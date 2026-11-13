"""Live Watch REST API (A10 real bodies; the frozen docstring contract).

Every handler talks ONLY to the LiveManager (``..live.manager.manager`` —
the module singleton) or to the A7 upload surface; never to a watcher, the
capture layer, or the store's runtime mutations. The manager import is
lazy inside each handler: importing at module scope would build a
LiveManager (and pull the whole capture/whisper stack) for every process
that imports the router, including CLI runs that never touch live.

JSON shapes mirror ``frontend/src/live/api.ts`` field-for-field (the wire
contract F1/F2/F3 coded against); ``schemas_live`` holds the models.

Endpoint summary (from the frozen skeleton docstring, all kept):

    GET  /api/live/streams                 list watched streams + live state
    POST /api/live/streams                 {url, label?} → add
    DELETE /api/live/streams/{id}          remove (stops capture, frees buffer)
    POST /api/live/streams/{id}/pause      pause
    POST /api/live/streams/{id}/resume     resume
    GET  /api/live/streams/{id}/events     SSE: state + clip events for stream
    GET  /api/live/clips                   list live clips (all streams)
    GET  /api/live/clips/{id}              one clip
    GET  /api/live/clips/{id}/media        the rendered mp4 (FileResponse)
    GET  /api/live/events                  SSE: global live feed (states+clips)
    GET  /api/live/uploads                 list upload records
    POST /api/live/clips/{id}/upload       {platform} manual upload trigger
    POST /api/live/clips/{id}/discard      drop a clip without rendering
    GET  /api/live/status                  {enabled, watching, clipping, caps}
    GET  /api/live/settings                LiveSettings (read)
    PUT  /api/live/settings                LiveSettings (write, validated)
    POST /api/live/uploads/youtube/start   OAuth start (returns redirect URL)
    POST /api/live/uploads/instagram/start OAuth start (returns redirect URL)
    GET  /api/live/uploads/youtube/status   token present? account ok?
    GET  /api/live/uploads/instagram/status token present? account ok?
    GET  /api/live/uploads/{platform}/callback    OAuth code exchange
    POST /api/live/uploads/{platform}/credentials user's own app id/secret
    GET/POST /api/live/uploads/attestation        rights-acceptance state

Error mapping (integration-ledger §4): RightsNotAccepted → 400 "accept
the rights notice", NotLinkedError → 409, UploadApiError → 502.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets as _secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

from .schemas_live import (
    LiveAttestationIn,
    LiveClipOut,
    LiveCredentialsIn,
    LiveSettingsOut,
    LiveStreamIn,
    LiveStreamOut,
    LiveUploadIn,
    LiveUploadOut,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live", tags=["live"])

#: The shared all-clips channel (watcher.py naming contract).
CLIPS_CHANNEL = "live:-:clips"


# --------------------------------------------------------------------------
# Manager access — lazy, 503 while the feature isn't running
# --------------------------------------------------------------------------


async def _manager():
    """The live manager singleton; 503 unless started.

    A disabled or stopped live feature must say "not running" rather than
    pretend to be an empty feature — the frozen docstring contract.
    """
    from ..live.manager import manager

    if not manager.started:
        raise HTTPException(status_code=503, detail="Live Watch is not running.")
    return manager


VALID_PLATFORMS = ("youtube", "instagram")


def _check_platform(platform: str) -> str:
    if platform not in VALID_PLATFORMS:
        raise HTTPException(status_code=404, detail=f"Unknown upload platform {platform!r}.")
    return platform


def _uploads_pkg():
    """The A7 upload package root, imported lazily (httpx/keyring chain)."""
    from ..live import uploads as pkg

    return pkg


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


def _status_payload(status: dict) -> dict:
    """manager.status() → the frontend ``LiveStatus`` shape (api.ts).

    Key renames, manager → api.ts:
    - ``clip_queue_depth`` → ``clip_queue_len``
    - ``pending_clip_bytes`` / 1 MiB → ``disk_used_mb``
    - ``pending_cap_bytes`` → dropped (internal bookkeeping)

    ``clipping`` (an api.ts field the manager doesn't expose) is derived
    from the per-stream snapshot states — the manager's ``watching`` set
    counts capture capacity, not the state-machine stage the UI means.
    """
    caps = dict(status.get("caps") or {})
    streams = status.get("streams") or []
    clipping = sum(1 for s in streams if isinstance(s, dict) and s.get("state") == "clipping")
    return {
        "enabled": bool(status.get("enabled")),
        "watching": int(status.get("watching") or 0),
        "clipping": clipping,
        "caps": {
            "max_concurrent_streams": int(caps.get("max_concurrent_streams") or 0),
            "clip_queue_len": int(caps.get("clip_queue_depth") or 0),
            "clip_queue_max": int(caps.get("clip_queue_max") or 0),
            "disk_used_mb": round(float(caps.get("pending_clip_bytes") or 0) / (1024 * 1024), 1),
        },
    }


@router.get("/status")
async def live_status() -> dict:
    """Whole-feature status strip for the streams panel header."""
    m = await _manager()
    return _status_payload(m.status())


@router.get("/streams")
async def list_streams() -> list[LiveStreamOut]:
    """List every watched stream + its live state (the manager's view)."""
    m = await _manager()
    out: list[LiveStreamOut] = []
    for snapshot in m.status().get("streams") or []:
        try:
            out.append(LiveStreamOut.of_snapshot(m.stream_snapshot(snapshot["id"])))
        except KeyError:
            continue  # removed between list and snapshot; the next poll has it
    return out


@router.post("/streams", status_code=201)
async def add_stream(payload: LiveStreamIn) -> LiveStreamOut:
    """Register a stream; the watcher launches when started+enabled."""
    m = await _manager()
    try:
        sid = await m.add_stream(payload.url, payload.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LiveStreamOut.of_snapshot(m.stream_snapshot(sid))


@router.delete("/streams/{stream_id}", status_code=204)
async def remove_stream(stream_id: str) -> Response:
    """Stop capture, free the buffer, delete every persisted trace."""
    m = await _manager()
    try:
        await m.remove_stream(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Stream not found.") from exc
    return Response(status_code=204)


@router.post("/streams/{stream_id}/pause")
async def pause_stream(stream_id: str) -> LiveStreamOut:
    """Pause: stop capture+signals, keep config and the row (state=off)."""
    m = await _manager()
    try:
        await m.pause_stream(stream_id)
        return LiveStreamOut.of_snapshot(m.stream_snapshot(stream_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Stream not found.") from exc


@router.post("/streams/{stream_id}/resume")
async def resume_stream(stream_id: str) -> LiveStreamOut:
    """Resume: fresh watcher start for a paused stream."""
    m = await _manager()
    try:
        await m.resume_stream(stream_id)
        return LiveStreamOut.of_snapshot(m.stream_snapshot(stream_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Stream not found.") from exc


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def _stored_live_section():
    """The raw ``live`` dict from config.json, or None when absent/corrupt.

    Reading the raw file (not config.load()) is deliberate: the live
    section is NOT part of pydantic Settings (``extra="ignore"``), so the
    model round-trip would silently drop it.
    """
    import json as _json

    from .. import paths

    path = paths.config_path()
    if not path.exists():
        return None
    try:
        raw = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw.get("live") if isinstance(raw, dict) else None


def _read_settings_model():
    """LiveSettings exactly as the manager would load it (same validation)."""
    from ..live.manager import json_read_live_section

    return json_read_live_section(_stored_live_section())


def _write_live_settings(validated) -> None:
    """Persist a validated LiveSettings as config.json's ``live`` section.

    The raw dict is read, the ``live`` key replaced, and the whole file
    written back atomically (tmp + replace — config.save's own crash
    safety can't be reused because it serializes a Settings model that
    doesn't know the section exists). Every other key round-trips
    untouched, including ``_fallback_secrets``.
    """
    import json as _json
    import os

    from .. import paths

    path = paths.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = _json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw["live"] = validated.model_dump(mode="json")

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(raw, indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):  # platform dependent
        tmp.chmod(0o600)
    os.replace(tmp, path)


@router.get("/settings")
async def get_live_settings() -> LiveSettingsOut:
    return LiveSettingsOut.of(_read_settings_model())


@router.put("/settings")
async def put_live_settings(payload: dict) -> LiveSettingsOut:
    """Full-body LiveSettings replace (frozen contract; F1/F3 send the whole).

    Validated through ``LiveSettings.model_validate`` before anything is
    written so an invalid tree can never corrupt config.json. A malformed
    body is a 400 with a readable ``detail`` string — the frontend decodes
    ``detail`` strings from non-2xx responses (api.ts request()).
    """
    from ..live.manager import json_read_live_section
    from ..live.settings import LiveSettings

    try:
        validated = LiveSettings.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid live settings: {exc}") from exc

    _write_live_settings(validated)
    return LiveSettingsOut.of(json_read_live_section(_stored_live_section()))


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


@router.get("/clips")
async def list_clips(limit: int = 100) -> list[LiveClipOut]:
    """Newest-first clips across every stream."""
    from ..live import store

    rows = await asyncio.to_thread(store.list_clips, None, limit)
    return [LiveClipOut.of(row) for row in rows]


@router.get("/clips/{clip_id}")
async def get_clip(clip_id: str) -> LiveClipOut:
    from ..live import store

    row = await asyncio.to_thread(store.get_clip, clip_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return LiveClipOut.of(row)


@router.get("/clips/{clip_id}/media")
async def clip_media(clip_id: str) -> FileResponse:
    """The rendered mp4; 404 unless the clip is ready and the file exists."""
    from ..live import store

    row = await asyncio.to_thread(store.get_clip, clip_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    if row.status != "ready" or not row.rendered_path:
        raise HTTPException(status_code=404, detail="Clip is not ready.")
    path = Path(row.rendered_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Clip media file is missing.")
    return FileResponse(path, media_type="video/mp4")


@router.post("/clips/{clip_id}/discard")
async def discard_clip(clip_id: str) -> LiveClipOut:
    """Drop a clip without rendering: status → 'rejected' on the row."""
    from ..live import store
    from ..live.events import publish

    row = await asyncio.to_thread(store.get_clip, clip_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    await asyncio.to_thread(
        store.update_clip, clip_id, status="rejected", reason="discarded by user"
    )
    updated = await asyncio.to_thread(store.get_clip, clip_id)
    assert updated is not None
    # The response patches the caller's card; the SSE feed carries the same
    # verdict to every other open panel.
    publish(CLIPS_CHANNEL, "clip_status", LiveClipOut.of(updated).model_dump())
    return LiveClipOut.of(updated)


# --------------------------------------------------------------------------
# Uploads — records list, manual trigger, OAuth, credentials, attestation
# --------------------------------------------------------------------------


@router.get("/uploads")
async def list_uploads(limit: int = 200) -> list[LiveUploadOut]:
    from ..live import store

    rows = await asyncio.to_thread(store.list_uploads, None, limit)
    return [LiveUploadOut.of(row) for row in rows]


def _upload_meta_for(row, platform: str):
    """UploadMeta for one clip row, prefixed per the saved live settings."""
    from ..live.uploads import UploadMeta

    uploads_cfg = getattr(_read_settings_model(), "uploads", None)
    prefix = ""
    if uploads_cfg is not None:
        section = getattr(uploads_cfg, platform, None)
        attr = "title_prefix" if platform == "youtube" else "caption_prefix"
        prefix = getattr(section, attr, "") or ""
    title = f"{prefix}{row.title or ''}".strip() or f"AutoClip live clip {row.id[:8]}"
    return UploadMeta(title=title, description=row.hook or "")


async def _record_upload_outcome(
    clip_id: str,
    platform: str,
    *,
    result=None,
    error: str | None = None,
    status: str = "done",
) -> LiveUploadOut:
    """Persist one live_uploads row and publish it on the clips channel.

    Mirrors the manager's auto-upload path (integration ledger §3): refusals
    land as 'skipped' rows with the reason, platform failures as 'failed',
    successes carry ``media_id``/``url`` into ``platform_media_id``/
    ``upload_url``. F2 renders the row either from the POST response or
    from the 'upload' SSE frame — same shape, both paths.
    """
    from ..live import store
    from ..live.events import publish
    from ..live.models import LiveUploadRow, new_id, utcnow

    row = LiveUploadRow(
        id=new_id(),
        clip_id=clip_id,
        platform=platform,
        status=status,
        upload_url=(getattr(result, "url", "") or "") if result is not None else "",
        platform_media_id=(getattr(result, "media_id", "") or "") if result is not None else "",
        error=error,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    await asyncio.to_thread(store.create_upload, row)
    out = LiveUploadOut.of(row)
    publish(CLIPS_CHANNEL, "upload", out.model_dump())
    return out


def _upload_http_error(exc: Exception) -> HTTPException:
    """A7's error hierarchy → HTTP (integration ledger §4)."""
    from ..live.uploads.tokens import NotLinkedError, RightsNotAccepted, UploadError

    if isinstance(exc, RightsNotAccepted):
        return HTTPException(
            status_code=400,
            detail="Uploads are locked — accept the content-rights notice in the "
            "Live settings first.",
        )
    if isinstance(exc, NotLinkedError):
        platform = "platform"
        message = str(exc)
        for name in VALID_PLATFORMS:
            if name in message:
                platform = name
                break
        return HTTPException(status_code=409, detail=f"Connect the {platform} account first.")
    if isinstance(exc, UploadError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=502, detail=f"Upload failed: {exc}")


@router.post("/clips/{clip_id}/upload")
async def upload_clip(clip_id: str, payload: LiveUploadIn) -> LiveUploadOut:
    """Manual upload trigger → A7's upload_youtube / upload_instagram.

    Returns the LiveUpload row (F2 renders it). The error hierarchy maps to
    400/409/502 per the ledger; RIGHTS and NOT-LINKED refusals persist a
    'skipped' row (ledger §3), platform failures a 'failed' one — the wall
    shows why nothing was posted in both cases.
    """
    from ..live import store
    from ..live.uploads import upload_instagram, upload_youtube
    from ..live.uploads.tokens import (
        NotLinkedError,
        RightsNotAccepted,
        UploadError,
    )

    platform = _check_platform(payload.platform)
    row = await asyncio.to_thread(store.get_clip, clip_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    if row.status != "ready" or not row.rendered_path:
        raise HTTPException(status_code=409, detail="Only ready clips can be uploaded.")

    fn = upload_youtube if platform == "youtube" else upload_instagram
    meta = _upload_meta_for(row, platform)
    try:
        result = await fn(Path(row.rendered_path), meta)
    except RightsNotAccepted as exc:
        await _record_upload_outcome(clip_id, platform, error=str(exc), status="skipped")
        raise _upload_http_error(exc) from exc
    except NotLinkedError as exc:
        await _record_upload_outcome(clip_id, platform, error=str(exc), status="skipped")
        raise _upload_http_error(exc) from exc
    except UploadError as exc:
        await _record_upload_outcome(clip_id, platform, error=str(exc), status="failed")
        raise _upload_http_error(exc) from exc
    except Exception as exc:
        await _record_upload_outcome(
            clip_id, platform, error=str(exc) or type(exc).__name__, status="failed"
        )
        raise _upload_http_error(exc) from exc
    return await _record_upload_outcome(clip_id, platform, result=result)


# --------------------------------------------------------------------------
# OAuth start / callback / credentials / status / attestation
# --------------------------------------------------------------------------

#: CSRF state tokens minted by /start, one per pending consent. In-memory:
#: a restart invalidates pending consents, which is correct — state binds
#: one consent round-trip, it doesn't need to survive one.
_pending_states: dict[str, str] = {}


def _redirect_uri(platform: str) -> str:
    """The frozen callback location this server advertises as redirect_uri."""
    return f"http://localhost:8000/api/live/uploads/{platform}/callback"


@router.post("/uploads/{platform}/start")
async def start_upload_auth(platform: str) -> dict:
    """Build the consent URL; CSRF state is minted and held here."""
    _check_platform(platform)
    uploads = _uploads_pkg()
    state = _secrets.token_urlsafe(16)
    _pending_states[platform] = state
    try:
        redirect = uploads.authorization_url(
            platform, redirect_uri=_redirect_uri(platform), state=state
        )
    except Exception as exc:
        _pending_states.pop(platform, None)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"redirect_url": redirect}


@router.get("/uploads/{platform}/callback")
async def upload_auth_callback(
    platform: str, code: str = "", state: str = "", error: str = ""
) -> Response:
    """Exchange the code, then bounce the browser to /live?connected=1.

    An HTML page (not JSON): this endpoint's client is the user's consent
    tab, not fetch(). A missing or mismatched state fails closed — a plain
    400 page, no token exchange, no redirect.
    """
    _check_platform(platform)
    uploads = _uploads_pkg()

    if error:
        return HTMLResponse(
            _callback_page("not connected", f"The platform refused the consent: {error}."),
            status_code=400,
        )
    if not code:
        return HTMLResponse(
            _callback_page("not connected", "No authorization code was provided."),
            status_code=400,
        )
    if state != _pending_states.get(platform):
        return HTMLResponse(
            _callback_page("not connected", "This authorization link has expired — start again."),
            status_code=400,
        )
    _pending_states.pop(platform, None)

    try:
        await uploads.exchange_code(platform, code, redirect_uri=_redirect_uri(platform))
    except Exception as exc:
        log.warning("OAuth code exchange failed for %s: %s", platform, exc)
        return HTMLResponse(
            _callback_page("not connected", f"Connecting failed: {type(exc).__name__}."),
            status_code=502,
        )
    return HTMLResponse(_callback_page("connected", None))


def _callback_page(status: str, message: str | None) -> str:
    """Minimal page that bounces the consent tab back to the Live page."""
    target = "/live?connected=1" if status == "connected" else "/live"
    note = f"\n  <p>{message}</p>" if message else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>AutoClip Live Watch</title>\n"
        f'  <meta http-equiv="refresh" content="0; url={target}">\n'
        "</head><body>\n"
        f"  <p>AutoClip Live — {status}.</p>{note}\n"
        f'  <p><a href="{target}">Continue to Live Watch</a></p>\n'
        "</body></html>"
    )


@router.get("/uploads/{platform}/status")
async def upload_platform_status(platform: str) -> dict:
    """``{connected, account}`` — A7's status dicts, verbatim."""
    _check_platform(platform)
    uploads = _uploads_pkg()
    fn = uploads.youtube_status if platform == "youtube" else uploads.instagram_status
    return await fn()


@router.post("/uploads/{platform}/credentials")
async def save_credentials(platform: str, payload: LiveCredentialsIn) -> dict:
    """Store the user's own OAuth app credentials; never echoed back."""
    _check_platform(platform)
    uploads = _uploads_pkg()
    if not payload.client_id.strip() or not payload.client_secret.strip():
        raise HTTPException(status_code=400, detail="Both client id and secret are required.")
    try:
        uploads.save_client_credentials(
            platform, payload.client_id.strip(), payload.client_secret.strip()
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True}


@router.get("/uploads/attestation")
async def get_attestation() -> dict:
    """The persisted rights-acceptance state for the settings page."""
    uploads = _uploads_pkg()
    stored = uploads.rights_attestation()
    accepted = bool(stored) and str(stored).startswith("accepted:")
    return {
        "accepted": accepted,
        "accepted_at": str(stored).removeprefix("accepted:") if accepted else None,
    }


@router.post("/uploads/attestation")
async def post_attestation(payload: LiveAttestationIn) -> dict:
    """Record acceptance via A7's accept_rights() (keyring flag)."""
    uploads = _uploads_pkg()
    if not payload.accepted:
        raise HTTPException(
            status_code=400,
            detail="The rights notice must be explicitly accepted.",
        )
    stored = uploads.accept_rights()
    return {
        "accepted": True,
        "accepted_at": str(stored).removeprefix("accepted:"),
    }


# --------------------------------------------------------------------------
# SSE — the global feed and one stream's feed
# --------------------------------------------------------------------------


#: watcher/manager event name → the six SSE names the frontend has listeners
#: for (subscribeLiveEvents in api.ts). Anything absent here is dropped at
#: the boundary: the frontend would never dispatch it, so emitting the raw
#: name is dead traffic. 'clip' (full row) vs 'clip_status' (thin patch) is
#: exactly the split F2's coerceClip/mergeClipFields pair implements.
_FRONTEND_EVENT: dict[str, str] = {
    "state": "state",
    "clip": "clip",
    "clip_status": "clip_status",
    "upload": "upload",
    "heartbeat": "heartbeat",
    # clip lifecycle → status patches
    "clip_detecting": "clip_status",
    "clip_queued": "clip_status",
    "clip_rejected": "clip_status",
    "clip_dropped": "clip_status",
    "clip_failed": "clip_status",
    "clip_ready": "clip_status",
    "clip_producing": "clip_status",
    # stream lifecycle → state patches (F1 streamPatch)
    "stream_added": "state",
    "stream_removed": "state",
    "stream_paused": "state",
    "stream_resumed": "state",
}


def _frontend_event_name(watcher_event: str) -> str | None:
    """Map one watcher/manager event name to the frontend's vocabulary."""
    if watcher_event in ("snapshot", "heartbeat"):
        return watcher_event
    return _FRONTEND_EVENT.get(watcher_event)


async def _sse_generator(request: Request, channels: list[str]):
    """The SSE body: initial snapshot, then every channel's mapped events.

    A queue per feed (not per subscriber — the broker owns those) merges
    the channels so one SSE connection receives both state and clip
    traffic in arrival order. The broker is bound to THIS loop first: in a
    no-worker process nothing else has bound it, and discard/upload
    publishes from request handlers must still marshal safely.
    """
    from ..live import events as live_events

    live_events.bind_loop(asyncio.get_running_loop())
    yield {"event": "snapshot", "data": json.dumps({"resync": True})}

    queue: asyncio.Queue = asyncio.Queue()
    pumps = [asyncio.create_task(_pump(channel, queue)) for channel in channels]
    global _sse_feeds_open
    _sse_feeds_open += 1
    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except TimeoutError:
                yield {"event": "heartbeat", "data": json.dumps({})}
                continue
            name = _frontend_event_name(str(event.get("type") or ""))
            if name is None:
                continue
            data = _state_patch(event, channels)
            yield {"event": name, "data": json.dumps(data)}
    finally:
        for pump in pumps:
            pump.cancel()
        # Unbind only if we are the last live feed (refcounted so a second
        # open feed never loses its binding).
        _sse_feeds_open = max(0, _sse_feeds_open - 1)
        if _sse_feeds_open == 0:
            live_events.bind_loop(None)


#: Open SSE feeds. The binding is shared state: /events and a per-stream
#: feed can be open at once, and the last one to close unbinds the broker.
_sse_feeds_open = 0


def _state_patch(event: dict, channels: list[str]) -> dict:
    """Normalize one event's data for the frontend's patch merge.

    F1's streamPatch reads ``state_detail``/``last_error``; the watcher's
    state events carry ``detail``/``error``. Renaming here keeps the
    watcher vocabulary internal and the wire matching api.ts.
    """
    data = dict(event.get("data") or {})
    name = str(event.get("type") or "")
    if name == "state" and isinstance(data, dict):
        renamed = {}
        for key, value in data.items():
            if key == "detail":
                renamed["state_detail"] = value
            elif key == "error":
                renamed["last_error"] = value
            else:
                renamed[key] = value
        if (
            "stream_id" not in renamed
            and len(channels) == 1
            and channels[0].startswith("live:")
            and not channels[0].startswith("live:-:")
        ):
            renamed["stream_id"] = channels[0].removeprefix("live:")
        data = renamed
    return data


async def _pump(channel: str, queue: asyncio.Queue) -> None:
    """Forward one channel's events into the shared feed queue, forever."""
    from ..live.events import subscribe

    async for event in subscribe(channel):
        await queue.put(event)


@router.get("/events")
async def global_events(request: Request) -> EventSourceResponse:
    """The global feed: the shared clips channel (all streams' clip events).

    Per-stream state traffic rides the frontend's 5s poll; the manager's
    state publishes go to each stream's own channel where its dedicated
    endpoint (/streams/{id}/events) picks them up.
    """
    return EventSourceResponse(_sse_generator(request, [CLIPS_CHANNEL]))


@router.get("/streams/{stream_id}/events")
async def stream_events(request: Request, stream_id: str) -> EventSourceResponse:
    """One stream's feed: its own channel + the shared clips channel."""
    from ..live.watcher import stream_channel

    return EventSourceResponse(_sse_generator(request, [stream_channel(stream_id), CLIPS_CHANNEL]))
