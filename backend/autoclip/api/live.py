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
    GET  /api/live/clips?status=a,b        list live clips (all streams)
    GET  /api/live/clips/{id}              one clip
    GET  /api/live/clips/{id}/media        the rendered mp4 (FileResponse)
    POST /api/live/clips/{id}/retry        re-render a failed clip (409 if not)
    GET  /api/live/events                  SSE: global live feed (states+clips)
    GET  /api/live/uploads                 list upload records
    POST /api/live/clips/{id}/upload       {platform} manual upload trigger
    POST /api/live/clips/{id}/discard      drop a clip (tombstone)
    GET  /api/live/status                  {enabled, running, watching, clipping, caps}
    GET  /api/live/settings                LiveSettings (read)
    PUT  /api/live/settings                LiveSettings (write, validated, applied)
    POST /api/live/uploads/youtube/start   OAuth start (returns redirect URL)
    POST /api/live/uploads/instagram/start OAuth start (returns redirect URL)
    GET  /api/live/uploads/youtube/status   token present? account ok?
    GET  /api/live/uploads/instagram/status token present? account ok?
    GET  /api/live/uploads/{platform}/callback    OAuth code exchange
    POST /api/live/uploads/{platform}/credentials user's own app id/secret
    GET/POST /api/live/uploads/attestation        rights-acceptance state

Gate: ``/status``, ``/streams`` (GET), settings and the store-backed reads
always answer. Mutating manager endpoints answer 409 ("Live Watch is turned
off…") while the feature is switched off, and 503 while it is on but the
process runs without workers.

Error mapping (integration-ledger §4): RightsNotAccepted → 400 "accept
the rights notice", NotLinkedError → 409, UploadApiError → 502. Upload
refusals of the rights/linking pre-checks are NOT persisted (nothing was
attempted); an upload already running or done for the clip/platform is 409.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import secrets as _secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

from ..live.models import UPLOADS_IN_FLIGHT
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
#: The shared stream-state channel (``stream_state`` events).
STREAMS_CHANNEL = "live:-:streams"

#: 409 detail while the feature switch is off (frontend contract).
LIVE_OFF_DETAIL = "Live Watch is turned off. Turn it on in Live settings."
NOT_RUNNING_DETAIL = "Live Watch is not running."


# --------------------------------------------------------------------------
# Manager access — lazy; 409 while switched off, 503 while not running
# --------------------------------------------------------------------------


def _singleton():
    from ..live.manager import manager

    return manager


def _manager_enabled(m) -> bool:
    enabled = getattr(m, "enabled", None)
    if isinstance(enabled, bool):
        return enabled
    with contextlib.suppress(Exception):
        return bool(m.status().get("enabled", True))
    return True


async def _manager():
    """The live manager singleton, for endpoints that need it RUNNING.

    Switched off in settings → 409 with the "turn it on" detail; on but not
    running (a no-worker process) → 503. Never pretends to be an empty
    running feature.
    """
    manager = _singleton()
    if not manager.started:
        if not _manager_enabled(manager):
            raise HTTPException(status_code=409, detail=LIVE_OFF_DETAIL)
        raise HTTPException(status_code=503, detail=NOT_RUNNING_DETAIL)
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
    enabled = bool(status.get("enabled"))
    running = bool(status.get("running", enabled))
    if not enabled:
        # Switched off: the strip says so and every number is zero.
        return {
            "enabled": False,
            "running": False,
            "watching": 0,
            "clipping": 0,
            "caps": {
                "max_concurrent_streams": 0,
                "clip_queue_len": 0,
                "clip_queue_max": 0,
                "disk_used_mb": 0.0,
            },
        }
    clipping = sum(1 for s in streams if isinstance(s, dict) and s.get("state") == "clipping")
    return {
        "enabled": True,
        "running": running,
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
    """Whole-feature status strip — ALWAYS 200 (``enabled`` says if it's on)."""
    m = _singleton()
    return _status_payload(m.status())


def _snapshots(m) -> list[dict]:
    lister = getattr(m, "list_stream_snapshots", None)
    if lister is not None:
        return list(lister())
    return list(m.status().get("streams") or [])


@router.get("/streams")
async def list_streams() -> list[LiveStreamOut]:
    """Every watched stream + its live state — also while Live is off.

    A stopped feature still has a watch list; its streams read ``off``.
    """
    m = _singleton()
    out: list[LiveStreamOut] = []
    # On the loop, not in a thread: the manager's state is loop-owned.
    for snapshot in _snapshots(m):
        try:
            out.append(LiveStreamOut.of_snapshot(snapshot))
        except (KeyError, TypeError, ValueError):
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
    from ..live.settings import stored_live_section

    return stored_live_section()


def _read_settings_model():
    """LiveSettings exactly as the manager loads it (same parser)."""
    from ..live.settings import parse_live_section

    return parse_live_section(_stored_live_section())


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
async def put_live_settings(payload: dict) -> dict:
    """Full-body LiveSettings replace, persisted AND applied to the running feature.

    Validated through ``LiveSettings.model_validate`` before anything is
    written so an invalid tree can never corrupt config.json. A malformed
    body is a 400 with a readable ``detail`` string — the frontend decodes
    ``detail`` strings from non-2xx responses (api.ts request()). The
    response is the stored settings plus ``"applied": true`` once the
    running manager took them (enable/disable, budgets, thresholds,
    workers, auto-upload); on an apply failure ``applied`` is false and
    ``apply_error`` says why (the save itself stands).
    """
    from ..live.settings import LiveSettings

    try:
        validated = LiveSettings.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid live settings: {exc}") from exc

    await asyncio.to_thread(_write_live_settings, validated)
    stored = _read_settings_model()
    body = LiveSettingsOut.of(stored).model_dump()
    m = _singleton()
    apply = getattr(m, "apply_settings", None)
    try:
        if apply is not None:
            await apply(stored)
        body["applied"] = True
    except Exception as exc:
        log.exception("Applying live settings failed.")
        body["applied"] = False
        body["apply_error"] = str(exc) or type(exc).__name__
    return body


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


@router.get("/clips")
async def list_clips(limit: int = 100, status: str | None = None) -> list[LiveClipOut]:
    """Newest-first clips across every stream.

    ``status`` filters by a comma-separated list (``?status=ready,failed``).
    Rejected clips carry their ``reason``.
    """
    from ..live import store

    statuses = [s.strip() for s in (status or "").split(",") if s.strip()] or None
    rows = await asyncio.to_thread(
        lambda: store.list_clips(None, max(1, int(limit)), statuses=statuses)
    )
    return [LiveClipOut.of(row) for row in rows]


@router.post("/clips/{clip_id}/retry")
async def retry_clip(clip_id: str) -> LiveClipOut:
    """Re-enqueue a failed clip whose window is still on disk (409 otherwise)."""
    from ..live import store
    from ..live.manager import RetryRefused

    row = await asyncio.to_thread(store.get_clip, clip_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    m = _singleton()
    if not m.started:
        detail = LIVE_OFF_DETAIL if not _manager_enabled(m) else NOT_RUNNING_DETAIL
        raise HTTPException(status_code=409, detail=detail)
    try:
        updated = await m.retry_clip(clip_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Clip not found.") from exc
    except RetryRefused as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LiveClipOut.of(updated)


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
    """Drop a clip: status → 'rejected' on the row — a TOMBSTONE.

    Every worker checks it atomically before each status write, before the
    render and before an auto-upload: a queued clip is never rendered, a
    rendering one never surfaces as ready, a detecting one stops before
    the paid judge call.
    """
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


async def _finish_upload_row(
    row,
    *,
    result=None,
    error: str | None = None,
    status: str = "done",
) -> LiveUploadOut:
    """Settle the in-flight ``uploading`` row and publish it on the clips feed.

    Platform failures land as 'failed' with the reason, successes carry
    ``media_id``/``url`` into ``platform_media_id``/``upload_url``. F2
    renders the row either from the POST response or from the 'upload' SSE
    frame — same shape, both paths.
    """
    from ..live import store
    from ..live.events import publish

    fields: dict = {"status": status, "error": error}
    if result is not None:
        fields["upload_url"] = getattr(result, "url", "") or ""
        fields["platform_media_id"] = getattr(result, "media_id", "") or ""
    await asyncio.to_thread(store.update_upload, row.id, **fields)
    for key, value in fields.items():
        setattr(row, key, value)
    stored = next(
        (r for r in await asyncio.to_thread(store.list_uploads, row.clip_id) if r.id == row.id),
        row,
    )
    out = LiveUploadOut.of(stored)
    publish(CLIPS_CHANNEL, "upload", out.model_dump())
    return out


#: (clip_id, platform) pairs with an upload in flight in this process —
#: checked and claimed with no await in between, so a double click cannot
#: slip two uploads past the DB check. Shared with the manager's
#: auto-upload, which claims the same keys.
_uploads_in_flight: set[tuple[str, str]] = UPLOADS_IN_FLIGHT


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

    Returns the LiveUpload row (F2 renders it). 409 while an upload of this
    clip to this platform is running or once one succeeded; a new attempt
    is allowed after ``failed``/``skipped``. The rights and account-linking
    pre-checks refuse with 400/409 WITHOUT persisting anything (nothing was
    attempted, and a stored refusal would only block the retry); platform
    failures persist a 'failed' row; errors map to 400/409/502 per the
    ledger.
    """
    from ..live import store
    from ..live.models import LiveUploadRow, new_id, utcnow
    from ..live.uploads import upload_instagram, upload_youtube
    from ..live.uploads.tokens import (
        NotLinkedError,
        RightsNotAccepted,
        UploadError,
    )

    platform = _check_platform(payload.platform)
    key = (clip_id, platform)
    if key in _uploads_in_flight:
        raise HTTPException(
            status_code=409, detail=f"An upload of this clip to {platform} is already running."
        )
    _uploads_in_flight.add(key)  # claimed before the first await
    try:
        row = await asyncio.to_thread(store.get_clip, clip_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Clip not found.")
        if row.status != "ready" or not row.rendered_path:
            raise HTTPException(status_code=409, detail="Only ready clips can be uploaded.")
        previous = await asyncio.to_thread(store.list_uploads, clip_id)
        for upload in previous:
            if upload.platform != platform:
                continue
            if upload.status == "uploading":
                raise HTTPException(
                    status_code=409,
                    detail=f"An upload of this clip to {platform} is already running.",
                )
            if upload.status == "done":
                raise HTTPException(
                    status_code=409, detail=f"This clip was already uploaded to {platform}."
                )

        upload_row = LiveUploadRow(
            id=new_id(),
            clip_id=clip_id,
            platform=platform,
            status="uploading",
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        await asyncio.to_thread(store.create_upload, upload_row)
        fn = upload_youtube if platform == "youtube" else upload_instagram
        meta = _upload_meta_for(row, platform)
        try:
            result = await fn(Path(row.rendered_path), meta)
        except (RightsNotAccepted, NotLinkedError) as exc:
            # A pre-check refusal: nothing was sent. Not persisted.
            await asyncio.to_thread(store.delete_upload, upload_row.id)
            raise _upload_http_error(exc) from exc
        except UploadError as exc:
            await _finish_upload_row(upload_row, error=str(exc), status="failed")
            raise _upload_http_error(exc) from exc
        except Exception as exc:
            await _finish_upload_row(
                upload_row, error=str(exc) or type(exc).__name__, status="failed"
            )
            raise _upload_http_error(exc) from exc
        return await _finish_upload_row(upload_row, result=result)
    finally:
        _uploads_in_flight.discard(key)


# --------------------------------------------------------------------------
# OAuth start / callback / credentials / status / attestation
# --------------------------------------------------------------------------

#: CSRF state tokens minted by /start, one per pending consent, with the
#: redirect_uri that consent was started for (the code exchange must send
#: the identical value). In-memory: a restart invalidates pending consents,
#: which is correct — state binds one consent round-trip.
_pending_states: dict[str, tuple[str, str]] = {}
#: Which AutoClip page the consent tab returns to, per platform. Only known
#: in-app paths are accepted (never an arbitrary URL: open-redirect).
_return_pages: dict[str, str] = {}
RETURN_PAGES = frozenset({"/live", "/gta"})


def _redirect_uri(request: Request, platform: str) -> str:
    """The callback location for THIS server, derived from the request.

    Built from the request's base URL (scheme, host and port the browser
    used), so a server on any port — or behind the dev proxy — advertises
    a callback that actually reaches it. Register exactly this URL in your
    OAuth app (the status endpoint reports it).
    """
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/live/uploads/{platform}/callback"


@router.post("/uploads/{platform}/start")
async def start_upload_auth(platform: str, request: Request, return_to: str = "/live") -> dict:
    """Build the consent URL; CSRF state is minted and held here."""
    _check_platform(platform)
    uploads = _uploads_pkg()
    state = _secrets.token_urlsafe(16)
    redirect_uri = _redirect_uri(request, platform)
    _pending_states[platform] = (state, redirect_uri)
    _return_pages[platform] = return_to if return_to in RETURN_PAGES else "/live"
    try:
        redirect = uploads.authorization_url(platform, redirect_uri=redirect_uri, state=state)
    except Exception as exc:
        _pending_states.pop(platform, None)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"redirect_url": redirect}


#: The consent tab's page is static: no script, nothing loads from anywhere.
CALLBACK_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)


def _callback_response(
    status: str, message: str | None, status_code: int = 200, *, page: str = "/live"
) -> Response:
    response = HTMLResponse(_callback_page(status, message, page), status_code=status_code)
    response.headers["Content-Security-Policy"] = CALLBACK_CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/uploads/{platform}/callback")
async def upload_auth_callback(
    platform: str, request: Request, code: str = "", state: str = "", error: str = ""
) -> Response:
    """Exchange the code, then bounce the browser to /live?connected=1.

    An HTML page (not JSON): this endpoint's client is the user's consent
    tab, not fetch(). A missing or mismatched state fails closed — a plain
    400 page, no token exchange, no redirect. Everything echoed into the
    page is HTML-escaped and the page ships a strict CSP: the query string
    is attacker-controlled on an unauthenticated endpoint.
    """
    _check_platform(platform)
    uploads = _uploads_pkg()
    page = _return_pages.get(platform, "/live")

    if error:
        return _callback_response(
            "not connected", f"The platform refused the consent: {error}.", 400, page=page
        )
    if not code:
        return _callback_response(
            "not connected", "No authorization code was provided.", 400, page=page
        )
    pending = _pending_states.get(platform)
    if pending is None or not state or not _secrets.compare_digest(state, pending[0]):
        return _callback_response(
            "not connected", "This authorization link has expired — start again.", 400, page=page
        )
    _pending_states.pop(platform, None)
    _return_pages.pop(platform, None)

    try:
        await uploads.exchange_code(platform, code, redirect_uri=pending[1])
    except Exception as exc:
        log.warning("OAuth code exchange failed for %s: %s", platform, exc)
        return _callback_response(
            "not connected", f"Connecting failed: {type(exc).__name__}.", 502, page=page
        )
    return _callback_response("connected", None, page=page)


def _callback_page(status: str, message: str | None, page: str = "/live") -> str:
    """Minimal page that bounces the consent tab back to the page it came from.

    Every interpolated value is HTML-escaped (``message`` carries query
    parameters verbatim).
    """
    page = page if page in RETURN_PAGES else "/live"
    target = f"{page}?connected=1" if status == "connected" else page
    safe_target = html.escape(target, quote=True)
    safe_status = html.escape(status, quote=True)
    note = f"\n  <p>{html.escape(message, quote=True)}</p>" if message else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>AutoClip Live Watch</title>\n"
        f'  <meta http-equiv="refresh" content="0; url={safe_target}">\n'
        "</head><body>\n"
        f"  <p>AutoClip Live — {safe_status}.</p>{note}\n"
        f'  <p><a href="{safe_target}">Continue to Live Watch</a></p>\n'
        "</body></html>"
    )


@router.get("/uploads/{platform}/status")
async def upload_platform_status(platform: str, request: Request) -> dict:
    """A7's ``{connected, account}`` plus ``redirect_uri`` and ``has_credentials``.

    ``redirect_uri`` is the exact callback URL to register in the OAuth app;
    ``has_credentials`` says whether that app's client id/secret are stored.
    """
    _check_platform(platform)
    uploads = _uploads_pkg()
    fn = uploads.youtube_status if platform == "youtube" else uploads.instagram_status
    body = dict(await fn())
    try:
        has_credentials = (
            await asyncio.to_thread(uploads.load_client_credentials, platform)
        ) is not None
    except Exception:
        log.warning("Could not read %s client credentials.", platform, exc_info=True)
        has_credentials = False
    body["redirect_uri"] = _redirect_uri(request, platform)
    body["has_credentials"] = has_credentials
    return body


@router.post("/uploads/{platform}/credentials")
async def save_credentials(platform: str, payload: LiveCredentialsIn) -> dict:
    """Store (or replace) the user's own OAuth app credentials; never echoed back.

    Replacing them with a DIFFERENT app's client id also drops the stored
    connection: tokens issued to the old app can't be refreshed with the new
    one, so the user reconnects (consents) with the new app.
    """
    _check_platform(platform)
    uploads = _uploads_pkg()
    client_id = payload.client_id.strip()
    if not client_id or not payload.client_secret.strip():
        raise HTTPException(status_code=400, detail="Both client id and secret are required.")
    try:
        previous = uploads.load_client_credentials(platform)
    except Exception:
        previous = None
    app_changed = previous is not None and previous[0] != client_id
    try:
        uploads.save_client_credentials(platform, client_id, payload.client_secret.strip())
        if app_changed:
            uploads.clear_tokens(platform)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True, "disconnected": app_changed}


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
    "clip_rendering": "clip_status",
    # stream lifecycle → state patches (F1 streamPatch)
    "stream_added": "state",
    "stream_removed": "state",
    "stream_paused": "state",
    "stream_resumed": "state",
    # every stream state change, on the global feed
    "stream_state": "stream_state",
}

#: Clip lifecycle event → the row status it announces. Every clip event on
#: the wire carries ``status`` so a patch never has to guess.
_CLIP_EVENT_STATUS: dict[str, str] = {
    "clip_detecting": "detecting",
    "clip_queued": "queued",
    "clip_producing": "rendering",
    "clip_rendering": "rendering",
    "clip_ready": "ready",
    "clip_failed": "failed",
    "clip_rejected": "rejected",
    "clip_dropped": "rejected",
}


def _frontend_event_name(watcher_event: str) -> str | None:
    """Map one watcher/manager event name to the frontend's vocabulary."""
    if watcher_event in ("snapshot", "heartbeat"):
        return watcher_event
    return _FRONTEND_EVENT.get(watcher_event)


#: Merged events one SSE connection may hold for a slow client. Past it the
#: feed drops events and sends a ``snapshot`` resync instead (the frontend
#: refetches on it) — an unbounded merge queue would defeat the broker's
#: own per-subscriber bound and grow without limit behind a stalled socket.
SSE_FEED_QUEUE = 256


class _Feed:
    """One SSE connection's merged, bounded event queue."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_FEED_QUEUE)
        #: Set when an event had to be dropped: the client must resync.
        self.overflowed = False

    def offer(self, event: dict) -> None:
        if event.get("type") == "resync":  # the broker dropped events for us
            self.overflowed = True
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.overflowed = True

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return


async def _sse_generator(request: Request, channels: list[str]):
    """The SSE body: initial snapshot, then every channel's mapped events.

    A queue per feed (not per subscriber — the broker owns those) merges
    the channels so one SSE connection receives both state and clip
    traffic in arrival order. The broker is bound to THIS loop first: in a
    no-worker process nothing else has bound it, and discard/upload
    publishes from request handlers must still marshal safely.

    The channels are subscribed BEFORE the snapshot frame goes out: the
    client refetches on ``snapshot``, so anything published after that
    refetch must already be on its way to this feed. A feed that falls
    behind (slow client) is bounded: overflow drops the backlog and sends
    another ``snapshot`` so the client refetches instead of silently
    missing a state change.
    """
    from ..live import events as live_events

    live_events.bind_loop(asyncio.get_running_loop())
    feed = _Feed()
    pumps = [asyncio.create_task(_pump(channel, feed)) for channel in channels]
    global _sse_feeds_open
    _sse_feeds_open += 1
    try:
        # Let every pump register its subscription before the snapshot.
        await asyncio.sleep(0)
        yield {"event": "snapshot", "data": json.dumps({"resync": True})}
        while True:
            if await request.is_disconnected():
                break
            if feed.overflowed:
                feed.overflowed = False
                feed.drain()
                yield {"event": "snapshot", "data": json.dumps({"resync": True})}
                continue
            try:
                event = await asyncio.wait_for(feed.queue.get(), timeout=15.0)
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
    status = _CLIP_EVENT_STATUS.get(name)
    if status is not None:
        data["status"] = status
    if name == "stream_state":
        data["type"] = "stream_state"
        data["enabled"] = bool(data.get("enabled", True))
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


async def _pump(channel: str, feed: _Feed) -> None:
    """Forward one channel's events into the connection's bounded feed."""
    from ..live.events import subscribe

    async for event in subscribe(channel):
        feed.offer(event)


@router.get("/events")
async def global_events(request: Request) -> EventSourceResponse:
    """The global feed: every stream's clip events AND state changes.

    Clip events arrive as ``clip_status`` (with ``status``) / ``upload``;
    stream state changes as ``stream_state`` frames whose data is
    ``{"type": "stream_state", "stream_id", "state", "enabled", ...}``.
    A heartbeat frame keeps idle connections alive.
    """
    return EventSourceResponse(_sse_generator(request, [CLIPS_CHANNEL, STREAMS_CHANNEL]))


@router.get("/streams/{stream_id}/events")
async def stream_events(request: Request, stream_id: str) -> EventSourceResponse:
    """One stream's feed: its own channel + the shared clips channel."""
    from ..live.watcher import stream_channel

    return EventSourceResponse(_sse_generator(request, [stream_channel(stream_id), CLIPS_CHANNEL]))
