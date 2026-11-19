"""GTA Mode API: the stream queue, clips, GTA Live, settings, feedback loop.

Clip edits (hook choice, flash-forward, variants, picks on the Hype Curve)
live in :mod:`autoclip.gta.edits`; the routes here only translate them.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..gta import store
from ..gta.fetch import detect_platform, is_supported_url
from ..gta.pipeline import cleanup_stream_files, post_text
from ..gta.queue import queue
from ..gta.settings import GtaSettings
from ..gta.settings import load as load_settings
from ..gta.settings import save as save_settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gta", tags=["gta"])

#: Streams that may be retried.
_RETRYABLE = frozenset({"failed", "cancelled"})


class StreamsIn(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=50)
    #: Optional range (seconds) applied to every link in this request.
    range_start_s: float | None = Field(default=None, ge=0)
    range_end_s: float | None = Field(default=None, gt=0)

    @field_validator("urls")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for url in value:
            url = url.strip()
            if url and url not in out:
                out.append(url)
        if not out:
            raise ValueError("Paste at least one stream link.")
        return out


class LiveIn(BaseModel):
    url: str


class FeedbackIn(BaseModel):
    channel_url: str


class TwitchCredentialsIn(BaseModel):
    client_id: str = ""
    client_secret: str = ""


def _stream_out(
    stream: store.GtaStream, clips: list[store.GtaClip] | None = None
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for clip in clips or []:
        counts[clip.status] = counts.get(clip.status, 0) + 1
    return {
        "id": stream.id,
        "url": stream.url,
        "platform": stream.platform,
        "title": stream.title,
        "channel": stream.channel,
        "duration_s": stream.duration_s,
        "range_start_s": stream.range_start_s,
        "range_end_s": stream.range_end_s,
        "status": stream.status,
        "stage": stream.stage,
        "progress": stream.progress,
        "detail": stream.detail,
        "error": stream.error,
        "signals": stream.signals,
        "is_live": stream.signals.get("mode") == "live",
        "clip_counts": counts,
        "created_at": stream.created_at,
        "updated_at": stream.updated_at,
    }


def _upload_out(upload: store.GtaUpload | None) -> dict[str, Any] | None:
    if upload is None:
        return None
    return {
        "id": upload.id,
        "clip_id": upload.clip_id,
        "platform": upload.platform,
        "status": upload.status,
        "slot_at": upload.slot_at,
        "video_id": upload.video_id,
        "url": upload.url,
        "media_url": upload.media_url,
        "title": upload.title,
        "error": upload.error,
        "attempts": upload.attempts,
        "updated_at": upload.updated_at,
    }


def _clip_out(
    clip: store.GtaClip, uploads: dict[str, store.GtaUpload] | None = None
) -> dict[str, Any]:
    exists = bool(clip.path) and Path(clip.path).exists()
    return {
        "id": clip.id,
        "stream_id": clip.stream_id,
        "status": clip.status,
        "start_s": clip.start_s,
        "end_s": clip.end_s,
        "duration_s": round(clip.end_s - clip.start_s, 2),
        "hook_title": clip.hook_title,
        "hook_quote": clip.hook_quote,
        "moment_type": clip.moment_type,
        "summary": clip.summary,
        "hype_score": clip.hype_score,
        "score": clip.score,
        "rank": clip.rank,
        "verdict": clip.verdict,
        "reject_reason": clip.reject_reason,
        "caption": clip.caption,
        "hashtags": clip.hashtags,
        "post_text": post_text(clip),
        "path": clip.path,
        "has_video": exists,
        "qa": clip.qa,
        "error": clip.error,
        "uploads": {p: _upload_out(u) for p, u in (uploads or {}).items()},
        # v8: hook options, flash-forward, variants, caption language.
        "hook_options": clip.hook_options,
        "flash_forward": clip.flash_forward,
        "teaser_start_s": clip.teaser_start_s,
        "teaser_end_s": clip.teaser_end_s,
        "variant_of": clip.variant_of,
        "family_id": clip.family_id,
        "motion_seed": clip.motion_seed,
        "caption_style": clip.caption_style,
        "localised": clip.localised,
        "source": str(clip.verdict.get("source") or ("variant" if clip.variant_of else "judge")),
    }


def _uploads_by_clip() -> dict[str, dict[str, store.GtaUpload]]:
    out: dict[str, dict[str, store.GtaUpload]] = {}
    for upload in store.list_uploads(limit=100_000):
        out.setdefault(upload.clip_id, {})[upload.platform] = upload
    return out


# --------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------


@router.get("/streams")
async def list_streams() -> list[dict[str, Any]]:
    def work() -> list[dict[str, Any]]:
        streams = store.list_streams(limit=500)
        clips = store.list_clips(limit=100_000)
        by_stream: dict[str, list[store.GtaClip]] = {}
        for clip in clips:
            by_stream.setdefault(clip.stream_id, []).append(clip)
        return [_stream_out(s, by_stream.get(s.id, [])) for s in streams]

    return await asyncio.to_thread(work)


@router.post("/streams", status_code=201)
async def add_streams(payload: StreamsIn) -> dict[str, Any]:
    if (
        payload.range_start_s is not None
        and payload.range_end_s is not None
        and payload.range_end_s <= payload.range_start_s
    ):
        raise HTTPException(status_code=400, detail="The range end must be after its start.")

    def work() -> dict[str, Any]:
        existing = {
            s.url
            for s in store.list_streams(limit=10_000)
            if s.status not in _RETRYABLE and s.signals.get("mode") != "live"
        }
        added: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for url in payload.urls:
            if not is_supported_url(url):
                skipped.append({"url": url, "reason": "Not a YouTube, Twitch or Kick link."})
                continue
            if url in existing:
                # Never clip the exact same moments twice.
                skipped.append({"url": url, "reason": "Already queued or clipped."})
                continue
            stream = store.create_stream(
                url, range_start_s=payload.range_start_s, range_end_s=payload.range_end_s
            )
            store.update_stream(stream.id, platform=detect_platform(url))
            existing.add(url)
            added.append(_stream_out(store.get_stream(stream.id) or stream))
        return {"added": added, "skipped": skipped}

    result = await asyncio.to_thread(work)
    queue.notify()
    return result


def _get_stream(stream_id: str) -> store.GtaStream:
    stream = store.get_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found.")
    return stream


@router.get("/streams/{stream_id}")
async def get_stream(stream_id: str) -> dict[str, Any]:
    stream = await asyncio.to_thread(_get_stream, stream_id)
    clips = await asyncio.to_thread(store.list_clips, stream_id)
    uploads = await asyncio.to_thread(_uploads_by_clip)
    return {
        **_stream_out(stream, clips),
        "clips": [_clip_out(c, uploads.get(c.id)) for c in clips],
    }


@router.post("/streams/{stream_id}/cancel")
async def cancel_stream(stream_id: str) -> dict[str, Any]:
    stream = await asyncio.to_thread(_get_stream, stream_id)
    if stream.signals.get("mode") == "live":
        from ..gta.live import live_manager

        live_manager.stop(stream_id)
        return _stream_out(stream)
    if stream.status not in store.ACTIVE_STREAM_STATUSES:
        raise HTTPException(status_code=409, detail="This stream is not running.")
    await asyncio.to_thread(
        store.update_stream, stream_id, status="cancelled", detail="Cancelled", progress=0.0
    )
    queue.cancel(stream_id)
    return _stream_out(await asyncio.to_thread(_get_stream, stream_id))


@router.post("/streams/{stream_id}/retry")
async def retry_stream(stream_id: str) -> dict[str, Any]:
    stream = await asyncio.to_thread(_get_stream, stream_id)
    if stream.signals.get("mode") == "live":
        from ..gta.live import live_manager

        if live_manager.is_running(stream_id):
            raise HTTPException(
                status_code=409, detail="This live stream is already being watched."
            )
        await asyncio.to_thread(live_manager.restart, stream_id)
        return _stream_out(await asyncio.to_thread(_get_stream, stream_id))
    if stream.status not in _RETRYABLE:
        raise HTTPException(
            status_code=409, detail="Only failed or cancelled streams can be retried."
        )
    resume = stream.stage if stream.stage in store.ACTIVE_STREAM_STATUSES else "queued"
    if resume == "probing":
        resume = "queued"

    def work() -> None:
        store.update_stream(stream_id, status=resume, error=None, detail="", progress=0.0)
        # Clips that failed on a transient error get another go.
        for clip in store.list_clips(stream_id, statuses=frozenset({"failed"})):
            store.update_clip(clip.id, status="approved", error=None)

    await asyncio.to_thread(work)
    queue.notify()
    return _stream_out(await asyncio.to_thread(_get_stream, stream_id))


@router.delete("/streams/{stream_id}", status_code=204)
async def delete_stream(stream_id: str) -> None:
    stream = await asyncio.to_thread(_get_stream, stream_id)
    if stream.signals.get("mode") == "live":
        from ..gta.live import live_manager

        if live_manager.is_running(stream_id):
            raise HTTPException(status_code=409, detail="Stop the live watch before removing it.")
    elif stream.status in store.ACTIVE_STREAM_STATUSES:
        raise HTTPException(status_code=409, detail="Cancel the stream before removing it.")
    if any(k.startswith((f"stream:{stream_id}", f"clip:{stream_id}:")) for k in queue.busy_keys()):
        raise HTTPException(
            status_code=409, detail="The stream is still stopping; try again shortly."
        )
    from ..gta.poster import poster

    for uploading in poster.uploading_clip_ids():
        clip = await asyncio.to_thread(store.get_clip, uploading)
        if clip is not None and clip.stream_id == stream_id:
            raise HTTPException(
                status_code=409,
                detail="One of this stream's clips is uploading right now; try again shortly.",
            )
    await asyncio.to_thread(store.delete_stream, stream_id)
    # Working files only (audio, sections). Finished clips stay where they are.
    await asyncio.to_thread(cleanup_stream_files, stream_id)


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


@router.get("/clips")
async def list_clips(status: str | None = None) -> list[dict[str, Any]]:
    statuses = frozenset({status}) if status else None
    clips = await asyncio.to_thread(store.list_clips, None, statuses=statuses, limit=100_000)
    uploads = await asyncio.to_thread(_uploads_by_clip)
    return [_clip_out(c, uploads.get(c.id)) for c in clips]


class ClipEditIn(BaseModel):
    """Change a finished clip and render it again (fields left out are kept)."""

    hook_title: str | None = Field(default=None, max_length=120)
    #: True/False override the GTA setting; null follows it again.
    flash_forward: bool | None = None


class PickRangeIn(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)


class PicksIn(BaseModel):
    ranges: list[PickRangeIn] = Field(min_length=1, max_length=50)


def _edit_error(exc: Exception) -> HTTPException:
    from ..gta.edits import EditError

    status = exc.status if isinstance(exc, EditError) else 400
    return HTTPException(status_code=status, detail=str(exc))


@router.patch("/clips/{clip_id}")
async def edit_clip(clip_id: str, payload: ClipEditIn) -> dict[str, Any]:
    """Pick another hook title and/or flip the flash-forward; then re-render."""
    from ..gta import edits

    sent = payload.model_fields_set
    if not sent & {"hook_title", "flash_forward"}:
        raise HTTPException(status_code=400, detail="Nothing to change.")
    try:
        clip = await asyncio.to_thread(
            edits.rerender,
            clip_id,
            hook_title=payload.hook_title if "hook_title" in sent else None,
            flash_forward=payload.flash_forward,
            flash_forward_set="flash_forward" in sent,
        )
    except edits.EditError as exc:
        raise _edit_error(exc) from exc
    return _clip_out(clip)


@router.post("/clips/{clip_id}/variant", status_code=201)
async def make_variant(clip_id: str) -> dict[str, Any]:
    """A new clip of the same moment: next caption look, next hook, new motion."""
    from ..gta import edits

    try:
        clip = await asyncio.to_thread(edits.make_variant, clip_id)
    except edits.EditError as exc:
        raise _edit_error(exc) from exc
    return _clip_out(clip)


@router.get("/streams/{stream_id}/hype")
async def stream_hype(stream_id: str) -> dict[str, Any]:
    """The stream's Hype Curve (pooled), its markers and its clips, for picking."""
    from ..gta import edits

    try:
        return await asyncio.to_thread(edits.hype_graph, stream_id)
    except edits.EditError as exc:
        raise _edit_error(exc) from exc


@router.post("/streams/{stream_id}/picks", status_code=201)
async def make_picks(stream_id: str, payload: PicksIn) -> dict[str, Any]:
    """Clips from ranges dragged on the Hype Curve (no reject phase)."""
    from ..gta import edits

    ranges = [(r.start_s, r.end_s) for r in payload.ranges]
    try:
        clips = await asyncio.to_thread(edits.make_picks, stream_id, ranges)
    except edits.EditError as exc:
        raise _edit_error(exc) from exc
    return {"clips": [_clip_out(c) for c in clips]}


@router.get("/clips/{clip_id}/video")
async def clip_video(clip_id: str) -> FileResponse:
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None or not clip.path or not Path(clip.path).exists():
        raise HTTPException(status_code=404, detail="That clip has no video yet.")
    path = Path(clip.path)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


# --------------------------------------------------------------------------
# GTA Live
# --------------------------------------------------------------------------


@router.post("/live", status_code=201)
async def start_live(payload: LiveIn) -> dict[str, Any]:
    from ..gta.live import live_manager

    url = payload.url.strip()
    if not is_supported_url(url):
        raise HTTPException(status_code=400, detail="Not a YouTube, Twitch or Kick link.")
    for stream in await asyncio.to_thread(store.list_streams, limit=10_000):
        if stream.url == url and live_manager.is_running(stream.id):
            raise HTTPException(status_code=409, detail="That stream is already being watched.")
    stream = await asyncio.to_thread(live_manager.start, url)
    return _stream_out(stream)


@router.get("/live/{stream_id}/frame")
async def live_frame(stream_id: str) -> Response:
    """A near-live JPEG of a watched stream (the newest finished buffer segment)."""
    from ..gta.live import latest_frame

    data = await asyncio.to_thread(latest_frame, stream_id)
    if not data:
        raise HTTPException(status_code=404, detail="No picture from that stream yet.")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/live/{stream_id}/judge-now")
async def live_judge_now(stream_id: str) -> dict[str, Any]:
    """Send the stream's last minute to the AI judge now (a manual end-to-end check)."""
    from ..gta.live import live_manager

    if not live_manager.judge_now(stream_id):
        raise HTTPException(status_code=409, detail="That stream isn't being watched.")
    return {"queued": True}


@router.get("/ai-usage")
async def ai_usage() -> dict[str, Any]:
    """Calls, tokens, audio seconds and rate-limit headroom per AI key (today + total)."""
    from ..providers import usage

    return await asyncio.to_thread(usage.snapshot)


@router.post("/live/{stream_id}/stop")
async def stop_live(stream_id: str) -> dict[str, Any]:
    from ..gta.live import live_manager

    stream = await asyncio.to_thread(_get_stream, stream_id)
    live_manager.stop(stream_id)
    return _stream_out(stream)


# --------------------------------------------------------------------------
# Settings + feedback loop
# --------------------------------------------------------------------------


@router.get("/settings")
async def get_settings() -> dict[str, Any]:
    from ..gta.signals import twitch_credentials

    settings = await asyncio.to_thread(load_settings)
    has_twitch = await asyncio.to_thread(twitch_credentials)
    return {**settings.model_dump(mode="json"), "twitch_credentials_set": bool(has_twitch)}


@router.put("/settings")
async def put_settings(payload: dict[str, Any]) -> dict[str, Any]:
    current = await asyncio.to_thread(load_settings)
    merged = {**current.model_dump(mode="json"), **payload}
    merged.pop("twitch_credentials_set", None)
    # The learned bias is owned by the feedback loop, not the settings form.
    merged["bias"] = current.bias.model_dump(mode="json")
    try:
        validated = GtaSettings.model_validate(merged)
    except ValidationError as exc:
        # "Slot '25:00' is not a valid time of day", not pydantic's dump.
        detail = "; ".join(
            str(err.get("msg", "")).removeprefix("Value error, ") for err in exc.errors()
        )
        raise HTTPException(status_code=400, detail=detail or "Invalid settings.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    problem = await asyncio.to_thread(_logo_problem, validated)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    await asyncio.to_thread(save_settings, validated)
    from ..gta.poster import poster

    poster.notify()  # a changed schedule takes effect now, not at the next tick
    return await get_settings()


def _logo_problem(settings: GtaSettings) -> str:
    """Why the configured logo can't be used ('' = fine), checked before saving."""
    from ..gta import logo

    raw = settings.watermark_path.strip().strip('"')
    if raw:
        return "" if Path(raw).expanduser().is_file() else f"Logo file not found: {raw}"
    if not settings.watermark_text.strip():
        return ""
    try:
        logo.text_logo(settings.watermark_text, settings.watermark_font)
    except (logo.LogoError, OSError) as exc:
        return str(exc)
    return ""


@router.get("/logo")
async def current_logo() -> FileResponse:
    """The logo PNG clips are watermarked with (for the settings preview)."""
    from ..gta.pipeline import logo_path

    settings = await asyncio.to_thread(load_settings)
    path = await asyncio.to_thread(logo_path, settings)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="No logo set.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.put("/twitch-credentials")
async def put_twitch_credentials(payload: TwitchCredentialsIn) -> dict[str, Any]:
    from ..config import delete_secret, set_secret
    from ..gta.signals import TWITCH_CLIENT_ID_KEY, TWITCH_CLIENT_SECRET_KEY

    client_id = payload.client_id.strip()
    secret = payload.client_secret.strip()
    if bool(client_id) != bool(secret):
        raise HTTPException(
            status_code=400, detail="Enter both the Client ID and the Client Secret."
        )
    if client_id:
        await asyncio.to_thread(set_secret, TWITCH_CLIENT_ID_KEY, client_id)
        await asyncio.to_thread(set_secret, TWITCH_CLIENT_SECRET_KEY, secret)
    else:
        await asyncio.to_thread(delete_secret, TWITCH_CLIENT_ID_KEY)
        await asyncio.to_thread(delete_secret, TWITCH_CLIENT_SECRET_KEY)
    return {"twitch_credentials_set": bool(client_id)}


@router.get("/feedback")
async def latest_feedback() -> dict[str, Any] | None:
    return await asyncio.to_thread(store.latest_report)


@router.post("/feedback")
async def run_feedback(payload: FeedbackIn) -> dict[str, Any]:
    from ..gta.feedback import look_at_whats_working
    from ..pipeline.ingest import IngestError
    from ..providers.base import ProviderError

    url = payload.channel_url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Paste the channel's full link.")
    try:
        report = await asyncio.to_thread(look_at_whats_working, url)
    except (IngestError, ProviderError, ValueError) as exc:
        hint = getattr(exc, "hint", "")
        raise HTTPException(status_code=400, detail=f"{exc}\n{hint}".strip()) from exc
    return report


# --------------------------------------------------------------------------
# Posting schedule (YouTube + Instagram via Zernio)
# --------------------------------------------------------------------------


class SkipPostingIn(BaseModel):
    skip: bool


def _posting_overview() -> dict[str, Any]:
    from ..gta.poster import poster
    from ..gta.settings import YOUTUBE_CATEGORIES

    settings = load_settings()
    clips = {c.id: c for c in store.list_clips(statuses=frozenset({"ready"}), limit=100_000)}
    uploads = store.list_uploads(limit=200)
    for upload in uploads:  # titles of rows whose clip is no longer "ready"
        if upload.clip_id not in clips:
            clip = store.get_clip(upload.clip_id)
            if clip is not None:
                clips[clip.id] = clip

    def title(clip_id: str | None) -> str:
        clip = clips.get(clip_id or "")
        return clip.hook_title if clip else ""

    plan = poster.plan(settings)
    for entry in plan:
        entry["title"] = title(entry.get("clip_id"))
    return {
        "states": {p: st.as_dict() for p, st in poster.states.items()},
        "running": poster.running,
        "posting": settings.posting.model_dump(mode="json"),
        "categories": YOUTUBE_CATEGORIES,
        "plan": plan,
        "uploads": [
            {**(_upload_out(u) or {}), "clip_title": title(u.clip_id) or u.title} for u in uploads
        ],
    }


@router.get("/posting")
async def posting_overview() -> dict[str, Any]:
    return await asyncio.to_thread(_posting_overview)


@router.post("/posting/run")
async def posting_run_now() -> dict[str, Any]:
    """Wake the poster now (e.g. right after connecting or saving the schedule)."""
    from ..gta.poster import poster

    poster.notify()
    return {"running": poster.running}


class PostNowIn(BaseModel):
    title: str = Field(default="", max_length=100)
    #: When set (ISO time with timezone), schedule instead of posting now.
    publish_at: datetime | None = None


@router.post("/clips/{clip_id}/post-now")
async def post_clip_now(clip_id: str, payload: PostNowIn) -> dict[str, Any]:
    """Upload a ready clip to YouTube as public right now, with an optional title."""
    from ..gta.poster import poster

    try:
        outcome = await poster.post_now(clip_id, payload.title.strip(), payload.publish_at)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    row = await asyncio.to_thread(store.upload_for_clip, clip_id, "youtube")
    return {
        "outcome": outcome,
        "upload": _upload_out(row),
        "message": poster.states["youtube"].message,
    }


@router.post("/clips/{clip_id}/posting")
async def set_clip_posting(clip_id: str, payload: SkipPostingIn) -> dict[str, Any]:
    """Exclude a clip from auto-posting on every platform (skip) or allow it again.

    Skipping marks each platform the clip isn't already on (or being sent
    to); rows that are scheduled/posted/checking there are left alone.
    """

    def apply() -> dict[str, store.GtaUpload]:
        if store.get_clip(clip_id) is None:
            raise HTTPException(status_code=404, detail="Clip not found.")
        rows = store.uploads_for_clip(clip_id)
        if payload.skip:
            if any(r.status == "uploading" for r in rows.values()):
                raise HTTPException(
                    status_code=409, detail="This clip is uploading right now; try again shortly."
                )
            changed = False
            for platform in store.PLATFORMS:
                row = rows.get(platform)
                if row is None:
                    store.create_upload(clip_id, status="skipped", platform=platform)
                    changed = True
                elif row.status in {"failed", "skipped"}:
                    store.update_upload(row.id, status="skipped", slot_at=None)
                    changed = True
            if not changed:
                raise HTTPException(
                    status_code=409, detail="This clip is already posted or scheduled everywhere."
                )
        else:
            for row in rows.values():
                if row.status == "skipped":
                    store.delete_upload(row.id)
        return store.uploads_for_clip(clip_id)

    uploads = await asyncio.to_thread(apply)
    from ..gta.poster import poster

    poster.notify()
    return {"uploads": {p: _upload_out(u) for p, u in uploads.items()}}


@router.post("/uploads/{upload_id}/retry")
async def retry_upload(upload_id: str) -> dict[str, Any]:
    """Put a failed, interrupted or checking clip back in line (fresh attempts).

    For ``checking`` (Instagram) the UI asks the user to confirm the reel is
    really not at Zernio/Instagram first: AutoClip never re-sends on its own.
    """
    from ..gta.poster import poster

    def apply() -> None:
        row = store.get_upload(upload_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Upload not found.")
        if row.status not in {"failed", "interrupted", "checking"}:
            raise HTTPException(
                status_code=409,
                detail="Only failed, interrupted or checking uploads can be retried.",
            )
        store.delete_upload(row.id)

    await asyncio.to_thread(apply)
    poster.notify()
    return {"retried": True}


# --------------------------------------------------------------------------
# Instagram via Zernio (API key + connection test)
# --------------------------------------------------------------------------


class ZernioKeyIn(BaseModel):
    api_key: str = ""


class ZernioTestIn(BaseModel):
    account: str = ""


def _instagram_status() -> dict[str, Any]:
    from ..gta import zernio

    try:
        saved = bool(zernio.load_api_key())
    except Exception:
        saved = False
    return {"key_saved": saved, "reelrepost_key_available": bool(zernio.reelrepost_api_key())}


@router.get("/instagram")
async def instagram_status() -> dict[str, Any]:
    return await asyncio.to_thread(_instagram_status)


@router.put("/instagram/key")
async def put_instagram_key(payload: ZernioKeyIn) -> dict[str, Any]:
    from ..gta import zernio
    from ..gta.poster import poster

    try:
        await asyncio.to_thread(zernio.save_api_key, payload.api_key.strip())
    except zernio.ZernioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    poster.notify()
    return await asyncio.to_thread(_instagram_status)


@router.post("/instagram/import-reelrepost")
async def import_reelrepost_key() -> dict[str, Any]:
    """Copy the Zernio key ReelRepost saved (it never leaves this PC)."""
    from ..gta import zernio
    from ..gta.poster import poster

    key = await asyncio.to_thread(zernio.reelrepost_api_key)
    if not key:
        raise HTTPException(status_code=404, detail="ReelRepost has no Zernio API key saved.")
    try:
        await asyncio.to_thread(zernio.save_api_key, key)
    except zernio.ZernioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    poster.notify()
    return await asyncio.to_thread(_instagram_status)


@router.post("/instagram/test")
async def test_instagram(payload: ZernioTestIn) -> dict[str, Any]:
    from ..gta import zernio

    key = await asyncio.to_thread(zernio.load_api_key)
    if not key:
        return {"ok": False, "message": "Save your Zernio API key first.", "username": ""}
    account = payload.account.strip().lstrip("@")
    ok, message, username = await asyncio.to_thread(zernio.test_connection, key, account)
    return {"ok": ok, "message": message, "username": username}
