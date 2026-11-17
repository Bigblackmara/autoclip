"""GTA Mode API: the stream queue, clips, GTA Live, settings, feedback loop."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

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


def _clip_out(clip: store.GtaClip) -> dict[str, Any]:
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
    }


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
    return {**_stream_out(stream, clips), "clips": [_clip_out(c) for c in clips]}


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
    return [_clip_out(c) for c in clips]


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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    problem = await asyncio.to_thread(_logo_problem, validated)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    await asyncio.to_thread(save_settings, validated)
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
