"""Posting normal-mode clips, the Schedule page, and the health report."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..publishing import clipsource, remote, service
from ..publishing import posts as post_store
from ..publishing import settings as publish_settings
from ..publishing.core import instagram_readiness, readiness
from ..publishing.poster import CancelError, poster

router = APIRouter(tags=["publishing"])


class PostIn(BaseModel):
    clip_id: str
    platforms: list[Literal["youtube", "instagram"]] = Field(min_length=1)
    #: None = post now; else an ISO time with a zone (UTC "Z" is fine).
    scheduled_at: datetime | None = None
    #: YouTube privacy for a post-now (a scheduled post is public at its time).
    privacy: Literal["public", "unlisted", "private"] = "public"
    title: str | None = Field(default=None, max_length=100)
    caption: str | None = Field(default=None, max_length=2200)
    hashtags: list[str] | None = None


class CancelIn(BaseModel):
    #: YouTube only: "private" unschedules (stays private), "delete" removes it.
    action: Literal["private", "delete"] = "private"


def _readiness() -> dict[str, Any]:
    return {
        "youtube": readiness(),
        "instagram": instagram_readiness(),
        "youtube_can_edit": remote.youtube_can_edit(),
    }


@router.get("/api/publishing/status")
async def publishing_status() -> dict[str, Any]:
    ready = await asyncio.to_thread(_readiness)
    settings = await asyncio.to_thread(publish_settings.load)
    return {**poster.overview(), "ready": ready, "settings": settings.model_dump(mode="json")}


@router.put("/api/publishing/settings")
async def put_publishing_settings(payload: publish_settings.PublishSettings) -> dict[str, Any]:
    saved = await asyncio.to_thread(publish_settings.save, payload)
    poster.notify()
    return saved.model_dump(mode="json")


@router.get("/api/publishing/clips")
async def postable_clips(limit: int = 40) -> list[dict[str, Any]]:
    """Recently exported clips, with each platform's active post (if any)."""

    def build() -> list[dict[str, Any]]:
        out = []
        for item in clipsource.recent(limit=max(1, min(limit, 200))):
            active = service.clip_posts(item.clip_id)["active"]
            out.append(
                {
                    "clip_id": item.clip_id,
                    "job_id": item.job_id,
                    "title": item.title,
                    "text": item.text,
                    "hashtags": item.hashtags,
                    "source_title": item.source_title,
                    "export_id": item.export_id,
                    "download_url": f"/api/exports/{item.export_id}/download"
                    if item.export_id
                    else None,
                    "active": active,
                }
            )
        return out

    return await asyncio.to_thread(build)


@router.get("/api/publishing/clips/{clip_id}")
async def clip_posting(clip_id: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        item = clipsource.find(clip_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Clip not found.")
        return {
            "clip_id": item.clip_id,
            "title": item.title,
            "text": item.text,
            "hashtags": item.hashtags,
            "exported": item.ready,
            **service.clip_posts(clip_id),
        }

    return await asyncio.to_thread(build)


@router.get("/api/publishing/posts")
async def list_posts(clip_id: str | None = None) -> list[dict[str, Any]]:
    rows = await asyncio.to_thread(post_store.list_posts, clip_id=clip_id)
    return [p.as_dict() for p in rows]


@router.post("/api/publishing/posts", status_code=201)
async def create_posts(payload: PostIn) -> dict[str, Any]:
    when = payload.scheduled_at
    if when is not None and when.tzinfo is None:
        when = when.replace(tzinfo=UTC)

    def apply() -> list[post_store.Post]:
        try:
            return service.create_posts(
                payload.clip_id,
                list(payload.platforms),
                scheduled_at=when,
                privacy=payload.privacy,
                title=payload.title,
                caption=payload.caption,
                hashtags=payload.hashtags,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="Clip not found.") from exc
        except post_store.DoublePostError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except service.PostRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    made = await asyncio.to_thread(apply)
    poster.notify()
    ready = await asyncio.to_thread(_readiness)
    warnings = [ready[p.platform] for p in made if ready.get(p.platform)]
    return {"posts": [p.as_dict() for p in made], "warnings": list(dict.fromkeys(warnings))}


@router.post("/api/publishing/posts/{post_id}/cancel")
async def cancel_post(post_id: str, payload: CancelIn | None = None) -> dict[str, Any]:
    action = payload.action if payload else "private"
    try:
        post = await poster.cancel(post_id, action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Post not found.") from exc
    except post_store.StateChanged as exc:
        raise HTTPException(
            status_code=409, detail=f"The post changed meanwhile (now {exc.post.status})."
        ) from exc
    except CancelError as exc:
        raise HTTPException(
            status_code=409, detail={"message": str(exc), "studio": exc.studio}
        ) from exc
    return post.as_dict()


@router.post("/api/publishing/posts/{post_id}/retry")
async def retry_post(post_id: str) -> dict[str, Any]:
    def apply() -> post_store.Post:
        try:
            return poster.retry(post_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Post not found.") from exc
        except post_store.DoublePostError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (CancelError, post_store.StateChanged) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return (await asyncio.to_thread(apply)).as_dict()


@router.post("/api/publishing/run")
async def run_poster_now() -> dict[str, Any]:
    poster.notify()
    return {"running": poster.running}


@router.get("/api/schedule")
async def schedule() -> dict[str, Any]:
    """Every upcoming and past post: normal, Autopilot and GTA Mode."""
    return await asyncio.to_thread(service.schedule)


@router.get("/api/system/doctor")
async def system_doctor(live: bool = False, refresh: bool = False) -> dict[str, Any]:
    """Health checks (ok / warn / fail + fix). ``live`` contacts the AI and Zernio."""
    from .. import doctor, system

    def build() -> dict[str, Any]:
        if refresh:
            system.refresh()
        return doctor.doctor_report(live=live)

    return await asyncio.to_thread(build)
