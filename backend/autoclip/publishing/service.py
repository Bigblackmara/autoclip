"""Creating posts and the Schedule page's view of every post.

The Schedule shows normal-mode posts (JSON records) next to GTA Mode's
uploads (the ``gta_uploads`` table, read-only here) and GTA's planned
slots, with what can be done about each one.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from . import clipsource, remote
from . import posts as store
from .core import MIN_LEAD_S, PLATFORMS, iso, parse_iso
from .posts import DoublePostError, Post

log = logging.getLogger(__name__)

#: A scheduled post must be at least this far ahead (else use "now").
MIN_SCHEDULE_AHEAD_S = MIN_LEAD_S + 180
MAX_SCHEDULE_AHEAD = timedelta(days=180)


class PostRequestError(ValueError):
    """The request can't be turned into posts (bad time, no file, ...)."""


def create_posts(
    clip_id: str,
    platforms: list[str],
    *,
    scheduled_at: datetime | None = None,
    privacy: str = "public",
    title: str | None = None,
    caption: str | None = None,
    hashtags: list[str] | None = None,
    autopilot_id: str = "",
    slot_key: str = "",
    now: datetime | None = None,
) -> list[Post]:
    """One queued post per platform for an exported clip.

    Raises LookupError (no such clip), PostRequestError (not exported, bad
    time, no platform) or DoublePostError (already active on a platform:
    nothing is created then).
    """
    now = now or datetime.now(UTC)
    wanted = [p for p in dict.fromkeys(platforms) if p]
    if not wanted:
        raise PostRequestError("Pick YouTube, Instagram or both.")
    unknown = [p for p in wanted if p not in PLATFORMS]
    if unknown:
        raise PostRequestError(f"Unknown platform: {', '.join(unknown)}")
    item = clipsource.find(clip_id)
    if item is None:
        raise LookupError(clip_id)
    if not item.ready:
        raise PostRequestError("Export this clip first: there is no rendered file to post.")
    if scheduled_at is not None:
        if scheduled_at.tzinfo is None:
            raise PostRequestError("The time needs a time zone.")
        ahead = (scheduled_at - now).total_seconds()
        if ahead < MIN_SCHEDULE_AHEAD_S:
            raise PostRequestError(
                f"Pick a time at least {int(MIN_SCHEDULE_AHEAD_S // 60)} minutes ahead, "
                "or post now."
            )
        if scheduled_at - now > MAX_SCHEDULE_AHEAD:
            raise PostRequestError("Pick a time within the next 180 days.")
    for platform in wanted:
        existing = store.active_post(clip_id, platform)
        if existing is not None:
            raise DoublePostError(existing)
    made: list[Post] = []
    for platform in wanted:
        made.append(
            store.create_post(
                clip_id,
                platform,
                job_id=item.job_id,
                scheduled_at=iso(scheduled_at) if scheduled_at else None,
                privacy=privacy,
                title=(title if title is not None else item.title)[:100] or "New clip",
                caption=(caption if caption is not None else item.text)[:2200],
                hashtags=list(hashtags if hashtags is not None else item.hashtags),
                autopilot_id=autopilot_id,
                slot_key=slot_key,
            )
        )
    return made


# --------------------------------------------------------------------------
# The Schedule page
# --------------------------------------------------------------------------


def _normal_item(post: Post, now: datetime, can_edit: bool) -> dict[str, Any]:
    when = parse_iso(post.scheduled_at) or parse_iso(post.sent_at) or parse_iso(post.created_at)
    upcoming = post.status in {"queued", "uploading", "checking"} or (
        post.status == "scheduled" and (when is None or when > now)
    )
    cancel_options: list[str] = []
    cancel_note = ""
    # queued: withdrawn here; failed: stops the automatic retries
    can_cancel = post.status in {"queued", "failed"}
    if post.status == "scheduled" and upcoming:
        if post.platform == "instagram":
            can_cancel = bool(post.remote_id)
            if not can_cancel:
                cancel_note = "Zernio never confirmed a post id; remove it at zernio.com."
        elif can_edit:
            can_cancel = True
            cancel_options = ["private", "delete"]
        else:
            cancel_note = remote.STUDIO_NOTE
    return {
        "key": f"normal:{post.id}",
        "source": "autopilot" if post.autopilot_id else "normal",
        "id": post.id,
        "clip_id": post.clip_id,
        "job_id": post.job_id,
        "autopilot_id": post.autopilot_id,
        "platform": post.platform,
        "status": post.status,
        "when": iso(when) if when else None,
        "scheduled_at": post.scheduled_at,
        "title": post.title,
        "url": post.url,
        "error": post.error,
        "attempts": post.attempts,
        "upcoming": upcoming,
        "can_cancel": can_cancel,
        "cancel_options": cancel_options,
        "cancel_note": cancel_note,
        "can_retry": post.status in {"failed", "interrupted", "checking"},
        "retry_confirm": post.status in {"interrupted", "checking"},
    }


def _gta_items(now: datetime) -> list[dict[str, Any]]:
    """GTA Mode's uploads and planned slots (never fatal: GTA owns that data)."""
    try:
        from ..gta import store as gta_store
        from ..gta.poster import poster as gta_poster
        from ..gta.settings import load as load_gta_settings
    except Exception:  # pragma: no cover - GTA module missing/broken
        log.warning("GTA Mode data unavailable for the schedule.", exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    try:
        uploads = gta_store.list_uploads(limit=500)
        titles: dict[str, str] = {}
        for row in uploads:
            if row.clip_id not in titles:
                clip = gta_store.get_clip(row.clip_id)
                titles[row.clip_id] = clip.hook_title if clip else ""
        for row in uploads:
            if row.status == "skipped":
                continue
            when = parse_iso(row.slot_at) or parse_iso(row.updated_at)
            upcoming = row.status in {"uploading", "checking"} or (
                row.status == "scheduled" and (when is None or when > now)
            )
            out.append(
                {
                    "key": f"gta:{row.id}",
                    "source": "gta",
                    "id": row.id,
                    "clip_id": row.clip_id,
                    "job_id": "",
                    "autopilot_id": "",
                    "platform": row.platform,
                    "status": row.status,
                    "when": iso(when) if when else None,
                    "scheduled_at": row.slot_at,
                    "title": titles.get(row.clip_id) or row.title,
                    "url": row.url,
                    "error": row.error,
                    "attempts": row.attempts,
                    "upcoming": upcoming,
                    "can_cancel": False,
                    "cancel_options": [],
                    "cancel_note": "GTA Mode posts are managed in GTA Mode's Posting tab.",
                    "can_retry": row.status in {"failed", "interrupted", "checking"},
                    "retry_confirm": row.status in {"interrupted", "checking"},
                }
            )
        settings = load_gta_settings()
        if settings.posting.enabled:
            for entry in gta_poster.plan(settings, now):
                if entry.get("status") != "planned":
                    continue
                clip = gta_store.get_clip(entry.get("clip_id") or "")
                out.append(
                    {
                        "key": f"gta-plan:{entry['slot_at']}:{entry['platform']}",
                        "source": "gta",
                        "id": "",
                        "clip_id": entry.get("clip_id") or "",
                        "job_id": "",
                        "autopilot_id": "",
                        "platform": entry["platform"],
                        "status": "planned",
                        "when": entry["slot_at"],
                        "scheduled_at": entry["slot_at"],
                        "title": clip.hook_title if clip else "",
                        "url": "",
                        "error": None,
                        "attempts": 0,
                        "upcoming": True,
                        "can_cancel": False,
                        "cancel_options": [],
                        "cancel_note": "Planned by GTA Mode's schedule; skip the clip there.",
                        "can_retry": False,
                        "retry_confirm": False,
                    }
                )
    except Exception:
        log.warning("Could not read GTA Mode's posting data.", exc_info=True)
    return out


def schedule(now: datetime | None = None, *, can_edit: bool | None = None) -> dict[str, Any]:
    """Every upcoming (soonest first) and past (newest first) post."""
    now = now or datetime.now(UTC)
    if can_edit is None:
        can_edit = remote.youtube_can_edit()
    items = [_normal_item(p, now, can_edit) for p in store.list_posts()]
    items += _gta_items(now)
    upcoming = sorted((i for i in items if i["upcoming"]), key=lambda i: i["when"] or "")
    past = sorted((i for i in items if not i["upcoming"]), key=lambda i: i["when"] or "")[::-1]
    return {
        "now": iso(now),
        "youtube_can_edit": can_edit,
        "upcoming": upcoming,
        "past": past[:300],
        "counts": {
            "upcoming": len(upcoming),
            "posted": sum(1 for i in past if i["status"] == "posted"),
            "failed": sum(1 for i in items if i["status"] in {"failed", "interrupted"}),
        },
    }


def clip_posts(clip_id: str) -> dict[str, Any]:
    """The clip's post records and whether each platform is free to post to."""
    records = store.list_posts(clip_id=clip_id)
    active: dict[str, Any] = {}
    for platform in PLATFORMS:
        post = store.active_post(clip_id, platform)
        active[platform] = post.as_dict() if post else None
    return {"posts": [r.as_dict() for r in records], "active": active}


__all__ = [
    "DoublePostError",
    "PostRequestError",
    "clip_posts",
    "create_posts",
    "schedule",
]
