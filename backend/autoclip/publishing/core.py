"""Posting logic shared by GTA Mode and normal-mode posts.

GTA Mode's poster (:mod:`autoclip.gta.poster`) owns the battle-tested rules:
slot maths, retry/quota constants, connection readiness. They are imported
from there unchanged, so both posters keep exactly the same behaviour, and
generalised here where GTA's versions take a ``GtaClip``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..gta.poster import (
    IG_CAPTION_MAX,
    IG_CHECK_EVERY_S,
    IG_CONFIRM_AFTER_S,
    IG_CONFIRM_EVERY_S,
    IG_CONFIRM_FOR_S,
    IG_HASHTAGS_MAX,
    IG_INTERRUPTED_NOTE,
    IG_NOT_LISTED_NOTE,
    IG_NOT_SENT_NOTE,
    IG_PAUSE_S,
    INTERRUPTED_NOTE,
    MAX_ATTEMPTS,
    MIN_LEAD_S,
    PLATFORM_LABEL,
    QUOTA_PAUSE_S,
    RETRY_AFTER_S,
    TAGS_MAX_CHARS,
    instagram_readiness,
    parse_slot_key,
    readiness,
    slot_datetimes,
    slot_key,
)
from ..gta.poster import _is_quota_error as is_quota_error

__all__ = [
    "IG_CAPTION_MAX",
    "IG_CHECK_EVERY_S",
    "IG_CONFIRM_AFTER_S",
    "IG_CONFIRM_EVERY_S",
    "IG_CONFIRM_FOR_S",
    "IG_HASHTAGS_MAX",
    "IG_INTERRUPTED_NOTE",
    "IG_NOT_LISTED_NOTE",
    "IG_NOT_SENT_NOTE",
    "IG_PAUSE_S",
    "INTERRUPTED_NOTE",
    "MAX_ATTEMPTS",
    "MIN_LEAD_S",
    "PLATFORMS",
    "PLATFORM_LABEL",
    "QUOTA_PAUSE_S",
    "RETRY_AFTER_S",
    "age_s",
    "clean_text",
    "free_slots",
    "instagram_caption",
    "instagram_readiness",
    "is_quota_error",
    "iso",
    "parse_iso",
    "parse_slot_key",
    "readiness",
    "slot_datetimes",
    "slot_key",
    "youtube_description",
    "youtube_tags",
    "youtube_title",
]

PLATFORMS = ("youtube", "instagram")


def iso(moment: datetime) -> str:
    """UTC, second precision, ``Z`` suffix (the same shape as a slot key)."""
    return slot_key(moment)


def parse_iso(value: str | None) -> datetime | None:
    """A stored timestamp (``...Z`` / ``+00:00`` / naive = UTC) as aware UTC."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def age_s(stamp: str | None, now: datetime) -> float:
    """Seconds since ``stamp`` (infinite when unreadable)."""
    moment = parse_iso(stamp)
    return float("inf") if moment is None else (now - moment).total_seconds()


def free_slots(
    slots: list[str],
    now: datetime,
    taken: set[str],
    *,
    days: int = 3,
    tz: Any = None,
    min_lead_s: float = MIN_LEAD_S,
) -> list[datetime]:
    """Upcoming slots (wall-clock ``HH:MM`` in ``tz``) nobody holds yet.

    Like GTA's ``due_slot``: a slot closer than ``min_lead_s`` is never
    offered (no rushed or back-filled post).
    """
    out: list[datetime] = []
    for moment in slot_datetimes(slots, now, days=days, tz=tz):
        if slot_key(moment) in taken:
            continue
        if (moment - now).total_seconds() < min_lead_s:
            continue
        out.append(moment)
    return out


def clean_text(text: str) -> str:
    """YouTube rejects titles/descriptions containing angle brackets."""
    return (text or "").replace("<", "").replace(">", "").strip()


def youtube_title(*candidates: str, fallback: str = "New clip") -> str:
    for candidate in candidates:
        text = clean_text((candidate or "").splitlines()[0] if candidate else "")
        if text:
            return text[:100]
    return fallback[:100]


def youtube_tags(hashtags: list[str]) -> list[str]:
    """Hashtags as YouTube tags: no '#', deduplicated, within the 500-char limit."""
    tags: list[str] = []
    seen: set[str] = set()
    used = 0
    for tag in hashtags:
        word = clean_text(str(tag).lstrip("#"))
        if not word or word.lower() in seen:
            continue
        if used + len(word) + 1 > TAGS_MAX_CHARS:
            break
        tags.append(word)
        seen.add(word.lower())
        used += len(word) + 1
    return tags


def youtube_description(text: str, hashtags: list[str], *, shorts_tag: bool = True) -> str:
    tags = " ".join(_hashtags(hashtags, limit=15))
    description = clean_text(f"{text}\n\n{tags}".strip())
    if shorts_tag and "#shorts" not in description.lower():
        description = f"{description} #Shorts".strip()
    return description[:5000]


def _hashtags(hashtags: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in hashtags:
        word = str(tag).strip().lstrip("#")
        if not word or " " in word or word.lower() in seen:
            continue
        out.append(f"#{word}")
        seen.add(word.lower())
        if len(out) == limit:
            break
    return out


def instagram_caption(text: str, hashtags: list[str]) -> str:
    """Caption + hashtags for a Reel, within Instagram's 2200 chars / 30 tags."""
    tail = " ".join(_hashtags(hashtags, limit=IG_HASHTAGS_MAX))
    room = IG_CAPTION_MAX - (len(tail) + 2 if tail else 0)
    body = (text or "").strip()
    if len(body) > room:
        body = body[: max(0, room - 1)].rstrip() + "…"
    return f"{body}\n\n{tail}".strip()[:IG_CAPTION_MAX]
