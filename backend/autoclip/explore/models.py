"""What Explore returns, plus small readers for loosely typed yt-dlp JSON."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ItemKind = Literal["video", "short", "channel", "live", "vod", "clip"]
Platform = Literal["youtube", "twitch"]


class ExploreItem(BaseModel):
    """One result card: a video, Short, live stream, channel, Twitch VOD or clip."""

    kind: ItemKind
    source: Platform
    id: str
    url: str
    title: str
    channel: str | None = None
    channel_id: str | None = None
    channel_url: str | None = None
    thumbnail: str | None = None
    duration_s: float | None = None
    views: int | None = None
    subscribers: int | None = None
    #: ISO date (YYYY-MM-DD) when known.
    upload_date: str | None = None
    is_live: bool = False
    description: str | None = None


class ExploreChannel(BaseModel):
    source: Platform
    title: str
    url: str
    avatar: str | None = None
    banner: str | None = None
    subscribers: int | None = None
    description: str | None = None


class ChannelPage(BaseModel):
    channel: ExploreChannel
    tab: str
    items: list[ExploreItem]


class SearchResult(BaseModel):
    items: list[ExploreItem]
    #: True when Twitch keyword search is available (app credentials saved).
    twitch_search: bool = False
    #: Set when the query was taken as something other than a keyword search
    #: (e.g. a Twitch channel name without credentials).
    note: str | None = None


class HeatPoint(BaseModel):
    start_s: float
    end_s: float
    value: float


class Peak(BaseModel):
    start_s: float
    end_s: float
    peak_s: float
    value: float
    reasons: list[str] = Field(default_factory=list)


class ChapterOut(BaseModel):
    start_s: float
    end_s: float
    title: str


class PeekResult(BaseModel):
    platform: str
    id: str
    url: str
    title: str
    channel: str | None = None
    channel_url: str | None = None
    duration_s: float = 0.0
    views: int | None = None
    likes: int | None = None
    upload_date: str | None = None
    thumbnail: str | None = None
    is_live: bool = False
    description: str = ""
    chapters: list[ChapterOut] = Field(default_factory=list)
    #: YouTube "Most replayed", as yt-dlp reports it (~100 points), 0..1.
    heatmap: list[HeatPoint] = Field(default_factory=list)
    #: The strongest rewatch peaks, strongest first (at most 5).
    peaks: list[Peak] = Field(default_factory=list)


Raw = dict[str, Any]


def text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if value == value else None  # NaN check
    return None


def integer(value: Any) -> int | None:
    n = number(value)
    return int(n) if n is not None else None


def best_thumbnail(entry: Raw) -> str | None:
    """The widest thumbnail yt-dlp listed, else its single ``thumbnail``."""
    thumbs = [
        t for t in entry.get("thumbnails") or [] if isinstance(t, dict) and text(t.get("url"))
    ]
    thumbs.sort(key=lambda t: -(number(t.get("width")) or 0))
    url = text(thumbs[0]["url"]) if thumbs else text(entry.get("thumbnail"))
    if url and url.startswith("//"):
        url = "https:" + url
    return url


def iso_date(entry: Raw) -> str | None:
    """``upload_date`` (YYYYMMDD) or a unix ``timestamp`` as YYYY-MM-DD."""
    raw = text(entry.get("upload_date"))
    if raw and len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    ts = number(entry.get("timestamp")) or number(entry.get("release_timestamp"))
    if ts:
        return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
    return None
