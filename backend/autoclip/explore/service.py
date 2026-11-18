"""Explore service: search, channel browse and Peek, cached.

Every function here is synchronous and blocking (yt-dlp does network I/O);
the API runs them in a worker thread.
"""

from __future__ import annotations

import json
from typing import Any

from . import twitch, youtube, ytdlp
from .cache import cache
from .models import (
    ChannelPage,
    ChapterOut,
    HeatPoint,
    Peak,
    PeekResult,
    SearchResult,
    best_thumbnail,
    integer,
    iso_date,
    number,
    text,
)
from .ytdlp import ExploreError

__all__ = ["ExploreError", "channel", "clear_cache", "peek", "search", "status"]

SEARCH_TTL_S = 10 * 60
PEEK_TTL_S = 10 * 60
#: Peaks shown in the Peek panel.
MAX_PEAKS = 5
#: Description characters returned by Peek.
DESCRIPTION_CHARS = 1200


def _key(kind: str, **params: Any) -> str:
    return f"{kind}:{json.dumps(params, sort_keys=True)}"


def clear_cache() -> None:
    cache.clear()


def status() -> dict[str, bool]:
    return {"twitch_search": twitch.credentials() is not None}


def search(
    query: str,
    *,
    source: str = "youtube",
    type: str = "videos",
    duration: str = "any",
    sort: str = "relevance",
    uploaded: str = "any",
    limit: int = 24,
) -> SearchResult:
    """YouTube search with filters, or Twitch channel search."""
    query = query.strip()
    if not query:
        raise ExploreError("Type something to search for.")
    if source == "twitch":
        return _twitch_search(query, limit=limit)
    key = _key(
        "yt-search",
        q=query,
        type=type,
        duration=duration,
        sort=sort,
        uploaded=uploaded,
        limit=limit,
    )
    items = cache.cached(
        key,
        lambda: youtube.search(
            query, type=type, duration=duration, sort=sort, uploaded=uploaded, limit=limit
        ),
        SEARCH_TTL_S,
    )
    return SearchResult(items=items, twitch_search=status()["twitch_search"])


def _twitch_search(query: str, *, limit: int) -> SearchResult:
    creds = twitch.credentials()
    if creds is None:
        login = twitch.login_of(query)
        if not twitch.is_valid_login(login):
            return SearchResult(
                items=[],
                twitch_search=False,
                note=(
                    "Without Twitch app credentials a search is read as a channel name. "
                    "Save a Client ID and secret in GTA Mode settings for keyword search."
                ),
            )
        return SearchResult(
            items=[twitch.channel_card(login)],
            twitch_search=False,
            note="Read as a channel name (keyword search needs Twitch app credentials).",
        )
    items = cache.cached(
        _key("tw-search", q=query, limit=limit),
        lambda: twitch.search_channels(query, creds=creds, limit=limit),
        SEARCH_TTL_S,
    )
    return SearchResult(items=items, twitch_search=True)


def channel(
    url: str, *, tab: str = "videos", limit: int = 30, sort: str = "time", range: str = "7d"
) -> ChannelPage:
    """Browse one channel's tab. Twitch tabs: ``vods`` / ``clips``."""
    url = url.strip()
    if not url:
        raise ExploreError("Which channel? Pass its link or name.")
    if "twitch.tv" in url.lower() or tab in ("vods", "clips"):
        key = _key(
            "tw-channel", url=twitch.login_of(url), tab=tab, sort=sort, range=range, limit=limit
        )
        return cache.cached(
            key,
            lambda: twitch.channel(url, tab=tab, sort=sort, range=range, limit=limit),
            SEARCH_TTL_S,
        )
    key = _key("yt-channel", url=youtube.channel_base(url), tab=tab, limit=limit)
    return cache.cached(key, lambda: youtube.channel(url, tab=tab, limit=limit), SEARCH_TTL_S)


# --------------------------------------------------------------------------
# Peek
# --------------------------------------------------------------------------


def peek(url: str) -> PeekResult:
    """Full details for one video, with its "Most replayed" graph and peaks."""
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise ExploreError("Peek needs a video link (https://…).")
    return cache.cached(
        _key("peek", url=url),
        lambda: peek_from_info(ytdlp.extract(url, flat=False), url),
        PEEK_TTL_S,
    )


def peek_from_info(info: dict[str, Any], url: str) -> PeekResult:
    """Build a :class:`PeekResult` from a full yt-dlp info dict (no network)."""
    from ..gta.fetch import detect_platform, parse_heatmap
    from ..pipeline.signals import build_curve, parse_chapters

    if info.get("_type") == "playlist":
        raise ExploreError(
            "That link is a channel or playlist, not one video.",
            hint="Open it from the channel view instead, then pick a video.",
        )
    page_url = text(info.get("webpage_url")) or url
    duration = number(info.get("duration")) or 0.0
    heatmap = parse_heatmap(info.get("heatmap"))
    chapters = parse_chapters(info)
    peaks: list[Peak] = []
    if heatmap and duration > 0:
        curve = build_curve(duration, heatmap=heatmap, chapters=chapters)
        peaks = top_peaks(curve.curve, curve.hotspots)
    platform = detect_platform(page_url)
    live_status = info.get("live_status")
    return PeekResult(
        platform=platform if platform in ("youtube", "twitch") else "other",
        id=text(info.get("id")) or "",
        url=page_url,
        title=text(info.get("title")) or text(info.get("fulltitle")) or "Untitled",
        channel=text(info.get("channel")) or text(info.get("uploader")),
        channel_url=text(info.get("channel_url")) or text(info.get("uploader_url")),
        duration_s=duration,
        views=integer(info.get("view_count")),
        likes=integer(info.get("like_count")),
        upload_date=iso_date(info),
        thumbnail=best_thumbnail(info),
        is_live=info.get("is_live") is True or live_status == "is_live",
        description=(text(info.get("description")) or "")[:DESCRIPTION_CHARS],
        chapters=[ChapterOut(start_s=c.start_s, end_s=c.end_s, title=c.title) for c in chapters],
        heatmap=[HeatPoint(start_s=a, end_s=b, value=round(v, 4)) for a, b, v in heatmap],
        peaks=peaks,
    )


#: Smallest gap between two extra peaks, as a fraction of the video (and at
#: least MIN_PEAK_GAP_S seconds), so five peaks don't all sit on one hill.
PEAK_GAP_FRACTION = 1 / 25
MIN_PEAK_GAP_S = 10
#: Extra peaks below this share of the curve's top are not worth showing.
MIN_EXTRA_PEAK = 0.3


def top_peaks(curve: Any, hotspots: list[Any], *, count: int = MAX_PEAKS) -> list[Peak]:
    """The strongest ``count`` rewatch peaks of a per-second audience curve.

    The audience curve's own hotspots (pipeline.signals) come first. Its
    threshold is strict — it is tuned to feed the clip finder — so a video
    with one dominant moment can have a single hotspot; the rest are filled
    with the curve's next-highest local maxima, spaced apart.
    """
    import numpy as np

    peaks = [
        Peak(
            start_s=h.start_s,
            end_s=h.end_s,
            peak_s=h.peak_s,
            value=h.value,
            reasons=list(h.reasons),
        )
        for h in hotspots[:count]
    ]
    values = np.asarray(curve, dtype=np.float64)
    n = len(values)
    if len(peaks) >= count or n < 3:
        return peaks
    if n >= 5:
        padded = np.pad(values, 2, mode="edge")
        values = np.convolve(padded, np.ones(5) / 5, mode="valid")
    gap = max(MIN_PEAK_GAP_S, int(n * PEAK_GAP_FRACTION))
    is_max = (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    candidates = sorted((int(i) + 1 for i in np.flatnonzero(is_max)), key=lambda i: -values[i])
    for i in candidates:
        if len(peaks) >= count or values[i] < MIN_EXTRA_PEAK:
            break
        if any(p.start_s - gap <= i <= p.end_s + gap for p in peaks):
            continue
        floor = values[i] * 0.85
        start, end = i, i
        while start > 0 and i - start < gap and values[start - 1] >= floor:
            start -= 1
        while end < n - 1 and end - i < gap and values[end + 1] >= floor:
            end += 1
        peaks.append(
            Peak(
                start_s=float(start),
                end_s=float(end + 1),
                peak_s=float(i),
                value=round(float(values[i]), 3),
                reasons=[f"rewatched more than around it ({round(values[i] * 100)}% of the top)"],
            )
        )
    peaks.sort(key=lambda p: -p.value)
    return peaks
