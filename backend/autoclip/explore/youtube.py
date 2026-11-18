"""YouTube for Explore: filtered search and channel tabs, flat via yt-dlp.

No API key: yt-dlp reads YouTube's own results pages. Filters are expressed
the way YouTube's search page does it — the ``sp`` query parameter.
"""

from __future__ import annotations

import base64
import re
from typing import Literal
from urllib.parse import quote, quote_plus

from . import ytdlp
from .models import (
    ChannelPage,
    ExploreChannel,
    ExploreItem,
    Raw,
    best_thumbnail,
    integer,
    iso_date,
    number,
    text,
)

SearchType = Literal["videos", "shorts", "channels", "live"]
Duration = Literal["any", "short", "medium", "long"]
Sort = Literal["relevance", "views", "date"]
Uploaded = Literal["any", "today", "week", "month", "year"]
ChannelTab = Literal["videos", "shorts", "streams"]

_UPLOADED = {"hour": 1, "today": 2, "week": 3, "month": 4, "year": 5}
_TYPE = {"videos": 1, "live": 1, "channels": 2, "shorts": 9}
#: YouTube's duration filter: 1 = under 4 min, 2 = over 20 min, 3 = 4-20 min.
_DURATION = {"short": 1, "long": 2, "medium": 3}
_SORT = {"rating": 1, "date": 2, "views": 3}
#: Shorts are at most 3 minutes.
SHORT_MAX_S = 180


def search_param(
    *, type: str, duration: str = "any", sort: str = "relevance", uploaded: str = "any"
) -> str:
    """YouTube's ``sp`` search filter: a tiny protobuf, base64-encoded.

    Top level: field 1 = sort, field 2 = filters { 1 upload date, 2 type,
    3 duration, 8 "live now" }.
    """
    filters: list[int] = []
    if type != "channels" and uploaded in _UPLOADED:
        filters += [0x08, _UPLOADED[uploaded]]
    if type in _TYPE:
        filters += [0x10, _TYPE[type]]
    if type == "videos" and duration in _DURATION:
        filters += [0x18, _DURATION[duration]]
    if type == "live":
        filters += [0x40, 1]
    top: list[int] = []
    if type != "channels" and sort in _SORT:
        top += [0x08, _SORT[sort]]
    if filters:
        top += [0x12, len(filters), *filters]
    return base64.b64encode(bytes(top)).decode("ascii") if top else ""


def search_url(query: str, sp: str) -> str:
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    return f"{url}&sp={quote(sp, safe='')}" if sp else url


_CHANNEL_PATH = re.compile(r"/(channel/|c/|user/|@)")


def to_item(
    entry: Raw, *, force_kind: str | None = None, channel: Raw | None = None
) -> ExploreItem | None:
    """One flat yt-dlp entry as an :class:`ExploreItem` (None for playlists etc.)."""
    url = text(entry.get("url")) or text(entry.get("webpage_url"))
    vid = text(entry.get("id"))
    if not url or not vid:
        return None
    parent = channel or {}
    is_channel = entry.get("ie_key") == "YoutubeTab" or bool(_CHANNEL_PATH.search(url))
    if is_channel:
        if "list=" in url or "/playlist" in url:
            return None
        return ExploreItem(
            kind="channel",
            source="youtube",
            id=vid,
            url=url,
            title=text(entry.get("title")) or text(entry.get("channel")) or "Channel",
            channel=text(entry.get("channel")) or text(entry.get("title")),
            channel_id=text(entry.get("channel_id")) or vid,
            channel_url=url,
            thumbnail=best_thumbnail(entry),
            subscribers=integer(entry.get("channel_follower_count")),
            description=text(entry.get("description")),
        )
    if (
        entry.get("ie_key") not in (None, "Youtube")
        and "youtube" not in url
        and "youtu.be" not in url
    ):
        return None
    live_status = entry.get("live_status")
    if live_status == "is_upcoming":
        return None  # a premiere or scheduled stream: nothing to watch yet
    is_live = live_status == "is_live" or entry.get("is_live") is True
    is_short = force_kind == "short" or "/shorts/" in url
    kind = "live" if is_live else "short" if is_short else "video"
    if not url.startswith("http"):
        url = f"https://www.youtube.com/watch?v={vid}"
    fallback_thumb = f"https://i.ytimg.com/vi/{vid}/{'oar2' if is_short else 'hqdefault'}.jpg"
    return ExploreItem(
        kind=kind,
        source="youtube",
        id=vid,
        url=url,
        title=text(entry.get("title")) or "Untitled",
        channel=text(entry.get("channel"))
        or text(entry.get("uploader"))
        or text(parent.get("channel")),
        channel_id=text(entry.get("channel_id")) or text(parent.get("channel_id")),
        channel_url=(
            text(entry.get("channel_url"))
            or text(entry.get("uploader_url"))
            or text(parent.get("channel_url"))
        ),
        thumbnail=best_thumbnail(entry) or fallback_thumb,
        duration_s=number(entry.get("duration")),
        views=integer(entry.get("view_count")),
        upload_date=iso_date(entry),
        is_live=is_live,
        description=text(entry.get("description")),
    )


def _entries(info: Raw) -> list[Raw]:
    return [e for e in info.get("entries") or [] if isinstance(e, dict)]


def _search_raw(
    query: str, *, type: str, duration: str, sort: str, uploaded: str, limit: int
) -> list[ExploreItem]:
    # A plain video search uses yt-dlp's own search prefix; anything with a
    # filter goes through the results page and its ``sp`` parameter.
    plain = type == "videos" and (duration, sort, uploaded) == ("any", "relevance", "any")
    sp = search_param(type=type, duration=duration, sort=sort, uploaded=uploaded)
    url = f"ytsearch{limit}:{query}" if plain else search_url(query, sp)
    info = ytdlp.extract(url, flat=True, limit=limit)
    items = [i for e in _entries(info) if (i := to_item(e)) is not None]

    if type == "channels":
        items = [i for i in items if i.kind == "channel"]
    elif type == "shorts":
        # A /shorts/ link, or anything short enough to be one (the Shorts
        # filter is sometimes ignored and plain videos come back).
        items = [
            i.model_copy(update={"kind": "short", "url": f"https://www.youtube.com/shorts/{i.id}"})
            for i in items
            if i.kind == "short"
            or (i.kind == "video" and i.duration_s is not None and i.duration_s <= SHORT_MAX_S)
        ]
    elif type == "live":
        items = [
            i.model_copy(update={"kind": "live", "is_live": True})
            for i in items
            if i.kind != "channel"
        ]
    else:
        items = [i for i in items if i.kind != "channel"]
    return items[:limit]


def search(
    query: str,
    *,
    type: str = "videos",
    duration: str = "any",
    sort: str = "relevance",
    uploaded: str = "any",
    limit: int = 24,
) -> list[ExploreItem]:
    """Search YouTube with the same filters as its own search page."""
    query = query.strip()
    if not query:
        return []
    if type != "shorts":
        return _search_raw(
            query, type=type, duration=duration, sort=sort, uploaded=uploaded, limit=limit
        )
    # YouTube's own Shorts filter returns a shelf layout yt-dlp often reads as
    # nothing (slowly — it pages through everything first). Short videos
    # tagged #shorts are fast and reliable, so they go first; the real
    # filter is only the fallback.
    loose = _search_raw(
        f"{query} #shorts",
        type="videos",
        duration="short",
        sort=sort,
        uploaded=uploaded,
        limit=limit * 2,
    )
    items = [
        i.model_copy(update={"kind": "short", "url": f"https://www.youtube.com/shorts/{i.id}"})
        for i in loose
        if i.kind == "short" or (i.duration_s is not None and i.duration_s <= SHORT_MAX_S)
    ][:limit]
    if items:
        return items
    return _search_raw(
        query, type="shorts", duration="any", sort=sort, uploaded=uploaded, limit=limit
    )


def channel_base(value: str) -> str:
    """A channel URL without a trailing tab, from a URL, @handle, UC… id or name."""
    s = value.strip()
    if re.match(r"^https?://", s):
        s = re.sub(r"\?.*$", "", s)
        s = re.sub(r"/(videos|shorts|streams|featured|live|playlists|community|about)/?$", "", s)
        return s.rstrip("/")
    if s.startswith("@"):
        return f"https://www.youtube.com/{s}"
    if re.match(r"^UC[\w-]{20,}$", s):
        return f"https://www.youtube.com/channel/{s}"
    handle = re.sub(r"\s+", "", s)
    return f"https://www.youtube.com/@{handle}"


def _channel_meta(info: Raw, base: str) -> ExploreChannel:
    thumbs = [t for t in info.get("thumbnails") or [] if isinstance(t, dict) and text(t.get("url"))]
    avatar = next((t for t in thumbs if "avatar" in str(t.get("id") or "")), None) or next(
        (t for t in thumbs if number(t.get("width")) and t.get("width") == t.get("height")), None
    )
    banner = next((t for t in thumbs if "banner" in str(t.get("id") or "")), None)
    raw_title = text(info.get("title")) or ""
    title = (
        text(info.get("channel"))
        or text(info.get("uploader"))
        or re.sub(r" - (Videos|Shorts|Live)$", "", raw_title)
        or base
    )
    return ExploreChannel(
        source="youtube",
        title=title,
        url=text(info.get("channel_url")) or text(info.get("uploader_url")) or base,
        avatar=text(avatar.get("url")) if avatar else None,
        banner=text(banner.get("url")) if banner else None,
        subscribers=integer(info.get("channel_follower_count")),
        description=text(info.get("description")),
    )


def channel(value: str, *, tab: str = "videos", limit: int = 30) -> ChannelPage:
    """A channel's Videos, Shorts or Streams tab (most recent first)."""
    tab = tab if tab in ("videos", "shorts", "streams") else "videos"
    base = channel_base(value)
    try:
        info = ytdlp.extract(f"{base}/{tab}", flat=True, limit=limit)
    except ytdlp.MissingTabError:
        # e.g. a channel that never streamed has no Streams tab: an empty
        # tab, not a failure. The header comes from any tab that exists.
        info = {"entries": []}
        if tab != "videos":
            try:
                meta = ytdlp.extract(f"{base}/videos", flat=True, limit=1)
                info = {**meta, "entries": []}
            except ytdlp.ExploreError:
                pass
    force = "short" if tab == "shorts" else None
    items = [
        i
        for e in _entries(info)
        if (i := to_item(e, force_kind=force, channel=info)) is not None and i.kind != "channel"
    ]
    return ChannelPage(channel=_channel_meta(info, base), tab=tab, items=items[:limit])
