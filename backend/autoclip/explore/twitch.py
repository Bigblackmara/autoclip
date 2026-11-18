"""Twitch for Explore: a channel's past broadcasts and top clips (yt-dlp).

Keyword channel search needs the official Helix API, so it is offered only
when GTA Mode's Twitch app credentials are saved; without them a query is
taken as a channel name.

Broadcasts still in progress are hidden: their recording keeps growing and
cannot be clipped until the stream ends.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

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

_LOGIN = re.compile(r"^[a-z0-9_]{2,25}$")
_URL_LOGIN = re.compile(r"twitch\.tv/([A-Za-z0-9_]{2,25})")
#: yt-dlp / Twitch clip ranges.
CLIP_RANGES = ("24hr", "7d", "30d", "all")
#: Twitch's preview for a VOD whose stream is still running.
_PROCESSING_THUMB = "404_processing"


def login_of(value: str) -> str:
    """The channel login from a twitch.tv URL, ``@name`` or bare name."""
    s = value.strip()
    match = _URL_LOGIN.search(s)
    return (match.group(1) if match else s.lstrip("@")).lower()


def is_valid_login(login: str) -> bool:
    return bool(_LOGIN.match(login))


def credentials() -> tuple[str, str] | None:
    """GTA Mode's saved Twitch app credentials, or None."""
    try:
        from ..gta.signals import twitch_credentials

        return twitch_credentials()
    except Exception:  # keyring missing or broken: just no Helix features
        return None


def _is_live(entry: Raw) -> bool:
    if entry.get("live_status") == "is_live" or entry.get("is_live") is True:
        return True
    thumb = text(entry.get("thumbnail")) or ""
    return _PROCESSING_THUMB in thumb


def _sharper(url: str | None) -> str | None:
    """VOD previews come as 320x180; Twitch's CDN serves any size."""
    return url.replace("-320x180.", "-640x360.") if url else url


def to_item(entry: Raw, *, kind: str, login: str) -> ExploreItem | None:
    url = text(entry.get("url")) or text(entry.get("webpage_url"))
    vid = text(entry.get("id"))
    if not url or not vid:
        return None
    return ExploreItem(
        kind="clip" if kind == "clip" else "vod",
        source="twitch",
        id=vid,
        url=url,
        title=text(entry.get("title")) or ("Twitch clip" if kind == "clip" else "Twitch broadcast"),
        channel=text(entry.get("uploader")) or text(entry.get("channel")) or login,
        channel_id=text(entry.get("uploader_id")) or login,
        channel_url=f"https://www.twitch.tv/{login}",
        thumbnail=_sharper(best_thumbnail(entry)),
        duration_s=number(entry.get("duration")),
        views=integer(entry.get("view_count")),
        upload_date=iso_date(entry),
        is_live=_is_live(entry),
    )


def channel_url(login: str, *, tab: str, sort: str = "time", range: str = "7d") -> str:
    if tab == "clips":
        r = range if range in CLIP_RANGES else "7d"
        return f"https://www.twitch.tv/{login}/clips?filter=clips&range={r}"
    s = "views" if sort == "views" else "time"
    return f"https://www.twitch.tv/{login}/videos?filter=archives&sort={s}"


def channel(
    value: str, *, tab: str = "vods", sort: str = "time", range: str = "7d", limit: int = 30
) -> ChannelPage:
    """A Twitch channel's past broadcasts (``vods``) or top ``clips``."""
    login = login_of(value)
    if not is_valid_login(login):
        raise ytdlp.ExploreError(
            f'"{value}" is not a Twitch channel name.',
            hint="Use the name from the channel's link, e.g. twitch.tv/shroud → shroud.",
        )
    tab = "clips" if tab == "clips" else "vods"
    info = ytdlp.extract(
        channel_url(login, tab=tab, sort=sort, range=range), flat=True, limit=limit
    )
    kind = "clip" if tab == "clips" else "vod"
    items = [
        i
        for e in info.get("entries") or []
        if isinstance(e, dict) and (i := to_item(e, kind=kind, login=login)) is not None
    ]
    items = [i for i in items if not i.is_live][:limit]
    return ChannelPage(
        channel=ExploreChannel(source="twitch", title=login, url=f"https://www.twitch.tv/{login}"),
        tab=tab,
        items=items,
    )


def channel_card(login: str, *, description: str | None = None) -> ExploreItem:
    return ExploreItem(
        kind="channel",
        source="twitch",
        id=login,
        url=f"https://www.twitch.tv/{login}",
        title=login,
        channel=login,
        channel_id=login,
        channel_url=f"https://www.twitch.tv/{login}",
        description=description or "Open to see past broadcasts and top clips",
    )


# --------------------------------------------------------------------------
# Helix (only with saved credentials)
# --------------------------------------------------------------------------

HttpGet = Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]]
HttpPost = Callable[[str, dict[str, str]], dict[str, Any]]


def _http_get(url: str, params: dict[str, str], headers: dict[str, str]) -> dict[str, Any]:
    import httpx

    response = httpx.get(url, params=params, headers=headers, timeout=20.0)
    response.raise_for_status()
    return response.json()


def _http_post(url: str, params: dict[str, str]) -> dict[str, Any]:
    import httpx

    response = httpx.post(url, params=params, timeout=20.0)
    response.raise_for_status()
    return response.json()


_token_lock = threading.Lock()
_token: tuple[str, str, float] | None = None  # (client_id, token, expires_at)


def _app_token(creds: tuple[str, str], http_post: HttpPost) -> str:
    global _token
    client_id, secret = creds
    with _token_lock:
        if _token and _token[0] == client_id and time.time() < _token[2]:
            return _token[1]
        data = http_post(
            "https://id.twitch.tv/oauth2/token",
            {"client_id": client_id, "client_secret": secret, "grant_type": "client_credentials"},
        )
        token = str(data["access_token"])
        _token = (client_id, token, time.time() + max(60, int(data.get("expires_in", 3600)) - 120))
        return token


def reset_token() -> None:
    global _token
    with _token_lock:
        _token = None


def search_channels(
    query: str,
    *,
    creds: tuple[str, str],
    limit: int = 24,
    http_get: HttpGet = _http_get,
    http_post: HttpPost = _http_post,
) -> list[ExploreItem]:
    """Twitch channels matching ``query`` (Helix ``search/channels``)."""
    try:
        token = _app_token(creds, http_post)
        data = http_get(
            "https://api.twitch.tv/helix/search/channels",
            {"query": query, "first": str(max(1, min(limit, 100)))},
            {"Client-Id": creds[0], "Authorization": f"Bearer {token}"},
        )
    except Exception as exc:
        reset_token()
        raise ytdlp.ExploreError(
            "Twitch channel search failed.",
            hint=f"Check the Twitch app credentials in GTA Mode settings.\n\nOriginal error: {exc}",
        ) from exc
    items: list[ExploreItem] = []
    for row in data.get("data") or []:
        login = text(row.get("broadcaster_login"))
        if not login:
            continue
        name = text(row.get("display_name")) or login
        items.append(
            ExploreItem(
                kind="channel",
                source="twitch",
                id=login,
                url=f"https://www.twitch.tv/{login}",
                title=name,
                channel=name,
                channel_id=text(row.get("id")) or login,
                channel_url=f"https://www.twitch.tv/{login}",
                thumbnail=text(row.get("thumbnail_url")),
                description=text(row.get("title")) or text(row.get("game_name")),
            )
        )
    return items[:limit]
