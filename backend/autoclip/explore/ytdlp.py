"""The one place Explore talks to yt-dlp.

Kept tiny and module-level on purpose: tests replace :func:`extract` with a
function returning canned info dicts, so no test ever touches the network.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class ExploreError(RuntimeError):
    """A lookup failed in a way worth explaining to the user."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class MissingTabError(ExploreError):
    """The channel exists but has no such tab (e.g. no Streams)."""


def _options(*, flat: bool, limit: int | None) -> dict[str, Any]:
    from ..config import load as load_settings
    from ..pipeline.ingest import js_runtime_options

    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        # Metadata only: a live or members-only video with no playable format
        # must still describe itself instead of failing the whole lookup.
        "ignore_no_formats_error": True,
        "socket_timeout": 20,
        **js_runtime_options(),
    }
    if flat:
        # One request per page of results instead of one per video.
        options["extract_flat"] = "in_playlist"
        if limit:
            options["playlistend"] = int(limit)
    else:
        options["noplaylist"] = True
        options["getcomments"] = False
    cookies = load_settings().ingest.cookies_from_browser
    if cookies:
        options["cookiesfrombrowser"] = (cookies,)
    return options


def extract(url: str, *, flat: bool, limit: int | None = None) -> dict[str, Any]:
    """yt-dlp's info dict for ``url`` without downloading anything.

    ``flat`` lists a playlist / search / channel tab without opening each
    entry. Raises :class:`ExploreError` with an actionable message on failure.
    """
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(_options(flat=flat, limit=limit)) as ydl:
            info = ydl.extract_info(url, download=False)
            info = ydl.sanitize_info(info)
    except yt_dlp.utils.DownloadError as exc:
        raise translate_error(exc) from exc
    except Exception as exc:  # yt-dlp extractors raise all sorts on site changes
        log.warning("yt-dlp failed on %s: %s", url, exc)
        raise translate_error(exc) from exc
    if not isinstance(info, dict):
        raise ExploreError(f"yt-dlp returned nothing for {url}.")
    return info


def translate_error(exc: Exception) -> ExploreError:
    """A yt-dlp failure as something the user can act on."""
    from ..pipeline.ingest import _BLOCKED_MARKERS, _BOT_CHECK_MARKERS

    message = str(exc).lower()
    if "does not have a" in message and "tab" in message:
        return MissingTabError("This channel has nothing on that tab.")
    if any(marker in message for marker in _BOT_CHECK_MARKERS):
        return ExploreError(
            "YouTube asked for proof you're not a bot.",
            hint=(
                "Set a browser to pull cookies from in Settings (Ingest → cookies from "
                "browser), sign in to YouTube there, and close that browser while searching."
            ),
        )
    if any(marker in message for marker in _BLOCKED_MARKERS):
        return ExploreError(
            "The site refused to answer from your network.",
            hint=(
                "VPNs such as Cloudflare WARP are often rate-limited. Turn the VPN off or "
                "exclude youtube.com / twitch.tv from it and try again."
            ),
        )
    if "does not exist" in message or "404" in message or "not found" in message:
        return ExploreError("Nothing was found at that link.", hint=f"yt-dlp said: {exc}")
    if "private" in message or "members-only" in message:
        return ExploreError("This video is private or members-only.")
    if "unavailable" in message or "removed" in message:
        return ExploreError("This video is unavailable or has been removed.")
    return ExploreError(
        "yt-dlp could not read this page.",
        hint=(
            "The site may have changed; `autoclip update-ytdlp` pulls the latest "
            f"yt-dlp.\n\nOriginal error: {exc}"
        ),
    )
