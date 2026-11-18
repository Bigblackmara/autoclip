"""Explore: find source videos worth clipping (YouTube + Twitch), via yt-dlp.

Ported from CrestClip's Explore. Everything here is read-only metadata — no
media is downloaded. Results are kept in a small in-memory TTL cache so going
back and forth between tabs, channels and previews stays instant.
"""

from __future__ import annotations

from .service import (
    ExploreError,
    channel,
    clear_cache,
    peek,
    search,
    status,
)

__all__ = ["ExploreError", "channel", "clear_cache", "peek", "search", "status"]
