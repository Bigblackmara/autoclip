"""Explore API: find videos worth clipping on YouTube and Twitch.

Read-only metadata through yt-dlp; every call runs in a worker thread and is
cached for a few minutes. Failures come back as 502 with ``{message, hint}``
so the UI can show the fix beside the problem.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal, TypeVar

from fastapi import APIRouter, HTTPException, Query

from .. import explore
from ..explore.models import ChannelPage, PeekResult, SearchResult

router = APIRouter(prefix="/api/explore", tags=["explore"])

T = TypeVar("T")


async def _run(fn: Callable[[], T]) -> T:
    try:
        return await asyncio.to_thread(fn)
    except explore.ExploreError as exc:
        raise HTTPException(
            status_code=502, detail={"message": str(exc), "hint": exc.hint}
        ) from exc


@router.get("/status")
async def explore_status() -> dict[str, bool]:
    """Which optional Explore features are available on this machine."""
    return await asyncio.to_thread(explore.status)


@router.get("/search", response_model=SearchResult)
async def explore_search(
    q: str = Query(min_length=1, max_length=200),
    source: Literal["youtube", "twitch"] = "youtube",
    type: Literal["videos", "shorts", "channels", "live"] = "videos",
    duration: Literal["any", "short", "medium", "long"] = "any",
    sort: Literal["relevance", "views", "date"] = "relevance",
    uploaded: Literal["any", "today", "week", "month", "year"] = "any",
    limit: int = Query(default=24, ge=1, le=60),
) -> SearchResult:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Type something to search for.")
    return await _run(
        lambda: explore.search(
            q,
            source=source,
            type=type,
            duration=duration,
            sort=sort,
            uploaded=uploaded,
            limit=limit,
        )
    )


@router.get("/channel", response_model=ChannelPage)
async def explore_channel(
    url: str = Query(min_length=1, max_length=500),
    tab: Literal["videos", "shorts", "streams", "vods", "clips"] = "videos",
    sort: Literal["time", "views"] = "time",
    range: Literal["24hr", "7d", "30d", "all"] = "7d",
    limit: int = Query(default=30, ge=1, le=100),
) -> ChannelPage:
    """A channel's tab. YouTube: videos / shorts / streams; Twitch: vods / clips."""
    return await _run(lambda: explore.channel(url, tab=tab, sort=sort, range=range, limit=limit))


@router.get("/peek", response_model=PeekResult)
async def explore_peek(url: str = Query(min_length=1, max_length=500)) -> PeekResult:
    """One video's details, "Most replayed" graph, top peaks and chapters."""
    if not url.strip().lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Peek needs a video link (https://…).")
    return await _run(lambda: explore.peek(url))
