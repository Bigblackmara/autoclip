"""Housekeeping status: yt-dlp self-update and retention."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

from .. import config, retention, ytdlp_update

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/ytdlp")
async def ytdlp_status() -> dict[str, Any]:
    """Installed / in-memory / latest yt-dlp versions and the last check."""
    auto_update = config.load().ingest.ytdlp_auto_update
    return await asyncio.to_thread(ytdlp_update.updater.status, auto_update=auto_update)


@router.post("/ytdlp/check")
async def ytdlp_check() -> dict[str, Any]:
    """Check PyPI now and upgrade when newer (works with auto-update off too)."""
    await asyncio.to_thread(ytdlp_update.updater.check, reason="manual")
    return await ytdlp_status()


@router.get("/retention")
async def retention_status() -> dict[str, Any]:
    days = config.load().housekeeping.retention_days
    return {"retention_days": days, **retention.service.status()}
