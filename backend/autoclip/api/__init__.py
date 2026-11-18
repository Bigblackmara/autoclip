"""REST API routers."""

from __future__ import annotations

from fastapi import APIRouter

from . import clips, explore, gta, housekeeping, jobs, live, picks, settings, sources

__all__ = ["api_router"]

api_router = APIRouter()
api_router.include_router(sources.router)
api_router.include_router(jobs.router)
api_router.include_router(clips.router)
api_router.include_router(picks.router)
api_router.include_router(settings.router)
api_router.include_router(live.router)
api_router.include_router(gta.router)
api_router.include_router(housekeeping.router)
api_router.include_router(explore.router)
