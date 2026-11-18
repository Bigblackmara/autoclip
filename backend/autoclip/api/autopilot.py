"""Autopilot rules: create, edit, pause/resume, run now, review clips."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..autopilot import store
from ..autopilot.engine import engine, stock, upcoming_slots
from ..autopilot.models import Autopilot, AutopilotIn
from ..publishing import posts as post_store

router = APIRouter(prefix="/api/autopilots", tags=["autopilot"])


class ClipActionIn(BaseModel):
    action: Literal["approve", "reject"]


def _counts(state: dict[str, Any]) -> dict[str, int]:
    videos = state.get("videos", {}).values()
    clips = state.get("clips", [])
    return {
        "videos_waiting": sum(1 for v in videos if v.get("status") in {"waiting", "ready"}),
        "videos_clipping": sum(1 for v in videos if v.get("status") == "clipping"),
        "videos_seen": len(state.get("videos", {})),
        "clips_review": sum(1 for c in clips if c.get("status") == "review"),
        "clips_ready": sum(1 for c in clips if c.get("status") == "ready"),
        "clips_booked": sum(1 for c in clips if c.get("status") == "booked"),
        "clips_posted": sum(1 for c in clips if c.get("status") == "posted"),
        "stock": stock(state),
    }


def _summary(rule: Autopilot) -> dict[str, Any]:
    state = store.get_state(rule.id)
    return {
        **rule.model_dump(mode="json"),
        "state": {
            "last_scan_at": state.get("last_scan_at"),
            "last_run_at": state.get("last_run_at"),
            "last_error": state.get("last_error"),
            "message": state.get("message"),
        },
        "counts": _counts(state),
    }


def _detail(rule: Autopilot) -> dict[str, Any]:
    state = store.get_state(rule.id)
    videos = sorted(
        state.get("videos", {}).values(),
        key=lambda v: v.get("first_seen") or "",
        reverse=True,
    )
    posts = post_store.list_posts(autopilot_id=rule.id)
    return {
        **_summary(rule),
        "runs": list(reversed(state.get("runs", [])))[:30],
        "videos": videos[:120],
        "clips": list(reversed(state.get("clips", [])))[:120],
        "posts": [p.as_dict() for p in posts[:120]],
        "plan": upcoming_slots(rule, datetime.now(UTC)),
    }


def _get(autopilot_id: str) -> Autopilot:
    rule = store.get_autopilot(autopilot_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Autopilot not found.")
    return rule


@router.get("")
async def list_autopilots() -> dict[str, Any]:
    def build() -> dict[str, Any]:
        return {
            "running": engine.running,
            "autopilots": [_summary(r) for r in store.list_autopilots()],
        }

    return await asyncio.to_thread(build)


@router.post("", status_code=201)
async def create_autopilot(payload: AutopilotIn) -> dict[str, Any]:
    if not payload.rights_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Confirm you have the rights to clip and repost these videos.",
        )
    rule = await asyncio.to_thread(store.create_autopilot, payload)
    engine.notify(scan=rule.id)
    return await asyncio.to_thread(_detail, rule)


@router.get("/{autopilot_id}")
async def get_autopilot(autopilot_id: str) -> dict[str, Any]:
    def build() -> dict[str, Any]:
        return _detail(_get(autopilot_id))

    return await asyncio.to_thread(build)


@router.put("/{autopilot_id}")
async def update_autopilot(autopilot_id: str, payload: AutopilotIn) -> dict[str, Any]:
    if not payload.rights_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Confirm you have the rights to clip and repost these videos.",
        )

    def apply() -> dict[str, Any]:
        _get(autopilot_id)
        rule = store.update_autopilot(autopilot_id, **payload.model_dump())
        return _detail(rule)

    out = await asyncio.to_thread(apply)
    engine.notify(scan=autopilot_id)
    return out


@router.delete("/{autopilot_id}", status_code=204)
async def delete_autopilot(autopilot_id: str) -> None:
    def apply() -> None:
        _get(autopilot_id)
        engine.cancel_unsent(autopilot_id)
        store.delete_autopilot(autopilot_id)

    await asyncio.to_thread(apply)


async def _set_enabled(autopilot_id: str, enabled: bool) -> dict[str, Any]:
    def apply() -> dict[str, Any]:
        _get(autopilot_id)
        return _detail(store.update_autopilot(autopilot_id, enabled=enabled))

    out = await asyncio.to_thread(apply)
    engine.notify(scan=autopilot_id if enabled else None)
    return out


@router.post("/{autopilot_id}/pause")
async def pause_autopilot(autopilot_id: str) -> dict[str, Any]:
    return await _set_enabled(autopilot_id, False)


@router.post("/{autopilot_id}/resume")
async def resume_autopilot(autopilot_id: str) -> dict[str, Any]:
    return await _set_enabled(autopilot_id, True)


@router.post("/{autopilot_id}/run")
async def run_autopilot(autopilot_id: str) -> dict[str, Any]:
    """Scan this autopilot's sources now (in the background)."""
    await asyncio.to_thread(_get, autopilot_id)
    engine.notify(scan=autopilot_id)
    return {"queued": True, "running": engine.running}


@router.post("/{autopilot_id}/clips/{clip_id}")
async def clip_action(autopilot_id: str, clip_id: str, payload: ClipActionIn) -> dict[str, Any]:
    def apply() -> dict[str, Any]:
        rule = _get(autopilot_id)
        try:
            return engine.set_clip(rule, clip_id, payload.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Clip not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return await asyncio.to_thread(apply)
