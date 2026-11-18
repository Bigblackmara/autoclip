"""Autopilot rules and their working state, as JSON under ``<home>/autopilot/``.

``autopilots.json`` holds the rules (what the editor saves). ``state.json``
holds, per autopilot, what the engine has learned: every video it has seen
(so nothing is looked at twice), the clips it made and booked, and its
recent runs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from ..publishing.jsonstore import JsonFile
from .models import Autopilot, AutopilotIn

_RULES = JsonFile("autopilot", "autopilots.json", lambda: {"autopilots": []})
_STATE = JsonFile("autopilot", "state.json", dict)

#: Videos remembered per autopilot (oldest forgotten first; a forgotten
#: video is far past ``max_age_days`` by then, so it can't come back).
MAX_VIDEOS = 600
MAX_RUNS = 50
MAX_CLIPS = 400

#: Video statuses: waiting (not good enough yet / no rewatch graph yet),
#: ready (qualifies, next in line to clip), clipping (a job is running),
#: done, skipped (never: too old/short/...), failed.
VIDEO_STATUSES = frozenset({"waiting", "ready", "clipping", "done", "skipped", "failed"})
#: Clip statuses: review (waits for approval), ready (approved, waits for a
#: slot), booked (posts created for a slot), posted, rejected, failed.
CLIP_STATUSES = frozenset({"review", "ready", "booked", "posted", "rejected", "failed"})


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ rules
def _rule_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("autopilots")
    if not isinstance(rows, list):
        rows = data["autopilots"] = []
    return rows


def list_autopilots() -> list[Autopilot]:
    out: list[Autopilot] = []
    for row in _rule_rows(_RULES.read()):
        try:
            out.append(Autopilot.model_validate(row))
        except ValueError:
            continue
    return out


def get_autopilot(autopilot_id: str) -> Autopilot | None:
    for rule in list_autopilots():
        if rule.id == autopilot_id:
            return rule
    return None


def create_autopilot(payload: AutopilotIn) -> Autopilot:
    rule = Autopilot(**payload.model_dump())
    with _RULES.edit() as data:
        _rule_rows(data).append(rule.model_dump(mode="json"))
    return rule


def update_autopilot(autopilot_id: str, **changes: Any) -> Autopilot:
    with _RULES.edit() as data:
        for row in _rule_rows(data):
            if row.get("id") != autopilot_id:
                continue
            merged = {**row, **changes, "id": autopilot_id, "updated_at": now_iso()}
            rule = Autopilot.model_validate(merged)
            row.clear()
            row.update(rule.model_dump(mode="json"))
            return rule
    raise KeyError(autopilot_id)


def delete_autopilot(autopilot_id: str) -> None:
    with _RULES.edit() as data:
        rows = _rule_rows(data)
        rows[:] = [r for r in rows if r.get("id") != autopilot_id]
    with _STATE.edit() as state:
        state.pop(autopilot_id, None)


# ------------------------------------------------------------------ state
def empty_state() -> dict[str, Any]:
    return {
        "last_scan_at": None,
        "last_run_at": None,
        "last_error": None,
        "message": "Waiting for the first run.",
        "videos": {},
        "clips": [],
        "runs": [],
    }


def get_state(autopilot_id: str) -> dict[str, Any]:
    state = _STATE.read().get(autopilot_id)
    base = empty_state()
    if isinstance(state, dict):
        base.update(state)
    return base


def save_state(autopilot_id: str, state: dict[str, Any]) -> None:
    _trim(state)
    with _STATE.edit() as data:
        data[autopilot_id] = state


@contextlib.contextmanager
def edit_state(autopilot_id: str) -> Iterator[dict[str, Any]]:
    """``with edit_state(id) as st:`` - the autopilot's state, saved on a clean exit."""
    with _STATE.edit() as data:
        state = data.get(autopilot_id)
        base = empty_state()
        if isinstance(state, dict):
            base.update(state)
        yield base
        _trim(base)
        data[autopilot_id] = base


def _trim(state: dict[str, Any]) -> None:
    videos = state.get("videos") or {}
    if len(videos) > MAX_VIDEOS:
        keep = sorted(
            videos.items(),
            key=lambda kv: (kv[1].get("status") in {"clipping", "ready"}, kv[1].get("first_seen")),
            reverse=True,
        )[:MAX_VIDEOS]
        state["videos"] = dict(keep)
    runs = state.get("runs") or []
    state["runs"] = runs[-MAX_RUNS:]
    clips = state.get("clips") or []
    if len(clips) > MAX_CLIPS:
        active = [c for c in clips if c.get("status") in {"review", "ready", "booked"}]
        rest = [c for c in clips if c not in active]
        state["clips"] = rest[-(MAX_CLIPS - len(active)) :] + active
