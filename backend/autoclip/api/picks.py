"""Pick your own parts on the rewatch graph (see :mod:`autoclip.pipeline.picks`).

* ``GET  /api/jobs/{id}/audience`` - the audience curve to draw.
* ``POST /api/jobs/{id}/picked-clips`` - make clips from picked ranges.
* ``GET  /api/jobs/{id}/picked-clips`` - progress of that render.
* ``GET  /api/clips/{id}/segments`` - the parts of a joined clip.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import load as load_settings
from ..db import store
from ..pipeline import picks
from ..pipeline import signals as audience_signals
from ..pipeline.prepare import Loudness
from ..pipeline.runner import JobWorkspace, clip_spans
from .clips import _cached_silences, _load_transcript

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["picks"])

#: Most points per series sent to the graph; longer curves are max-pooled so
#: no peak disappears.
MAX_GRAPH_POINTS = 2000


class RangeIn(BaseModel):
    start_s: float
    end_s: float


class PickedClipsIn(BaseModel):
    ranges: list[RangeIn] = Field(min_length=1)
    #: True joins every range into one clip, in source order.
    join: bool = False
    #: Framing for these clips; None uses the export settings.
    framing: Literal["auto", "fill", "split", "fit"] | None = None


class PickedClipOut(BaseModel):
    id: str
    title: str
    start_s: float
    end_s: float
    duration_s: float
    spans: list[tuple[float, float]]
    score: int


class PickTaskOut(BaseModel):
    job_id: str
    clip_ids: list[str]
    state: str
    stage: str
    progress: float
    message: str
    error: str | None


class PickedClipsOut(BaseModel):
    clips: list[PickedClipOut]
    task: PickTaskOut


# --------------------------------------------------------------------------
# The curve
# --------------------------------------------------------------------------


def _pool(series: np.ndarray, step: int) -> list[float]:
    if step <= 1:
        return [round(float(v), 3) for v in series]
    n = math.ceil(len(series) / step)
    padded = np.pad(series.astype(np.float32), (0, n * step - len(series)), mode="edge")
    return [round(float(v), 3) for v in padded.reshape(n, step).max(axis=1)]


def _curve_for(job_id: str) -> audience_signals.AudienceCurve | None:
    """The job's cached audience curve, or one built from what's on disk.

    Building here never touches the network: comments are used only when a
    previous run cached them. The result is not written to the job's
    ``audience.json``, so a later pipeline run still gathers every signal.
    """
    workspace = JobWorkspace(job_id)
    if workspace.audience.exists():
        try:
            return audience_signals.AudienceCurve.load(workspace.audience)
        except (OSError, ValueError, TypeError, KeyError):
            log.warning("Ignoring an unreadable audience curve for job %s.", job_id)
    job = store.get_job(job_id)
    source = store.get_source(job.source_id) if job else None
    if source is None:
        return None
    loudness = None
    if workspace.loudness.exists():
        try:
            loudness = Loudness.from_dict(
                json.loads(workspace.loudness.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError, AttributeError):
            loudness = None
    source_dir = Path(source.path).parent
    try:
        return audience_signals.build_for_source(
            source_dir=source_dir,
            url=source.url,
            is_youtube=source.type == "youtube",
            duration_s=source.duration_s,
            loudness_values=loudness.values if loudness else None,
            loudness_step_s=loudness.step_s if loudness else 0.1,
            use_comments=(source_dir / audience_signals.COMMENTS_FILE).exists(),
            use_energy=True,
            fetch=lambda url: [],
        )
    except Exception:
        log.warning("Could not build an audience curve for job %s.", job_id, exc_info=True)
        return None


def _audience_payload(job_id: str) -> dict:
    job = store.get_job(job_id)
    source = store.get_source(job.source_id) if job else None
    if job is None or source is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    curve = _curve_for(job_id)
    clips = [
        {
            "id": c.id,
            "title": c.title,
            "rank": c.rank,
            "start_s": c.start_s,
            "end_s": c.end_s,
            "status": c.status,
            "spans": [[s.start, s.end] for s in (clip_spans(JobWorkspace(job_id), c) or [])],
        }
        for c in store.list_clips(job_id)
    ]
    duration = source.duration_s or (curve.seconds if curve else 0)
    if curve is None or not curve.has_data:
        return {
            "duration_s": duration,
            "kind": "none",
            "sources": [],
            "step_s": 1,
            "curve": [],
            "components": {},
            "hotspots": [],
            "chapters": [],
            "clips": clips,
            "has_transcript": JobWorkspace(job_id).transcript.exists(),
        }
    step = max(1, math.ceil(curve.seconds / MAX_GRAPH_POINTS))
    data = curve.to_dict()
    return {
        "duration_s": duration,
        "kind": curve.kind,
        "sources": curve.sources,
        "step_s": step,
        "curve": _pool(curve.curve, step),
        "components": {name: _pool(series, step) for name, series in curve.components.items()},
        "hotspots": data["hotspots"],
        "chapters": data["chapters"],
        "clips": clips,
        "has_transcript": JobWorkspace(job_id).transcript.exists(),
    }


@router.get("/jobs/{job_id}/audience")
async def job_audience(job_id: str) -> dict:
    """The audience curve for the rewatch graph, plus the job's clips as bands."""
    return await asyncio.to_thread(_audience_payload, job_id)


# --------------------------------------------------------------------------
# Picking
# --------------------------------------------------------------------------


def _launch(work: Callable[[], None]) -> None:
    """Run the render off the request (tests replace this to run inline)."""
    threading.Thread(target=work, name="autoclip-picks", daemon=True).start()


def _create(job_id: str, payload: PickedClipsIn) -> PickedClipsOut:
    job = store.get_job(job_id)
    source = store.get_source(job.source_id) if job else None
    if job is None or source is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Parts can be picked once the job has finished (it is {job.status}).",
        )
    if picks.is_running(job_id):
        raise HTTPException(
            status_code=409, detail="Clips picked earlier are still rendering; wait for them."
        )

    transcript = _load_transcript(job_id)
    workspace = JobWorkspace(job_id)
    try:
        planned = picks.plan(
            transcript,
            [(r.start_s, r.end_s) for r in payload.ranges],
            duration_s=source.duration_s,
            join=payload.join,
            silences=_cached_silences(workspace),
        )
    except picks.PickError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    curve = _curve_for(job_id)
    clips = picks.create(job, workspace.segments, planned, curve=curve)
    task = picks.start_task(job_id, [c.id for c in clips])
    framing = payload.framing or load_settings().export.framing
    _launch(lambda: picks.render(task, job, source, clips, framing=framing))

    return PickedClipsOut(
        clips=[
            PickedClipOut(
                id=clip.id,
                title=clip.title,
                start_s=clip.start_s,
                end_s=clip.end_s,
                duration_s=round(pick.duration_s, 3),
                spans=[(s.start, s.end) for s in pick.spans],
                score=clip.score,
            )
            for clip, pick in zip(clips, planned, strict=True)
        ],
        task=PickTaskOut(**task.to_dict()),
    )


@router.post("/jobs/{job_id}/picked-clips", response_model=PickedClipsOut, status_code=202)
async def create_picked_clips(job_id: str, payload: PickedClipsIn) -> PickedClipsOut:
    """Make clips from ranges picked on the rewatch graph.

    ``join`` false makes one clip per range; true makes one clip of all the
    ranges back to back. The clips are created at once and rendered in the
    background through the normal reframe/captions/export stages; poll
    ``GET`` on the same path for progress.
    """
    return await asyncio.to_thread(_create, job_id, payload)


@router.get("/jobs/{job_id}/picked-clips", response_model=PickTaskOut)
async def picked_clips_status(job_id: str) -> PickTaskOut:
    task = picks.task_for(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="No picked clips are rendering for this job.")
    return PickTaskOut(**task.to_dict())


@router.get("/clips/{clip_id}/segments")
async def clip_segments(clip_id: str) -> dict:
    """The parts of a joined clip (``spans`` is empty for an ordinary clip)."""
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    spans = await asyncio.to_thread(clip_spans, JobWorkspace(clip.job_id), clip)
    return {
        "clip_id": clip_id,
        "spans": [[s.start, s.end] for s in spans or []],
        "duration_s": round(sum(s.duration for s in spans), 3)
        if spans
        else round(clip.end_s - clip.start_s, 3),
    }
