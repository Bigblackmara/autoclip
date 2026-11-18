"""Pick your own parts: clips cut from ranges chosen on the rewatch graph.

The review page shows the job's audience curve (YouTube's "most replayed"
heatmap, timestamped comments, audio energy). The user drags across it to
select one or more ranges, then either

* **makes N clips** - one clip per range - or
* **joins them into one clip** - the ranges played back to back, in source
  order (see :mod:`autoclip.pipeline.segments`).

Either way the clips go through the normal pipeline from the boundary stage
on: edges snap to sentence starts/ends and measured silences (only nearby -
the user chose these edges), then reframe, captions and export run exactly as
for detected clips. Ported from CrestClip's ``createPickedClips`` /
``preparePickedClips`` / ``tidyRanges``.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..db import store
from ..db.models import Clip, Job, Source, new_id
from . import Stage, segments
from .boundaries import (
    SentenceBreaks,
    _keep_out_of_neighbours,
    align_end,
    align_start,
    snap_end_to_sentence,
    snap_start_to_sentence,
)
from .prepare import Silence
from .segments import Span
from .signals import AudienceCurve
from .transcript import Transcript

log = logging.getLogger(__name__)

#: A pick shorter than this (after clamping to the video) is rejected.
MIN_PICK_S = 1.0
#: Most ranges one request may carry.
MAX_PICKS = 12
#: Most a single clip may run - each picked clip, and a joined clip in total.
#: Ranges past it are cut short (CrestClip keeps every clip within 3 minutes).
MAX_CLIP_S = 180.0
#: How far an edge may move to land on a sentence boundary. The user put the
#: edge there; snapping is a nudge onto a clean cut, never a re-pick.
SNAP_WINDOW_S = 2.0
#: A snapped range that shrank below ``min(SNAP_MIN_KEEP_S, SNAP_MIN_KEEP_SHARE
#: x original)`` is replaced by the original range.
SNAP_MIN_KEEP_S = 3.0
SNAP_MIN_KEEP_SHARE = 0.7


class PickError(ValueError):
    """The picked ranges can't be made into clips; the message says why."""


@dataclass(frozen=True)
class PickedClip:
    """One clip to create: its parts (one, or several when joined) and words."""

    spans: list[Span]
    start_word: int
    end_word: int

    @property
    def start_s(self) -> float:
        return self.spans[0].start

    @property
    def end_s(self) -> float:
        return self.spans[-1].end

    @property
    def duration_s(self) -> float:
        return segments.total(self.spans)

    @property
    def is_joined(self) -> bool:
        return len(self.spans) > 1


# --------------------------------------------------------------------------
# Validation and snapping
# --------------------------------------------------------------------------


def clean_ranges(ranges: Sequence[tuple[float, float]], *, duration_s: float) -> list[Span]:
    """Validate picked ranges: clamped to the video, sorted, overlaps merged.

    Raises :class:`PickError` for no ranges, too many, non-finite numbers, a
    range outside the video, or one shorter than :data:`MIN_PICK_S`.
    """
    if not ranges:
        raise PickError("Pick at least one part of the video.")
    if len(ranges) > MAX_PICKS:
        raise PickError(f"Pick at most {MAX_PICKS} parts at a time.")
    limit = duration_s if duration_s > 0 else math.inf
    spans: list[Span] = []
    for start, end in ranges:
        if not (math.isfinite(start) and math.isfinite(end)):
            raise PickError("A picked part has an invalid start or end.")
        start, end = min(start, end), max(start, end)
        if end <= 0 or start >= limit:
            raise PickError("A picked part is outside the video.")
        start, end = max(0.0, start), min(limit, end)
        if end - start < MIN_PICK_S:
            raise PickError(f"Each picked part must be at least {MIN_PICK_S:g} second long.")
        spans.append(Span(round(start, 3), round(end, 3)))

    spans.sort(key=lambda s: s.start)
    merged: list[Span] = []
    for span in spans:
        if merged and span.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


def group(spans: Sequence[Span], *, join: bool) -> list[list[Span]]:
    """One group per clip: each span alone, or all of them joined.

    Every group is held to :data:`MAX_CLIP_S` in total; whatever is past it
    is cut (a joined clip drops trailing parts once it is full).
    """
    groups = [list(spans)] if join else [[s] for s in spans]
    out: list[list[Span]] = []
    for g in groups:
        room = MAX_CLIP_S
        capped: list[Span] = []
        for span in g:
            if room < MIN_PICK_S:
                break
            end = min(span.end, span.start + room)
            capped.append(Span(span.start, round(end, 3)))
            room -= end - span.start
        if capped:
            out.append(capped)
    return out


def snap_span(
    transcript: Transcript,
    span: Span,
    *,
    silences: Sequence[Silence] = (),
    breaks: SentenceBreaks | None = None,
) -> tuple[Span, int, int] | None:
    """Nudge a picked range onto clean cuts; returns it with its word range.

    The start moves to the first whole word (never mid-word), then to a
    sentence start if one is within :data:`SNAP_WINDOW_S`; the end likewise to
    the last whole word and a sentence end. Both edges then settle into a
    measured silence, without reaching into the neighbouring words. A snap
    that would shrink the range too much keeps the user's original edges.

    Returns None when the range holds no words at all (music, silence) - the
    caller keeps the raw range with no captions.
    """
    words = transcript.words
    if not words:
        return None
    if breaks is None:
        breaks = SentenceBreaks(transcript)
    inside = [i for i, w in enumerate(words) if span.start <= (w.start + w.end) / 2 < span.end]
    if not inside:
        return None
    first, last = inside[0], inside[-1]

    start_word = first
    candidate = snap_start_to_sentence(transcript, first, breaks)
    if candidate <= last and abs(words[candidate].start - span.start) <= SNAP_WINDOW_S:
        start_word = candidate
    end_word = last
    candidate = snap_end_to_sentence(transcript, last, breaks)
    if candidate >= start_word and abs(words[candidate].end - span.end) <= SNAP_WINDOW_S:
        end_word = candidate

    start_s = max(0.0, align_start(words[start_word].start, list(silences)))
    end_s = align_end(words[end_word].end, list(silences))
    start_s, end_s = _keep_out_of_neighbours(transcript, start_word, end_word, start_s, end_s)
    if end_s - start_s < min(SNAP_MIN_KEEP_S, span.duration * SNAP_MIN_KEEP_SHARE):
        return span, first, last
    return Span(round(start_s, 3), round(end_s, 3)), start_word, end_word


def plan(
    transcript: Transcript,
    ranges: Sequence[tuple[float, float]],
    *,
    duration_s: float,
    join: bool,
    silences: Sequence[Silence] = (),
) -> list[PickedClip]:
    """Validated, snapped and grouped clips for a pick request.

    Raises :class:`PickError` when the ranges are invalid.
    """
    spans = clean_ranges(ranges, duration_s=duration_s)
    breaks = SentenceBreaks(transcript) if transcript.words else None
    snapped: list[Span] = []
    for span in spans:
        result = snap_span(transcript, span, silences=silences, breaks=breaks) if breaks else None
        snapped.append(result[0] if result is not None else span)

    # Snapping only ever moves an edge a little, but two picks that nearly
    # touched can now overlap: merge them again.
    merged: list[Span] = []
    for span in sorted(snapped, key=lambda s: s.start):
        if merged and span.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)

    clips: list[PickedClip] = []
    for g in group(merged, join=join):
        start_word, end_word = word_range(transcript, g)
        clips.append(PickedClip(spans=g, start_word=start_word, end_word=end_word))
    return clips


def word_range(transcript: Transcript, spans: Sequence[Span]) -> tuple[int, int]:
    """First and last transcript word playing in ``spans``.

    A clip with no words at all (music, silence) gets an empty range at its
    start; its captions come out empty either way, because words are always
    filtered to the spans before rendering.
    """
    words = transcript.words
    inside = [
        i
        for i, w in enumerate(words)
        if any(s.start <= (w.start + w.end) / 2 < s.end for s in spans)
    ]
    if inside:
        return inside[0], inside[-1]
    index = next((i for i, w in enumerate(words) if w.start >= spans[0].start), len(words))
    index = max(0, min(index, len(words) - 1)) if words else 0
    return index, index


# --------------------------------------------------------------------------
# Clip rows
# --------------------------------------------------------------------------


def clock(seconds: float) -> str:
    """``m:ss`` (or ``h:mm:ss``) for clip titles."""
    s = int(max(0.0, seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def build_clips(
    job_id: str,
    picked: Sequence[PickedClip],
    *,
    first_rank: int,
    curve: AudienceCurve | None = None,
) -> list[Clip]:
    """Clip rows for picked clips, ranked after the job's existing clips.

    The score is how re-watched the picked parts are against the rest of the
    video (the audience curve's percentile), or a neutral 50 without one.
    """
    clips: list[Clip] = []
    for index, pick in enumerate(picked):
        number = f" #{index + 1}" if len(picked) > 1 else ""
        title = f"Your pick{number} · {clock(pick.start_s)}"
        score = _audience_score(curve, pick.spans)
        reason = "Picked by you on the rewatch graph"
        if pick.is_joined:
            reason += f" · {len(pick.spans)} parts joined"
        clips.append(
            Clip(
                id=new_id(),
                job_id=job_id,
                start_s=pick.start_s,
                end_s=pick.end_s,
                rank=first_rank + index,
                start_word=pick.start_word,
                end_word=pick.end_word,
                title=title,
                hook="",
                score=score if score is not None else 50,
                reason=reason,
                status="candidate",
            )
        )
    return clips


def _audience_score(curve: AudienceCurve | None, spans: Sequence[Span]) -> int | None:
    if curve is None or not curve.has_data:
        return None
    scores = [(curve.range_score(s.start, s.end), s.duration) for s in spans]
    weighted = [(v, d) for v, d in scores if v is not None]
    if not weighted:
        return None
    return round(sum(v * d for v, d in weighted) / sum(d for _, d in weighted))


def create(
    job: Job,
    workspace_segments: Callable[[str], Path],
    picked: Sequence[PickedClip],
    *,
    curve: AudienceCurve | None = None,
) -> list[Clip]:
    """Store the clip rows (and the sidecar of every joined clip)."""
    existing = store.list_clips(job.id)
    first_rank = max((c.rank for c in existing), default=0) + 1
    clips = build_clips(job.id, picked, first_rank=first_rank, curve=curve)
    for clip, pick in zip(clips, picked, strict=True):
        if pick.is_joined:
            segments.save(workspace_segments(clip.id), pick.spans)
    store.add_clips(clips)
    return clips


# --------------------------------------------------------------------------
# Rendering in the background
# --------------------------------------------------------------------------


@dataclass
class PickTask:
    """Progress of one job's picked-clip render, for the status endpoint."""

    job_id: str
    clip_ids: list[str]
    state: str = "running"  # "running" | "done" | "failed"
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    finished: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "clip_ids": list(self.clip_ids),
            "state": self.state,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
        }


_tasks: dict[str, PickTask] = {}
_lock = threading.Lock()


def task_for(job_id: str) -> PickTask | None:
    with _lock:
        return _tasks.get(job_id)


def is_running(job_id: str) -> bool:
    task = task_for(job_id)
    return task is not None and task.state == "running"


def start_task(job_id: str, clip_ids: list[str]) -> PickTask:
    task = PickTask(job_id=job_id, clip_ids=list(clip_ids))
    with _lock:
        _tasks[job_id] = task
    return task


#: The stages a picked clip still needs; detection is skipped (the user chose
#: the moments), and transcription already ran for the job.
_PICK_WEIGHTS = {Stage.REFRAME: 0.45, Stage.CAPTIONS: 0.02, Stage.EXPORT: 0.53}


def render(
    task: PickTask,
    job: Job,
    source: Source,
    clips: list[Clip],
    *,
    framing: str | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Reframe, caption and export picked clips with the job's own pipeline.

    Runs the same :class:`~autoclip.pipeline.runner.PipelineRunner` stages a
    detected clip goes through, but reports into ``task`` instead of the job
    row: the job itself stays "done" while its new clips render.
    """
    from .runner import PipelineRunner

    class _PickRunner(PipelineRunner):
        def _emit(self, stage, stage_progress, message="", *, overall=None) -> None:  # type: ignore[override]
            done = sum(w for s, w in _PICK_WEIGHTS.items() if _order(s) < _order(stage))
            task.stage = stage.value
            task.progress = min(1.0, done + _PICK_WEIGHTS.get(stage, 0.0) * stage_progress)
            task.message = message or stage.label

    try:
        runner = _PickRunner(job, source, is_cancelled=cancelled)
        if framing:
            runner.settings.export.framing = framing  # type: ignore[assignment]
        transcript = Transcript.load(runner.workspace.transcript)
        crop_paths = runner._stage_reframe(clips, transcript)
        runner._stage_captions(clips, transcript)
        runner._stage_export(clips, transcript, crop_paths)
    except Exception as exc:
        log.exception("Rendering picked clips for job %s failed.", job.id)
        task.state = "failed"
        task.error = str(exc) or type(exc).__name__
        return
    task.finished = [c.id for c in clips]
    task.progress = 1.0
    task.state = "done"
    task.message = f"{len(clips)} clip{'s' if len(clips) != 1 else ''} ready"


def _order(stage: Stage) -> int:
    return list(_PICK_WEIGHTS).index(stage) if stage in _PICK_WEIGHTS else -1
