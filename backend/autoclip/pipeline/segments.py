"""Multi-part ("joined") clips: several source spans played back to back.

A joined clip has two time bases:

* **source time** - absolute seconds in the source video, where its spans,
  the transcript's words and the measured silences live;
* **joined time** - seconds from the start of the rendered clip, 0 to the
  summed length of the spans. The export, its crop path, captions and edit
  plan all work here.

This module converts between the two, and stores a clip's spans in a JSON
sidecar beside the job's other per-clip artifacts (``work/<job>/segments/
<clip_id>.json``). The ``clips`` row keeps the envelope - first span start to
last span end - so lists and sorting still work; the sidecar is what makes it
a joined clip. No sidecar means an ordinary one-span clip.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .edit import Keep, retime_words
from .prepare import Silence
from .reframe.croppath import CropPath
from .transcript import Word

log = logging.getLogger(__name__)

#: Spans shorter than this are dropped when a sidecar is read or clipped.
MIN_SPAN_S = 0.1


@dataclass(frozen=True)
class Span:
    """One part of a joined clip, in source seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def total(spans: Sequence[Span]) -> float:
    """Length of the joined clip."""
    return sum(s.duration for s in spans)


def offsets(spans: Sequence[Span]) -> list[float]:
    """Joined-time start of each span."""
    out: list[float] = []
    t = 0.0
    for span in spans:
        out.append(t)
        t += span.duration
    return out


def to_joined(t: float, spans: Sequence[Span]) -> float | None:
    """Source time ``t`` on the joined timeline, or None if no span holds it."""
    for span, offset in zip(spans, offsets(spans), strict=True):
        if span.start <= t <= span.end:
            return offset + t - span.start
    return None


def to_source(t: float, spans: Sequence[Span]) -> float:
    """Joined time ``t`` back in source time (clamped to the clip)."""
    if not spans:
        return t
    for span, offset in zip(spans, offsets(spans), strict=True):
        if t < offset + span.duration:
            return span.start + max(0.0, t - offset)
    return spans[-1].end


def range_to_joined(
    rng: tuple[float, float] | None, spans: Sequence[Span]
) -> tuple[float, float] | None:
    """A source-time range on the joined timeline, or None.

    Only a range that lies inside a single span maps; one that reaches into a
    gap between parts (or is None) doesn't exist in the joined clip.
    """
    if rng is None:
        return None
    start, end = rng
    for span, offset in zip(spans, offsets(spans), strict=True):
        if span.start <= start and end <= span.end:
            return (offset + start - span.start, offset + end - span.start)
    return None


def words_on_timeline(words: Sequence[Word], spans: Sequence[Span]) -> list[Word]:
    """Words (source time) re-timed onto the joined clip.

    A word belongs to the span holding its midpoint and is clipped to it;
    words in the gaps between spans (or whose middle falls outside every span)
    are dropped. That is what keeps the captions in sync across every join.
    """
    return retime_words(words, [Keep(s.start, s.end) for s in spans])


def words_in(words: Sequence[Word], spans: Sequence[Span]) -> list[Word]:
    """The words (still in source time) that play in a joined clip."""
    return [w for w in words if any(s.start <= (w.start + w.end) / 2 < s.end for s in spans)]


def silences_on_timeline(silences: Sequence[Silence], spans: Sequence[Span]) -> list[Silence]:
    """Measured silences (source time) moved onto the joined clip, cut at the joins."""
    out: list[Silence] = []
    for span, offset in zip(spans, offsets(spans), strict=True):
        for silence in silences:
            start = max(silence.start, span.start)
            end = min(silence.end, span.end)
            if end > start:
                out.append(
                    Silence(
                        start=round(offset + start - span.start, 4),
                        end=round(offset + end - span.start, 4),
                    )
                )
    return out


def clip_to(spans: Sequence[Span], start: float, end: float) -> list[Span]:
    """The spans intersected with ``[start, end]`` (a joined clip that was trimmed)."""
    out: list[Span] = []
    for span in spans:
        a, b = max(span.start, start), min(span.end, end)
        if b - a >= MIN_SPAN_S:
            out.append(Span(a, b))
    return out


def joined_crop_path(paths: Sequence[CropPath]) -> CropPath:
    """Per-span crop paths laid end to end on the joined timeline."""
    from .reframe.croppath import join_paths

    return join_paths(list(paths))


# --------------------------------------------------------------------------
# Sidecar storage
# --------------------------------------------------------------------------


def save(path: Path, spans: Sequence[Span]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "spans": [[s.start, s.end] for s in spans]}),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load(path: Path) -> list[Span] | None:
    """The spans saved for a clip; None when it is an ordinary clip.

    An unreadable sidecar is logged and treated as absent: the clip then
    renders as its envelope, which is wrong but recoverable, rather than not
    at all.
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        spans = [Span(float(a), float(b)) for a, b in raw["spans"]]
    except (OSError, ValueError, TypeError, KeyError):
        log.warning("Ignoring an unreadable segments file %s.", path)
        return None
    spans = [s for s in spans if s.duration >= MIN_SPAN_S]
    return spans or None


def as_pairs(spans: Sequence[Span]) -> list[tuple[float, float]]:
    """The ``ExportRequest.spans`` form."""
    return [(s.start, s.end) for s in spans]
