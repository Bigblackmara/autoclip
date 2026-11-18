"""Crop path representation and rendering to ffmpeg expressions.

A crop path is a list of :class:`CropSegment` — one per shot. Within a segment
the crop window has a fixed size and may pan; between segments the size may
change, because each segment is scaled to the same output dimensions before
being concatenated. That's what lets a wide two-shot and a tight single coexist
in one clip without a mid-stream frame-size change, which ffmpeg filtergraphs
don't allow.

Panning is emitted as a piecewise-linear expression over ``t`` rather than a
``sendcmd`` script: it keeps everything in one filtergraph, survives seeking,
and is far easier to inspect when a render looks wrong.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Cap on keyframes per segment. Beyond this the expression gets unwieldy and
#: the extra precision is invisible — the path is smoothed and dead-zoned
#: upstream, so adjacent keyframes differ by well under a pixel.
MAX_KEYFRAMES_PER_SEGMENT = 48

#: Movement below this many pixels is dropped during decimation.
KEYFRAME_EPSILON_PX = 1.0


class Strategy(StrEnum):
    """How a shot was framed.

    Recorded per segment so the review UI can explain a framing decision, and so
    regression tests can assert on strategy rather than raw pixels.
    """

    #: One clear subject; crop follows them, or locks if they barely move.
    TRACK = "track"
    #: Several faces with no clear active speaker; frame wide enough to hold them.
    WIDE = "wide"
    #: No usable face; centre crop, optionally with a slow push-in.
    GENERAL = "general"
    #: Two people far apart (a podcast two-shot): each gets their own half of
    #: the frame, stacked top and bottom. See :attr:`CropSegment.panes`.
    SPLIT = "split"


@dataclass
class CropKeyframe:
    """Top-left corner of the crop window at time ``t`` (clip-relative seconds)."""

    t: float
    x: float
    y: float


@dataclass
class CropRect:
    """A fixed window in source pixels (top-left corner plus size)."""

    x: float
    y: float
    width: int
    height: int


@dataclass
class CropSegment:
    start_s: float
    end_s: float
    width: int
    height: int
    keyframes: list[CropKeyframe] = field(default_factory=list)
    strategy: Strategy = Strategy.GENERAL
    #: Fractional zoom applied over the segment, for the GENERAL slow push-in.
    #: 0.0 disables it.
    zoom: float = 0.0
    #: Fit the whole source frame into the output instead of cropping, filling
    #: the remainder with a blurred copy. Used when the subjects are spread
    #: wider than any crop of the target ratio can hold — cropping there would
    #: cut someone out of frame, which the acceptance bar forbids.
    fit: bool = False
    #: Two-speaker split screen: the top and bottom halves of the output, each
    #: a steady window around one person. Empty for every other strategy. The
    #: segment's own ``width``/``height``/``keyframes`` still describe an
    #: ordinary crop of the target ratio, used where a split can't render
    #: (non-vertical ratios) and by the review preview.
    panes: list[CropRect] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def is_split(self) -> bool:
        return len(self.panes) == 2

    @property
    def is_static(self) -> bool:
        """True when the crop never moves, allowing a constant-value expression."""
        if len(self.keyframes) <= 1:
            return True
        first = self.keyframes[0]
        return all(
            abs(k.x - first.x) < KEYFRAME_EPSILON_PX and abs(k.y - first.y) < KEYFRAME_EPSILON_PX
            for k in self.keyframes
        )


@dataclass
class CropPath:
    """The full crop plan for one clip."""

    source_width: int
    source_height: int
    segments: list[CropSegment] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0

    @property
    def split_share(self) -> float:
        """Fraction of the clip rendered as a two-speaker split (0..1)."""
        total = self.duration_s
        if total <= 0:
            return 0.0
        return sum(s.duration_s for s in self.segments if s.is_split) / total

    @property
    def has_split(self) -> bool:
        return any(s.is_split for s in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "segments": [{**asdict(s), "strategy": s.strategy.value} for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropPath:
        segments = []
        for raw in data.get("segments", []):
            keyframes = [CropKeyframe(**k) for k in raw.get("keyframes", [])]
            segments.append(
                CropSegment(
                    start_s=raw["start_s"],
                    end_s=raw["end_s"],
                    width=raw["width"],
                    height=raw["height"],
                    keyframes=keyframes,
                    strategy=Strategy(raw.get("strategy", "general")),
                    zoom=raw.get("zoom", 0.0),
                    fit=raw.get("fit", False),
                    panes=[CropRect(**p) for p in raw.get("panes") or []],
                )
            )
        return cls(
            source_width=data["source_width"],
            source_height=data["source_height"],
            segments=segments,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> CropPath:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------


def fit_segment(start_s: float, end_s: float, source_width: int, source_height: int) -> CropSegment:
    """The whole frame fitted over a blurred copy of itself.

    The answer for screen recordings, slides and gameplay with no presenter: a
    blind centre crop of those throws away most of what the viewer needs.
    """
    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=source_width,
        height=source_height,
        keyframes=[CropKeyframe(t=start_s, x=0.0, y=0.0)],
        strategy=Strategy.GENERAL,
        fit=True,
    )


def fit_path(source_width: int, source_height: int, duration_s: float) -> CropPath:
    """A one-segment path that fits the whole clip (see :func:`fit_segment`)."""
    return CropPath(
        source_width=source_width,
        source_height=source_height,
        segments=[fit_segment(0.0, duration_s, source_width, source_height)],
    )


def shifted(segment: CropSegment, offset_s: float) -> CropSegment:
    """A copy of ``segment`` moved ``offset_s`` later on the timeline."""
    return CropSegment(
        start_s=segment.start_s + offset_s,
        end_s=segment.end_s + offset_s,
        width=segment.width,
        height=segment.height,
        keyframes=[CropKeyframe(k.t + offset_s, k.x, k.y) for k in segment.keyframes],
        strategy=segment.strategy,
        zoom=segment.zoom,
        fit=segment.fit,
        panes=[CropRect(p.x, p.y, p.width, p.height) for p in segment.panes],
    )


def join_paths(paths: list[CropPath]) -> CropPath:
    """Concatenate per-part crop paths onto one joined timeline.

    Each path covers its own part from 0; part ``k`` is shifted by the summed
    length of the parts before it, so segment boundaries land exactly on the
    joins and no segment ever straddles two parts.

    Preconditions:
        ``paths`` is non-empty and every path has at least one segment.
    """
    offset = 0.0
    segments: list[CropSegment] = []
    for path in paths:
        segments.extend(shifted(s, offset) for s in path.segments)
        offset += path.duration_s
    return CropPath(
        source_width=paths[0].source_width,
        source_height=paths[0].source_height,
        segments=segments,
    )


def split_at(path: CropPath, times: list[float]) -> CropPath:
    """Cut segments at ``times`` (e.g. the joins of a multi-part clip).

    Keyframes are left as they are — they're absolute on the timeline, so each
    piece still evaluates to the same positions. A path built by
    :func:`join_paths` comes back unchanged; this is for paths that weren't,
    such as a centre-crop fallback spanning the whole joined clip.
    """
    cuts = sorted(t for t in times if t > 0)
    out: list[CropSegment] = []
    for segment in path.segments:
        edges = [segment.start_s]
        edges += [t for t in cuts if segment.start_s + 1e-6 < t < segment.end_s - 1e-6]
        edges.append(segment.end_s)
        for a, b in zip(edges, edges[1:], strict=False):
            piece = shifted(segment, 0.0)
            piece.start_s, piece.end_s = a, b
            inside = [k for k in piece.keyframes if a <= k.t <= b]
            if not inside or inside[0].t > a:
                at = _position_at(segment.keyframes, a)
                inside.insert(0, CropKeyframe(a, at[0], at[1]))
            piece.keyframes = inside
            out.append(piece)
    return CropPath(path.source_width, path.source_height, out)


def _position_at(keyframes: list[CropKeyframe], t: float) -> tuple[float, float]:
    """Linear interpolation of the crop position at ``t``."""
    if not keyframes:
        return 0.0, 0.0
    if t <= keyframes[0].t:
        return keyframes[0].x, keyframes[0].y
    for k0, k1 in zip(keyframes, keyframes[1:], strict=False):
        if k0.t <= t <= k1.t:
            span = k1.t - k0.t
            r = (t - k0.t) / span if span > 0 else 0.0
            return k0.x + (k1.x - k0.x) * r, k0.y + (k1.y - k0.y) * r
    return keyframes[-1].x, keyframes[-1].y


def target_crop_size(
    source_width: int, source_height: int, aspect_w: int, aspect_h: int
) -> tuple[int, int]:
    """Largest window of the target aspect ratio that fits inside the source.

    Dimensions are forced even, because H.264 with yuv420p chroma subsampling
    requires it and an odd value fails the encode.
    """
    target_ratio = aspect_w / aspect_h
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        # Source is wider than the target: full height, cropped width.
        height = source_height
        width = round(height * target_ratio)
    else:
        width = source_width
        height = round(width / target_ratio)

    width = min(source_width, width - (width % 2))
    height = min(source_height, height - (height % 2))
    return max(2, width), max(2, height)


def centre_crop(
    source_width: int,
    source_height: int,
    duration_s: float,
    *,
    aspect_w: int = 9,
    aspect_h: int = 16,
    zoom: float = 0.0,
) -> CropPath:
    """Build a static centred crop path.

    The Phase 1 fallback, and still the correct answer whenever no face is
    detected in a shot.
    """
    width, height = target_crop_size(source_width, source_height, aspect_w, aspect_h)
    x = (source_width - width) / 2
    y = (source_height - height) / 2

    return CropPath(
        source_width=source_width,
        source_height=source_height,
        segments=[
            CropSegment(
                start_s=0.0,
                end_s=duration_s,
                width=width,
                height=height,
                keyframes=[CropKeyframe(t=0.0, x=x, y=y)],
                strategy=Strategy.GENERAL,
                zoom=zoom,
            )
        ],
    )


def decimate(
    keyframes: list[CropKeyframe], *, limit: int = MAX_KEYFRAMES_PER_SEGMENT
) -> list[CropKeyframe]:
    """Reduce keyframes to at most ``limit``, keeping the ones that matter.

    Drops keyframes whose position is within :data:`KEYFRAME_EPSILON_PX` of the
    straight line between their neighbours, then thins uniformly if still over
    budget. Endpoints are always preserved.
    """
    if len(keyframes) <= 2:
        return list(keyframes)

    kept: list[CropKeyframe] = [keyframes[0]]
    for previous, current, following in zip(keyframes, keyframes[1:], keyframes[2:], strict=False):
        span = following.t - previous.t
        if span <= 0:
            continue
        ratio = (current.t - previous.t) / span
        predicted_x = previous.x + (following.x - previous.x) * ratio
        predicted_y = previous.y + (following.y - previous.y) * ratio
        if (
            abs(current.x - predicted_x) > KEYFRAME_EPSILON_PX
            or abs(current.y - predicted_y) > KEYFRAME_EPSILON_PX
        ):
            kept.append(current)
    kept.append(keyframes[-1])

    if len(kept) <= limit:
        return kept

    step = len(kept) / (limit - 1)
    thinned = [kept[min(len(kept) - 1, round(i * step))] for i in range(limit - 1)]
    thinned.append(kept[-1])

    deduped: list[CropKeyframe] = []
    for frame in thinned:
        if not deduped or frame.t > deduped[-1].t:
            deduped.append(frame)
    return deduped


# --------------------------------------------------------------------------
# ffmpeg expressions
# --------------------------------------------------------------------------


def axis_expression(keyframes: list[CropKeyframe], axis: str, *, offset_s: float = 0.0) -> str:
    """Build a piecewise-linear ffmpeg expression for one axis.

    ``offset_s`` is subtracted from keyframe times, which is what makes an
    expression usable inside a ``trim``-ed segment whose PTS was reset to zero.

    Preconditions:
        keyframes are sorted by ``t`` and non-empty; axis is "x" or "y".
    """
    if not keyframes:
        return "0"

    values = [(round(k.t - offset_s, 4), round(getattr(k, axis), 2)) for k in keyframes]

    if len(values) == 1:
        return str(values[0][1])

    # Built from the last pair backwards so each nested else-branch is the
    # already-constructed expression for the remaining time.
    expression = str(values[-1][1])
    for (t0, v0), (t1, v1) in zip(reversed(values[:-1]), reversed(values[1:]), strict=True):
        span = t1 - t0
        if span <= 0:
            continue
        slope = round((v1 - v0) / span, 6)
        ramp = f"({v0}+{slope}*(t-{t0}))"
        expression = f"if(lt(t,{t1}),{ramp},{expression})"

    return expression


def segment_crop_filter(segment: CropSegment, *, relative: bool = True) -> str:
    """Render one segment's ``crop`` filter.

    ``relative`` reflects whether the segment's PTS has been reset to zero by a
    preceding ``trim`` — true in the normal multi-segment render path.
    """
    offset = segment.start_s if relative else 0.0
    width, height = segment.width, segment.height

    if segment.is_static or len(segment.keyframes) <= 1:
        first = segment.keyframes[0] if segment.keyframes else CropKeyframe(0.0, 0.0, 0.0)
        x_expr, y_expr = str(round(first.x, 2)), str(round(first.y, 2))
    else:
        frames = decimate(segment.keyframes)
        x_expr = axis_expression(frames, "x", offset_s=offset)
        y_expr = axis_expression(frames, "y", offset_s=offset)

    # Escape commas so the expression survives filtergraph parsing.
    x_expr = x_expr.replace(",", "\\,")
    y_expr = y_expr.replace(",", "\\,")

    return f"crop=w={width}:h={height}:x='{x_expr}':y='{y_expr}'"
