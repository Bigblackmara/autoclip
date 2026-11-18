"""Reframe stage — 16:9 source to a 9:16 crop path that follows the speaker.

Pipeline for one clip:

1. Split into shots. Crop paths never interpolate across a cut.
2. Sample face landmarks across the clip.
3. Link observations into per-person tracks.
4. Map diarized speakers onto tracks once, then let turn boundaries drive framing.
5. Per shot, pick a strategy — TRACK, WIDE, GENERAL, or (with the "auto" and
   "split" framings) SPLIT for a two-person shot and a blurred fit for a shot
   with no faces at all.
6. Build a raw crop path, smooth it, and emit segments.

The quality bar is "never jarring": no visible jitter, no cut-off faces, the
speaker on screen for essentially all of their speaking time. Every default here
is biased toward stillness — a locked frame that is slightly off-centre beats a
frame that is always correct and always moving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .. import ffmpeg
from ..transcript import Transcript
from .croppath import (
    CropKeyframe,
    CropPath,
    CropRect,
    CropSegment,
    Strategy,
    centre_crop,
    fit_path,
    fit_segment,
    target_crop_size,
)
from .faces import FaceDetectionUnavailable, FaceObservation, sample_faces
from .scenes import Shot, detect_shots
from .smoothing import SmoothingConfig, smooth_series
from .speaker import SILENT_MAR_VARIANCE, assign_speakers, map_tracks_to_speakers
from .tracker import FaceTrack, build_tracks

log = logging.getLogger(__name__)

__all__ = [
    "FRAMINGS",
    "CropPath",
    "CropSegment",
    "ReframeConfig",
    "Strategy",
    "build_crop_path",
    "plan_shot_layout",
]

#: The framing modes (``ExportSettings.framing``):
#:
#: * ``auto`` - the planner decides per shot: two people too far apart for one
#:   crop become a top/bottom split, a shot with no faces (screen recording,
#:   slides, gameplay) is fitted over a blurred copy, everything else gets the
#:   speaker-following crop.
#: * ``fill`` - always crop (the original behaviour; faceless shots centre-crop).
#: * ``split`` - split whenever two persistent faces share a shot, even fairly
#:   close together or with no speech evidence; otherwise as ``auto``.
#: * ``fit`` - the whole frame over a blurred copy, for the whole clip.
FRAMINGS = ("auto", "fill", "split", "fit")

#: Where the eye line sits within the crop, as a fraction of crop height.
#: Portrait convention puts the eyes near the upper third; 0.38 reads as
#: composed without leaving the chin tight to the bottom edge.
EYE_LINE_RATIO = 0.38

#: A face wider than this fraction of the crop is already a close-up; don't
#: track it, lock the frame.
LOCK_IF_FACE_WIDER_THAN = 0.55

#: If detected faces span more than this fraction of the crop width, no crop can
#: hold them and we fall back to a fitted (blurred-background) frame.
FIT_IF_SPREAD_EXCEEDS = 0.95

#: A shot with faces in fewer than this share of its samples has no presenter
#: (screen, slides, gameplay) and is fitted rather than cropped.
FIT_IF_FACE_PRESENCE_BELOW = 0.25

#: Faces narrower than this fraction of the source width are ignored when
#: deciding whether a shot has a presenter: crowds, posters, HUD portraits.
MIN_PRESENTER_FACE_WIDTH = 0.025

#: A face must be seen in at least this share of a shot's samples to be one of
#: the two people of a split.
SPLIT_MIN_PRESENCE = 0.6

#: In "auto", the two faces must be further apart than this fraction of the
#: crop width; closer than that a single wide crop holds them both.
SPLIT_MIN_SEPARATION = 0.85

#: In "split", any two faces further apart than this multiple of the wider
#: face's width qualify.
SPLIT_FORCED_SEPARATION = 1.2

#: The smaller face's area must be at least this share of the larger one's: a
#: small face in the background is not the other half of a two-shot.
SPLIT_MIN_AREA_RATIO = 0.25

#: Shots shorter than this never split; not worth re-laying out the frame.
SPLIT_MIN_SHOT_S = 1.5

#: Pane width as a multiple of the face width: a medium close-up.
SPLIT_PANE_FACE_SCALE = 3.2

#: Smallest pane as a fraction of the source width, bounding the upscale.
SPLIT_MIN_PANE_WIDTH = 0.22

#: Where the eye line sits within each pane, as a fraction of its height.
SPLIT_EYE_LINE_RATIO = 0.40

#: Slow push-in applied to shots with no face at all, to stop a static wide from
#: feeling like a still image. Off by default — see the note in export._zoom_filter.
GENERAL_ZOOM = 0.0


@dataclass
class ReframeConfig:
    aspect_w: int = 9
    aspect_h: int = 16
    sample_fps: float = 5.0
    smoothing: SmoothingConfig | None = None
    #: Skip face detection entirely and centre-crop everything.
    centre_only: bool = False
    #: One of :data:`FRAMINGS`. Defaults to the original crop-only behaviour so
    #: callers that never asked for it (Live Watch) are unchanged; normal-mode
    #: jobs pass ``ExportSettings.framing``.
    framing: str = "fill"

    @property
    def pane_aspect(self) -> float | None:
        """Aspect of each half of a split (9:16 gives 9:8), or None when the
        target isn't tall enough to stack two panes."""
        if self.aspect_h <= self.aspect_w:
            return None
        return self.aspect_w / (self.aspect_h / 2)


def build_crop_path(
    video: Path,
    *,
    start_s: float,
    end_s: float,
    transcript: Transcript | None = None,
    config: ReframeConfig | None = None,
) -> CropPath:
    """Compute the crop path for one clip.

    Falls back to a centred crop whenever face detection is unavailable or finds
    nothing — a centre crop is an acceptable result, an exception is not.

    Preconditions:
        video exists; 0 <= start_s < end_s <= source duration.
    """
    config = config or ReframeConfig()
    duration = end_s - start_s

    info = ffmpeg.probe(video)
    if not info.has_video or not info.width or not info.height:
        raise ValueError(f"{video.name} has no video stream to reframe.")

    source_w, source_h = info.width, info.height
    crop_w, crop_h = target_crop_size(source_w, source_h, config.aspect_w, config.aspect_h)

    if config.centre_only:
        return centre_crop(
            source_w, source_h, duration, aspect_w=config.aspect_w, aspect_h=config.aspect_h
        )

    framing = config.framing if config.framing in FRAMINGS else "fill"
    if framing == "fit":
        return fit_path(source_w, source_h, duration)

    try:
        observations = sample_faces(
            video, start_s=start_s, end_s=end_s, sample_fps=config.sample_fps
        )
    except FaceDetectionUnavailable as exc:
        log.warning("Face detection unavailable (%s); using a centre crop.", exc)
        return centre_crop(
            source_w,
            source_h,
            duration,
            aspect_w=config.aspect_w,
            aspect_h=config.aspect_h,
            zoom=GENERAL_ZOOM,
        )

    if not observations and framing in ("auto", "split"):
        log.info("No faces detected in %s; fitting the whole frame.", video.name)
        return fit_path(source_w, source_h, duration)

    if not observations:
        log.info("No faces detected in %s; using a centre crop.", video.name)
        return centre_crop(
            source_w,
            source_h,
            duration,
            aspect_w=config.aspect_w,
            aspect_h=config.aspect_h,
            zoom=GENERAL_ZOOM,
        )

    tracks = build_tracks(observations)
    shots = detect_shots(video, start_s=start_s, end_s=end_s)

    turns = _clip_relative_turns(transcript, start_s, end_s)
    speaker_map = map_tracks_to_speakers(tracks, turns) if turns else {}

    segments: list[CropSegment] = []
    for shot in shots:
        if framing in ("auto", "split"):
            planned = plan_shot_layout(
                shot,
                observations=observations,
                tracks=tracks,
                speaker_map=speaker_map,
                source_w=source_w,
                source_h=source_h,
                crop_w=crop_w,
                crop_h=crop_h,
                config=config,
            )
            if planned is not None:
                segments.append(planned)
                continue
        segments.extend(
            _segments_for_shot(
                shot,
                tracks=tracks,
                turns=turns,
                speaker_map=speaker_map,
                source_w=source_w,
                source_h=source_h,
                crop_w=crop_w,
                crop_h=crop_h,
                config=config,
            )
        )

    if not segments:
        return centre_crop(
            source_w, source_h, duration, aspect_w=config.aspect_w, aspect_h=config.aspect_h
        )

    # Guarantee the segments tile the clip exactly; a gap or overlap would
    # desync the concatenated render from the audio.
    segments[0].start_s = 0.0
    segments[-1].end_s = duration
    for index in range(len(segments) - 1):
        segments[index].end_s = segments[index + 1].start_s

    segments = [s for s in segments if s.duration_s > 0.01]

    log.info(
        "Reframed %s into %d segment(s): %s",
        video.name,
        len(segments),
        ", ".join(s.strategy.value for s in segments),
    )
    return CropPath(source_width=source_w, source_height=source_h, segments=segments)


def plan_shot_layout(
    shot: Shot,
    *,
    observations: list[FaceObservation],
    tracks: list[FaceTrack],
    speaker_map: dict[str, FaceTrack],
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    config: ReframeConfig,
) -> CropSegment | None:
    """The whole-shot layout for "auto"/"split" framing, or None to crop as usual.

    * No presenter in the shot: a blurred fit of the whole frame.
    * Two persistent people too far apart for one crop: a SPLIT, each person
      steady in their own half for the whole shot.

    Pure: decisions come from the observations alone, so they can be tested
    with synthetic face tracks.
    """
    expected = max(1.0, shot.duration_s * config.sample_fps)
    min_width = source_w * MIN_PRESENTER_FACE_WIDTH
    present = {
        round(o.t, 3)
        for o in observations
        if shot.start_s <= o.t < shot.end_s and o.width >= min_width
    }
    if len(present) / expected < FIT_IF_FACE_PRESENCE_BELOW:
        return fit_segment(shot.start_s, shot.end_s, source_w, source_h)

    pane_aspect = config.pane_aspect
    if pane_aspect is None or shot.duration_s < SPLIT_MIN_SHOT_S:
        return None
    pair = _split_pair(
        shot,
        tracks,
        speaker_map=speaker_map,
        expected=expected,
        min_width=min_width,
        crop_w=crop_w,
        forced=config.framing == "split",
    )
    if pair is None:
        return None
    return _split_segment(
        shot,
        pair,
        source_w=source_w,
        source_h=source_h,
        crop_w=crop_w,
        crop_h=crop_h,
        pane_aspect=pane_aspect,
    )


def _split_pair(
    shot: Shot,
    tracks: list[FaceTrack],
    *,
    speaker_map: dict[str, FaceTrack],
    expected: float,
    min_width: float,
    crop_w: int,
    forced: bool,
) -> tuple[list[FaceObservation], list[FaceObservation]] | None:
    """The two people of a split shot, left one first; None if it isn't one."""
    persistent: list[tuple[FaceTrack, list[FaceObservation]]] = []
    for track in tracks:
        seen = track.observations_between(shot.start_s, shot.end_s)
        if not seen or len(seen) / expected < SPLIT_MIN_PRESENCE:
            continue
        if _median([o.width for o in seen]) < min_width:
            continue
        persistent.append((track, seen))
    if len(persistent) < 2:
        return None

    persistent.sort(key=lambda item: -_median([o.area for o in item[1]]))
    (track_a, seen_a), (track_b, seen_b) = persistent[:2]
    area_a = _median([o.area for o in seen_a])
    area_b = _median([o.area for o in seen_b])
    if min(area_a, area_b) < max(area_a, area_b) * SPLIT_MIN_AREA_RATIO:
        return None

    cx_a = _median([o.cx for o in seen_a])
    cx_b = _median([o.cx for o in seen_b])
    separation = abs(cx_a - cx_b)
    if forced:
        wider = max(_median([o.width for o in seen_a]), _median([o.width for o in seen_b]))
        if separation < wider * SPLIT_FORCED_SEPARATION:
            return None
    else:
        if separation <= crop_w * SPLIT_MIN_SEPARATION:
            return None
        # Two people talking: both mapped to diarized speakers, or both mouths
        # moving somewhere in the clip. A listener who never speaks, or a face
        # on a poster, doesn't earn half the frame.
        speakers = {t.id for t in speaker_map.values()}
        if speakers:
            if not {track_a.id, track_b.id} <= speakers:
                return None
        elif not all(
            t.mouth_activity(t.start_s, t.end_s + 1e-3) > SILENT_MAR_VARIANCE
            for t in (track_a, track_b)
        ):
            return None

    return (seen_a, seen_b) if cx_a <= cx_b else (seen_b, seen_a)


def _split_segment(
    shot: Shot,
    pair: tuple[list[FaceObservation], list[FaceObservation]],
    *,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    pane_aspect: float,
) -> CropSegment:
    """One steady window per person (left person on top), held for the shot."""
    left, right = pair
    face_w = (_median([o.width for o in left]) + _median([o.width for o in right])) / 2
    max_w = min(source_w, source_h * pane_aspect)
    min_w = min(max_w, source_w * SPLIT_MIN_PANE_WIDTH)
    pane_w = _even(max(min_w, min(face_w * SPLIT_PANE_FACE_SCALE, max_w)))
    pane_h = min(_even(pane_w / pane_aspect), source_h - source_h % 2)

    panes: list[CropRect] = []
    for seen in (left, right):
        cx = _median([o.cx for o in seen])
        eye_y = _median([o.eye_y for o in seen])
        panes.append(
            CropRect(
                x=round(_clamp(cx - pane_w / 2, 0, source_w - pane_w), 2),
                y=round(_clamp(eye_y - SPLIT_EYE_LINE_RATIO * pane_h, 0, source_h - pane_h), 2),
                width=pane_w,
                height=pane_h,
            )
        )

    # The ordinary-crop stand-in (other ratios, the review preview): the
    # centroid of both faces.
    centre = (_median([o.cx for o in left]) + _median([o.cx for o in right])) / 2
    eye = (_median([o.eye_y for o in left]) + _median([o.eye_y for o in right])) / 2
    return CropSegment(
        start_s=shot.start_s,
        end_s=shot.end_s,
        width=crop_w,
        height=crop_h,
        keyframes=[
            CropKeyframe(
                t=shot.start_s,
                x=_clamp(centre - crop_w / 2, 0, source_w - crop_w),
                y=_clamp(eye - EYE_LINE_RATIO * crop_h, 0, source_h - crop_h),
            )
        ],
        strategy=Strategy.SPLIT,
        panes=panes,
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _even(value: float) -> int:
    whole = int(value)
    return max(2, whole - whole % 2)


def _clip_relative_turns(
    transcript: Transcript | None, start_s: float, end_s: float
) -> list[tuple[str, float, float]]:
    """Rebase diarization turns onto the clip's timeline, dropping non-overlapping ones."""
    if transcript is None or not transcript.has_diarization:
        return []

    turns: list[tuple[str, float, float]] = []
    for speaker, turn_start, turn_end in transcript.speaker_turns():
        overlap_start = max(turn_start, start_s)
        overlap_end = min(turn_end, end_s)
        if overlap_end > overlap_start:
            turns.append((speaker, overlap_start - start_s, overlap_end - start_s))
    return turns


def _segments_for_shot(
    shot: Shot,
    *,
    tracks: list[FaceTrack],
    turns: list[tuple[str, float, float]],
    speaker_map: dict[str, FaceTrack],
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    config: ReframeConfig,
) -> list[CropSegment]:
    assignments = assign_speakers(
        tracks,
        shot_start_s=shot.start_s,
        shot_end_s=shot.end_s,
        turns=turns,
        speaker_map=speaker_map,
    )

    segments: list[CropSegment] = []
    for assignment in assignments:
        visible = [
            t for t in tracks if t.observations_between(assignment.start_s, assignment.end_s)
        ]

        if assignment.track is not None:
            segments.append(
                _track_segment(
                    assignment.track,
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    config=config,
                )
            )
        elif visible:
            segments.append(
                _wide_segment(
                    visible,
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
            )
        else:
            segments.append(
                _general_segment(
                    start_s=assignment.start_s,
                    end_s=assignment.end_s,
                    source_w=source_w,
                    source_h=source_h,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
            )

    return segments


def _track_segment(
    track: FaceTrack,
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
    config: ReframeConfig,
) -> CropSegment:
    """Frame a single subject, tracking them or locking on if they barely move."""
    observations = track.observations_between(start_s, end_s)
    if not observations:
        return _general_segment(
            start_s=start_s,
            end_s=end_s,
            source_w=source_w,
            source_h=source_h,
            crop_w=crop_w,
            crop_h=crop_h,
        )

    raw = [
        (
            o.t,
            _clamp(o.cx - crop_w / 2, 0, source_w - crop_w),
            _clamp(o.eye_y - EYE_LINE_RATIO * crop_h, 0, source_h - crop_h),
        )
        for o in observations
    ]

    mean_face_width = sum(o.width for o in observations) / len(observations)
    spread_x = max(x for _, x, _ in raw) - min(x for _, x, _ in raw)

    # A close-up, or a subject who barely moves, is better locked than tracked.
    should_lock = (
        mean_face_width > crop_w * LOCK_IF_FACE_WIDER_THAN
        or spread_x < (config.smoothing or SmoothingConfig()).dead_zone_px
    )

    if should_lock or len(raw) < 3:
        x = sum(item[1] for item in raw) / len(raw)
        y = sum(item[2] for item in raw) / len(raw)
        keyframes = [CropKeyframe(t=start_s, x=x, y=y)]
    else:
        smoothing = config.smoothing or SmoothingConfig()
        xs = smooth_series([(t, x) for t, x, _ in raw], smoothing)
        ys = smooth_series([(t, y) for t, _, y in raw], smoothing)
        keyframes = [CropKeyframe(t=t, x=x, y=y) for (t, x), (_, y) in zip(xs, ys, strict=True)]
        # Anchor the ends so the expression covers the whole segment.
        if keyframes[0].t > start_s:
            keyframes.insert(0, CropKeyframe(start_s, keyframes[0].x, keyframes[0].y))
        if keyframes[-1].t < end_s:
            keyframes.append(CropKeyframe(end_s, keyframes[-1].x, keyframes[-1].y))

    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=keyframes,
        strategy=Strategy.TRACK,
    )


def _wide_segment(
    visible: list[FaceTrack],
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
) -> CropSegment:
    """Hold several faces at once — locked, never panning.

    Panning between two people who are both on screen is the most obviously
    robotic thing an auto-reframer does. If they fit, lock on their centroid; if
    they don't, fit the whole frame rather than cutting someone out.
    """
    observations: list[FaceObservation] = [
        o for track in visible for o in track.observations_between(start_s, end_s)
    ]
    if not observations:
        return _general_segment(
            start_s=start_s,
            end_s=end_s,
            source_w=source_w,
            source_h=source_h,
            crop_w=crop_w,
            crop_h=crop_h,
        )

    left = min(o.left for o in observations)
    right = max(o.right for o in observations)
    spread = right - left

    if spread > crop_w * FIT_IF_SPREAD_EXCEEDS:
        return CropSegment(
            start_s=start_s,
            end_s=end_s,
            width=source_w,
            height=source_h,
            keyframes=[CropKeyframe(t=start_s, x=0.0, y=0.0)],
            strategy=Strategy.WIDE,
            fit=True,
        )

    centre_x = (left + right) / 2
    eye_y = sum(o.eye_y for o in observations) / len(observations)

    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=[
            CropKeyframe(
                t=start_s,
                x=_clamp(centre_x - crop_w / 2, 0, source_w - crop_w),
                y=_clamp(eye_y - EYE_LINE_RATIO * crop_h, 0, source_h - crop_h),
            )
        ],
        strategy=Strategy.WIDE,
    )


def _general_segment(
    *,
    start_s: float,
    end_s: float,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
) -> CropSegment:
    """No usable face — centre crop."""
    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=crop_w,
        height=crop_h,
        keyframes=[CropKeyframe(t=start_s, x=(source_w - crop_w) / 2, y=(source_h - crop_h) / 2)],
        strategy=Strategy.GENERAL,
        zoom=GENERAL_ZOOM,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high)) if high > low else max(0.0, low)
