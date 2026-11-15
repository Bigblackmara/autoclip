"""Facecam overlay detection for gameplay streams.

A streamer's facecam is a fixed rectangle composited over the gameplay. Two
things give it away, and both survive across a clip while the game moves:

1. A *small* face that stays put. Face detection runs on overlapping tiles,
   because a 60 px webcam face in a 1080p frame is invisible to the landmark
   model's whole-frame pass. Faces are clustered by position; a cluster seen in
   most samples is a facecam candidate. A large face (a talking-head video) or
   a face that wanders (a game character) is not.
2. The overlay's border: a closed rectangle around every observed face
   position whose sides are lines of edges that persist across the samples
   *more than the lines beside them* (so busy static textures — a lobby, a
   bookshelf — cancel out). Rectangles are scored on their weakest side, the
   difference in motion across the sides (a static room against moving
   gameplay), a penalty for sides that run on past the corners (a HUD bar the
   rect happens to share) and one for straying from webcam aspects (16:9,
   4:3). Sides flush with the frame edge need no line.

When no closed rectangle is found the rect is estimated from the face boxes —
always containing every observed face position, never covering the frame.

With several cams (a squad stream) the streamer's is taken to be the docked,
off-centre one that is biggest and most consistently present. Detection is
per clip: a long stream changes layout (lobby, match, a full-screen "just
chatting" stretch) and the cam moves with it.

Everything returns ``None`` rather than guessing when the signal is weak: the
caller then keeps the ordinary speaker-following crop, which is the right
answer for every video that is not a gameplay stream.

All of it runs on the CPU (ffmpeg decode, MediaPipe's CPU delegate, numpy).
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: A facecam face is small: wider than this fraction of the frame and it is a
#: talking head (or a full-screen "just chatting" scene), not an overlay.
MAX_FACE_WIDTH_RATIO = 0.08

#: Fraction of sampled frames a face cluster must appear in to count as stable.
#: Webcam faces drop out when the streamer turns away, so this is well below 1.
MIN_PRESENCE = 0.4

#: Face centres within this many face widths of a cluster join it. Streamers
#: move around inside their cam; a game character crosses the whole frame.
CLUSTER_DISTANCE_FACTOR = 2.0

#: A facecam overlay never covers more than this fraction of either dimension.
MAX_CAM_SPAN = 0.6

#: Nor more than this fraction of the frame's area.
MAX_CAM_AREA = 0.36

#: Width analysed for the border search. Plenty for locating straight lines,
#: and it keeps the per-pixel statistics cheap.
ANALYSIS_WIDTH = 480

#: Grey-level step that counts as an edge between two pixels.
EDGE_THRESHOLD = 10.0

#: Minimum mean prominence along a side for it to be a border: how much more
#: often an edge is on the line than on the parallel lines either side.
MIN_SIDE_PERSISTENCE = 0.3

#: Distance, in analysis pixels, of the lines a candidate is compared against.
PROMINENCE_OFFSET = 4

#: Upper bound on the cam's size in face widths. Streamers sit a little back
#: from the camera; past this, the "cam" is a scene with a face in it.
MAX_CAM_FACE_WIDTHS = 12.0
MAX_CAM_FACE_HEIGHTS = 9.0

#: Minimum prominence over the face's span for a line to be a candidate side.
MIN_CANDIDATE_PROMINENCE = 0.15

#: Candidate lines kept per side for the rectangle search.
CANDIDATES_PER_SIDE = 24

#: Weight of the motion contrast across a side, next to its edge persistence.
CONTRAST_WEIGHT = 0.5

#: Width in analysis pixels of the bands compared across a candidate side.
BAND = 3

#: How far past a corner (analysis pixels) a side is checked for running on,
#: and how much a side that does is penalised.
EXTENSION = 8
OVERRUN_WEIGHT = 0.25

#: Penalty per unit of log-distance from the nearest webcam aspect (16:9, 4:3).
ASPECT_WEIGHT = 0.5

#: Rectangles scoring within this of the best compete on size; the largest wins.
NEAR_BEST_SCORE = 0.01

#: Samples per clip: about one every two seconds, within these bounds.
MIN_SAMPLES = 12
MAX_SAMPLES = 30

#: A cam this close to a frame edge (fraction of the frame) counts as docked.
EDGE_DOCK = 0.03

CACHE_VERSION = 1


@dataclass(frozen=True)
class FaceBox:
    """One detected face in source-pixel coordinates."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def cx(self) -> float:
        return self.left + self.width / 2

    @property
    def cy(self) -> float:
        return self.top + self.height / 2


@dataclass
class Facecam:
    """A detected facecam overlay. Rect and face box are in source pixels."""

    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int
    #: Median face box inside the cam.
    face_left: float
    face_top: float
    face_width: float
    face_height: float
    #: Fraction of sampled frames the face was found in.
    presence: float
    #: "border" when the overlay's edges were found, "estimate" when the rect
    #: was derived from the face boxes alone.
    method: str = "border"

    @property
    def face_width_ratio(self) -> float:
        return self.face_width / self.source_width if self.source_width else 1.0

    @property
    def area_ratio(self) -> float:
        frame = self.source_width * self.source_height
        return (self.width * self.height) / frame if frame else 1.0

    @property
    def well_placed(self) -> bool:
        """Docked against a frame edge and off-centre, where overlays go."""
        sw, sh = self.source_width, self.source_height
        if not sw or not sh:
            return False
        docked = (
            self.x <= sw * EDGE_DOCK
            or self.y <= sh * EDGE_DOCK
            or self.x + self.width >= sw * (1 - EDGE_DOCK)
            or self.y + self.height >= sh * (1 - EDGE_DOCK)
        )
        centre = (self.x + self.width / 2) / sw
        return docked and not 0.35 < centre < 0.65

    @property
    def is_overlay(self) -> bool:
        """True when this looks like a facecam overlay rather than a talking head.

        What the "auto" layout keys on: a small face, a cam well short of the
        whole frame, docked at the side of the frame.
        """
        return (
            self.face_width_ratio <= MAX_FACE_WIDTH_RATIO
            and self.area_ratio <= MAX_CAM_AREA
            and self.well_placed
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Facecam:
        return cls(**{name: data[name] for name in cls.__dataclass_fields__ if name in data})


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def save_detection(path: Path, facecam: Facecam | None) -> Path:
    """Cache a detection result, including "no facecam" — that took work too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "facecam": facecam.to_dict() if facecam else None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_detection(path: Path) -> tuple[bool, Facecam | None]:
    """Return ``(cached, facecam)``; ``cached`` is False for a missing or stale file."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return False, None
    data = raw.get("facecam")
    if data is None:
        return True, None
    try:
        return True, Facecam.from_dict(data)
    except (KeyError, TypeError):
        return False, None


# --------------------------------------------------------------------------
# Detection entry point
# --------------------------------------------------------------------------

#: ``(rgb_frame) -> face boxes`` in that frame's pixel coordinates.
FaceDetector = Callable[[Any], list[FaceBox]]
#: ``(video, times) -> rgb frames`` (H x W x 3 uint8 arrays), one per time.
FrameReader = Callable[[Path, Sequence[float]], list[Any]]


def sample_times(start_s: float, end_s: float, count: int | None = None) -> list[float]:
    """Evenly spread sample times inside ``[start_s, end_s)``."""
    duration = max(0.0, end_s - start_s)
    if count is None:
        count = int(min(MAX_SAMPLES, max(MIN_SAMPLES, round(duration / 2))))
    if duration <= 0 or count <= 0:
        return [start_s]
    return [start_s + duration * (index + 0.5) / count for index in range(count)]


def detect_facecam(
    video: Path,
    *,
    start_s: float,
    end_s: float,
    source_width: int,
    source_height: int,
    samples: int | None = None,
    detector: FaceDetector | None = None,
    reader: FrameReader | None = None,
) -> Facecam | None:
    """Find a facecam overlay in ``video`` between ``start_s`` and ``end_s``.

    Returns None when there is no stable small face — the normal result for
    anything that is not a gameplay stream.
    """
    times = sample_times(start_s, end_s, samples)
    reader = reader or read_frames
    frames = reader(video, times)
    if not frames:
        return None

    owned_detector = None
    if detector is None:
        owned_detector = TiledFaceDetector()
        detector = owned_detector
    try:
        detections = [detector(frame) for frame in frames]
    finally:
        if owned_detector is not None:
            owned_detector.close()

    grey = _analysis_frames(frames)
    return locate_facecam(detections, grey, source_width, source_height)


def locate_facecam(
    detections: Sequence[Sequence[FaceBox]],
    grey_frames: Any | None,
    source_width: int,
    source_height: int,
) -> Facecam | None:
    """The pure half of detection: faces per frame (+ optional frames) to a rect.

    ``grey_frames`` is an ``(N, h, w)`` float array of the sampled frames at
    analysis resolution, or None to skip the border search.
    """
    clusters = stable_face_clusters(detections, source_width, source_height)
    if not clusters:
        return None

    candidates: list[Facecam] = []
    for cluster in clusters:
        rect, method = _cam_rect(cluster.faces, grey_frames, source_width, source_height)
        if rect is None:
            continue
        x, y, w, h = rect
        median = cluster.median_face
        candidates.append(
            Facecam(
                x=x,
                y=y,
                width=w,
                height=h,
                source_width=source_width,
                source_height=source_height,
                face_left=median.left,
                face_top=median.top,
                face_width=median.width,
                face_height=median.height,
                presence=round(cluster.presence, 3),
                method=method,
            )
        )

    if not candidates:
        return None

    # Overlays dock against a frame edge, away from the centre where the game
    # (and its characters' faces) lives. Among several cams — a squad stream —
    # the streamer's own is by convention the biggest and always on screen;
    # a found border counts for a little more than an estimate.
    chosen = max(
        candidates,
        key=lambda c: (
            c.well_placed,
            c.width * c.height * c.presence * (1.0 if c.method == "border" else 0.9),
        ),
    )
    log.info(
        "Facecam at %dx%d+%d+%d (%s, face in %.0f%% of samples).",
        chosen.width,
        chosen.height,
        chosen.x,
        chosen.y,
        chosen.method,
        chosen.presence * 100,
    )
    return chosen


# --------------------------------------------------------------------------
# Stable faces
# --------------------------------------------------------------------------


@dataclass
class FaceCluster:
    faces: list[FaceBox]
    #: Number of distinct frames the cluster appears in, over frames sampled.
    presence: float

    @property
    def median_face(self) -> FaceBox:
        return FaceBox(
            left=statistics.median(f.left for f in self.faces),
            top=statistics.median(f.top for f in self.faces),
            width=statistics.median(f.width for f in self.faces),
            height=statistics.median(f.height for f in self.faces),
        )


def stable_face_clusters(
    detections: Sequence[Sequence[FaceBox]], source_width: int, source_height: int
) -> list[FaceCluster]:
    """Small faces that hold their position across most samples.

    Greedy clustering by centre distance, scaled by face size. Each cluster
    counts at most one face per frame towards its presence.
    """
    frame_count = len(detections)
    if frame_count == 0:
        return []

    max_face_width = source_width * MAX_FACE_WIDTH_RATIO
    groups: list[dict[str, Any]] = []
    for index, faces in enumerate(detections):
        for face in faces:
            if face.width <= 1 or face.width > max_face_width:
                continue
            best = None
            best_distance = float("inf")
            for group in groups:
                ref = group["ref"]
                limit = CLUSTER_DISTANCE_FACTOR * max(ref.width, face.width)
                distance = ((face.cx - ref.cx) ** 2 + (face.cy - ref.cy) ** 2) ** 0.5
                size_ratio = face.width / ref.width if ref.width else 0.0
                if distance <= limit and 0.5 <= size_ratio <= 2.0 and distance < best_distance:
                    best, best_distance = group, distance
            if best is None:
                groups.append({"ref": face, "faces": [face], "frames": {index}})
            else:
                best["faces"].append(face)
                best["frames"].add(index)

    clusters = [
        FaceCluster(faces=g["faces"], presence=len(g["frames"]) / frame_count) for g in groups
    ]
    stable = [c for c in clusters if c.presence >= MIN_PRESENCE]
    stable.sort(key=lambda c: c.presence, reverse=True)
    return stable


# --------------------------------------------------------------------------
# Rectangle estimation
# --------------------------------------------------------------------------


def _cam_rect(
    faces: list[FaceBox],
    grey_frames: Any | None,
    source_width: int,
    source_height: int,
) -> tuple[tuple[int, int, int, int] | None, str]:
    union = (
        min(f.left for f in faces),
        min(f.top for f in faces),
        max(f.right for f in faces),
        max(f.bottom for f in faces),
    )
    face_width = statistics.median(f.width for f in faces)

    if grey_frames is not None and len(grey_frames) >= 2:
        found = _border_rect(union, face_width, grey_frames, source_width, source_height)
        if found is not None:
            return found, "border"

    return _estimated_rect(union, face_width, source_width, source_height), "estimate"


def _estimated_rect(
    union: tuple[float, float, float, float],
    face_width: float,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int] | None:
    """A 16:9 box around the face positions when no border was found.

    Sized to hold every observed face position with room for shoulders, which
    is what a webcam frame shows; clamped into the frame.
    """
    left, top, right, bottom = union
    width = max((right - left) + 2.0 * face_width, 4.0 * face_width)
    height = max((bottom - top) + 1.5 * face_width, width * 9 / 16)
    width = max(width, height * 16 / 9)
    width = min(width, source_width * MAX_CAM_SPAN)
    height = min(height, source_height * MAX_CAM_SPAN)

    cx = (left + right) / 2
    # Faces sit in the upper half of a webcam frame.
    cy = top + (bottom - top) / 2 + height * 0.12
    x = _clamp(cx - width / 2, 0, source_width - width)
    y = _clamp(cy - height / 2, 0, source_height - height)
    return _even_rect(x, y, width, height, source_width, source_height)


def _border_rect(
    union: tuple[float, float, float, float],
    face_width: float,
    grey_frames: Any,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int] | None:
    """Find the overlay's border as a closed rectangle of persistent edges."""
    import numpy as np

    frames = np.asarray(grey_frames, dtype=np.float32)
    _, h, w = frames.shape
    # How much each pixel changes across the samples: the overlay and the
    # gameplay behind it change independently, so the border separates two
    # different levels of motion, where a line inside the streamer's room
    # does not.
    motion = frames.std(axis=0)
    sx = source_width / w
    sy = source_height / h

    # Fraction of frames with an edge *at* each pixel boundary (a grey step
    # between neighbours), for vertical and for horizontal lines.
    vertical = np.zeros((h, w + 1), dtype=np.float32)
    vertical[:, 1:w] = (np.abs(frames[:, :, 1:] - frames[:, :, :-1]) >= EDGE_THRESHOLD).mean(0)
    horizontal = np.zeros((h + 1, w), dtype=np.float32)
    horizontal[1:h, :] = (np.abs(frames[:, 1:, :] - frames[:, :-1, :]) >= EDGE_THRESHOLD).mean(0)
    # Raw persistence pins a found side to its exact pixel; prominence finds it.
    vertical_raw = np.zeros((h + 1, w + 1), dtype=np.float64)
    vertical_raw[1:, :] = np.cumsum(vertical, axis=0)
    horizontal_raw = np.zeros((h + 1, w + 1), dtype=np.float64)
    horizontal_raw[:, 1:] = np.cumsum(horizontal, axis=1)
    vertical = _line_prominence(vertical, axis=1)
    horizontal = _line_prominence(horizontal, axis=0)

    ul = int(np.floor(union[0] / sx))
    ut = int(np.floor(union[1] / sy))
    ur = int(np.ceil(union[2] / sx))
    ub = int(np.ceil(union[3] / sy))
    ul, ur = max(0, ul), min(w, ur)
    ut, ub = max(0, ut), min(h, ub)
    if ur <= ul or ub <= ut:
        return None

    pad = max(1, int(round(0.25 * face_width / sx)))
    # A webcam frames a person, not a landscape: bound the cam by the face too.
    max_w = int(min(w * MAX_CAM_SPAN, MAX_CAM_FACE_WIDTHS * face_width / sx))
    max_h = int(min(h * MAX_CAM_SPAN, MAX_CAM_FACE_HEIGHTS * face_width / sy))

    # Prefix sums make any side's mean an O(1) lookup in the rectangle search.
    vertical_sum = np.zeros((h + 1, w + 1), dtype=np.float64)
    vertical_sum[1:, :] = np.cumsum(vertical, axis=0)
    horizontal_sum = np.zeros((h + 1, w + 1), dtype=np.float64)
    horizontal_sum[:, 1:] = np.cumsum(horizontal, axis=1)

    def column_score(x: int, y0: int, y1: int) -> float:
        if x <= 0 or x >= w:
            return 1.0
        y0, y1 = max(0, y0), min(h, y1)
        if y1 <= y0:
            return 0.0
        # Best of the line and its neighbours: a candidate picked over the
        # face's span can sit a pixel off the line along the full side.
        return max(
            float(vertical_sum[y1, c] - vertical_sum[y0, c]) / (y1 - y0)
            for c in (x - 1, x, x + 1)
            if 0 < c < w
        )

    def column_contrast(x: int, y0: int, y1: int, outward: int) -> float:
        inside = motion[y0:y1, max(0, x - BAND) : x] if outward > 0 else motion[y0:y1, x : x + BAND]
        outside = (
            motion[y0:y1, x : x + BAND] if outward > 0 else motion[y0:y1, max(0, x - BAND) : x]
        )
        return _contrast(inside, outside)

    def row_contrast(y: int, x0: int, x1: int, outward: int) -> float:
        inside = motion[max(0, y - BAND) : y, x0:x1] if outward > 0 else motion[y : y + BAND, x0:x1]
        outside = (
            motion[y : y + BAND, x0:x1] if outward > 0 else motion[max(0, y - BAND) : y, x0:x1]
        )
        return _contrast(inside, outside)

    def row_score(y: int, x0: int, x1: int) -> float:
        if y <= 0 or y >= h:
            return 1.0
        x0, x1 = max(0, x0), min(w, x1)
        if x1 <= x0:
            return 0.0
        return max(
            float(horizontal_sum[r, x1] - horizontal_sum[r, x0]) / (x1 - x0)
            for r in (y - 1, y, y + 1)
            if 0 < r < h
        )

    def candidates(
        positions: Iterable[int], score: Callable[[int], float], edge: int | None
    ) -> list[int]:
        ordered = [p for p in positions if p != edge]
        scores = {p: score(p) for p in ordered}
        # Local peaks only: a border blurred over two pixels is one line.
        # Scored over the face's span only, where a HUD element against the
        # cam can mask part of the border, so the bar is lower here than for
        # the finished rectangle's full sides.
        peaks = [
            p
            for p in ordered
            if scores[p] >= MIN_CANDIDATE_PROMINENCE
            and all(scores[p] >= scores.get(p + d, -1.0) for d in (-2, -1, 1, 2))
        ]
        picked: list[int] = []
        for p in sorted(peaks, key=lambda p: scores[p], reverse=True):
            if all(abs(p - q) > 2 for q in picked):
                picked.append(p)
            if len(picked) >= CANDIDATES_PER_SIDE:
                break
        # The frame edge is always a candidate when a cam reaching it would
        # still be small enough — even when a face sits right against it.
        if edge is not None:
            picked.append(edge)
        return picked

    lefts = candidates(
        range(max(1, ur - max_w), ul - pad + 1),
        lambda x: column_score(x, ut, ub),
        0 if ur <= max_w else None,
    )
    rights = candidates(
        range(ur + pad, min(w - 1, ul + max_w) + 1),
        lambda x: column_score(x, ut, ub),
        w if w - ul <= max_w else None,
    )
    tops = candidates(
        range(max(1, ub - max_h), ut - pad + 1),
        lambda y: row_score(y, ul, ur),
        0 if ub <= max_h else None,
    )
    bottoms = candidates(
        range(ub + pad, min(h - 1, ut + max_h) + 1),
        lambda y: row_score(y, ul, ur),
        h if h - ut <= max_h else None,
    )

    min_cam_w = face_width / sx / 0.6
    valid: list[tuple[float, int, tuple[int, int, int, int]]] = []
    for left in lefts:
        for right in rights:
            width = right - left
            if width > max_w or width < min_cam_w:
                continue
            for top in tops:
                for bottom in bottoms:
                    height = bottom - top
                    if height > max_h or height <= 0:
                        continue
                    if (width * height) > MAX_CAM_AREA * w * h:
                        continue
                    if not 0.5 <= width / height <= 2.6:
                        continue
                    sides = [
                        column_score(left, top, bottom),
                        column_score(right, top, bottom),
                        row_score(top, left, right),
                        row_score(bottom, left, right),
                    ]
                    detected = [
                        s
                        for s, pos, limit in zip(
                            sides, (left, right, top, bottom), (w, w, h, h), strict=True
                        )
                        if 0 < pos < limit
                    ]
                    if len(detected) < 2:
                        # A rect made of frame edges is the frame, not a cam.
                        continue
                    persistence = min(detected)
                    if persistence < MIN_SIDE_PERSISTENCE:
                        continue
                    contrasts = [
                        c
                        for c, pos, limit in zip(
                            (
                                column_contrast(left, top, bottom, -1),
                                column_contrast(right, top, bottom, 1),
                                row_contrast(top, left, right, -1),
                                row_contrast(bottom, left, right, 1),
                            ),
                            (left, right, top, bottom),
                            (w, w, h, h),
                            strict=True,
                        )
                        if 0 < pos < limit
                    ]
                    # A border stops at the cam's corners. A side whose line
                    # runs on past them is a scene line the rect happens to
                    # share (lobby UI, a HUD bar), not the overlay's edge.
                    reach = EXTENSION
                    overruns = [
                        o
                        for o, pos, limit in zip(
                            (
                                max(
                                    column_score(left, top - reach, top),
                                    column_score(left, bottom, bottom + reach),
                                ),
                                max(
                                    column_score(right, top - reach, top),
                                    column_score(right, bottom, bottom + reach),
                                ),
                                max(
                                    row_score(top, left - reach, left),
                                    row_score(top, right, right + reach),
                                ),
                                max(
                                    row_score(bottom, left - reach, left),
                                    row_score(bottom, right, right + reach),
                                ),
                            ),
                            (left, right, top, bottom),
                            (w, w, h, h),
                            strict=True,
                        )
                        if 0 < pos < limit
                    ]
                    overrun = max(0.0, sum(overruns) / len(overruns))
                    # Webcams are 16:9 or 4:3, and overlays mostly keep that.
                    aspect = width * sx / (height * sy)
                    off_aspect = min(abs(math.log(aspect / 1.78)), abs(math.log(aspect / 1.33)))
                    score = (
                        persistence
                        + CONTRAST_WEIGHT * sum(contrasts) / len(contrasts)
                        - OVERRUN_WEIGHT * overrun
                        - ASPECT_WEIGHT * off_aspect
                    )
                    valid.append((score, width * height, (left, top, right, bottom)))

    if not valid:
        return None

    # Lines inside the cam (the streamer's room) also close rectangles, but
    # always smaller ones: among near-best scores, the largest is the overlay.
    best_score = max(item[0] for item in valid)
    near_best = [item for item in valid if item[0] >= best_score - NEAR_BEST_SCORE]
    _, _, (left, top, right, bottom) = max(near_best, key=lambda item: item[1])

    # Candidates were found with slack (a blurred line, peaks spaced apart);
    # settle each found side on its exact line before stepping inside it.
    def settle_column(x: int, y0: int, y1: int) -> int:
        if x <= 0 or x >= w:
            return x
        options = [c for c in range(x - 2, x + 3) if 0 < c < w]
        return max(options, key=lambda c: vertical_raw[y1, c] - vertical_raw[y0, c])

    def settle_row(y: int, x0: int, x1: int) -> int:
        if y <= 0 or y >= h:
            return y
        options = [r for r in range(y - 2, y + 3) if 0 < r < h]
        return max(options, key=lambda r: horizontal_raw[r, x1] - horizontal_raw[r, x0])

    left, right = settle_column(left, top, bottom), settle_column(right, top, bottom)
    top, bottom = settle_row(top, left, right), settle_row(bottom, left, right)

    # Step one analysis pixel inside detected borders so no gameplay or
    # border line bleeds into the facecam panel.
    inset_l = 1 if left > 0 else 0
    inset_r = 1 if right < w else 0
    inset_t = 1 if top > 0 else 0
    inset_b = 1 if bottom < h else 0
    x0 = (left + inset_l) * sx
    x1 = (right - inset_r) * sx
    y0 = (top + inset_t) * sy
    y1 = (bottom - inset_b) * sy
    return _even_rect(x0, y0, x1 - x0, y1 - y0, source_width, source_height)


def _line_prominence(persistence: Any, *, axis: int) -> Any:
    """How much more often an edge sits on a line than on lines either side.

    A border is a thin line of persistent edges; a busy static texture (a
    lobby scene, a bookshelf) is persistent everywhere. Subtracting the
    neighbouring lines' persistence keeps the first and cancels the second.
    The line itself takes the max over +-1 px, tolerating a border blurred
    across the downscale.
    """
    import numpy as np

    padded = np.pad(persistence, [(1, 1) if a == axis else (0, 0) for a in range(2)], mode="edge")
    count = persistence.shape[axis]
    line = np.maximum.reduce(
        [np.take(padded, range(offset, offset + count), axis=axis) for offset in (0, 1, 2)]
    )
    far = PROMINENCE_OFFSET
    padded = np.pad(
        persistence, [(far, far) if a == axis else (0, 0) for a in range(2)], mode="edge"
    )
    before = np.take(padded, range(0, count), axis=axis)
    after = np.take(padded, range(2 * far, 2 * far + count), axis=axis)
    return line - (before + after) / 2


def _contrast(inside: Any, outside: Any) -> float:
    if inside.size == 0 or outside.size == 0:
        return 0.0
    a, b = float(inside.mean()), float(outside.mean())
    return abs(a - b) / (a + b + 1.0)


def _even_rect(
    x: float, y: float, width: float, height: float, source_width: int, source_height: int
) -> tuple[int, int, int, int] | None:
    """Round to even pixels inside the frame (yuv420p crops need even sizes)."""
    ix = max(0, int(round(x)) & ~1)
    iy = max(0, int(round(y)) & ~1)
    iw = min(source_width - ix, int(round(width)))
    ih = min(source_height - iy, int(round(height)))
    iw -= iw % 2
    ih -= ih % 2
    if iw < 16 or ih < 16:
        return None
    return ix, iy, iw, ih


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high)) if high > low else max(0.0, low)


# --------------------------------------------------------------------------
# Frames and the default detector
# --------------------------------------------------------------------------


def _analysis_frames(frames: Sequence[Any]) -> Any | None:
    """Downscaled greyscale stack ``(N, h, w)`` for the border search."""
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - both are hard dependencies
        return None

    grey = []
    for frame in frames:
        height, width = frame.shape[:2]
        target_w = min(ANALYSIS_WIDTH, width)
        target_h = max(1, round(height * target_w / width))
        small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        grey.append(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32))
    if not grey or any(g.shape != grey[0].shape for g in grey):
        return None
    return np.stack(grey)


def read_frames(video: Path, times: Sequence[float]) -> list[Any]:
    """Decode one RGB frame per time with ffmpeg (CPU decode, input seeking).

    ffmpeg rather than OpenCV: the bundled OpenCV build can't decode every
    codec ffmpeg can (AV1 streams in particular), and an input-side seek per
    sample is far cheaper than decoding a long clip end to end.
    """
    import numpy as np

    from .. import ffmpeg

    info = ffmpeg.probe(video)
    if not info.has_video or not info.width or not info.height:
        return []
    width, height = info.width, info.height
    frame_bytes = width * height * 3

    frames = []
    for t in times:
        command = [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, t):.3f}",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:",
        ]
        try:
            result = subprocess.run(command, capture_output=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("Could not read a frame at %.2fs from %s: %s", t, video.name, exc)
            continue
        if len(result.stdout) < frame_bytes:
            continue
        frame = np.frombuffer(result.stdout[:frame_bytes], dtype=np.uint8)
        frames.append(frame.reshape(height, width, 3))
    return frames


class TiledFaceDetector:
    """MediaPipe face landmarks run over overlapping tiles of the frame.

    The landmarker's detector sees a downscaled whole frame, where a webcam
    overlay's face is a few pixels across. Tiles about half the frame height
    bring it back into range. Runs on the CPU.
    """

    def __init__(self, *, max_faces: int = 4) -> None:
        from .faces import FaceDetectionUnavailable

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
        except ImportError as exc:
            raise FaceDetectionUnavailable(f"MediaPipe is unavailable: {exc}") from exc

        from ... import models

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models.ensure("face_landmarker"))),
            running_mode=RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __call__(self, frame: Any) -> list[FaceBox]:
        import numpy as np

        height, width = frame.shape[:2]
        tile = max(64, min(width, height, round(height / 2.25)))
        stride = max(1, tile * 2 // 3)
        xs = sorted({*range(0, max(1, width - tile + 1), stride), max(0, width - tile)})
        ys = sorted({*range(0, max(1, height - tile + 1), stride), max(0, height - tile)})

        boxes: list[FaceBox] = []
        for y in ys:
            for x in xs:
                crop = np.ascontiguousarray(frame[y : y + tile, x : x + tile])
                image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=crop)
                result = self._landmarker.detect(image)
                crop_h, crop_w = crop.shape[:2]
                for landmarks in result.face_landmarks or []:
                    lx = [lm.x * crop_w for lm in landmarks]
                    ly = [lm.y * crop_h for lm in landmarks]
                    left, right = min(lx), max(lx)
                    top, bottom = min(ly), max(ly)
                    # A face cut by an inner tile edge is seen whole by a
                    # neighbouring tile; only frame edges may truncate one.
                    if (
                        (left < 1 and x > 0)
                        or (top < 1 and y > 0)
                        or (right > crop_w - 1 and x + tile < width)
                        or (bottom > crop_h - 1 and y + tile < height)
                    ):
                        continue
                    boxes.append(
                        FaceBox(left=x + left, top=y + top, width=right - left, height=bottom - top)
                    )
        return _suppress_duplicates(boxes)


def _suppress_duplicates(boxes: list[FaceBox]) -> list[FaceBox]:
    """Merge the same face found in overlapping tiles (keep the larger box)."""
    kept: list[FaceBox] = []
    for box in sorted(boxes, key=lambda b: b.width * b.height, reverse=True):
        if all(_iou(box, other) < 0.3 for other in kept):
            kept.append(box)
    return kept


def _iou(a: FaceBox, b: FaceBox) -> float:
    overlap_w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    overlap_h = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    intersection = overlap_w * overlap_h
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0 else 0.0
