"""Quality checks every exported GTA clip must pass.

- length: under 60 s (and not trivially short),
- loudness: integrated loudness within tolerance of the target, true peak
  under the ceiling,
- black frames: none at all,
- face-cam: the streamer's face is actually visible in the clip.

Each check reports a plain result the UI can show; :func:`run_qa` never
raises for a failed check, only for a file that cannot be read at all.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..pipeline import ffmpeg

log = logging.getLogger(__name__)

#: Shorts/Reels/TikTok hard limit is 60 s; the clip must be strictly under.
MAX_LENGTH_S = 60.0
MIN_LENGTH_S = 5.0
#: Frames sampled for the face-cam check.
FACE_SAMPLES = 12

_I_RE = re.compile(r"\bI:\s*(-?\d+(?:\.\d+)?)\s*LUFS")
_PEAK_RE = re.compile(r"\bPeak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dBFS")
_BLACK_RE = re.compile(r"black_start:\s*(-?\d+(?:\.\d+)?)\s+black_end:\s*(-?\d+(?:\.\d+)?)")


@dataclass
class Check:
    ok: bool
    detail: str
    value: Any = None


@dataclass
class QaReport:
    length: Check
    loudness: Check
    black_frames: Check
    facecam: Check
    #: Black spans (output seconds) — used to auto-fix edge blacks.
    black_spans: list[tuple[float, float]] = field(default_factory=list)
    loudness_lufs: float | None = None

    @property
    def passed(self) -> bool:
        return all(c.ok for c in (self.length, self.loudness, self.black_frames, self.facecam))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "length": asdict(self.length),
            "loudness": asdict(self.loudness),
            "black_frames": asdict(self.black_frames),
            "facecam": asdict(self.facecam),
        }


def _run(command: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, check=False, timeout=timeout, errors="replace"
    )


def measure_loudness(path: Path) -> tuple[float | None, float | None]:
    """(integrated LUFS, true peak dBFS) of a file, via ebur128."""
    result = _run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-vn",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ]
    )
    text = result.stderr or ""
    # The summary is the LAST occurrence of each value.
    i_values = _I_RE.findall(text)
    peak_values = _PEAK_RE.findall(text)
    lufs = float(i_values[-1]) if i_values else None
    peak = None
    if peak_values:
        raw = peak_values[-1]
        peak = float("-inf") if raw == "-inf" else float(raw)
    return lufs, peak


def black_spans(path: Path) -> list[tuple[float, float]] | None:
    """Every black span in the output (one frame or longer). None on error."""
    result = _run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-an",
            "-vf",
            "blackdetect=d=0:pic_th=0.97:pix_th=0.10",
            "-f",
            "null",
            "-",
        ]
    )
    if result.returncode != 0:
        return None
    return [(float(a), float(b)) for a, b in _BLACK_RE.findall(result.stderr or "")]


FrameReader = Callable[[Path, Sequence[float]], list[Any]]
FaceCounter = Callable[[Any], int]


def face_ratio(
    section: Path,
    start_s: float,
    end_s: float,
    *,
    samples: int = FACE_SAMPLES,
    reader: FrameReader | None = None,
    counter: FaceCounter | None = None,
) -> float:
    """Share of sampled frames (section-local times) with a visible face."""
    from ..pipeline.reframe.facecam import TiledFaceDetector, read_frames

    span = max(0.0, end_s - start_s)
    if span <= 0:
        return 0.0
    times = [start_s + span * (i + 0.5) / samples for i in range(samples)]
    frames = (reader or read_frames)(section, times)
    if not frames:
        return 0.0
    owned = None
    if counter is None:
        owned = TiledFaceDetector()

        def counter(frame: Any) -> int:
            return len(owned(frame))

    try:
        hits = sum(1 for frame in frames if counter(frame) > 0)
    finally:
        if owned is not None:
            owned.close()
    return hits / len(times)


def run_qa(
    output: Path,
    *,
    section: Path,
    section_start_s: float,
    section_end_s: float,
    target_lufs: float,
    tolerance_lu: float,
    true_peak_db: float,
    face_min_ratio: float,
    face_ratio_fn: Callable[[Path, float, float], float] | None = None,
) -> QaReport:
    info = ffmpeg.probe(output)
    duration = float(info.duration_s or 0.0)
    if not info.has_video or not info.has_audio:
        length = Check(False, "the file is missing its video or audio track", duration)
    elif duration >= MAX_LENGTH_S:
        length = Check(False, f"{duration:.1f} s — must be under {MAX_LENGTH_S:.0f} s", duration)
    elif duration < MIN_LENGTH_S:
        length = Check(False, f"{duration:.1f} s — too short", duration)
    else:
        length = Check(True, f"{duration:.1f} s", duration)

    lufs, peak = measure_loudness(output)
    if lufs is None:
        loudness = Check(False, "could not measure loudness", None)
    elif abs(lufs - target_lufs) > tolerance_lu:
        loudness = Check(
            False, f"{lufs:.1f} LUFS (target {target_lufs:.0f} ±{tolerance_lu:g})", lufs
        )
    elif peak is not None and peak > true_peak_db + 0.5:
        loudness = Check(False, f"true peak {peak:.1f} dBFS (ceiling {true_peak_db:g})", lufs)
    else:
        loudness = Check(
            True,
            f"{lufs:.1f} LUFS, peak {peak if peak is not None else float('nan'):.1f} dBFS",
            lufs,
        )

    spans = black_spans(output)
    if spans is None:
        black = Check(False, "could not scan for black frames", None)
        spans = []
    elif spans:
        listed = ", ".join(f"{a:.2f}-{b:.2f}s" for a, b in spans[:4])
        black = Check(False, f"black frames at {listed}", len(spans))
    else:
        black = Check(True, "no black frames", 0)

    try:
        ratio = (face_ratio_fn or face_ratio)(section, section_start_s, section_end_s)
    except Exception as exc:  # detector unavailable / model missing
        log.warning("Face-cam check could not run: %s", exc)
        facecam = Check(False, f"face detector unavailable ({exc})", None)
    else:
        if ratio >= face_min_ratio:
            facecam = Check(True, f"face visible in {ratio:.0%} of sampled frames", round(ratio, 3))
        else:
            facecam = Check(
                False,
                f"face visible in only {ratio:.0%} of sampled frames (need {face_min_ratio:.0%})",
                round(ratio, 3),
            )

    return QaReport(
        length=length,
        loudness=loudness,
        black_frames=black,
        facecam=facecam,
        black_spans=spans,
        loudness_lufs=lufs,
    )


def fix_loudness(path: Path, measured_lufs: float, target_lufs: float, true_peak_db: float) -> None:
    """Re-mux the clip with a gain correction (video stream copied)."""
    gain = target_lufs - measured_lufs
    partial = path.with_name(path.stem + ".gain.mp4")
    ffmpeg.run(
        [
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-af",
            f"volume={gain:.2f}dB,alimiter=limit={10 ** (true_peak_db / 20):.4f}:level=disabled",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(partial),
        ]
    )
    partial.replace(path)


def edge_trims(
    spans: list[tuple[float, float]], duration_s: float, *, edge_s: float = 1.5
) -> tuple[float, float] | None:
    """(trim_start, trim_end) that removes black spans touching the edges.

    None when a black span sits in the middle (can't be fixed by trimming).
    """
    trim_start = 0.0
    trim_end = 0.0
    for a, b in spans:
        if a <= edge_s:
            trim_start = max(trim_start, b + 0.05)
        elif b >= duration_s - edge_s:
            trim_end = max(trim_end, duration_s - a + 0.05)
        else:
            return None
    return trim_start, trim_end
