"""Prepare stage — normalise media into what later stages expect.

Produces a 16 kHz mono WAV (what Whisper wants), a thumbnail strip for UI
scrubbing, a silence map used later to place clip boundaries in audio
troughs rather than mid-breath, and a loudness profile that lets a clip's end
run on through the reaction (the laugh, the scream) instead of cutting just
before it.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ffmpeg

log = logging.getLogger(__name__)

#: Whisper resamples to 16 kHz mono internally; doing it once up front avoids
#: repeating the work on every model invocation.
AUDIO_SAMPLE_RATE = 16_000

THUMBNAIL_INTERVAL_S = 5
THUMBNAIL_WIDTH = 160


@dataclass
class Silence:
    start: float
    end: float

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    @property
    def duration(self) -> float:
        return self.end - self.start


#: Loudness values at or below this are treated as digital silence.
LOUDNESS_FLOOR_LUFS = -120.0


@dataclass
class Loudness:
    """Momentary loudness (EBU R128, 400 ms window) sampled every ``step_s``.

    ebur128 emits one value per ``step_s`` frame; ``values[i]`` covers the
    400 ms window that ends with frame ``i``, so its centre sits at
    ``(i + 1) * step_s - WINDOW_S / 2``. :meth:`frames` reports those centres,
    so callers can reason about "how loud is the audio around time t".
    """

    values: list[float]
    step_s: float = 0.1

    WINDOW_S = 0.4

    def __bool__(self) -> bool:
        return bool(self.values)

    def _centre(self, index: int) -> float:
        return (index + 1) * self.step_s - self.WINDOW_S / 2

    def frames(self, start: float, end: float) -> list[tuple[float, float]]:
        """``(centre_time, loudness)`` for every window centred in ``[start, end]``."""
        if not self.values or end < start:
            return []
        offset = self.WINDOW_S / 2 - self.step_s
        first = max(0, math.ceil((start + offset) / self.step_s - 1e-9))
        last = min(len(self.values) - 1, math.floor((end + offset) / self.step_s + 1e-9))
        return [(self._centre(i), self.values[i]) for i in range(first, last + 1)]

    def to_dict(self) -> dict[str, object]:
        return {"step_s": self.step_s, "values": [round(v, 1) for v in self.values]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Loudness:
        return cls(
            values=[float(v) for v in data.get("values") or []],
            step_s=float(data.get("step_s") or 0.1),
        )


#: Look far enough into a file to find an audio track that starts late
#: (``analyzeduration`` is in microseconds: 100 s).
DEEP_PROBE = ("-analyzeduration", "100M", "-probesize", "100M")


def audio_lead_s(source: Path) -> float:
    """Seconds the first audio starts after the file's start (0 if unknown)."""
    try:
        result = subprocess.run(  # noqa: S603
            [
                ffmpeg.ffprobe_path(),
                "-v",
                "error",
                *DEEP_PROBE,
                "-show_entries",
                "format=start_time:stream=codec_type,start_time",
                "-of",
                "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        import json

        data = json.loads(result.stdout or "{}")
        file_start = float(data.get("format", {}).get("start_time"))
        audio = [
            float(s["start_time"])
            for s in data.get("streams", [])
            if s.get("codec_type") == "audio" and s.get("start_time") not in (None, "N/A")
        ]
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return 0.0
    if not audio:
        return 0.0
    return max(0.0, min(audio) - file_start)


def extract_audio(
    source: Path,
    destination: Path,
    *,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Extract mono 16 kHz PCM WAV from any input.

    ffmpeg writes to a sibling ``.partial.wav`` that is renamed into place
    only on success. The runner reuses any non-empty ``audio.wav`` on a
    retry or after a restart, and a WAV cut short by a crash, a cancel, or a
    full disk is still a valid, non-empty file — so the job silently went on
    to transcribe and clip only the first part of the video.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".partial.wav")
    lead = audio_lead_s(source)
    # Word times are read against the file's start: an audio track that
    # starts late (live capture after a reconnect) is padded with silence so
    # second N of the WAV is second N of the video.
    pad = ["-af", f"adelay={round(lead * 1000)}:all=1"] if lead > 0.02 else []
    try:
        ffmpeg.run(
            [
                *DEEP_PROBE,
                "-i",
                str(source),
                "-vn",
                *pad,
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(partial),
            ],
            total_duration_s=duration_s,
            on_progress=on_progress,
            cancelled=cancelled,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, destination)
    return destination


def generate_thumbnails(
    source: Path,
    destination_dir: Path,
    *,
    interval_s: int = THUMBNAIL_INTERVAL_S,
    width: int = THUMBNAIL_WIDTH,
) -> list[Path]:
    """Write one thumbnail every ``interval_s`` seconds for UI scrubbing.

    Returns the generated files in chronological order. An audio-only source
    yields an empty list rather than an error.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    pattern = destination_dir / "thumb_%05d.jpg"

    try:
        ffmpeg.run(
            [
                "-i",
                str(source),
                "-vf",
                f"fps=1/{interval_s},scale={width}:-2",
                "-q:v",
                "5",
                str(pattern),
            ]
        )
    except ffmpeg.FFmpegError:
        # Audio-only input has no video stream to sample.
        log.debug("No thumbnails generated for %s (likely audio-only).", source.name)
        return []

    return sorted(destination_dir.glob("thumb_*.jpg"))


_SILENCE_START = re.compile(r"silence_start:\s*([-\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([-\d.]+)")


def detect_silences(
    audio: Path,
    *,
    noise_db: float = -32.0,
    min_duration_s: float = 0.20,
) -> list[Silence]:
    """Find silent spans using ffmpeg's ``silencedetect``.

    Clip boundaries land far better inside these than at a fixed offset from a
    sentence end: a fixed pad clips breaths and plosives, while a trough is
    where a human editor would cut.

    Preconditions:
        audio is a decodable audio file; video inputs work but waste decode time.
    """
    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-i",
        str(audio),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_duration_s}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        command, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        log.warning("silencedetect failed; boundary refinement will fall back to fixed padding.")
        return []

    silences: list[Silence] = []
    pending_start: float | None = None
    for line in (proc.stderr or "").splitlines():
        if match := _SILENCE_START.search(line):
            pending_start = float(match.group(1))
        elif match := _SILENCE_END.search(line):
            end = float(match.group(1))
            start = pending_start if pending_start is not None else end
            silences.append(Silence(start=max(0.0, start), end=end))
            pending_start = None

    return silences


_LOUDNESS_M = re.compile(r"lavfi\.r128\.M=(\S+)")


def measure_loudness(audio: Path) -> Loudness:
    """Measure momentary loudness every 100 ms with ffmpeg's ``ebur128``.

    Streamers' reactions are loud: a cut placed right before a loudness spike
    is the cut that loses the laugh. The boundary pass uses this profile to
    carry a clip's end through the reaction. An empty profile (ffmpeg missing
    or failing) simply disables that step.
    """
    try:
        command = [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-nostdin",
            "-i",
            str(audio),
            "-af",
            "ebur128=metadata=1,ametadata=mode=print:key=lavfi.r128.M",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(
            command, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace"
        )
    except (OSError, ffmpeg.FFmpegError) as exc:
        log.warning("Loudness measurement unavailable (%s); clip ends won't follow reactions.", exc)
        return Loudness(values=[])
    if proc.returncode != 0:
        log.warning("ebur128 failed; clip ends won't follow reactions.")
        return Loudness(values=[])
    return parse_loudness(proc.stderr or "")


def parse_loudness(log_text: str) -> Loudness:
    """Parse ``ametadata`` print output into a 100 ms loudness profile."""
    values: list[float] = []
    for match in _LOUDNESS_M.finditer(log_text):
        try:
            value = float(match.group(1))
        except ValueError:
            value = LOUDNESS_FLOOR_LUFS
        if value != value or value < LOUDNESS_FLOOR_LUFS:  # nan or -inf
            value = LOUDNESS_FLOOR_LUFS
        values.append(value)
    return Loudness(values=values, step_s=0.1)
