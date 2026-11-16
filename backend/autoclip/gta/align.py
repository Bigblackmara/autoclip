"""Find exactly where a downloaded clip section starts in stream time.

A section is requested from ``start`` but stream-copy downloads cut on the
nearest keyframe/fragment, so the file can begin a few seconds early. Instead
of trusting the request, the section's audio is matched against the analysis
audio (already on disk, in stream time): the lag with the best correlation of
the two loudness envelopes is the true offset, to 10 ms.
"""

from __future__ import annotations

import logging
import subprocess
import wave
from pathlib import Path

import numpy as np

from ..pipeline import ffmpeg

log = logging.getLogger(__name__)

RATE = 16000
HOP = 160  # 10 ms envelope resolution
#: How far from the requested start the true start may be searched.
SEARCH_S = 25.0
#: Correlation below this is not trusted; the requested offset is kept.
MIN_CORRELATION = 0.5


def _envelope(samples: np.ndarray) -> np.ndarray:
    count = len(samples) // HOP
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    blocks = np.abs(samples[: count * HOP].astype(np.float32)).reshape(count, HOP)
    env = np.log1p(blocks.mean(axis=1) * 50.0)
    return env.astype(np.float32)


def decode_audio(path: Path, *, seconds: float | None = None) -> np.ndarray:
    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
    ]
    if seconds is not None:
        command += ["-t", f"{seconds:.3f}"]
    command += ["-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]
    result = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    if result.returncode != 0:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(result.stdout, dtype=np.int16)


def read_reference(
    parts: list[Path], part_s: float, range_start_s: float, a: float, b: float
) -> np.ndarray:
    """Analysis audio between stream times ``a`` and ``b`` (16 kHz mono)."""
    out: list[np.ndarray] = []
    for index, part in enumerate(parts):
        p0 = range_start_s + index * part_s
        p1 = p0 + part_s
        lo, hi = max(a, p0), min(b, p1)
        if hi <= lo:
            continue
        with wave.open(str(part), "rb") as handle:
            frames = handle.getnframes()
            first = int((lo - p0) * RATE)
            last = min(frames, int((hi - p0) * RATE))
            if last <= first:
                continue
            handle.setpos(min(first, frames))
            out.append(np.frombuffer(handle.readframes(last - first), dtype=np.int16))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


def best_lag(reference: np.ndarray, probe: np.ndarray) -> tuple[int, float]:
    """(lag in envelope steps, normalised correlation) of ``probe`` in ``reference``."""
    if len(probe) < 50 or len(reference) < len(probe):
        return 0, 0.0
    probe = probe - probe.mean()
    norm_p = float(np.sqrt(np.sum(probe * probe)))
    if norm_p <= 1e-9:
        return 0, 0.0
    n = len(reference) + len(probe)
    size = 1 << int(np.ceil(np.log2(n)))
    corr = np.fft.irfft(np.fft.rfft(reference, size) * np.conj(np.fft.rfft(probe, size)), size)
    valid = len(reference) - len(probe) + 1
    corr = corr[:valid]
    # Normalise by the reference's local energy (sliding window).
    csum = np.concatenate([[0.0], np.cumsum(reference.astype(np.float64))])
    csq = np.concatenate([[0.0], np.cumsum(reference.astype(np.float64) ** 2)])
    m = len(probe)
    window_sum = csum[m : m + valid] - csum[:valid]
    window_sq = csq[m : m + valid] - csq[:valid]
    local_var = np.maximum(window_sq - window_sum**2 / m, 1e-9)
    score = corr / (np.sqrt(local_var) * norm_p)
    lag = int(np.argmax(score))
    return lag, float(score[lag])


def section_offset(
    section: Path,
    *,
    expected_start_s: float,
    parts: list[Path],
    part_s: float,
    range_start_s: float,
    probe_s: float = 30.0,
) -> tuple[float, float]:
    """(stream time where the section file starts, confidence 0..1).

    Falls back to ``expected_start_s`` (confidence 0) when the audio can't be
    matched — silence, a muted VOD stretch, or missing analysis parts.
    """
    sample = decode_audio(section, seconds=probe_s)
    if len(sample) < RATE * 3 or not parts:
        return expected_start_s, 0.0
    a = max(range_start_s, expected_start_s - SEARCH_S)
    b = expected_start_s + SEARCH_S + len(sample) / RATE
    reference = read_reference(parts, part_s, range_start_s, a, b)
    lag, score = best_lag(_envelope(reference), _envelope(sample))
    if score < MIN_CORRELATION:
        log.info(
            "Section %s: audio match too weak (%.2f); using the requested start.",
            section.name,
            score,
        )
        return expected_start_s, score
    return a + lag * HOP / RATE, score
