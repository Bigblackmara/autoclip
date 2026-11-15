"""Audio energy spike detection for Live Watch.

The audio detector answers one question per second of stream audio: "is the
stream louder right now than it usually is?" Each committed buffer segment is
downmixed once by ffmpeg to mono 16 kHz PCM; numpy turns that into per-second
RMS levels; each level is z-scored against a rolling baseline of recent
seconds. Runs of consecutive spike seconds ("bursts") become very few
DetectorSignals — one at onset, updates only while the burst's peak z grows
materially — so a long hype section flags its edges without spamming the
detector (A4).

Why RMS only — no FFT, laughter, or spectral classification (v1 scope):
spectral features need wider windows and more CPU per second of audio, and
their thresholds drift per game, language, and mic; on a 16 kHz mono downmix
of a compressed live stream they also inherit every codec artifact. An energy
z-score costs microseconds per second of audio, needs no training data, and
maps directly onto A4's strength formula (audio strength = z / spike_z,
capped at 2.0). If v1 recall proves too low, ``process_second`` is the
extension point: richer per-second features can reuse the baseline and burst
machinery unchanged.

Emission policy (what A4/A8 can rely on):

- burst onset -> one DetectorSignal at the onset second;
- peak updates only while z improves by >= ``_UPDATE_Z_GAIN`` over the last
  emitted value;
- the open burst's own seconds are excluded from the baseline ("louder than
  usual" needs the usual, not the excitement itself), so a burst whose
  pre-onset history has scrolled out of the window starves and ends on its
  own; ``max_sustained_s`` is the hard stop for configurations where that
  wouldn't happen in time, and after it no further signals fire until a
  non-spike second (or an unobserved time gap) re-arms the edge.

Stream time base: second ``i`` of a segment lands at ``t_s = start_s + i``,
where ``start_s`` is the segment's REAL start from the buffer's per-segment
index (stream copy cuts at keyframes, so segments are not ``SEGMENT_S``
long). A segment-like without ``start_s`` falls back to the nominal
``index * SEGMENT_S``. Each .ts file decodes standalone (ffmpeg resets its
timeline per input), so cross-second alignment is bounded by the segment's
own encoder priming.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autoclip.live.buffer import SEGMENT_S, RingBuffer
from autoclip.live.settings import LiveAudioSettings, LiveSettings
from autoclip.live.signals import DetectorSignal, SignalKind
from autoclip.pipeline.ffmpeg import FFmpegError, ffmpeg_path

log = logging.getLogger(__name__)

#: Analysis sample rate. 16 kHz mono captures the energy envelope with lots of
#: headroom; anything higher only feeds spectral features we deliberately
#: don't run (see the module docstring).
SAMPLE_RATE = 16_000

#: Committed-but-unanalyzed segments beyond this halve the analysis rate.
_SKIP_AHEAD_THRESHOLD = 4

#: Hard cap on one ffmpeg decode; a wedged child must never pin the worker.
_DECODE_TIMEOUT_S = 15.0

#: Watchdog granularity for stop/timeout checks during a decode.
_WATCHDOG_TICK_S = 0.05

#: Missing seconds between two evaluated seconds (skipped segment, epoch
#: boundary) beyond which an open burst is broken rather than bridged.
_GAP_S = 1.5

#: History required before z-scores mean anything; below this a couple of
#: early loud seconds would "spike" against nothing. Never more than the
#: baseline window can hold.
_MIN_BASELINE_S = 10

#: Baseline std under which the z formula degenerates (digital silence).
_STD_EPS = 1e-6

#: z assigned to a level jump over a flat baseline — large but finite, so
#: peak_z survives JSON serialization on the SSE event stream.
_FLAT_BASELINE_Z = 1000.0

#: z improvement required before a running burst re-emits an updated peak.
_UPDATE_Z_GAIN = 1.0


class AnalysisCancelled(RuntimeError):
    """Raised when a stop_event interrupts the worker mid-analysis."""


@dataclass(frozen=True)
class PassResult:
    """One analysis sweep, for logging, tests, and manager status."""

    analyzed: int = 0
    skipped: int = 0
    failed: int = 0
    emitted: int = 0
    pending: int = 0


@dataclass
class _Burst:
    """A run of consecutive spike seconds currently being tracked."""

    onset_s: float
    peak_z: float
    peak_s: float
    #: z of the last emitted signal; the next update must beat it by
    #: ``_UPDATE_Z_GAIN`` so a plateaued burst stays quiet.
    emitted_z: float


def audio_settings(settings: LiveAudioSettings | LiveSettings) -> LiveAudioSettings:
    """Accept either the LiveSettings tree or just its audio subsection.

    The manager holds the whole LiveSettings; narrower callers may pass the
    subsection — accepting both keeps the worker's signature frozen.
    """
    if isinstance(settings, LiveAudioSettings):
        return settings
    audio = getattr(getattr(settings, "detection", None), "audio", None)
    if isinstance(audio, LiveAudioSettings):
        return audio
    raise TypeError(
        "settings must be LiveAudioSettings or LiveSettings (.detection.audio); "
        f"got {type(settings).__name__}"
    )


def decode_segment_pcm(
    path: Path,
    *,
    sample_rate: int = SAMPLE_RATE,
    stop_event: threading.Event | None = None,
) -> np.ndarray:
    """Downmix one buffer segment to mono float32 PCM in [-1, 1].

    ffmpeg probes by content, so synthetic WAV bytes in a ``.ts``-named file
    decode identically to real MPEG-TS segments — that is what lets unit
    tests feed exact PCM without an encoder round-trip.

    Raises:
        AnalysisCancelled: ``stop_event`` was set (checked before the spawn
            and by a watchdog every ``_WATCHDOG_TICK_S`` mid-decode).
        FFmpegError: ffmpeg is missing, the segment is undecodable, or the
            decode exceeded ``_DECODE_TIMEOUT_S``.

    Preconditions:
        path points at a fully committed segment (never a file ffmpeg is
        still writing).
    """
    if stop_event is not None and stop_event.is_set():
        raise AnalysisCancelled("stop requested before decode")

    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finished = threading.Event()
    timed_out = threading.Event()

    def _watchdog() -> None:
        # communicate() blocks on the child's pipes, so the only prompt way
        # out of a slow/wedged child is killing it; the kill also closes the
        # pipes and unblocks the blocked read.
        deadline = time.monotonic() + _DECODE_TIMEOUT_S
        while not finished.wait(_WATCHDOG_TICK_S):
            if stop_event is not None and stop_event.is_set():
                proc.kill()
                return
            if time.monotonic() > deadline:
                timed_out.set()
                proc.kill()
                return

    watchdog = threading.Thread(target=_watchdog, name="audio-decode-watchdog", daemon=True)
    watchdog.start()
    try:
        stdout, stderr = proc.communicate()
    finally:
        finished.set()
        watchdog.join(timeout=_WATCHDOG_TICK_S * 4)

    if stop_event is not None and stop_event.is_set():
        raise AnalysisCancelled(f"stop requested during decode of {path.name}")
    if timed_out.is_set():
        raise FFmpegError(
            f"segment decode exceeded {_DECODE_TIMEOUT_S:.0f}s: {path.name}", command=command
        )
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg could not decode segment {path.name}",
            command=command,
            stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
        )
    if len(stdout) % 2:
        stdout = stdout[:-1]  # defensive: s16le is always even, never crash on odd
    return np.frombuffer(stdout, dtype="<i2").astype(np.float32) / 32768.0


class AudioSignalAnalyzer:
    """Per-second RMS -> baseline z-score -> burst signals, for one stream.

    ``on_signal`` receives one DetectorSignal per burst onset and per material
    peak update; it is called synchronously inside ``process_second``.

    Preconditions:
        seconds are fed strictly increasing in ``t_s`` (the worker guarantees
        it via segment order); ``on_signal`` is cheap and thread-safe.
    """

    def __init__(
        self,
        settings: LiveAudioSettings | LiveSettings,
        on_signal: Callable[[DetectorSignal], None],
        *,
        segment_s: float = SEGMENT_S,
    ) -> None:
        self.settings = audio_settings(settings)
        self._on_signal = on_signal
        self._segment_s = segment_s
        self._history: deque[tuple[float, float]] = deque()
        self._burst: _Burst | None = None
        self._saturated = False
        self._last_s: float | None = None
        # With the default 30 s window that's 30 samples; a shorter window
        # can't demand more history than it can hold, and under 2 samples is
        # statistically nothing.
        self._min_samples = max(2, min(_MIN_BASELINE_S, int(self.settings.baseline_window_s)))
        self._baseline_window_s = self.settings.baseline_window_s

    # -- analysis ----------------------------------------------------------

    def analyze_segment(
        self,
        index: int,
        path: Path,
        *,
        stop_event: threading.Event | None = None,
        start_s: float | None = None,
    ) -> list[DetectorSignal]:
        """Decode and analyze buffer segment ``index``; emits via ``on_signal``.

        ``start_s`` is the segment's real stream start (the buffer's index);
        None falls back to the nominal ``index * segment_s``.
        """
        pcm = decode_segment_pcm(path, stop_event=stop_event)
        t_start_s = index * self._segment_s if start_s is None else float(start_s)
        return self.analyze_pcm(pcm, t_start_s=t_start_s)

    def analyze_pcm(self, pcm: np.ndarray, t_start_s: float) -> list[DetectorSignal]:
        """Analyze raw PCM starting at stream time ``t_start_s``.

        Preconditions:
            pcm is mono float32 in [-1, 1] at ``SAMPLE_RATE``; ``t_start_s``
            is greater than every second previously fed.
        """
        n = pcm.size // SAMPLE_RATE
        if n == 0:
            return []
        frames = pcm[: n * SAMPLE_RATE].reshape(n, SAMPLE_RATE)
        levels = np.sqrt(np.square(frames).mean(axis=1))
        emitted = []
        for i in range(n):
            signal = self.process_second(t_start_s + i, float(levels[i]))
            if signal is not None:
                emitted.append(signal)
        return emitted

    def process_second(self, t_s: float, rms: float) -> DetectorSignal | None:
        """Feed one second of energy; emit a signal at burst edges.

        A second spikes when its z against the trailing baseline reaches
        ``spike_z`` AND its absolute RMS clears ``floor_rms`` (a z-score over
        near-silence is numerically huge but means nothing — a quiet room
        getting slightly less quiet is not a viral moment).
        """
        if self._last_s is not None and t_s - self._last_s > _GAP_S:
            # Unobserved time (skipped segment, epoch jump): a burst can't
            # bridge it, and saturation re-arms — the next rise is a new edge.
            self._burst = None
            self._saturated = False
        self._last_s = t_s

        mean, std, count = self._baseline()
        z = self._z_score(rms, mean, std, count)
        is_spike = (
            count >= self._min_samples
            and rms >= self.settings.floor_rms
            and z >= self.settings.spike_z
        )

        signal: DetectorSignal | None = None
        if is_spike and not self._saturated:
            if self._burst is None:
                self._burst = _Burst(onset_s=t_s, peak_z=z, peak_s=t_s, emitted_z=z)
                signal = self._signal(t_s, z, rms, mean, std, count, onset=True)
            elif t_s - self._burst.onset_s >= self.settings.max_sustained_s:
                # Sustained past the cap: stop updating this burst; only a
                # non-spike second (or a time gap) re-arms the edge.
                self._burst = None
                self._saturated = True
            elif z >= self._burst.emitted_z + _UPDATE_Z_GAIN:
                self._burst.peak_z = z
                self._burst.peak_s = t_s
                self._burst.emitted_z = z
                signal = self._signal(t_s, z, rms, mean, std, count, onset=False)
            elif z > self._burst.peak_z:
                self._burst.peak_z = z
                self._burst.peak_s = t_s
        elif not is_spike:
            self._burst = None
            self._saturated = False

        self._history.append((t_s, rms))
        while self._history and self._history[0][0] <= t_s - self._baseline_window_s:
            self._history.popleft()

        if signal is not None:
            self._emit(signal)
        return signal

    # -- state -------------------------------------------------------------

    def reset_baseline(self) -> None:
        """Forget all learned levels.

        After a reconnect (epoch change) the loudness distribution is new;
        scoring fresh audio against the pre-reconnect baseline manufactures
        phantom spikes exactly at the moment the stream is least trustworthy.
        """
        self._history.clear()
        self._burst = None
        self._saturated = False

    def baseline_state(self) -> tuple[float, float, int]:
        """(mean, std, seconds) of the current baseline — for tests and status."""
        return self._baseline()

    def burst_active(self) -> bool:
        return self._burst is not None

    # -- internals ---------------------------------------------------------

    def _baseline(self) -> tuple[float, float, int]:
        """Mean/std of trailing levels, excluding any open burst's seconds.

        "Louder than usual" needs the usual, not the excitement itself:
        scoring a burst against a baseline that already contains the burst
        collapses z within a couple of seconds and kills peak updates exactly
        when they matter. Burst seconds still land in history — once the
        burst ends they are the new normal.
        """
        # +inf when no burst is open (every observed second is "usual");
        # the onset second once one is (only pre-burst history is usual).
        cutoff = self._burst.onset_s if self._burst is not None else float("inf")
        levels = [rms for t, rms in self._history if t < cutoff]
        if not levels:
            return 0.0, 0.0, 0
        # float64, not float32: a float32 mean of identical values lands a
        # hair *below* the values themselves, so every subsequent identical
        # second tests as "above the mean" over a flat baseline and fires a
        # phantom z=_FLAT_BASELINE_Z spike ~20 s into any steady stream.
        # This is tens of samples per second — precision here costs nothing.
        values = np.asarray(levels, dtype=np.float64)
        return float(values.mean()), float(values.std()), values.size

    def _z_score(self, rms: float, mean: float, std: float, count: int) -> float:
        """z of ``rms`` against the baseline; 0.0 when it can't be trusted.

        A flat baseline (std ~ 0, e.g. digital silence) divides by zero and
        mislabels dithered quiet as infinitely extreme; a genuine level jump
        over a flat baseline is still a spike, so it gets a large-but-finite z.
        """
        if count < self._min_samples:
            return 0.0
        if std <= _STD_EPS:
            return _FLAT_BASELINE_Z if rms > mean else 0.0
        return (rms - mean) / std

    def _signal(
        self,
        t_s: float,
        z: float,
        rms: float,
        mean: float,
        std: float,
        count: int,
        *,
        onset: bool,
    ) -> DetectorSignal:
        strength = min(z / self.settings.spike_z, 2.0)
        if onset:
            detail = (
                f"audio energy spike onset: rms={rms:.3f} z={z:.1f} "
                f"(baseline mean={mean:.3f} std={std:.3f} over {count}s)"
            )
        else:
            # Only reachable while a burst is open (see process_second).
            onset_s = self._burst.onset_s if self._burst is not None else t_s
            detail = (
                f"audio burst peak: z={z:.1f} rms={rms:.3f} "
                f"(burst onset t={onset_s:.1f}s, baseline mean={mean:.3f} std={std:.3f})"
            )
        return DetectorSignal(
            kind=SignalKind.AUDIO_ENERGY,
            t_s=t_s,
            strength=strength,
            detail=detail,
            peak_z=z,
        )

    def _emit(self, signal: DetectorSignal) -> None:
        """Deliver via callback; a broken callback must not kill the worker.

        A4's aggregator hiccuping for one signal is survivable — a dead
        analyzer thread silently stops detecting anything at all.
        """
        try:
            self._on_signal(signal)
        except Exception:
            log.exception("on_signal callback failed; signal dropped: %r", signal)


def process_pass(
    buffer: RingBuffer,
    analyzer: AudioSignalAnalyzer,
    next_index: int | None,
    last_epoch: int | None,
    *,
    stop_event: threading.Event | None = None,
    skip_threshold: int = _SKIP_AHEAD_THRESHOLD,
) -> tuple[PassResult, int | None, int | None]:
    """Analyze the committed-but-unanalyzed backlog once (maybe every other).

    Returns ``(result, next_index, last_epoch)`` — the caller threads the last
    two through the next call. Split out of :func:`start` so tests and the
    manager can drive one deterministic pass without a thread.

    Preconditions:
        ``buffer.segments`` lists only fully committed segments (the capture
        side's contract); the analyzer is used from a single worker thread.
    """
    snapshot = sorted(getattr(buffer, "segments", ()), key=lambda s: s.index)
    if not snapshot:
        return PassResult(), next_index, last_epoch
    if stop_event is not None and stop_event.is_set():
        # Already stopped: report no work rather than a half-truth about skips.
        return PassResult(pending=len(snapshot)), next_index, last_epoch
    if next_index is None:
        next_index = snapshot[0].index
        if last_epoch is None:
            last_epoch = snapshot[0].epoch
    pending = [s for s in snapshot if s.index >= next_index]
    if not pending:
        return PassResult(), next_index, last_epoch

    # Behind schedule: halve the work. The baseline needs enough seconds, not
    # every second, and skipped segments show up as time gaps that cleanly
    # break bursts instead of smearing them.
    step = 2 if len(pending) > skip_threshold else 1
    analyzed = failed = emitted = 0
    last_processed: int | None = None
    for seg in pending[::step]:
        if stop_event is not None and stop_event.is_set():
            break
        if seg.epoch != last_epoch:
            if last_epoch is not None:
                analyzer.reset_baseline()
                log.info("epoch changed at segment %d; baseline reset", seg.index)
            last_epoch = seg.epoch
        try:
            signals = analyzer.analyze_segment(
                seg.index,
                seg.path,
                stop_event=stop_event,
                start_s=getattr(seg, "start_s", None),
            )
        except AnalysisCancelled:
            raise
        except (FFmpegError, OSError) as exc:
            # One torn segment (prune race, muxer hiccup) must not blind the
            # worker to everything after it.
            log.warning("segment %d undecodable, skipping: %s", seg.index, exc)
            failed += 1
            continue
        analyzed += 1
        emitted += len(signals)
        last_processed = seg.index

    result = PassResult(
        analyzed=analyzed,
        skipped=len(pending) - analyzed - failed,
        failed=failed,
        emitted=emitted,
        pending=len(pending),
    )
    if last_processed is not None:
        next_index = last_processed + 1
    return result, next_index, last_epoch


def start(
    buffer: RingBuffer,
    settings: LiveAudioSettings | LiveSettings,
    on_signal: Callable[[DetectorSignal], None],
    stop_event: threading.Event,
    *,
    poll_s: float = 1.0,
) -> None:
    """Per-stream analyzer worker: run until ``stop_event`` is set.

    Designed as the target of a plain :class:`threading.Thread` started by
    the manager (A8): no asyncio, at most one ffmpeg child alive at any
    moment, sequential segment analysis, and every idle wait parks on
    ``stop_event`` (not ``sleep``) so ``stop()`` is prompt.

    Preconditions:
        ``on_signal`` runs on this worker's thread and must be cheap and
        thread-safe; ``buffer`` is the live RingBuffer — or anything exposing
        a ``.segments`` list of committed ``SegmentInfo``-likes.
    """
    analyzer = AudioSignalAnalyzer(settings, on_signal)
    next_index: int | None = None
    last_epoch: int | None = None
    was_behind = False
    log.info("audio analyzer worker started")
    try:
        while not stop_event.is_set():
            result, next_index, last_epoch = process_pass(
                buffer, analyzer, next_index, last_epoch, stop_event=stop_event
            )
            log.debug("audio analyzer pass: %s", result)
            behind = result.skipped > 0
            if behind and not was_behind:
                log.warning(
                    "audio analyzer is behind (%d segments pending); "
                    "analyzing every other segment until caught up",
                    result.pending,
                )
            was_behind = behind
            newest = max((s.index for s in getattr(buffer, "segments", ())), default=None)
            if newest is None or next_index is None or next_index > newest:
                # Backlog drained — park on the event so stop() stays prompt.
                stop_event.wait(poll_s)
    except AnalysisCancelled:
        pass  # stop_event set mid-analysis; we're done either way
    finally:
        log.info("audio analyzer worker stopped")


__all__ = [
    "AnalysisCancelled",
    "AudioSignalAnalyzer",
    "PassResult",
    "SAMPLE_RATE",
    "audio_settings",
    "decode_segment_pcm",
    "process_pass",
    "start",
]
