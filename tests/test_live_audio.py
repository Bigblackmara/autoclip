"""Audio energy spike analyzer — unit tests (A2).

Two layers, deliberately separate:

- Pure math/state tests drive ``process_second`` with exact RMS levels and
  assert the baseline mean/std, z gating, and burst state transitions. No
  ffmpeg anywhere near them, so threshold math is verified exactly.
- Segment tests write synthetic WAV PCM into ``.ts``-named files (ffmpeg
  probes by content, so no encoder round-trip is needed to exercise the real
  decode pipe), commit them through a fake buffer, and assert DetectorSignal
  values — t_s math, floor suppression, epoch resets, worker lifecycle.

The real ffmpeg binary is used, not mocked: a mocked decoder tests nothing
about the one thing that breaks in the field (segment decode), and ffmpeg is
a hard runtime dependency of the project anyway. Every decode here is a
sub-second job on a ~100 KB synthetic file.
"""

from __future__ import annotations

import inspect
import math
import time
import wave
from pathlib import Path
from threading import Event, Thread
from typing import Any

import numpy as np
import pytest
from autoclip.live import buffer as live_buffer
from autoclip.live.audio_signals import (
    SAMPLE_RATE,
    AnalysisCancelled,
    AudioSignalAnalyzer,
    PassResult,
    audio_settings,
    decode_segment_pcm,
    process_pass,
    start,
)
from autoclip.live.buffer import SEGMENT_S, SegmentInfo
from autoclip.live.settings import LiveAudioSettings, LiveSettings
from autoclip.live.signals import DetectorSignal, SignalKind
from autoclip.pipeline.ffmpeg import FFmpegError

# --------------------------------------------------------------------------
# Helpers: synthetic segments + fake buffer
# --------------------------------------------------------------------------


def write_wav_segment(path: Path, samples: np.ndarray) -> None:
    """Write mono s16 PCM as a WAV file — into a ``.ts``-named path.

    ffmpeg probes by content rather than suffix, so the buffer's real
    ``seg_NNNNNN.ts`` files can be faked with exact-valued PCM.
    """
    pcm = np.clip(samples, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())


def constant_second(amplitude: float) -> np.ndarray:
    """One second of constant-amplitude audio — RMS is exactly the amplitude."""
    return np.full(SAMPLE_RATE, amplitude, dtype=np.float64)


def pure_tone_second(amplitude: float, freq: float = 440.0) -> np.ndarray:
    """One second of a sine, for content that is not a flat line."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    return amplitude * np.sin(2 * math.pi * freq * t)


class FakeBuffer:
    """Minimal stand-in for A1's RingBuffer: just an ordered segment list.

    The analyzer contract is ``.segments`` of index/path/epoch objects; pins,
    pruning, and extraction are A1's concern and irrelevant to analysis.
    Rebinds (never mutates) the list so a concurrent worker always holds a
    stable snapshot — the same guarantee the real buffer's commit path must
    give its readers.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.segments: list[SegmentInfo] = []

    def add(self, index: int, *, epoch: int = 0) -> Path:
        path = self.segment_path(index)
        info = SegmentInfo(
            index=index,
            path=path,
            size_bytes=path.stat().st_size,
            epoch=epoch,
        )
        self.segments = sorted(self.segments + [info], key=lambda s: s.index)
        return path

    def segment_path(self, index: int) -> Path:
        return self.directory / f"seg_{index:06d}.ts"


@pytest.fixture
def buffer_dir(tmp_path: Path) -> Path:
    d = tmp_path / "buffer"
    d.mkdir()
    return d


@pytest.fixture
def fake_buffer(buffer_dir: Path) -> FakeBuffer:
    return FakeBuffer(buffer_dir)


def make_segments(
    fake_buffer: FakeBuffer,
    per_second_amplitudes: list[list[float]],
    *,
    start_index: int = 0,
    epoch: int = 0,
) -> None:
    """Write + commit one 6 s segment per list of per-second amplitudes."""
    for offset, amps in enumerate(per_second_amplitudes):
        index = start_index + offset
        path = fake_buffer.segment_path(index)
        write_wav_segment(path, np.concatenate([constant_second(a) for a in amps]))
        fake_buffer.add(index, epoch=epoch)


class Collector:
    """Signal sink: records every DetectorSignal the analyzer emits."""

    def __init__(self) -> None:
        self.signals: list[DetectorSignal] = []

    def __call__(self, signal: DetectorSignal) -> None:
        self.signals.append(signal)


def quiet_analyzer(**overrides: Any) -> AudioSignalAnalyzer:
    """Analyzer with default settings (optionally overridden); sink discards."""
    return AudioSignalAnalyzer(LiveAudioSettings(**overrides), Collector())


# --------------------------------------------------------------------------
# Settings plumbing
# --------------------------------------------------------------------------


class TestAudioSettings:
    def test_accepts_subsection(self) -> None:
        sub = LiveAudioSettings()
        assert audio_settings(sub) is sub

    def test_accepts_full_tree(self) -> None:
        tree = LiveSettings()
        assert audio_settings(tree) is tree.detection.audio

    def test_rejects_unrelated_types(self) -> None:
        with pytest.raises(TypeError):
            audio_settings("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# decode_segment_pcm — the real ffmpeg pipe
# --------------------------------------------------------------------------


class TestDecodeSegmentPcm:
    def test_decodes_constant_amplitude_to_exact_rms(self, buffer_dir: Path) -> None:
        path = buffer_dir / "seg_000000.ts"
        write_wav_segment(path, constant_second(0.25))
        pcm = decode_segment_pcm(path)
        assert pcm.size == SAMPLE_RATE
        assert float(np.sqrt(np.square(pcm).mean())) == pytest.approx(0.25, abs=1e-3)

    def test_downmixes_stereo_source_to_mono(self, buffer_dir: Path) -> None:
        # Two channels in a .ts-named WAV: left loud, right silent. The mono
        # downmix must average the channels, not take either one alone.
        path = buffer_dir / "seg_stereo.ts"
        stereo = np.stack([constant_second(0.5), constant_second(0.0)], axis=1).reshape(-1)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())
        pcm = decode_segment_pcm(path)
        assert pcm.size >= SAMPLE_RATE - 1
        assert float(np.sqrt(np.square(pcm).mean())) == pytest.approx(0.25, abs=0.02)

    def test_garbage_file_raises_ffmpeg_error(self, buffer_dir: Path) -> None:
        path = buffer_dir / "seg_bad.ts"
        path.write_bytes(b"\x00garbage-not-media")
        with pytest.raises(FFmpegError):
            decode_segment_pcm(path)

    def test_empty_file_raises_ffmpeg_error(self, buffer_dir: Path) -> None:
        (buffer_dir / "seg_empty.ts").write_bytes(b"")
        with pytest.raises(FFmpegError):
            decode_segment_pcm(buffer_dir / "seg_empty.ts")

    def test_missing_file_raises(self, buffer_dir: Path) -> None:
        with pytest.raises(FFmpegError):
            decode_segment_pcm(buffer_dir / "seg_nope.ts")

    def test_stop_event_before_decode_raises_cancelled(self, buffer_dir: Path) -> None:
        path = buffer_dir / "seg_000000.ts"
        write_wav_segment(path, constant_second(0.1))
        stop = Event()
        stop.set()
        with pytest.raises(AnalysisCancelled):
            decode_segment_pcm(path, stop_event=stop)


# --------------------------------------------------------------------------
# Baseline math (pure, no ffmpeg)
# --------------------------------------------------------------------------


class TestBaselineMath:
    def test_no_spike_before_minimum_history(self) -> None:
        # Below the minimum-history guard z is untrusted (0): even a 50x
        # jump must wait until there is enough baseline to mean anything —
        # otherwise every stream "spikes" in its first nine seconds.
        a = quiet_analyzer()
        for t in range(9):
            assert a.process_second(float(t), 0.01) is None
        assert a.process_second(9.0, 0.5) is None  # only 9 baseline samples
        assert a.process_second(10.0, 0.5) is not None  # 10 samples: fires

    def test_baseline_mean_and_std_match_manual_computation(self) -> None:
        a = quiet_analyzer(baseline_window_s=10)
        for t in range(10):
            a.process_second(float(t), 0.1 + 0.01 * (t % 3))
        mean, std, count = a.baseline_state()
        vals = np.array([0.1, 0.11, 0.12] * 3 + [0.1])
        assert count == 10
        assert mean == pytest.approx(vals.mean(), abs=1e-6)
        assert std == pytest.approx(vals.std(), abs=1e-5)

    def test_baseline_excludes_open_burst_seconds(self) -> None:
        a = quiet_analyzer(baseline_window_s=30)
        for t in range(20):
            a.process_second(float(t), 0.05 + 0.001 * (t % 5))
        assert a.process_second(20.0, 0.6) is not None
        mean, std, count = a.baseline_state()
        assert count == 20  # onset second excluded from its own baseline
        assert mean < 0.1

    def test_baseline_rolls_off_with_window(self) -> None:
        a = quiet_analyzer(baseline_window_s=10)
        for t in range(30):
            a.process_second(float(t), 0.5 if t < 10 else 0.05)
        mean, std, count = a.baseline_state()
        assert count == 10
        assert mean == pytest.approx(0.05, abs=1e-6)

    def test_flat_baseline_jump_gets_finite_z(self) -> None:
        # 30 identical quiet seconds -> std=0; the jump is still a spike,
        # with a large-but-finite z (peak_z rides the SSE event stream).
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        for t in range(30):
            a.process_second(float(t), 0.05)
        sig = a.process_second(30.0, 0.9)
        assert sig is not None
        assert math.isfinite(sig.peak_z)
        assert sig.peak_z > 100

    def test_reset_baseline_forgets_history(self) -> None:
        a = quiet_analyzer()
        for t in range(15):
            a.process_second(float(t), 0.05)
        a.reset_baseline()
        assert a.baseline_state() == (0.0, 0.0, 0)

    def test_time_gap_breaks_open_burst(self) -> None:
        # Six unobserved seconds (skipped segment / epoch jump): the loud
        # continuation fires a NEW onset instead of silently continuing the
        # old burst — whose z was ~400 against the quiet baseline, while a
        # continuation's z against the polluted history would be ~3.5.
        a = quiet_analyzer()
        for t in range(12):
            a.process_second(float(t), 0.05 + 0.001 * (t % 5))
        sig = a.process_second(12.0, 0.7)
        assert sig is not None
        assert "onset" in sig.detail
        sig2 = a.process_second(18.0, 0.7)
        assert sig2 is not None
        assert "onset" in sig2.detail
        assert sig2.t_s == 18.0


# --------------------------------------------------------------------------
# Spike gating: z AND floor_rms
# --------------------------------------------------------------------------


class TestFloorRmsSuppression:
    def test_high_z_over_silence_does_not_fire(self) -> None:
        # True-silence baseline (std 0), one second of tiny dither: z would
        # be huge, but 0.005 < floor_rms 0.02 — a quiet room getting
        # slightly less quiet is not a viral moment.
        a = quiet_analyzer()
        for t in range(30):
            a.process_second(float(t), 0.0)
        assert a.process_second(30.0, 0.005) is None

    def test_lower_floor_lets_the_same_level_fire(self) -> None:
        a = quiet_analyzer(floor_rms=0.001)
        for t in range(30):
            a.process_second(float(t), 0.0)
        assert a.process_second(30.0, 0.005) is not None

    def test_loud_but_unremarkable_level_does_not_fire(self) -> None:
        # 0.5 clears the floor easily but equals its own baseline: a
        # loudness-war stream must not "spike" at its steady level.
        a = quiet_analyzer()
        for t in range(30):
            a.process_second(float(t), 0.5)
        assert a.process_second(30.0, 0.5) is None


# --------------------------------------------------------------------------
# Burst / sustained handling (pure)
# --------------------------------------------------------------------------


class TestBurstHandling:
    def _prime(self, a: AudioSignalAnalyzer, seconds: int = 12) -> None:
        for t in range(seconds):
            a.process_second(float(t), 0.05 + 0.001 * (t % 5))

    def test_one_onset_signal_per_burst(self) -> None:
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        self._prime(a)
        fired = [a.process_second(float(12 + i), 0.8) for i in range(5)]  # plateau
        signals = [s for s in fired if s is not None]
        assert len(signals) == 1
        assert "onset" in signals[0].detail
        assert len(sink.signals) == 1

    def test_peak_update_only_on_material_z_gain(self) -> None:
        # Alternating 0.03/0.07 baseline: mean 0.05, std 0.02 exactly.
        # z(0.12)=3.5 fires the onset; z(0.125)=3.75 is under the last
        # emitted z + 1.0 so it updates nothing; z(0.20)=7.5 is material.
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        for t in range(12):
            a.process_second(float(t), 0.03 if t % 2 == 0 else 0.07)
        onset = a.process_second(12.0, 0.12)
        assert onset is not None
        assert "onset" in onset.detail
        assert a.process_second(13.0, 0.125) is None
        update = a.process_second(14.0, 0.20)
        assert update is not None
        assert "peak" in update.detail
        assert update.t_s == 14.0
        assert [s.t_s for s in sink.signals] == [12.0, 14.0]

    def test_non_spike_second_closes_burst(self) -> None:
        a = quiet_analyzer()
        self._prime(a)
        a.process_second(12.0, 0.8)
        assert a.burst_active()
        a.process_second(13.0, 0.05)
        assert not a.burst_active()

    def test_burst_excludes_its_own_seconds_from_baseline(self) -> None:
        # Ten plateau seconds: the baseline must stay at the pre-burst level
        # — proof the burst never became its own "usual".
        a = quiet_analyzer()
        self._prime(a)
        primed_mean = a.baseline_state()[0]
        fired = [a.process_second(float(12 + i), 0.8) for i in range(10)]
        assert a.baseline_state()[0] == pytest.approx(primed_mean, abs=1e-6)
        assert fired[0] is not None
        assert all(s is None for s in fired[1:])  # a plateau stays quiet

    def test_max_sustained_caps_burst_updates(self) -> None:
        # Once a burst is max_sustained_s old, further spike seconds fire
        # nothing even if z keeps climbing.
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(max_sustained_s=3.0), sink)
        self._prime(a)
        assert a.process_second(12.0, 0.6) is not None  # onset at t=12
        a.process_second(13.0, 0.55)  # within the cap
        a.process_second(14.0, 0.6)  # within the cap
        a.process_second(15.0, 0.65)  # burst is 3 s old: saturates, no signal
        after = [a.process_second(16.0 + i, 0.9) for i in range(3)]
        assert all(s is None for s in after)
        assert len(sink.signals) == 1

    def test_quiet_second_re_arms_after_saturation(self) -> None:
        # After saturation the loud level has legitimately joined the
        # baseline, so re-arming needs a quiet second and an even LOUDER
        # rise — 0.6 was absorbed, 0.95 stands out again.
        a = quiet_analyzer(max_sustained_s=2.0)
        self._prime(a)
        a.process_second(12.0, 0.6)
        a.process_second(13.0, 0.6)
        a.process_second(14.0, 0.6)  # 2 s old: saturates
        assert a.process_second(15.0, 0.6) is None
        a.process_second(16.0, 0.05)  # quiet second re-arms the edge
        assert a.process_second(17.0, 0.95) is not None


# --------------------------------------------------------------------------
# Signal values (pure)
# --------------------------------------------------------------------------


class TestSignalValues:
    def _varying(self, a: AudioSignalAnalyzer, seconds: int = 30) -> None:
        for t in range(seconds):
            a.process_second(float(t), 0.05 + 0.001 * (t % 4))

    def test_strength_is_z_over_spike_z_uncapped(self) -> None:
        # A near-threshold spike (z ~4.5 vs spike_z 2.5): the linear strength
        # formula actually holds below A4's 2.0 cap — ~1.8, not 2.0.
        a = quiet_analyzer()
        self._varying(a)
        mean, std, count = a.baseline_state()
        assert count == 30
        sig = a.process_second(30.0, 0.0565)
        assert sig is not None
        expected_z = (0.0565 - mean) / std
        assert sig.peak_z == pytest.approx(expected_z, rel=0.01)
        assert sig.strength == pytest.approx(expected_z / 2.5, rel=0.01)
        assert sig.strength < 2.0

    def test_strength_capped_at_two(self) -> None:
        # A 0.9 jump over a 0.05-ish baseline: z in the hundreds, strength
        # must not follow it past A4's cap.
        a = quiet_analyzer()
        self._varying(a)
        sig = a.process_second(30.0, 0.9)
        assert sig is not None
        assert sig.peak_z > 100
        assert sig.strength == pytest.approx(2.0)

    def test_kind_and_fields(self) -> None:
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        self._varying(a)
        sig = a.process_second(30.0, 0.8)
        assert sig is not None
        assert isinstance(sig, DetectorSignal)
        assert sig.kind is SignalKind.AUDIO_ENERGY
        assert sig.t_s == 30.0
        assert "onset" in sig.detail


# --------------------------------------------------------------------------
# Segment-level analysis through the real ffmpeg pipe
# --------------------------------------------------------------------------


class TestAnalyzeSegment:
    def test_spike_detected_with_correct_stream_time(self, fake_buffer: FakeBuffer) -> None:
        # Three quiet segments then a loud one: onset must land at the first
        # second of segment 3, i.e. t_s == 3 * SEGMENT_S exactly.
        make_segments(fake_buffer, [[0.05] * 6] * 3 + [[0.7] * 6])
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        for seg in fake_buffer.segments:
            a.analyze_segment(seg.index, seg.path)
        assert len(sink.signals) == 1
        onset = sink.signals[0]
        assert onset.kind is SignalKind.AUDIO_ENERGY
        assert onset.t_s == pytest.approx(3 * SEGMENT_S)

    def test_spike_mid_segment_maps_to_index_times_segment_s_plus_offset(
        self, fake_buffer: FakeBuffer
    ) -> None:
        make_segments(fake_buffer, [[0.05] * 6] * 5 + [[0.05, 0.05, 0.8, 0.8, 0.8, 0.8]])
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        for seg in fake_buffer.segments:
            a.analyze_segment(seg.index, seg.path)
        assert len(sink.signals) == 1
        assert sink.signals[0].t_s == pytest.approx(5 * SEGMENT_S + 2)

    def test_pure_tones_detected_same_as_constants(self, fake_buffer: FakeBuffer) -> None:
        # The same RMS delivered by sines instead of flat lines: detection
        # is content-agnostic and the time base holds for real waveforms.
        for index in range(3):
            write_wav_segment(fake_buffer.segment_path(index), np.tile(pure_tone_second(0.05), 6))
            fake_buffer.add(index)
        write_wav_segment(fake_buffer.segment_path(3), np.tile(pure_tone_second(0.7), 6))
        fake_buffer.add(3)
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        for seg in fake_buffer.segments:
            a.analyze_segment(seg.index, seg.path)
        assert len(sink.signals) == 1
        assert sink.signals[0].t_s == pytest.approx(3 * SEGMENT_S)

    def test_floor_suppresses_spike_through_real_pipe(self, fake_buffer: FakeBuffer) -> None:
        # Near-silence baseline, then a tenfold-louder dither that is still
        # under floor_rms=0.02: z is real but the fire is suppressed.
        make_segments(fake_buffer, [[0.001] * 6] * 4 + [[0.015] * 6])
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        for seg in fake_buffer.segments:
            a.analyze_segment(seg.index, seg.path)
        assert not sink.signals

    def test_partial_seconds_floored_not_rounded_up(self, fake_buffer: FakeBuffer) -> None:
        # A 3.5 s file: 3 full seconds analyzed, the half second dropped —
        # a truncated segment must never fabricate a phantom fourth second.
        path = fake_buffer.segment_path(0)
        write_wav_segment(path, np.full(int(3.5 * SAMPLE_RATE), 0.05))
        fake_buffer.add(0)
        a = quiet_analyzer()
        a.analyze_segment(0, path)
        assert a.baseline_state()[2] == 3


# --------------------------------------------------------------------------
# process_pass — backlog sweep + skip-ahead + epochs
# --------------------------------------------------------------------------


class TestProcessPass:
    def _fill(self, fake_buffer: FakeBuffer, count: int, *, start: int = 0, epoch: int = 0) -> None:
        make_segments(
            fake_buffer,
            [[0.05] * 6 for _ in range(count)],
            start_index=start,
            epoch=epoch,
        )

    def test_first_pass_processes_all_pending(self, fake_buffer: FakeBuffer) -> None:
        self._fill(fake_buffer, 3)
        a = quiet_analyzer()
        result, next_index, last_epoch = process_pass(fake_buffer, a, None, None)
        assert result.analyzed == 3
        assert result.failed == 0
        assert next_index == 3
        assert last_epoch == 0

    def test_second_pass_with_no_new_segments_is_idle(self, fake_buffer: FakeBuffer) -> None:
        self._fill(fake_buffer, 3)
        a = quiet_analyzer()
        _, next_index, last_epoch = process_pass(fake_buffer, a, None, None)
        result, next_index, last_epoch = process_pass(fake_buffer, a, next_index, last_epoch)
        assert result == PassResult()
        assert next_index == 3
        assert last_epoch == 0

    def test_empty_buffer_is_idle(self, fake_buffer: FakeBuffer) -> None:
        result, next_index, last_epoch = process_pass(fake_buffer, quiet_analyzer(), None, None)
        assert result == PassResult()
        assert next_index is None
        assert last_epoch is None

    def test_backlog_beyond_threshold_halves_work(self, fake_buffer: FakeBuffer) -> None:
        self._fill(fake_buffer, 10)
        a = quiet_analyzer()
        result, next_index, _ = process_pass(fake_buffer, a, None, None)
        assert result.analyzed == 5  # every other segment while behind
        assert result.skipped == 5
        assert next_index == 9  # last analyzed index + 1

    def test_after_skip_next_pass_picks_up_remainder(self, fake_buffer: FakeBuffer) -> None:
        self._fill(fake_buffer, 10)
        a = quiet_analyzer()
        _, next_index, _ = process_pass(fake_buffer, a, None, None)
        result, next_index, _ = process_pass(fake_buffer, a, next_index, None)
        assert result.analyzed == 1  # only segment 9 was left behind
        assert result.skipped == 0
        assert next_index == 10

    def test_stop_event_freezes_work_before_any_analysis(self, fake_buffer: FakeBuffer) -> None:
        self._fill(fake_buffer, 4)
        a = quiet_analyzer()
        stop = Event()
        stop.set()
        result, next_index, _ = process_pass(fake_buffer, a, None, None, stop_event=stop)
        assert result.analyzed == 0
        assert result.pending == 4
        assert result.skipped == 0  # stopping is not "skipping ahead"
        assert next_index is None  # cursor unmoved: the backlog survives

    def test_failed_segments_are_skipped_not_fatal(self, fake_buffer: FakeBuffer) -> None:
        # A torn segment between two good ones: the pass survives it and
        # keeps analyzing — one bad file must not blind the worker.
        self._fill(fake_buffer, 3)
        fake_buffer.segment_path(1).write_bytes(b"\x00\x01\x02 not media at all")
        a = quiet_analyzer()
        result, next_index, _ = process_pass(fake_buffer, a, None, None)
        assert result.failed == 1
        assert result.analyzed == 2
        assert result.pending == 3
        assert next_index == 3  # cursor advances past the failure

    def test_epoch_change_resets_baseline(self, fake_buffer: FakeBuffer) -> None:
        # Reconnect (new epoch) with a much louder mix: without the reset,
        # post-reconnect audio would spike against the pre-reconnect quiet
        # baseline and fire a phantom event at the worst moment.
        self._fill(fake_buffer, 2, start=0, epoch=0)
        write_wav_segment(fake_buffer.segment_path(2), np.full(6 * SAMPLE_RATE, 0.4))
        fake_buffer.add(2, epoch=1)
        sink = Collector()
        a = AudioSignalAnalyzer(LiveAudioSettings(), sink)
        result, _, last_epoch = process_pass(fake_buffer, a, None, None)
        assert result.analyzed == 3
        assert last_epoch == 1
        assert not sink.signals  # nothing to spike against post-reset
        mean, _, count = a.baseline_state()
        assert count == 6  # only the epoch-1 segment's seconds remain
        assert mean == pytest.approx(0.4, abs=0.01)

    def test_epoch_listed_mid_pass_does_not_crash(self, fake_buffer: FakeBuffer) -> None:
        # Refusing epoch mixing at extract time is the buffer's job; the
        # analyzer just resets its baseline at whatever boundary it sees.
        self._fill(fake_buffer, 2, start=0, epoch=0)
        self._fill(fake_buffer, 2, start=2, epoch=9)
        a = quiet_analyzer()
        result, _, last_epoch = process_pass(fake_buffer, a, None, None)
        assert result.analyzed == 4
        assert result.failed == 0
        assert last_epoch == 9


# --------------------------------------------------------------------------
# start() — the threaded worker (real ffmpeg decodes)
# --------------------------------------------------------------------------


class TestWorkerStart:
    def test_stop_event_before_start_exits_immediately(self, fake_buffer: FakeBuffer) -> None:
        stop = Event()
        stop.set()
        # Must not block or spin despite an empty buffer; reaching the
        # return at all is the assertion.
        start(fake_buffer, LiveAudioSettings(), Collector(), stop, poll_s=60)

    def test_worker_analyzes_arriving_segments_and_stops(self, fake_buffer: FakeBuffer) -> None:
        stop = Event()
        sink = Collector()
        # Two quiet segments pre-committed so a loud segment can fire
        # without waiting for the 10 s minimum baseline to accumulate live.
        make_segments(fake_buffer, [[0.05] * 6] * 2)
        worker = Thread(
            target=start,
            args=(fake_buffer, LiveAudioSettings(), sink, stop),
            kwargs={"poll_s": 0.05},
            daemon=True,
        )
        worker.start()
        try:
            # Commit a loud segment while the worker runs: exactly one onset.
            make_segments(fake_buffer, [[0.7] * 6], start_index=2)
            deadline = time.monotonic() + 3.0
            while not sink.signals and time.monotonic() < deadline:
                time.sleep(0.02)
            assert len(sink.signals) == 1
            assert sink.signals[0].t_s == pytest.approx(2 * SEGMENT_S)
        finally:
            stop.set()
            worker.join(timeout=5)
        assert not worker.is_alive()

    def test_worker_stops_promptly_when_parked(self, fake_buffer: FakeBuffer) -> None:
        # The idle wait must park on stop_event (not time.sleep), so stop()
        # breaks it immediately instead of at the next poll tick.
        make_segments(fake_buffer, [[0.05] * 6 for _ in range(12)])
        stop = Event()
        worker = Thread(
            target=start,
            args=(fake_buffer, LiveAudioSettings(), Collector(), stop),
            kwargs={"poll_s": 30},
            daemon=True,
        )
        worker.start()
        time.sleep(0.3)  # fast decodes: the backlog drains and parks quickly
        t0 = time.monotonic()
        stop.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert time.monotonic() - t0 < 2.0  # nowhere near the 30 s poll

    def test_worker_idles_cleanly_on_empty_buffer(self, fake_buffer: FakeBuffer) -> None:
        stop = Event()
        worker = Thread(
            target=start,
            args=(fake_buffer, LiveAudioSettings(), Collector(), stop),
            kwargs={"poll_s": 0.05},
            daemon=True,
        )
        worker.start()
        time.sleep(0.2)
        assert worker.is_alive()  # idle is not crash
        stop.set()
        worker.join(timeout=5)
        assert not worker.is_alive()


# --------------------------------------------------------------------------
# On-signal callback robustness
# --------------------------------------------------------------------------


class TestCallbackRobustness:
    def test_broken_callback_does_not_kill_analyzer(self, fake_buffer: FakeBuffer) -> None:
        def boom(signal: DetectorSignal) -> None:
            raise RuntimeError("aggregator is having a moment")

        # 3 quiet + 1 loud segment: the onset's callback exception must be
        # swallowed, not propagated — a dead analyzer thread would silently
        # stop detecting anything at all.
        make_segments(fake_buffer, [[0.05] * 6] * 3 + [[0.8] * 6])
        a = AudioSignalAnalyzer(LiveAudioSettings(), boom)
        for seg in fake_buffer.segments:
            a.analyze_segment(seg.index, seg.path)  # must not raise


# --------------------------------------------------------------------------
# Manager-facing plumbing
# --------------------------------------------------------------------------


class TestManagerPlumbing:
    def test_accepts_full_live_settings_tree(self, fake_buffer: FakeBuffer) -> None:
        # The manager passes the whole LiveSettings, not the audio subtree.
        make_segments(fake_buffer, [[0.05] * 6] * 3 + [[0.9] * 6])
        sink = Collector()
        a = AudioSignalAnalyzer(LiveSettings(), sink)
        for seg in fake_buffer.segments:
            a.analyze_segment(seg.index, seg.path)
        assert len(sink.signals) == 1
        assert sink.signals[0].t_s == pytest.approx(3 * SEGMENT_S)

    def test_segment_s_default_follows_buffer_contract(self) -> None:
        # t_s math hinges on SEGMENT_S: assert the analyzer's default tracks
        # the buffer constant rather than a frozen literal.
        params = inspect.signature(AudioSignalAnalyzer.__init__).parameters
        assert params["segment_s"].default == SEGMENT_S == live_buffer.SEGMENT_S
