"""Stall watchdog + coverage grace adapt to the stream's REAL segment lengths.

Stream copy cuts at keyframes, so a stream with a 20 s GOP (common on IRL /
mobile encoders) closes a segment only every ~20 s whatever ``-segment_time``
says. The old fixed timers (stall 18 s, coverage grace 24 s) were multiples
of the nominal 6 s segment: such a stream tripped the stall watchdog after
every segment and restarted capture in a loop.

The capture tests run the REAL tailer against a scaled clock (capture's
``time.monotonic`` runs SCALE× faster than wall time), so "20 s segments"
take a fraction of a second. No network, no ffmpeg, no real data dirs.
"""

from __future__ import annotations

import time
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from autoclip.live.buffer import (
    RECENT_SEGMENTS,
    SEGMENT_S,
    RingBuffer,
    adaptive_timeout_s,
    segment_name,
)
from autoclip.live.capture import (
    FIRST_SEGMENT_TIMEOUT_S,
    STALL_CEILING_S,
    STALL_SEGMENTS,
    STALL_TIMEOUT_S,
    StreamCapture,
)
from autoclip.live.settings import LiveRollingBufferSettings, LiveSettings
from autoclip.live.watcher import (
    COVERAGE_GRACE_CEILING_S,
    COVERAGE_GRACE_S,
    StreamWatcher,
    WatcherDeps,
)

from tests.test_live_capture import FakeProc

#: Virtual seconds per wall second inside capture.py.
SCALE = 50.0
#: The long keyframe interval under test.
GOP20 = 20.0
_WAIT_S = 10.0


@pytest.fixture(autouse=True)
def _no_real_tree_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    # FakeProc carries a made-up PID: never taskkill a real process.
    monkeypatch.setattr("autoclip.live.capture.kill_process_tree", lambda pid: None)


@pytest.fixture
def fast_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[], float]:
    """Make capture.py's monotonic clock run SCALE× faster than wall time."""
    real_monotonic = time.monotonic
    origin = real_monotonic()

    def virtual() -> float:
        return (real_monotonic() - origin) * SCALE

    monkeypatch.setattr(
        "autoclip.live.capture.time",
        types.SimpleNamespace(monotonic=virtual, sleep=time.sleep),
    )
    return virtual


def wait_until(predicate: Callable[[], bool], timeout_s: float = _WAIT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _append_row(buffer: RingBuffer, cap: StreamCapture, index: int, duration: float) -> None:
    """What ffmpeg does when segment ``index`` closes: one CSV row."""
    assert cap.segment_list_path is not None
    start = index * duration
    with open(cap.segment_list_path, "a", encoding="utf-8") as handle:
        handle.write(f"{segment_name(index)},{start:.3f},{start + duration:.3f}\n")


def _write_segment(buffer: RingBuffer, index: int) -> None:
    (buffer.directory / segment_name(index)).write_bytes(b"S" * 256)


def _capture(
    buffer: RingBuffer, proc: FakeProc, events: list[tuple[str, str]], **kwargs
) -> StreamCapture:
    return StreamCapture(
        "https://twitch.tv/irl",
        buffer,
        on_event=lambda k, d: events.append((k, d)),
        resolver=lambda u: "x",
        spawn=lambda u: proc,
        poll_s=0.01,
        drain_grace_s=0.05,
        **kwargs,
    )


def _produce(
    buffer: RingBuffer, cap: StreamCapture, count: int, duration: float, *, real_gap_s: float
) -> None:
    """Emit ``count`` closed segments, one per ``duration`` virtual seconds."""
    _write_segment(buffer, 0)
    for index in range(count):
        time.sleep(real_gap_s)
        _append_row(buffer, cap, index, duration)  # row first, like ffmpeg
        _write_segment(buffer, index + 1)  # N+1 opening closes N


# -- pure helpers -----------------------------------------------------------------


class TestAdaptiveTimeout:
    def test_nominal_segments_keep_the_old_timers(self) -> None:
        # 6 s segments: exactly the old fixed values.
        assert STALL_TIMEOUT_S == 3 * SEGMENT_S
        assert (
            adaptive_timeout_s(
                STALL_TIMEOUT_S, SEGMENT_S, segments=STALL_SEGMENTS, ceiling_s=STALL_CEILING_S
            )
            == STALL_TIMEOUT_S
        )

    def test_long_segments_stretch_it(self) -> None:
        assert adaptive_timeout_s(18.0, GOP20, segments=3, ceiling_s=90.0) == 60.0

    def test_ceiling_bounds_it(self) -> None:
        assert adaptive_timeout_s(18.0, 500.0, segments=3, ceiling_s=90.0) == 90.0

    def test_never_below_the_base(self) -> None:
        assert adaptive_timeout_s(18.0, 1.0, segments=3, ceiling_s=90.0) == 18.0
        # A ceiling below the base never shortens the base either.
        assert adaptive_timeout_s(18.0, 500.0, segments=3, ceiling_s=5.0) == 18.0

    @pytest.mark.parametrize("bad", [None, 0.0, -4.0, float("nan"), float("inf")])
    def test_unknown_lengths_fall_back_to_the_base(self, bad: float | None) -> None:
        assert adaptive_timeout_s(18.0, bad, segments=3, ceiling_s=90.0) == 18.0


class TestRecentSegmentMax:
    def test_empty_buffer_is_unknown(self, tmp_path: Path) -> None:
        assert RingBuffer(tmp_path / "b").recent_segment_max_s() is None

    def test_only_recent_segments_count(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "b")
        durations = [40.0] + [GOP20] * RECENT_SEGMENTS
        for index, duration in enumerate(durations):
            _write_segment(buffer, index)
            buffer.commit_segment(index, duration_s=duration)
        # The one 40 s freak segment has aged out of the window.
        assert buffer.recent_segment_max_s() == GOP20
        assert buffer.recent_segment_max_s(count=len(durations)) == 40.0


# -- the stall watchdog -------------------------------------------------------------


class TestLongGopStallWatchdog:
    def test_fixed_timeout_would_restart_a_20s_gop_stream(self, tmp_path: Path, fast_clock) -> None:
        # Control: the old behaviour (no adaptation) kills a HEALTHY capture
        # whose segments are 20 s apart, because 20 s > 18 s.
        events: list[tuple[str, str]] = []
        proc = FakeProc(alive_s=60)
        buffer = RingBuffer(tmp_path / "buf")
        cap = _capture(buffer, proc, events, stall_segments=0.0)
        try:
            cap.start()
            _produce(buffer, cap, 3, GOP20, real_gap_s=GOP20 / SCALE)
            assert wait_until(proc.killed.is_set)
        finally:
            cap.stop()
        assert any(k == "error" and "stalled" in d for k, d in events)

    def test_20s_segments_do_not_trip_the_watchdog(self, tmp_path: Path, fast_clock) -> None:
        events: list[tuple[str, str]] = []
        proc = FakeProc(alive_s=60)
        buffer = RingBuffer(tmp_path / "buf")
        cap = _capture(buffer, proc, events)  # production defaults
        try:
            cap.start()
            _produce(buffer, cap, 6, GOP20, real_gap_s=GOP20 / SCALE)
            assert wait_until(lambda: cap.committed == 6)
            assert not proc.killed.is_set()
            assert cap.stall_limit_s() == pytest.approx(STALL_SEGMENTS * GOP20)
        finally:
            cap.stop()
        assert not any(k == "error" for k, _ in events)
        assert [s.duration_s for s in buffer.segments] == [GOP20] * 6

    def test_a_genuinely_stalled_20s_capture_is_still_restarted(
        self, tmp_path: Path, fast_clock
    ) -> None:
        events: list[tuple[str, str]] = []
        proc = FakeProc(alive_s=600, kill_rc=1)
        buffer = RingBuffer(tmp_path / "buf")
        cap = _capture(buffer, proc, events)
        try:
            cap.start()
            _produce(buffer, cap, 3, GOP20, real_gap_s=GOP20 / SCALE)
            assert wait_until(lambda: cap.committed == 3)
            # ffmpeg stays alive but writes nothing more: a frozen capture.
            frozen_at = fast_clock()
            assert wait_until(proc.killed.is_set)
            idle = fast_clock() - frozen_at
            assert wait_until(lambda: any(k == "error" for k, _ in events))
        finally:
            cap.stop()
        detail = next(d for k, d in events if k == "error")
        assert "stalled" in detail  # the watcher reconnects on this "error"
        # Not before 3 real segments of silence, and well inside the ceiling.
        limit = STALL_SEGMENTS * GOP20
        assert idle >= limit - GOP20  # last commit happened before frozen_at
        assert idle < STALL_CEILING_S + 30.0

    def test_a_freak_long_segment_cannot_hide_a_dead_capture(
        self, tmp_path: Path, fast_clock
    ) -> None:
        # One 500 s "segment" (a source hiccup) would stretch 3× to 25 min;
        # the ceiling keeps the stall limit at STALL_CEILING_S.
        events: list[tuple[str, str]] = []
        proc = FakeProc(alive_s=600, kill_rc=1)
        buffer = RingBuffer(tmp_path / "buf")
        cap = _capture(buffer, proc, events)
        try:
            cap.start()
            _produce(buffer, cap, 1, 500.0, real_gap_s=0.05)
            assert wait_until(lambda: cap.committed == 1)
            assert cap.stall_limit_s() == STALL_CEILING_S
            assert wait_until(proc.killed.is_set)
        finally:
            cap.stop()
        assert any(k == "error" and "stalled" in d for k, d in events)

    def test_first_segment_grace_is_unchanged(self, tmp_path: Path) -> None:
        cap = _capture(RingBuffer(tmp_path / "buf"), FakeProc(), [])
        assert cap.stall_limit_s() == FIRST_SEGMENT_TIMEOUT_S

    def test_disabled_watchdog_stays_disabled(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        _write_segment(buffer, 0)
        buffer.commit_segment(0, duration_s=GOP20)
        cap = _capture(buffer, FakeProc(), [], stall_timeout_s=None)
        cap.committed = 1
        assert cap.stall_limit_s() is None


# -- the watcher's coverage grace ----------------------------------------------------


def _watcher(buffer: object, live: LiveSettings | None = None) -> StreamWatcher:
    deps = WatcherDeps(buffer_factory=lambda: buffer, capture_factory=lambda *a: None)
    watcher = StreamWatcher("s-gop", "url", "", live or LiveSettings(), deps=deps)
    watcher._buffer = buffer
    return watcher


class TestCoverageGrace:
    def test_nominal_segments_keep_the_old_grace(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        for index in range(3):
            _write_segment(buffer, index)
            buffer.commit_segment(index)  # nominal 6 s
        live = LiveSettings()
        assert _watcher(buffer, live)._coverage_timeout_s() == pytest.approx(
            live.rolling_buffer.post_roll_s + COVERAGE_GRACE_S
        )

    def test_20s_segments_stretch_the_grace(self, tmp_path: Path) -> None:
        # The segment holding a window's end commits only once it closes, up
        # to one REAL segment later: 24 s is too short for a 20 s GOP.
        buffer = RingBuffer(tmp_path / "buf")
        for index in range(4):
            _write_segment(buffer, index)
            buffer.commit_segment(index, duration_s=GOP20)
        live = LiveSettings()
        timeout = _watcher(buffer, live)._coverage_timeout_s()
        grace = timeout - live.rolling_buffer.post_roll_s
        assert grace >= 2 * GOP20
        assert grace <= COVERAGE_GRACE_CEILING_S

    def test_grace_is_capped(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        _write_segment(buffer, 0)
        buffer.commit_segment(0, duration_s=900.0)
        live = LiveSettings()
        assert _watcher(buffer, live)._coverage_timeout_s() == pytest.approx(
            live.rolling_buffer.post_roll_s + COVERAGE_GRACE_CEILING_S
        )

    def test_buffer_doubles_without_the_method_still_adapt(self) -> None:
        seg = types.SimpleNamespace(index=0, start_s=0.0, duration_s=GOP20)
        double = types.SimpleNamespace(segments=[seg])
        live = LiveSettings()
        grace = _watcher(double, live)._coverage_timeout_s() - live.rolling_buffer.post_roll_s
        assert grace >= 2 * GOP20

    def test_injected_timeout_wins(self, tmp_path: Path) -> None:
        deps = WatcherDeps(coverage_timeout_s=3.0, capture_factory=lambda *a: None)
        watcher = StreamWatcher("s", "url", "", LiveSettings(), deps=deps)
        assert watcher._coverage_timeout_s() == 3.0


# -- rolling_buffer.segment_s is a read-only report -----------------------------------


class TestSegmentSetting:
    def test_reports_what_capture_records(self) -> None:
        assert LiveRollingBufferSettings().segment_s == SEGMENT_S

    def test_a_stored_value_is_pinned_back(self) -> None:
        # Capture always asks ffmpeg for SEGMENT_S; a hand-edited config must
        # not make the UI claim a different segment length.
        assert LiveRollingBufferSettings(segment_s=10.0).segment_s == SEGMENT_S
        live = LiveSettings.model_validate({"rolling_buffer": {"segment_s": 2}})
        assert live.rolling_buffer.segment_s == SEGMENT_S
