"""StreamCapture unit tests — resolution, tailer commits, cancellation, events.

The failure modes that actually matter here are process-related: a resolver
that hangs, an ffmpeg that ignores a naive kill() (Windows kills must walk the
tree), a tailer committing a half-written file, and a start() that hides a
dead channel instead of raising. All of those are testable without touching
the network — resolution goes through the injected seam, ffmpeg through fake
Popen objects with an explicit kill/exit protocol, and real files on disk for
the tailer, exactly like the work order's test strategy freezes.

Timing hygiene: every wait on a thread is deadline-bounded, and stop() is
always called in a finally — a hung fake here would balloon the suite into
minutes (a first draft of this file did exactly that), so the harness makes
deadlocks fail tests instead of hanging them.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from autoclip.live.buffer import RingBuffer, segment_name
from autoclip.live.capture import (
    DRAIN_GRACE_S,
    SEGMENT_TIME_S,
    TAILER_POLL_S,
    CaptureError,
    StreamCapture,
    index_from_name,
    is_segment_name,
    kill_process_tree,
    resolve_stream_url,
)

# Generous for slow CI, but finite, so a deadlock fails a test instead of
# hanging the whole suite.
_WAIT_S = 10.0


@pytest.fixture(autouse=True)
def _no_real_tree_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never run a real ``taskkill /PID 4242 /T /F`` against the host.

    FakeProc carries a made-up PID; stop() on a live fake would otherwise
    tree-kill whatever real process happens to own that PID. Tests that
    exercise kill_process_tree itself import the real function directly.
    """
    monkeypatch.setattr("autoclip.live.capture.kill_process_tree", lambda pid: None)


def wait_until(predicate: Callable[[], bool], timeout_s: float = _WAIT_S) -> bool:
    """Poll ``predicate`` until true or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeProc:
    """Popen stand-in with an explicit alive→exit protocol.

    ``rc`` is what a *waited* exit reports; ``kill_rc`` is what a killed
    process reports, so tests can tell "stream ended" from "user stop" —
    mirroring real semantics closely enough for the waiter thread.
    """

    def __init__(self, *, alive_s: float = 30.0, rc: int = 0, kill_rc: int = -9) -> None:
        self.pid = 4242
        self.stderr = None
        self.returncode: int | None = None
        self._exit_at = time.monotonic() + alive_s
        self._rc = rc
        self._kill_rc = kill_rc
        self.killed = threading.Event()

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self.killed.is_set() or time.monotonic() >= self._exit_at:
            self.returncode = self._kill_rc if self.killed.is_set() else self._rc
            return self.returncode
        return None

    def wait(self, timeout: float | None = None) -> int:
        deadline = time.monotonic() + (timeout if timeout else 60.0)
        while self.poll() is None:
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            time.sleep(0.01)
        return self.returncode or 0

    def kill(self) -> None:
        self.killed.set()


def make_capture(
    buffer: RingBuffer,
    *,
    url: str = "https://twitch.tv/example",
    resolver: Callable[[str], str] | None = None,
    spawn: Callable[[str], Any] | None = None,
    events: list[tuple[str, str]] | None = None,
    poll_s: float = 0.02,
    drain_grace_s: float = 0.1,
) -> StreamCapture:
    """Build a capture with all effects observable (no real subprocesses)."""
    if events is None:
        events = []
    on_event = lambda kind, detail: events.append((kind, detail))  # noqa: E731
    return StreamCapture(
        url,
        buffer,
        on_event=on_event,
        resolver=resolver,
        spawn=spawn,
        poll_s=poll_s,
        drain_grace_s=drain_grace_s,
    )


def seed_segment(buffer: RingBuffer, index: int, *, size: int = 256) -> Path:
    """Write a segment file WITHOUT committing — exactly what ffmpeg does."""
    path = buffer.directory / segment_name(index)
    path.write_bytes(b"S" * size)
    return path


class TestNameParsing:
    def test_is_segment_name(self) -> None:
        assert is_segment_name("seg_000000.ts")
        assert is_segment_name("seg_123456.ts")
        # ffmpeg's %06d widens past six digits instead of truncating, so
        # seven-digit names (85+ days of 24/7 capture) must still parse.
        assert is_segment_name("seg_1234567.ts")
        assert not is_segment_name("seg_000000.mp4")
        assert not is_segment_name(".hidden.ts")
        assert not is_segment_name("seg_a0000.ts")

    def test_index_round_trip(self) -> None:
        for i in (0, 5, 999999):
            assert index_from_name(segment_name(i)) == i

    def test_bad_index_raises(self) -> None:
        with pytest.raises(ValueError):
            index_from_name("nonsense")


class TestConstants:
    def test_segment_time_matches_buffer(self) -> None:
        # The -segment_time asked of ffmpeg, and the nominal fallback length
        # of a segment without a list row. Real times come from the list
        # (tests/test_live_segment_timing.py), never from index * this.
        assert SEGMENT_TIME_S == 6.0

    def test_tailer_defaults_are_sane(self) -> None:
        # Small enough to commit promptly, large enough to see stable sizes.
        assert TAILER_POLL_S <= 1.0
        assert DRAIN_GRACE_S >= TAILER_POLL_S


class TestStartContract:
    """start() blocks on resolution and raises for dead channels."""

    def test_start_raises_for_unplayable_url(self, tmp_path: Path) -> None:
        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda _url: "")
        with pytest.raises(CaptureError, match="No playable streams"):
            cap.start()

    def test_dead_channel_error_is_offline(self, tmp_path: Path) -> None:
        # The watcher parks offline-shaped errors in waiting_for_live
        # instead of burning the reconnect budget — this is the A1↔A8
        # contract, pinned here.
        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda _url: "")
        with pytest.raises(CaptureError) as excinfo:
            cap.start()
        assert excinfo.value.offline is True
        assert "no playable streams" in str(excinfo.value).lower()
        assert excinfo.value.hint

    def test_custom_resolver_used(self, tmp_path: Path) -> None:
        got: list[str] = []
        spawn_urls: list[str] = []

        def resolver(url: str) -> str:
            got.append(url)
            return "http://media.example/live.m3u8"

        def spawn(stream_url: str):
            spawn_urls.append(stream_url)
            return FakeProc(alive_s=30)

        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            url="https://twitch.tv/x",
            resolver=resolver,
            spawn=spawn,
        )
        try:
            cap.start()
            assert got == ["https://twitch.tv/x"]
            assert spawn_urls == ["http://media.example/live.m3u8"]
        finally:
            cap.stop()

    def test_start_twice_raises(self, tmp_path: Path) -> None:
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x/live",
            spawn=lambda u: FakeProc(alive_s=30),
        )
        try:
            cap.start()
            with pytest.raises(RuntimeError, match="already started"):
                cap.start()
        finally:
            cap.stop()

    def test_start_emits_started(self, tmp_path: Path) -> None:
        events: list[tuple[str, str]] = []
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x/live",
            spawn=lambda u: FakeProc(alive_s=30),
            events=events,
        )
        try:
            cap.start()
            assert events
            assert events[0][0] == "started"
        finally:
            cap.stop()

    def test_start_refuses_when_stopped_mid_resolution(self, tmp_path: Path) -> None:
        # stop() racing a slow resolver must never spawn an unmanaged proc.
        gate = threading.Event()

        def slow_resolver(url: str) -> str:
            gate.wait(timeout=_WAIT_S)
            return "http://x/live"

        spawned: list[FakeProc] = []

        def spawn(url: str):
            spawned.append(FakeProc())
            return spawned[-1]

        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=slow_resolver, spawn=spawn)
        starter = threading.Thread(target=cap.start)
        starter.start()
        # Let the resolver block, then stop() before it returns.
        time.sleep(0.05)
        cap.stop()
        gate.set()
        starter.join(timeout=_WAIT_S)
        assert spawned == []


class TestTailerCommit:
    """A segment is committed once it is provably closed: N+1 exists, or
    ffmpeg is gone. Size stability is NOT closure (live HLS arrives in
    bursts, so a half-written file holds still between them)."""

    def test_segment_committed_once_its_successor_exists(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        cap = make_capture(buffer, resolver=lambda u: "http://x", spawn=lambda u: FakeProc())
        try:
            cap.start()
            seed_segment(buffer, 0)
            time.sleep(0.2)
            assert buffer.segments == []  # newest file: still being written
            seed_segment(buffer, 1)
            assert wait_until(lambda: len(buffer.segments) >= 1)
            assert [s.index for s in buffer.segments] == [0]
        finally:
            cap.stop()

    def test_stable_but_unfinished_segment_is_not_committed(self, tmp_path: Path) -> None:
        # The live failure: a segment written in bursts holds its size
        # still between bursts. However long it holds still, as the newest
        # file of a running ffmpeg it is not closed.
        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda u: "http://x")
        cap._proc = FakeProc(alive_s=30)
        path = cap.buffer.directory / segment_name(0)
        path.write_bytes(b"S" * 100)
        for _ in range(5):
            assert cap._poll_once() is False
        assert cap.buffer.segments == []

        path.write_bytes(b"S" * 400)  # the next burst lands
        (cap.buffer.directory / segment_name(1)).write_bytes(b"S" * 10)  # muxer moved on
        assert cap._poll_once() is True
        # Committed with the size re-read at commit time — the whole file.
        assert [(s.index, s.size_bytes) for s in cap.buffer.segments] == [(0, 400)]

    def test_final_segment_is_committed_once_ffmpeg_is_gone(self, tmp_path: Path) -> None:
        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda u: "http://x")
        (cap.buffer.directory / segment_name(0)).write_bytes(b"S" * 50)
        assert cap._poll_once() is False
        assert cap._poll_once(final=True) is True
        assert [s.index for s in cap.buffer.segments] == [0]

    def test_zero_byte_file_is_never_committed(self, tmp_path: Path) -> None:
        # A zero-byte .ts is a muxer error artifact, not a segment.
        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda u: "http://x")
        (cap.buffer.directory / segment_name(0)).write_bytes(b"")
        cap._poll_once(final=True)
        cap._poll_once(final=True)
        assert cap.buffer.segments == []
        # ...and it does not block the sequence behind it.
        (cap.buffer.directory / segment_name(1)).write_bytes(b"S" * 10)
        cap._poll_once(final=True)
        assert [s.index for s in cap.buffer.segments] == [1]

    def test_files_outside_the_expected_sequence_are_never_committed(self, tmp_path: Path) -> None:
        # Stale leftovers (a previous ffmpeg's numbering) must not be
        # adopted: committing seg 950 would make every new segment look
        # out of order — the "blind after restart" failure.
        buffer = RingBuffer(tmp_path / "buf")
        for i in range(950, 953):
            seed_segment(buffer, i)
        cap = make_capture(buffer, resolver=lambda u: "http://x")
        cap._next_index = 0
        cap._poll_once(final=True)
        assert buffer.segments == []
        seed_segment(buffer, 0)
        seed_segment(buffer, 1)
        cap._poll_once()
        assert [s.index for s in buffer.segments] == [0]

    def test_commits_arrive_in_order(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        cap = make_capture(buffer, resolver=lambda u: "http://x", spawn=lambda u: FakeProc())
        try:
            cap.start()
            for i in range(5):
                seed_segment(buffer, i)
                # Segment i-1 is closed the moment segment i appears.
                assert wait_until(lambda i=i: len(buffer.segments) >= i)
        finally:
            cap.stop()
        assert [s.index for s in buffer.segments] == [0, 1, 2, 3]
        assert cap.committed == 4

    def test_non_segment_files_are_ignored(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        cap = make_capture(buffer, resolver=lambda u: "http://x", spawn=lambda u: FakeProc())
        try:
            cap.start()
            (buffer.directory / "notes.txt").write_text("hello", encoding="utf-8")
            (buffer.directory / "seg_000000.mp4").write_bytes(b"nope")
            time.sleep(0.3)
            assert buffer.segments == []
        finally:
            cap.stop()

    def test_commit_refusal_does_not_kill_tailing(self, tmp_path: Path) -> None:
        # A refused commit (duplicate from a restart race) logs and moves
        # on; the tailer must keep serving whatever lands next.
        buffer = RingBuffer(tmp_path / "buf")
        cap = make_capture(buffer, resolver=lambda u: "http://x", spawn=lambda u: FakeProc())
        try:
            cap.start()
            seed_segment(buffer, 0)
            seed_segment(buffer, 1)
            assert wait_until(lambda: len(buffer.segments) == 1)
            # Force the failure: register the same index again by hand.
            with pytest.raises(ValueError):
                buffer.commit_segment(0)
            # Tailer still alive and committing *new* files.
            seed_segment(buffer, 2)
            assert wait_until(lambda: len(buffer.segments) == 2)
        finally:
            cap.stop()

    def test_tailer_drains_after_ffmpeg_death(self, tmp_path: Path) -> None:
        # The last segment can hit the directory after the process is gone;
        # the drain grace must give it time to be committed.
        buffer = RingBuffer(tmp_path / "buf")
        cap = make_capture(
            buffer,
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=0.05),
            drain_grace_s=0.5,
        )
        try:
            cap.start()
            seed_segment(buffer, 0)
            assert wait_until(lambda: len(buffer.segments) == 1)
        finally:
            cap.stop()
        assert [s.index for s in buffer.segments] == [0]


class TestEvents:
    def test_ended_emitted_on_clean_exit(self, tmp_path: Path) -> None:
        events: list[tuple[str, str]] = []
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=0.05, rc=0),
            events=events,
        )
        cap.start()
        assert wait_until(lambda: ("ended", "stream ended") in events)
        cap.stop()

    def test_error_emitted_on_dirty_exit(self, tmp_path: Path) -> None:
        events: list[tuple[str, str]] = []
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=0.05, rc=3),
            events=events,
        )
        cap.start()
        assert wait_until(lambda: any(k == "error" and "3" in d for k, d in events))
        cap.stop()

    def test_stop_does_not_emit_ended(self, tmp_path: Path) -> None:
        # A user removal must not look like the broadcaster signing off
        # (the watcher would park the stream in waiting_for_live).
        events: list[tuple[str, str]] = []
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=30),
            events=events,
        )
        cap.start()
        cap.stop()
        time.sleep(0.2)
        assert not any(k == "ended" for k, _ in events)
        assert not any(k == "error" for k, _ in events)

    def test_callback_crash_is_swallowed(self, tmp_path: Path) -> None:
        def bad_on_event(kind: str, detail: str) -> None:
            raise RuntimeError("callback bug")

        cap = StreamCapture(
            "https://twitch.tv/x",
            RingBuffer(tmp_path / "buf"),
            on_event=bad_on_event,
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=0.05),
            poll_s=0.02,
            drain_grace_s=0.1,
        )
        cap.start()  # must not raise despite the exploding callback
        assert wait_until(lambda: cap._waiter is not None and not cap._waiter.is_alive())
        cap.stop()


class TestStop:
    """stop() is the cancellation contract: idempotent, tree-killing, fast."""

    def test_stop_is_idempotent(self, tmp_path: Path) -> None:
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=30),
        )
        cap.start()
        cap.stop()
        cap.stop()  # second call returns immediately
        assert cap.stopped

    def test_stop_kills_process_tree(self, tmp_path: Path, monkeypatch) -> None:
        killed: list[int] = []
        monkeypatch.setattr(
            "autoclip.live.capture.kill_process_tree", lambda pid: killed.append(pid)
        )
        proc = FakeProc(alive_s=30)
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x",
            spawn=lambda u: proc,
        )
        try:
            cap.start()
            assert cap.running
        finally:
            cap.stop()
        assert killed == [proc.pid]
        assert not cap.running

    def test_stop_before_start_is_safe(self, tmp_path: Path) -> None:
        cap = make_capture(RingBuffer(tmp_path / "buf"))
        cap.stop()
        assert cap.stopped

    def test_stop_from_any_thread_completes(self, tmp_path: Path) -> None:
        # The watcher may call stop() from its own thread or a callback;
        # neither may deadlock on joining the capture's threads. joining
        # yourself is the classic self-join hang, so the stopper thread
        # stands in for "not the current thread".
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=30),
        )
        cap.start()

        def stopper() -> None:
            cap.stop()

        t = threading.Thread(target=stopper)
        t.start()
        t.join(timeout=_WAIT_S)
        assert not t.is_alive()
        assert cap.stopped

    def test_threads_do_not_outlive_stop(self, tmp_path: Path) -> None:
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "http://x",
            spawn=lambda u: FakeProc(alive_s=30),
        )
        cap.start()
        threads = [t for t in (cap._tailer, cap._waiter) if t is not None]
        cap.stop()
        assert all(not t.is_alive() for t in threads)


class _ChattyProc(FakeProc):
    """A FakeProc with a real stderr pipe that floods like a noisy ffmpeg."""

    def __init__(self, payload: bytes, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        import io

        self.stderr = io.BufferedReader(io.BytesIO(payload))


class TestCaptureRobustness:
    """Freezes and orphans: the failure modes a 24/7 capture must survive."""

    def test_stderr_is_drained_and_its_tail_kept(self, tmp_path: Path) -> None:
        # An unread PIPE blocks ffmpeg once the OS buffer fills; the reader
        # must consume it, and keep the last lines for error events.
        noise = b"".join(f"warning {i}\n".encode() for i in range(5000)) + b"fatal: 403\n"
        proc = _ChattyProc(noise, alive_s=30)
        cap = make_capture(
            RingBuffer(tmp_path / "buf"), resolver=lambda u: "x", spawn=lambda u: proc
        )
        try:
            cap.start()
            assert wait_until(lambda: "fatal: 403" in cap.stderr_tail)
        finally:
            cap.stop()

    def test_error_event_carries_the_stderr_tail(self, tmp_path: Path) -> None:
        events: list[tuple[str, str]] = []
        proc = _ChattyProc(b"Server returned 403 Forbidden\n", alive_s=0.1, rc=1)
        cap = make_capture(
            RingBuffer(tmp_path / "buf"),
            resolver=lambda u: "x",
            spawn=lambda u: proc,
            events=events,
        )
        try:
            cap.start()
            assert wait_until(lambda: any(k == "error" for k, _ in events))
        finally:
            cap.stop()
        detail = next(d for k, d in events if k == "error")
        assert "code 1" in detail and "403 Forbidden" in detail

    def test_stalled_capture_is_killed_and_reported(self, tmp_path: Path) -> None:
        # ffmpeg alive but nothing new for ~3 segments: kill + "error" so
        # the watcher reconnects instead of sitting on a frozen capture.
        events: list[tuple[str, str]] = []
        proc = FakeProc(alive_s=30, kill_rc=1)
        buffer = RingBuffer(tmp_path / "buf")
        cap = StreamCapture(
            "https://twitch.tv/x",
            buffer,
            on_event=lambda k, d: events.append((k, d)),
            resolver=lambda u: "x",
            spawn=lambda u: proc,
            poll_s=0.02,
            drain_grace_s=0.05,
            stall_timeout_s=0.3,
            first_segment_timeout_s=0.3,
        )
        try:
            cap.start()
            assert wait_until(lambda: any(k == "error" for k, _ in events))
        finally:
            cap.stop()
        assert proc.killed.is_set()
        detail = next(d for k, d in events if k == "error")
        assert "stalled" in detail

    def test_steady_segments_keep_the_watchdog_quiet(self, tmp_path: Path) -> None:
        events: list[tuple[str, str]] = []
        proc = FakeProc(alive_s=30)
        buffer = RingBuffer(tmp_path / "buf")
        cap = StreamCapture(
            "https://twitch.tv/x",
            buffer,
            on_event=lambda k, d: events.append((k, d)),
            resolver=lambda u: "x",
            spawn=lambda u: proc,
            poll_s=0.02,
            stall_timeout_s=0.5,
            first_segment_timeout_s=0.5,
        )
        try:
            cap.start()
            for i in range(8):
                seed_segment(buffer, i)
                time.sleep(0.1)
            assert not proc.killed.is_set()
        finally:
            cap.stop()
        assert not any(k == "error" for k, _ in events)
        assert cap.committed == 7

    def test_stop_during_popen_never_orphans_ffmpeg(self, tmp_path: Path) -> None:
        # stop() landing between the stop-flag check and Popen returning
        # used to leave a live process nobody managed.
        spawned: list[FakeProc] = []
        in_popen = threading.Event()

        def slow_spawn(url: str) -> FakeProc:
            in_popen.set()
            time.sleep(0.3)  # a slow Popen (antivirus scan, loaded box)
            spawned.append(FakeProc(alive_s=30))
            return spawned[-1]

        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda u: "x", spawn=slow_spawn)
        starter = threading.Thread(target=cap.start)
        starter.start()
        assert in_popen.wait(timeout=_WAIT_S)
        cap.stop()  # lands while Popen is still running
        starter.join(timeout=_WAIT_S)
        assert spawned, "the spawn should have completed"
        assert spawned[0].killed.is_set(), "the process spawned during stop() was orphaned"
        assert cap._tailer is None and cap._waiter is None

    def test_argv_bounds_blocking_reads(self, tmp_path: Path, monkeypatch) -> None:
        argv: list[str] = []
        monkeypatch.setattr("autoclip.live.capture._ffmpeg_binary", lambda: "ffmpeg")
        monkeypatch.setattr("autoclip.live.capture._new_process_group_flags", lambda: 0)
        monkeypatch.setattr(
            subprocess, "Popen", lambda command, **kwargs: argv.extend(command) or FakeProc()
        )
        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda u: "x")
        cap._spawn_ffmpeg("http://media/x.m3u8")
        cap.stop()
        assert "-rw_timeout" in argv
        assert argv.index("-rw_timeout") < argv.index("-i")
        assert int(argv[argv.index("-rw_timeout") + 1]) >= 1_000_000


class TestRealArgv:
    """The frozen ffmpeg argv — pinned so a refactor can't silently drift it."""

    def test_spawn_argv_contains_frozen_flags(self, tmp_path: Path, monkeypatch) -> None:
        argv: list[str] = []
        monkeypatch.setattr("autoclip.live.capture._ffmpeg_binary", lambda: "ffmpeg")
        monkeypatch.setattr("autoclip.live.capture._new_process_group_flags", lambda: 0)
        monkeypatch.setattr(
            subprocess,
            "Popen",
            lambda command, **kwargs: argv.extend(command) or FakeProc(),
        )
        buffer = RingBuffer(tmp_path / "buf")
        cap = make_capture(buffer, resolver=lambda u: "http://media/x.m3u8")
        cap._spawn_ffmpeg("http://media/x.m3u8")
        cap.stop()
        joined = " ".join(argv)
        # Work-order-pinned flags: segment muxer (NOT HLS), copy codecs, all
        # streams, 6s segments, one continuous clock, faststart.
        assert "-f" in argv and argv[argv.index("-f") + 1] == "segment"
        assert "-c" in argv and argv[argv.index("-c") + 1] == "copy"
        assert "-map" in argv and argv[argv.index("-map") + 1] == "0"
        assert argv[argv.index("-segment_time") + 1] == "6"
        # No per-segment reset: it made audio drift at every join.
        assert "-reset_timestamps" not in argv
        assert "-segment_start_number" in argv
        assert argv[argv.index("-movflags") + 1] == "+faststart"
        assert "-nostdin" in argv
        # The output pattern targets the buffer dir with the frozen template.
        pattern = str(buffer.directory / "seg_%06d.ts").replace("\\", "/")
        assert pattern in joined.replace("\\", "/")
        # And NOT the HLS muxer — the work order pins the segment muxer
        # because .ts files must be plain cut-mid-GOP-resilient fragments.
        assert argv.count("hls") == 0

    def test_start_number_continues_past_committed(self, tmp_path: Path) -> None:
        # Reconnects must not renumber from 0: index → seconds is global,
        # and a restart that renumbers would corrupt every downstream map.
        buffer = RingBuffer(tmp_path / "buf")
        for i in range(3):
            path = buffer.next_segment_path(i)
            path.write_bytes(b"x" * 10)
            buffer.commit_segment(i)
        cap = make_capture(buffer, resolver=lambda u: "http://x")
        assert cap._next_start_number() == 3

    def test_start_number_zero_when_empty(self, tmp_path: Path) -> None:
        cap = make_capture(RingBuffer(tmp_path / "buf"), resolver=lambda u: "http://x")
        assert cap._next_start_number() == 0


class TestResolutionFunctions:
    """The module-level resolver chain: streamlink first, yt-dlp fallback."""

    def test_both_refusals_raise_offline_with_markers(self, monkeypatch) -> None:
        def refuse(url: str) -> str:
            raise CaptureError("No playable streams found", offline=True)

        monkeypatch.setattr("autoclip.live.capture._resolve_with_streamlink", refuse)
        monkeypatch.setattr("autoclip.live.capture._resolve_with_ytdlp", refuse)
        with pytest.raises(CaptureError) as excinfo:
            resolve_stream_url("https://twitch.tv/nobody")
        assert excinfo.value.offline is True
        assert "no playable streams" in str(excinfo.value).lower()

    def test_streamlink_success_short_circuits_ytdlp(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            "autoclip.live.capture._resolve_with_streamlink",
            lambda url: calls.append("sl") or "http://s/live",
        )
        monkeypatch.setattr(
            "autoclip.live.capture._resolve_with_ytdlp",
            lambda url: calls.append("ytdlp") or "http://y/live",
        )
        assert resolve_stream_url("https://x.example/live") == "http://s/live"
        assert calls == ["sl"]

    def test_streamlink_failure_falls_back_to_ytdlp(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "autoclip.live.capture._resolve_with_streamlink",
            lambda url: (_ for _ in ()).throw(CaptureError("plugin exploded")),
        )
        monkeypatch.setattr(
            "autoclip.live.capture._resolve_with_ytdlp",
            lambda url: "http://y/live",
        )
        assert resolve_stream_url("https://youtube.example/watch?v=x") == "http://y/live"

    def test_hard_failure_is_not_offline(self, monkeypatch) -> None:
        # A timeout/hang is a connect failure (gets backoff), not "offline".
        monkeypatch.setattr(
            "autoclip.live.capture._resolve_with_streamlink",
            lambda url: (_ for _ in ()).throw(CaptureError("socket timed out")),
        )
        monkeypatch.setattr(
            "autoclip.live.capture._resolve_with_ytdlp",
            lambda url: (_ for _ in ()).throw(CaptureError("connection refused")),
        )
        with pytest.raises(CaptureError) as excinfo:
            resolve_stream_url("https://x.example/live")
        assert excinfo.value.offline is False

    def test_streamlink_stream_object_is_used(self, monkeypatch) -> None:
        # Verified against the real streamlink 8.6.3 API: streams() returns
        # {'best': HLSStream, ...} whose .url is the manifest URL ffmpeg
        # ingests. The imports under test are function-local, so inject a
        # fake `streamlink` package into sys.modules.
        import sys
        import types

        class FakeStream:
            url = "http://real/hls.m3u8"

        fake_session = types.SimpleNamespace(streams=lambda url: {"best": FakeStream()})
        streamlink_pkg = types.ModuleType("streamlink")
        streamlink_session = types.ModuleType("streamlink.session")
        streamlink_session.Streamlink = lambda: fake_session
        streamlink_exceptions = types.ModuleType("streamlink.exceptions")
        streamlink_exceptions.NoStreamsError = type("NoStreamsError", (Exception,), {})
        streamlink_exceptions.PluginError = type("PluginError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "streamlink", streamlink_pkg)
        monkeypatch.setitem(sys.modules, "streamlink.session", streamlink_session)
        monkeypatch.setitem(sys.modules, "streamlink.exceptions", streamlink_exceptions)

        assert resolve_stream_url("https://twitch.tv/x") == "http://real/hls.m3u8"


class TestKillProcessTree:
    def test_windows_calls_taskkill(self, monkeypatch) -> None:
        import sys

        commands: list[list[str]] = []
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kwargs: commands.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0),
        )
        kill_process_tree(1234)
        assert commands == [["taskkill", "/PID", "1234", "/T", "/F"]]

    def test_posix_signals_process_group(self, monkeypatch) -> None:
        # On Windows the whole POSIX branch is structurally untestable (os
        # is frozen there and exposes no killpg), so drive the branch the
        # way an embedder would: patch the two things the branch reads —
        # sys.platform and the signal constants — against the REAL os module
        # semantics via a tiny indirection-free smoke check. Skip instead
        # of faking os internals that don't exist on this interpreter.
        import sys

        if sys.platform == "win32" and not hasattr(__import__("os"), "killpg"):
            pytest.skip("POSIX killpg branch cannot run on this Windows interpreter")

        import os
        import signal

        sent: list[tuple[int, int]] = []
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append((pid, sig)))
        kill_process_tree(777)
        assert sent[0] == (777, signal.SIGTERM)


def test_ytdlp_resolve_retries_over_ipv4_on_bot_check(monkeypatch):
    """YouTube's "confirm you're not a bot" page (VPN IPv6) -> retry over IPv4."""
    from autoclip.live import capture

    seen: list = []

    def fake_extract(yt_dlp, options, url):
        seen.append(options.get("source_address"))
        if options.get("source_address") != "0.0.0.0":
            raise RuntimeError("ERROR: [youtube] x: Sign in to confirm you're not a bot.")
        return {"url": "https://manifest.example/hls.m3u8"}

    monkeypatch.setattr(capture, "_extract", fake_extract)
    assert capture._resolve_with_ytdlp("https://www.youtube.com/watch?v=x") == (
        "https://manifest.example/hls.m3u8"
    )
    assert seen == [None, "0.0.0.0"]


def test_ytdlp_resolve_other_errors_not_retried(monkeypatch):
    from autoclip.live import capture

    calls: list = []

    def fake_extract(yt_dlp, options, url):
        calls.append(1)
        raise RuntimeError("This live event has ended.")

    monkeypatch.setattr(capture, "_extract", fake_extract)
    with pytest.raises(capture.CaptureError, match="has ended"):
        capture._resolve_with_ytdlp("https://www.youtube.com/watch?v=x")
    assert len(calls) == 1


def test_ytdlp_resolve_falls_back_to_mweb_client(monkeypatch):
    """Bot check on both routes with the web client -> IPv4 + mweb client."""
    from autoclip.live import capture

    seen: list = []

    def fake_extract(yt_dlp, options, url):
        client = (options.get("extractor_args") or {}).get("youtube", {}).get("player_client")
        seen.append((options.get("source_address"), client))
        if client != ["mweb"]:
            raise RuntimeError("Sign in to confirm you're not a bot")
        return {"url": "https://manifest.example/hls.m3u8"}

    monkeypatch.setattr(capture, "_extract", fake_extract)
    assert capture._resolve_with_ytdlp("https://www.youtube.com/watch?v=x").endswith("hls.m3u8")
    assert seen == [(None, None), ("0.0.0.0", None), ("0.0.0.0", ["mweb"])]
