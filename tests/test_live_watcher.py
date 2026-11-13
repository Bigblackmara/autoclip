"""Unit tests for the Live Watch per-stream watcher (A8).

Everything is faked: no network, no streamlink, no whisper, no ffmpeg. Each
test injects a minimal ``WatcherDeps`` whose contracts match the landed
sibling bodies (capture on_event(kind, detail), run_chat_reader, judge).

Async tests rely on this suite's asyncio_mode = "auto".
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from autoclip.live import store
from autoclip.live.judge import LiveJudgement
from autoclip.live.models import LiveStreamRow
from autoclip.live.settings import LiveSettings
from autoclip.live.signals import DetectorSignal, SignalKind, SignalWindow
from autoclip.live.states import StreamState
from autoclip.live.watcher import (
    BACKOFF_CAP_S,
    DEFAULT_MAX_CONNECT_ATTEMPTS,
    ClipWorkItem,
    StreamWatcher,
    WatcherDeps,
    _looks_offline,
    stream_channel,
)

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeBuffer:
    """RingBuffer stand-in: extract writes a stub file, prune counts calls."""

    def __init__(self) -> None:
        self.segments: list[Any] = []
        self.extract_calls: list[tuple[float, float]] = []
        self.prunes = 0
        self.closed = False
        self.epochs = 0

    def begin_epoch(self) -> int:
        self.epochs += 1
        return self.epochs

    def prune(self, *_: Any, **__: Any) -> int:
        self.prunes += 1
        return 0

    def extract(self, start_s: float, end_s: float, target: Path) -> Path:
        self.extract_calls.append((start_s, end_s))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-window")
        return target

    def close(self) -> None:
        self.closed = True


class FakeCapture:
    """StreamCapture stand-in: on_event(kind, detail), start() may raise."""

    def __init__(self, buffer: Any, on_event: Any, *, url: str = "") -> None:
        self.buffer = buffer
        self.on_event = on_event
        self.url = url
        self.started = False
        self.stop_calls = 0
        self.fail_start: BaseException | None = None

    def start(self) -> None:
        if self.fail_start is not None:
            raise self.fail_start
        self.started = True
        self.on_event("started", "fake capture started")

    def stop(self) -> None:
        self.stop_calls += 1

    def emit(self, kind: str, detail: str = "") -> None:
        self.on_event(kind, detail)


class FakeAggregator:
    def __init__(self) -> None:
        self.signals: list[DetectorSignal] = []

    def add(self, signal: DetectorSignal) -> None:
        self.signals.append(signal)

    def evaluate(self, window_s: float = 30.0) -> SignalWindow:
        """Roll the last ``window_s`` of signals into a window (A4 shape)."""
        times = [s.t_s for s in self.signals]
        end = max(times) if times else 0.0
        return SignalWindow(
            start_s=max(0.0, end - window_s),
            end_s=end,
            signals=[s for s in self.signals if end - window_s <= s.t_s <= end],
        )


@dataclass
class SimpleDecision:
    fire: bool = False


class FakeTrigger:
    def __init__(self) -> None:
        self.windows: list[SignalWindow] = []
        self.fire = False

    def decide(self, window: SignalWindow) -> Any:
        self.windows.append(window)
        return SimpleDecision(fire=self.fire)


class AlwaysBucket:
    """TokenBucket that succeeds; replaces A4's pending stub."""

    def try_acquire(self, n: float = 1.0) -> bool:
        return True


class NeverBucket:
    def try_acquire(self, n: float = 1.0) -> bool:
        return False


class AlwaysCooldown:
    def can_trigger(self, stream_id: str, **_: Any) -> bool:
        return True

    def mark_clip(self, stream_id: str, **_: Any) -> None:
        pass


@dataclass
class FakeWord:
    text: str
    start: float
    end: float


@dataclass
class FakeTranscript:
    words: list = field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path


def make_transcript(n: int = 3) -> FakeTranscript:
    return FakeTranscript(
        words=[
            FakeWord(text=f"w{i}", start=float(i * 2.0), end=float(i * 2.0 + 1.5)) for i in range(n)
        ]
    )


def make_deps(
    buffer: FakeBuffer | None = None,
    **overrides: Any,
) -> tuple[WatcherDeps, dict[str, Any]]:
    """Build WatcherDeps around fakes; returns (deps, parts registry)."""
    buffer = buffer if buffer is not None else FakeBuffer()
    created: list[FakeCapture] = []

    def capture_factory(buf: Any, on_event: Any) -> FakeCapture:
        capture = FakeCapture(buf, on_event)
        created.append(capture)
        return capture

    async def default_judge(label: str, words: list, signals: dict) -> LiveJudgement:
        return LiveJudgement(score=90, title="T", hook="H", reason="R")

    defaults: dict[str, Any] = {
        "capture_factory": capture_factory,
        "buffer_factory": lambda: buffer,
        "aggregator_factory": FakeAggregator,
        "trigger": FakeTrigger(),
        "cooldown": AlwaysCooldown(),
        "judge_minute": AlwaysBucket(),
        "judge_hour": AlwaysBucket(),
        "judge_day": AlwaysBucket(),
        "transcribe": lambda path: make_transcript(),
        "audio_extract": lambda source, dest: dest.parent.mkdir(parents=True, exist_ok=True),
        "judge": default_judge,
        "enqueue_clip": lambda item: True,
        "may_capture": lambda: True,
        "publish": lambda *a, **k: None,
        "sleep": immediate_sleep(),
        "chat_runner": lambda *a, **k: False,
        "max_connect_attempts": DEFAULT_MAX_CONNECT_ATTEMPTS,
    }
    # Test-ergonomic aliases → actual WatcherDeps field names.
    if "enqueue" in overrides:
        overrides["enqueue_clip"] = overrides.pop("enqueue")
    if "semaphore" in overrides:
        overrides["judge_semaphore"] = overrides.pop("semaphore")
    defaults.update(overrides)
    deps = WatcherDeps(**defaults)
    parts = {"captures": created, "buffer": buffer, "trigger": defaults["trigger"]}
    return deps, parts


def immediate_sleep() -> Any:
    """A sleep seam that yields control but never actually waits."""

    async def _sleep(seconds: float) -> None:
        await asyncio.sleep(0)

    return _sleep


async def wait_until(predicate: Any, timeout: float = 5.0) -> None:
    """Await ``predicate()`` true or fail; polls on the running loop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"condition not met within {timeout}s")


async def stop_watcher(watcher: StreamWatcher, task: asyncio.Task) -> None:
    """The manager's teardown pattern: cancel the task, then stop()."""
    watcher.stop()
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


@pytest.fixture
def primed_db(initialised_db: int) -> int:
    """An initialized DB with the parent live_streams row for clip FKs.

    The real store enforces live_clips.stream_id → live_streams.id; the
    manager (not the watcher) creates stream rows, so candidate tests must
    register their stream first or every create_clip raises IntegrityError.
    """
    store.upsert_stream(LiveStreamRow(id="s-cand", url="https://twitch.tv/x", label="x"))
    for sid in (
        "s-rej",
        "s-budget",
        "s-full",
        "s-busy",
        "s-cancel",
        "s-row",
        "s-xfail",
    ):
        store.upsert_stream(LiveStreamRow(id=sid, url="url", label=""))
    return initialised_db


# --------------------------------------------------------------------------
# Module helpers
# --------------------------------------------------------------------------


def test_stream_channel_naming() -> None:
    assert stream_channel("abc") == "live:abc"


def test_looks_offline_markers() -> None:
    class Captureish(Exception):
        pass

    assert _looks_offline(Captureish("No playable streams found for this url")) is True
    assert _looks_offline(Captureish("connection refused")) is False

    class Offline(Exception):
        offline = True

    assert _looks_offline(Offline("anything")) is True


# --------------------------------------------------------------------------
# run() lifecycle + crash isolation
# --------------------------------------------------------------------------


async def test_run_reaches_watching_when_capture_starts() -> None:
    deps, _ = make_deps()
    watcher = StreamWatcher("s-live", "https://twitch.tv/x", "x", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        assert watcher.buffer is not None
    finally:
        await stop_watcher(watcher, task)


async def test_run_parks_in_error_when_buffer_fails() -> None:
    def broken_factory() -> Any:
        raise RuntimeError("disk on fire")

    deps, _ = make_deps(buffer_factory=broken_factory)
    watcher = StreamWatcher("s-err", "url", "", LiveSettings(), deps=deps)
    await watcher.run()  # returns: terminal error, never raises
    assert watcher.state is StreamState.ERROR
    assert "disk on fire" in watcher.state_detail


async def test_watcher_crash_is_isolated_to_error_state() -> None:
    """A supervisor-loop exception lands THIS stream in error, not the loop.

    _maybe_prune/_drain_signals/_connect all guard their own subsystems, so
    the crash path is exercised by making one loop step raise — the same
    path any uncaught loop error takes — and expecting run() to convert it
    into a terminal error for THIS stream only.
    """
    deps, _ = make_deps()
    watcher = StreamWatcher("s-crash", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)

        def boom() -> None:
            raise RuntimeError("loop exploded")

        watcher._drain_signals = boom  # type: ignore[method-assign]
        await wait_until(lambda: watcher.state is StreamState.ERROR)
        assert "watcher crashed" in watcher.state_detail
        assert "loop exploded" in watcher.state_detail
    finally:
        await stop_watcher(watcher, task)


async def test_stop_cancels_cleanly_and_closes_buffer_only_when_asked() -> None:
    deps, _ = make_deps()
    watcher = StreamWatcher("s-stop", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        buffer = watcher.buffer
        watcher.stop()  # pause semantics: buffer kept
        await asyncio.sleep(0)
        assert buffer is not None and not buffer.closed
        watcher.stop(close_buffer=True)  # removal semantics: buffer closed
        assert buffer.closed
    finally:
        await stop_watcher(watcher, task)


# --------------------------------------------------------------------------
# Reconnect + backoff
# --------------------------------------------------------------------------


async def test_capture_error_event_sets_reconnecting_with_backoff() -> None:
    deps, parts = make_deps()
    watcher = StreamWatcher("s-rc1", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        parts["captures"][0].emit("error", "ffmpeg exited with code 1")
        await wait_until(lambda: watcher.state is StreamState.RECONNECTING)
        assert watcher._backoff_wait_s > 0
    finally:
        await stop_watcher(watcher, task)


def test_backoff_grows_exponentially_and_caps_at_120s() -> None:
    deps, _ = make_deps()
    watcher = StreamWatcher("s-bk", "url", "", LiveSettings(), deps=deps)
    # Slot doubles per attempt: 2, 4, 8 ... all within ±20% jitter.
    for attempt in range(1, 7):
        watcher._connect_attempts = attempt
        wait = watcher._next_backoff()
        expected = min(2.0 * 2 ** (attempt - 1), BACKOFF_CAP_S)
        assert wait == pytest.approx(expected, rel=0.21)
    # Beyond the cap the slot stays at 120s (±20%).
    watcher._connect_attempts = 50
    for _ in range(20):
        wait = watcher._next_backoff()
        assert BACKOFF_CAP_S * 0.79 <= wait <= BACKOFF_CAP_S * 1.21


async def test_repeated_connect_failures_terminate_in_error() -> None:
    def failing_capture(buf: Any, on_event: Any) -> FakeCapture:
        capture = FakeCapture(buf, on_event)
        capture.fail_start = RuntimeError("resolution blew up")
        return capture

    deps, _ = make_deps(capture_factory=failing_capture, max_connect_attempts=3)
    watcher = StreamWatcher("s-dead", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.ERROR)
        assert "3 times" in watcher.state_detail
    finally:
        await stop_watcher(watcher, task)


async def test_offline_resolution_parks_waiting_for_live_not_error() -> None:
    def offline_capture(buf: Any, on_event: Any) -> FakeCapture:
        capture = FakeCapture(buf, on_event)
        capture.fail_start = RuntimeError("No playable streams found")
        return capture

    states: list[StreamState] = []
    deps, _ = make_deps(capture_factory=offline_capture)
    watcher = StreamWatcher("s-off", "url", "", LiveSettings(), deps=deps)
    watcher._on_state = lambda sid, old, new, detail: states.append(new)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WAITING_FOR_LIVE)
        await asyncio.sleep(0.05)
        # Not-live parks on the offline poll — never an error, and the
        # failure budget must not accumulate across polls.
        assert watcher.state is not StreamState.ERROR
        assert StreamState.ERROR not in states
        assert watcher._connect_attempts == 0
        assert watcher._backoff_wait_s == 0.0
    finally:
        await stop_watcher(watcher, task)


async def test_clean_ended_event_parks_on_offline_poll() -> None:
    deps, parts = make_deps()
    watcher = StreamWatcher("s-end", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        parts["captures"][0].emit("ended", "stream ended")
        await wait_until(lambda: watcher.state is StreamState.WAITING_FOR_LIVE)
        # The offline park cleared the failure budget: a next broadcast
        # starts fresh, not with 1 attempt already spent.
        assert watcher._connect_attempts == 0
        assert watcher._backoff_wait_s == 0.0
    finally:
        await stop_watcher(watcher, task)


async def test_capacity_gate_parks_waiting() -> None:
    deps, _ = make_deps(may_capture=lambda: False)
    watcher = StreamWatcher("s-cap", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WAITING_FOR_LIVE)
        assert "capacity" in watcher.state_detail
    finally:
        await stop_watcher(watcher, task)


# --------------------------------------------------------------------------
# State machine hard edges
# --------------------------------------------------------------------------


def test_illegal_state_jump_is_ignored() -> None:
    deps, _ = make_deps()
    watcher = StreamWatcher("s-jump", "url", "", LiveSettings(), deps=deps)
    watcher.set_state(StreamState.OFF, "", initial=True)
    # off → connecting is illegal (only a fresh watcher start may do it).
    assert watcher.set_state(StreamState.CONNECTING) is False
    assert watcher.state is StreamState.OFF


def test_store_failure_never_kills_state_transition() -> None:
    """The in-memory state is the source of truth; a broken DB can't stop it."""
    import autoclip.live.watcher as watcher_mod

    deps, _ = make_deps()
    watcher = StreamWatcher("s-db", "url", "", LiveSettings(), deps=deps)

    class Exploding:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("db down")

    original = watcher_mod.store
    watcher_mod.store = Exploding()  # type: ignore[assignment]
    try:
        moved = watcher.set_state(StreamState.WATCHING, "live", initial=True)
        assert moved is True
        assert watcher.state is StreamState.WATCHING
    finally:
        watcher_mod.store = original


# --------------------------------------------------------------------------
# Signals → trigger → candidate pipeline
# --------------------------------------------------------------------------


async def test_signal_drain_fires_candidate_and_full_pipeline(primed_db: int) -> None:
    enqueued: list[ClipWorkItem] = []
    judge_calls: list[tuple[str, list, dict]] = []

    async def judge(label: str, words: list, signals: dict) -> LiveJudgement:
        judge_calls.append((label, words, signals))
        return LiveJudgement(score=95, title="Big Moment", hook="h", reason="r")

    deps, parts = make_deps(enqueue=lambda item: enqueued.append(item) or True, judge=judge)
    trigger: FakeTrigger = parts["trigger"]
    trigger.fire = True
    buffer: FakeBuffer = parts["buffer"]
    watcher = StreamWatcher("s-cand", "https://twitch.tv/x", "x", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=120.0, strength=1.5))
        await wait_until(lambda: len(enqueued) == 1)
        # Judge saw the label and the window-derived signals.
        label, words, signals = judge_calls[0]
        assert label == "x"
        assert len(words) == 3
        assert signals.get("audio_energy") == 1.5
        item = enqueued[0]
        assert item.stream_id == "s-cand"
        assert item.clip_start_s == pytest.approx(100.0)  # 120 - pre_roll 20
        assert item.clip_end_s == pytest.approx(130.0)  # 120 + post_roll 10
        assert item.window_bytes == len(b"fake-window")
        # The buffer extracted exactly the pre/post-roll window.
        assert buffer.extract_calls and buffer.extract_calls[0] == (100.0, 130.0)
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
    finally:
        await stop_watcher(watcher, task)


async def test_below_threshold_judgement_rejects_without_enqueue(primed_db: int) -> None:
    enqueued: list[ClipWorkItem] = []

    async def harsh(label: str, words: list, signals: dict) -> LiveJudgement:
        return LiveJudgement(score=10, title="Nope", hook="", reason="bland")

    deps, parts = make_deps(enqueue=lambda item: enqueued.append(item) or True, judge=harsh)
    parts["trigger"].fire = True
    watcher = StreamWatcher("s-rej", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.CHAT_HYPE, t_s=60.0, strength=1.9))
        await wait_until(lambda: watcher.state is StreamState.CLIPPING)
        await wait_until(lambda: watcher.state is not StreamState.CLIPPING)
        # watching → clipping → watching is the legal loop; nothing enqueued.
        assert watcher.state is StreamState.WATCHING
        assert enqueued == []
    finally:
        await stop_watcher(watcher, task)


async def test_judge_budget_exhaustion_rejects_and_publishes(primed_db: int) -> None:
    enqueued: list[ClipWorkItem] = []
    published: list[tuple[str, str, dict]] = []

    def publish(channel: str, event_type: str, data: dict | None = None) -> None:
        published.append((channel, event_type, data or {}))

    deps, parts = make_deps(
        enqueue=lambda item: enqueued.append(item) or True,
        judge_minute=NeverBucket(),
        publish=publish,
    )
    parts["trigger"].fire = True
    watcher = StreamWatcher("s-budget", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=50.0, strength=1.0))
        await wait_until(lambda: any(e[1] == "clip_rejected" for e in published))
        rejected = [e for e in published if e[1] == "clip_rejected"][0]
        assert "judge minute budget exhausted" in rejected[2]["reason"]
        assert enqueued == []
    finally:
        await stop_watcher(watcher, task)


async def test_queue_full_drop_marks_rejected(primed_db: int) -> None:
    published: list[tuple[str, str, dict]] = []

    def publish(channel: str, event_type: str, data: dict | None = None) -> None:
        published.append((channel, event_type, data or {}))

    deps, parts = make_deps(enqueue=lambda item: False, publish=publish)
    parts["trigger"].fire = True
    watcher = StreamWatcher("s-full", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=50.0, strength=1.0))
        await wait_until(lambda: any(e[1] == "clip_dropped" for e in published))
        dropped = [e for e in published if e[1] == "clip_dropped"][0]
        assert dropped[2]["reason"] == "queue full or disk pressure"
    finally:
        await stop_watcher(watcher, task)


async def test_candidate_busy_skips_second_trigger(primed_db: int) -> None:
    enqueued: list[ClipWorkItem] = []
    release = asyncio.Event()

    async def slow_judge(label: str, words: list, signals: dict) -> LiveJudgement:
        await release.wait()
        return LiveJudgement(score=95, title="T", hook="h", reason="r")

    deps, parts = make_deps(enqueue=lambda item: enqueued.append(item) or True, judge=slow_judge)
    parts["trigger"].fire = True
    watcher = StreamWatcher("s-busy", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=10.0, strength=1.0))
        await wait_until(lambda: watcher.state is StreamState.CLIPPING)
        # Second trigger while the first is mid-flight must not spawn another.
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=12.0, strength=1.0))
        await asyncio.sleep(0.05)
        release.set()
        await wait_until(lambda: len(enqueued) == 1)
        await asyncio.sleep(0.05)
        assert len(enqueued) == 1
    finally:
        release.set()
        await stop_watcher(watcher, task)


async def test_judge_semaphore_limits_concurrency() -> None:
    """The manager-injected semaphore caps simultaneous judge calls."""
    concurrent = 0
    peak = 0
    lock = asyncio.Lock()

    async def probe(label: str, words: list, signals: dict) -> LiveJudgement:
        nonlocal concurrent, peak
        async with lock:
            concurrent += 1
            peak = max(peak, concurrent)
        await asyncio.sleep(0.05)
        async with lock:
            concurrent -= 1
        return LiveJudgement(score=95, title="T", hook="", reason="")

    deps, _ = make_deps(judge=probe, semaphore=asyncio.Semaphore(1))
    watcher = StreamWatcher("s-sem", "url", "", LiveSettings(), deps=deps)
    # Two concurrent judge calls through the watcher's semaphore must
    # never overlap.
    results = await asyncio.gather(watcher._call_judge([], {}), watcher._call_judge([], {}))
    assert all(r.score == 95 for r in results)
    assert peak == 1


def test_pending_circuit_fail_closes() -> None:
    class PendingCircuit:
        def can_trigger(self, stream_id: str, **_: Any) -> bool:
            raise NotImplementedError("A4 pending")

    deps, _ = make_deps(cooldown=PendingCircuit())
    watcher = StreamWatcher("s-failclosed", "url", "", LiveSettings(), deps=deps)
    assert watcher._cooldown_ok() is False


# --------------------------------------------------------------------------
# Signal worker threads
# --------------------------------------------------------------------------


async def test_audio_worker_thread_runs_and_stops() -> None:
    started = threading.Event()

    def audio_start(buffer: Any, settings: Any, on_signal: Any, stop_event: Any) -> None:
        started.set()
        stop_event.wait(timeout=5)

    deps, _ = make_deps(audio_start=audio_start)
    watcher = StreamWatcher("s-audio", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        # Poll on the loop: a blocking Event.wait would starve the watcher
        # task and deadlock the test.
        await wait_until(started.is_set)
    finally:
        await stop_watcher(watcher, task)


async def test_chat_worker_thread_runs_and_stops() -> None:
    started = threading.Event()

    def chat_runner(url: str, settings: Any, **kwargs: Any) -> bool:
        started.set()
        kwargs["stop_event"].wait(timeout=5)
        return True

    deps, _ = make_deps(chat_runner=chat_runner)
    watcher = StreamWatcher("s-chat", "https://twitch.tv/example", "x", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(started.is_set)
    finally:
        await stop_watcher(watcher, task)


async def test_chat_worker_not_started_for_non_twitch() -> None:
    started = threading.Event()

    def chat_runner(url: str, settings: Any, **kwargs: Any) -> bool:
        started.set()
        return True

    deps, _ = make_deps(chat_runner=chat_runner)
    watcher = StreamWatcher("s-yt", "https://youtube.com/watch?v=x", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await asyncio.sleep(0.1)
        assert not started.is_set(), "chat started for a non-Twitch URL"
    finally:
        await stop_watcher(watcher, task)


async def test_signal_workers_survive_a_crashing_audio_worker() -> None:
    def exploding_audio(buffer: Any, settings: Any, on_signal: Any, stop_event: Any) -> None:
        raise RuntimeError("audio worker exploded")

    deps, _ = make_deps(audio_start=exploding_audio)
    watcher = StreamWatcher("s-acrash", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        # The audio thread dying must not surface in the supervisor.
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        assert watcher._audio_thread is not None and not watcher._audio_thread.is_alive()
    finally:
        await stop_watcher(watcher, task)


# --------------------------------------------------------------------------
# Teardown of cancelled candidates
# --------------------------------------------------------------------------


async def test_cancelled_candidate_unwinds_and_leaves_no_clipping_state(primed_db: int) -> None:
    gate = asyncio.Event()

    async def stuck_judge(label: str, words: list, signals: dict) -> LiveJudgement:
        await gate.wait()
        return LiveJudgement(score=95, title="T", hook="", reason="")

    deps, parts = make_deps(judge=stuck_judge)
    parts["trigger"].fire = True
    states: list[StreamState] = []
    watcher = StreamWatcher("s-cancel", "url", "", LiveSettings(), deps=deps)
    watcher._on_state = lambda sid, old, new, detail: states.append(new)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=10.0, strength=1.0))
        await wait_until(lambda: watcher.state is StreamState.CLIPPING)
        # Full teardown cancels the in-flight candidate; the watcher's
        # finally must still unwind and must not leave CLIPPING standing.
        watcher.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        assert watcher.state is not StreamState.CLIPPING
        assert watcher._stopping is True
    finally:
        gate.set()


# --------------------------------------------------------------------------
# Store integration (real A9 store under an initialized DB)
# --------------------------------------------------------------------------


async def test_candidate_writes_live_clips_row(primed_db: int) -> None:
    enqueued: list[ClipWorkItem] = []

    async def judge(label: str, words: list, signals: dict) -> LiveJudgement:
        return LiveJudgement(score=95, title="Kept", hook="h", reason="r")

    deps, parts = make_deps(enqueue=lambda item: enqueued.append(item) or True, judge=judge)
    parts["trigger"].fire = True
    watcher = StreamWatcher("s-row", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=120.0, strength=1.0))
        await wait_until(lambda: len(enqueued) == 1)
        clip_id = enqueued[0].clip_id
        row = store.get_clip(clip_id)
        assert row is not None
        assert row.status == "queued"
        assert row.title == "Kept"
        assert row.stream_id == "s-row"
        assert row.clip_start_s == pytest.approx(100.0)
        assert row.clip_end_s == pytest.approx(130.0)
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
    finally:
        await stop_watcher(watcher, task)


async def test_candidate_marks_clip_row_failed_when_extraction_fails(primed_db: int) -> None:
    class ExplodingBuffer(FakeBuffer):
        def extract(self, start_s: float, end_s: float, target: Path) -> Path:
            raise ValueError("segments 9..25 are not all held")

    deps, parts = make_deps(buffer=ExplodingBuffer())
    parts["trigger"].fire = True
    watcher = StreamWatcher("s-xfail", "url", "", LiveSettings(), deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=120.0, strength=1.0))
        await wait_until(lambda: any(f.status == "failed" for f in (store.list_clips() or [])))
        rows = [r for r in store.list_clips() or [] if r.status == "failed"]
        assert rows[0].error
        assert "window extraction failed" in rows[0].error
    finally:
        await stop_watcher(watcher, task)


# --------------------------------------------------------------------------
# Trim advice
# --------------------------------------------------------------------------


def test_trim_window_applies_word_offsets() -> None:
    deps, _ = make_deps()
    watcher = StreamWatcher("s-trim", "url", "", LiveSettings(), deps=deps)
    # Trimmed span must stay above TRIM_MIN_S (10s) to be accepted.
    words = [FakeWord(text="a", start=0.0, end=1.0), FakeWord(text="b", start=4.0, end=16.0)]
    judgement = LiveJudgement(score=90, start_word_offset=0, end_word_offset=1)
    start, end = watcher._trim_window(judgement, words, clip_start_s=100.0, clip_end_s=130.0)
    assert start == pytest.approx(100.0)
    assert end == pytest.approx(116.0)


def test_trim_window_ignores_advice_below_minimum() -> None:
    deps, _ = make_deps()
    watcher = StreamWatcher("s-trim2", "url", "", LiveSettings(), deps=deps)
    words = [FakeWord(text="a", start=0.0, end=0.5), FakeWord(text="b", start=1.0, end=1.2)]
    judgement = LiveJudgement(score=90, start_word_offset=0, end_word_offset=1)
    start, end = watcher._trim_window(judgement, words, clip_start_s=100.0, clip_end_s=130.0)
    # tightening 30s → 1.2s is below the minimum: ignored wholesale.
    assert (start, end) == (100.0, 130.0)


def test_trim_window_ignores_offsets_without_words() -> None:
    deps, _ = make_deps()
    watcher = StreamWatcher("s-trim3", "url", "", LiveSettings(), deps=deps)
    judgement = LiveJudgement(score=90, start_word_offset=2, end_word_offset=5)
    start, end = watcher._trim_window(judgement, [], clip_start_s=100.0, clip_end_s=130.0)
    assert (start, end) == (100.0, 130.0)
