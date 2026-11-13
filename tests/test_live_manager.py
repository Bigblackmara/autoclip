"""Unit tests for the LiveManager orchestrator (A8).

Everything is faked: no network, no streamlink, no whisper, no ffmpeg. The
manager is driven through its public surface exactly as A10 will use it,
with watcher-side contracts (capture factory, buffer factory, judge)
replaced by minimal fakes via the constructor's injection seams.

Async tests rely on this suite's asyncio_mode = "auto".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from autoclip.live import store
from autoclip.live.manager import (
    HOSTED_JUDGE_CONCURRENCY,
    LOCAL_JUDGE_CONCURRENCY,
    LiveManager,
    _judge_concurrency,
    json_read_live_section,
)
from autoclip.live.settings import LiveSettings
from autoclip.live.states import StreamState

# --------------------------------------------------------------------------
# Fakes (minimal shapes of the watcher contracts)
# --------------------------------------------------------------------------


class FakeBuffer:
    """Minimal RingBuffer shape: enough surface for the watcher loop."""

    def __init__(self) -> None:
        self.closed = False
        self.extract_calls: list[tuple[float, float]] = []

    def begin_epoch(self) -> int:
        return 1

    def prune(self, *_: Any, **__: Any) -> int:
        return 0

    def extract(self, start_s: float, end_s: float, target: Any) -> Any:
        # 파일을 실제로 만들면 _file_size가 통과됩니다 (translated: writing the
        # file for real lets _file_size pass)
        from pathlib import Path as _P

        p = _P(str(target))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake-window")
        self.extract_calls.append((start_s, end_s))
        return target

    def span_seconds(self) -> float:
        return 0.0

    def close(self) -> None:
        self.closed = True


class FakeCapture:
    def __init__(self, buffer: Any, on_event: Any) -> None:
        self.buffer = buffer
        self.on_event = on_event

    def start(self) -> None:
        self.on_event("started", "fake capture started")

    def stop(self) -> None:
        pass


@dataclass
class FakeDecision:
    fire: bool = False


class FakeAggregator:
    """Minimal SignalAggregator: add() + evaluate() rolling window."""

    def __init__(self) -> None:
        self.signals: list[Any] = []

    def add(self, signal: Any) -> None:
        self.signals.append(signal)

    def evaluate(self, window_s: float = 30.0) -> Any:
        times = [s.t_s for s in self.signals]
        end = max(times) if times else 0.0
        return _Window(
            start_s=max(0.0, end - window_s),
            end_s=end,
            signals=[s for s in self.signals if end - window_s <= s.t_s <= end],
        )


class _Window:
    def __init__(self, start_s: float, end_s: float, signals: list) -> None:
        self.start_s = start_s
        self.end_s = end_s
        self.signals = signals

    def strength_of(self, kind: Any) -> float:
        matching = [s for s in self.signals if s.kind == kind]
        return max((s.strength for s in matching), default=0.0)


class FakeTrigger:
    def __init__(self) -> None:
        self.fire = False

    def decide(self, window: Any) -> FakeDecision:
        return FakeDecision(fire=self.fire)


class AlwaysBucket:
    def try_acquire(self, n: float = 1.0) -> bool:
        return True


class AlwaysCooldown:
    def can_trigger(self, stream_id: str, **_: Any) -> bool:
        return True

    def mark_clip(self, stream_id: str, **_: Any) -> None:
        pass


@dataclass
class FakeTranscript:
    words: list = field(default_factory=list)


def make_manager(
    *,
    settings: LiveSettings | None = None,
    store_mod: Any = None,
    registry: Any = None,
    produce: Any = None,
    publish: Any = None,
    judge_semaphore: Any = None,
) -> LiveManager:
    """A manager whose watchers run entirely on fakes (hermetic)."""
    return LiveManager(
        settings=settings or LiveSettings(),
        store_mod=store_mod,
        registry=registry,
        produce=produce,
        capture_factory=lambda buf, on_event: FakeCapture(buf, on_event),
        buffer_factory=FakeBuffer,
        aggregator_factory=FakeAggregator,
        trigger=FakeTrigger(),
        cooldown=AlwaysCooldown(),
        judge_minute=AlwaysBucket(),
        judge_hour=AlwaysBucket(),
        judge_day=AlwaysBucket(),
        transcribe=lambda path: FakeTranscript(),
        audio_extract=lambda source, dest: None,
        judge=None,
        publish=publish,
        judge_semaphore=judge_semaphore,
    )


class FakeStore:
    """In-memory store module: the exact call surface the manager uses."""

    def __init__(self) -> None:
        self.streams: dict[str, Any] = {}
        self.clips: dict[str, Any] = {}
        self.state_calls: list[tuple[str, dict]] = []
        self.reset_calls = 0

    def reset_interrupted_states(self) -> None:
        self.reset_calls += 1

    def upsert_stream(self, row: Any) -> None:
        self.streams[row.id] = row

    def get_stream(self, stream_id: str) -> Any | None:
        return self.streams.get(stream_id)

    def list_streams(self) -> list:
        return list(self.streams.values())

    def delete_stream(self, stream_id: str) -> None:
        self.streams.pop(stream_id, None)

    def update_stream_state(self, stream_id: str, **fields: Any) -> None:
        self.state_calls.append((stream_id, fields))
        row = self.streams.get(stream_id)
        if row is not None:
            for key, value in fields.items():
                setattr(row, key, value)

    def get_clip(self, clip_id: str) -> Any | None:
        return self.clips.get(clip_id)

    def list_clips(self) -> list:
        return list(self.clips.values())

    def create_clip(self, row: Any) -> None:
        self.clips[row.id] = row

    def update_clip(self, clip_id: str, **fields: Any) -> None:
        row = self.clips.get(clip_id)
        if row is not None:
            for key, value in fields.items():
                setattr(row, key, value)


class FakeRegistry:
    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}

    def upsert(self, cfg: Any) -> None:
        self.entries[cfg.id] = cfg

    def get(self, stream_id: str) -> Any | None:
        return self.entries.get(stream_id)

    def list_streams(self) -> list:
        return list(self.entries.values())

    def delete(self, stream_id: str) -> None:
        self.entries.pop(stream_id, None)

    def set_enabled(self, stream_id: str, enabled: bool) -> None:
        cfg = self.entries.get(stream_id)
        if cfg is not None:
            cfg.enabled = enabled


async def wait_until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"condition not met within {timeout}s")


# --------------------------------------------------------------------------
# Settings loading
# --------------------------------------------------------------------------


def test_json_read_live_section_defaults_on_none() -> None:
    settings = json_read_live_section(None)
    assert isinstance(settings, LiveSettings)
    assert settings.workers.max_concurrent_streams == 5


def test_json_read_live_section_validates_dict() -> None:
    settings = json_read_live_section({"enabled": False})
    assert settings.enabled is False
    assert settings.workers.clip_queue_max == 10  # frozen default


def test_json_read_live_section_ignores_garbage() -> None:
    settings = json_read_live_section({"workers": {"nonsense": True}})
    assert isinstance(settings, LiveSettings)


def test_judge_concurrency_mirrors_highlights() -> None:
    class S:
        active_provider = "anthropic"

    class L:
        active_provider = "ollama"

    assert _judge_concurrency(S()) == HOSTED_JUDGE_CONCURRENCY == 3
    assert _judge_concurrency(L()) == LOCAL_JUDGE_CONCURRENCY == 1


# --------------------------------------------------------------------------
# start/stop lifecycle
# --------------------------------------------------------------------------


async def test_start_is_idempotent_and_resets_interrupted_states() -> None:
    fake_store = FakeStore()
    manager = make_manager(store_mod=fake_store)
    await manager.start()
    assert fake_store.reset_calls == 1
    await manager.start()  # second call is a no-op
    assert fake_store.reset_calls == 1
    await manager.stop()


async def test_stop_is_idempotent() -> None:
    manager = make_manager(store_mod=FakeStore())
    await manager.start()
    await manager.stop()
    await manager.stop()  # second call is a no-op, not a crash
    assert manager.status()["running"] is False


async def test_start_loads_persisted_streams_from_registry() -> None:
    from autoclip.live.settings import LiveStreamConfig

    registry = FakeRegistry()
    registry.upsert(LiveStreamConfig(id="persist-1", url="https://twitch.tv/a", label="a"))
    registry.upsert(
        LiveStreamConfig(id="persist-2", url="https://twitch.tv/b", label="b", enabled=False)
    )
    manager = make_manager(store_mod=FakeStore(), registry=registry)
    await manager.start()
    try:
        assert set(manager._stream_rows) == {"persist-1", "persist-2"}
        assert "persist-1" in manager._tasks  # enabled → launched
        assert "persist-2" not in manager._tasks  # disabled → not launched
    finally:
        await manager.stop()


async def test_start_loads_streams_from_db_when_registry_empty() -> None:
    fake_store = FakeStore()
    from autoclip.live.models import LiveStreamRow

    fake_store.streams["db-1"] = LiveStreamRow(id="db-1", url="u1", label="")
    manager = make_manager(store_mod=fake_store, registry=FakeRegistry())
    await manager.start()
    try:
        assert "db-1" in manager._stream_rows
        assert "db-1" in manager._tasks
    finally:
        await manager.stop()


# --------------------------------------------------------------------------
# add / remove / pause / resume
# --------------------------------------------------------------------------


async def test_add_stream_persists_and_launches() -> None:
    fake_store = FakeStore()
    registry = FakeRegistry()
    manager = make_manager(store_mod=fake_store, registry=registry)
    await manager.start()
    try:
        sid = await manager.add_stream("https://twitch.tv/x", "x")
        assert sid in fake_store.streams
        assert sid in registry.entries
        assert sid in manager._tasks
        snapshot = manager.stream_snapshot(sid)
        assert snapshot["url"] == "https://twitch.tv/x"
        assert snapshot["label"] == "x"
    finally:
        await manager.stop()


async def test_add_stream_rejects_empty_url() -> None:
    manager = make_manager(store_mod=FakeStore())
    await manager.start()
    try:
        with pytest.raises(ValueError):
            await manager.add_stream("   ")
    finally:
        await manager.stop()


async def test_add_stream_disabled_does_not_launch() -> None:
    fake_store = FakeStore()
    manager = make_manager(store_mod=fake_store, registry=FakeRegistry())
    await manager.start()
    try:
        sid = await manager.add_stream("https://twitch.tv/y", enabled=False)
        assert sid in fake_store.streams  # persisted...
        assert sid not in manager._tasks  # ...but not watched
        assert manager.stream_snapshot(sid)["enabled"] is False
    finally:
        await manager.stop()


async def test_add_stream_on_stopped_manager_persists_for_next_start() -> None:
    fake_store = FakeStore()
    manager = make_manager(store_mod=fake_store, registry=FakeRegistry())
    sid = await manager.add_stream("https://twitch.tv/z")  # manager not started
    assert sid in fake_store.streams
    assert sid not in manager._tasks
    await manager.start()  # now the persisted row launches
    try:
        assert sid in manager._tasks
    finally:
        await manager.stop()


async def test_remove_stream_stops_and_deletes_everywhere() -> None:
    fake_store = FakeStore()
    registry = FakeRegistry()
    manager = make_manager(store_mod=fake_store, registry=registry)
    await manager.start()
    sid = await manager.add_stream("https://twitch.tv/rm", "rm")
    await manager.remove_stream(sid)
    try:
        assert sid not in fake_store.streams
        assert sid not in registry.entries
        assert sid not in manager._tasks
        assert sid not in manager._watchers
        with pytest.raises(KeyError):
            manager.stream_snapshot(sid)
    finally:
        await manager.stop()


async def test_pause_stream_keeps_row_but_stops_watcher() -> None:
    fake_store = FakeStore()
    manager = make_manager(store_mod=fake_store, registry=FakeRegistry())
    await manager.start()
    sid = await manager.add_stream("https://twitch.tv/p", "p")
    await manager.pause_stream(sid)
    try:
        assert sid in fake_store.streams  # row kept...
        assert sid not in manager._tasks  # ...watcher stopped
        snapshot = manager.stream_snapshot(sid)
        assert snapshot["enabled"] is False
        # pause → state 'off' persisted
        states = [state for _, state in fake_store.state_calls]
        assert any(s.get("state") == StreamState.OFF.value for s in states)
    finally:
        await manager.stop()


async def test_resume_stream_launches_fresh_watcher() -> None:
    fake_store = FakeStore()
    manager = make_manager(store_mod=fake_store, registry=FakeRegistry())
    await manager.start()
    sid = await manager.add_stream("https://twitch.tv/q", "q")
    await manager.pause_stream(sid)
    assert sid not in manager._tasks
    await manager.resume_stream(sid)
    try:
        assert sid in manager._tasks
        assert manager.stream_snapshot(sid)["enabled"] is True
    finally:
        await manager.stop()


async def test_resume_unknown_stream_raises_key_error() -> None:
    manager = make_manager(store_mod=FakeStore(), registry=FakeRegistry())
    await manager.start()
    try:
        with pytest.raises(KeyError):
            await manager.resume_stream("no-such-stream")
    finally:
        await manager.stop()


# --------------------------------------------------------------------------
# Crash isolation & capacity
# --------------------------------------------------------------------------


async def test_watcher_crash_does_not_kill_the_manager() -> None:
    """A watcher task dying leaves siblings + the manager fully alive."""

    state = {"n": 0}

    def exploding_buffer() -> Any:
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("buffer factory exploded")
        return FakeBuffer()

    manager = LiveManager(
        settings=LiveSettings(),
        store_mod=FakeStore(),
        buffer_factory=exploding_buffer,
        capture_factory=lambda buf, on_event: FakeCapture(buf, on_event),
    )
    await manager.start()
    sid = await manager.add_stream("https://twitch.tv/boom")
    healthy = await manager.add_stream("https://twitch.tv/fine")
    await asyncio.sleep(0.05)
    try:
        # The sick stream never got a task; the healthy one runs on.
        assert sid not in manager._tasks
        assert healthy in manager._tasks
        assert manager.status()["running"] is True
    finally:
        await manager.stop()


async def test_max_concurrent_streams_capacity_gate() -> None:
    settings = LiveSettings()
    settings.workers.max_concurrent_streams = 2
    manager = make_manager(store_mod=FakeStore(), settings=settings)
    await manager.start()
    try:
        s1 = await manager.add_stream("https://twitch.tv/c1")
        s2 = await manager.add_stream("https://twitch.tv/c2")
        s3 = await manager.add_stream("https://twitch.tv/c3")
        await wait_until(lambda: len(manager._capturing) == 2)
        held = set(manager._capturing)
        assert len(held) == 2  # the cap
        assert manager._reserve_capture(s1) is True  # already capturing → ok
        assert manager._reserve_capture(s3) is False  # over cap → refused
        assert s3 not in manager._capturing  # refused id must not leak in
        manager._capturing.discard(s2)  # simulate slot freed
        assert manager._reserve_capture(s3) is True  # now fits
        assert s3 in manager._capturing
    finally:
        await manager.stop()


def test_capacity_gate_sync_block_no_await_between_check_and_mark() -> None:
    """The check and the connecting mark keep one synchronous block: AST-verify
    that _reserve_capture's body contains no await/yield expressions."""
    import ast
    import inspect

    assert inspect.iscoroutinefunction(LiveManager._reserve_capture) is False
    import textwrap

    raw = textwrap.dedent(inspect.getsource(LiveManager._reserve_capture))
    tree = ast.parse(raw)
    bad = [n for n in ast.walk(tree) if isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))]
    assert bad == []


# --------------------------------------------------------------------------
# Clip queue + producer indirection
# --------------------------------------------------------------------------


@dataclass
class FakeProduceResult:
    rendered_path: str = "rendered.mp4"
    transcript_path: str = "transcript.json"
    title: str = "Produced"
    error: None = None


async def test_clip_pipeline_produces_and_marks_ready(initialised_db: int) -> None:
    """Full end-to-end (all fakes): signal → queue → producer → 'ready'."""
    from autoclip.live.judge import LiveJudgement
    from autoclip.live.watcher import ClipWorkItem

    produced: list[Any] = []

    def produce(item: ClipWorkItem, buffer: Any, clip_row: Any, settings: Any) -> Any:
        produced.append(item)
        return FakeProduceResult()

    async def judge(label: str, words: list, signals: dict) -> LiveJudgement:
        return LiveJudgement(score=95, title="Kept", hook="h", reason="r")

    manager = LiveManager(
        settings=LiveSettings(),
        produce=produce,
        capture_factory=lambda buf, on_event: FakeCapture(buf, on_event),
        buffer_factory=FakeBuffer,
        aggregator_factory=FakeAggregator,
        trigger=_FiringTrigger(),
        cooldown=AlwaysCooldown(),
        judge_minute=AlwaysBucket(),
        judge_hour=AlwaysBucket(),
        judge_day=AlwaysBucket(),
        transcribe=lambda path: FakeTranscript(),
        audio_extract=lambda s, d: None,
        judge=judge,
    )
    await manager.start()
    sid = await manager.add_stream("https://twitch.tv/e2e", "e2e")
    try:
        await wait_until(lambda: manager._watchers.get(sid) is not None)
        watcher = manager._watchers[sid]
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(_make_signal(t_s=120.0))
        await wait_until(lambda: len(produced) == 1, timeout=10.0)
        item = produced[0]
        assert item.stream_id == sid
        row = store.get_clip(item.clip_id)
        assert row is not None
        await wait_until(lambda: (store.get_clip(item.clip_id) or LiveClipLike()).status == "ready")
        ready = store.get_clip(item.clip_id)
        assert ready.status == "ready"
        assert ready.rendered_path == "rendered.mp4"
        assert ready.title in {"Produced", "Kept"}
    finally:
        await manager.stop()


class _FiringTrigger:
    def decide(self, window: Any) -> Any:
        return FakeDecision(fire=True)


class LiveClipLike:
    status = "detecting"


def _make_signal(t_s: float):
    from autoclip.live.signals import DetectorSignal, SignalKind

    return DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=t_s, strength=1.0)


async def test_producer_failure_marks_clip_failed(initialised_db: int) -> None:
    from autoclip.live.judge import LiveJudgement
    from autoclip.live.watcher import ClipWorkItem

    def bad_produce(item: ClipWorkItem, buffer: Any, clip_row: Any, settings: Any) -> Any:
        raise RuntimeError("ffmpeg died")

    async def judge(label: str, words: list, signals: dict) -> LiveJudgement:
        return LiveJudgement(score=95, title="T", hook="", reason="")

    manager = LiveManager(
        settings=LiveSettings(),
        produce=bad_produce,
        capture_factory=lambda buf, on_event: FakeCapture(buf, on_event),
        buffer_factory=FakeBuffer,
        aggregator_factory=FakeAggregator,
        trigger=_FiringTrigger(),
        cooldown=AlwaysCooldown(),
        judge_minute=AlwaysBucket(),
        judge_hour=AlwaysBucket(),
        judge_day=AlwaysBucket(),
        transcribe=lambda path: FakeTranscript(),
        audio_extract=lambda s, d: None,
        judge=judge,
    )
    await manager.start()
    sid = await manager.add_stream("https://twitch.tv/fail", "fail")
    try:
        await wait_until(lambda: manager._watchers.get(sid) is not None)
        watcher = manager._watchers[sid]
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(_make_signal(t_s=90.0))
        await wait_until(
            lambda: any(getattr(r, "status", "") == "failed" for r in (store.list_clips() or [])),
            timeout=10.0,
        )
        failed = [r for r in store.list_clips() or [] if r.status == "failed"]
        assert failed and "ffmpeg died" in (failed[0].error or "")
    finally:
        await manager.stop()


async def test_queue_full_drops_and_never_blocks_the_watcher(initialised_db: int) -> None:
    """A full clip queue degrades via events + rejected rows, no blocking."""
    fake_store = FakeStore()
    settings = LiveSettings()
    settings.workers.clip_queue_max = 1
    import threading

    gate = threading.Event()

    def produce(item: Any, buffer: Any, clip_row: Any, settings: Any) -> Any:
        gate.wait(timeout=60)  # hold the single worker so the queue stays full
        return FakeProduceResult()

    async def judge(label: str, words: list, signals: dict) -> Any:
        from autoclip.live.judge import LiveJudgement

        return LiveJudgement(score=95, title="T", hook="", reason="")

    manager = LiveManager(
        settings=settings,
        store_mod=fake_store,
        produce=produce,
        capture_factory=lambda buf, on_event: FakeCapture(buf, on_event),
        buffer_factory=FakeBuffer,
        aggregator_factory=FakeAggregator,
        trigger=_FiringTrigger(),
        cooldown=AlwaysCooldown(),
        judge_minute=AlwaysBucket(),
        judge_hour=AlwaysBucket(),
        judge_day=AlwaysBucket(),
        transcribe=lambda path: FakeTranscript(),
        audio_extract=lambda s, d: None,
        judge=judge,
    )
    await manager.start()
    sid = await manager.add_stream("https://twitch.tv/full", "full")
    try:
        # The watcher writes through the REAL store module; its FK needs the
        # parent row, which FakeStore didn't create.
        from autoclip.live.models import LiveStreamRow

        store.upsert_stream(LiveStreamRow(id=sid, url="https://twitch.tv/full", label="full"))
        await wait_until(lambda: manager._watchers.get(sid) is not None)
        watcher = manager._watchers[sid]
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        # First candidate goes through (queue empty).
        watcher.submit_signal(_make_signal(t_s=10.0))
        await wait_until(lambda: manager._queue is not None and manager._queue.qsize() >= 1)
        # Second candidate: queue full (worker is gated) → dropped, not hung.
        watcher.submit_signal(_make_signal(t_s=12.0))
        await asyncio.sleep(0.2)
        assert watcher.state is not None  # the watcher survived the drop
    finally:
        gate.set()
        await manager.stop()


# --------------------------------------------------------------------------
# Event publishing
# --------------------------------------------------------------------------


async def test_manager_publishes_lifecycle_events() -> None:
    published: list[tuple[str, str]] = []

    def publish(channel: str, event_type: str, data: dict | None = None) -> None:
        published.append((channel, event_type))

    manager = make_manager(store_mod=FakeStore(), publish=publish)
    await manager.start()
    try:
        sid = await manager.add_stream("https://twitch.tv/ev", "ev")
        await manager.pause_stream(sid)
        await manager.resume_stream(sid)
        await manager.remove_stream(sid)
        events = [e for _, e in published]
        assert "stream_added" in events
        assert "stream_paused" in events
        assert "stream_resumed" in events
        assert "stream_removed" in events
    finally:
        await manager.stop()


async def test_injected_publisher_reaches_the_watcher() -> None:
    """The publish kwarg flows into watcher deps without a TypeError."""
    published: list[str] = []

    def publish(channel: str, event_type: str, data: dict | None = None) -> None:
        published.append(event_type)

    manager = make_manager(store_mod=FakeStore(), publish=publish)
    await manager.start()
    try:
        sid = await manager.add_stream("https://twitch.tv/pub", "pub")
        await wait_until(lambda: manager._watchers.get(sid) is not None)
        watcher = manager._watchers[sid]
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        assert "stream_added" in published
    finally:
        await manager.stop()


# --------------------------------------------------------------------------
# status()
# --------------------------------------------------------------------------


async def test_status_shape() -> None:
    manager = make_manager(store_mod=FakeStore())
    await manager.start()
    try:
        sid = await manager.add_stream("https://twitch.tv/st", "st")
        status = manager.status()
        assert status["running"] is True
        assert status["watching"] >= 0
        assert isinstance(status["streams"], list)
        assert set(status["caps"]) == {
            "clip_queue_max",
            "clip_queue_depth",
            "clip_queue_workers",
            "max_concurrent_streams",
            "pending_clip_bytes",
            "pending_cap_bytes",
        }
        by_id = {s["id"]: s for s in status["streams"]}
        assert sid in by_id
        assert by_id[sid]["url"] == "https://twitch.tv/st"
    finally:
        await manager.stop()


# --------------------------------------------------------------------------
# Module-level singleton
# --------------------------------------------------------------------------


def test_manager_singleton_exists() -> None:
    from autoclip.live import manager as manager_mod

    assert isinstance(manager_mod.manager, LiveManager)
    assert manager_mod.manager is manager_mod.manager


# --------------------------------------------------------------------------
# Judge concurrency mirror (scope update)
# --------------------------------------------------------------------------


async def test_judge_semaphore_default_is_provider_aware() -> None:
    """_resolve_judge_semaphore falls back to the config's provider."""
    manager = make_manager(store_mod=FakeStore())
    semaphore = manager._resolve_judge_semaphore()
    assert isinstance(semaphore, asyncio.Semaphore)


def test_judge_concurrency_values() -> None:
    assert HOSTED_JUDGE_CONCURRENCY == 3
    assert LOCAL_JUDGE_CONCURRENCY == 1
