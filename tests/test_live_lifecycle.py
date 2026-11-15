"""Live Watch lifecycle regressions: the reviewed P0/P1/P2 failure modes.

Each test reproduces one reviewed failure through the manager/watcher public
surface (fakes for capture, judge and render; the real SQLite store under a
temp AUTOCLIP_HOME; the real RingBuffer/StreamCapture where the bug lived)
and pins the fixed behaviour. No network, no real ffmpeg, no keyring.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from autoclip import paths
from autoclip.live import clip_factory, store
from autoclip.live.buffer import SEGMENT_S, RingBuffer, segment_name
from autoclip.live.capture import StreamCapture
from autoclip.live.judge import LiveJudgement
from autoclip.live.manager import LiveManager, RetryRefused
from autoclip.live.models import LiveClipRow, LiveStreamRow, LiveUploadRow
from autoclip.live.paths import clip_dir, stream_buffer_dir
from autoclip.live.settings import LiveSettings, LiveStreamConfig
from autoclip.live.signals import DetectorSignal, SignalKind, SignalWindow
from autoclip.live.states import StreamState
from autoclip.live.watcher import (
    JUDGE_AUDIO_NAME,
    JUDGE_TRANSCRIPT_NAME,
    JUDGE_VERDICT_NAME,
    JUDGE_WINDOW_NAME,
    ClipWorkItem,
    StreamWatcher,
    WatcherDeps,
)

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_tree_kills(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake processes carry made-up PIDs: never taskkill a real one."""
    monkeypatch.setattr("autoclip.live.capture.kill_process_tree", lambda pid: None)


class FakeBuffer:
    """RingBuffer stand-in: extract writes ``payload`` bytes, tracks pins."""

    def __init__(self, payload: bytes = b"fake-window") -> None:
        self.segments: list[Any] = []
        self.payload = payload
        self.extract_calls: list[tuple[float, float]] = []
        self.pins: list[int] = []
        self.unpins: list[int] = []
        self.retain_s = 300.0
        self.max_disk_bytes = 512 * 1024 * 1024

    def begin_epoch(self) -> int:
        return 1

    def prune(self, *_: Any, **__: Any) -> int:
        return 0

    def extract(self, start_s: float, end_s: float, target: Path) -> Path:
        self.extract_calls.append((start_s, end_s))
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload)
        return target

    def pin_oldest(self, index: int) -> None:
        self.pins.append(index)

    def unpin(self, index: int | None = None) -> None:
        self.unpins.append(index)

    def close(self) -> None:
        pass


class FakeCapture:
    def __init__(self, buffer: Any, on_event: Any, *, fail: BaseException | None = None) -> None:
        self.buffer = buffer
        self.on_event = on_event
        self.fail = fail
        self.stopped = False

    def start(self) -> None:
        if self.fail is not None:
            raise self.fail

    def stop(self) -> None:
        self.stopped = True


class Offline(Exception):
    offline = True


@dataclass
class Decision:
    fire: bool = True


class FiringTrigger:
    def decide(self, window: Any) -> Decision:
        return Decision(fire=True)


class Aggregator:
    def __init__(self) -> None:
        self.signals: list[DetectorSignal] = []

    def add(self, signal: DetectorSignal) -> None:
        self.signals.append(signal)

    def evaluate(self, window_s: float = 30.0) -> SignalWindow:
        end = max(s.t_s for s in self.signals)
        return SignalWindow(
            start_s=max(0.0, end - window_s),
            end_s=end,
            signals=[s for s in self.signals if s.t_s >= end - window_s],
        )


class AlwaysCooldown:
    def can_trigger(self, stream_id: str, **_: Any) -> bool:
        return True

    def mark_clip(self, stream_id: str, **_: Any) -> None:
        pass


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    words: list = field(default_factory=lambda: [Word("hey", 0.0, 0.5)])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"words": len(self.words)}), encoding="utf-8")
        return path


def audio_extract(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"RIFF" + b"\0" * 2000)


def immediate_sleep():
    async def _sleep(_seconds: float) -> None:
        await asyncio.sleep(0)

    return _sleep


def signal(t_s: float, strength: float = 1.5) -> DetectorSignal:
    return DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=t_s, strength=strength, peak_z=6.0)


async def wait_until(predicate: Any, timeout: float = 5.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    pytest.fail(f"condition not met within {timeout}s")


def watcher_for(
    stream_id: str,
    *,
    judge: Any = None,
    transcribe: Any = None,
    enqueue: Any = None,
    buffer: Any = None,
    settings: LiveSettings | None = None,
    **extra: Any,
) -> tuple[StreamWatcher, dict]:
    """A hermetic watcher writing through the REAL store (needs a stream row)."""
    calls: dict[str, list] = {"judge": [], "enqueued": [], "captures": []}

    async def default_judge(label: str, words: list, signals: dict) -> LiveJudgement:
        calls["judge"].append(words)
        return LiveJudgement(score=10, title="meh", reason="nothing happens")

    def capture_factory(buf: Any, on_event: Any) -> FakeCapture:
        cap = FakeCapture(buf, on_event)
        calls["captures"].append(cap)
        return cap

    def default_enqueue(item: ClipWorkItem) -> bool:
        calls["enqueued"].append(item)
        return True

    the_buffer = buffer if buffer is not None else FakeBuffer()
    deps = WatcherDeps(
        capture_factory=extra.pop("capture_factory", capture_factory),
        buffer_factory=lambda: the_buffer,
        aggregator_factory=Aggregator,
        trigger=FiringTrigger(),
        cooldown=extra.pop("cooldown", AlwaysCooldown()),
        transcribe=transcribe or (lambda path: Transcript()),
        audio_extract=audio_extract,
        judge=judge or default_judge,
        enqueue_clip=enqueue or default_enqueue,
        publish=lambda *a, **k: None,
        sleep=immediate_sleep(),
        chat_runner=lambda *a, **k: False,
        audio_start=lambda *a, **k: None,
        **extra,
    )
    watcher = StreamWatcher(
        stream_id, "https://kick.com/x", "x", settings or LiveSettings(), deps=deps
    )
    calls["buffer"] = the_buffer
    return watcher, calls


async def stop_watcher(watcher: StreamWatcher, task: asyncio.Task) -> None:
    watcher.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


@pytest.fixture
def stream_row(initialised_db: int) -> str:
    store.upsert_stream(LiveStreamRow(id="s1", url="https://kick.com/x", label="x"))
    return "s1"


# --------------------------------------------------------------------------
# P0 #2 — settings are loaded, preserved, and applied live
# --------------------------------------------------------------------------


def _write_live_section(section: dict) -> None:
    paths.ensure_layout()
    paths.config_path().write_text(json.dumps({"live": section}), encoding="utf-8")


class TestSettingsTakeEffect:
    async def test_manager_runs_on_the_stored_section(self, initialised_db: int) -> None:
        _write_live_section({"enabled": False, "rate_limit": {"judge_min_score": 90}})
        manager = LiveManager(registry=FakeRegistry())
        assert manager.settings.enabled is False
        assert manager.settings.rate_limit.judge_min_score == 90
        await manager.start()
        try:
            assert manager.started is False  # off means off, not "defaults"
        finally:
            await manager.shutdown()

    async def test_enabling_starts_and_disabling_stops_the_feature(
        self, initialised_db: int
    ) -> None:
        _write_live_section({"enabled": False})
        registry = FakeRegistry()
        registry.upsert(LiveStreamConfig(id="s1", url="https://kick.com/x", label="x"))
        captures: list[FakeCapture] = []

        def capture_factory(buf: Any, on_event: Any) -> FakeCapture:
            captures.append(FakeCapture(buf, on_event))
            return captures[-1]

        manager = LiveManager(
            registry=registry,
            capture_factory=capture_factory,
            buffer_factory=FakeBuffer,
            audio_start=lambda *a, **k: None,
            chat_runner=lambda *a, **k: False,
        )
        await manager.start()
        try:
            assert manager.started is False
            await manager.apply_settings(LiveSettings(enabled=True))
            assert manager.started is True
            await wait_until(lambda: "s1" in manager._watchers)
            watcher = manager._watchers["s1"]
            await wait_until(lambda: watcher.state is StreamState.WATCHING)

            await manager.apply_settings(LiveSettings(enabled=False))
            assert manager.started is False
            assert manager._watchers == {}
            await wait_until(lambda: all(c.stopped for c in captures))
            assert manager.stream_snapshot("s1")["state"] == "off"
        finally:
            await manager.shutdown()

    async def test_thresholds_budgets_and_workers_apply_live(self, initialised_db: int) -> None:
        buffer = FakeBuffer()
        manager = LiveManager(
            settings=LiveSettings(),
            registry=FakeRegistry(),
            capture_factory=lambda b, e: FakeCapture(b, e),
            buffer_factory=lambda: buffer,
            audio_start=lambda *a, **k: None,
            chat_runner=lambda *a, **k: False,
        )
        await manager.start()
        try:
            sid = await manager.add_stream("https://kick.com/x", "x")
            await wait_until(lambda: sid in manager._watchers)
            watcher = manager._watchers[sid]
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            new = LiveSettings()
            new.rate_limit.judge_min_score = 97
            new.rate_limit.judge_per_minute = 1
            new.rate_limit.cooldown_s = 5.0
            new.detection.audio.spike_z = 9.0
            new.rolling_buffer.retain_minutes = 2.0
            new.workers.clip_queue_workers = 3
            new.workers.clip_queue_max = 2
            new.uploads.youtube.auto_upload = True
            result = await manager.apply_settings(new)
            assert result == {"running": True, "restarted": False}
            assert manager._watchers[sid] is watcher  # no restart needed
            live = watcher._live
            assert live is manager.settings
            assert live.rate_limit.judge_min_score == 97
            assert live.detection.audio.spike_z == 9.0
            assert live.uploads.youtube.auto_upload is True
            assert watcher._deps.cooldown.cooldown_s == 5.0 or isinstance(
                watcher._deps.cooldown, AlwaysCooldown
            )
            assert buffer.retain_s == 120.0
            assert len([w for w in manager._workers if not w.done()]) == 3
            # The queue bound is read live too.
            assert manager._live.workers.clip_queue_max == 2
        finally:
            await manager.shutdown()

    async def test_bound_at_start_settings_restart_the_watchers(self, initialised_db: int) -> None:
        manager = LiveManager(
            settings=LiveSettings(),
            registry=FakeRegistry(),
            capture_factory=lambda b, e: FakeCapture(b, e),
            buffer_factory=FakeBuffer,
            audio_start=lambda *a, **k: None,
            chat_runner=lambda *a, **k: False,
        )
        await manager.start()
        try:
            sid = await manager.add_stream("https://kick.com/x", "x")
            await wait_until(lambda: sid in manager._watchers)
            first = manager._watchers[sid]
            new = LiveSettings()
            new.detection.chat.enabled = False
            result = await manager.apply_settings(new)
            assert result["restarted"] is True
            await wait_until(lambda: manager._watchers.get(sid) not in (None, first))
        finally:
            await manager.shutdown()


class FakeRegistry:
    def __init__(self) -> None:
        self.entries: dict[str, LiveStreamConfig] = {}

    def upsert(self, cfg: LiveStreamConfig) -> None:
        self.entries[cfg.id] = cfg

    def list_streams(self) -> list[LiveStreamConfig]:
        return list(self.entries.values())

    def delete(self, stream_id: str) -> None:
        self.entries.pop(stream_id, None)

    def set_enabled(self, stream_id: str, enabled: bool) -> None:
        cfg = self.entries.get(stream_id)
        if cfg is not None:
            self.entries[stream_id] = cfg.model_copy(update={"enabled": enabled})


# --------------------------------------------------------------------------
# P0 #3 — nothing under live/ grows forever
# --------------------------------------------------------------------------


class TestDiskHygiene:
    async def test_judge_media_is_deleted_once_the_verdict_is_recorded(
        self, stream_row: str
    ) -> None:
        watcher, calls = watcher_for(stream_row, buffer=FakeBuffer(b"V" * 50_000))
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(120.0))
            await wait_until(
                lambda: store.list_clips() and store.list_clips()[0].status == "rejected"
            )
            clip = store.list_clips()[0]
            work = clip_dir(clip.id)
            assert not (work / JUDGE_WINDOW_NAME).exists()
            assert not (work / JUDGE_AUDIO_NAME).exists()
            assert (work / JUDGE_TRANSCRIPT_NAME).is_file()
            verdict = json.loads((work / JUDGE_VERDICT_NAME).read_text(encoding="utf-8"))
            assert verdict["score"] == 10
        finally:
            await stop_watcher(watcher, task)

    async def test_ready_render_drops_its_intermediates(self, stream_row: str) -> None:
        def produce(item: Any, buffer: Any, clip_row: Any, settings: Any) -> Any:
            work = clip_dir(item.clip_id)
            for name in ("raw.ts", "normalized.mp4", "audio.wav", "transcript.json"):
                (work / name).write_bytes(b"x" * 100)
            rendered = work / "final.mp4"
            rendered.write_bytes(b"mp4")
            return _Produced(rendered)

        manager = LiveManager(settings=LiveSettings(), produce=produce, registry=FakeRegistry())
        item = _queued_clip(stream_row, "c-ready")
        await manager._render_one(item)
        work = clip_dir("c-ready")
        assert store.get_clip("c-ready").status == "ready"
        assert not (work / "raw.ts").exists()
        assert not (work / "normalized.mp4").exists()
        assert not (work / "audio.wav").exists()
        assert (work / "final.mp4").is_file()
        assert (work / "transcript.json").is_file()

    async def test_removing_a_stream_deletes_its_buffer_and_clip_folders(
        self, initialised_db: int
    ) -> None:
        manager = LiveManager(
            settings=LiveSettings(),
            registry=FakeRegistry(),
            capture_factory=lambda b, e: FakeCapture(b, e),
            buffer_factory=FakeBuffer,
            audio_start=lambda *a, **k: None,
            chat_runner=lambda *a, **k: False,
        )
        await manager.start()
        try:
            sid = await manager.add_stream("https://kick.com/x", "x")
            other = await manager.add_stream("https://kick.com/y", "y")
            buf = stream_buffer_dir(sid)
            buf.mkdir(parents=True, exist_ok=True)
            (buf / segment_name(3)).write_bytes(b"seg")
            for clip_id, owner in (("c-a", sid), ("c-b", sid), ("c-other", other)):
                store.create_clip(LiveClipRow(id=clip_id, stream_id=owner, status="ready"))
                clip_dir(clip_id).mkdir(parents=True, exist_ok=True)
                (clip_dir(clip_id) / "final.mp4").write_bytes(b"mp4")

            await manager.remove_stream(sid)

            assert not buf.exists()
            assert not clip_dir("c-a").exists() and not clip_dir("c-b").exists()
            assert clip_dir("c-other").is_dir()  # another stream's clips stay
            assert store.get_stream(sid) is None
        finally:
            await manager.shutdown()

    def test_janitor_frees_the_oldest_non_ready_artifacts_first(self, stream_row: str) -> None:
        settings = LiveSettings()
        settings.workers.disk_budget_mb = 1
        manager = LiveManager(settings=settings, registry=FakeRegistry())
        mb = 1024 * 1024

        def clip(clip_id: str, status: str, updated: str, size: int) -> Path:
            store.create_clip(
                LiveClipRow(id=clip_id, stream_id=stream_row, status=status, updated_at=updated)
            )
            store.update_clip(clip_id, status=status)  # keeps updated_at monotonic below
            work = clip_dir(clip_id)
            work.mkdir(parents=True, exist_ok=True)
            (work / "raw.ts").write_bytes(b"x" * size)
            (work / "judge_verdict.json").write_text("{}", encoding="utf-8")
            return work

        old_failed = clip("c-old", "failed", "2026-01-01T00:00:00+00:00", int(0.6 * mb))
        time.sleep(1.1)  # distinct updated_at seconds
        new_rejected = clip("c-new", "rejected", "2026-01-02T00:00:00+00:00", int(0.6 * mb))
        ready = clip("c-ready", "ready", "2026-01-01T00:00:00+00:00", 2 * mb)
        busy = clip("c-busy", "queued", "2026-01-01T00:00:00+00:00", int(0.3 * mb))

        report = manager.enforce_disk_budget()

        assert not (old_failed / "raw.ts").exists()  # oldest reclaimable first
        assert (old_failed / "judge_verdict.json").exists()  # evidence stays
        assert (new_rejected / "raw.ts").exists()  # back under budget: kept
        assert (ready / "raw.ts").exists()  # ready clips are never touched
        assert (busy / "raw.ts").exists()  # in-flight work is never touched
        assert report["artifacts_freed_bytes"] >= int(0.6 * mb)
        assert manager._admit_candidate() == (True, "")

    def test_still_over_budget_refuses_new_candidates(self, stream_row: str) -> None:
        settings = LiveSettings()
        settings.workers.disk_budget_mb = 1
        manager = LiveManager(settings=settings, registry=FakeRegistry())
        store.create_clip(LiveClipRow(id="c-busy", stream_id=stream_row, status="rendering"))
        clip_dir("c-busy").mkdir(parents=True, exist_ok=True)
        (clip_dir("c-busy") / "raw.ts").write_bytes(b"x" * (2 * 1024 * 1024))

        manager.enforce_disk_budget()

        ok, reason = manager._admit_candidate()
        assert ok is False and "disk budget" in reason

    def test_unowned_folders_are_garbage(self, stream_row: str, monkeypatch) -> None:
        from autoclip.live import manager as manager_mod

        monkeypatch.setattr(manager_mod, "ORPHAN_MIN_AGE_S", 0.0)
        manager = LiveManager(settings=LiveSettings(), registry=FakeRegistry())
        orphan_clip = clip_dir("c-ghost")
        orphan_clip.mkdir(parents=True)
        (orphan_clip / "raw.ts").write_bytes(b"x")
        orphan_buffer = stream_buffer_dir("gone-stream")
        orphan_buffer.mkdir(parents=True)
        (orphan_buffer / segment_name(1)).write_bytes(b"x")
        live_buffer = stream_buffer_dir(stream_row)
        live_buffer.mkdir(parents=True)

        report = manager.enforce_disk_budget()

        assert report["orphans_removed"] == 2
        assert not orphan_clip.exists() and not orphan_buffer.exists()
        assert live_buffer.exists()

    def test_unreadable_clip_list_never_deletes_clip_folders(
        self, stream_row: str, monkeypatch
    ) -> None:
        # a transient DB error must not make finished clips look ownerless
        from autoclip.live import manager as manager_mod

        monkeypatch.setattr(manager_mod, "ORPHAN_MIN_AGE_S", 0.0)
        manager = LiveManager(settings=LiveSettings(), registry=FakeRegistry())
        store.create_clip(LiveClipRow(id="c-keep", stream_id=stream_row, status="ready"))
        keep = clip_dir("c-keep")
        keep.mkdir(parents=True)
        (keep / "final.mp4").write_bytes(b"mp4")

        def locked(*_args, **_kwargs):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(manager._store, "list_clips", locked)

        report = manager.enforce_disk_budget()

        assert (keep / "final.mp4").exists()
        assert report["orphans_removed"] == 0

    async def test_low_free_disk_refuses_the_candidate_before_any_work(
        self, stream_row: str, monkeypatch
    ) -> None:
        from autoclip.live import manager as manager_mod

        usage = shutil.disk_usage(paths.root().parent)
        monkeypatch.setattr(
            manager_mod.shutil,
            "disk_usage",
            lambda _p: type(usage)(usage.total, usage.total - 10 * 1024 * 1024, 10 * 1024 * 1024),
        )
        manager = LiveManager(settings=LiveSettings(), registry=FakeRegistry())
        watcher, calls = watcher_for(stream_row, admit_candidate=manager._admit_candidate)
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(60.0))
            await wait_until(
                lambda: store.list_clips() and store.list_clips()[0].status == "rejected"
            )
            row = store.list_clips()[0]
            assert "low disk space" in row.reason
            assert calls["judge"] == [] and calls["buffer"].extract_calls == []
        finally:
            await stop_watcher(watcher, task)


@dataclass
class _Produced:
    rendered_path: Path
    transcript_path: Path | None = None
    title: str = "Produced"
    error: str | None = None


def _queued_clip(stream_id: str, clip_id: str, **row: Any) -> ClipWorkItem:
    store.create_clip(
        LiveClipRow(
            id=clip_id,
            stream_id=stream_id,
            status=row.pop("status", "queued"),
            title=row.pop("title", "Big moment"),
            clip_start_s=100.0,
            clip_end_s=130.0,
            trigger_end_s=120.0,
            **row,
        )
    )
    clip_dir(clip_id).mkdir(parents=True, exist_ok=True)
    return ClipWorkItem(
        clip_id=clip_id,
        stream_id=stream_id,
        label="x",
        source_path=clip_dir(clip_id) / "raw.ts",
        clip_start_s=100.0,
        clip_end_s=130.0,
        trigger_start_s=118.0,
        trigger_end_s=120.0,
        judgement=LiveJudgement(score=90, title="Big moment"),
    )


# --------------------------------------------------------------------------
# P0 #4 + #5 — a second lifecycle still sees new segments
# --------------------------------------------------------------------------


class FakeFfmpeg:
    """Writes numbered segments like the segment muxer: N closes when N+1 opens."""

    def __init__(self, directory: Path, start: int, marker: bytes) -> None:
        self.directory = directory
        self.next = start
        self.marker = marker
        self.pid = 4242
        self.stderr = None
        self.returncode: int | None = None
        self.killed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self.killed.wait(0.03):
            (self.directory / segment_name(self.next)).write_bytes(self.marker * 64)
            self.next += 1

    def poll(self) -> int | None:
        if self.killed.is_set():
            self.returncode = -9
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.killed.wait(timeout if timeout else 60.0)
        self.returncode = -9
        return -9

    def kill(self) -> None:
        self.killed.set()


class TestSecondLifecycle:
    async def test_pause_resume_lifecycle_accepts_new_segments(self, stream_row: str) -> None:
        procs: list[FakeFfmpeg] = []

        def capture_factory(buffer: RingBuffer, on_event: Any) -> StreamCapture:
            capture = StreamCapture(
                "https://kick.com/x",
                buffer,
                on_event=on_event,
                resolver=lambda u: "http://local/live.m3u8",
                poll_s=0.02,
                drain_grace_s=0.05,
            )
            marker = b"L1" if not procs else b"L2"

            def spawn(_url: str) -> FakeFfmpeg:
                procs.append(FakeFfmpeg(buffer.directory, capture._next_index, marker))
                return procs[-1]

            capture._custom_spawn = spawn
            return capture

        def make() -> StreamWatcher:
            deps = WatcherDeps(
                capture_factory=capture_factory,
                aggregator_factory=lambda: None,
                audio_start=lambda *a, **k: None,
                chat_runner=lambda *a, **k: False,
                publish=lambda *a, **k: None,
                enqueue_clip=lambda item: True,
            )
            return StreamWatcher(stream_row, "https://kick.com/x", "x", LiveSettings(), deps=deps)

        first = make()
        task = asyncio.create_task(first.run())
        await wait_until(lambda: first.buffer is not None and len(first.buffer.segments) >= 5)
        old_max = first.buffer.segments[-1].index
        await stop_watcher(first, task)
        # Leftovers of lifecycle 1 sit in the folder, high indices included.
        folder = stream_buffer_dir(stream_row)
        (folder / segment_name(old_max + 500)).write_bytes(b"stale")
        assert any(folder.glob("seg_*.ts"))

        second = make()
        task = asyncio.create_task(second.run())
        try:
            await wait_until(lambda: second.buffer is not None and len(second.buffer.segments) >= 5)
            indices = [s.index for s in second.buffer.segments]
            assert indices[:5] == [0, 1, 2, 3, 4]  # fresh numbering, nothing refused
            assert all(s.path.read_bytes().startswith(b"L2") for s in second.buffer.segments)
            assert not (folder / segment_name(old_max + 500)).exists()
        finally:
            await stop_watcher(second, task)
        for proc in procs:
            assert proc.killed.is_set()


# --------------------------------------------------------------------------
# P0 #6 — auto-upload fires from a real render, with the prefixes
# --------------------------------------------------------------------------


class TestAutoUploadThroughRender:
    async def test_render_one_uploads_the_rendered_file_with_prefixes(
        self, stream_row: str, monkeypatch
    ) -> None:
        import autoclip.live.uploads.instagram as ig_mod
        import autoclip.live.uploads.youtube as yt_mod

        uploaded: list[tuple[str, str, str]] = []

        @dataclass
        class Result:
            url: str = "https://example.invalid/v"
            media_id: str = "m1"

        async def fake_yt(path: Any, meta: Any) -> Result:
            uploaded.append(("youtube", str(path), meta.title))
            return Result()

        async def fake_ig(path: Any, meta: Any) -> Result:
            uploaded.append(("instagram", str(path), meta.title))
            return Result()

        monkeypatch.setattr(yt_mod, "upload_youtube", fake_yt)
        monkeypatch.setattr(ig_mod, "upload_instagram", fake_ig)

        def produce(item: Any, buffer: Any, clip_row: Any, settings: Any) -> Any:
            rendered = clip_dir(item.clip_id) / "final.mp4"
            rendered.write_bytes(b"mp4")
            return _Produced(rendered, title="Huge play")

        settings = LiveSettings()
        settings.uploads.youtube.auto_upload = True
        settings.uploads.youtube.title_prefix = "[LIVE] "
        settings.uploads.instagram.auto_upload = True
        settings.uploads.instagram.caption_prefix = "#clip "
        manager = LiveManager(settings=settings, produce=produce, registry=FakeRegistry())

        await manager._render_one(_queued_clip(stream_row, "c-up"))
        # Auto-upload runs beside the render worker, not inside it.
        await asyncio.gather(*manager._upload_tasks)

        rendered = str(clip_dir("c-up") / "final.mp4")
        assert uploaded == [
            ("youtube", rendered, "[LIVE] Huge play"),
            ("instagram", rendered, "#clip Huge play"),
        ]
        statuses = sorted((u.platform, u.status) for u in store.list_uploads("c-up"))
        assert statuses == [("instagram", "done"), ("youtube", "done")]

    async def test_no_second_upload_when_one_already_ran(
        self, stream_row: str, monkeypatch
    ) -> None:
        import autoclip.live.uploads.youtube as yt_mod

        calls: list[str] = []

        async def fake_yt(path: Any, meta: Any) -> Any:
            calls.append(str(path))
            return None

        monkeypatch.setattr(yt_mod, "upload_youtube", fake_yt)
        settings = LiveSettings()
        settings.uploads.youtube.auto_upload = True
        manager = LiveManager(settings=settings, registry=FakeRegistry())
        item = _queued_clip(stream_row, "c-dup", status="ready")
        store.create_upload(
            LiveUploadRow(id="u1", clip_id="c-dup", platform="youtube", status="done")
        )
        row = store.get_clip("c-dup")
        row.rendered_path = "x.mp4"
        await manager._maybe_auto_upload(item, row, "t")
        assert calls == []


# --------------------------------------------------------------------------
# P0 #7 — discard is a tombstone everywhere
# --------------------------------------------------------------------------


class TestDiscardTombstone:
    async def test_a_discarded_queued_clip_is_never_rendered(self, stream_row: str) -> None:
        produced: list[str] = []

        def produce(item: Any, *_: Any) -> Any:
            produced.append(item.clip_id)
            return _Produced(clip_dir(item.clip_id) / "final.mp4")

        manager = LiveManager(settings=LiveSettings(), produce=produce, registry=FakeRegistry())
        item = _queued_clip(stream_row, "c-q")
        store.update_clip("c-q", status="rejected", reason="discarded by user")
        await manager._render_one(item)
        assert produced == []
        row = store.get_clip("c-q")
        assert row.status == "rejected" and row.reason == "discarded by user"

    async def test_a_clip_discarded_mid_render_never_turns_ready(
        self, stream_row: str, monkeypatch
    ) -> None:
        import autoclip.live.uploads.youtube as yt_mod

        uploads: list[str] = []

        async def fake_yt(path: Any, meta: Any) -> Any:
            uploads.append(str(path))

        monkeypatch.setattr(yt_mod, "upload_youtube", fake_yt)

        def produce(item: Any, *_: Any) -> Any:
            # The user discards while the render runs.
            store.update_clip(item.clip_id, status="rejected", reason="discarded by user")
            rendered = clip_dir(item.clip_id) / "final.mp4"
            rendered.write_bytes(b"mp4")
            return _Produced(rendered)

        settings = LiveSettings()
        settings.uploads.youtube.auto_upload = True
        manager = LiveManager(settings=settings, produce=produce, registry=FakeRegistry())
        await manager._render_one(_queued_clip(stream_row, "c-r"))
        row = store.get_clip("c-r")
        assert row.status == "rejected" and row.rendered_path == ""
        assert not (clip_dir("c-r") / "final.mp4").exists()
        assert uploads == [] and store.list_uploads("c-r") == []

    async def test_a_clip_discarded_while_detecting_never_reaches_the_judge(
        self, stream_row: str
    ) -> None:
        gate = threading.Event()
        judged: list[int] = []

        def slow_transcribe(path: Path) -> Transcript:
            gate.wait(timeout=5)
            return Transcript()

        async def judge(label: str, words: list, signals: dict) -> LiveJudgement:
            judged.append(1)
            return LiveJudgement(score=99)

        watcher, calls = watcher_for(stream_row, judge=judge, transcribe=slow_transcribe)
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(60.0))
            await wait_until(lambda: bool(store.list_clips()))
            clip_id = store.list_clips()[0].id
            store.update_clip(clip_id, status="rejected", reason="discarded by user")
            gate.set()
            await wait_until(lambda: not watcher._candidate_busy)
            assert judged == [] and calls["enqueued"] == []
            assert store.get_clip(clip_id).reason == "discarded by user"
        finally:
            gate.set()
            await stop_watcher(watcher, task)


# --------------------------------------------------------------------------
# P1 #8 + #9 — capture slots and the reconnect budget
# --------------------------------------------------------------------------


class TestCaptureSlotsAndReconnects:
    async def test_an_offline_stream_does_not_hold_a_capture_slot(
        self, initialised_db: int
    ) -> None:
        settings = LiveSettings()
        settings.workers.max_concurrent_streams = 1

        def capture_factory(buf: Any, on_event: Any) -> FakeCapture:
            return FakeCapture(buf, on_event)

        manager = LiveManager(
            settings=settings,
            registry=FakeRegistry(),
            capture_factory=capture_factory,
            buffer_factory=FakeBuffer,
            audio_start=lambda *a, **k: None,
            chat_runner=lambda *a, **k: False,
        )
        offline = {"on": True}
        real_factory = manager._make_deps

        def deps_for(stream_id: str) -> WatcherDeps:
            deps = real_factory(stream_id)
            if stream_id == manager._offline_id:
                deps.capture_factory = lambda b, e: FakeCapture(
                    b, e, fail=Offline("No playable streams") if offline["on"] else None
                )
            return deps

        manager._offline_id = "pending"
        manager._make_deps = deps_for  # type: ignore[method-assign]
        await manager.start()
        try:
            store.upsert_stream(LiveStreamRow(id="parked", url="https://kick.com/a"))
            manager._offline_id = "parked"
            manager._stream_rows["parked"] = LiveStreamRow(id="parked", url="https://kick.com/a")
            manager._launch("parked", manager._stream_rows["parked"])
            await wait_until(
                lambda: (
                    manager._watchers.get("parked") is not None
                    and manager._watchers["parked"].state is StreamState.WAITING_FOR_LIVE
                )
            )
            assert "parked" not in manager._capturing
            live = await manager.add_stream("https://kick.com/b", "b")
            await wait_until(lambda: manager._watchers[live].state is StreamState.WATCHING)
            assert manager._capturing == {live}
        finally:
            await manager.shutdown()

    async def test_ffmpeg_dying_right_after_spawn_exhausts_the_budget(self) -> None:
        def dies_fast(buf: Any, on_event: Any) -> Any:
            capture = FakeCapture(buf, on_event)

            def start() -> None:
                threading.Timer(0.02, on_event, args=("error", "ffmpeg exited with code 1")).start()

            capture.start = start  # type: ignore[method-assign]
            return capture

        watcher, _ = watcher_for(
            "s-dies", capture_factory=dies_fast, max_connect_attempts=4, judge=None
        )
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.ERROR, timeout=10.0)
            assert "4 times" in watcher.state_detail
        finally:
            await stop_watcher(watcher, task)

    async def test_a_delivering_capture_resets_the_budget(self) -> None:
        class Delivering(FakeCapture):
            committed = 1  # the real capture's "a segment landed" counter

        watcher, _ = watcher_for("s-ok", capture_factory=lambda b, e: Delivering(b, e))
        watcher._connect_attempts = 3
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            await wait_until(lambda: watcher._connect_attempts == 0)
        finally:
            await stop_watcher(watcher, task)


# --------------------------------------------------------------------------
# P1 #10 — the judge cannot run away with the bill
# --------------------------------------------------------------------------


class TestJudgeCost:
    async def test_a_rejected_spike_is_judged_once(self, stream_row: str) -> None:
        watcher, calls = watcher_for(stream_row)
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(100.0))
            await wait_until(lambda: len(calls["judge"]) == 1 and not watcher._candidate_busy)
            for t in (104.0, 108.0, 110.0):  # the same moment's tail
                watcher.submit_signal(signal(t, 0.4))
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.1)
            assert len(calls["judge"]) == 1
            watcher.submit_signal(signal(200.0))  # a genuinely new moment
            await wait_until(lambda: len(calls["judge"]) == 2)
        finally:
            await stop_watcher(watcher, task)

    async def test_provider_errors_open_a_shared_breaker(self, stream_row: str) -> None:
        from autoclip.live.circuit import JudgeCircuitBreaker

        attempts: list[int] = []

        async def overloaded(label: str, words: list, signals: dict) -> LiveJudgement:
            attempts.append(1)
            raise RuntimeError("Service temporarily at capacity")

        breaker = JudgeCircuitBreaker(base_s=60.0)
        watcher, _ = watcher_for(stream_row, judge=overloaded, judge_breaker=breaker)
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(100.0))
            await wait_until(lambda: len(attempts) == 1 and not watcher._candidate_busy)
            watcher.submit_signal(signal(300.0))
            await wait_until(lambda: len(store.list_clips()) == 2 and not watcher._candidate_busy)
            assert len(attempts) == 1  # refused locally while backing off
            newest = next(c for c in store.list_clips() if c.trigger_end_s == 300.0)
            assert newest.status == "rejected" and "backing off" in newest.reason
            assert breaker.remaining() > 0
        finally:
            await stop_watcher(watcher, task)

    async def test_no_speech_is_a_local_reject(self, stream_row: str) -> None:
        from autoclip.pipeline.transcribe import TranscriptionError

        def silent(path: Path) -> Any:
            raise TranscriptionError("No speech was detected in this audio.")

        watcher, calls = watcher_for(stream_row, transcribe=silent)
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(100.0))
            await wait_until(
                lambda: store.list_clips() and store.list_clips()[0].status == "rejected"
            )
            row = store.list_clips()[0]
            assert "no speech" in row.reason
            assert calls["judge"] == []
            assert row.judge_calls == 0
        finally:
            await stop_watcher(watcher, task)

    async def test_budgets_survive_a_restart(self, stream_row: str) -> None:
        settings = LiveSettings()
        settings.rate_limit.judge_per_minute = 1
        first, calls1 = watcher_for(stream_row, settings=settings)
        task = asyncio.create_task(first.run())
        try:
            await wait_until(lambda: first.state is StreamState.WATCHING)
            first.submit_signal(signal(100.0))
            await wait_until(lambda: len(calls1["judge"]) == 1 and not first._candidate_busy)
        finally:
            await stop_watcher(first, task)

        # A brand-new watcher (as after a process restart): no in-memory
        # bucket to reset — the spend is read back from live_clips.
        second, calls2 = watcher_for(stream_row, settings=settings)
        task = asyncio.create_task(second.run())
        try:
            await wait_until(lambda: second.state is StreamState.WATCHING)
            second.submit_signal(signal(500.0))
            await wait_until(lambda: len(store.list_clips()) == 2 and not second._candidate_busy)
            assert calls2["judge"] == []
            newest = next(c for c in store.list_clips() if c.trigger_end_s == 500.0)
            assert newest.reason.startswith("rate limited") and "minute" in newest.reason
        finally:
            await stop_watcher(second, task)

    async def test_the_repair_retry_counts_as_a_second_call(
        self, stream_row: str, monkeypatch
    ) -> None:
        from autoclip.live import judge as judge_mod
        from autoclip.providers import ProviderStatus
        from autoclip.providers.base import LLMProvider

        class Scripted(LLMProvider):
            name = "scripted"
            requires_key = False

            def __init__(self) -> None:
                super().__init__("m")
                self.responses = ["not json", '{"score": 20, "reason": "dull"}']

            async def _complete(self, system: str, user: str, config: Any) -> str:
                return self.responses.pop(0)

            async def health_check(self) -> ProviderStatus:
                return ProviderStatus(name=self.name, available=True)

        from autoclip.pipeline.transcript import Word as RealWord

        provider = Scripted()
        monkeypatch.setattr(judge_mod, "build_provider", lambda: provider)
        watcher, _ = watcher_for(
            stream_row,
            transcribe=lambda path: Transcript(words=[RealWord(text="hey", start=0.0, end=0.5)]),
        )
        watcher._deps.judge = judge_mod.judge_window  # the real judge
        watcher._judge_is_default = True
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(100.0))
            await wait_until(
                lambda: store.list_clips() and store.list_clips()[0].status == "rejected"
            )
            assert store.list_clips()[0].judge_calls == 2
        finally:
            await stop_watcher(watcher, task)

    def test_every_watcher_shares_one_judge_semaphore(self) -> None:
        manager = LiveManager(settings=LiveSettings(), registry=FakeRegistry())
        a, b = manager._make_deps("a"), manager._make_deps("b")
        assert a.judge_semaphore is b.judge_semaphore is not None
        assert a.judge_breaker is b.judge_breaker


# --------------------------------------------------------------------------
# P1 #12 — pause/resume and registry state are honest
# --------------------------------------------------------------------------


class TestPauseResumeState:
    async def test_pause_persists_disabled_off_and_resume_enables(
        self, initialised_db: int
    ) -> None:
        manager = LiveManager(
            settings=LiveSettings(),
            registry=FakeRegistry(),
            capture_factory=lambda b, e: FakeCapture(b, e),
            buffer_factory=FakeBuffer,
            audio_start=lambda *a, **k: None,
            chat_runner=lambda *a, **k: False,
        )
        await manager.start()
        try:
            sid = await manager.add_stream("https://kick.com/x", "x")
            await wait_until(lambda: manager._watchers[sid].state is StreamState.WATCHING)
            await manager.pause_stream(sid)
            row = store.get_stream(sid)
            assert row.enabled is False and row.state == "off"
            assert manager.stream_snapshot(sid)["state"] == "off"
            assert manager.stream_snapshot(sid)["enabled"] is False
            await manager.resume_stream(sid)
            assert store.get_stream(sid).enabled is True
            await wait_until(lambda: manager.stream_snapshot(sid)["state"] == "watching")
        finally:
            await manager.shutdown()

    async def test_registry_disabled_streams_read_off(self, initialised_db: int) -> None:
        registry = FakeRegistry()
        registry.upsert(LiveStreamConfig(id="d1", url="https://kick.com/x", enabled=False))
        manager = LiveManager(
            settings=LiveSettings(),
            registry=registry,
            capture_factory=lambda b, e: FakeCapture(b, e),
            buffer_factory=FakeBuffer,
        )
        await manager.start()
        try:
            snap = manager.stream_snapshot("d1")
            assert snap["state"] == "off" and snap["enabled"] is False
            assert store.get_stream("d1") is not None  # DB row exists for clip FKs
        finally:
            await manager.shutdown()

    def test_a_stopping_watcher_records_only_off_or_ended(self) -> None:
        watcher, _ = watcher_for("s-stop")
        watcher.set_state(StreamState.WATCHING, "live", initial=True)
        watcher._stopping = True
        assert watcher.set_state(StreamState.RECONNECTING, "late callback") is False
        assert watcher.state is StreamState.WATCHING
        assert watcher.set_state(StreamState.OFF, "paused") is True

    async def test_cancelled_candidate_cannot_paint_reconnecting_over_off(
        self, stream_row: str
    ) -> None:
        gate = asyncio.Event()

        async def stuck(label: str, words: list, signals: dict) -> LiveJudgement:
            await gate.wait()
            return LiveJudgement(score=95)

        states: list[StreamState] = []
        watcher, _ = watcher_for(stream_row, judge=stuck)
        watcher._on_state = lambda sid, old, new, detail: states.append(new)
        task = asyncio.create_task(watcher.run())
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(signal(60.0))
        await wait_until(lambda: watcher.state is StreamState.CLIPPING)
        watcher._capture = None  # capture gone too: the old path said "reconnecting"
        await stop_watcher(watcher, task)
        gate.set()
        assert watcher.state is StreamState.OFF
        assert StreamState.RECONNECTING not in states


# --------------------------------------------------------------------------
# P1 #13 — interrupted work can be retried
# --------------------------------------------------------------------------


class TestRetry:
    async def test_failed_clip_with_a_staged_window_rerenders(self, stream_row: str) -> None:
        def produce(item: Any, buffer: Any, clip_row: Any, settings: Any) -> Any:
            assert clip_factory.window_staged(clip_row)  # rendered without a buffer
            rendered = clip_dir(item.clip_id) / "final.mp4"
            rendered.write_bytes(b"mp4")
            return _Produced(rendered)

        manager = LiveManager(
            settings=LiveSettings(),
            produce=produce,
            registry=FakeRegistry(),
            capture_factory=lambda b, e: FakeCapture(b, e),
            buffer_factory=FakeBuffer,
        )
        _queued_clip(stream_row, "c-f", status="failed", error="interrupted")
        clip_factory.stage_window(FakeBuffer(), "c-f", 100.0, 130.0)
        await manager.start()
        try:
            row = await manager.retry_clip("c-f")
            assert row.status == "queued"
            await wait_until(lambda: store.get_clip("c-f").status == "ready")
        finally:
            await manager.shutdown()

    async def test_retry_refuses_without_a_window_or_when_not_failed(self, stream_row: str) -> None:
        manager = LiveManager(settings=LiveSettings(), registry=FakeRegistry())
        _queued_clip(stream_row, "c-gone", status="failed")
        _queued_clip(stream_row, "c-ready", status="ready")
        await manager.start()
        try:
            with pytest.raises(RetryRefused, match="no longer on disk"):
                await manager.retry_clip("c-gone")
            with pytest.raises(RetryRefused, match="Only failed"):
                await manager.retry_clip("c-ready")
            with pytest.raises(KeyError):
                await manager.retry_clip("nope")
        finally:
            await manager.shutdown()

    async def test_stop_marks_queued_work_interrupted(self, stream_row: str) -> None:
        gate = threading.Event()

        def produce(item: Any, *_: Any) -> Any:
            gate.wait(timeout=10)
            return _Produced(clip_dir(item.clip_id) / "final.mp4")

        manager = LiveManager(settings=LiveSettings(), produce=produce, registry=FakeRegistry())
        await manager.start()
        try:
            first = _queued_clip(stream_row, "c-1")
            second = _queued_clip(stream_row, "c-2")
            assert manager._enqueue_clip(stream_row, first)
            await wait_until(lambda: store.get_clip("c-1").status == "rendering")
            assert manager._enqueue_clip(stream_row, second)
        finally:
            gate.set()
            await manager.shutdown()
        assert store.get_clip("c-2").status == "failed"
        assert store.get_clip("c-2").error == "interrupted"


# --------------------------------------------------------------------------
# P2 #15 — the clip window is held, and its post-roll is waited for
# --------------------------------------------------------------------------


class GrowingBuffer(FakeBuffer):
    """Reports a newest committed segment that advances over time."""

    def __init__(self, covered_s: float) -> None:
        super().__init__()
        self.covered = covered_s

    def covered_until_s(self) -> float:
        return self.covered


class TestWindowHeldUntilRender:
    async def test_cut_waits_for_the_post_roll_and_stages_the_window(self, stream_row: str) -> None:
        buffer = GrowingBuffer(covered_s=120.0)  # the trigger is the newest audio

        async def accept(label: str, words: list, signals: dict) -> LiveJudgement:
            return LiveJudgement(score=95, title="Kept")

        watcher, calls = watcher_for(stream_row, judge=accept, buffer=buffer)
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(119.0))
            await asyncio.sleep(0.3)
            assert buffer.extract_calls == []  # post-roll (to 129 s) not captured yet
            buffer.covered = 132.0
            await wait_until(lambda: len(calls["enqueued"]) == 1)
            item = calls["enqueued"][0]
            assert buffer.pins and buffer.pins[0] == int(item.clip_start_s // SEGMENT_S)
            assert buffer.unpins == buffer.pins  # released once staged
            row = store.get_clip(item.clip_id)
            assert row.status == "queued"
            assert clip_factory.window_staged(row)  # renderable without the buffer
            assert item.source_path == clip_dir(item.clip_id) / "raw.ts"
        finally:
            await stop_watcher(watcher, task)

    async def test_a_dead_capture_stops_the_wait(self, stream_row: str) -> None:
        buffer = GrowingBuffer(covered_s=120.0)
        watcher, calls = watcher_for(stream_row, buffer=buffer, coverage_timeout_s=30.0)
        task = asyncio.create_task(watcher.run())
        try:
            await wait_until(lambda: watcher.state is StreamState.WATCHING)
            watcher.submit_signal(signal(119.0))
            await wait_until(lambda: watcher.state is StreamState.CLIPPING)
            watcher._capture = None  # the stream died: nothing more will arrive
            await wait_until(lambda: len(calls["judge"]) == 1, timeout=3.0)
        finally:
            await stop_watcher(watcher, task)


# --------------------------------------------------------------------------
# P2 #17 — teardown never freezes the event loop
# --------------------------------------------------------------------------


class TestTeardownKeepsTheLoopResponsive:
    async def test_slow_signal_threads_do_not_stall_the_loop(self) -> None:
        release = threading.Event()

        def slow_chat(url: str, settings: Any, *, on_signal: Any, stop_event: Any, clock: Any):
            release.wait(timeout=3.0)  # a reader stuck in recv()
            return True

        deps = WatcherDeps(
            capture_factory=lambda b, e: FakeCapture(b, e),
            buffer_factory=FakeBuffer,
            aggregator_factory=lambda: None,
            audio_start=lambda *a, **k: None,
            chat_runner=slow_chat,
            publish=lambda *a, **k: None,
            enqueue_clip=lambda item: True,
        )
        watcher = StreamWatcher("s-loop", "https://twitch.tv/chan", "x", LiveSettings(), deps=deps)
        task = asyncio.create_task(watcher.run())
        await wait_until(lambda: watcher.state is StreamState.WATCHING)
        gaps: list[float] = []

        async def ticker() -> None:
            last = time.monotonic()
            while True:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        tick = asyncio.create_task(ticker())
        try:
            watcher.stop()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait({task}, timeout=10.0)
        finally:
            release.set()
            tick.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tick
        assert max(gaps) < 0.5, f"event loop stalled {max(gaps):.2f}s during teardown"
