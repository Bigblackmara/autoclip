"""Live Watch round-2 regressions (second bug-hunt pass).

Everything is faked: no network, no GPU, no real keys, no real data dirs.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from autoclip.live import chat_signals as cs
from autoclip.live.judge import LiveJudgement
from autoclip.live.manager import LiveManager
from autoclip.live.models import UPLOADS_IN_FLIGHT, LiveClipRow, LiveUploadRow
from autoclip.live.settings import LiveChatSettings, LiveSettings
from autoclip.live.signals import SignalKind
from autoclip.live.watcher import ClipWorkItem, ClipWorkResult, StreamWatcher, WatcherDeps

# --------------------------------------------------------------------------
# Chat is not scored while the stream is offline
# --------------------------------------------------------------------------


class TestChatPausedWhileOffline:
    def test_reader_skips_messages_while_the_clock_says_not_capturing(self) -> None:
        """Offline chat must not pile into one bucket and fire on reconnect.

        Stream time stands still while the stream is offline; scoring the
        whole offline chat at that frozen instant made a hypetrain (and a
        giant rate bucket) out of ordinary chatter — a paid judge call on
        every reconnect.
        """
        signals: list[Any] = []
        now: list[float | None] = [None]
        reader = cs.TwitchChatReader(
            "chan",
            LiveChatSettings(hypetrain_msgs_per_min=20.0),
            on_signal=signals.append,
            clock=lambda: now[0],
        )
        for _ in range(200):  # ten minutes of offline chat, one frozen instant
            reader._observe("viewer", "hi")  # noqa: SLF001
        assert signals == []
        now[0] = 600.0  # live again
        for i in range(3):
            now[0] = 600.0 + i
            reader._observe("viewer", "hi")  # noqa: SLF001
        hype = [s for s in signals if s.kind is SignalKind.CHAT_HYPE]
        assert hype == [], "offline messages leaked into the live velocity window"

    def test_watcher_chat_clock_is_none_without_a_capture(self) -> None:
        class Seg:
            def __init__(self, index: int) -> None:
                self.index = index

        class Buf:
            segments = [Seg(9)]

        watcher = StreamWatcher(
            "s-chat", "https://twitch.tv/chan", "x", LiveSettings(), deps=WatcherDeps()
        )
        watcher._buffer = Buf()  # noqa: SLF001
        assert watcher._chat_clock() is None  # noqa: SLF001 - offline / reconnecting
        watcher._capture = object()  # noqa: SLF001
        assert watcher._chat_clock() == 9 * 6.0  # noqa: SLF001
        watcher._stopping = True  # noqa: SLF001
        assert watcher._chat_clock() is None  # noqa: SLF001


# --------------------------------------------------------------------------
# Auto-upload: off the render worker, outcome recorded by the upload thread
# --------------------------------------------------------------------------


class _UploadStore:
    """Minimal live.store double for the render + auto-upload path."""

    def __init__(self) -> None:
        self.clips: dict[str, LiveClipRow] = {}
        self.uploads: dict[str, LiveUploadRow] = {}

    def get_clip(self, clip_id: str) -> LiveClipRow | None:
        return self.clips.get(clip_id)

    def update_clip_guarded(self, clip_id: str, **fields: Any) -> bool:
        row = self.clips.get(clip_id)
        if row is None or row.status == "rejected":
            return False
        for key, value in fields.items():
            setattr(row, key, value)
        return True

    def create_upload(self, row: LiveUploadRow) -> LiveUploadRow:
        self.uploads[row.id] = row
        return row

    def update_upload(self, upload_id: str, **fields: Any) -> None:
        for key, value in fields.items():
            setattr(self.uploads[upload_id], key, value)

    def list_uploads(self, clip_id: str | None = None, limit: int = 200) -> list[LiveUploadRow]:
        return [u for u in self.uploads.values() if clip_id in (None, u.clip_id)]

    def list_clips(self, *_: Any, **__: Any) -> list[LiveClipRow]:
        return list(self.clips.values())


def _work_item(clip_id: str) -> ClipWorkItem:
    return ClipWorkItem(
        clip_id=clip_id,
        stream_id="s1",
        label="stream",
        source_path=Path("unused.ts"),
        clip_start_s=0.0,
        clip_end_s=10.0,
        trigger_start_s=0.0,
        trigger_end_s=10.0,
        judgement=LiveJudgement(score=90, title="Huge"),
    )


@pytest.fixture
def blocking_youtube(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """upload_youtube faked: blocks until ``release`` is set, then succeeds."""
    import autoclip.live.uploads.youtube as yt_mod

    state: dict[str, Any] = {"release": threading.Event(), "calls": 0}

    @dataclass
    class Result:
        url: str = "https://example.invalid/v1"
        media_id: str = "v1"

    async def fake(path: Any, meta: Any) -> Result:
        state["calls"] += 1
        await asyncio.to_thread(state["release"].wait, 10)
        return Result()

    monkeypatch.setattr(yt_mod, "upload_youtube", fake)
    return state


def _upload_manager(store: _UploadStore, events: list) -> LiveManager:
    settings = LiveSettings()
    settings.uploads.youtube.auto_upload = True

    def produce(item: Any, *_: Any) -> Any:
        return ClipWorkResult(rendered_path=Path("final.mp4"), title="Huge")

    return LiveManager(
        settings=settings,
        produce=produce,
        store_mod=store,
        registry=None,
        publish=lambda channel, kind, data: events.append((kind, dict(data or {}))),
    )


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


class TestAutoUploadRound2:
    async def test_render_worker_is_not_held_by_the_upload(self, blocking_youtube) -> None:
        store = _UploadStore()
        store.clips["c1"] = LiveClipRow(id="c1", stream_id="s1", status="queued")
        events: list = []
        manager = _upload_manager(store, events)
        try:
            await asyncio.wait_for(manager._render_one(_work_item("c1")), timeout=3.0)  # noqa: SLF001
            assert store.clips["c1"].status == "ready"
            await _until(lambda: blocking_youtube["calls"] == 1)
            # The upload runs on; the UI was told at once.
            [row] = store.uploads.values()
            assert row.status == "uploading"
            assert (
                "upload",
                {"id": row.id, "clip_id": "c1", "platform": "youtube", "status": "uploading"},
            ) in events
        finally:
            blocking_youtube["release"].set()
            await asyncio.gather(*manager._upload_tasks, return_exceptions=True)  # noqa: SLF001
        assert row.status == "done" and row.upload_url == "https://example.invalid/v1"

    async def test_cancelled_upload_still_records_its_outcome(self, blocking_youtube) -> None:
        """stop() cancels the awaiting task; the finished upload is recorded.

        Before: the row stayed 'uploading', a restart turned it into
        'failed: interrupted' and a retry posted the clip a second time.
        """
        store = _UploadStore()
        manager = _upload_manager(store, [])
        clip = LiveClipRow(id="c2", stream_id="s1", status="ready", rendered_path="final.mp4")
        store.clips["c2"] = clip
        task = asyncio.create_task(manager._maybe_auto_upload(_work_item("c2"), clip, "t"))  # noqa: SLF001
        await _until(lambda: blocking_youtube["calls"] == 1)
        assert ("c2", "youtube") in UPLOADS_IN_FLIGHT
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        blocking_youtube["release"].set()
        [row] = store.uploads.values()
        await _until(lambda: row.status != "uploading")
        assert row.status == "done"
        await _until(lambda: ("c2", "youtube") not in UPLOADS_IN_FLIGHT)

    async def test_manual_upload_claim_blocks_the_auto_upload(self, blocking_youtube) -> None:
        store = _UploadStore()
        manager = _upload_manager(store, [])
        clip = LiveClipRow(id="c3", stream_id="s1", status="ready", rendered_path="final.mp4")
        store.clips["c3"] = clip
        UPLOADS_IN_FLIGHT.add(("c3", "youtube"))  # the manual route is mid-upload
        try:
            await manager._maybe_auto_upload(_work_item("c3"), clip, "t")  # noqa: SLF001
        finally:
            UPLOADS_IN_FLIGHT.discard(("c3", "youtube"))
        assert store.uploads == {} and blocking_youtube["calls"] == 0


# --------------------------------------------------------------------------
# Windows: capture ffmpeg dies with the backend (no orphans after a crash)
# --------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
class TestCaptureDiesWithTheHost:
    def _sleeper(self) -> subprocess.Popen:
        return subprocess.Popen(  # noqa: S603 - fixed argv, a harmless sleeper
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    def test_closing_the_job_kills_a_detached_child(self) -> None:
        import ctypes

        from autoclip.live import capture

        proc = self._sleeper()
        try:
            job = capture._create_kill_on_close_job()  # noqa: SLF001
            capture._assign_to_job(job, proc)  # noqa: SLF001
            assert proc.poll() is None
            # What the OS does when the backend process dies: the last
            # handle to its job closes.
            ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(job))
            assert proc.wait(timeout=10) is not None
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_real_spawns_are_bound_to_the_host_job(self) -> None:
        from autoclip.live import capture

        proc = self._sleeper()
        try:
            assert capture.bind_to_host_lifetime(proc) is True
            assert capture._HOST_JOB is not None  # noqa: SLF001
        finally:
            proc.kill()
            proc.wait(timeout=10)


# --------------------------------------------------------------------------
# SSE: subscribed before the snapshot; a slow client is bounded + resynced
# --------------------------------------------------------------------------


class _ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


class TestSSEFeed:
    async def test_subscribes_before_the_snapshot_and_resyncs_on_overflow(self) -> None:
        from autoclip.api import live as api_live
        from autoclip.live import events

        channels = [api_live.CLIPS_CHANNEL, api_live.STREAMS_CHANNEL]
        gen = api_live._sse_generator(_ConnectedRequest(), channels)  # noqa: SLF001
        try:
            first = await gen.__anext__()
            assert first["event"] == "snapshot"
            # The client refetches on the snapshot: anything published from
            # here on must already reach this feed.
            assert events.broker.subscriber_count(api_live.CLIPS_CHANNEL) == 1
            assert events.broker.subscriber_count(api_live.STREAMS_CHANNEL) == 1

            # A client that stops reading: events past the bound are dropped
            # (memory stays bounded) and the client is told to resync rather
            # than silently missing a state change.
            for burst in range(8):
                for i in range(60):
                    events.publish(
                        api_live.CLIPS_CHANNEL, "clip_ready", {"clip_id": f"{burst}-{i}"}
                    )
                await asyncio.sleep(0.05)
            names = []
            for _ in range(400):
                frame = await asyncio.wait_for(gen.__anext__(), timeout=5.0)
                names.append(frame["event"])
                if frame["event"] == "snapshot":
                    break
            assert names[-1] == "snapshot"
            assert len(names) < 300, "the whole backlog was buffered for a stalled client"
            events.publish(api_live.CLIPS_CHANNEL, "clip_ready", {"clip_id": "after"})
            frame = await asyncio.wait_for(gen.__anext__(), timeout=5.0)
            assert frame["event"] == "clip_status" and '"after"' in frame["data"]
        finally:
            await gen.aclose()
            await asyncio.sleep(0.05)
        assert events.broker.subscriber_count(api_live.CLIPS_CHANNEL) == 0


# --------------------------------------------------------------------------
# stream_state frames: a cleared error is sent as null; removal is explicit
# --------------------------------------------------------------------------


class _Registry:
    def list_streams(self) -> list:
        return []

    def upsert(self, cfg: Any) -> None:
        pass

    def delete(self, stream_id: str) -> None:
        pass

    def set_enabled(self, stream_id: str, enabled: bool) -> None:
        pass


class TestStreamStateFrames:
    async def test_cleared_error_is_sent_and_removal_is_flagged(self, initialised_db) -> None:
        frames: list[dict] = []
        manager = LiveManager(
            settings=LiveSettings(),
            registry=_Registry(),
            publish=lambda channel, kind, data: (
                frames.append(dict(data or {})) if kind == "stream_state" else None
            ),
        )
        sid = await manager.add_stream("https://twitch.tv/chan", enabled=False)
        assert "last_error" in frames[-1] and frames[-1]["last_error"] is None
        await manager.remove_stream(sid)
        assert frames[-1]["stream_id"] == sid and frames[-1]["removed"] is True
