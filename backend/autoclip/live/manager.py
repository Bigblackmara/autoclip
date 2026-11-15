"""LiveManager — the single Live Watch entry point (A8 owns bodies).

Owns: the watcher task registry, add/remove/pause/resume, the bounded clip
queue + its render workers, capture capacity, the shared judge concurrency
semaphore and provider circuit breaker, the feature-wide disk budget, event
publishing, crash recovery, and applying settings saves to the running
feature.

Public API (what A10 / the API layer codes against; all coroutines):

    await manager.start()                        # idempotent; loads persisted
                                                # streams, launches watchers
    await manager.stop()                        # cancel all, kill captures
    await manager.apply_settings(settings)      # a settings save, live
    sid = await manager.add_stream(url, label="")     # register + watch
    await manager.remove_stream(sid)                  # stop + delete (+ files)
    await manager.pause_stream(sid)                   # stop, keep config
    await manager.resume_stream(sid)                   # fresh watcher start
    await manager.retry_clip(clip_id)                 # re-render a failed clip
    snapshot = manager.status()     # {"enabled", "running", "watching",
                                    #  "streams": [...], "caps": {...}}
    snap = manager.stream_snapshot(sid)   # one stream's API-facing view

Construction parameters (all optional, all injectable for tests):

    settings=None      LiveSettings; defaults to config.json's ``live``
                       section merged over the frozen defaults (re-read on
                       every start()).
    produce=None       the clip renderer: (ClipWorkItem, buffer, clip_row,
                       app_settings) -> producer result, or any sync/async
                       callable accepting the item. Defaults to a wrapper
                       over A8b's clip_factory.produce (the standard render
                       path); inject a fake in tests.
    store_mod=None     live.store module (or a compatible test double).
    registry=None      A3's persisted registry; None = auto (module import).
    capture_factory= / buffer_factory= / aggregator_factory= / trigger= /
    cooldown= / judge_minute= / judge_hour= / judge_day= / transcribe= /
    judge= / audio_start= / chat_runner= / publish=
                       forwarded verbatim into each watcher's WatcherDeps,
                       so tests fake every sibling contract without
                       monkeypatching module internals.

Judge concurrency mirrors the pipeline's highlights stage
(HOSTED_CONCURRENCY=3 / LOCAL_CONCURRENCY=1): the active provider decides —
a hosted provider (anthropic/openai/gemini) may run up to 3 concurrent
judge calls across streams; ollama (local) serializes to 1. ONE semaphore
lives on the manager and every watcher shares it.

Guarantees the API layer relies on:
- start() / stop() are idempotent; stop() cancels every task and awaits
  each unwind bounded, so no watcher, capture, or render leaks past it.
- add/remove/pause/resume are safe on a not-started manager (registration
  still lands; listing works; watchers launch on the next start()).
- One watcher task per stream at most; a crashed watcher is reaped without
  ever taking the manager down.
- pause persists ``enabled=0``/``state=off``; remove deletes every persisted
  trace of the stream, including its buffer folder and clip folders.
- A discarded clip (status ``rejected``) is a tombstone: no worker writes
  it back to life, renders it, or uploads it.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from . import events
from .buffer import PENDING_CAP_BYTES
from .circuit import JudgeCircuitBreaker
from .models import UPLOADS_IN_FLIGHT, LiveClipRow, LiveStreamRow, LiveUploadRow, utcnow
from .settings import (
    LiveSettings,
    LiveStreamConfig,
    load_live_settings,
    parse_live_section,
    update_in_place,
)
from .states import StreamState
from .watcher import (
    CLIPS_CHANNEL,
    STREAMS_CHANNEL,
    ClipWorkItem,
    StreamWatcher,
    WatcherDeps,
    publish_event,
    store_call,
    stream_channel,
)

log = logging.getLogger(__name__)

#: Judge-call concurrency, mirroring pipeline/highlights.py. Hosted
#: providers parallelize; a local ollama process serializes (vLLM-style
#: single-slot serving) — one slow local call must not queue behind
#: itself.
HOSTED_JUDGE_CONCURRENCY = 3
LOCAL_JUDGE_CONCURRENCY = 1

#: live_clips statuses whose window/intermediate files still hold disk.
PENDING_CLIP_STATUSES = ("detecting", "queued", "rendering")

#: Statuses whose artifacts the disk janitor may delete (terminal, not ready).
_RECLAIMABLE_STATUSES = ("failed", "rejected")

#: How long remove/pause wait for a watcher's teardown (it is never
#: cancelled twice: the capture stop inside it must complete).
TEARDOWN_WAIT_S = 20.0

#: Disk janitor cadence, and the age an unowned folder must reach before it
#: is treated as garbage (never race a candidate creating its folder).
JANITOR_INTERVAL_S = 300.0
ORPHAN_MIN_AGE_S = 600.0

#: Auto-uploads running at once (each one is a long network transfer).
AUTO_UPLOAD_CONCURRENCY = 2

#: Raised to the API when Live Watch is switched off in settings.
LIVE_OFF_DETAIL = "Live Watch is turned off. Turn it on in Live settings."


class RetryRefused(RuntimeError):
    """A clip cannot be re-rendered (not failed, window gone, queue full)."""


def _load_settings() -> LiveSettings:
    """LiveSettings from config.json's ``live`` section, defaults elsewhere."""
    try:
        return load_live_settings()
    except Exception:
        log.warning("Falling back to default LiveSettings.", exc_info=True)
        return LiveSettings()


def json_read_live_section(raw: Any) -> LiveSettings:
    """Validate the raw ``live`` config value (or None) into LiveSettings."""
    return parse_live_section(raw)


def _default_store_mod():
    from . import store

    return store


def _default_registry() -> Any:
    """A3's persisted registry module, or None when it cannot import."""
    try:
        from . import registry as registry_mod
    except ImportError:
        log.info("Stream registry unavailable (A3 pending): streams persist in DB only.")
        return None
    return registry_mod


def _file_size(path_repr: Any) -> int:
    if not path_repr:
        return 0
    try:
        return int(Path(str(path_repr)).stat().st_size)
    except OSError:
        return 0


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            with contextlib.suppress(OSError):
                if entry.is_file():
                    total += entry.stat().st_size
    except OSError:
        pass
    return total


def _rmtree(path: Path, attempts: int = 3) -> bool:
    """Delete a folder tree; retried (Windows file locks); True when gone."""
    for attempt in range(attempts):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return True
        time.sleep(0.2 * (attempt + 1))
    log.warning("Could not fully delete %s (files still in use).", path)
    return False


def _judge_concurrency(app_settings: Any) -> int:
    """Mirror highlights: hosted providers 3, local (ollama) 1."""
    try:
        name = str(getattr(app_settings, "active_provider", "") or "")
    except Exception:
        name = ""
    return LOCAL_JUDGE_CONCURRENCY if name == "ollama" else HOSTED_JUDGE_CONCURRENCY


def _default_produce(
    item: ClipWorkItem,
    buffer: Any,
    clip_row: Any,
    app_settings: Any,
) -> Any:
    """Standard render path: A8b's clip_factory (reuses the staged window).

    Wrapped (never called directly by the manager) so a failing import or a
    stub factory lands as a failed clip row instead of a worker crash.
    """
    from . import clip_factory  # A8b owns this module

    return clip_factory.produce(buffer, clip_row, app_settings)


def _run_producer(
    produce: Any, item: ClipWorkItem, buffer: Any, clip_row: Any, settings: Any
) -> Any:
    """Run the injected producer: sync inline, async in a private loop.

    Runs inside a worker thread (the manager awaits it via to_thread): the
    real producer (A8b) drives ffmpeg and the pipeline synchronously, and
    blocking the manager's loop would stall every watcher heartbeat.
    Accepting either a sync or async callable keeps this seam from
    dictating A8b's internals.
    """
    if produce is None:
        raise NotImplementedError("no clip producer configured")
    spec = None
    try:
        spec = inspect.signature(produce).parameters
    except (TypeError, ValueError):
        spec = None
    # Accept 1-arg (item), 4-arg (item, buffer, clip_row, settings) and
    # dataclass-style producers; the default 4-arg factory is the seam A8b
    # landed against, the 1-arg shape is what tests fake.
    accepts = spec is not None and len(spec) >= 4
    if inspect.iscoroutinefunction(produce):
        import asyncio as _asyncio

        if accepts:
            return _asyncio.run(produce(item, buffer, clip_row, settings))
        return _asyncio.run(produce(item))
    if accepts:
        return produce(item, buffer, clip_row, settings)
    return produce(item)


def _needs_restart(old: LiveSettings, new: LiveSettings) -> bool:
    """Settings that are bound at watcher start and cannot change in place."""
    return (
        old.detection.chat.enabled != new.detection.chat.enabled
        or old.detection.audio.baseline_window_s != new.detection.audio.baseline_window_s
        or old.rolling_buffer.segment_s != new.rolling_buffer.segment_s
    )


class LiveManager:
    """Owns every watcher task, the clip queue, and the whole-feature snapshot.

    The API layer (A10) talks ONLY to this class — never to a watcher, the
    capture layer, or the store directly.
    """

    def __init__(
        self,
        *,
        settings: LiveSettings | None = None,
        produce: Any = None,
        store_mod: Any = None,
        registry: Any = None,
        capture_factory: Any = None,
        buffer_factory: Any = None,
        aggregator_factory: Any = None,
        trigger: Any = None,
        cooldown: Any = None,
        judge_minute: Any = None,
        judge_hour: Any = None,
        judge_day: Any = None,
        transcribe: Any = None,
        audio_extract: Any = None,
        judge: Any = None,
        audio_start: Any = None,
        chat_runner: Any = None,
        publish: Any = None,
        judge_semaphore: Any = None,
    ) -> None:
        #: Injected settings are authoritative; otherwise config.json is
        #: re-read on every start() (the singleton is built at import time).
        self._settings_injected = settings is not None
        self._live = settings if settings is not None else _load_settings()
        self._store = store_mod if store_mod is not None else _default_store_mod()
        self._produce = produce
        self._registry = registry

        # Forwarded verbatim into every watcher's WatcherDeps so one shared
        # trigger / circuit / judge / publisher can serve all streams.
        self._inject: dict[str, Any] = {
            "capture_factory": capture_factory,
            "buffer_factory": buffer_factory,
            "aggregator_factory": aggregator_factory,
            "trigger": trigger,
            "cooldown": cooldown,
            "judge_minute": judge_minute,
            "judge_hour": judge_hour,
            "judge_day": judge_day,
            "transcribe": transcribe,
            "audio_extract": audio_extract,
            "judge": judge,
            "audio_start": audio_start,
            "chat_runner": chat_runner,
            "publish": publish,
        }

        self._watchers: dict[str, StreamWatcher] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._capturing: set[str] = set()
        self._stream_rows: dict[str, LiveStreamRow] = {}
        self._started = False
        #: Set by start() even when the feature is off: the host (serve)
        #: allows Live Watch to run, so enabling it later starts it live.
        self._host_ready = False
        self._semaphore_injected = judge_semaphore is not None
        self._judge_semaphore: Any = judge_semaphore
        #: One provider-error breaker for every stream (same provider/key).
        self._breaker = JudgeCircuitBreaker()
        # Loop-bound state, created in start() — asyncio primitives bind to
        # the running loop (see jobs/queue.py for the same reasoning).
        self._queue: asyncio.Queue[ClipWorkItem | None] | None = None
        self._workers: list[asyncio.Task] = []
        self._busy_workers: set[asyncio.Task] = set()
        self._janitor: asyncio.Task | None = None
        #: Auto-uploads run beside the render workers, never inside them.
        self._upload_tasks: set[asyncio.Task] = set()
        self._upload_slots: asyncio.Semaphore | None = None
        self._lifecycle: asyncio.Lock | None = None
        #: Disk budget verdict from the last janitor pass.
        self._disk_over_budget = False
        self._disk_reason = ""

    # -- wiring ----------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """The ``live.enabled`` feature switch as currently applied."""
        return bool(self._live.enabled)

    @property
    def settings(self) -> LiveSettings:
        return self._live

    def _resolve_judge_semaphore(self) -> Any:
        """The ONE judge semaphore every watcher shares (hosted 3, ollama 1)."""
        if self._judge_semaphore is not None:
            return self._judge_semaphore
        concurrency = HOSTED_JUDGE_CONCURRENCY
        try:
            from .. import config

            concurrency = _judge_concurrency(config.load())
        except Exception:
            log.warning(
                "Could not read active provider; defaulting judge concurrency to %d.",
                HOSTED_JUDGE_CONCURRENCY,
                exc_info=True,
            )
        self._judge_semaphore = asyncio.Semaphore(concurrency)
        return self._judge_semaphore

    def _make_deps(self, stream_id: str) -> WatcherDeps:
        """One WatcherDeps per watcher: injections over manager-owned hooks.

        ``publish`` is dropped from the injections — a caller-injected
        publisher is captured by ``self._publish`` (the manager's
        choke-point), so re-forwarding it here would collide with the
        explicit ``publish=`` keyword below.
        """
        fresh = {k: v for k, v in self._inject.items() if v is not None and k != "publish"}

        def enqueue_clip(item: ClipWorkItem) -> bool:
            return self._enqueue_clip(stream_id, item)

        return WatcherDeps(
            enqueue_clip=enqueue_clip,
            may_capture=lambda: self._reserve_capture(stream_id),
            release_capture=lambda: self._release_capture(stream_id),
            publish=self._publish,
            judge_semaphore=self._resolve_judge_semaphore(),
            judge_breaker=self._breaker,
            admit_candidate=self._admit_candidate,
            **fresh,
        )

    def _publish(self, channel: str, event_type: str, data: dict | None = None) -> None:
        """Single publish choke-point: injected publisher or the stub-safe one."""
        publisher = self._inject.get("publish")
        if publisher is not None:
            publisher(channel, event_type, data)
        else:
            publish_event(channel, event_type, data)

    def _publish_stream_state(self, stream_id: str) -> None:
        """``stream_state`` on the shared streams feed (the global SSE feed)."""
        row = self._stream_rows.get(stream_id)
        try:
            snapshot = self.stream_snapshot(stream_id)
        except KeyError:
            snapshot = None
        payload: dict[str, Any] = {
            "type": "stream_state",
            "stream_id": stream_id,
            "state": (snapshot or {}).get("state") or (row.state if row else "ended"),
            "enabled": bool((snapshot or {}).get("enabled", row.enabled if row else False)),
        }
        if snapshot is not None:
            payload["state_detail"] = snapshot.get("detail") or ""
            # Always sent, null included: a patch without the key kept a
            # cleared error on screen until the next poll.
            payload["last_error"] = snapshot.get("error") or None
        with contextlib.suppress(Exception):
            self._publish(STREAMS_CHANNEL, "stream_state", payload)

    def _lifecycle_lock(self) -> asyncio.Lock:
        if self._lifecycle is None:
            self._lifecycle = asyncio.Lock()
        return self._lifecycle

    # -- lifecycle -------------------------------------------------------------

    async def start(self, *, reload_settings: bool = True) -> None:
        """Load persisted streams and launch their watchers. Idempotent.

        A no-op while the feature flag ``live.enabled`` is off: watchers,
        capture subprocesses, and clip workers must not exist at all in that
        state, not merely idle — a disabled feature should cost nothing. The
        call still records that the host allows Live Watch, so turning the
        flag on later (apply_settings) starts it without a restart.
        Settings are re-read from config.json first (unless they were
        injected, or ``apply_settings`` is the caller with fresher ones).
        """
        self._host_ready = True
        if self._started:
            return
        if reload_settings and not self._settings_injected:
            update_in_place(self._live, _load_settings())
        if not self._live.enabled:
            log.info("Live Watch disabled in settings; manager not started.")
            return
        self._started = True
        loop = asyncio.get_running_loop()
        # Bind events.publish to this loop (A10's broker marshals threads
        # onto it); stub-safe — a pending A10 is skipped inside publish_event.
        with contextlib.suppress(Exception):
            events.bind_loop(loop)
        if not self._semaphore_injected:
            self._judge_semaphore = None  # rebuilt for this loop below
        self._resolve_judge_semaphore()
        # Unbounded queue: the configured bound is enforced in
        # _enqueue_clip against the CURRENT settings (resizable live).
        self._queue = asyncio.Queue()
        # Crash recovery FIRST (A9 contract): rows a previous process left
        # mid-flight are reset before this one asserts new states over them.
        await asyncio.to_thread(store_call, self._store.reset_interrupted_states)
        self._resize_workers()
        self._load_and_launch()
        self._janitor = loop.create_task(self._janitor_loop(), name="live-disk-janitor")
        log.info("LiveManager started: %d stream(s) registered.", len(self._stream_rows))

    async def stop(self) -> None:
        """Cancel everything this manager owns and await the unwind. Idempotent."""
        awaitables: list[asyncio.Task] = []
        stopping_ids = list(self._tasks)
        for stream_id, task in list(self._tasks.items()):
            watcher = self._watchers.get(stream_id)
            if watcher is not None:
                with contextlib.suppress(Exception):
                    watcher.stop()  # flags stopping + kills the capture
            if task is not None and not task.done():
                task.cancel()
                awaitables.append(task)
        self._tasks.clear()
        if self._janitor is not None and not self._janitor.done():
            self._janitor.cancel()
            awaitables.append(self._janitor)
        self._janitor = None
        # Auto-uploads are only cancelled, not awaited: an upload already
        # running in its thread finishes and records its own outcome.
        for task in list(self._upload_tasks):
            task.cancel()
        self._upload_slots = None
        queue = self._queue
        if queue is not None:
            # Queued (never started) work is interrupted, not silently lost:
            # the row says so, and a staged window can be retried later.
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is not None:
                    self._mark_interrupted(item.clip_id)
            # Poison the queue: idle workers exit on the sentinel without
            # waiting out their get().
            for _ in self._workers:
                queue.put_nowait(None)
        for worker in self._workers:
            if not worker.done():
                worker.cancel()
                awaitables.append(worker)
        if awaitables:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait(awaitables, timeout=TEARDOWN_WAIT_S)
        self._workers = []
        self._watchers.clear()
        self._capturing.clear()
        was_started = self._started
        self._queue = None
        self._started = False
        if was_started:
            for stream_id in stopping_ids:
                row = self._stream_rows.get(stream_id)
                if row is not None:
                    row.state = StreamState.OFF.value
                self._publish_stream_state(stream_id)
        self._stream_rows.clear()
        log.info("LiveManager stopped.")

    async def shutdown(self) -> None:
        """Host going away: stop, and do not start again on a settings save."""
        self._host_ready = False
        await self.stop()

    def _mark_interrupted(self, clip_id: str) -> None:
        from .store import INTERRUPTED

        with contextlib.suppress(Exception):
            self._guarded_update(clip_id, status="failed", error=INTERRUPTED, reason=INTERRUPTED)

    async def apply_settings(self, new: LiveSettings) -> dict:
        """Apply a settings save to the RUNNING feature; returns what happened.

        The manager's settings tree is updated in place, so every watcher,
        detector and scorer holding a reference sees the new thresholds,
        rolls and budgets immediately. Beyond that:

        - ``enabled`` off → the whole feature stops (captures killed);
          on → it starts (when the host allows Live Watch at all).
        - render workers are added/retired to match ``clip_queue_workers``;
          the queue bound and capture capacity are read live.
        - settings bound at watcher start (chat on/off, the audio baseline
          window, segment length) restart the watchers.
        """
        async with self._lifecycle_lock():
            restart = self._started and _needs_restart(self._live, new)
            update_in_place(self._live, new)
            if not self._live.enabled:
                if self._started:
                    await self.stop()
                return {"running": False, "restarted": False}
            if not self._started:
                if self._host_ready:
                    await self.start(reload_settings=False)
                return {"running": self._started, "restarted": False}
            for watcher in list(self._watchers.values()):
                with contextlib.suppress(Exception):
                    watcher.update_settings(self._live)
            self._resize_workers()
            if restart:
                await self._restart_watchers()
            return {"running": True, "restarted": restart}

    async def _restart_watchers(self) -> None:
        """Fresh lifecycles for every running watcher (settings rebind)."""
        for stream_id in list(self._tasks):
            row = self._stream_rows.get(stream_id)
            await self._stop_watcher(stream_id)
            if row is not None and row.enabled and self._started:
                self._launch(stream_id, row)

    def _resize_workers(self) -> None:
        """Match the render worker count to ``workers.clip_queue_workers``."""
        if self._queue is None:
            return
        self._workers = [w for w in self._workers if not w.done()]
        wanted = max(1, int(self._live.workers.clip_queue_workers))
        loop = asyncio.get_running_loop()
        while len(self._workers) < wanted:
            self._workers.append(loop.create_task(self._clip_worker(), name="live-clip-worker"))
        # Surplus: idle workers go now; a busy one retires after its item.
        for worker in list(reversed(self._workers)):
            if len(self._workers) <= wanted:
                break
            if worker not in self._busy_workers:
                worker.cancel()
                self._workers.remove(worker)

    # -- stream registry operations --------------------------------------------

    async def add_stream(self, url: str, label: str = "", *, enabled: bool = True) -> str:
        """Register a stream, persist it, and (when started+enabled) watch it.

        Registration works on a not-started manager too: the row and the
        registry entry land, and the watcher launches on the next start().
        """
        if not url.strip():
            raise ValueError("Stream URL must not be empty.")
        row = LiveStreamRow(
            id=new_stream_id(),
            url=url,
            label=label or "",
            enabled=enabled,
            state=StreamState.WAITING_FOR_LIVE.value if enabled else StreamState.OFF.value,
        )
        await asyncio.to_thread(store_call, self._store.upsert_stream, row)
        self._registry_upsert(row)
        self._stream_rows[row.id] = row
        self._publish(stream_channel(row.id), "stream_added", self._row_event(row))
        if enabled and self._started:
            self._launch(row.id, row)
        self._publish_stream_state(row.id)
        return row.id

    def _known(self, stream_id: str) -> bool:
        if stream_id in self._stream_rows or stream_id in self._tasks:
            return True
        with contextlib.suppress(Exception):
            if store_call(self._store.get_stream, stream_id) is not None:
                return True
        return self._registry_row(stream_id) is not None

    async def _stop_watcher(self, stream_id: str, *, close_buffer: bool = False) -> None:
        """Stop one watcher and await its full teardown (bounded, uncancelled)."""
        task = self._tasks.pop(stream_id, None)
        watcher = self._watchers.pop(stream_id, None)
        if watcher is not None:
            with contextlib.suppress(Exception):
                watcher.stop()  # stopping flag makes late callbacks no-ops
        if task is not None and not task.done():
            task.cancel()
            # asyncio.wait, NOT wait_for: a timeout must not cancel the
            # teardown a second time — the capture stop inside it has to
            # finish before the stream's folder is reused or deleted.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait({task}, timeout=TEARDOWN_WAIT_S)
        if watcher is not None:
            with contextlib.suppress(Exception):
                await watcher.wait_stopped()
            if close_buffer:
                with contextlib.suppress(Exception):
                    watcher.stop(close_buffer=True)
        self._capturing.discard(stream_id)

    async def remove_stream(self, stream_id: str) -> None:
        """Stop the watcher, delete every persisted trace and every file."""
        if not self._known(stream_id):
            raise KeyError(stream_id)
        clip_ids: list[str] = []
        with contextlib.suppress(Exception):
            rows = await asyncio.to_thread(store_call, self._store.list_clips, stream_id, 1_000_000)
            clip_ids = [r.id for r in rows or []]
        # Terminal state first: a capture-end callback racing the removal
        # must not leave the row saying "watching" after the stream is gone.
        await asyncio.to_thread(
            store_call,
            self._store.update_stream_state,
            stream_id,
            state=StreamState.ENDED.value,
        )
        await self._stop_watcher(stream_id, close_buffer=True)
        self._stream_rows.pop(stream_id, None)
        await asyncio.to_thread(store_call, self._store.delete_stream, stream_id)
        if self._registry is not None:
            with contextlib.suppress(Exception):
                self._registry.delete(stream_id)
        await asyncio.to_thread(self._purge_stream_files, stream_id, clip_ids)
        self._publish(stream_channel(stream_id), "stream_removed", {"stream_id": stream_id})
        with contextlib.suppress(Exception):
            self._publish(
                STREAMS_CHANNEL,
                "stream_state",
                {
                    "type": "stream_state",
                    "stream_id": stream_id,
                    "state": "ended",
                    "enabled": False,
                    # Other open tabs drop the row instead of showing a
                    # paused stream whose Resume answers 404.
                    "removed": True,
                },
            )

    @staticmethod
    def _purge_stream_files(stream_id: str, clip_ids: list[str]) -> None:
        """A removed stream's buffer folder and every clip folder of it."""
        from .paths import clip_dir, stream_buffer_dir

        _rmtree(stream_buffer_dir(stream_id))
        for clip_id in clip_ids:
            _rmtree(clip_dir(clip_id))

    async def pause_stream(self, stream_id: str) -> None:
        """Stop capture + signals; keep config. Persists enabled=0, state=off."""
        if not self._known(stream_id):
            raise KeyError(stream_id)
        row = self._stream_rows.get(stream_id) or self._lookup_row(stream_id)
        # Persist BEFORE stopping: the store row, the registry and the row
        # cache all say paused/off, so nothing can resurrect "watching".
        await asyncio.to_thread(
            store_call,
            self._store_set_enabled,
            stream_id,
            False,
            StreamState.OFF.value,
        )
        if row is not None:
            row.enabled = False
            row.state = StreamState.OFF.value
            row.state_detail = "paused"
            self._stream_rows[stream_id] = row
        if self._registry is not None:
            with contextlib.suppress(Exception):
                self._registry.set_enabled(stream_id, False)
        await self._stop_watcher(stream_id)
        if row is not None:  # the teardown's final OFF write may say "stopped"
            row.state = StreamState.OFF.value
        self._publish(stream_channel(stream_id), "stream_paused", {"stream_id": stream_id})
        self._publish_stream_state(stream_id)

    def _store_set_enabled(self, stream_id: str, enabled: bool, state: str) -> None:
        setter = getattr(self._store, "set_stream_enabled", None)
        if setter is not None:
            setter(stream_id, enabled, state=state)
            return
        # Minimal store doubles: state through the frozen patch call, the
        # flag through a full-row rewrite.
        self._store.update_stream_state(stream_id, state=state)
        row = self._store.get_stream(stream_id)
        if row is not None:
            row.enabled = enabled
            self._store.upsert_stream(row)

    async def resume_stream(self, stream_id: str) -> None:
        """Fresh watcher start for a paused stream (resume = new lifecycle)."""
        row = self._stream_rows.get(stream_id) or await asyncio.to_thread(
            store_call, self._store.get_stream, stream_id
        )
        if row is None:
            row = self._registry_row(stream_id)
        if row is None:
            raise KeyError(stream_id)
        row.enabled = True
        row.state = StreamState.WAITING_FOR_LIVE.value
        row.state_detail = ""
        self._stream_rows[stream_id] = row
        await asyncio.to_thread(store_call, self._store.upsert_stream, row)
        if self._registry is not None:
            with contextlib.suppress(Exception):
                self._registry.set_enabled(stream_id, True)
        if self._started and stream_id not in self._tasks:
            self._launch(stream_id, row)
        self._publish(stream_channel(stream_id), "stream_resumed", {"stream_id": stream_id})
        self._publish_stream_state(stream_id)

    # -- snapshots for the API ---------------------------------------------------

    @property
    def started(self) -> bool:
        """Whether start() has actually launched watchers and workers.

        Distinct from the ``live.enabled`` flag: it is False both before
        start() and when start() was a no-op because the feature is off.
        """
        return self._started

    def _clip_counts(self) -> dict[str, int]:
        counter = getattr(self._store, "clip_counts", None)
        if counter is None:
            return {}
        try:
            return dict(counter() or {})
        except Exception:
            return {}

    def list_stream_snapshots(self) -> list[dict]:
        """Every registered stream's API view — also on a not-started manager."""
        if not self._started:
            for cfg in self._registry_list():
                if cfg.id not in self._stream_rows:
                    self._stream_rows[cfg.id] = self._merged_row(cfg)
            with contextlib.suppress(Exception):
                for row in store_call(self._store.list_streams) or []:
                    self._stream_rows.setdefault(row.id, row)
        counts = self._clip_counts()
        out = []
        for stream_id in list(self._stream_rows):
            with contextlib.suppress(KeyError):
                out.append(self.stream_snapshot(stream_id, clip_counts=counts))
        return out

    def status(self) -> dict:
        """Whole-feature snapshot: enabled flag, per-stream views, queue caps."""
        streams = self.list_stream_snapshots() if self._started else []
        queue = self._queue
        return {
            "enabled": self._live.enabled,
            "running": self._started,
            "watching": len(self._capturing),
            "streams": streams,
            "caps": {
                "clip_queue_max": self._live.workers.clip_queue_max,
                "clip_queue_depth": 0 if queue is None else queue.qsize(),
                "clip_queue_workers": self._live.workers.clip_queue_workers,
                "max_concurrent_streams": self._live.workers.max_concurrent_streams,
                "pending_clip_bytes": self._pending_clip_bytes() if self._started else 0,
                "pending_cap_bytes": PENDING_CAP_BYTES,
            },
        }

    def stream_snapshot(self, stream_id: str, *, clip_counts: dict | None = None) -> dict:
        """API-facing view of one stream: identity + live state + last error.

        ``state`` is ``off`` for a paused/disabled stream and for every
        stream while the feature itself is not running.
        """
        row = self._stream_rows.get(stream_id)
        if row is None:
            row = self._lookup_row(stream_id)
        if row is None:
            raise KeyError(stream_id)
        watcher = self._watchers.get(stream_id)
        if not row.enabled:
            state: Any = StreamState.OFF
            detail = row.state_detail or "paused"
        elif not self._started:
            state = StreamState.OFF
            detail = "Live Watch is off" if not self._live.enabled else "Live Watch is not running"
        else:
            state = (watcher.state if watcher is not None else None) or row.state
            detail = watcher.state_detail if watcher is not None else row.state_detail
        counts = clip_counts if clip_counts is not None else self._clip_counts()
        snapshot: dict = {
            "id": stream_id,
            "url": row.url,
            "label": row.label,
            "enabled": bool(row.enabled),
            "state": state if isinstance(state, str) else getattr(state, "value", str(state)),
            "detail": detail or "",
            "capturing": stream_id in self._capturing,
            "clip_count": int(counts.get(stream_id, 0)),
            "created_at": getattr(row, "created_at", "") or "",
            "updated_at": getattr(row, "updated_at", "") or "",
        }
        if getattr(row, "last_error", None):
            snapshot["error"] = row.last_error
        return snapshot

    # -- rows: load, lookup, persist ----------------------------------------------

    def _lookup_row(self, stream_id: str) -> LiveStreamRow | None:
        """Row from the live store, then the A3 registry (stub-tolerant)."""
        row = store_call(self._store.get_stream, stream_id)
        if row is not None:
            self._stream_rows[stream_id] = row
            return row
        return self._registry_row(stream_id)

    def _registry_row(self, stream_id: str) -> LiveStreamRow | None:
        for cfg in self._registry_list():
            if cfg.id == stream_id:
                return self._merged_row(cfg)
        return None

    def _merged_row(self, cfg: LiveStreamConfig) -> LiveStreamRow:
        """The registry entry (authoritative enable flag) over the DB row."""
        db_row = None
        with contextlib.suppress(Exception):
            db_row = store_call(self._store.get_stream, cfg.id)
        row = _stream_row_from_config(cfg)
        if db_row is not None:
            row.state = db_row.state
            row.state_detail = db_row.state_detail
            row.last_error = db_row.last_error
            row.created_at = db_row.created_at
            row.updated_at = db_row.updated_at
        if not row.enabled:
            row.state = StreamState.OFF.value
        return row

    def _registry_list(self) -> list[LiveStreamConfig]:
        if self._registry is None:
            self._registry = _default_registry()
        if self._registry is None:
            return []
        try:
            return list(self._registry.list_streams())
        except NotImplementedError:
            log.info("Stream registry is a stub (A3 pending): no persisted streams loaded.")
        except Exception:
            log.warning("Stream registry read failed; starting from the DB only.", exc_info=True)
        return []

    def _registry_upsert(self, row: LiveStreamRow) -> None:
        if self._registry is None:
            self._registry = _default_registry()
        if self._registry is None:
            return
        try:
            self._registry.upsert(
                LiveStreamConfig(id=row.id, url=row.url, label=row.label, enabled=row.enabled)
            )
        except NotImplementedError:
            log.debug("Registry upsert skipped (A3 pending).")
        except Exception:
            log.warning("Registry upsert failed for %s.", row.id, exc_info=True)

    def _load_and_launch(self) -> None:
        """Pull persisted streams (registry first, DB fallback) and launch."""
        loaded: dict[str, LiveStreamRow] = {}
        for cfg in self._registry_list():
            loaded[cfg.id] = self._merged_row(cfg)
        if not loaded:
            for row in store_call(self._store.list_streams) or []:
                if not row.enabled:
                    row.state = StreamState.OFF.value
                loaded[row.id] = row
        for row in loaded.values():
            # Clip rows reference live_streams: a registry-only entry (hand
            # edit) gets its DB row before anything can clip it.
            with contextlib.suppress(Exception):
                if store_call(self._store.get_stream, row.id) is None:
                    store_call(self._store.upsert_stream, row)
            self._stream_rows[row.id] = row
            if row.enabled:
                self._launch(row.id, row)

    def _row_event(self, row: LiveStreamRow) -> dict:
        return {
            "stream_id": row.id,
            "url": row.url,
            "label": row.label,
            "enabled": row.enabled,
        }

    # -- launch / capacity --------------------------------------------------------

    def _launch(self, stream_id: str, row: LiveStreamRow) -> None:
        """Create + start one watcher task (never two for one id via manager).

        Construction is guarded: a raising capture/buffer factory must land
        as one stream's error, not as a TypeError inside add_stream().
        """
        existing = self._tasks.get(stream_id)
        if existing is not None and not existing.done():
            return
        try:
            watcher = StreamWatcher(
                stream_id,
                row.url,
                row.label,
                self._live,
                deps=self._make_deps(stream_id),
                on_state=self._on_watcher_state,
            )
        except NotImplementedError:
            log.info("Watcher deps for %s still stubbed; skipping launch.", stream_id)
            return
        except Exception as exc:
            log.exception("Watcher construction failed for %s.", stream_id)
            row.last_error = f"watcher construction failed: {exc}"
            self._stream_rows[stream_id] = row
            return
        self._watchers[stream_id] = watcher
        task = asyncio.create_task(watcher.run(), name=f"live-watcher-{stream_id}")
        self._tasks[stream_id] = task
        task.add_done_callback(lambda t, sid=stream_id: self._watcher_done(sid, t))

    def _on_watcher_state(
        self, stream_id: str, old: StreamState | None, new: StreamState, detail: str
    ) -> None:
        """Mirror watcher state into the row cache + capacity set (callback)."""
        row = self._stream_rows.get(stream_id)
        if row is not None:
            row.state = new.value
            row.state_detail = detail
            row.updated_at = utcnow()
        if new is StreamState.OFF or is_terminal_state(new):
            self._capturing.discard(stream_id)
        self._publish_stream_state(stream_id)

    def _watcher_done(self, stream_id: str, task: asyncio.Task) -> None:
        """Reap a finished watcher: log crashes, drop registries, free cap."""
        if self._tasks.get(stream_id) is task:
            self._tasks.pop(stream_id, None)
            self._watchers.pop(stream_id, None)
            self._capturing.discard(stream_id)
        if not task.cancelled() and task.done() and task.exception() is not None:
            log.error("Watcher task for %s died: %r", stream_id, task.exception())

    def _reserve_capture(self, stream_id: str) -> bool:
        """Capacity gate; atomic within one loop tick (callers never await).

        The watcher's CONNECTING mark and this check form one synchronous
        block on the loop, so concurrent watchers can't all pass and
        overshoot settings.workers.max_concurrent_streams.
        """
        if stream_id in self._capturing:
            return True
        if len(self._capturing) >= max(1, self._live.workers.max_concurrent_streams):
            return False
        self._capturing.add(stream_id)
        return True

    def _release_capture(self, stream_id: str) -> None:
        """A watcher without a running capture gives its slot back."""
        self._capturing.discard(stream_id)

    # -- disk budget ------------------------------------------------------------------

    def _admit_candidate(self) -> tuple[bool, str]:
        """Refuse new candidates on low free disk or an exceeded budget."""
        from .paths import live_dir

        floor_mb = int(getattr(self._live.workers, "min_free_disk_mb", 0) or 0)
        if floor_mb > 0:
            target = live_dir()
            while not target.exists() and target.parent != target:
                target = target.parent
            try:
                free = shutil.disk_usage(target).free
            except OSError:
                free = None
            if free is not None and free < floor_mb * 1024 * 1024:
                return (
                    False,
                    f"low disk space ({free // (1024 * 1024)} MB free, need {floor_mb} MB)",
                )
        if self._disk_over_budget:
            return False, self._disk_reason or "Live Watch disk budget exceeded"
        return True, ""

    async def _janitor_loop(self) -> None:
        while True:
            try:
                # Snapshot loop-owned state here; the pass itself runs in a thread.
                known = set(self._stream_rows)
                await asyncio.to_thread(self.enforce_disk_budget, known)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Live Watch disk janitor pass failed.")
            await asyncio.sleep(JANITOR_INTERVAL_S)

    def enforce_disk_budget(self, known_streams: set[str] | None = None) -> dict:
        """One janitor pass: drop orphans, then oldest non-ready artifacts.

        Counted against ``workers.disk_budget_mb``: stream buffers plus every
        clip folder whose clip is not ``ready`` (in flight, failed, rejected)
        plus orphans. Orphans (folders no stream/clip row owns any more) are
        deleted outright once old enough; then, while over budget, the
        artifacts of the oldest failed/rejected clips go (their small JSON
        evidence stays). Ready clips and in-flight work are never touched.
        Still over budget afterwards → new candidates are refused.
        """
        from .paths import buffers_dir, clips_dir

        budget = int(getattr(self._live.workers, "disk_budget_mb", 0) or 0) * 1024 * 1024
        now = time.time()
        known_streams = set(known_streams if known_streams is not None else self._stream_rows)
        with contextlib.suppress(Exception):
            known_streams |= {r.id for r in store_call(self._store.list_streams) or []}
        known_streams |= {cfg.id for cfg in self._registry_list()}
        rows: dict[str, Any] = {}
        # Only treat folders as orphans when the clip list was really read: a
        # transient DB error must never make every clip (ready ones included)
        # look ownerless and get deleted.
        rows_ok = False
        try:
            rows = {r.id: r for r in store_call(self._store.list_clips, None, 1_000_000) or []}
            rows_ok = True
        except Exception:  # noqa: BLE001 - the janitor must never crash the manager
            log.warning("disk janitor: could not list clips; skipping orphan cleanup")
        report = {"orphans_removed": 0, "artifacts_freed_bytes": 0, "usage_bytes": 0}

        def old_enough(path: Path) -> bool:
            try:
                return now - path.stat().st_mtime >= ORPHAN_MIN_AGE_S
            except OSError:
                return False

        usage = 0
        if buffers_dir().is_dir():
            for entry in buffers_dir().iterdir():
                if not entry.is_dir():
                    continue
                if entry.name not in known_streams and old_enough(entry):
                    if _rmtree(entry):
                        report["orphans_removed"] += 1
                    continue
                usage += _dir_size(entry)
        reclaimable: list[tuple[str, Path]] = []
        if clips_dir().is_dir():
            for entry in clips_dir().iterdir():
                if not entry.is_dir():
                    continue
                row = rows.get(entry.name)
                if row is None:
                    if rows_ok and old_enough(entry) and _rmtree(entry):
                        report["orphans_removed"] += 1
                    continue
                if row.status == "ready":
                    continue
                usage += _dir_size(entry)
                if row.status in _RECLAIMABLE_STATUSES:
                    reclaimable.append((row.updated_at or row.created_at or "", entry))
        if budget > 0 and usage > budget:
            for _, entry in sorted(reclaimable):
                if usage <= budget:
                    break
                for item in entry.rglob("*"):
                    if item.is_file() and item.suffix.lower() != ".json":
                        with contextlib.suppress(OSError):
                            size = item.stat().st_size
                            item.unlink()
                            usage -= size
                            report["artifacts_freed_bytes"] += size
        report["usage_bytes"] = usage
        over = budget > 0 and usage > budget
        self._disk_over_budget = over
        self._disk_reason = (
            f"Live Watch disk budget exceeded ({usage // (1024 * 1024)} MB of "
            f"{budget // (1024 * 1024)} MB in buffers and unfinished clips)"
            if over
            else ""
        )
        if report["orphans_removed"] or report["artifacts_freed_bytes"]:
            log.info("Live Watch disk janitor: %s", report)
        return report

    # -- clip queue -----------------------------------------------------------------

    def _enqueue_clip(self, stream_id: str, item: ClipWorkItem) -> bool:
        """Bounded enqueue; overflow degrades, never blocks the watcher.

        The watcher turns a False into a rejected live_clips row + its own
        events; this side publishes the queue-full / disk-pressure alarms.
        """
        queue = self._queue
        if queue is None:
            log.warning("Clip queue not running; dropping candidate %s.", item.clip_id)
            return False
        if queue.qsize() >= max(1, int(self._live.workers.clip_queue_max)):
            self._publish(
                CLIPS_CHANNEL,
                "clip_dropped",
                {"clip_id": item.clip_id, "stream_id": stream_id, "reason": "clip queue full"},
            )
            return False
        window_bytes = item.window_bytes or _file_size(item.source_path)
        if self._pending_clip_bytes() + window_bytes > PENDING_CAP_BYTES:
            self._publish(
                CLIPS_CHANNEL,
                "disk_pressure",
                {
                    "clip_id": item.clip_id,
                    "stream_id": stream_id,
                    "reason": "pending clip bytes above cap",
                },
            )
            return False
        queue.put_nowait(item)
        return True

    def _guarded_update(self, clip_id: str, **fields: Any) -> bool:
        """Tombstone-guarded clip write; False when discarded or missing."""
        guarded = getattr(self._store, "update_clip_guarded", None)
        if guarded is not None:
            return bool(store_call(guarded, clip_id, **fields))
        row = store_call(self._store.get_clip, clip_id)
        if row is None or getattr(row, "status", "") == "rejected":
            return False
        store_call(self._store.update_clip, clip_id, **fields)
        return True

    def _clip_tombstoned(self, clip_id: str) -> bool:
        row = store_call(self._store.get_clip, clip_id)
        return row is not None and getattr(row, "status", "") == "rejected"

    async def retry_clip(self, clip_id: str) -> LiveClipRow:
        """Re-enqueue a FAILED clip whose window is still staged on disk.

        Raises KeyError for an unknown clip and :class:`RetryRefused` (the
        API's 409) when the clip is not failed, its window is gone, the
        feature is not running, or the queue is full.
        """
        from . import clip_factory

        row = await asyncio.to_thread(store_call, self._store.get_clip, clip_id)
        if row is None:
            raise KeyError(clip_id)
        if row.status != "failed":
            raise RetryRefused(f"Only failed clips can be retried (this one is {row.status}).")
        if not self._started:
            raise RetryRefused("Live Watch is not running.")
        staged = await asyncio.to_thread(clip_factory.window_staged, row)
        if not staged:
            raise RetryRefused(
                "This clip's window is no longer on disk (it was never staged, or it was"
                " cleaned up), so it cannot be re-rendered."
            )
        item = ClipWorkItem(
            clip_id=row.id,
            stream_id=row.stream_id,
            label=row.title or row.stream_id,
            source_path=clip_factory.clip_dir(row.id) / clip_factory.RAW_NAME,
            clip_start_s=row.clip_start_s,
            clip_end_s=row.clip_end_s,
            trigger_start_s=row.trigger_start_s,
            trigger_end_s=row.trigger_end_s,
            judgement=_RowJudgement(row),
        )
        # Status first, synchronously (no await before the enqueue): a
        # worker picking the item up must find it "queued", never write
        # "rendering" that this call then overwrites.
        if not self._guarded_update(clip_id, status="queued", error=None, reason=""):
            raise RetryRefused("This clip was discarded.")
        queued = store_call(self._store.get_clip, clip_id) or row  # the "queued" snapshot
        if not self._enqueue_clip(row.stream_id, item):
            self._guarded_update(clip_id, status="failed", error=row.error or "queue full")
            raise RetryRefused("The clip queue is full; try again in a moment.")
        self._publish(
            CLIPS_CHANNEL,
            "clip_queued",
            {"clip_id": clip_id, "stream_id": row.stream_id, "retry": True},
        )
        return queued

    async def _maybe_auto_upload(self, item: ClipWorkItem, clip_row: Any, title: str) -> None:
        """Upload a freshly ready clip when the user opted in per platform.

        ``clip_row`` must be the row AS RENDERED (with ``rendered_path``).
        Opt-in comes from settings.live.uploads.{youtube,instagram}.auto_upload
        and the per-platform title/caption prefixes are applied. Each failure
        class becomes a durable row — rights not attested or platform not
        linked record ``skipped`` (user action needed, not an error), while
        token/API failures record ``failed`` — plus an ``upload`` event so
        the UI wall explains itself. A discarded clip is never uploaded, and
        a platform with an upload already running or done is skipped.
        """
        rendered_path = getattr(clip_row, "rendered_path", "") or ""
        if not rendered_path:
            return
        if getattr(clip_row, "status", "ready") == "rejected":
            return
        uploads_cfg = getattr(self._live, "uploads", None)
        wanted: list[tuple[str, str]] = []
        yt = getattr(uploads_cfg, "youtube", None) if uploads_cfg else None
        ig = getattr(uploads_cfg, "instagram", None) if uploads_cfg else None
        base = title or "Untitled clip"
        if yt is not None and getattr(yt, "auto_upload", False):
            wanted.append(("youtube", f"{getattr(yt, 'title_prefix', '') or ''}{base}"))
        if ig is not None and getattr(ig, "auto_upload", False):
            wanted.append(("instagram", f"{getattr(ig, 'caption_prefix', '') or ''}{base}"))
        for platform, upload_title in wanted:
            if await asyncio.to_thread(self._clip_tombstoned, item.clip_id):
                log.info("Clip %s was discarded; auto-upload skipped.", item.clip_id)
                return
            key = (item.clip_id, platform)
            # Check + claim + row with no await in between: a manual upload
            # of the same clip/platform (which claims the same key) can
            # never run alongside this one.
            if key in UPLOADS_IN_FLIGHT or self._upload_blocked(item.clip_id, platform):
                continue
            row_id = new_stream_id()
            try:
                from .uploads import instagram as _ig_mod
                from .uploads import youtube as _yt_mod
                from .uploads.tokens import UploadMeta as _Meta
            except Exception:
                log.exception("Upload modules unavailable; skipping auto-upload.")
                return
            upload_row = LiveUploadRow(
                id=row_id, clip_id=item.clip_id, platform=platform, status="uploading"
            )
            store_call(self._store.create_upload, upload_row)
            UPLOADS_IN_FLIGHT.add(key)
            # The UI shows the upload as running at once (no manual
            # "Upload" button that would only answer 409).
            self._publish(
                CLIPS_CHANNEL,
                "upload",
                {
                    "id": row_id,
                    "clip_id": item.clip_id,
                    "platform": platform,
                    "status": "uploading",
                },
            )
            fn = _yt_mod.upload_youtube if platform == "youtube" else _ig_mod.upload_instagram
            meta = _Meta(title=upload_title.strip() or "Untitled clip")
            started = threading.Event()
            try:
                await asyncio.to_thread(
                    self._upload_in_thread, started, key, row_id, fn, rendered_path, meta
                )
            except asyncio.CancelledError:
                if not started.is_set():
                    # The thread never ran (cancelled in the executor queue):
                    # nothing was sent, so the claim and the row are ours.
                    UPLOADS_IN_FLIGHT.discard(key)
                    with contextlib.suppress(Exception):
                        store_call(
                            self._store.update_upload, row_id, status="failed", error="interrupted"
                        )
                raise

    def _upload_in_thread(
        self,
        started: threading.Event,
        key: tuple[str, str],
        row_id: str,
        fn: Any,
        rendered_path: str,
        meta: Any,
    ) -> None:
        """Run one auto-upload to its end and record the outcome (worker thread).

        The outcome is written HERE, not by the awaiting coroutine: a
        stop() cancels that coroutine but cannot cancel this thread, and an
        upload that finished with nobody recording it stayed "uploading",
        then "failed: interrupted" after a restart — inviting a retry that
        posts the clip twice.
        """
        started.set()
        clip_id, platform = key
        try:
            try:
                result = asyncio.run(fn(rendered_path, meta))
            except Exception as exc:
                name = type(exc).__name__
                skipped = name in ("RightsNotAccepted", "NotLinkedError")
                status = "skipped" if skipped else "failed"
                store_call(
                    self._store.update_upload,
                    row_id,
                    status=status,
                    error=f"{name}: {exc}"[:512],
                )
                self._publish(
                    CLIPS_CHANNEL,
                    "upload",
                    {"id": row_id, "clip_id": clip_id, "platform": platform, "status": status},
                )
                log.info("Auto-upload %s for clip %s: %s", status, clip_id, exc)
                return
            url = str(getattr(result, "url", "") or "")
            store_call(
                self._store.update_upload,
                row_id,
                status="done",
                upload_url=url,
                platform_media_id=str(getattr(result, "media_id", "") or ""),
            )
            self._publish(
                CLIPS_CHANNEL,
                "upload",
                {
                    "id": row_id,
                    "clip_id": clip_id,
                    "platform": platform,
                    "status": "done",
                    "url": url,
                },
            )
        finally:
            UPLOADS_IN_FLIGHT.discard(key)

    def _upload_blocked(self, clip_id: str, platform: str) -> bool:
        """An upload for this clip/platform is running or already succeeded."""
        lister = getattr(self._store, "list_uploads", None)
        if lister is None:
            return False
        try:
            rows = lister(clip_id) or []
        except Exception:
            return False
        return any(r.platform == platform and r.status in ("uploading", "done") for r in rows)

    def _pending_clip_bytes(self) -> int:
        """Bytes held by not-yet-rendered clips (staged windows + transcripts).

        Deliberately cheap: the staged windows dominate disk use. The store
        rows add whatever intermediates exist. The guard only needs a safe
        upper estimate that trends to zero as the queue drains.
        """
        total = 0
        if self._queue is not None:
            for item in tuple(self._queue._queue):  # noqa: SLF001 - bounded, loop-local
                total += _file_size(getattr(item, "source_path", None))
        try:
            rows = self._store.list_clips() or []
        except NotImplementedError:
            rows = []
        except Exception:
            rows = []
        for row in rows:
            if getattr(row, "status", "") in PENDING_CLIP_STATUSES:
                total += _file_size(getattr(row, "transcript_path", ""))
        return int(total)

    def _should_retire(self) -> bool:
        """A surplus worker (settings shrank) leaves after its current item."""
        current = asyncio.current_task()
        alive = [w for w in self._workers if not w.done()]
        wanted = max(1, int(self._live.workers.clip_queue_workers))
        if len(alive) > wanted and current in alive and alive[-1] is current:
            self._workers = [w for w in self._workers if w is not current]
            return True
        return False

    async def _clip_worker(self) -> None:
        """Render queued candidates; one task per clip_queue_workers."""
        queue = self._queue
        assert queue is not None
        while True:
            if self._should_retire():
                return
            item = await queue.get()
            if item is None:  # poison sentinel from stop()
                queue.task_done()
                return
            current = asyncio.current_task()
            if current is not None:
                self._busy_workers.add(current)
            try:
                await self._render_one(item)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    self._guarded_update(
                        item.clip_id, status="failed", error="cancelled during render"
                    )
                raise
            except Exception:
                log.exception("Clip worker crashed on a candidate.")
            finally:
                if current is not None:
                    self._busy_workers.discard(current)
                queue.task_done()

    def _discard_outputs(self, clip_id: str) -> None:
        """A tombstoned clip's render outputs and intermediates (kept: JSON)."""
        from .paths import clip_dir

        work = clip_dir(clip_id)
        if not work.is_dir():
            return
        for item in work.iterdir():
            if item.is_file() and item.suffix.lower() != ".json":
                with contextlib.suppress(OSError):
                    item.unlink()

    async def _render_one(self, item: ClipWorkItem) -> None:
        """Hand one accepted candidate to the producer (A8b indirection).

        The tombstone is checked before the render (a discarded clip is not
        rendered at all), at every status write, and before auto-upload.
        """
        if not await asyncio.to_thread(self._guarded_update, item.clip_id, status="rendering"):
            log.info("Clip %s was discarded (or removed); not rendering it.", item.clip_id)
            await asyncio.to_thread(self._discard_outputs, item.clip_id)
            return
        self._publish(
            CLIPS_CHANNEL,
            "clip_producing",
            {"clip_id": item.clip_id, "stream_id": item.stream_id},
        )
        watcher = self._watchers.get(item.stream_id)
        buffer = watcher.buffer if watcher is not None else None
        clip_row = store_call(self._store.get_clip, item.clip_id)
        app_settings = None
        try:
            from .. import config

            app_settings = config.load()
        except Exception:
            app_settings = None
        produce = self._produce if self._produce is not None else _default_produce
        try:
            result = await asyncio.to_thread(
                _run_producer, produce, item, buffer, clip_row, app_settings
            )
        except asyncio.CancelledError:
            self._guarded_update(item.clip_id, status="failed", error="cancelled during render")
            raise
        except Exception as exc:
            log.exception("Clip production failed for %s.", item.clip_id)
            if self._guarded_update(item.clip_id, status="failed", error=str(exc)):
                self._publish(
                    CLIPS_CHANNEL,
                    "clip_failed",
                    {"clip_id": item.clip_id, "stream_id": item.stream_id, "error": str(exc)},
                )
            return
        rendered = getattr(result, "rendered_path", None)
        error = getattr(result, "error", None)
        if error:
            if self._guarded_update(item.clip_id, status="failed", error=error):
                self._publish(
                    CLIPS_CHANNEL,
                    "clip_failed",
                    {"clip_id": item.clip_id, "stream_id": item.stream_id, "error": error},
                )
            return
        transcript_path = getattr(result, "transcript_path", None)
        fields: dict = {
            "status": "ready",
            "rendered_path": str(rendered) if rendered else "",
            "error": None,
        }
        if transcript_path:
            fields["transcript_path"] = str(transcript_path)
        title = getattr(result, "title", "") or getattr(item.judgement, "title", "")
        if title:
            fields["title"] = title
        if not await asyncio.to_thread(self._guarded_update, item.clip_id, **fields):
            # Discarded while rendering: the result must not surface.
            log.info("Clip %s was discarded during its render; dropping the result.", item.clip_id)
            await asyncio.to_thread(self._discard_outputs, item.clip_id)
            return
        if rendered:
            # Ready: the render intermediates (raw/normalized/audio) are
            # dead weight — only a failed render resumes from them.
            from . import clip_factory

            with contextlib.suppress(Exception):
                await asyncio.to_thread(clip_factory.cleanup_intermediates, item.clip_id)
        self._publish(
            CLIPS_CHANNEL,
            "clip_ready",
            {
                "clip_id": item.clip_id,
                "stream_id": item.stream_id,
                "rendered_path": str(rendered) if rendered else "",
                "title": title,
            },
        )
        # Auto-upload is opt-in per platform (settings.live.uploads) and
        # runs after the clip is durably "ready" so a failure here never
        # touches the render itself. The row is RE-READ: the pre-render one
        # has no rendered_path. All upload errors are swallowed into rows +
        # events: one bad token must not stall the render worker.
        ready_row = store_call(self._store.get_clip, item.clip_id)
        if ready_row is None or not getattr(ready_row, "rendered_path", ""):
            ready_row = LiveClipRow(
                id=item.clip_id,
                stream_id=item.stream_id,
                status="ready",
                rendered_path=str(rendered) if rendered else "",
                title=title,
            )
        self._spawn_auto_upload(item, ready_row, title)

    def _spawn_auto_upload(self, item: ClipWorkItem, clip_row: Any, title: str) -> None:
        """Auto-upload off the render worker (an Instagram publish polls for
        minutes; with one worker every queued render — and every new
        candidate past ``clip_queue_max`` — waited behind it)."""
        uploads_cfg = getattr(self._live, "uploads", None)
        if not any(
            getattr(getattr(uploads_cfg, name, None), "auto_upload", False)
            for name in ("youtube", "instagram")
        ):
            return
        if self._upload_slots is None:
            self._upload_slots = asyncio.Semaphore(AUTO_UPLOAD_CONCURRENCY)
        slots = self._upload_slots

        async def run() -> None:
            async with slots:
                await self._maybe_auto_upload(item, clip_row, title)

        task = asyncio.get_running_loop().create_task(run(), name=f"live-upload-{item.clip_id}")
        self._upload_tasks.add(task)
        task.add_done_callback(self._upload_done)

    def _upload_done(self, task: asyncio.Task) -> None:
        self._upload_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("Auto-upload task died: %r", task.exception())


class _RowJudgement:
    """LiveJudgement-shaped view of a stored row (retry work items)."""

    def __init__(self, row: Any) -> None:
        self.score = int(getattr(row, "score", 0) or 0)
        self.title = getattr(row, "title", "") or ""
        self.hook = getattr(row, "hook", "") or ""
        self.reason = getattr(row, "reason", "") or ""
        self.start_word_offset = None
        self.end_word_offset = None


def _stream_row_from_config(cfg: LiveStreamConfig) -> LiveStreamRow:
    row = LiveStreamRow(id=cfg.id, url=cfg.url, label=cfg.label, enabled=cfg.enabled)
    if not cfg.enabled:
        row.state = StreamState.OFF.value
    if cfg.created_at:
        row.created_at = cfg.created_at
    return row


def new_stream_id() -> str:
    from .models import new_id

    return new_id()


def is_terminal_state(state: Any) -> bool:
    """Terminal-state check tolerant of raw string states from the store."""
    from .states import is_terminal

    if isinstance(state, str):
        try:
            state = StreamState(state)
        except ValueError:
            return False
    return is_terminal(state)


#: Process-wide manager. One per app; tests build fresh instances.
manager = LiveManager()
