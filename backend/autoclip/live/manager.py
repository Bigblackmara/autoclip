"""LiveManager — the single Live Watch entry point (A8 owns bodies).

Owns: the watcher task registry, add/remove/pause/resume, the bounded clip
queue + its render workers, capture capacity, judge rate limits (circuit.py)
and the judge concurrency mirror, event publishing, crash recovery.

Public API (what A10 / the API layer codes against; all coroutines):

    await manager.start()                        # idempotent; loads persisted
                                                # streams, launches watchers
    await manager.stop()                        # cancel all, kill captures
    sid = await manager.add_stream(url, label="")     # register + watch
    await manager.remove_stream(sid)                  # stop + delete
    await manager.pause_stream(sid)                   # stop, keep config
    await manager.resume_stream(sid)                   # fresh watcher start
    snapshot = manager.status()     # {"enabled", "running", "watching",
                                    #  "streams": [...], "caps": {...}}
    snap = manager.stream_snapshot(sid)   # one stream's API-facing view

Construction parameters (all optional, all injectable for tests):

    settings=None      LiveSettings; defaults from config.json's ``live``
                       section merged over the frozen defaults (the tree is
                       additive only — extra="ignore" tolerates old configs).
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
judge calls across streams; ollama (local) serializes to 1.

Guarantees the API layer relies on:
- start() / stop() are idempotent; stop() cancels every task and awaits
  each unwind bounded, so no watcher, capture, or render leaks past it.
- add/remove/pause/resume are safe on a not-started manager (registration
  still lands; listing works; watchers launch on the next start()).
- One watcher task per stream at most; a crashed watcher is reaped without
  ever taking the manager down.
- pause keeps the ring buffer (segments survive for the resume); remove
  closes it and deletes every persisted trace of the stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from pathlib import Path
from typing import Any

from . import events
from .buffer import PENDING_CAP_BYTES
from .models import LiveStreamRow, utcnow
from .settings import LiveSettings, LiveStreamConfig
from .states import StreamState
from .watcher import (
    CLIPS_CHANNEL,
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

#: Default clip renderer: A8b's factory, re-extracting from the buffer.
_DEFAULT_PRODUCE_LOCK = asyncio.Lock() if hasattr(asyncio, "Lock") else None


def _load_settings() -> LiveSettings:
    """LiveSettings from config.json's ``live`` section, defaults elsewhere.

    ``extra="ignore"`` plus a plain ``model_validate`` of the stored dict
    keeps this additive: an old config without the section, or with fields
    this version no longer knows, both land on the frozen defaults instead
    of crashing the app on startup.
    """
    try:
        from .. import config

        raw = json_read_live_section(getattr(config.load(), "live", None))
    except Exception:
        log.warning("Falling back to default LiveSettings.", exc_info=True)
        return LiveSettings()
    return raw


def json_read_live_section(raw: Any) -> LiveSettings:
    """Validate the raw ``live`` config value (or None) into LiveSettings."""
    if not isinstance(raw, dict):
        return LiveSettings()
    try:
        return LiveSettings.model_validate(raw)
    except Exception:
        log.warning("live settings failed validation; using defaults.", exc_info=True)
        return LiveSettings()


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
    """Standard render path: A8b's clip_factory, re-extracting the window.

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
        self._judge_semaphore: Any = judge_semaphore
        # Loop-bound state, created in start() — asyncio primitives bind to
        # the running loop (see jobs/queue.py for the same reasoning).
        self._queue: asyncio.Queue[ClipWorkItem | None] | None = None
        self._workers: list[asyncio.Task] = []

    # -- wiring ----------------------------------------------------------------

    def _resolve_judge_semaphore(self) -> Any:
        """Judge concurrency mirror (scope update): hosted 3, ollama 1."""
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
        try:
            return asyncio.Semaphore(concurrency)
        except RuntimeError:
            return None  # no running loop yet (built outside start())

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
            publish=self._publish,
            judge_semaphore=self._resolve_judge_semaphore(),
            **fresh,
        )

    def _publish(self, channel: str, event_type: str, data: dict | None = None) -> None:
        """Single publish choke-point: injected publisher or the stub-safe one."""
        publisher = self._inject.get("publish")
        if publisher is not None:
            publisher(channel, event_type, data)
        else:
            publish_event(channel, event_type, data)

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Load persisted streams and launch their watchers. Idempotent."""
        if self._started:
            return
        self._started = True
        loop = asyncio.get_running_loop()
        # Bind events.publish to this loop (A10's broker marshals threads
        # onto it); stub-safe — a pending A10 is skipped inside publish_event.
        with contextlib.suppress(Exception):
            events.bind_loop(loop)
        self._queue = asyncio.Queue(maxsize=max(1, self._live.workers.clip_queue_max))
        # Crash recovery FIRST (A9 contract): rows a previous process left
        # mid-flight are reset before this one asserts new states over them.
        await asyncio.to_thread(store_call, self._store.reset_interrupted_states)
        for _ in range(max(1, self._live.workers.clip_queue_workers)):
            worker = loop.create_task(self._clip_worker())
            self._workers.append(worker)
        self._load_and_launch()
        log.info("LiveManager started: %d stream(s) registered.", len(self._stream_rows))

    async def stop(self) -> None:
        """Cancel everything this manager owns and await the unwind. Idempotent."""
        awaitables: list[asyncio.Task] = []
        for stream_id, task in list(self._tasks.items()):
            watcher = self._watchers.get(stream_id)
            if watcher is not None:
                with contextlib.suppress(Exception):
                    watcher.stop()  # flags stopping + kills the capture
            if task is not None and not task.done():
                task.cancel()
                awaitables.append(task)
        self._tasks.clear()
        queue = self._queue
        if queue is not None:
            # Poison the queue: idle workers exit on the sentinel without
            # waiting out their get(); a FULL queue can't block a no-wait put.
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)
        for worker in self._workers:
            if not worker.done():
                worker.cancel()
                awaitables.append(worker)
        if awaitables:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait(awaitables, timeout=8.0)
        self._workers = []
        self._watchers.clear()
        self._capturing.clear()
        self._stream_rows.clear()
        self._queue = None
        self._started = False
        log.info("LiveManager stopped.")

    # -- stream registry operations --------------------------------------------

    async def add_stream(self, url: str, label: str = "", *, enabled: bool = True) -> str:
        """Register a stream, persist it, and (when started+enabled) watch it.

        Registration works on a not-started manager too: the row and the
        registry entry land, and the watcher launches on the next start().
        """
        if not url.strip():
            raise ValueError("Stream URL must not be empty.")
        row = LiveStreamRow(id=new_stream_id(), url=url, label=label or "", enabled=enabled)
        await asyncio.to_thread(store_call, self._store.upsert_stream, row)
        self._registry_upsert(row)
        self._stream_rows[row.id] = row
        self._publish(stream_channel(row.id), "stream_added", self._row_event(row))
        if enabled and self._started:
            self._launch(row.id, row)
        return row.id

    async def remove_stream(self, stream_id: str) -> None:
        """Stop the watcher, free its buffer, delete every persisted trace."""
        task = self._tasks.pop(stream_id, None)
        watcher = self._watchers.pop(stream_id, None)
        # Terminal state first: a capture-end callback racing the removal
        # must not leave the row saying "watching" after the stream is gone.
        await asyncio.to_thread(
            store_call,
            self._store.update_stream_state,
            stream_id,
            state=StreamState.ENDED.value,
        )
        if watcher is not None:
            with contextlib.suppress(Exception):
                watcher.stop()  # close_buffer=False: the task's teardown owns it
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
                await asyncio.wait_for(task, timeout=5.0)
        if watcher is not None:
            # Removal (unlike pause) drops the buffer's directory claim.
            with contextlib.suppress(Exception):
                watcher.stop(close_buffer=True)
        self._capturing.discard(stream_id)
        self._stream_rows.pop(stream_id, None)
        await asyncio.to_thread(store_call, self._store.delete_stream, stream_id)
        if self._registry is not None:
            with contextlib.suppress(Exception):
                self._registry.delete(stream_id)
        self._publish(stream_channel(stream_id), "stream_removed", {"stream_id": stream_id})

    async def pause_stream(self, stream_id: str) -> None:
        """Stop capture + signals; keep config and the DB row (state=off)."""
        task = self._tasks.pop(stream_id, None)
        watcher = self._watchers.pop(stream_id, None)
        # Mark off BEFORE cancelling: the task's final set_state call from a
        # capture callback can't overwrite the user-visible pause.
        await asyncio.to_thread(
            store_call,
            self._store.update_stream_state,
            stream_id,
            state=StreamState.OFF.value,
        )
        if watcher is not None:
            with contextlib.suppress(Exception):
                watcher.stop()  # stopping flag makes late callbacks no-ops
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
                await asyncio.wait_for(task, timeout=5.0)
        self._capturing.discard(stream_id)
        row = self._stream_rows.get(stream_id)
        if row is not None:
            row.enabled = False
        if self._registry is not None:
            with contextlib.suppress(Exception):
                self._registry.set_enabled(stream_id, False)
        self._publish(stream_channel(stream_id), "stream_paused", {"stream_id": stream_id})

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
        self._stream_rows[stream_id] = row
        await asyncio.to_thread(store_call, self._store.upsert_stream, row)
        if self._registry is not None:
            with contextlib.suppress(Exception):
                self._registry.set_enabled(stream_id, True)
        if self._started and stream_id not in self._tasks:
            self._launch(stream_id, row)
        self._publish(stream_channel(stream_id), "stream_resumed", {"stream_id": stream_id})

    # -- snapshots for the API ---------------------------------------------------

    def status(self) -> dict:
        """Whole-feature snapshot: enabled flag, per-stream views, queue caps."""
        streams = []
        for stream_id in self._stream_rows:
            with contextlib.suppress(KeyError):
                streams.append(self.stream_snapshot(stream_id))
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
                "pending_clip_bytes": self._pending_clip_bytes(),
                "pending_cap_bytes": PENDING_CAP_BYTES,
            },
        }

    def stream_snapshot(self, stream_id: str) -> dict:
        """API-facing view of one stream: identity + live state + last error."""
        row = self._stream_rows.get(stream_id)
        if row is None:
            row = self._lookup_row(stream_id)
        if row is None:
            raise KeyError(stream_id)
        watcher = self._watchers.get(stream_id)
        state = (watcher.state if watcher is not None else None) or row.state
        detail = watcher.state_detail if watcher is not None else row.state_detail
        snapshot: dict = {
            "id": stream_id,
            "url": row.url,
            "label": row.label,
            "enabled": row.enabled,
            "state": state if isinstance(state, str) else getattr(state, "value", str(state)),
            "detail": detail or "",
            "capturing": stream_id in self._capturing,
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
                return _stream_row_from_config(cfg)
        return None

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
            loaded[cfg.id] = _stream_row_from_config(cfg)
        if not loaded:
            for row in store_call(self._store.list_streams) or []:
                loaded[row.id] = row
        for row in loaded.values():
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
        if queue.full():
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

    def _pending_clip_bytes(self) -> int:
        """Bytes held by not-yet-rendered clips (queue windows + transcripts).

        Deliberately cheap: the extracted windows dominate disk use. The
        store rows add whatever intermediates exist. The guard only needs a
        safe upper estimate that trends to zero as the queue drains.
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

    async def _clip_worker(self) -> None:
        """Render queued candidates; one task per clip_queue_workers."""
        assert self._queue is not None
        while True:
            item = await self._queue.get()
            if item is None:  # poison sentinel from stop()
                self._queue.task_done()
                return
            try:
                await self._render_one(item)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    store_call(
                        self._store.update_clip,
                        item.clip_id,
                        status="failed",
                        error="cancelled during render",
                    )
                raise
            except Exception:
                log.exception("Clip worker crashed on a candidate.")
            finally:
                self._queue.task_done()

    async def _render_one(self, item: ClipWorkItem) -> None:
        """Hand one accepted candidate to the producer (A8b indirection)."""
        store_call(self._store.update_clip, item.clip_id, status="rendering")
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
            store_call(
                self._store.update_clip,
                item.clip_id,
                status="failed",
                error="cancelled during render",
            )
            raise
        except Exception as exc:
            log.exception("Clip production failed for %s.", item.clip_id)
            store_call(self._store.update_clip, item.clip_id, status="failed", error=str(exc))
            self._publish(
                CLIPS_CHANNEL, "clip_failed", {"clip_id": item.clip_id, "error": str(exc)}
            )
            return
        rendered = getattr(result, "rendered_path", None)
        error = getattr(result, "error", None)
        if error:
            store_call(self._store.update_clip, item.clip_id, status="failed", error=error)
            self._publish(CLIPS_CHANNEL, "clip_failed", {"clip_id": item.clip_id, "error": error})
            return
        transcript_path = getattr(result, "transcript_path", None)
        fields: dict = {
            "status": "ready",
            "rendered_path": str(rendered) if rendered else "",
        }
        if transcript_path:
            fields["transcript_path"] = str(transcript_path)
        title = getattr(result, "title", "") or getattr(item.judgement, "title", "")
        if title:
            fields["title"] = title
        store_call(self._store.update_clip, item.clip_id, **fields)
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


def _stream_row_from_config(cfg: LiveStreamConfig) -> LiveStreamRow:
    return LiveStreamRow(id=cfg.id, url=cfg.url, label=cfg.label, enabled=cfg.enabled)


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
