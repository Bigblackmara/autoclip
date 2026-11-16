"""GTA Mode queue: many streams at once, VRAM-safe.

Two worker pools:

- **network** (several threads): link probing, audio and section downloads,
  chat/viewer signals, the Hype Curve and the AI judge — waiting on the
  internet, so they run in parallel across streams.
- **heavy** (one thread by default): Whisper transcription and rendering —
  each wants the GPU, so they run one after another. Slower, never out of
  memory.

State lives in SQLite (each stream's ``status`` is the stage it needs next,
each clip's ``status`` likewise), so a restart simply resumes: the stages
reuse every file they already produced.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from ..pipeline.ingest import IngestError
from ..providers.base import ProviderError
from . import pipeline, store
from .pipeline import Reporter, StageError, StreamCancelled
from .settings import load as load_settings

log = logging.getLogger(__name__)

#: How often the scheduler looks for new work when nobody wakes it.
TICK_S = 2.0
#: A clip section that fails this many times is marked failed.
MAX_CLIP_ATTEMPTS = 3


class GtaQueue:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._net: ThreadPoolExecutor | None = None
        self._heavy: ThreadPoolExecutor | None = None
        self._busy: set[str] = set()
        self._busy_lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._clip_failures: dict[str, int] = {}
        self._stopping = False

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        settings = load_settings()
        self._stopping = False
        self._loop = asyncio.get_running_loop()
        self._net = ThreadPoolExecutor(settings.network_workers, thread_name_prefix="gta-net")
        self._heavy = ThreadPoolExecutor(settings.heavy_workers, thread_name_prefix="gta-heavy")
        self._wakeup = asyncio.Event()
        self._recover_clips()
        self._task = asyncio.create_task(self._scheduler(), name="gta-queue")
        self.notify()

    async def stop(self) -> None:
        self._stopping = True
        # Running stages see "cancelled" and stop at their next check; their
        # rows keep the stage status, so the next start resumes them.
        with self._busy_lock:
            self._cancelled.update(self._busy)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for pool in (self._net, self._heavy):
            if pool is not None:
                await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
        self._net = self._heavy = None
        self._wakeup = None
        with self._busy_lock:
            self._busy.clear()
            self._cancelled.clear()

    @property
    def running(self) -> bool:
        return self._task is not None

    def notify(self) -> None:
        if self._loop is None or self._wakeup is None:
            return
        with contextlib.suppress(RuntimeError):  # loop closed during shutdown
            self._loop.call_soon_threadsafe(self._wakeup.set)

    def cancel(self, stream_id: str) -> None:
        """Ask the stages of ``stream_id`` to stop (the row is updated by the API)."""
        with self._busy_lock:
            self._cancelled.add(stream_id)
        self.notify()

    def busy_keys(self) -> set[str]:
        with self._busy_lock:
            return set(self._busy)

    # -- recovery --------------------------------------------------------

    def _recover_clips(self) -> None:
        """Clips caught mid-download by a crash go back to "approved".

        A clip caught mid-render keeps "rendering": the render reuses its
        downloaded section, or sends the clip back for a fresh download if
        the section turns out to be missing.
        """
        for clip in store.list_clips(statuses=frozenset({"downloading"})):
            store.update_clip(clip.id, status="approved")

    # -- scheduling ------------------------------------------------------

    async def _scheduler(self) -> None:
        assert self._wakeup is not None
        while not self._stopping:
            try:
                await asyncio.to_thread(self._schedule_once)
            except Exception:
                log.exception("GTA scheduler pass failed.")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=TICK_S)
            self._wakeup.clear()

    def _claim(self, key: str) -> bool:
        with self._busy_lock:
            if key in self._busy:
                return False
            self._busy.add(key)
            return True

    def _release(self, key: str) -> None:
        with self._busy_lock:
            self._busy.discard(key)
        self.notify()

    def _is_cancelled(self, stream_id: str) -> bool:
        if self._stopping:
            return True
        with self._busy_lock:
            if stream_id in self._cancelled:
                return True
        stream = store.get_stream(stream_id)
        return stream is None or stream.status == "cancelled"

    def _schedule_once(self) -> None:
        if self._net is None or self._heavy is None:
            return
        for stream in store.list_streams(statuses=store.ACTIVE_STREAM_STATUSES):
            if stream.status == "analyzed":
                self._schedule_clips(stream)
                continue
            key = f"stream:{stream.id}"
            if not self._claim(key):
                continue
            with self._busy_lock:
                self._cancelled.discard(stream.id)
            pool = self._heavy if stream.status in pipeline.HEAVY_STAGES else self._net
            self._submit(pool, key, self._run_stream_stage, stream.id)

    def _schedule_clips(self, stream: store.GtaStream) -> None:
        assert self._net is not None and self._heavy is not None
        clips = store.list_clips(stream.id, statuses=store.PENDING_CLIP_STATUSES)
        if not clips:
            if not any(k.startswith(f"clip:{stream.id}:") for k in self.busy_keys()):
                self._finish_stream(stream.id)
            return
        for clip in clips:
            key = f"clip:{stream.id}:{clip.id}"
            if not self._claim(key):
                continue
            if clip.status in {"approved", "downloading"}:
                self._submit(self._net, key, self._run_section, stream.id, clip.id)
            else:
                self._submit(self._heavy, key, self._run_render, stream.id, clip.id)

    def _submit(self, pool: ThreadPoolExecutor, key: str, fn, *args) -> None:
        try:
            future: Future = pool.submit(fn, *args)
        except RuntimeError:  # pool shut down
            self._release(key)
            return
        future.add_done_callback(lambda _f, k=key: self._release(k))

    def _finish_stream(self, stream_id: str) -> None:
        clips = store.list_clips(stream_id)
        ready = sum(1 for c in clips if c.status == "ready")
        failed = sum(1 for c in clips if c.status in {"qa_failed", "failed"})
        rejected = sum(1 for c in clips if c.status == "rejected")
        detail = f"{ready} clip(s) ready, {rejected} moment(s) rejected"
        if failed:
            detail += f", {failed} failed QA/render"
        store.update_stream(stream_id, status="done", progress=1.0, detail=detail, stage="")

    # -- workers ---------------------------------------------------------

    def _run_stream_stage(self, stream_id: str) -> None:
        stream = store.get_stream(stream_id)
        if stream is None or stream.status not in store.ACTIVE_STREAM_STATUSES:
            return
        status = stream.status
        report = Reporter(stream_id, lambda: self._is_cancelled(stream_id))
        settings = load_settings()
        store.update_stream(stream_id, stage=status, error=None, progress=0.0)
        try:
            if status in {"queued", "probing"}:
                store.update_stream(stream_id, status="probing")
                pipeline.stage_probe(stream, report)
            elif status == "downloading_audio":
                pipeline.stage_audio(stream, report, settings)
            elif status == "fetching_signals":
                pipeline.stage_signals(stream, report)
            elif status == "transcribing":
                pipeline.stage_transcribe(stream, report)
            elif status == "building_hype":
                pipeline.stage_hype(stream, report)
            elif status == "judging":
                pipeline.stage_judge(stream, report, settings)
            else:
                return
        except StreamCancelled:
            self._after_cancel(stream_id, status)
            return
        except Exception as exc:
            if self._is_cancelled(stream_id):
                self._after_cancel(stream_id, status)
                return
            self._fail(stream_id, status, exc)
            return
        current = store.get_stream(stream_id)
        if current is None or current.status == "cancelled":
            return
        store.update_stream(
            stream_id, status=pipeline.NEXT[status], progress=0.0, detail="", stage=status
        )

    def _after_cancel(self, stream_id: str, status: str) -> None:
        current = store.get_stream(stream_id)
        if current is None:
            return
        if self._stopping and current.status != "cancelled":
            # App shutting down: keep the stage so the next start resumes it.
            store.update_stream(stream_id, status="queued" if status == "probing" else status)
            return
        store.update_stream(stream_id, detail="Cancelled", progress=0.0)

    def _fail(self, stream_id: str, status: str, exc: Exception) -> None:
        message = str(exc) or type(exc).__name__
        hint = getattr(exc, "hint", "")
        if isinstance(exc, IngestError | ProviderError | StageError) and hint:
            message = f"{message}\n{hint}"
        if not isinstance(exc, IngestError | ProviderError | StageError):
            log.exception("GTA stage %s failed for %s.", status, stream_id)
        store.update_stream(
            stream_id, status="failed", error=message[:2000], stage=status, detail=""
        )

    def _run_section(self, stream_id: str, clip_id: str) -> None:
        stream = store.get_stream(stream_id)
        clip = store.get_clip(clip_id)
        if stream is None or clip is None or clip.status not in {"approved", "downloading"}:
            return
        report = Reporter(stream_id, lambda: self._is_cancelled(stream_id))
        store.update_clip(clip_id, status="downloading", error=None)
        try:
            pipeline.stage_section(clip, stream, report, load_settings())
        except StreamCancelled:
            store.update_clip(clip_id, status="approved")
            return
        except Exception as exc:
            if self._is_cancelled(stream_id):
                store.update_clip(clip_id, status="approved")
                return
            self._clip_failed(clip_id, exc)
            return
        store.update_clip(clip_id, status="rendering")

    def _run_render(self, stream_id: str, clip_id: str) -> None:
        stream = store.get_stream(stream_id)
        clip = store.get_clip(clip_id)
        if stream is None or clip is None or clip.status != "rendering":
            return
        report = Reporter(stream_id, lambda: self._is_cancelled(stream_id))
        try:
            result = pipeline.stage_render(clip, stream, report, load_settings())
        except StreamCancelled:
            return
        except StageError as exc:
            # Section missing: download it again.
            store.update_clip(clip_id, status="approved", error=str(exc))
            return
        except Exception as exc:
            if self._is_cancelled(stream_id):
                return
            self._clip_failed(clip_id, exc)
            return
        passed = bool(result["qa"].get("passed"))
        store.update_clip(
            clip_id,
            status="ready" if passed else "qa_failed",
            path=result["path"],
            qa=result["qa"],
            error=None if passed else "Failed QA — see the checks.",
        )

    def _clip_failed(self, clip_id: str, exc: Exception) -> None:
        count = self._clip_failures.get(clip_id, 0) + 1
        self._clip_failures[clip_id] = count
        message = str(exc) or type(exc).__name__
        hint = getattr(exc, "hint", "")
        if hint:
            message = f"{message}\n{hint}"
        if count >= MAX_CLIP_ATTEMPTS:
            log.warning("GTA clip %s failed %d times: %s", clip_id, count, message)
            store.update_clip(clip_id, status="failed", error=message[:2000])
        else:
            log.info("GTA clip %s failed (attempt %d), retrying: %s", clip_id, count, message)
            store.update_clip(clip_id, status="approved", error=message[:2000])


queue = GtaQueue()
