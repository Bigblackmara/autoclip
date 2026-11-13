"""Watcher task: per-stream state machine + reconnect loop (A8 owns bodies).

One ``StreamWatcher`` per watched stream: a single asyncio task supervising
that stream's capture subprocess, cheap signal workers, and the judge
pipeline. Hard rules that keep a 24/7 watch survivable:

- crash isolation: everything a watcher touches (capture, judge, whisper)
  is allowed to raise; nothing may take the manager down. A fatal error
  lands THIS stream in ``error`` state and the manager keeps supervising
  its siblings.
- reconnection backs off exponentially (2s doubling to a 120s cap, ±20%
  jitter) so a site outage can't become a hot loop. "No playable streams"
  (channel not live) parks the stream in ``waiting_for_live`` on a 30s poll
  instead of counting it as a failure.
- heavy work never happens inline on the event loop: extraction and
  transcription run in threads; rendering is handed to the manager's
  bounded clip queue via the injected ``enqueue_clip`` hook.

Sibling contracts this module relies on (bodies land with their own work
orders; every one of them is injectable through :class:`WatcherDeps`):

- capture (A1): ``StreamCapture(url, buffer, on_event=...)`` with
  ``start()`` / ``stop()``; ``on_event("started"|"ended"|"error", detail)``;
  a not-live URL raises with "No playable streams" in the message.
- detector (A4): ``SignalAggregator.add(sig)`` / ``.evaluate(window_s)`` and
  ``Trigger.decide(window)`` → decision with ``.fire`` / ``.strength`` /
  ``.reasons``. Absence degrades the watcher to capture-only (logged).
- circuit (A4): ``TokenBucket.try_acquire()`` and
  ``PerStreamCooldown.can_trigger / mark_clip``.
- judge (A6): ``judge_window(label, words, signals)`` → ``LiveJudgement``;
  until it lands, candidates are rejected with that reason, not crashed.
- audio (A2): ``audio_signals.start(buffer, settings, on_signal,
  stop_event)`` run in a daemon thread; chat (A5): ``TwitchChatReader``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import random
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import events, store
from .buffer import PENDING_CAP_BYTES, SEGMENT_S  # noqa: F401  (PENDING_CAP_BYTES: manager)
from .buffer import RingBuffer
from .circuit import PerStreamCooldown, TokenBucket
from .judge import LiveJudgement, judge_window
from .models import LiveClipRow, new_id
from .paths import clip_dir, stream_buffer_dir
from .settings import LiveSettings
from .signals import DetectorSignal, SignalKind, SignalWindow
from .states import StreamState, can_transition, is_terminal

log = logging.getLogger(__name__)

# -- wiring constants -----------------------------------------------------------

#: Reconnect backoff: 2s, 4s, 8s ... capped, each ± this fraction of jitter.
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 120.0
BACKOFF_JITTER = 0.2
#: Connect attempts before the stream is parked in ``error``. A successful
#: connect resets the budget; only rapid-fire failures exhaust it.
DEFAULT_MAX_CONNECT_ATTEMPTS = 10
#: Watcher heartbeat: how often signals are drained and the buffer pruned.
DEFAULT_TICK_S = 1.0
#: Poll interval while the channel is not live ("No playable streams").
DEFAULT_OFFLINE_POLL_S = 30.0
#: How often the rolling buffer is pruned while watching.
PRUNE_INTERVAL_S = 10.0
#: Signal window handed to the detector/aggregator, in stream seconds.
SIGNAL_WINDOW_S = 30.0

#: SSE channel naming (events.py contract): per-stream and all-clips feeds.
CLIPS_CHANNEL = "live:-:clips"


def stream_channel(stream_id: str) -> str:
    return f"live:{stream_id}"


# -- stub-safe helpers -----------------------------------------------------------


_WARNED_STUBS: set[str] = set()


def store_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sibling-contract function, tolerating its pending stub.

    The parallel workers ship bodies behind the frozen signatures; until
    then several of them raise ``NotImplementedError`` (live.store, live
    events, A1's buffer). A watch must survive that — the manager mirrors
    state in memory — so the call is skipped with ONE debug log per function
    instead of killing the stream. The moment the sibling lands, the calls
    simply start working; no watcher restart needed.
    """
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        if fn.__name__ not in _WARNED_STUBS:
            _WARNED_STUBS.add(fn.__name__)
            log.info("%s is a stub (sibling work order pending); call skipped.", fn.__qualname__)
        return None


def publish_event(channel: str, event_type: str, data: dict | None = None) -> None:
    """Stub-safe ``events.publish`` (A10 pending).

    Event publishing is fire-and-forget SSE fan-out: a failure here must
    never propagate into watcher control flow, and A10's skeleton raises
    ``NotImplementedError`` until it lands.
    """
    try:
        events.publish(channel, event_type, data)
    except NotImplementedError:
        if "publish" not in _WARNED_STUBS:
            _WARNED_STUBS.add("publish")
            log.info("live.events.publish is a stub (A10 pending); SSE events are dropped for now.")
    except Exception:
        log.warning("live event publish failed for %s/%s.", channel, event_type, exc_info=True)


_OFFLINE_MARKERS = ("no playable streams", "no streams found", "stream is offline", "not live")


def _looks_offline(exc: BaseException) -> bool:
    """Whether a capture error means 'channel not live' rather than 'broken'.

    A1 encodes this in the exception hint text (streamlink's exact words are
    "No playable streams found"); a capture may also set ``exc.offline``.
    Offline is a poll-and-wait case; anything else is a connect failure that
    gets backoff.
    """
    if getattr(exc, "offline", False):
        return True
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _OFFLINE_MARKERS)


def _default_transcribe(path: Path, *_: Any) -> Any:
    """Default transcriber: A8 reuses the pipeline stage unchanged."""
    from ..pipeline.transcribe import transcribe

    return transcribe(path)


# -- clip queue work items --------------------------------------------------------


@dataclass
class ClipWorkItem:
    """One accepted candidate handed to the manager's bounded clip queue.

    ``source_path`` is the already-extracted window (A1's buffer range,
    concatenated once by the watcher under the extraction pin), so the
    producer (A8b) never re-touches the volatile rolling buffer. Trimming
    advice rides inside ``judgement`` word offsets.
    """

    clip_id: str
    stream_id: str
    label: str
    source_path: Path
    clip_start_s: float
    clip_end_s: float
    trigger_start_s: float
    trigger_end_s: float
    judgement: LiveJudgement
    words: list = field(default_factory=list)
    caption_style: str = "bold_pop"
    ratio: str = "9:16"


@dataclass
class ClipWorkResult:
    """What a producer must return for one work item."""

    rendered_path: Path | None = None
    error: str | None = None


async def placeholder_produce(item: ClipWorkItem) -> ClipWorkResult:
    """Default producer (A8b placeholder): the real clip factory lands later."""
    raise NotImplementedError(
        "Live clip rendering is not implemented yet (A8b). Inject a producer "
        "via LiveManager(produce=...) to render live clips."
    )


# -- dependency bundle ------------------------------------------------------------


@dataclass
class WatcherDeps:
    """Everything a watcher needs from the outside world, injectable.

    Every ``None`` entry is auto-defaulted at watcher construction so the
    module compiles and runs while sibling work orders are in flight (see
    the module docstring for each contract's expected shape). Tests inject
    minimal fakes with the same method names.
    """

    capture_factory: Callable[[Any, Callable[[str, str], None]], Any] | None = None
    buffer_factory: Callable[[], Any] | None = None
    aggregator_factory: Callable[[], Any] | None = None
    trigger: Any = None
    cooldown: Any = None
    judge_minute: Any = None
    judge_hour: Any = None
    judge_day: Any = None
    transcribe: Callable[..., Any] | None = None
    judge: Callable[..., Awaitable[LiveJudgement]] | None = None
    audio_start: Callable[..., None] | None = None
    chat_factory: Callable[..., Any] | None = None
    enqueue_clip: Callable[[ClipWorkItem], bool] | None = None
    may_capture: Callable[[], bool] | None = None
    publish: Callable[[str, str, dict | None], None] | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None
    max_connect_attempts: int | None = None


# -- the watcher --------------------------------------------------------------------


class StreamWatcher:
    """Per-stream supervisor task: state machine + connect/detect/judge loop."""

    def __init__(
        self,
        stream_id: str,
        url: str,
        label: str,
        settings: LiveSettings,
        *,
        deps: WatcherDeps | None = None,
        on_state: Callable[[str, StreamState | None, StreamState, str], None] | None = None,
        tick_s: float = DEFAULT_TICK_S,
        offline_poll_s: float = DEFAULT_OFFLINE_POLL_S,
    ) -> None:
        self.id = stream_id
        self.url = url
        self.label = label
        self._live = settings
        self._deps = deps if deps is not None else WatcherDeps()
        self._on_state = on_state
        self._tick_s = tick_s
        self._offline_poll_s = offline_poll_s
        self._resolve_defaults()

        # Loop-bound state — created in run(), never here: asyncio
        # primitives bind to the loop running at first use, and this object
        # is built outside it (the manager, tests, the API layer).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._buffer: Any = None
        self._aggregator: Any = None
        self._signal_q: queue.SimpleQueue[DetectorSignal] = queue.SimpleQueue()
        self._bg_tasks: set[asyncio.Task] = set()
        self._audio_thread: threading.Thread | None = None
        self._signal_stop: threading.Event | None = None
        self._chat: Any = None
        self._aggregator_broken = False

        self._capture: Any = None
        self._state: StreamState | None = None
        self._state_detail = ""
        self._stopping = False
        self._connect_attempts = 0
        # Set when the capture layer reports the STREAM ended (ffmpeg exit):
        # the next supervisor pass sleeps out the offline poll in
        # waiting_for_live before looking for a new broadcast.
        self._park_pending = False
        self._park_reason = ""
        self._backoff_wait_s = 0.0
        self._next_prune = 0.0
        self._prune_broken = False
        self._candidate_busy = False
        self._audio_started = False

    # -- dependency defaults ---------------------------------------------------

    def _resolve_defaults(self) -> None:
        deps = self._deps
        rate = self._live.rate_limit
        if deps.buffer_factory is None:
            deps.buffer_factory = self._default_buffer_factory()
        if deps.capture_factory is None:
            deps.capture_factory = self._default_capture_factory()
        if deps.aggregator_factory is None:
            deps.aggregator_factory = self._default_aggregator_factory
        if deps.trigger is None:
            deps.trigger = self._default_trigger()
        if deps.cooldown is None:
            deps.cooldown = PerStreamCooldown(rate.cooldown_s)
        if deps.judge_minute is None:
            # Global budgets shared across streams (circuit.py contract):
            # N per period means capacity N, refilled N/period per second.
            deps.judge_minute = TokenBucket(rate.judge_per_minute / 60.0, rate.judge_per_minute)
        if deps.judge_hour is None:
            deps.judge_hour = TokenBucket(rate.judge_per_hour / 3600.0, rate.judge_per_hour)
        if deps.judge_day is None:
            deps.judge_day = TokenBucket(rate.judge_per_day / 86400.0, rate.judge_per_day)
        if deps.transcribe is None:
            deps.transcribe = _default_transcribe
        if deps.judge is None:
            deps.judge = judge_window
        if deps.enqueue_clip is None:
            deps.enqueue_clip = self._no_clip_sink
            log.warning("Stream %s has no clip queue; accepted candidates will be dropped.", self.id)
        if deps.may_capture is None:
            deps.may_capture = lambda: True
        if deps.publish is None:
            deps.publish = publish_event
        if deps.sleep is None:
            deps.sleep = asyncio.sleep
        if deps.max_connect_attempts is None:
            deps.max_connect_attempts = DEFAULT_MAX_CONNECT_ATTEMPTS

    def _default_buffer_factory(self) -> Callable[[], Any]:
        def factory() -> Any:
            directory = stream_buffer_dir(self.id)
            directory.mkdir(parents=True, exist_ok=True)
            return RingBuffer(
                directory,
                retain_minutes=self._live.rolling_buffer.retain_minutes,
                max_disk_mb=self._live.rolling_buffer.max_disk_mb,
            )

        return factory

    def _default_capture_factory(self) -> Callable[[Any, Callable[[str, str], None]], Any]:
        def factory(buffer: Any, on_event: Callable[[str, str], None]) -> Any:
            from .capture import StreamCapture  # A1 owns this class

            return StreamCapture(self.url, buffer, on_event=on_event)

        return factory

    def _default_aggregator_factory(self) -> Any:
        try:
            from .detector import SignalAggregator  # A4 owns this class
        except ImportError:
            log.warning("SignalAggregator unavailable (A4 pending): %s runs capture-only.", self.id)
            return None
        return SignalAggregator()

    def _default_trigger(self) -> Any:
        try:
            from .detector import Trigger  # A4 owns this class
        except ImportError:
            log.info("Trigger unavailable (A4 pending): %s will not fire candidates.", self.id)
            return None
        # A4's constructor is not frozen; try the two plausible shapes once.
        for args in ((), (self._live.detection,)):
            try:
                return Trigger(*args)
            except TypeError:
                continue
        return None

    def _no_clip_sink(self, item: ClipWorkItem) -> bool:
        log.warning("Dropping accepted clip %s: no clip queue is wired.", item.clip_id)
        return False

    # -- lifecycle ---------------------------------------------------------------

    async def run(self) -> None:
        """Supervise this stream until stopped, paused, or terminally errored."""
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._buffer = self._deps.buffer_factory()  # type: ignore[misc]
        self._aggregator = self._deps.aggregator_factory()  # type: ignore[misc]
        self._next_prune = self._loop.time() + PRUNE_INTERVAL_S
        # A fresh watcher start is not an in-place transition — resume is
        # modelled as a brand-new lifecycle, so the initial state bypasses
        # the guard but is still persisted + published.
        self.set_state(StreamState.WAITING_FOR_LIVE, "watch started", initial=True)
        self._start_signal_workers()
        try:
            await self._supervise()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Crash isolation: the manager survives; only this stream dies.
            log.exception("Watcher for stream %s crashed.", self.id)
            self.set_state(StreamState.ERROR, f"watcher crashed: {exc}", error=str(exc))
        finally:
            with contextlib.suppress(Exception):
                await self._teardown()

    async def _teardown(self) -> None:
        """``finally``-side teardown: stop children, await their exits."""
        self._stopping = True
        if self._wake is not None:
            self._wake.set()
        self._shutdown()
        pending = [task for task in self._bg_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            # A cancelled candidate task unwinds through its own ``finally``
            # (marking the live_clips row failed); await that, bounded.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait(pending, timeout=2.0)
        self._bg_tasks.clear()

    def stop(self, *, close_buffer: bool = False) -> None:
        """Flag this watcher to wind down (idempotent).

        The supervisor task must be cancelled for the flag to be observed
        promptly — the manager does that; a parked ``offline_poll_s`` sleep
        otherwise finishes first. The manager pattern is therefore:
        ``task.cancel()`` then ``stop()`` (the task's own ``finally`` runs
        the full teardown, including awaiting cancelled candidate tasks).

        ``close_buffer`` additionally drops the segment directory claim
        (stream removal); pause keeps the buffer for the resume.
        """
        self._stopping = True
        if self._wake is not None:
            self._wake.set()
        self._shutdown()
        if close_buffer and self._buffer is not None:
            store_call(self._buffer.close)

    def _shutdown(self) -> None:
        self._stop_capture()
        if self._signal_stop is not None:
            self._signal_stop.set()
        thread = self._audio_thread
        if thread is not None and thread.is_alive():
            with contextlib.suppress(RuntimeError):
                thread.join(timeout=1.0)
        if self._chat is not None:
            with contextlib.suppress(Exception):
                self._chat.stop()
            self._chat = None
        if self._buffer is not None:
            with contextlib.suppress(Exception):
                self._buffer.unpin(None)

    def set_state(
        self,
        new: StreamState,
        detail: str = "",
        *,
        error: str | None = None,
        initial: bool = False,
    ) -> bool:
        """Move to ``new`` if the frozen state machine allows it.

        Legal transitions are persisted to the store and published as SSE
        ``state`` events; illegal jumps are logged and ignored so a late
        capture callback can never corrupt what the UI shows.
        """
        old = self._state
        if not initial and not can_transition(old, new):
            log.warning("Stream %s: ignoring illegal state jump %s → %s.", self.id, old, new)
            return False
        if old == new and detail == self._state_detail and error is None:
            return True  # duplicate — no store churn, no duplicate SSE event
        self._state = new
        self._state_detail = detail
        data: dict = {"stream_id": self.id, "state": new.value, "detail": detail}
        if old is not None:
            data["old"] = old.value
        if error is not None:
            data["error"] = error
        store_call(
            store.update_stream_state, self.id, state=new.value, state_detail=detail, error=error
        )
        self._deps.publish(stream_channel(self.id), "state", data)
        if self._on_state is not None:
            with contextlib.suppress(Exception):
                self._on_state(self.id, old, new, detail)
        return True

    # -- supervisor loop -----------------------------------------------------------

    @property
    def state(self) -> StreamState | None:
        return self._state

    @property
    def state_detail(self) -> str:
        return self._state_detail

    async def _supervise(self) -> None:
        while not self._stopping:
            if self._state is not None and is_terminal(self._state):
                break
            if self._capture is None:
                await self._connect()
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._tick_s)
            except TimeoutError:
                pass
            self._wake.clear()
            self._drain_signals()
            await self._maybe_prune()

    async def _connect(self) -> None:
        # The stream ended on its own (ffmpeg exited cleanly, e.g. the
        # broadcaster stopped): park on the offline cadence, not backoff —
        # a next broadcast is expected, it's just not up yet.
        if self._park_pending:
            self._park_pending = False
            reason, self._park_reason = self._park_reason, ""
            await self._park_waiting(f"stream ended{f': {reason}' if reason else ''}")
            return
        # Capacity check and the CONNECTING mark are one synchronous block:
        # no await between them, so concurrent watchers can't all pass the
        # check and overshoot settings.workers.max_concurrent_streams.
        if not self._deps.may_capture():
            await self._park_waiting("waiting for capture capacity")
            return
        if self._backoff_wait_s > 0:
            # A failed connect sleeps out its backoff BEFORE marking
            # connecting: the UI must not show "connecting" for the whole
            # 120s cap when nothing is being attempted yet.
            delay, self._backoff_wait_s = self._backoff_wait_s, 0.0
            self.set_state(StreamState.RECONNECTING, "backoff before retry")
            await self._deps.sleep(delay)
            if self._stopping:
                return
        self.set_state(StreamState.CONNECTING, "connecting")
        try:
            capture = await asyncio.to_thread(self._spawn_capture)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._connect_failed(exc)
            return
        self._capture = capture
        self._connect_attempts = 0
        store_call(self._buffer.begin_epoch)  # reconnect = new epoch (A1)
        self.set_state(StreamState.WATCHING, "live")

    def _spawn_capture(self) -> Any:
        capture = self._deps.capture_factory(self._buffer, self._on_capture_event_thread)
        capture.start()
        return capture

    async def _park_waiting(self, detail: str) -> None:
        self.set_state(StreamState.WAITING_FOR_LIVE, detail)
        await self._deps.sleep(self._offline_poll_s)

    async def _connect_failed(self, exc: Exception) -> None:
        if isinstance(exc, (ImportError, AttributeError)):
            # A missing implementation — retrying can't fix an import error.
            self.set_state(StreamState.ERROR, f"capture unavailable: {exc}", error=str(exc))
            return
        if _looks_offline(exc):
            # Not-live is not a failure: poll on the offline cadence.
            self._connect_attempts = 0
            await self._park_waiting(f"waiting for live: {exc}")
            return
        self._connect_attempts += 1
        if self._connect_attempts >= (
            self._deps.max_connect_attempts or DEFAULT_MAX_CONNECT_ATTEMPTS
        ):
            self.set_state(
                StreamState.ERROR,
                f"connect failed {self._connect_attempts} times: {exc}",
                error=str(exc),
            )
            return
        self.set_state(StreamState.RECONNECTING, f"{type(exc).__name__}: {exc}")
        self._backoff_wait_s = self._next_backoff()

    def _next_backoff(self) -> float:
        slot = min(
            BACKOFF_BASE_S * (2 ** max(0, self._connect_attempts - 1)),
            BACKOFF_CAP_S,
        )
        jitter = 1.0 + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
        return slot * jitter

    # -- capture events (any thread) ------------------------------------------------

    def _on_capture_event_thread(self, kind: str, detail: str = "") -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._handle_capture_event, kind, detail)
        except RuntimeError:
            pass  # loop shut down between the check and the marshal

    def _handle_capture_event(self, kind: str, detail: str) -> None:
        if self._stopping:
            return
        if kind == "started":
            return  # the connect path already records "live"
        if kind not in ("ended", "error"):
            log.debug("Stream %s: unknown capture event %r.", self.id, kind)
            return
        self._stop_capture()
        if kind == "ended":
            # Clean end (broadcaster stopped): the channel is expected to
            # come back, so park in waiting_for_live on the 30s poll —
            # SUCCESSIVE live sessions must not eat the reconnect budget.
            self._connect_attempts = 0
            self._backoff_wait_s = 0.0
            self._park_pending = True
            self._park_reason = detail
            # While a candidate is in flight (clipping) the frozen machine
            # refuses clipping → waiting_for_live; the candidate's own
            # unwind takes the legal hop and the park flag drives the rest.
            if self._state is not StreamState.CLIPPING:
                self.set_state(StreamState.WAITING_FOR_LIVE, detail or "stream ended")
            return
        self._connect_attempts += 1
        if self._connect_attempts >= (
            self._deps.max_connect_attempts or DEFAULT_MAX_CONNECT_ATTEMPTS
        ):
            self.set_state(
                StreamState.ERROR,
                f"capture lost {self._connect_attempts} times: {detail}",
                error=detail or kind,
            )
            return
        self.set_state(StreamState.RECONNECTING, detail or "capture error")
        self._backoff_wait_s = self._next_backoff()

    def _stop_capture(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            with contextlib.suppress(Exception):
                capture.stop()

    # -- signals ----------------------------------------------------------------------

    def submit_signal(self, signal: DetectorSignal) -> None:
        """Thread-safe entry point used by the audio/chat workers."""
        self._signal_q.put(signal)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._wake.set)

    def _start_signal_workers(self) -> None:
        if self._audio_started:
            return
        self._audio_started = True
        self._signal_stop = threading.Event()
        fn = self._deps.audio_start
        if fn is None:
            # Auto-detect A2's contract: a module-level start() function.
            from . import audio_signals

            fn = getattr(audio_signals, "start", None)
        if fn is not None:
            thread = threading.Thread(
                target=self._run_signal_worker,
                args=(fn, self._buffer, self._live, self.submit_signal, self._signal_stop),
                name=f"live-audio-{self.id}",
                daemon=True,
            )
            self._audio_thread = thread
            thread.start()
        if self._live.detection.chat.enabled and "twitch.tv" in self.url.lower():
            self._start_chat()

    def _run_signal_worker(
        self, fn: Any, buffer: Any, settings: Any, on_signal: Any, stop_event: Any
    ) -> None:
        try:
            fn(buffer, settings, on_signal, stop_event)
        except NotImplementedError:
            log.info("Audio signals unavailable (A2 pending) for %s; audio detection is off.", self.id)
        except Exception:
            # A detector thread dying must never surface in the supervisor.
            log.exception("Audio signal worker for %s died.", self.id)

    def _start_chat(self) -> None:
        if self._deps.chat_factory is not None:
            factory = self._deps.chat_factory
        else:
            from . import chat_signals as chat_module  # A5 owns this class

            reader_cls = getattr(chat_module, "TwitchChatReader", None)
            if reader_cls is None:
                log.info(
                    "Twitch chat unavailable (A5 pending): %s runs without chat signals.", self.id
                )
                return

            def factory() -> Any:
                return reader_cls(self.url, on_signal=self.submit_signal)

        try:
            self._chat = factory()
            if self._chat is not None:
                self._chat.start()
        except NotImplementedError:
            log.info("Twitch chat unavailable (A5 pending) for %s.", self.id)
        except Exception:
            # Chat is a cheap side-signal: a refusal degrades, never errors.
            log.info("Chat signals disabled for %s: reader refused to start.", self.id)

    def _drain_signals(self) -> None:
        moved = False
        while True:
            try:
                signal = self._signal_q.get_nowait()
            except queue.Empty:
                break
            moved = True
            self._add_signal(signal)
        if not moved:
            return
        trigger = self._deps.trigger
        if self._aggregator is None or trigger is None or self._aggregator_broken:
            return
        window = self._evaluate_window()
        if window is None:
            return
        try:
            decision = trigger.decide(window)
        except NotImplementedError:
            self._aggregator_broken = True
            log.info("Trigger.decide unavailable (A4 pending) for %s.", self.id)
            return
        except Exception:
            log.exception("Trigger decision failed for %s.", self.id)
            return
        if getattr(decision, "fire", False) and not self._candidate_busy:
            self._spawn(self._process_candidate(window, decision))

    def _add_signal(self, signal: DetectorSignal) -> None:
        aggregator = self._aggregator
        if aggregator is None or self._aggregator_broken:
            return
        add = getattr(aggregator, "add", None) or getattr(aggregator, "append", None)
        if add is None:
            self._aggregator_broken = True
            log.warning("Signal aggregator for %s has no add/append; detection is off.", self.id)
            return
        try:
            add(signal)
        except NotImplementedError:
            self._aggregator_broken = True
            log.info("Signal aggregator unavailable (A4 pending) for %s.", self.id)
        except Exception:
            log.exception("Aggregator rejected a signal on %s.", self.id)

    def _evaluate_window(self) -> SignalWindow | None:
        try:
            try:
                return self._aggregator.evaluate(SIGNAL_WINDOW_S)
            except TypeError:
                return self._aggregator.evaluate()
        except NotImplementedError:
            self._aggregator_broken = True
            log.info("SignalAggregator.evaluate unavailable (A4 pending) for %s.", self.id)
        except Exception:
            log.exception("Signal window evaluation failed for %s.", self.id)
        return None

    async def _maybe_prune(self) -> None:
        loop = self._loop
        if loop is None or self._prune_broken:
            return
        if loop.time() < self._next_prune:
            return
        self._next_prune = loop.time() + PRUNE_INTERVAL_S
        try:
            await asyncio.to_thread(self._buffer.prune)
        except NotImplementedError:
            self._prune_broken = True
            log.info("RingBuffer.prune is a stub (A1 pending) for %s.", self.id)
        except Exception:
            self._prune_broken = True
            log.exception("Buffer prune failed for %s; pruning disabled for this watch.", self.id)

    def _spawn(self, coro: Any) -> None:
        """Track a background task so teardown can always reach it."""
        loop = self._loop
        assert loop is not None
        task = loop.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # -- candidate → judge → clip queue -------------------------------------------

    def _cooldown_ok(self) -> bool:
        try:
            return bool(self._deps.cooldown.can_trigger(self.id))
        except NotImplementedError:
            log.info("Cooldown circuit unavailable (A4 pending); skipping candidate on %s.", self.id)
            return False
        except Exception:
            log.exception("Cooldown check failed for %s.", self.id)
            return False

    def _cooldown_mark(self) -> None:
        try:
            self._deps.cooldown.mark_clip(self.id)
        except NotImplementedError:
            log.info("Cooldown mark unavailable (A4 pending) for %s.", self.id)
        except Exception:
            log.exception("Cooldown mark failed for %s.", self.id)

    def _acquire_judge_budget(self) -> tuple[bool, str]:
        """Take one token from each global judge bucket (A4 circuit)."""
        for name, bucket in (
            ("minute", self._deps.judge_minute),
            ("hour", self._deps.judge_hour),
            ("day", self._deps.judge_day),
        ):
            if bucket is None:
                continue
            try:
                ok = bool(bucket.try_acquire())
            except NotImplementedError:
                return False, f"judge {name} budget unavailable (circuit not implemented yet)"
            if not ok:
                return False, f"judge {name} budget exhausted"
        return True, ""

    def _pin_for(self, clip_start_s: float) -> int:
        """Index covering ``clip_start_s`` — pin so prune can't race us."""
        try:
            span = float(self._buffer.span_seconds())
        except (AttributeError, NotImplementedError, TypeError):
            span = clip_start_s  # best effort until A1 lands
        behind = max(0.0, span - clip_start_s) + SEGMENT_S  # one segment of slack
        try:
            return int(self._buffer.indices_back(behind))
        except NotImplementedError:
            return 0

    def _update_clip(self, clip_id: str, **fields: Any) -> None:
        store_call(store.update_clip, clip_id, **fields)

    def _update_clip_safe(self, clip_id: str, **fields: Any) -> None:
        with contextlib.suppress(Exception):
            self._update_clip(clip_id, **fields)

    def _publish_clip(self, event_type: str, data: dict) -> None:
        payload = {"stream_id": self.id, **data}
        self._deps.publish(stream_channel(self.id), event_type, payload)
        self._deps.publish(CLIPS_CHANNEL, event_type, payload)

    def _save_transcript(self, transcript: Any, directory: Path) -> str:
        if not hasattr(transcript, "save"):
            return ""
        try:
            path = directory / "transcript.json"
            transcript.save(path)
            return str(path)
        except Exception:
            log.debug("Transcript save failed for %s.", self.id, exc_info=True)
            return ""

    async def _process_candidate(self, window: SignalWindow, decision: Any) -> None:
        """Turn one triggered window into a judged, enqueued clip candidate.

        Steps are cost-ordered: cooldown (free) → candidate row → global
        judge budgets → buffer extraction + transcription (heavy) → the LLM
        judge → clip queue. Every drop writes the live_clips row so the UI
        can show WHY a candidate vanished instead of a silent miss.
        """
        if self._candidate_busy:
            log.debug("Stream %s: candidate already in flight; ignoring trigger.", self.id)
            return
        self._candidate_busy = True
        clip_id = new_id()
        entered_clipping = False
        try:
            rate_limit = self._live.rate_limit
            rolling = self._live.rolling_buffer
            if not self._cooldown_ok():
                return
            times = [s.t_s for s in window.signals] or [window.end_s]
            trigger_start_s = min(times)
            trigger_end_s = max(times)
            clip_start_s = max(0.0, trigger_end_s - rolling.pre_roll_s)
            clip_end_s = trigger_end_s + rolling.post_roll_s
            store_call(
                store.create_clip,
                LiveClipRow(
                    id=clip_id,
                    stream_id=self.id,
                    status="detecting",
                    trigger_start_s=trigger_start_s,
                    trigger_end_s=trigger_end_s,
                    clip_start_s=clip_start_s,
                    clip_end_s=clip_end_s,
                ),
            )
            self._publish_clip("clip_detecting", {"clip_id": clip_id, "trigger_end_s": trigger_end_s})
            budget_ok, budget_reason = self._acquire_judge_budget()
            if not budget_ok:
                self._update_clip(clip_id, status="rejected", reason=f"rate limited: {budget_reason}")
                self._publish_clip("clip_rejected", {"clip_id": clip_id, "reason": budget_reason})
                return

            # The capture loop keeps running underneath: the state machine
            # explicitly allows watching → clipping → watching|reconnecting.
            entered_clipping = self.set_state(StreamState.CLIPPING, "candidate detected")
            directory = clip_dir(clip_id)
            directory.mkdir(parents=True, exist_ok=True)
            source_path = directory / "window.ts"
            pin_index = self._pin_for(clip_start_s)
            try:
                self._buffer.pin_oldest(pin_index)
                await asyncio.to_thread(self._buffer.extract, clip_start_s, clip_end_s, source_path)
            finally:
                store_call(self._buffer.unpin, pin_index)

            transcript = await asyncio.to_thread(self._deps.transcribe, source_path)
            words = list(getattr(transcript, "words", None) or [])
            transcript_path = self._save_transcript(transcript, directory)
            signals = {
                kind.value: window.strength_of(kind)
                for kind in (
                    SignalKind.AUDIO_ENERGY,
                    SignalKind.CHAT_RATE,
                    SignalKind.CHAT_HYPE,
                )
                if window.strength_of(kind) > 0
            }
            try:
                judgement = await self._deps.judge(self.label or self.url, words, signals)
            except Exception as exc:
                # A6 pending or provider failure — the stream keeps watching.
                self._update_clip(clip_id, status="rejected", reason=f"judge unavailable: {exc}")
                self._publish_clip(
                    "clip_rejected", {"clip_id": clip_id, "reason": f"judge error: {exc}"}
                )
                return

            if judgement.score < rate_limit.judge_min_score:
                self._update_clip(
                    clip_id,
                    status="rejected",
                    title=judgement.title,
                    score=int(judgement.score),
                    reason=judgement.reason
                    or f"below judge_min_score ({rate_limit.judge_min_score})",
                    hook=judgement.hook,
                    transcript_path=transcript_path,
                )
                self._publish_clip(
                    "clip_rejected", {"clip_id": clip_id, "score": int(judgement.score)}
                )
                return

            # Verdict passed: mark the cooldown so this stream cools down even
            # if the enqueue below drops the item under queue/disk pressure.
            self._cooldown_mark()
            self._update_clip(
                clip_id,
                status="queued",
                title=judgement.title,
                score=int(judgement.score),
                reason=judgement.reason,
                hook=judgement.hook,
                transcript_path=transcript_path,
            )
            item = ClipWorkItem(
                clip_id=clip_id,
                stream_id=self.id,
                label=self.label or self.url,
                source_path=source_path,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                trigger_start_s=trigger_start_s,
                trigger_end_s=trigger_end_s,
                judgement=judgement,
                words=words,
            )
            enqueued = bool(self._deps.enqueue_clip(item))
            if not enqueued:
                self._update_clip(clip_id, status="rejected", reason="clip queue full or disk pressure")
                self._publish_clip(
                    "clip_dropped", {"clip_id": clip_id, "reason": "queue full or disk pressure"}
                )
                return
            self._publish_clip(
                "clip_queued",
                {"clip_id": clip_id, "score": int(judgement.score), "title": judgement.title},
            )
        except asyncio.CancelledError:
            self._update_clip_safe(clip_id, status="failed", error="cancelled before verdict")
            raise
        except Exception as exc:
            log.exception("Candidate pipeline failed for stream %s.", self.id)
            self._update_clip_safe(clip_id, status="failed", error=str(exc))
            self._publish_clip("clip_failed", {"clip_id": clip_id, "error": str(exc)})
        finally:
            # Candidate work over. If the capture is still alive → back to
            # plain watching. If it died mid-clip, the capture callback has
            # already moved us (or parked us for the next poll); when the
            # frozen machine won't take the jump, the reconnecting hop is
            # the legal way out and the connect path re-parks from there.
            self._candidate_busy = False
            if entered_clipping and self._state is StreamState.CLIPPING:
                if self._capture is not None:
                    self.set_state(StreamState.WATCHING, "live")
                else:
                    self.set_state(StreamState.RECONNECTING, "capture lost during clip")

