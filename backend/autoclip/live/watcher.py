"""Watcher task: per-stream state machine + reconnect loop (A8 owns bodies).

One ``StreamWatcher`` per watched stream: a single asyncio task supervising
that stream's capture subprocess, cheap signal workers, and the judge
pipeline. Hard rules that keep a 24/7 watch survivable:

- crash isolation: everything a watcher touches (capture, judge, whisper)
  is allowed to raise; nothing may take the manager down. A fatal error
  lands THIS stream in ``error`` state and the manager keeps supervising its
  siblings. The state machine itself is memory-first: a broken store must
  never kill the loop.
- reconnection backs off exponentially (2s doubling to a 120s cap, ±20%
  jitter) so a site outage can't become a hot loop. A resolution failure
  that means "channel not live" (streamlink's "No playable streams", A1's
  ``CaptureError(hint=..., offline=True)``) parks the stream in
  ``waiting_for_live`` on the offline poll instead of counting as a
  failure — an ended broadcast is expected back; it is not the same failure
  class as a broken site.
- heavy work never happens inline on the event loop: capture start/stop,
  extraction, audio decode and transcription all run in threads (killing
  an ffmpeg tree on Windows can block for seconds, and blocking the loop
  would stall every other stream's heartbeat).

Sibling contracts this module wires (every one injectable through
:class:`WatcherDeps`; landed bodies are wired as defaults, pending ones
degrade with one log and the watch continues):

- capture (A1, landed): ``StreamCapture(url, buffer, on_event=...)``;
  ``start()`` blocks on resolution and raises :class:`CaptureError` for a
  dead URL — the watcher runs it in a thread and classifies the exception
  (offline → park, anything else → backoff). ``on_event(kind, detail)``
  fires with kind in ``{"started", "ended", "error"}``: "ended" is a clean
  broadcaster sign-off (park), "error" is an abnormal ffmpeg exit
  (backoff). User-initiated ``stop()`` emits nothing.
- buffer (A1, landed): ``RingBuffer.extract(start_s, end_s, target)`` pins
  internally, so the watcher never manages clip pins itself.
  ``begin_epoch()`` is the watcher's signal that a reconnect happened.
- audio (A2, landed): module-level ``start(buffer, settings, on_signal,
  stop_event)`` run in a daemon thread; dies quietly when stubbed.
- chat (A5, landed): ``run_chat_reader(url, settings, on_signal=,
  stop_event=, clock=)`` — a blocking thread body, returning False when the
  URL has no chat. Its ``clock`` is wired to the buffer's stream time so
  chat and audio signals share one timeline.
- detector (A4, pending): ``SignalAggregator.add/evaluate(window_s)`` and
  ``Trigger.decide(window)`` -> ``.fire``. Absence degrades the watcher to
  capture-only.
- circuit (A4, pending): ``PerStreamCooldown.can_trigger/mark_clip`` and
  ``TokenBucket.try_acquire``. A pending circuit is fail-closed: candidates
  are refused with that reason rather than spending unguarded LLM calls.
- judge (A6, landed): ``judge_window(label, words, signals)`` ->
  ``LiveJudgement``, built on the STANDARD provider layer (the active
  provider from config, temperature 0.2, one repair retry) — the same path
  the highlights stage uses. A manager-provided semaphore caps concurrent
  judge calls across streams (hosted: 3, local: 1).

Clip production is NOT done here: accepted candidates are handed to the
manager's bounded clip queue as :class:`ClipWorkItem`s; the manager's
producer (A8b's ``clip_factory.produce``) re-extracts the final window from
the ring buffer and renders. The watcher's own extraction exists only to
feed the judge, under ``judge_*`` artifact names that never collide with
the factory's (the factory wipes its own artifacts when a window is
re-judged).
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
from .buffer import PENDING_CAP_BYTES, SEGMENT_S, RingBuffer  # noqa: F401  (PENDING_CAP_BYTES: manager)
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
#: Watcher heartbeat: how often signals drain and the buffer is pruned.
DEFAULT_TICK_S = 1.0
#: Poll interval while the channel is not live (resolution said "offline",
#: or the broadcast ended cleanly). Spec default: 30s.
DEFAULT_OFFLINE_POLL_S = 30.0
#: How often the rolling buffer is pruned while watching.
PRUNE_INTERVAL_S = 10.0
#: Signal window handed to the detector/aggregator, in stream seconds.
SIGNAL_WINDOW_S = 30.0
#: Judge trim advice that would tighten the window below this is ignored:
#: the judge's taste in pacing is not worth rendering a 3-second nothing.
TRIM_MIN_S = 10.0

#: Judge-phase artifacts inside ``live/clips/<clip_id>/``. Distinct from
#: clip_factory's RAW_NAME/AUDIO_NAME/TRANSCRIPT_NAME on purpose: the factory
#: wipes its own artifacts when a window is re-judged, and the judge's
#: evidence must survive that.
JUDGE_WINDOW_NAME = "judge_window.ts"
JUDGE_AUDIO_NAME = "judge_audio.wav"
JUDGE_TRANSCRIPT_NAME = "judge_transcript.json"

#: SSE channel naming (events.py contract): per-stream feed and the shared
#: all-clips feed every stream publishes clip events to.
CLIPS_CHANNEL = "live:-:clips"


def stream_channel(stream_id: str) -> str:
    """Per-stream SSE channel (events.py frozen naming)."""
    return f"live:{stream_id}"


# -- stub-safe helpers -----------------------------------------------------------

_WARNED_STUBS: set[str] = set()


def store_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a sibling-contract function, tolerating a still-pending stub.

    The parallel workers ship bodies behind frozen signatures; until then a
    few (circuit, events) raise ``NotImplementedError``. A watch must
    survive that — so the call is skipped with ONE log per function instead
    of killing the stream. The moment the sibling lands, calls just work.
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
            log.info("live.events.publish is a stub (A10 pending); SSE events dropped for now.")
    except Exception:
        log.warning("live event publish failed for %s/%s.", channel, event_type, exc_info=True)


#: Resolution failures matching any of these mean "channel not live" rather
#: than "broken". A1 marks its not-live errors with ``offline=True`` and a
#: hint containing streamlink's exact words; both signals route here.
_OFFLINE_MARKERS = ("no playable streams", "no streams found", "offline", "not live")


def _looks_offline(exc: BaseException) -> bool:
    if getattr(exc, "offline", False):
        return True
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _OFFLINE_MARKERS)


def _default_transcribe(path: Path, *_args: Any) -> Any:
    """Default transcriber: A8 reuses the pipeline stage unchanged."""
    from ..pipeline.transcribe import transcribe

    return transcribe(path)


def _default_audio_extract(source: Path, destination: Path) -> Any:
    """Default Whisper-input extraction: the pipeline stage unchanged."""
    from ..pipeline.prepare import extract_audio

    return extract_audio(source, destination)


# -- clip queue work items ---------------------------------------------------------


@dataclass
class ClipWorkItem:
    """One accepted candidate handed to the manager's bounded clip queue.

    The manager's producer (A8b) re-reads the live_clips row and re-extracts
    the final window from the stream's ring buffer at render time, so the
    item carries identity and accounting, not payload: ``source_path`` is
    the already-extracted judge window (its size feeds the manager's
    pending-bytes ledger), and ``judgement``/``words`` ride along for
    logging and test fakes.
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
    #: Size of the extracted judge window on disk.
    window_bytes: int = 0
    words: list = field(default_factory=list)
    caption_style: str = "bold_pop"
    ratio: str = "9:16"


@dataclass
class ClipWorkResult:
    """What a producer returns for one work item.

    Duck-compatible with A8b's ``ProduceResult`` (rendered_path, title,
    transcript_path; srt has no row column); ``error`` set means the render
    failed and the message is stored on the row verbatim.
    """

    rendered_path: Path | None = None
    transcript_path: Path | None = None
    title: str = ""
    error: str | None = None


async def placeholder_produce(item: ClipWorkItem) -> ClipWorkResult:
    """Fallback producer when clip_factory is unavailable: fail loudly."""
    raise NotImplementedError(
        f"Live clip rendering unavailable for {item.clip_id}. Pass a producer "
        "to LiveManager(produce=...)."
    )


# -- dependency bundle ---------------------------------------------------------------


@dataclass
class WatcherDeps:
    """Everything a watcher needs from the outside world, injectable.

    Every ``None`` entry is auto-defaulted at watcher construction (see the
    module docstring for each contract's shape); tests inject minimal fakes
    with the same method names.
    """

    #: (buffer, on_event) -> capture object (A1 StreamCapture by default)
    capture_factory: Callable[..., Any] | None = None
    #: () -> ring buffer
    buffer_factory: Callable[[], Any] | None = None
    #: () -> signal aggregator (None allowed: capture-only watch)
    aggregator_factory: Callable[[], Any] | None = None
    #: Trigger.decide(window) -> decision(.fire)
    trigger: Any = None
    #: PerStreamCooldown.can_trigger(stream_id) / .mark_clip(stream_id)
    cooldown: Any = None
    #: TokenBucket.try_acquire() ×3 — global judge budgets
    judge_minute: Any = None
    judge_hour: Any = None
    judge_day: Any = None
    #: Shared asyncio.Semaphore capping concurrent judge calls across
    #: streams (hosted providers: 3, local models: 1). None = unbounded.
    judge_semaphore: Any = None
    #: (path) -> transcript with .words (pipeline `transcribe` by default)
    transcribe: Callable[..., Any] | None = None
    #: (source, destination) -> None — Whisper-input extraction (the
    #: pipeline's extract_audio by default). Heavy: always run threaded.
    audio_extract: Callable[..., Any] | None = None
    #: async (label, words, signals) -> LiveJudgement (A6 by default)
    judge: Callable[..., Awaitable[LiveJudgement]] | None = None
    #: (buffer, settings, on_signal, stop_event) -> None — audio worker body
    audio_start: Callable[..., None] | None = None
    #: (url, settings, on_signal=, stop_event=, clock=) -> bool — chat body
    chat_runner: Callable[..., bool] | None = None
    #: (ClipWorkItem) -> bool — bounded enqueue; False drops the candidate
    enqueue_clip: Callable[[ClipWorkItem], bool] | None = None
    #: () -> bool — capture capacity gate
    may_capture: Callable[[], bool] | None = None
    #: (channel, event_type, data) — SSE publisher
    publish: Callable[[str, str, dict | None], None] | None = None
    #: async (seconds) -> None — sleep seam (tests fast-forward waits)
    sleep: Callable[[float], Awaitable[None]] | None = None
    max_connect_attempts: int | None = None


# -- the watcher -------------------------------------------------------------------


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
        self._chat_thread: threading.Thread | None = None
        self._signal_stop: threading.Event | None = None
        self._aggregator_broken = False
        self._prune_broken = False

        self._capture: Any = None
        #: Capture built by an in-flight resolution thread but not yet
        #: adopted by the loop — the hand-off cell for cancellation races.
        self._pending_capture: Any = None
        self._state: StreamState | None = None
        self._state_detail = ""
        self._stopping = False
        self._connect_attempts = 0
        # Set when the capture layer reports the STREAM ended cleanly: the
        # next supervisor pass sleeps out the offline poll in
        # waiting_for_live before looking for a new broadcast.
        self._park_pending = False
        self._park_reason = ""
        self._backoff_wait_s = 0.0
        self._next_prune = 0.0
        self._candidate_busy = False
        self._workers_started = False

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
        if deps.audio_extract is None:
            deps.audio_extract = _default_audio_extract
        if deps.judge is None:
            deps.judge = judge_window
        if deps.chat_runner is None:
            deps.chat_runner = self._default_chat_runner
        if deps.enqueue_clip is None:
            deps.enqueue_clip = self._no_clip_sink
            log.warning("Stream %s has no clip queue; candidates will be dropped.", self.id)
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

    def _default_capture_factory(self) -> Callable[..., Any]:
        def factory(buffer: Any, on_event: Callable[[str, str], None]) -> Any:
            from .capture import StreamCapture  # A1 owns this class

            return StreamCapture(self.url, buffer, on_event=on_event)

        return factory

    def _default_aggregator_factory(self) -> Any:
        try:
            from .detector import SignalAggregator  # A4 owns this class
        except ImportError:
            log.info("SignalAggregator unavailable (A4 pending): %s runs capture-only.", self.id)
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

    def _default_chat_runner(self, url: str, settings: Any, **kwargs: Any) -> bool:
        from .chat_signals import run_chat_reader  # A5 owns this module

        return run_chat_reader(url, settings, **kwargs)

    def _no_clip_sink(self, item: ClipWorkItem) -> bool:
        log.warning("Dropping accepted clip %s: no clip queue is wired.", item.clip_id)
        return False

    # -- lifecycle ---------------------------------------------------------------

    @property
    def state(self) -> StreamState | None:
        return self._state

    @property
    def state_detail(self) -> str:
        return self._state_detail

    @property
    def buffer(self) -> Any:
        """This stream's ring buffer (None until run() starts it)."""
        return self._buffer

    async def run(self) -> None:
        """Supervise this stream until stopped, paused, or terminally errored."""
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        try:
            self._buffer = self._deps.buffer_factory()
        except Exception as exc:
            # Crash isolation: a broken factory parks THIS stream in error;
            # the manager and its other watchers keep running.
            log.exception("Watcher for stream %s could not build its buffer.", self.id)
            self.set_state(StreamState.ERROR, f"buffer unavailable: {exc}", error=str(exc))
            return
        self._aggregator = self._deps.aggregator_factory()
        if self._loop is not None:
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
        if self._signal_stop is not None:
            self._signal_stop.set()  # audio + chat workers watch this event
        for thread in (self._audio_thread, self._chat_thread):
            if thread is not None and thread.is_alive():
                with contextlib.suppress(RuntimeError):
                    thread.join(timeout=2.0)
        self._stop_capture_async()
        pending = [task for task in self._bg_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            # A cancelled candidate task unwinds through its own ``finally``
            # (marking the live_clips row failed); await that, bounded.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait(pending, timeout=3.0)
        self._bg_tasks.clear()

    def stop(self, *, close_buffer: bool = False) -> None:
        """Flag this watcher to wind down (idempotent; sync emergency path).

        The supervisor task must be cancelled for the flag to be observed
        promptly — the manager does that; a parked ``offline_poll_s`` sleep
        otherwise finishes first. The manager pattern is therefore:
        ``task.cancel()`` then ``stop()`` (the task's own ``finally`` runs
        the full teardown, including awaiting cancelled candidate tasks).

        ``close_buffer`` additionally drops the buffer's bookkeeping (stream
        removal); pause keeps the buffer segments for the resume.
        """
        self._stopping = True
        if self._wake is not None:
            self._wake.set()
        if self._signal_stop is not None:
            self._signal_stop.set()
        self._stop_capture_async()
        if close_buffer and self._buffer is not None:
            with contextlib.suppress(Exception):
                self._buffer.close()

    def _stop_capture_async(self) -> None:
        """Detach the current capture and stop it off-loop, never blocking.

        ``capture.stop()`` kills the ffmpeg tree and joins threads — up to
        seconds on Windows. The loop must not pay that: watchers other than
        this one share it. Detaching synchronously matters more: the state
        machine must immediately treat the capture as gone.
        """
        capture, self._capture = self._capture, None
        pending, self._pending_capture = self._pending_capture, None
        for obj in (capture, pending):
            if obj is None:
                continue
            loop = self._loop
            if loop is not None and not loop.is_closed():
                try:
                    self._spawn(asyncio.to_thread(obj.stop))
                except RuntimeError:
                    with contextlib.suppress(Exception):
                        obj.stop()
            else:
                with contextlib.suppress(Exception):
                    obj.stop()

    # -- state machine ------------------------------------------------------------

    def set_state(
        self,
        new: StreamState,
        detail: str = "",
        *,
        error: str | None = None,
        initial: bool = False,
    ) -> bool:
        """Move to ``new`` if the frozen state machine allows it.

        Legal transitions are published as SSE ``state`` events and mirrored
        to the store; illegal jumps are logged and ignored so a late capture
        callback can never corrupt what the UI shows. The store write is
        best-effort: the in-memory state is the source of truth and a broken
        DB must never kill the watch loop.
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
        with contextlib.suppress(Exception):
            store_call(
                store.update_stream_state,
                self.id,
                state=new.value,
                state_detail=detail,
                error=error,
            )
        self._deps.publish(stream_channel(self.id), "state", data)
        if self._on_state is not None:
            with contextlib.suppress(Exception):
                self._on_state(self.id, old, new, detail)
        return True

    # -- supervisor loop -----------------------------------------------------------

    async def _supervise(self) -> None:
        assert self._wake is not None
        while not self._stopping:
            if self._state is not None and is_terminal(self._state):
                break
            if self._capture is None:
                await self._connect()
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self._tick_s)
            self._wake.clear()
            self._drain_signals()
            await self._maybe_prune()

    async def _connect(self) -> None:
        # The stream ended cleanly (broadcaster stopped) or resolution said
        # "offline": park on the offline cadence, not backoff — a next
        # broadcast is expected, it's just not up yet.
        if self._park_pending:
            self._park_pending = False
            reason, self._park_reason = self._park_reason, ""
            suffix = f": {reason}" if reason else ""
            await self._park_waiting(f"stream ended{suffix}")
            return
        if self._backoff_wait_s > 0:
            # A failed connect sleeps out its backoff BEFORE the capacity
            # check and the connecting mark: the UI must not show
            # "connecting" for the whole 120s cap when nothing is being
            # attempted yet, and a capacity slot must not be held while
            # sleeping.
            delay, self._backoff_wait_s = self._backoff_wait_s, 0.0
            self.set_state(StreamState.RECONNECTING, "backoff before retry")
            await self._deps.sleep(delay)
            if self._stopping:
                return
        # Capacity check and the CONNECTING mark are one synchronous block:
        # no await between them, so concurrent watchers can't all pass the
        # check and overshoot settings.workers.max_concurrent_streams.
        if not self._deps.may_capture():
            await self._park_waiting("waiting for capture capacity")
            return
        self.set_state(StreamState.CONNECTING, "connecting")
        try:
            await asyncio.to_thread(self._spawn_capture)
        except asyncio.CancelledError:
            # The resolution thread may still be running; if it already
            # built a capture, nobody else will ever stop it. The thread
            # side re-checks _stopping after start() — between the two,
            # the _pending_capture cell closes the race.
            self._stop_capture_async()
            raise
        except (ImportError, AttributeError) as exc:
            # A missing implementation — retrying can't fix an import error.
            self.set_state(StreamState.ERROR, f"capture unavailable: {exc}", error=str(exc))
            return
        except Exception as exc:
            await self._connect_failed(exc)
            return
        if self._stopping or self._pending_capture is None:
            self._stop_capture_async()
            return
        capture, self._pending_capture = self._pending_capture, None
        self._capture = capture
        self._connect_attempts = 0
        store_call(self._buffer.begin_epoch)  # reconnect = new epoch (A1)
        self.set_state(StreamState.WATCHING, "live")

    def _spawn_capture(self) -> None:
        """Worker-thread body: resolve + spawn ffmpeg (A1 blocks in start()).

        The built capture goes into ``_pending_capture`` immediately — the
        loop side adopts it as ``self._capture`` only after the await
        resumes, so a torn-down watcher can still find and stop it. If the
        watcher stopped while resolution ran, the thread stops its own
        capture: ffmpeg must never leak.
        """
        capture = self._deps.capture_factory(self._buffer, self._on_capture_event_thread)
        self._pending_capture = capture
        capture.start()
        if self._stopping and self._pending_capture is capture:
            with contextlib.suppress(Exception):
                capture.stop()
            self._pending_capture = None

    async def _park_waiting(self, detail: str) -> None:
        self.set_state(StreamState.WAITING_FOR_LIVE, detail)
        await self._deps.sleep(self._offline_poll_s)

    async def _connect_failed(self, exc: Exception) -> None:
        """A connect attempt died before ffmpeg ever spawned."""
        if _looks_offline(exc):
            # Not-live is not a failure: poll on the offline cadence so
            # successive not-live polls never eat the reconnect budget.
            self._connect_attempts = 0
            self._backoff_wait_s = 0.0
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
        """Exponential backoff: 2s, 4s, 8s ... capped at 120s, ±20% jitter."""
        slot = min(
            BACKOFF_BASE_S * (2 ** max(0, self._connect_attempts - 1)),
            BACKOFF_CAP_S,
        )
        jitter = 1.0 + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
        return slot * jitter

    # -- capture events (any thread) ------------------------------------------------

    def _on_capture_event_thread(self, kind: str, detail: str = "") -> None:
        """Capture callback — runs on A1's threads, must never block them."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        # Loop shut-down between the outer check and the marshal is normal
        # teardown racing a capture thread: dropping the event is correct.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._handle_capture_event, kind, detail)

    def _handle_capture_event(self, kind: str, detail: str) -> None:
        if self._stopping:
            return  # user-initiated stop: A1 emits nothing, but be safe
        if kind == "started":
            return  # informational — the connect path already records "live"
        if kind not in ("ended", "error"):
            log.debug("Stream %s: unknown capture event %r.", self.id, kind)
            return
        # Detach synchronously (state decisions below depend on "no
        # capture"); the expensive tree-kill runs in a thread.
        self._stop_capture_async()
        if kind == "ended":
            # Clean end (broadcaster stopped): the channel is expected to
            # come back, so park in waiting_for_live on the offline poll —
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

    # -- signals ----------------------------------------------------------------------

    def submit_signal(self, signal: DetectorSignal) -> None:
        """Thread-safe signal entry used by the audio/chat worker threads."""
        self._signal_q.put(signal)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._wake.set)

    def _stream_now(self) -> float:
        """Stream seconds captured so far — the shared detector timeline.

        Segment indices are global to the watch (A1: they never reset within
        a lifecycle), so ``index * SEGMENT_S`` is a monotonic stream clock.
        Chat's ``clock`` and audio's segment-derived times therefore share
        one base — which is what makes the trigger's combined window math
        meaningful.
        """
        buffer = self._buffer
        if buffer is None:
            return 0.0
        try:
            segments = buffer.segments
            if segments:
                return float(segments[-1].index) * SEGMENT_S
        except Exception:
            pass
        return 0.0

    def _start_signal_workers(self) -> None:
        """Start audio + chat worker threads once per watcher lifecycle.

        The buffer object survives reconnects, so the workers outlive any
        single capture session — they idle on a disconnected buffer rather
        than being restarted per connect.
        """
        if self._workers_started:
            return
        self._workers_started = True
        self._signal_stop = threading.Event()
        audio = self._deps.audio_start
        if audio is None:
            # Auto-detect A2's contract: a module-level start() function.
            from . import audio_signals

            audio = getattr(audio_signals, "start", None)
        if audio is not None:
            self._audio_thread = threading.Thread(
                target=self._run_worker,
                args=(audio, self._buffer, self._live, self.submit_signal, self._signal_stop),
                name=f"live-audio-{self.id}",
                daemon=True,
            )
            self._audio_thread.start()
        if self._live.detection.chat.enabled and "twitch.tv" in self.url.lower():
            self._chat_thread = threading.Thread(
                target=self._run_chat,
                name=f"live-chat-{self.id}",
                daemon=True,
            )
            self._chat_thread.start()

    def _run_worker(self, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except NotImplementedError:
            log.info("Audio signals unavailable (A2 pending) for %s.", self.id)
        except Exception:
            # A detector thread dying must never surface in the supervisor.
            log.exception("Audio signal worker for %s died.", self.id)

    def _run_chat(self) -> None:
        stop_event = self._signal_stop
        assert stop_event is not None
        try:
            ran = self._deps.chat_runner(
                self.url,
                self._live,
                on_signal=self.submit_signal,
                stop_event=stop_event,
                clock=self._stream_now,
            )
            if ran is False:
                log.debug("Stream %s: no chat signal for this URL.", self.id)
        except NotImplementedError:
            log.info("Chat signals unavailable (A5 pending) for %s.", self.id)
        except Exception:
            # Chat is one cheap side-signal: its death degrades, never errors.
            log.exception("Chat worker for %s died.", self.id)

    def _drain_signals(self) -> None:
        """Fold queued signals into the aggregator; maybe fire a candidate."""
        added = False
        while True:
            try:
                signal = self._signal_q.get_nowait()
            except queue.Empty:
                break
            added = True
            self._add_signal(signal)
        if not added:
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
            # extract() pins its own range internally (A1), so a prune
            # running in a thread alongside it is safe by those pins.
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
        if loop is None or loop.is_closed():
            return
        task = loop.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # -- candidate → judge → clip queue -------------------------------------------

    def _cooldown_ok(self) -> bool:
        try:
            return bool(self._deps.cooldown.can_trigger(self.id))
        except NotImplementedError:
            # Fail closed: an unenforceable cooldown means unbounded LLM
            # spend, which is exactly what the circuit exists to prevent.
            log.info("Cooldown circuit pending; refusing candidate on %s.", self.id)
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
        """Persist the judge-phase transcript next to the window artifacts.

        Rejected candidates keep their evidence for the UI; accepted ones
        keep it under the judge_* name so A8b's artifact wipes never destroy
        the judge's input.
        """
        save = getattr(transcript, "save", None)
        if save is None:
            return ""
        try:
            path = directory / JUDGE_TRANSCRIPT_NAME
            save(path)
            return str(path)
        except Exception:
            log.debug("Transcript save failed for %s.", self.id, exc_info=True)
            return ""

    def _trim_window(
        self,
        judgement: LiveJudgement,
        words: list,
        clip_start_s: float,
        clip_end_s: float,
        judge_start_s: float | None = None,
    ) -> tuple[float, float]:
        """Apply the judge's word-offset trim advice to the render window.

        Offsets are relative to the JUDGE window's first word (which may
        start earlier than the render window when ``judge_context_s`` gave
        the judge extra lead-up), so a trim point is first converted to
        stream time via ``judge_start_s + word time``, then clamped into the
        render window. Advice that would tighten below TRIM_MIN_S, or trim
        to a range entirely outside the render window, is ignored wholesale
        — a judge with bad taste in pacing shouldn't shrink a clip into
        nothing or blow past the pre-roll the buffer holds.
        """
        start_off = judgement.start_word_offset
        end_off = judgement.end_word_offset
        if (start_off is None and end_off is None) or not words:
            return clip_start_s, clip_end_s
        base_s = judge_start_s if judge_start_s is not None else clip_start_s
        start, end = clip_start_s, clip_end_s
        if start_off is not None and 0 <= start_off < len(words):
            start = base_s + float(getattr(words[start_off], "start", 0.0))
        if end_off is not None and 0 <= end_off < len(words):
            end = base_s + float(getattr(words[end_off], "end", 0.0))
        if judge_start_s is not None:
            # The advice referred to a wider window: clamp into the render
            # range instead of trusting it to fall inside.
            start = max(start, clip_start_s)
            end = min(end, clip_end_s)
        if end - start < TRIM_MIN_S or start < clip_start_s - 1e-6 or end > clip_end_s + 1e-6:
            return clip_start_s, clip_end_s
        return start, end

    async def _call_judge(self, words: list, signals: dict) -> LiveJudgement:
        """One judge call under the manager's shared concurrency semaphore."""
        coro = self._deps.judge(self.label or self.url, words, signals)
        sem = self._deps.judge_semaphore
        if sem is None:
            return await coro
        async with sem:
            return await coro

    async def _process_candidate(self, window: SignalWindow, decision: Any) -> None:
        """Turn one triggered window into a judged, enqueued clip candidate.

        Steps are cost-ordered: cooldown (free) → candidate row → global
        judge budgets (cheap) → buffer extraction + transcription (heavy)
        → the LLM judge → the bounded clip queue. Every drop writes the
        live_clips row so the UI can show WHY a candidate vanished rather
        than a silent miss.
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
            self._publish_clip(
                "clip_detecting", {"clip_id": clip_id, "trigger_end_s": trigger_end_s}
            )
            budget_ok, budget_reason = self._acquire_judge_budget()
            if not budget_ok:
                self._update_clip(
                    clip_id, status="rejected", reason=f"rate limited: {budget_reason}"
                )
                self._publish_clip("clip_rejected", {"clip_id": clip_id, "reason": budget_reason})
                return

            # The capture loop keeps running underneath: the state machine
            # explicitly allows watching → clipping → watching|reconnecting.
            entered_clipping = self.set_state(StreamState.CLIPPING, "candidate detected")
            directory = clip_dir(clip_id)
            source_path = directory / JUDGE_WINDOW_NAME
            audio_path = directory / JUDGE_AUDIO_NAME
            # The judge reads a WIDER window than the eventual clip: it needs
            # the setup that led to the moment, not just the moment. The
            # final render window (clip_start_s..clip_end_s) is re-extracted
            # from the buffer at render time, so this judgment window never
            # bounds the clip. Anchor back from the trigger so the context
            # covers the lead-up; the buffer clamps to whatever it holds.
            context_s = getattr(rolling, "judge_context_s", 0.0) or 0.0
            judge_end_s = min(clip_end_s, trigger_end_s + max(rolling.post_roll_s, 5.0))
            judge_start_s = max(0.0, trigger_end_s - context_s) if context_s > 0 else clip_start_s
            try:
                # extract() validates + pins + concatenates + unpins in one
                # call (A1); a pruned or epoch-crossing range raises a clear
                # error the candidate path records verbatim.
                await asyncio.to_thread(
                    self._buffer.extract, judge_start_s, judge_end_s, source_path
                )
                await asyncio.to_thread(self._deps.audio_extract, source_path, audio_path)
            except Exception as exc:
                if context_s > 0:
                    # The 4-minute lookback can predate the buffer after a
                    # fresh start or reconnect; the candidate is still
                    # viable on the plain render window — retry tight rather
                    # than losing the moment.
                    log.info(
                        "Wide judge window unavailable on %s (%s); retrying clip window.",
                        self.id,
                        exc,
                    )
                    judge_start_s = clip_start_s
                    try:
                        await asyncio.to_thread(
                            self._buffer.extract, clip_start_s, clip_end_s, source_path
                        )
                        await asyncio.to_thread(self._deps.audio_extract, source_path, audio_path)
                    except Exception as exc2:
                        raise RuntimeError(f"window extraction failed: {exc2}") from exc2
                else:
                    raise RuntimeError(f"window extraction failed: {exc}") from exc

            transcript = await asyncio.to_thread(self._deps.transcribe, audio_path)
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
                judgement = await self._call_judge(words, signals)
            except Exception as exc:
                # A6/provider failure — the stream keeps watching.
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

            final_start, final_end = self._trim_window(
                judgement, words, clip_start_s, clip_end_s, judge_start_s
            )
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
                clip_start_s=final_start,
                clip_end_s=final_end,
            )
            try:
                window_bytes = int(source_path.stat().st_size)
            except OSError:
                window_bytes = 0
            item = ClipWorkItem(
                clip_id=clip_id,
                stream_id=self.id,
                label=self.label or self.url,
                source_path=source_path,
                clip_start_s=final_start,
                clip_end_s=final_end,
                trigger_start_s=trigger_start_s,
                trigger_end_s=trigger_end_s,
                judgement=judgement,
                window_bytes=window_bytes,
                words=words,
            )
            enqueued = bool(self._deps.enqueue_clip(item))
            if not enqueued:
                self._update_clip(
                    clip_id, status="rejected", reason="clip queue full or disk pressure"
                )
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
            # Crash isolation: the candidate dies, the watch does not.
            log.exception("Candidate pipeline failed for stream %s.", self.id)
            self._update_clip_safe(clip_id, status="failed", error=str(exc))
            self._publish_clip("clip_failed", {"clip_id": clip_id, "error": str(exc)})
        finally:
            # Candidate work over. If the capture is still alive → back to
            # plain watching. If it died mid-clip, a capture callback has
            # already parked or reconnected us; the reconnecting hop is the
            # legal exit when clipping → waiting was refused, and the park
            # flag drives the rest from the connect path.
            self._candidate_busy = False
            if entered_clipping and self._state is StreamState.CLIPPING:
                if self._capture is not None:
                    self.set_state(StreamState.WATCHING, "live")
                else:
                    self.set_state(StreamState.RECONNECTING, "capture lost during clip")
