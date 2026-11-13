"""SSE broker for Live Watch (mirrors jobs/events.py, separate channels).

Channel naming (watcher.py contract): per-stream feeds ``live:<stream_id>``
plus the shared all-clips feed ``live:-:clips``. Watchers — and anything
else running in a worker thread — publish through :func:`publish`, which
marshals onto the loop recorded by :func:`bind_loop` exactly like
``jobs.events``: ``call_soon_threadsafe`` is the only thread-safe entry
into loop state, and ``queue.put_nowait`` from a foreign thread would
corrupt it.

Each subscriber gets its own bounded queue, so a slow client drops events
instead of growing memory without limit; publishes with no subscribers are
dropped, never queued forever. A ``bind_loop`` before any publish is what
makes publishes safe — the manager calls it from ``start()``.

Event types the frontend subscribes to (F1/F2): ``snapshot``, ``state``,
``clip``, ``clip_status``, ``upload``, ``heartbeat``. Watchers publish
their own internal names (``clip_detecting``, ``clip_queued`` …);
:mod:`autoclip.api.live` maps those to the frontend vocabulary at the
subscriber boundary, so neither side knows the other's names.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Per-subscriber buffer, mirrored from jobs/events.py: deep enough to
#: absorb a burst, shallow enough that a dead client is noticed quickly.
QUEUE_SIZE = 64

#: Emitted when nothing has arrived for this long, so proxies and browsers
#: don't time an SSE connection out. 15s is comfortably under EventSource's
#: own reconnect heuristic, keeping the connection visibly alive.
HEARTBEAT_S = 15.0


@dataclass
class Event:
    """One SSE message on a live channel."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = json.dumps({"type": self.type, **self.data})
        return f"event: {self.type}\ndata: {payload}\n\n"


class EventBroker:
    """Fan-out of live events to any number of SSE subscribers."""

    def __init__(self) -> None:
        # Mutated only from the loop (subscribe registers from a coroutine,
        # _publish_on_loop runs via call_soon_threadsafe), so plain dict/set
        # access needs no lock — that is the invariant bind_loop establishes.
        self._subscribers: dict[str, set[asyncio.Queue[Event]]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Record the loop that owns the subscriber queues (manager.start)."""
        self._loop = loop

    # -- publishing --------------------------------------------------------

    def publish(self, channel: str, event: Event) -> None:
        """Publish an event to ``channel`` from any thread.

        A publish with no subscribers is a drop, not a queue-build: Live
        Watch can run for hours with nobody listening, and holding every
        state change for a subscriber who may never arrive would leak.
        An unbound broker (no bind_loop yet, or a closed loop) drops too —
        the watcher contract makes fire-and-forget publishing safe.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        # Raises if the loop closed between the check above and this call.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._publish_on_loop, channel, event)

    def _publish_on_loop(self, channel: str, event: Event) -> None:
        for queue in tuple(self._subscribers.get(channel, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Slow client: losing an intermediate frame is harmless,
                # the next one carries the current state.
                log.debug("Dropped live event for a slow subscriber on %s.", channel)

    # -- subscribing -------------------------------------------------------

    async def subscribe(self, channel: str) -> AsyncIterator[Event]:
        """Yield events for ``channel`` until the client disconnects."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._subscribers.setdefault(channel, set()).add(queue)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
                except TimeoutError:
                    yield Event(type="heartbeat")
                    continue
                yield event
        finally:
            subscribers = self._subscribers.get(channel)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    self._subscribers.pop(channel, None)

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers.get(channel, ()))


#: Process-wide broker. One per app; tests build fresh instances.
broker = EventBroker()


def bind_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Record the loop that publishes marshal onto (called by manager.start)."""
    broker.bind_loop(loop)


def publish(channel: str, event_type: str, data: dict | None = None) -> None:
    """Thread-safe publish of one event; callable from watcher threads."""
    broker.publish(channel, Event(type=event_type, data=dict(data or {})))


async def subscribe(channel: str) -> AsyncIterator[dict[str, Any]]:
    """Async iterator of ``{'type', 'data'}`` mappings for ``channel``.

    The wire shape :mod:`autoclip.api.live` serializes; ``heartbeat``
    frames arrive on idle channels.
    """
    async for event in broker.subscribe(channel):
        yield {"type": event.type, "data": event.data}


@contextlib.asynccontextmanager
async def bound_loop():
    """Bind the broker to the running loop for the app's lifetime."""
    broker.bind_loop(asyncio.get_running_loop())
    try:
        yield broker
    finally:
        broker.bind_loop(None)  # type: ignore[arg-type]
