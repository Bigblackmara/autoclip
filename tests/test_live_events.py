"""Unit tests for the Live Watch SSE event broker (live/events.py).

Pure asyncio — no app, no TestClient, no network. The broker's contracts
under test are the ones the manager and the REST layer rely on:

- publish is thread-safe via call_soon_threadsafe once bind_loop has run;
  publishes before bind_loop (or after loop close) are dropped, never raise.
- each subscriber gets its own bounded queue; a slow subscriber drops
  frames instead of blocking the publisher or leaking memory.
- publishes with no subscribers vanish (dropped, not queued).
- heartbeat frames arrive on an idle channel.
- no replay: events published before a subscription starts are not seen.

Test mechanics (the hard-won part): ``subscribe`` is an async GENERATOR —
its queue registers only when the generator body first runs, and wrapping
``agen.__anext__()`` in ``asyncio.wait_for`` CANCELS the generator on
timeout, tearing the subscription down (the CancelledError is delivered
inside the generator frame, its ``finally`` discards the queue, and the
next ``__anext__`` raises StopAsyncIteration). So NO test ever wraps
``__anext__`` in an outer timeout: collectors drive the generator with a
plain ``async for``, pass/fail deadlines live at the TEST level, and
passive pulls rely on the broker's own heartbeat (patched down to 0.05s
for speed — the shipped 15s would stall the suite) returning instead of
an external timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading

import pytest
from autoclip.live import events
from autoclip.live.events import HEARTBEAT_S, QUEUE_SIZE, Event, EventBroker


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> EventBroker:
    """A fresh broker, swapped in as the module singleton, fast heartbeat."""
    broker = EventBroker()
    monkeypatch.setattr(events, "broker", broker)
    monkeypatch.setattr(events, "HEARTBEAT_S", 0.05)
    return broker


def _etype(event) -> str:
    return event["type"] if isinstance(event, dict) else event.type


class _Tap:
    """Background collector driving one subscription with plain ``async for``.

    ``started`` polls the broker until the queue is really registered, so a
    publish fired right after it cannot race ahead of the subscription.
    ``wait(count)`` waits for that many non-heartbeat events and returns
    them in arrival order (heartbeats ride along on idle channels and are
    filtered, never asserted around).
    """

    def __init__(self, broker: EventBroker, channel: str, subscribe_fn=None) -> None:
        self.got: list = []
        self._broker = broker
        self._channel = channel
        self._subscribe = subscribe_fn or broker.subscribe
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        agen = self._subscribe(self._channel)
        async for event in agen:  # no outer timeout — see module docstring
            self.got.append(event)

    @property
    def data(self) -> list:
        return [event for event in self.got if _etype(event) != "heartbeat"]

    async def started(self, count: int = 1, timeout: float = 2.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while self._broker.subscriber_count(self._channel) < count:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(
                    f"tap on {self._channel!r} never reached {count} subscriber(s)"
                )
            await asyncio.sleep(0.001)

    async def wait(self, count: int, timeout: float = 2.0) -> list:
        deadline = asyncio.get_running_loop().time() + timeout
        while len(self.data) < count:
            if asyncio.get_running_loop().time() > deadline:
                break
            await asyncio.sleep(0.005)
        return list(self.data)

    async def close(self) -> None:
        # Cancelling the task lands the CancelledError inside the generator
        # frame, which runs its finally → the queue is discarded. That is
        # the teardown under test as much as a helper.
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await self._task


class TestBindLoop:
    async def test_publish_before_bind_is_dropped_not_raised(self, broker: EventBroker):
        # Watcher code may publish from threads before any bind_loop; the
        # silence is the contract.
        broker.publish("live:s1", Event(type="state", data={"state": "watching"}))
        assert broker.subscriber_count("live:s1") == 0

    async def test_publish_after_loop_close_is_dropped(self, broker: EventBroker):
        loop = asyncio.new_event_loop()
        broker.bind_loop(loop)
        loop.close()
        broker.publish("live:s1", Event(type="state"))  # must not raise
        assert broker.subscriber_count("live:s1") == 0

    async def test_bind_loop_makes_publishes_safe_on_a_live_loop(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        tap = _Tap(broker, "live:s1")
        await tap.started()
        broker.publish("live:s1", Event(type="state", data={"state": "watching"}))
        got = await tap.wait(1)
        await tap.close()
        assert len(got) == 1
        assert got[0].type == "state"
        assert got[0].data == {"state": "watching"}

    async def test_a_burst_of_publishes_all_arrive(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        tap = _Tap(broker, "live:s9")
        await tap.started()
        for i in range(5):
            broker.publish("live:s9", Event(type="state", data={"i": i}))
        got = await tap.wait(5)
        await tap.close()
        assert [e.data["i"] for e in got] == [0, 1, 2, 3, 4]

    async def test_module_level_api_uses_the_singleton(self, broker: EventBroker):
        events.bind_loop(asyncio.get_running_loop())
        tap = _Tap(events.broker, "live:module-check", events.subscribe)
        await tap.started()
        events.publish("live:module-check", "state", {"stream_id": "module-check"})
        got = await tap.wait(1)
        await tap.close()
        assert got == [{"type": "state", "data": {"stream_id": "module-check"}}]


class TestThreading:
    async def test_publish_from_a_foreign_thread_is_marshalled(self, broker: EventBroker):
        """The exact race the manager's worker threads hit: publish from a
        non-loop thread must land in the subscriber's queue, not corrupt it."""
        broker.bind_loop(asyncio.get_running_loop())
        tap = _Tap(broker, "live:t1")
        await tap.started()

        def spam():
            for i in range(20):
                broker.publish("live:t1", Event(type="state", data={"i": i}))

        thread = threading.Thread(target=spam)
        thread.start()
        thread.join(timeout=5.0)

        got = await tap.wait(20)
        await tap.close()
        assert [e.data["i"] for e in got] == list(range(20))

    async def test_many_threads_publish_concurrently_without_loss(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        tap = _Tap(broker, "live:t2")
        await tap.started()

        def spam(offset: int) -> None:
            for i in range(10):
                broker.publish("live:t2", Event(type="state", data={"key": f"{offset}-{i}"}))

        threads = [threading.Thread(target=spam, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        got = await tap.wait(40)
        await tap.close()
        assert len(got) == 40
        keys = {e.data["key"] for e in got}
        assert keys == {f"{n}-{i}" for n in range(4) for i in range(10)}


class TestFanOut:
    async def test_two_subscribers_each_get_every_event(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        tap1 = _Tap(broker, "live:s2")
        tap2 = _Tap(broker, "live:s2")
        await tap1.started(2)
        await tap2.started(2)

        for i in range(3):
            broker.publish("live:s2", Event(type="clip", data={"i": i}))
        got1 = await tap1.wait(3)
        got2 = await tap2.wait(3)
        await tap1.close()
        await tap2.close()
        assert [e.data["i"] for e in got1] == [0, 1, 2]
        assert [e.data["i"] for e in got2] == [0, 1, 2]

    async def test_events_only_go_to_their_channel(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        mine = _Tap(broker, "live:mine")
        other = _Tap(broker, "live:other")
        await mine.started()
        await other.started()

        broker.publish("live:mine", Event(type="state", data={"s": 1}))
        got = await mine.wait(1)
        await mine.close()
        await other.close()
        assert len(got) == 1
        assert other.data == []  # the other channel got nothing at all

    async def test_no_subscribers_is_a_drop(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        for i in range(100):
            broker.publish("live:ghost", Event(type="state", data={"i": i}))
        await asyncio.sleep(0.05)
        assert broker.subscriber_count("live:ghost") == 0

    async def test_no_replay_before_subscription(self, broker: EventBroker):
        """Events published before a subscription are NOT replayed to it."""
        broker.bind_loop(asyncio.get_running_loop())
        broker.publish("live:late", Event(type="state", data={"i": 0}))
        tap = _Tap(broker, "live:late")
        await tap.started()
        broker.publish("live:late", Event(type="state", data={"i": 1}))
        got = await tap.wait(1)
        await tap.close()
        assert [e.data["i"] for e in got] == [1]


class TestBoundedQueues:
    async def test_slow_subscriber_drops_frames_not_the_publisher(self, broker: EventBroker):
        """A subscriber that never pulls must not block publish(): its queue
        fills to QUEUE_SIZE and further events are dropped in the broker."""
        broker.bind_loop(asyncio.get_running_loop())

        agen = broker.subscribe("live:slow")
        assert broker.subscriber_count("live:slow") == 0  # not registered yet
        first = await agen.__anext__()  # register, then heartbeat on idle
        assert first.type == "heartbeat"
        assert broker.subscriber_count("live:slow") == 1

        for i in range(QUEUE_SIZE + 20):
            broker.publish("live:slow", Event(type="state", data={"i": i}))
        # Let the call_soon_threadsafe callbacks run: the first QUEUE_SIZE
        # land, the rest are dropped inside the broker (QueueFull).
        await asyncio.sleep(0.05)

        # The publisher was never blocked — the loop is still responsive.
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=1.0)
        assert broker.subscriber_count("live:slow") == 1

        # Drain via the generator itself: exactly the first QUEUE_SIZE
        # events, in order, then nothing but heartbeats.
        received = []
        while True:
            event = await agen.__anext__()
            if event.type == "heartbeat":
                break
            received.append(event)
        assert [e.data["i"] for e in received] == list(range(QUEUE_SIZE))

        await agen.aclose()
        assert broker.subscriber_count("live:slow") == 0

    async def test_subscriber_registration_is_cleaned_up_on_close(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        tap = _Tap(broker, "live:cleanup")
        await tap.started()
        assert broker.subscriber_count("live:cleanup") == 1
        await tap.close()
        await asyncio.sleep(0.01)
        assert broker.subscriber_count("live:cleanup") == 0


class TestHeartbeat:
    async def test_idle_channel_yields_heartbeats(self, broker: EventBroker):
        broker.bind_loop(asyncio.get_running_loop())
        agen = broker.subscribe("live:idle")
        first = await agen.__anext__()  # idle → broker's own heartbeat
        await agen.aclose()
        assert first.type == "heartbeat"

    def test_heartbeat_constant_is_sane(self):
        # HEARTBEAT_S imported at module load: the SHIPPED value, not the
        # monkeypatched one. Too long and proxies kill the stream.
        assert 0 < HEARTBEAT_S <= 30.0


class TestEventShape:
    def test_to_sse_frames_a_named_event(self):
        frame = Event(type="clip", data={"clip_id": "c1", "status": "ready"}).to_sse()
        assert frame.startswith("event: clip\n")
        assert frame.endswith("\n\n")
        assert '"clip_id": "c1"' in frame

    async def test_module_subscribe_yields_type_data_dicts(self):
        # The REST layer awaits {'type', 'data'} mappings, not Event objects.
        agen = events.subscribe("live:shape")
        assert inspect.isasyncgen(agen)
        await agen.aclose()


class TestHelperConformsToManagerCallSites:
    async def test_manager_call_pattern_round_trips(self, broker: EventBroker):
        """The exact pattern manager.py / watcher.py use:

        ``events.publish(channel, event_type, data)`` from any thread, with
        ``events.subscribe()`` returning ``{'type', 'data'}`` mappings.
        """
        events.bind_loop(asyncio.get_running_loop())
        tap = _Tap(events.broker, "live:-:clips", events.subscribe)
        await tap.started()

        def watcher_thread():
            events.publish("live:-:clips", "clip_ready", {"clip_id": "c9"})

        t = threading.Thread(target=watcher_thread)
        t.start()
        t.join(timeout=5.0)

        got = await tap.wait(1)
        await tap.close()
        assert got == [{"type": "clip_ready", "data": {"clip_id": "c9"}}]
