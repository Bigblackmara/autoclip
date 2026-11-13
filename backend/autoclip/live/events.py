"""SSE broker for Live Watch (mirrors jobs/events.py, separate channels).

A10 (API) owns this file. Live watchers publish state changes and clip events
on channels named ``live:<stream_id>`` and ``live:-:clips`` (all-clips feed).
Same thread-marshalling rules as jobs.events (call_soon_threadsafe).
"""

from __future__ import annotations


async def subscribe(channel: str):
    """Async iterator of events for ``channel``; see jobs.events.subscribe."""
    raise NotImplementedError("A10: implement")


def publish(channel: str, event_type: str, data: dict | None = None) -> None:
    """Thread-safe publish; callable from watcher threads."""
    raise NotImplementedError("A10: implement")


def bind_loop(loop) -> None:
    raise NotImplementedError("A10: implement")
