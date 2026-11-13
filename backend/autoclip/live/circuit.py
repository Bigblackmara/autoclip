"""Per-stream cooldown circuit (skeleton — A4 fills bodies via a tiny helper).

The manager consults this before running any candidate through the pipeline:
after a clip is produced, the SAME stream's triggers are muted for
settings.live.rate_limit.cooldown_s seconds, and a global judge token bucket
(shared across streams) caps LLM spend per minute/hour/day.
"""

from __future__ import annotations

import time


class TokenBucket:
    """Minimal token bucket: capacity ``rate`` tokens, refilled each second."""

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self._last = time.monotonic()

    def try_acquire(self, n: float = 1.0) -> bool:
        raise NotImplementedError("A4: implement")

    def _refill(self) -> None:
        raise NotImplementedError("A4: implement")


class PerStreamCooldown:
    """Tracks last-clip-per-stream timestamps; cooldown enforced per stream."""

    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self._last_clip_at: dict[str, float] = {}

    def can_trigger(self, stream_id: str, *, now: float | None = None) -> bool:
        raise NotImplementedError("A4: implement")

    def mark_clip(self, stream_id: str, *, now: float | None = None) -> None:
        raise NotImplementedError("A4: implement")

    def remaining(self, stream_id: str, *, now: float | None = None) -> float:
        raise NotImplementedError("A4: implement")
