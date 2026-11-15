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
        """Try to spend ``n`` tokens; True when the budget allows it."""
        self._refill()
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.rate)
        self._last = now


class JudgeCircuitBreaker:
    """Backs the LLM judge off after provider errors (shared by all streams).

    A provider at capacity ("Service temporarily at capacity", 429/529) or
    down answers every stream the same way, so retrying on every trigger
    only burns rate limits. After ``n`` consecutive failures the breaker is
    open for ``base_s * 2**(n-1)`` seconds (capped at ``cap_s``); while it
    is open candidates are refused locally. One success closes it.
    """

    def __init__(
        self,
        base_s: float = 30.0,
        cap_s: float = 600.0,
        *,
        clock=time.monotonic,
    ) -> None:
        self.base_s = base_s
        self.cap_s = cap_s
        self._clock = clock
        self.failures = 0
        self.last_error = ""
        self._open_until = 0.0

    def remaining(self) -> float:
        """Seconds the breaker stays open (0 when calls are allowed)."""
        return max(0.0, self._open_until - self._clock())

    def allow(self) -> bool:
        return self.remaining() <= 0.0

    def failure(self, error: str = "") -> float:
        """Record a provider failure; returns the new open duration."""
        self.failures += 1
        self.last_error = error[:300]
        delay = min(self.cap_s, self.base_s * (2 ** (self.failures - 1)))
        self._open_until = self._clock() + delay
        return delay

    def success(self) -> None:
        self.failures = 0
        self.last_error = ""
        self._open_until = 0.0


class PerStreamCooldown:
    """Tracks last-clip-per-stream timestamps; cooldown enforced per stream."""

    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self._last_clip_at: dict[str, float] = {}

    def can_trigger(self, stream_id: str, *, now: float | None = None) -> bool:
        """True when ``stream_id`` may trigger (past its cooldown)."""
        last = self._last_clip_at.get(stream_id)
        if last is None:
            return True
        current = now if now is not None else time.monotonic()
        return (current - last) >= self.cooldown_s

    def mark_clip(self, stream_id: str, *, now: float | None = None) -> None:
        """Start a fresh cooldown for ``stream_id``."""
        self._last_clip_at[stream_id] = now if now is not None else time.monotonic()

    def remaining(self, stream_id: str, *, now: float | None = None) -> float:
        """Seconds until ``stream_id`` may trigger again (0 when clear)."""
        last = self._last_clip_at.get(stream_id)
        if last is None:
            return 0.0
        current = now if now is not None else time.monotonic()
        return max(0.0, self.cooldown_s - (current - last))
