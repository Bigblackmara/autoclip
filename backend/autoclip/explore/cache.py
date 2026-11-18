"""A small thread-safe in-memory TTL cache for Explore results.

yt-dlp takes seconds per search; people flip between tabs and come back to
the same result list constantly. Failures are never cached, so a transient
YouTube refusal is retried on the next request.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

#: Default lifetime of a cached entry.
DEFAULT_TTL_S = 10 * 60
#: Oldest entries are dropped beyond this many.
MAX_ENTRIES = 256


class TTLCache:
    def __init__(
        self, *, max_entries: int = MAX_ENTRIES, clock: Callable[[], float] = time.monotonic
    ):
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries
        self._clock = clock

    def get(self, key: str) -> Any | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            expires, value = hit
            if self._clock() >= expires:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Any, ttl_s: float) -> None:
        with self._lock:
            self._data[key] = (self._clock() + ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def cached(self, key: str, compute: Callable[[], T], ttl_s: float = DEFAULT_TTL_S) -> T:
        """The cached value for ``key``, computing (and storing) it on a miss."""
        hit = self.get(key)
        if hit is not None:
            return hit
        value = compute()
        self.put(key, value, ttl_s)
        return value


cache = TTLCache()
