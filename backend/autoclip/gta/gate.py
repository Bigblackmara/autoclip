"""One GPU-heavy job at a time across all of GTA Mode, live work first.

Batch transcription/rendering and GTA Live's transcribe+render all go
through this gate, so two Whisper models or a Whisper plus an encode never
compete for VRAM. Live work has priority: a waiting live clip is served
before the next batch job, which is what keeps the live turnaround short.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator


class HeavyGate:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._live_waiting = 0

    @contextlib.contextmanager
    def hold(self, *, live: bool = False) -> Iterator[None]:
        with self._cond:
            if live:
                self._live_waiting += 1
            try:
                while self._busy or (not live and self._live_waiting > 0):
                    self._cond.wait()
            finally:
                if live:
                    self._live_waiting -= 1
            self._busy = True
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()


gate = HeavyGate()
