"""Signals shared across Live Watch detectors.

Frozen types every detector worker (audio, chat, later visual) codes against.
Aggregation and triggering live in detector.py (A4 owns that file's bodies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SignalKind(StrEnum):
    AUDIO_ENERGY = "audio_energy"
    CHAT_RATE = "chat_rate"
    CHAT_HYPE = "chat_hype"
    LLM_JUDGE = "llm_judge"


@dataclass
class DetectorSignal:
    """One cheap-detector observation, timestamped in stream time.

    ``t_s`` is seconds since capture start for the stream. ``strength``
    normalizes each detector's own scale; ``peak_z`` carries the detector's
    statistical extremity (z-score or equivalent) for logging/judging.
    """

    kind: SignalKind
    t_s: float
    strength: float
    detail: str = ""
    peak_z: float = 0.0


@dataclass
class SignalWindow:
    """Rolling multi-detector window the aggregator sees."""

    start_s: float
    end_s: float
    signals: list[DetectorSignal] = field(default_factory=list)

    def strength_of(self, kind: SignalKind) -> float:
        """Peak strength of ``kind`` in the window, 0 if absent."""
        return max((s.strength for s in self.signals if s.kind == kind), default=0.0)

    def strongest_of(self, kind: SignalKind) -> DetectorSignal | None:
        best: DetectorSignal | None = None
        for s in self.signals:
            if s.kind == kind and (best is None or s.strength > best.strength):
                best = s
        return best


__all__ = ["DetectorSignal", "SignalKind", "SignalWindow"]
