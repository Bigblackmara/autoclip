"""Trigger decision logic for viral-moment detection.

Cheap signals run continuously; the aggregator rolls them up and the trigger
decides when a window has enough combined evidence to spend an LLM call on.

Firing rule (v1, deliberately simple and fail-closed):

* A window with **no** audio-energy spike never fires — chat alone can hype
  over an ordinary moment, so audio is the mandatory signal.
* An extreme audio spike (z >= ``LONE_SPIKE_Z``) fires alone.
* Otherwise audio needs a corroborating chat signal in the same window:
  a chat-rate spike, a chat hype burst, or sustained excitement (a signal
  run above the strength floor lasting >= ``SUSTAINED_S``).
* Weak-but-above-threshold audio without corroboration does not fire.

The Trigger never raises ``NotImplementedError``; the watcher treats that as
"detection pending" and degrades to capture-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .settings import LiveDetectionSettings
from .signals import DetectorSignal, SignalKind, SignalWindow

log = logging.getLogger(__name__)

#: Signal strength floor — below this a signal is treated as noise.
MIN_STRENGTH = 0.05
#: How strong a chat hype signal must be to corroborate audio on its own.
HYPE_STRENGTH = 0.5
#: How long (stream seconds) sustained excitement must last to corroborate.
SUSTAINED_S = 3.0
#: z-score an audio spike needs to fire WITHOUT any chat corroboration.
LONE_SPIKE_Z = 4.0
#: Max gap (seconds) between signals still counted as a contiguous run.
_RUN_GAP_S = 8.0
#: Hard cap on retained signals (long-session memory guard).
_MAX_MEMORY = 4096


@dataclass
class WindowDecision:
    """Result of :meth:`Trigger.decide`.

    The watcher reads ``decision.fire`` — the attribute name is frozen by
    the watcher contract.
    """

    fire: bool = False
    reason: str = ""
    signals: list[DetectorSignal] = field(default_factory=list)


class SignalAggregator:
    """Collects cheap-detector signals and evaluates rolling windows.

    The watcher serializes calls onto its event-loop thread (its signal
    queue drains on the loop), so no internal lock is needed.
    """

    def __init__(self) -> None:
        self._signals: list[DetectorSignal] = []

    def add(self, signal: DetectorSignal) -> None:
        """Record one observation. Duplicate timestamps are allowed."""
        self._signals.append(signal)

    def evaluate(self, window_s: float = 30.0) -> SignalWindow:
        """Return the last ``window_s`` seconds of signals (stream time).

        Signals older than the window are dropped; a burst older than the
        window cannot fire a stale candidate.
        """
        if not self._signals:
            return SignalWindow(start_s=0.0, end_s=0.0, signals=[])
        newest_t = max(s.t_s for s in self._signals)
        oldest_t = max(0.0, newest_t - window_s)
        kept = [s for s in self._signals if s.t_s >= oldest_t]
        if len(self._signals) > _MAX_MEMORY:
            self._signals = kept  # retention guard for very long sessions
        return SignalWindow(start_s=oldest_t, end_s=newest_t, signals=kept)

    def __len__(self) -> int:
        return len(self._signals)


class Trigger:
    """Decides whether a signal window is worth an LLM judge call.

    Accepts the live ``detection`` settings (the watcher passes
    ``LiveSettings.detection``); the constructor also works with no
    arguments, using the frozen defaults.
    """

    def __init__(self, detection: LiveDetectionSettings | None = None) -> None:
        self._detection = detection or LiveDetectionSettings()

    def decide(self, window: SignalWindow) -> WindowDecision:
        """Return a :class:`WindowDecision` for ``window`` — never raises."""
        if not window.signals:
            return WindowDecision(fire=False, reason="no signals")

        audio = window.strongest_of(SignalKind.AUDIO_ENERGY)
        if audio is None:
            return WindowDecision(fire=False, reason="no audio energy signal")
        if audio.strength < MIN_STRENGTH:
            return WindowDecision(fire=False, reason="audio below strength floor")

        # Extreme spike fires on its own.
        if audio.peak_z >= LONE_SPIKE_Z:
            return WindowDecision(
                fire=True,
                reason=f"extreme audio spike z={audio.peak_z:.1f}",
                signals=list(window.signals),
            )

        chat_rate = window.strongest_of(SignalKind.CHAT_RATE)
        if chat_rate is not None and chat_rate.peak_z >= self._detection.chat.rate_z:
            return WindowDecision(
                fire=True,
                reason=f"audio spike + chat rate z={chat_rate.peak_z:.1f}",
                signals=list(window.signals),
            )

        chat_hype = window.strongest_of(SignalKind.CHAT_HYPE)
        if chat_hype is not None and chat_hype.strength >= HYPE_STRENGTH:
            return WindowDecision(
                fire=True,
                reason=f"audio spike + chat hype strength={chat_hype.strength:.2f}",
                signals=list(window.signals),
            )

        # Sustained excitement: a contiguous run of qualifying signals.
        run_s = self._longest_run_above_floor(window)
        if run_s >= SUSTAINED_S:
            return WindowDecision(
                fire=True,
                reason=f"audio spike + sustained {run_s:.0f}s",
                signals=list(window.signals),
            )

        return WindowDecision(fire=False, reason="audio spike without corroboration")

    def _longest_run_above_floor(self, window: SignalWindow) -> float:
        """Longest contiguous span (stream seconds) of signals above floor.

        Any signal kind counts; what matters is that excitement persisted
        across time, not that it kept re-spiking.
        """
        qualifying = sorted(
            (s for s in window.signals if s.strength >= MIN_STRENGTH),
            key=lambda s: s.t_s,
        )
        if not qualifying:
            return 0.0
        longest = 0.0
        run = 0.0
        prev_t = qualifying[0].t_s
        for s in qualifying[1:]:
            gap = s.t_s - prev_t
            if gap <= _RUN_GAP_S:
                run += gap
            else:
                longest = max(longest, run)
                run = 0.0
            prev_t = s.t_s
        return max(longest, run)


__all__ = ["SignalAggregator", "Trigger", "WindowDecision"]
