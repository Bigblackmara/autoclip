"""Unit tests for the A4 detection bodies: detector.py + circuit.py."""

from __future__ import annotations

import time

import pytest
from autoclip.live.circuit import PerStreamCooldown, TokenBucket
from autoclip.live.detector import (
    LONE_SPIKE_Z,
    MIN_STRENGTH,
    SignalAggregator,
    Trigger,
    WindowDecision,
)
from autoclip.live.settings import LiveDetectionSettings
from autoclip.live.signals import DetectorSignal, SignalKind


def audio(t: float, strength: float = 0.6, z: float = 2.0) -> DetectorSignal:
    return DetectorSignal(SignalKind.AUDIO_ENERGY, t_s=t, strength=strength, peak_z=z)


def rate(t: float, strength: float = 0.6, z: float = 4.0) -> DetectorSignal:
    return DetectorSignal(SignalKind.CHAT_RATE, t_s=t, strength=strength, peak_z=z)


def hype(t: float, strength: float = 0.9) -> DetectorSignal:
    return DetectorSignal(SignalKind.CHAT_HYPE, t_s=t, strength=strength, peak_z=3.0)


class TestTokenBucket:
    def test_acquire_within_capacity(self) -> None:
        bucket = TokenBucket(rate=1.0, capacity=3)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True

    def test_acquire_exhausted_fails(self) -> None:
        bucket = TokenBucket(rate=0.0, capacity=2)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False

    def test_refill_over_time(self) -> None:
        bucket = TokenBucket(rate=10.0, capacity=1)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False
        # Manually age the clock past the refill window.
        bucket._last = time.monotonic() - 1.0
        assert bucket.try_acquire() is True

    def test_refill_never_exceeds_capacity(self) -> None:
        bucket = TokenBucket(rate=100.0, capacity=2)
        bucket._last = time.monotonic() - 60.0
        assert bucket.try_acquire(1.5) is True
        assert bucket.try_acquire(1.5) is False  # capped at 2, spent 1.5

    def test_acquire_partial_token_fails_cleanly(self) -> None:
        bucket = TokenBucket(rate=0.0, capacity=1)
        assert bucket.try_acquire(0.5) is True
        assert bucket.try_acquire(0.5) is True
        assert bucket.try_acquire(0.1) is False


class TestPerStreamCooldown:
    def test_first_trigger_always_allowed(self) -> None:
        cooldown = PerStreamCooldown(cooldown_s=90.0)
        assert cooldown.can_trigger("s1") is True

    def test_cooldown_blocks_until_elapsed(self) -> None:
        cooldown = PerStreamCooldown(cooldown_s=90.0)
        cooldown.mark_clip("s1", now=100.0)
        assert cooldown.can_trigger("s1", now=100.0 + 89.9) is False
        assert cooldown.can_trigger("s1", now=100.0 + 90.0) is True

    def test_streams_are_independent(self) -> None:
        cooldown = PerStreamCooldown(cooldown_s=60.0)
        cooldown.mark_clip("s1", now=100.0)
        assert cooldown.can_trigger("s1", now=110.0) is False
        assert cooldown.can_trigger("s2", now=110.0) is True

    def test_remaining(self) -> None:
        cooldown = PerStreamCooldown(cooldown_s=60.0)
        assert cooldown.remaining("s1") == 0.0
        cooldown.mark_clip("s1", now=100.0)
        assert cooldown.remaining("s1", now=130.0) == pytest.approx(30.0)
        assert cooldown.remaining("s1", now=200.0) == 0.0


class TestSignalAggregator:
    def test_empty_evaluate_returns_empty_window(self) -> None:
        agg = SignalAggregator()
        window = agg.evaluate(30.0)
        assert window.signals == []
        assert window.start_s == 0.0 and window.end_s == 0.0

    def test_evaluate_keeps_recent_drops_old(self) -> None:
        agg = SignalAggregator()
        agg.add(audio(5.0))
        agg.add(audio(100.0))
        window = agg.evaluate(30.0)
        assert [s.t_s for s in window.signals] == [100.0]
        assert window.start_s == pytest.approx(70.0)
        assert window.end_s == pytest.approx(100.0)

    def test_retention_guard_trims_memory(self) -> None:
        agg = SignalAggregator()
        for i in range(5000):
            agg.add(audio(float(i)))
        agg.evaluate(30.0)
        assert len(agg) <= 31  # only the newest window survives

    def test_len(self) -> None:
        agg = SignalAggregator()
        assert len(agg) == 0
        agg.add(audio(1.0))
        assert len(agg) == 1


class TestTriggerDecide:
    def test_no_signals_never_fires(self) -> None:
        decision = Trigger().decide(
            __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
                start_s=0, end_s=10, signals=[]
            )
        )
        assert isinstance(decision, WindowDecision)
        assert decision.fire is False

    def test_chat_alone_never_fires(self) -> None:
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=10, signals=[rate(1.0), hype(2.0)]
        )
        decision = Trigger().decide(window)
        assert decision.fire is False

    def test_extreme_audio_spike_fires_alone(self) -> None:
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=10, signals=[audio(1.0, strength=0.8, z=LONE_SPIKE_Z)]
        )
        decision = Trigger().decide(window)
        assert decision.fire is True
        assert "extreme audio spike" in decision.reason

    def test_audio_below_floor_never_fires(self) -> None:
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=10, signals=[audio(1.0, strength=MIN_STRENGTH / 2, z=9.0)]
        )
        decision = Trigger().decide(window)
        assert decision.fire is False

    def test_audio_plus_chat_rate_fires(self) -> None:
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=10, signals=[audio(1.0), rate(2.0, z=4.0)]
        )
        decision = Trigger(LiveDetectionSettings()).decide(window)
        assert decision.fire is True
        assert "chat rate" in decision.reason

    def test_audio_plus_hype_fires(self) -> None:
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=10, signals=[audio(1.0), hype(2.0, strength=0.9)]
        )
        decision = Trigger().decide(window)
        assert decision.fire is True
        assert "chat hype" in decision.reason

    def test_weak_hype_does_not_corroborate(self) -> None:
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=10, signals=[audio(1.0), hype(2.0, strength=0.2)]
        )
        decision = Trigger().decide(window)
        assert decision.fire is False

    def test_sustained_excitement_fires(self) -> None:
        signals = [audio(float(t), strength=0.3, z=2.5) for t in range(0, 12, 2)]
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=12, signals=signals
        )
        decision = Trigger().decide(window)
        assert decision.fire is True
        assert "sustained" in decision.reason

    def test_moderate_audio_alone_does_not_fire(self) -> None:
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=10, signals=[audio(1.0, strength=0.4, z=2.5)]
        )
        decision = Trigger().decide(window)
        assert decision.fire is False
        assert "without corroboration" in decision.reason

    def test_constructor_accepts_detection_settings_and_none(self) -> None:
        Trigger(LiveDetectionSettings())
        Trigger(None)
        Trigger()

    def test_decision_carries_signals(self) -> None:
        signals = [audio(1.0, strength=0.9, z=10.0)]
        window = __import__("autoclip.live.signals", fromlist=["SignalWindow"]).SignalWindow(
            start_s=0, end_s=5, signals=signals
        )
        decision = Trigger().decide(window)
        assert decision.signals is not signals  # copied, not aliased
        assert decision.signals == signals
