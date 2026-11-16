"""The Hype Curve: one 0-100 score per second of stream, plus its peaks.

Built BEFORE anything reads the transcript. Each signal is scored against
the stream's own normal level rather than a fixed threshold — a screamer's
baseline is loud, a chill streamer's isn't — then the available signals are
combined. Peaks ("markers") carry human-readable reasons so the judge knows
WHY a moment is hot (the streamer screamed, chat typed "clip" 14 times...).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .signals import SILENCE_DB, ChatSeries

#: Signal weights; renormalised over the signals a stream actually has.
WEIGHTS = {"audio": 0.30, "chat": 0.25, "keywords": 0.20, "replayed": 0.15, "clips": 0.10}

#: Baselines are medians over blocks this long, smoothed over this many blocks.
BLOCK_S = 10
BASELINE_BLOCKS = 31  # ~5 minutes either side

#: A marker needs at least this score, and markers are this far apart.
MARKER_MIN_SCORE = 30.0
MARKER_SPACING_S = 25


@dataclass
class Marker:
    t: float
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class HypeCurve:
    score: np.ndarray
    markers: list[Marker]
    available: list[str]
    #: Per-signal 0..1 contributions (kept for the UI's breakdown).
    components: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def seconds(self) -> int:
        return len(self.score)

    def peak(self, start_s: float, end_s: float) -> float:
        a = max(0, int(start_s))
        b = min(self.seconds, int(np.ceil(end_s)))
        if b <= a:
            return 0.0
        return float(self.score[a:b].max())

    def markers_between(self, start_s: float, end_s: float) -> list[Marker]:
        return [m for m in self.markers if start_s <= m.t < end_s]

    # -- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "score": [round(float(v), 1) for v in self.score],
            "markers": [asdict(m) for m in self.markers],
            "available": self.available,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> HypeCurve:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            score=np.asarray(payload["score"], dtype=np.float32),
            markers=[Marker(**m) for m in payload["markers"]],
            available=list(payload["available"]),
        )


def _block_baseline(
    values: np.ndarray, *, statistic: str = "median"
) -> tuple[np.ndarray, np.ndarray]:
    """Slow-moving baseline and spread of ``values``, expanded back per second.

    Medians over 10 s blocks, then a centred running median over ~5 min of
    blocks: robust to the very spikes being looked for.
    """
    n = len(values)
    if n == 0:
        return values.copy(), values.copy()
    blocks = int(np.ceil(n / BLOCK_S))
    padded = np.full(blocks * BLOCK_S, np.nan, dtype=np.float64)
    padded[:n] = values
    per_block = np.nanmedian(padded.reshape(blocks, BLOCK_S), axis=1)
    half = BASELINE_BLOCKS // 2
    edge = np.pad(per_block, (half, half), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(edge, BASELINE_BLOCKS)
    centre = np.median(windows, axis=1)
    spread = np.median(np.abs(windows - centre[:, None]), axis=1) * 1.4826
    return (
        np.repeat(centre, BLOCK_S)[:n].astype(np.float32),
        np.repeat(spread, BLOCK_S)[:n].astype(np.float32),
    )


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(values) == 0:
        return values.astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / width
    # Edge padding, not zeros: zero padding made the first and last seconds
    # of every stream look like a loudness spike (and chat look quieter).
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(values.astype(np.float32), (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def audio_component(loudness_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(0..1 contribution, z-score) of the streamer getting louder than usual."""
    voiced = np.where(loudness_db <= SILENCE_DB + 1, np.nan, loudness_db)
    if np.all(np.isnan(voiced)):
        zeros = np.zeros(len(loudness_db), dtype=np.float32)
        return zeros, zeros
    filled = np.where(np.isnan(voiced), np.nanmedian(voiced), voiced)
    centre, spread = _block_baseline(filled)
    spread = np.maximum(spread, 1.5)  # dB: don't let a flat baseline explode z
    # A 3-second loudness envelope: one cough is not a scream.
    z = (_smooth(filled, 3) - centre) / spread
    z = np.where(np.isnan(voiced), 0.0, z).astype(np.float32)
    return np.clip((z - 1.0) / 3.0, 0.0, 1.0).astype(np.float32), z


def chat_component(chat: ChatSeries) -> tuple[np.ndarray, np.ndarray]:
    """(0..1 contribution, ratio to normal) of chat speeding up."""
    rate = _smooth(chat.rate, 10)
    centre, _ = _block_baseline(rate)
    floor = max(0.05, float(np.median(rate)) * 0.5)
    ratio = rate / np.maximum(centre, floor)
    return np.clip((ratio - 1.3) / 2.2, 0.0, 1.0).astype(np.float32), ratio.astype(np.float32)


def keyword_component(chat: ChatSeries) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(0..1 contribution, smoothed counts) of "clip"/"lmao"/"???" bursts."""
    clip = _smooth(chat.clip, 10) * 10
    hype = _smooth(chat.hype, 10) * 10
    walls = _smooth(chat.walls, 10) * 10
    msgs = np.maximum(_smooth(chat.rate, 10) * 10, 1.0)
    # Weighted keyword density per 10 s, relative to how busy chat is anyway.
    density = (3.0 * clip + hype + 2.0 * walls) / np.sqrt(msgs)
    centre, spread = _block_baseline(density)
    z = (density - centre) / np.maximum(spread, 0.25)
    strength = np.clip((z - 1.5) / 3.0, 0.0, 1.0)
    # An outright "clip it" chorus counts even when the baseline is noisy.
    strength = np.maximum(strength, np.clip((clip - 2.0) / 6.0, 0.0, 1.0))
    return strength.astype(np.float32), {"clip": clip, "hype": hype, "walls": walls}


def replayed_component(heat: np.ndarray) -> np.ndarray:
    """0..1 contribution of YouTube's "Most Replayed" graph."""
    if len(heat) == 0 or float(heat.max()) <= 0:
        return np.zeros(len(heat), dtype=np.float32)
    # The graph is relative already (1.0 = most replayed); only its upper
    # part means anything.
    return np.clip((heat - 0.35) / 0.55, 0.0, 1.0).astype(np.float32)


def clips_component(clips: np.ndarray) -> np.ndarray:
    return np.clip(clips / 3.0, 0.0, 1.0).astype(np.float32)


def build_curve(
    seconds: int,
    *,
    loudness_db: np.ndarray | None = None,
    chat: ChatSeries | None = None,
    heat: np.ndarray | None = None,
    viewer_clips: np.ndarray | None = None,
) -> HypeCurve:
    """Combine whatever signals exist into the curve and its markers."""

    def fit(values: np.ndarray) -> np.ndarray:
        out = np.zeros(seconds, dtype=np.float32)
        n = min(seconds, len(values))
        out[:n] = values[:n]
        return out

    components: dict[str, np.ndarray] = {}
    details: dict[str, np.ndarray] = {}

    if loudness_db is not None and len(loudness_db) and np.any(loudness_db > SILENCE_DB + 1):
        strength, z = audio_component(fit(loudness_db))
        components["audio"], details["audio_z"] = strength, z
    has_chat = chat is not None and float(chat.rate.sum()) >= 20
    if has_chat:
        assert chat is not None
        fitted = ChatSeries(
            rate=fit(chat.rate), clip=fit(chat.clip), hype=fit(chat.hype), walls=fit(chat.walls)
        )
        strength, ratio = chat_component(fitted)
        components["chat"], details["chat_ratio"] = strength, ratio
        kw_strength, counts = keyword_component(fitted)
        components["keywords"] = kw_strength
        details.update({f"kw_{k}": v for k, v in counts.items()})
    if heat is not None and len(heat) and float(heat.max()) > 0:
        components["replayed"] = replayed_component(fit(heat))
        details["heat"] = fit(heat)
    if viewer_clips is not None and len(viewer_clips) and float(viewer_clips.max()) > 0:
        components["clips"] = clips_component(fit(viewer_clips))
        details["clips"] = fit(viewer_clips)

    score = np.zeros(seconds, dtype=np.float32)
    total = sum(WEIGHTS[name] for name in components)
    if total > 0:
        for name, values in components.items():
            score += values * (WEIGHTS[name] / total)
        # Agreement bonus: two signals firing together is far more telling
        # than one firing hard.
        firing = sum((values > 0.35).astype(np.float32) for values in components.values())
        score = np.clip(score * (1.0 + 0.25 * np.maximum(firing - 1, 0)), 0.0, 1.0)
    score = (score * 100.0).astype(np.float32)

    markers = find_markers(score, components, details)
    return HypeCurve(
        score=score, markers=markers, available=sorted(components), components=components
    )


def find_markers(
    score: np.ndarray,
    components: dict[str, np.ndarray],
    details: dict[str, np.ndarray],
    *,
    min_score: float = MARKER_MIN_SCORE,
    spacing_s: int = MARKER_SPACING_S,
) -> list[Marker]:
    """Local maxima of the curve, strongest first claimed, then time-ordered."""
    order = np.argsort(-score, kind="stable")
    taken = np.zeros(len(score), dtype=bool)
    markers: list[Marker] = []
    for index in order:
        value = float(score[index])
        if value < min_score:
            break
        if taken[index]:
            continue
        a = max(0, index - spacing_s)
        b = min(len(score), index + spacing_s + 1)
        taken[a:b] = True
        markers.append(
            Marker(t=float(index), score=round(value, 1), reasons=_reasons(int(index), details))
        )
    markers.sort(key=lambda m: m.t)
    return markers


def _window_max(values: np.ndarray, index: int, radius: int = 6) -> float:
    a = max(0, index - radius)
    b = min(len(values), index + radius + 1)
    return float(values[a:b].max()) if b > a else 0.0


def _reasons(index: int, details: dict[str, np.ndarray]) -> list[str]:
    reasons: list[str] = []
    if "audio_z" in details:
        z = _window_max(details["audio_z"], index)
        if z >= 2.0:
            reasons.append(f"streamer much louder than normal (+{z:.1f} sigma)")
    if "chat_ratio" in details:
        ratio = _window_max(details["chat_ratio"], index)
        if ratio >= 1.8:
            reasons.append(f"chat {ratio:.1f}x faster than normal")
    for key, label in (
        ("kw_clip", '"clip"'),
        ("kw_hype", "laughing/lmao"),
        ("kw_walls", '"???" walls'),
    ):
        if key in details:
            count = _window_max(details[key], index)
            if count >= 3:
                reasons.append(f"{label} x{count:.0f} in chat")
    if "heat" in details and _window_max(details["heat"], index) >= 0.6:
        reasons.append("YouTube Most Replayed peak")
    if "clips" in details and _window_max(details["clips"], index) >= 1:
        reasons.append("viewers clipped this")
    return reasons
