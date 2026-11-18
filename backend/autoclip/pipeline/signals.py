"""Audience signals — where viewers actually watch, as one curve per second.

Ported from CrestClip's idea of starting from what viewers already
re-watched instead of only guessing what might go viral:

1. YouTube's "Most replayed" heatmap (in the yt-dlp info json saved at
   ingest): relative to the video's own peak, so it says *where*, not *how
   much*.
2. Timestamps viewers write in comments ("12:34 was insane"), weighted by
   the comment's likes.
3. Audio energy — loudness above the video's own running baseline (laughter,
   shouting). The only signal an upload has, and a weak one, so it is the
   fallback rather than an equal partner.

Chapters are kept beside the curve: they are context for the model, not a
score.

Everything here is best effort. A missing info json, a comment request that
fails, an unreadable loudness profile — each just leaves that signal out, and
the curve records which ones it had (``sources``) and what kind it is
(``kind``: "audience" when real viewer data went in, "energy" when only
loudness did, "none" otherwise).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..gta.fetch import parse_heatmap
from ..gta.hype import audio_component
from ..gta.signals import SILENCE_DB, heatmap_series

log = logging.getLogger(__name__)

#: Version of the saved curve format.
CURVE_VERSION = 1

#: File names next to a YouTube source's media.
INFO_FILE = "info.json"
COMMENTS_FILE = "comments.json"

#: Blend weights when viewer data exists (CrestClip's): the heatmap leads,
#: comments refine it, loudness only breaks ties.
HEAT_WEIGHT = 0.7
COMMENT_WEIGHT_WITH_HEAT = 0.2
COMMENT_WEIGHT_ALONE = 0.75
ENERGY_WEIGHT_WITH_HEAT = 0.1
ENERGY_WEIGHT_ALONE = 0.25

#: Every viewer starts at 0:00, so the opening of the heatmap is always
#: "replayed". The first this-fraction of the video is damped.
OPENING_FRACTION = 0.03
OPENING_FLOOR = 0.35

#: Fewer timestamp mentions than this is noise, not a signal.
MIN_COMMENT_HITS = 3

#: People type the timestamp a little before the moment; a mention is spread
#: from LEAD seconds before to TAIL seconds after it, peaking at PEAK_DELAY.
COMMENT_LEAD_S = 4
COMMENT_TAIL_S = 20
COMMENT_PEAK_DELAY_S = 4.0
COMMENT_SIGMA_S = 7.0

#: Loudness values (LUFS) at or below this count as silence.
LOUDNESS_FLOOR = -70.0

#: Hotspot detection.
HOTSPOT_MIN_THRESHOLD = 0.55
HOTSPOT_PERCENTILE = 88.0
HOTSPOT_CONTINUE = 0.85
HOTSPOT_MERGE_GAP_S = 5
MAX_HOTSPOTS = 12

#: ``12:34`` / ``1:02:03``, not part of a longer number or ratio.
_TIMESTAMP = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class Comment:
    text: str
    likes: int = 0


@dataclass
class Chapter:
    start_s: float
    end_s: float
    title: str


@dataclass
class Hotspot:
    start_s: float
    end_s: float
    peak_s: float
    #: Curve value at the peak, 0..1.
    value: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class AudienceCurve:
    """A 0..1 score per second of the source, plus its hotspots."""

    curve: np.ndarray
    kind: str = "none"  # "audience" | "energy" | "none"
    sources: list[str] = field(default_factory=list)
    hotspots: list[Hotspot] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    #: Per-signal 0..1 series, by name ("heatmap", "comments", "energy").
    components: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def seconds(self) -> int:
        return len(self.curve)

    @property
    def has_data(self) -> bool:
        return self.kind != "none" and self.seconds > 0

    @property
    def label(self) -> str:
        """What the curve is made of, for score notes and prompt hints."""
        return "+".join(self.sources) if self.sources else "none"

    def range_score(self, start_s: float, end_s: float) -> int | None:
        """How hot ``[start_s, end_s]`` is against the rest of the video, 0..100.

        The mean of the hotter half of the span's seconds, as a percentile
        of the whole curve ("more re-watched than X% of the video"). A mid
        rank is used for ties so a flat stretch never reads as a peak.
        """
        if not self.has_data:
            return None
        a = max(0, int(math.floor(start_s)))
        b = min(self.seconds, max(a + 1, int(math.ceil(end_s))))
        if a >= self.seconds:
            return None
        whole = self.curve.astype(np.float64)
        values = np.sort(whole[a:b])[::-1]
        top = values[: max(1, math.ceil(len(values) / 2))]
        v = float(top.mean())
        # float32 storage: compare with a tolerance, or a flat stretch would
        # rank above every second equal to it.
        below = float(np.count_nonzero(whole < v - 1e-6))
        equal = float(np.count_nonzero(np.abs(whole - v) <= 1e-6))
        rank = (below + 0.5 * equal) / self.seconds
        return max(0, min(100, round(rank * 100)))

    def hotspots_between(self, start_s: float, end_s: float) -> list[Hotspot]:
        """Hotspots whose peak lies in ``[start_s, end_s)``, strongest first."""
        found = [h for h in self.hotspots if start_s <= h.peak_s < end_s]
        return sorted(found, key=lambda h: -h.value)

    def chapters_between(self, start_s: float, end_s: float) -> list[Chapter]:
        return [c for c in self.chapters if start_s <= c.start_s < end_s]

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CURVE_VERSION,
            "kind": self.kind,
            "sources": self.sources,
            "curve": [round(float(v), 3) for v in self.curve],
            "hotspots": [asdict(h) for h in self.hotspots],
            "chapters": [asdict(c) for c in self.chapters],
            "components": {
                name: [round(float(v), 3) for v in series]
                for name, series in self.components.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudienceCurve:
        if int(data.get("version") or 0) != CURVE_VERSION:
            raise ValueError("unsupported audience curve version")
        return cls(
            curve=np.asarray(data.get("curve") or [], dtype=np.float32),
            kind=str(data.get("kind") or "none"),
            sources=[str(s) for s in data.get("sources") or []],
            hotspots=[Hotspot(**h) for h in data.get("hotspots") or []],
            chapters=[Chapter(**c) for c in data.get("chapters") or []],
            components={
                str(name): np.asarray(series, dtype=np.float32)
                for name, series in (data.get("components") or {}).items()
            },
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> AudienceCurve:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# Readers (never raise for bad input)
# --------------------------------------------------------------------------


def load_info(path: Path) -> dict[str, Any] | None:
    """The yt-dlp info json saved at ingest, or None when absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def parse_chapters(info: dict[str, Any] | None) -> list[Chapter]:
    out: list[Chapter] = []
    raw = (info or {}).get("chapters")
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start_time") or 0.0)
            end = float(item.get("end_time") or 0.0)
        except (TypeError, ValueError):
            continue
        title = str(item.get("title") or "").strip()
        if title and end > start >= 0:
            out.append(Chapter(start_s=start, end_s=end, title=title[:120]))
    out.sort(key=lambda c: c.start_s)
    return out


def parse_comments(raw: Any) -> list[Comment]:
    """yt-dlp ``comments`` entries (or our own cache) into :class:`Comment`."""
    out: list[Comment] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        if not text:
            continue
        try:
            likes = int(item.get("like_count", item.get("likes")) or 0)
        except (TypeError, ValueError):
            likes = 0
        out.append(Comment(text=text, likes=max(0, likes)))
    return out


def comment_timestamps(text: str, duration_s: float) -> list[int]:
    """Distinct in-video seconds a comment mentions (``12:34``, ``1:02:03``)."""
    seen: list[int] = []
    for match in _TIMESTAMP.finditer(text):
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        if seconds >= 60 or (hours and minutes >= 60):
            continue
        t = hours * 3600 + minutes * 60 + seconds
        if t < 1 or (duration_s and t >= duration_s):
            continue
        # "4:10 ... 4:12" in one comment is one moment, not two votes.
        if any(abs(t - other) < 10 for other in seen):
            continue
        seen.append(t)
    return seen


def loudness_per_second(values: list[float], step_s: float) -> np.ndarray:
    """Momentary loudness (one value per ``step_s``) as one dB value per second.

    Each second is the energy mean of its frames; seconds with no voiced
    frame are reported as :data:`SILENCE_DB`, which
    :func:`~autoclip.gta.hype.audio_component` treats as "no signal".
    """
    if not values or step_s <= 0:
        return np.zeros(0, dtype=np.float32)
    per = max(1, round(1.0 / step_s))
    arr = np.asarray(values, dtype=np.float64)
    seconds = math.ceil(len(arr) / per)
    padded = np.full(seconds * per, np.nan)
    padded[: len(arr)] = arr
    padded = np.where(padded <= LOUDNESS_FLOOR, np.nan, padded)
    blocks = padded.reshape(seconds, per)
    out = np.full(seconds, SILENCE_DB, dtype=np.float64)
    voiced = ~np.all(np.isnan(blocks), axis=1)
    if voiced.any():
        power = np.nanmean(10.0 ** (blocks[voiced] / 10.0), axis=1)
        out[voiced] = 10.0 * np.log10(np.maximum(power, 1e-12))
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def _normalise(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float32)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.zeros(len(values), dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def heat_component(heatmap: list[tuple[float, float, float]], seconds: int) -> np.ndarray | None:
    """The "Most replayed" graph per second, opening damped, 0..1."""
    if not heatmap or seconds <= 0:
        return None
    heat = heatmap_series(heatmap, seconds).astype(np.float64)
    if float(heat.max()) <= 0:
        return None
    opening = max(1, int(seconds * OPENING_FRACTION))
    ramp = OPENING_FLOOR + (1.0 - OPENING_FLOOR) * (np.arange(opening) / opening)
    heat[:opening] *= ramp
    return _normalise(heat)


def comment_component(
    comments: list[Comment], seconds: int
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Like-weighted timestamp mentions per second, 0..1, plus the raw hits.

    Returns None when fewer than :data:`MIN_COMMENT_HITS` timestamps were
    found — a couple of mentions say more about one commenter than the
    audience.
    """
    if not comments or seconds <= 0:
        return None
    series = np.zeros(seconds, dtype=np.float64)
    hits: list[tuple[int, int]] = []
    offsets = np.arange(-COMMENT_LEAD_S, COMMENT_TAIL_S + 1)
    kernel = np.exp(-((offsets - COMMENT_PEAK_DELAY_S) ** 2) / (2 * COMMENT_SIGMA_S**2))
    for comment in comments:
        weight = 1.0 + math.log10(1 + comment.likes)
        for t in comment_timestamps(comment.text, seconds):
            hits.append((t, comment.likes))
            for offset, k in zip(offsets, kernel, strict=True):
                i = t + int(offset)
                if 0 <= i < seconds:
                    series[i] += weight * k
    if len(hits) < MIN_COMMENT_HITS:
        return None
    return _normalise(series), hits


def energy_component(loudness_db: np.ndarray) -> np.ndarray | None:
    """Loudness above the video's own running baseline, squashed to 0..1."""
    if len(loudness_db) == 0 or not np.any(loudness_db > SILENCE_DB + 1):
        return None
    _, z = audio_component(loudness_db.astype(np.float32))
    squashed = 1.0 / (1.0 + np.exp(-1.4 * z.astype(np.float64)))
    squashed = np.where(loudness_db <= SILENCE_DB + 1, 0.0, squashed)
    # Light smoothing: one shout is a second, a moment is a few.
    if len(squashed) >= 3:
        padded = np.pad(squashed, 1, mode="edge")
        squashed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    return squashed.astype(np.float32)


def _fit(series: np.ndarray, seconds: int) -> np.ndarray:
    out = np.zeros(seconds, dtype=np.float32)
    n = min(seconds, len(series))
    out[:n] = series[:n]
    return out


# --------------------------------------------------------------------------
# Curve
# --------------------------------------------------------------------------


def build_curve(
    duration_s: float,
    *,
    heatmap: list[tuple[float, float, float]] | None = None,
    comments: list[Comment] | None = None,
    loudness_db: np.ndarray | None = None,
    chapters: list[Chapter] | None = None,
) -> AudienceCurve:
    """Combine whatever signals exist into one per-second curve."""
    seconds = max(0, int(math.ceil(duration_s)))
    if seconds == 0 and loudness_db is not None:
        seconds = len(loudness_db)
    components: dict[str, np.ndarray] = {}

    heat = heat_component(heatmap or [], seconds)
    if heat is not None:
        components["heatmap"] = heat
    comment_result = comment_component(comments or [], seconds)
    hits: list[tuple[int, int]] = []
    if comment_result is not None:
        components["comments"], hits = comment_result
    energy = energy_component(loudness_db) if loudness_db is not None else None
    if energy is not None:
        components["energy"] = _fit(energy, seconds)

    parts: list[tuple[np.ndarray, float]] = []
    if heat is not None:
        parts.append((heat, HEAT_WEIGHT))
    if "comments" in components:
        weight = COMMENT_WEIGHT_WITH_HEAT if heat is not None else COMMENT_WEIGHT_ALONE
        parts.append((components["comments"], weight))

    if parts:
        kind = "audience"
        if "energy" in components:
            weight = ENERGY_WEIGHT_WITH_HEAT if heat is not None else ENERGY_WEIGHT_ALONE
            parts.append((components["energy"], weight))
        total = sum(w for _, w in parts)
        curve = _normalise(sum(series.astype(np.float64) * w for series, w in parts) / total)
    elif "energy" in components:
        kind = "energy"
        curve = components["energy"]
    else:
        kind = "none"
        curve = np.zeros(seconds, dtype=np.float32)

    result = AudienceCurve(
        curve=curve.astype(np.float32),
        kind=kind,
        sources=list(components),
        chapters=list(chapters or []),
        components=components,
    )
    if kind != "none":
        result.hotspots = find_hotspots(result, hits)
    return result


def find_hotspots(curve: AudienceCurve, hits: list[tuple[int, int]] | None = None) -> list[Hotspot]:
    """Contiguous hot stretches of the curve, strongest first."""
    values = curve.curve.astype(np.float64)
    n = len(values)
    if n == 0 or float(values.max()) <= 0:
        return []
    if n >= 5:
        padded = np.pad(values, 2, mode="edge")
        values = np.convolve(padded, np.ones(5) / 5, mode="valid")
    threshold = max(HOTSPOT_MIN_THRESHOLD, float(np.percentile(values, HOTSPOT_PERCENTILE)))

    runs: list[list[int]] = []
    i = 0
    while i < n:
        if values[i] >= threshold:
            j = i
            while j + 1 < n and values[j + 1] >= threshold * HOTSPOT_CONTINUE:
                j += 1
            if runs and i - runs[-1][1] <= HOTSPOT_MERGE_GAP_S:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1

    spots: list[Hotspot] = []
    for start, end in runs:
        peak = start + int(np.argmax(values[start : end + 1]))
        spots.append(
            Hotspot(
                start_s=float(start),
                end_s=float(end + 1),
                peak_s=float(peak),
                value=round(float(values[peak]), 3),
                reasons=_reasons(curve, start, end, peak, hits or []),
            )
        )
    spots.sort(key=lambda h: -h.value)
    return spots[:MAX_HOTSPOTS]


def _reasons(
    curve: AudienceCurve, start: int, end: int, peak: int, hits: list[tuple[int, int]]
) -> list[str]:
    reasons: list[str] = []
    heat = curve.components.get("heatmap")
    if heat is not None and len(heat) > peak:
        level = float(heat[start : end + 1].max())
        if level >= 0.5:
            reasons.append(f"most replayed ({round(level * 100)}% of the video's peak)")
    near = [likes for t, likes in hits if start - COMMENT_TAIL_S <= t <= end + COMMENT_LEAD_S]
    if near:
        likes = sum(near)
        plural = "s" if len(near) != 1 else ""
        reasons.append(f"{len(near)} comment timestamp{plural} ({likes} likes)")
    energy = curve.components.get("energy")
    if energy is not None and len(energy) > peak and float(energy[start : end + 1].max()) >= 0.85:
        reasons.append("loud reaction in the audio")
    return reasons or ["above the video's usual level"]


# --------------------------------------------------------------------------
# Comments (network, best effort)
# --------------------------------------------------------------------------


def fetch_comments(url: str, *, max_comments: int, cookies_from_browser: str = "") -> list[Comment]:
    """Top comments of a YouTube video via yt-dlp. Raises on failure."""
    import yt_dlp

    from .ingest import js_runtime_options

    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "noplaylist": True,
        "getcomments": True,
        "socket_timeout": 30,
        "extractor_args": {
            "youtube": {
                "comment_sort": ["top"],
                # total, top-level threads, replies: no replies needed.
                "max_comments": [str(max_comments), "all", "0"],
            }
        },
        **js_runtime_options(),
    }
    if cookies_from_browser:
        options["cookiesfrombrowser"] = (cookies_from_browser,)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    return parse_comments(info.get("comments"))


CommentFetcher = Callable[[str], list[Comment]]


def load_or_fetch_comments(cache: Path, url: str, fetch: CommentFetcher) -> list[Comment]:
    """Cached comments for a source, fetching them once. Never raises.

    A failed fetch is not cached, so the next job on the same source tries
    again.
    """
    if cache.exists():
        try:
            return parse_comments(json.loads(cache.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            log.warning("Ignoring an unreadable comment cache at %s.", cache)
    try:
        comments = fetch(url)
    except Exception as exc:
        log.warning("Could not read comments for %s (%s); going without them.", url, exc)
        return []
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps([{"text": c.text, "like_count": c.likes} for c in comments]),
            encoding="utf-8",
        )
    except OSError:
        log.debug("Could not cache comments at %s.", cache)
    return comments


def build_for_source(
    *,
    source_dir: Path,
    url: str | None,
    is_youtube: bool,
    duration_s: float,
    loudness_values: list[float] | None = None,
    loudness_step_s: float = 0.1,
    use_comments: bool = True,
    use_energy: bool = True,
    fetch: CommentFetcher | None = None,
) -> AudienceCurve:
    """Gather every available signal for one source and build its curve.

    ``source_dir`` holds the source's media and its ingest sidecars
    (``info.json``; ``comments.json`` is cached there too). ``fetch`` is only
    called for YouTube sources with a URL, when ``use_comments`` is set.
    """
    info = load_info(source_dir / INFO_FILE) if is_youtube else None
    heatmap = parse_heatmap((info or {}).get("heatmap"))
    chapters = parse_chapters(info)
    if not duration_s and info:
        try:
            duration_s = float(info.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration_s = 0.0

    comments: list[Comment] = []
    if use_comments and is_youtube and url and fetch is not None:
        comments = load_or_fetch_comments(source_dir / COMMENTS_FILE, url, fetch)

    loudness_db = None
    if use_energy and loudness_values:
        loudness_db = loudness_per_second(loudness_values, loudness_step_s)

    curve = build_curve(
        duration_s,
        heatmap=heatmap,
        comments=comments,
        loudness_db=loudness_db,
        chapters=chapters,
    )
    log.info(
        "Audience curve: %s from %s, %d hotspot(s), %d chapter(s).",
        curve.kind,
        curve.label,
        len(curve.hotspots),
        len(curve.chapters),
    )
    return curve


def clock(seconds: float) -> str:
    s = max(0, round(seconds))
    h, rem = divmod(s, 3600)
    m, r = divmod(rem, 60)
    return f"{h}:{m:02d}:{r:02d}" if h else f"{m}:{r:02d}"
