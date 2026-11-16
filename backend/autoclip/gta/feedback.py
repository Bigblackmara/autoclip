""" "Look at what's working": learn from the channel's real view counts.

Reads the posted Shorts of a channel (title + views, via yt-dlp, no login),
works out which moment types and which streamers beat the channel's average,
and stores that as a bias the judge uses from then on.

Each Short is matched to the GTA clip it came from when its title matches a
clip's hook title or caption; Shorts that match nothing are classified by the
AI from their title alone.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, field_validator

from ..db.models import utcnow
from ..providers.base import DetectionConfig
from . import store
from .judging import MOMENT_TYPES, Complete, _ask, _clean_type
from .settings import GtaBias
from .settings import load as load_settings
from .settings import save as save_settings

log = logging.getLogger(__name__)

#: A group needs at least this many Shorts before it may steer selection.
MIN_GROUP = 3
#: Most Shorts read from a channel.
MAX_SHORTS = 300


@dataclass
class PostedShort:
    title: str
    views: int
    url: str = ""
    moment_type: str = ""
    streamer: str = ""
    matched_clip: str = ""


def shorts_url(channel_url: str) -> str:
    url = channel_url.strip().rstrip("/")
    if re.search(r"youtube\.com/(@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)$", url):
        return f"{url}/shorts"
    return url


def list_shorts(channel_url: str, *, limit: int = MAX_SHORTS) -> list[PostedShort]:
    import yt_dlp

    from .fetch import _translate, _ydl_base_options

    options = {
        **_ydl_base_options(),
        "extract_flat": "in_playlist",
        "playlistend": limit,
        "noplaylist": False,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(shorts_url(channel_url), download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise _translate(exc) from exc
    entries = (info or {}).get("entries") or []
    out: list[PostedShort] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        views = entry.get("view_count")
        if views is None:
            continue
        out.append(
            PostedShort(
                title=str(entry.get("title") or ""),
                views=int(views),
                url=str(entry.get("url") or entry.get("webpage_url") or ""),
            )
        )
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def match_to_clips(
    shorts: list[PostedShort], clips: list[store.GtaClip], streams: dict[str, str]
) -> None:
    """Fill moment_type/streamer from the GTA clip each Short came from."""
    for short in shorts:
        title = _norm(short.title)
        if not title:
            continue
        best: tuple[float, store.GtaClip | None] = (0.0, None)
        for clip in clips:
            for text in (clip.hook_title, clip.caption):
                candidate = _norm(text)
                if not candidate:
                    continue
                if candidate in title or title in candidate:
                    ratio = 1.0
                else:
                    ratio = difflib.SequenceMatcher(None, title, candidate).ratio()
                if ratio > best[0]:
                    best = (ratio, clip)
        ratio, clip = best
        if clip is not None and ratio >= 0.72:
            short.moment_type = clip.moment_type or "other"
            short.streamer = streams.get(clip.stream_id, "")
            short.matched_clip = clip.id


class _Label(BaseModel):
    index: int
    moment_type: str = "other"
    streamer: str = ""

    @field_validator("moment_type", "streamer", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value)


class _Labels(BaseModel):
    labels: list[_Label] = []


CLASSIFY_SYSTEM = (
    "You label posted YouTube Shorts that were clipped from livestreams. For each title, "
    "give the type of moment and the streamer's name if the title or common knowledge makes "
    "it clear (empty string if unknown). Moment types: "
    + ", ".join(MOMENT_TYPES)
    + '. Respond with ONLY JSON: {"labels": [{"index": <n>, "moment_type": "...", '
    + '"streamer": "..."}]}'
)


async def classify(shorts: list[PostedShort], complete: Complete) -> None:
    pending = [(i, s) for i, s in enumerate(shorts) if not s.moment_type]
    config = DetectionConfig(prompt_version="gta_feedback", temperature=0.1)
    for offset in range(0, len(pending), 60):
        batch = pending[offset : offset + 60]
        user = "\n".join(f"{i}. {s.title}" for i, s in batch)
        parsed = await _ask(complete, CLASSIFY_SYSTEM, user, config, _Labels)
        assert isinstance(parsed, _Labels)
        by_index = {label.index: label for label in parsed.labels}
        for i, short in batch:
            label = by_index.get(i)
            short.moment_type = _clean_type(label.moment_type) if label else "other"
            short.streamer = (label.streamer.strip() if label else "") or short.streamer


def analyse(shorts: list[PostedShort]) -> dict[str, Any]:
    """Average views per moment type and per streamer vs the channel average."""
    if not shorts:
        return {"channel_average": 0, "count": 0, "moment_types": {}, "streamers": {}}
    average = sum(s.views for s in shorts) / len(shorts)

    def groups(key: str) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[int]] = {}
        for short in shorts:
            name = getattr(short, key) or ""
            if not name:
                continue
            buckets.setdefault(name, []).append(short.views)
        out: dict[str, dict[str, float]] = {}
        for name, views in buckets.items():
            mean = sum(views) / len(views)
            out[name] = {
                "count": len(views),
                "average_views": round(mean),
                "vs_channel": round(mean / average, 2) if average else 0.0,
            }
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["vs_channel"]))

    return {
        "channel_average": round(average),
        "count": len(shorts),
        "moment_types": groups("moment_type"),
        "streamers": groups("streamer"),
    }


def bias_from(analysis: dict[str, Any], channel_url: str) -> GtaBias:
    moment_types = {
        name: float(data["vs_channel"])
        for name, data in analysis["moment_types"].items()
        if data["count"] >= MIN_GROUP
    }
    streamers = {
        name: float(data["vs_channel"])
        for name, data in analysis["streamers"].items()
        if data["count"] >= MIN_GROUP
    }
    lines: list[str] = []
    winners = [n for n, m in moment_types.items() if m > 1.15]
    losers = [n for n, m in moment_types.items() if m < 0.85]
    if winners:
        lines.append(
            "Beating the channel average: "
            + ", ".join(f"{n.replace('_', ' ')} ({moment_types[n]:.1f}x)" for n in winners)
        )
    if losers:
        lines.append(
            "Under the average: "
            + ", ".join(f"{n.replace('_', ' ')} ({moment_types[n]:.1f}x)" for n in losers)
        )
    top_streamers = [n for n, m in streamers.items() if m > 1.15]
    if top_streamers:
        lines.append(
            "Streamers beating the average: "
            + ", ".join(f"{n} ({streamers[n]:.1f}x)" for n in top_streamers)
        )
    return GtaBias(
        moment_types=moment_types,
        streamers=streamers,
        summary="\n".join(lines),
        updated_at=utcnow(),
        channel_url=channel_url,
    )


def look_at_whats_working(channel_url: str, *, complete: Complete | None = None) -> dict[str, Any]:
    """Run the whole feedback loop; returns the stored report."""
    shorts = list_shorts(channel_url)
    if not shorts:
        raise ValueError("No posted Shorts with view counts were found on that channel.")
    clips = store.list_clips(statuses=frozenset({"ready"}))
    streams = {s.id: s.channel for s in store.list_streams(limit=10_000)}
    match_to_clips(shorts, clips, streams)
    if any(not s.moment_type for s in shorts):
        if complete is None:
            from .pipeline import _provider_complete

            complete = _provider_complete()
        asyncio.run(classify(shorts, complete))
    analysis = analyse(shorts)
    bias = bias_from(analysis, channel_url)
    settings = load_settings()
    settings.bias = bias
    save_settings(settings)
    report = {
        **analysis,
        "bias": bias.model_dump(),
        "shorts": [asdict(s) for s in sorted(shorts, key=lambda s: -s.views)[:50]],
        "matched": sum(1 for s in shorts if s.matched_clip),
    }
    store.create_report(channel_url, report)
    return report
