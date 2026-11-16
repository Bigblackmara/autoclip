"""GTA Mode settings (persisted in config.json under ``gta``).

Kept apart from the normal pipeline's settings on purpose: GTA Mode has its
own clip lengths, judging and layout, and changing one mode must never move
the other. A missing or invalid section falls back to the defaults below
instead of failing startup, exactly like the ``live`` section.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

log = logging.getLogger(__name__)

#: Section key in config.json.
SECTION = "gta"

_write_lock = threading.Lock()


class GtaBias(BaseModel):
    """What the feedback loop learned works, fed back into judging.

    Multipliers are relative to the channel average (1.0 = average). Only
    groups with enough posted Shorts behind them are kept, so a single lucky
    upload can't steer selection.
    """

    moment_types: dict[str, float] = Field(default_factory=dict)
    streamers: dict[str, float] = Field(default_factory=dict)
    summary: str = ""
    updated_at: str = ""
    channel_url: str = ""


class GtaLiveSettings(BaseModel):
    #: How often the live watcher re-evaluates the stream, seconds.
    listen_interval_s: float = 20.0
    #: How much of the stream the judge reads when the chat explodes or the
    #: streamer loses it, seconds.
    context_s: float = 60.0
    #: The clip must be finished within this long of the moment, seconds.
    turnaround_s: float = 180.0


class GtaSettings(BaseModel):
    #: Transcript chunk the judge reads at once, minutes ("exactly 12").
    chunk_minutes: float = 12.0
    #: Read-only spill-over past each chunk so a payoff that lands just after
    #: the seam can still be judged. Clips must START inside their chunk.
    chunk_tail_s: float = 60.0

    #: Final clip length bounds AFTER dead air is cut, seconds. The upper
    #: bound stays under 60 s (Shorts/Reels/TikTok).
    min_clip_s: float = 12.0
    max_clip_s: float = 58.0
    #: Most clips kept per stream after the final judging round.
    max_clips_per_stream: int = 12

    #: Seconds of extra video downloaded either side of a clip section, so the
    #: frame-accurate cut never lands outside the downloaded file.
    section_pad_s: float = 6.0

    #: A silence at least this long inside a clip is dead air and is cut.
    dead_air_min_s: float = 0.6
    #: Silence left on each side of a cut so speech never sounds clipped.
    dead_air_keep_s: float = 0.12

    #: On-screen hook title duration, seconds.
    hook_title_s: float = 3.0
    #: Caption preset (see pipeline.captions.STYLES).
    caption_style: str = "bold_pop"

    #: Loudness target and the QA tolerance around it.
    loudness_lufs: float = -14.0
    loudness_tolerance_lu: float = 1.0
    true_peak_db: float = -1.0

    #: Share of sampled frames that must show a face for the face-cam check.
    face_min_ratio: float = 0.5

    #: yt-dlp format for the analysis download (audio only where possible).
    analysis_format: str = "bestaudio/worst"
    #: yt-dlp format for the clip sections that get rendered.
    clip_format: str = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"

    #: Concurrency. Network work (downloads, chat, the AI judge) runs in
    #: parallel; heavy work (Whisper, rendering) one at a time so VRAM never
    #: overflows — slower, never broken.
    network_workers: int = 3
    heavy_workers: int = 1

    #: Where finished clips go. Empty = ``<AutoClip home>/exports/gta``.
    output_dir: str = ""

    live: GtaLiveSettings = Field(default_factory=GtaLiveSettings)
    bias: GtaBias = Field(default_factory=GtaBias)

    @field_validator("chunk_minutes")
    @classmethod
    def _positive_chunk(cls, value: float) -> float:
        if value < 1.0:
            raise ValueError("chunk_minutes must be at least 1")
        return value

    @field_validator("max_clip_s")
    @classmethod
    def _under_a_minute(cls, value: float) -> float:
        if not 5.0 <= value < 60.0:
            raise ValueError("max_clip_s must be between 5 and 59.9 seconds")
        return value

    @field_validator("network_workers", "heavy_workers", "max_clips_per_stream")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return value

    @model_validator(mode="after")
    def _ordered_lengths(self) -> GtaSettings:
        if self.min_clip_s >= self.max_clip_s:
            raise ValueError("min_clip_s must be shorter than max_clip_s")
        return self


def _config_path():
    from .. import paths

    return paths.config_path()


def _read_raw() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def parse_section(raw: Any) -> GtaSettings:
    """Validate a raw ``gta`` value; anything unusable lands on the defaults."""
    if not isinstance(raw, dict):
        return GtaSettings()
    try:
        return GtaSettings.model_validate(raw)
    except Exception:
        log.warning("gta settings failed validation; using defaults.", exc_info=True)
        return GtaSettings()


def load() -> GtaSettings:
    return parse_section(_read_raw().get(SECTION))


def save(settings: GtaSettings) -> None:
    """Write the ``gta`` section back, leaving every other key untouched."""
    with _write_lock:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = _read_raw()
        raw[SECTION] = settings.model_dump(mode="json")
        # Own temp name: Live Watch's writer uses ``.json.tmp`` and the two
        # must never write over each other's half-finished file.
        tmp = path.with_suffix(".gta.tmp")
        tmp.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):  # platform dependent
            tmp.chmod(0o600)
        os.replace(tmp, path)


def output_root():
    """Directory finished GTA clips are written to."""
    from pathlib import Path

    from .. import paths

    settings = load()
    if settings.output_dir.strip():
        return Path(settings.output_dir).expanduser()
    return paths.exports_dir() / "gta"


def work_root():
    """Per-stream working files (analysis audio, chat, transcript, sections)."""
    from .. import paths

    return paths.root() / "gta"
