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
    #: Also judge what was SAID every this many seconds, even without a loud
    #: spike (something important, a reveal, a story worth watching). 0 = off.
    judge_every_s: float = 180.0


#: GTA caption looks -> the pipeline.captions preset each one renders with.
#: ``karaoke_fill`` is kept for settings saved before the looks were added.
CAPTION_STYLES = {
    "gta": "bold_pop",
    "pop": "pop",
    "neon": "neon",
    "comic": "comic",
    "minimal": "minimal",
    "karaoke_fill": "karaoke_fill",
}
#: Older saves stored the preset name of the GTA look.
CAPTION_ALIASES = {"bold_pop": "gta", "": "gta"}
MOTION_LEVELS = ("off", "subtle", "dynamic")
#: The order "Make variant" steps through caption looks.
VARIANT_LOOKS = ("gta", "pop", "neon", "comic", "minimal")
#: Caption language modes (see :mod:`autoclip.pipeline.language`).
CAPTION_LANGUAGES = ("as_spoken", "roman", "english")

#: YouTube video categories offered for GTA clips (id -> name).
YOUTUBE_CATEGORIES = {"20": "Gaming", "22": "People & Blogs", "23": "Comedy", "24": "Entertainment"}


class GtaPosting(BaseModel):
    """Posting schedule (YouTube, Instagram via Zernio) for finished GTA clips.

    Shortly before each daily slot (``lead_minutes``) the best-scored clip
    that passed QA is uploaded as private with ``publishAt`` set to the slot,
    so YouTube itself makes it public exactly then, even if AutoClip is
    closed by that time. Off until the user switches it on.
    """

    enabled: bool = False
    #: Local times of day ("HH:MM", 24 h) a Short goes public.
    slots: list[str] = Field(default_factory=lambda: ["12:00", "17:00", "21:00"])
    #: How long before its slot a clip is uploaded, minutes.
    lead_minutes: int = 120
    category_id: str = "24"
    #: Append #Shorts to the description when the clip's hashtags lack it.
    shorts_tag: bool = True
    #: Which platforms the schedule posts to (``enabled`` is the master switch).
    youtube: bool = True
    #: Instagram Reels through Zernio (zernio.com), scheduled for the same slot.
    instagram: bool = False
    #: Instagram username to post as when several are connected at Zernio.
    instagram_account: str = ""

    @field_validator("instagram_account")
    @classmethod
    def _account(cls, value: str) -> str:
        text = str(value).strip().lstrip("@").strip()
        if len(text) > 64 or any(ch.isspace() for ch in text):
            raise ValueError("Instagram account must be a username like clipgod (no spaces)")
        return text

    @field_validator("slots")
    @classmethod
    def _slots(cls, value: list[str]) -> list[str]:
        cleaned: set[str] = set()
        for raw in value:
            text = str(raw).strip()
            parts = text.split(":")
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                raise ValueError(f"Slot {text!r} is not a time like 17:00")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(f"Slot {text!r} is not a valid time of day")
            cleaned.add(f"{hour:02d}:{minute:02d}")
        if not 1 <= len(cleaned) <= 24:
            raise ValueError("Give between 1 and 24 posting times")
        return sorted(cleaned)

    @field_validator("lead_minutes")
    @classmethod
    def _lead(cls, value: int) -> int:
        if not 15 <= value <= 1440:
            raise ValueError("lead_minutes must be between 15 and 1440")
        return value

    @field_validator("category_id")
    @classmethod
    def _category(cls, value: str) -> str:
        if value not in YOUTUBE_CATEGORIES:
            raise ValueError(f"category_id must be one of {', '.join(YOUTUBE_CATEGORIES)}")
        return value


class GtaSettings(BaseModel):
    #: Transcript chunk the judge reads at once, minutes ("exactly 12").
    chunk_minutes: float = 12.0
    #: Read-only spill-over past each chunk so a payoff that lands just after
    #: the seam can still be judged. Clips must START inside their chunk.
    chunk_tail_s: float = 60.0

    #: Final clip length AFTER dead air is cut, seconds: aim for
    #: ``min_clip_s``-``target_clip_s`` (15-30 s); ``max_clip_s`` is the hard
    #: cap, used only when a payoff needs it, and stays under 60 s.
    min_clip_s: float = 15.0
    target_clip_s: float = 30.0
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
    #: Caption look (see :data:`CAPTION_STYLES`): ``gta`` is the classic GTA
    #: look (white, black outline, spoken word yellow); ``pop`` / ``neon`` /
    #: ``comic`` / ``minimal`` reuse the normal pipeline's presets. Every one
    #: stays in the bottom band, above the bottom safe zone, in 1-2 lines.
    caption_style: str = "gta"
    #: Draw 3-8 key words per clip bigger and in the style's accent colour
    #: (at most one per caption line).
    keyword_highlight: bool = True
    #: Subtle step zooms through the clip on the stream frame only (the
    #: blurred background and the text bands never move), on top of the
    #: punch-in: ``off`` / ``subtle`` (~every 4.5 s) / ``dynamic`` (~2.2 s).
    motion: str = "subtle"
    #: "fit" = whole stream frame over a blurred fill; "split" = the streamer's
    #: webcam across the top, gameplay zoomed to fill the bottom (falls back to
    #: "fit" when no webcam is found).
    layout: str = "fit"

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

    #: YouTube VODs: timestamps viewers wrote in the top comments ("2:13:05
    #: was insane") join the Hype Curve. One extra request, cached per
    #: stream; a failure just leaves the signal out.
    comment_signal: bool = True
    #: YouTube VODs: use YouTube's own captions as the transcript where they
    #: look right, skipping Whisper for those parts (Whisper on any doubt).
    #: Off by default: automatic captions have no punctuation.
    youtube_captions: bool = False

    #: Own-value editing: a punch-in zoom on the clip's key moment (its
    #: loudest beat), with the judge's short reaction text in the top band.
    punch_in: bool = True
    punch_in_zoom: float = 1.3
    punch_in_s: float = 1.5
    reaction_text: bool = True

    #: Flash-forward: open on the clip's peak reaction (the line the judge
    #: quoted, else the Hype Curve / loudest beat), punched in, for
    #: ``flash_forward_s`` seconds, then play the clip from its start. Each
    #: clip can override it. Off by default.
    flash_forward: bool = False
    flash_forward_s: float = 2.0
    #: A static white caption box with black text (the hook) on screen for
    #: the whole clip, near the bottom just above the platform's bottom bar
    #: (the subtitles move up to sit above it).
    static_caption: bool = True
    #: An AI narrator: a line of context at the start and a question at the
    #: end, written by the AI and voiced locally (Kokoro), with the stream
    #: audio ducked under it. Original commentary for YouTube's reused-content
    #: rule (Partner Program).
    voiceover: bool = True
    #: Narrator voice: a Microsoft neural voice (edge-tts, free; e.g.
    #: en-US-ChristopherNeural, en-US-GuyNeural, en-US-AndrewNeural) or an
    #: offline Kokoro voice (af_heart, am_michael …).
    voiceover_voice: str = "en-US-ChristopherNeural"
    voiceover_speed: float = 1.05
    #: Stream audio level while the narrator speaks, dB.
    voiceover_duck_db: float = -11.0
    #: Also close with a narrator question (clips of 14 s or more).
    voiceover_outro: bool = True
    #: Language/script of the burned captions, the hook title and the post
    #: text: ``as_spoken`` (Hindi as Hinglish, everything else as spoken),
    #: ``roman`` (Latin letters) or ``english`` (translated by the AI).
    caption_language: str = "as_spoken"

    #: Your logo (PNG, ideally with transparency) burned into a top corner.
    #: When set it wins over the text logo below.
    watermark_path: str = ""
    #: Text logo used when no PNG is set: drawn in ``watermark_font`` (empty =
    #: bundled Lemon Milk Bold), first word yellow, rest white, black outline.
    #: Both empty = no watermark (QA then warns that it is missing).
    watermark_text: str = "GTA Clipgod"
    watermark_font: str = ""
    watermark_corner: str = "top_right"
    watermark_width_px: int = 240
    watermark_opacity: float = 0.85

    live: GtaLiveSettings = Field(default_factory=GtaLiveSettings)
    posting: GtaPosting = Field(default_factory=GtaPosting)
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

    @field_validator("caption_style", mode="before")
    @classmethod
    def _caption_style(cls, value: object) -> str:
        key = str(value or "").strip().lower()
        key = CAPTION_ALIASES.get(key, key)
        if key not in CAPTION_STYLES:
            raise ValueError(f"caption_style must be one of {', '.join(CAPTION_STYLES)}")
        return key

    @field_validator("motion")
    @classmethod
    def _motion(cls, value: str) -> str:
        if value not in MOTION_LEVELS:
            raise ValueError(f"motion must be one of {', '.join(MOTION_LEVELS)}")
        return value

    @field_validator("flash_forward_s")
    @classmethod
    def _tease_length(cls, value: float) -> float:
        if not 1.0 <= value <= 3.0:
            raise ValueError("flash_forward_s must be between 1 and 3 seconds")
        return value

    @field_validator("caption_language")
    @classmethod
    def _language(cls, value: str) -> str:
        if value not in CAPTION_LANGUAGES:
            raise ValueError(f"caption_language must be one of {', '.join(CAPTION_LANGUAGES)}")
        return value

    @field_validator("watermark_corner")
    @classmethod
    def _corner(cls, value: str) -> str:
        # Only the top corners: the bottom band holds the captions, and the
        # bottom ~300 px are covered by the app UI.
        if value not in {"top_left", "top_right"}:
            raise ValueError("watermark_corner must be top_left or top_right")
        return value

    @field_validator("punch_in_zoom")
    @classmethod
    def _zoom(cls, value: float) -> float:
        if not 1.05 <= value <= 2.0:
            raise ValueError("punch_in_zoom must be between 1.05 and 2.0")
        return value

    @field_validator("watermark_opacity")
    @classmethod
    def _opacity(cls, value: float) -> float:
        if not 0.1 <= value <= 1.0:
            raise ValueError("watermark_opacity must be between 0.1 and 1.0")
        return value

    @field_validator("watermark_text")
    @classmethod
    def _logo_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) > 40:
            raise ValueError("watermark_text must be at most 40 characters")
        return value

    @field_validator("watermark_width_px")
    @classmethod
    def _logo_width(cls, value: int) -> int:
        if not 40 <= value <= 400:
            raise ValueError("watermark_width_px must be between 40 and 400")
        return value

    @model_validator(mode="after")
    def _ordered_lengths(self) -> GtaSettings:
        if self.min_clip_s >= self.max_clip_s:
            raise ValueError("min_clip_s must be shorter than max_clip_s")
        if not self.min_clip_s <= self.target_clip_s <= self.max_clip_s:
            raise ValueError("target_clip_s must lie between min_clip_s and max_clip_s")
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
