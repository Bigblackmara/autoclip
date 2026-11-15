"""Live Watch settings (persisted in config.json under ``live``).

Frozen defaults; A4 owns the bodies of detection-related subsections and may add
fields to LiveDetectionSettings, but must keep every existing field and default
so other workers' code keeps compiling against the skeleton as written.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class LiveRollingBufferSettings(BaseModel):
    #: Target seconds per .ts segment (capture currently always asks ffmpeg
    #: for buffer.SEGMENT_S). Real segments are keyframe-cut and vary; the
    #: buffer times clips from their real lengths, never from this.
    segment_s: float = 6.0
    #: How much of the stream to keep on disk, in minutes.
    retain_minutes: float = 5.0
    #: Hard disk cap for ONE stream's buffer, megabytes.
    max_disk_mb: int = 512
    #: Seconds captured before the trigger moment (pre-roll).
    pre_roll_s: float = 20.0
    #: Seconds captured after the moment ends (post-roll).
    post_roll_s: float = 10.0
    #: How much stream time the LLM judge gets to READ around a trigger —
    #: a wider window gives the judge setup/context for a better verdict.
    #: This is independent of pre_roll/post_roll (what gets RENDERED):
    #: the judge window is transcription-only input; the final clip is
    #: re-extracted from clip_start_s/clip_end_s at render time.
    judge_context_s: float = 240.0


class LiveRateLimitSettings(BaseModel):
    #: Max candidate windows sent to the LLM judge per stream per minute.
    judge_per_minute: int = 3
    #: Hard cap on judge calls per stream per hour (cost control).
    judge_per_hour: int = 60
    #: LLM gate threshold: candidates scoring below this are dropped silently.
    judge_min_score: int = 70
    #: Cooldown after a clipped moment before THAT stream can trigger again.
    cooldown_s: float = 90.0
    #: Hard cap on judge calls per stream per day (strict cost control).
    judge_per_day: int = 500


class LiveAudioSettings(BaseModel):
    #: Seconds of recent audio the RMS window uses as its baseline.
    baseline_window_s: float = 30.0
    #: RMS z-score a window must exceed to count as a spike.
    spike_z: float = 2.5
    #: Minimum absolute RMS (0–1) — protects against loudness war baselines.
    floor_rms: float = 0.02
    #: Max consecutive spike windows to count as sustained excitement.
    max_sustained_s: float = 30.0


class LiveChatSettings(BaseModel):
    #: Enable Twitch chat reading (the only chat source in v1).
    enabled: bool = True
    #: Seconds of chat history the rate baseline uses.
    baseline_window_s: float = 60.0
    #: Message-rate z-score a window needs to count as a chat spike.
    rate_z: float = 3.0
    #: Minimum messages/minute below which chat is "quiet" regardless of z.
    floor_msgs_per_min: float = 10.0
    #: EMOTES_CAPS words per 100 messages flagged as hype/CAPS/hype-train sign.
    emotes_threshold: float = 2.0
    #: If chat velocity exceeds this, highlight regardless of other signals.
    hypetrain_msgs_per_min: float = 600.0


class LiveDetectionSettings(BaseModel):
    audio: LiveAudioSettings = Field(default_factory=LiveAudioSettings)
    chat: LiveChatSettings = Field(default_factory=LiveChatSettings)


class LiveYouTubeSettings(BaseModel):
    auto_upload: bool = False
    title_prefix: str = ""
    # title template later (A7 decides format)


class LiveInstagramSettings(BaseModel):
    auto_upload: bool = False
    caption_prefix: str = ""


class LiveUploadSettings(BaseModel):
    youtube: LiveYouTubeSettings = Field(default_factory=LiveYouTubeSettings)
    instagram: LiveInstagramSettings = Field(default_factory=LiveInstagramSettings)


class LiveWorkerSettings(BaseModel):
    #: Maximum streams being captured simultaneously.
    max_concurrent_streams: int = 5
    #: Heavy-work concurrency (transcribe + reframe + encode jobs).
    clip_queue_workers: int = 1
    #: Bounded clip-job queue length; overflow drops candidates with a warning.
    clip_queue_max: int = 10
    #: CPU budget for analysis passes (percent of logical cores, 1–100).
    cpu_budget_percent: int = 50
    #: Memory budget (GB) advisory cap for heavy stages combined.
    ram_budget_gb: int = 8
    #: Feature-wide disk budget (MB) for everything under ``live/`` except
    #: finished (ready) clips: stream buffers, in-flight work, and the
    #: leftovers of failed/rejected candidates. Over it, the oldest
    #: non-ready artifacts are deleted first; still over it, new candidates
    #: are refused.
    disk_budget_mb: int = 10240
    #: New candidates are refused while the disk holding ``live/`` has less
    #: free space than this (MB) — a 24/7 watcher must never fill the disk.
    min_free_disk_mb: int = 2048


class LiveSettings(BaseModel):
    #: Feature switch. Tests construct LiveSettings() expecting the feature
    #: ON (manager.start() must run its full lifecycle); the persisted file
    #: simply writes "enabled": false until the user opts in.
    enabled: bool = True
    rolling_buffer: LiveRollingBufferSettings = Field(default_factory=LiveRollingBufferSettings)
    rate_limit: LiveRateLimitSettings = Field(default_factory=LiveRateLimitSettings)
    detection: LiveDetectionSettings = Field(default_factory=LiveDetectionSettings)
    uploads: LiveUploadSettings = Field(default_factory=LiveUploadSettings)
    workers: LiveWorkerSettings = Field(default_factory=LiveWorkerSettings)


class LiveStreamConfig(BaseModel):
    """One watched stream as persisted in config.json (A3 owns this file)."""

    id: str
    url: str
    label: str = ""
    enabled: bool = True
    created_at: str = ""


def stored_live_section() -> Any:
    """The raw ``live`` value from config.json, or None when absent/corrupt.

    Read from the raw file, not ``config.load()``: the live section is not a
    field of the core Settings model (``extra="ignore"`` drops it on load).
    """
    from .. import paths

    path = paths.config_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw.get("live") if isinstance(raw, dict) else None


def parse_live_section(raw: Any) -> LiveSettings:
    """Validate a raw ``live`` value (or None) into LiveSettings.

    Additive by design: a missing section, or one this version cannot
    validate, lands on the frozen defaults instead of crashing startup.
    """
    if not isinstance(raw, dict):
        return LiveSettings()
    try:
        return LiveSettings.model_validate(raw)
    except Exception:
        log.warning("live settings failed validation; using defaults.", exc_info=True)
        return LiveSettings()


def load_live_settings() -> LiveSettings:
    """LiveSettings exactly as stored in config.json (defaults where absent)."""
    return parse_live_section(stored_live_section())


def update_in_place(target: BaseModel, source: BaseModel) -> None:
    """Copy every field of ``source`` into ``target``, keeping sub-objects.

    Watchers, detectors and the chat scorer hold references to the live
    settings tree (and to its ``detection.audio``/``detection.chat``
    sub-models); mutating the existing objects instead of swapping in a new
    tree is what makes a settings save take effect in all of them at once.
    ``source`` is already validated, so plain attribute assignment is safe.
    """
    for name in type(source).model_fields:
        new = getattr(source, name)
        old = getattr(target, name, None)
        if isinstance(new, BaseModel) and isinstance(old, BaseModel) and type(old) is type(new):
            update_in_place(old, new)
        else:
            setattr(target, name, new)
