"""Live Watch settings (persisted in config.json under ``live``).

Frozen defaults; A4 owns the bodies of detection-related subsections and may add
fields to LiveDetectionSettings, but must keep every existing field and default
so other workers' code keeps compiling against the skeleton as written.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LiveRollingBufferSettings(BaseModel):
    #: Seconds of stream time per .ts segment.
    segment_s: float = 6.0
    #: How much of the stream to keep on disk, in minutes.
    retain_minutes: float = 5.0
    #: Hard disk cap for ONE stream's buffer, megabytes.
    max_disk_mb: int = 512
    #: Seconds captured before the trigger moment (pre-roll).
    pre_roll_s: float = 20.0
    #: Seconds captured after the moment ends (post-roll).
    post_roll_s: float = 10.0


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


class LiveSettings(BaseModel):
    enabled: bool = False
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
