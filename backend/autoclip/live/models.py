"""Row models for the Live Watch tables (migration v2).

Mirrors the conventions of ``db.models``: plain dataclasses, constructed from
trusted database output, converted to Pydantic at the API boundary.

Table overview (DDL in db/schema.py, migration v2 — A9 owns that file)::

    live_streams   one row per watched stream; runtime state synced here
    live_clips     one row per produced live clip (before/after render)
    live_uploads   one row per platform upload attempt (clip → platform)

A9 owns this file. The dataclass field lists below are FROZEN as the contract
other workers code against; A9 adds from_row() and store functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..db.models import new_id as _base_new_id  # reuse the id/utcnow helpers


def new_id() -> str:
    return _base_new_id()


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


LiveStreamStatus = str  # states.LiveStreamState values, stored as TEXT
LiveClipStatus = str  # detecting | queued | rendering | ready | failed | rejected
LiveUploadStatus = str  # pending | uploading | done | failed | skipped


@dataclass
class LiveStreamRow:
    id: str
    url: str
    label: str = ""
    enabled: bool = True
    state: LiveStreamStatus = "waiting_for_live"
    state_detail: str = ""
    last_error: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)


@dataclass
class LiveClipRow:
    id: str
    stream_id: str
    status: LiveClipStatus = "detecting"
    trigger_start_s: float = 0.0
    trigger_end_s: float = 0.0
    clip_start_s: float = 0.0
    clip_end_s: float = 0.0
    title: str = ""
    score: int = 0
    reason: str = ""
    hook: str = ""
    judge_model: str = ""
    transcript_path: str = ""
    rendered_path: str = ""
    caption_style: str = "bold_pop"
    ratio: str = "9:16"
    error: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)


@dataclass
class LiveUploadRow:
    id: str
    clip_id: str
    platform: str  # "youtube" | "instagram"
    status: LiveUploadStatus = "pending"
    upload_url: str = ""
    platform_media_id: str = ""
    error: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
