"""Row models for the tables in :mod:`autoclip.db.schema`.

Plain dataclasses rather than Pydantic models: these mirror SQLite rows and are
constructed from trusted database output, so validation would be dead weight.
The API layer converts them to Pydantic response models at the boundary.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

SourceType = Literal["youtube", "upload"]
JobStatus = Literal["queued", "running", "failed", "done", "cancelled"]
ClipStatus = Literal["candidate", "kept", "discarded", "exported"]


def new_id() -> str:
    """Generate a short, filesystem-safe, collision-resistant identifier."""
    return uuid.uuid4().hex[:16]


def utcnow() -> str:
    """Current UTC time as an ISO-8601 string — the storage format for all timestamps."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Source:
    id: str
    type: SourceType
    path: str
    title: str = ""
    url: str | None = None
    filename: str | None = None
    channel: str | None = None
    duration_s: float = 0.0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool = True
    has_video: bool = True
    created_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Source:
        return cls(
            id=row["id"],
            type=row["type"],
            path=row["path"],
            title=row["title"],
            url=row["url"],
            filename=row["filename"],
            channel=row["channel"],
            duration_s=row["duration_s"],
            width=row["width"],
            height=row["height"],
            fps=row["fps"],
            has_audio=bool(row["has_audio"]),
            has_video=bool(row["has_video"]),
            created_at=row["created_at"],
        )


@dataclass
class Job:
    id: str
    source_id: str
    status: JobStatus = "queued"
    current_stage: str = ""
    progress: float = 0.0
    error: str | None = None
    provider: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)
    started_at: str | None = None
    finished_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            status=row["status"],
            current_stage=row["current_stage"],
            progress=row["progress"],
            error=row["error"],
            provider=row["provider"],
            settings=json.loads(row["settings_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )


@dataclass
class Transcript:
    job_id: str
    json_path: str
    language: str = ""
    model: str = ""
    has_diarization: bool = False
    word_count: int = 0
    #: "whisper" or "youtube" — which path produced this transcript.
    source: str = "whisper"
    created_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Transcript:
        return cls(
            job_id=row["job_id"],
            json_path=row["json_path"],
            language=row["language"],
            model=row["model"],
            has_diarization=bool(row["has_diarization"]),
            word_count=row["word_count"],
            source=row["source"],
            created_at=row["created_at"],
        )


@dataclass
class Clip:
    id: str
    job_id: str
    start_s: float
    end_s: float
    rank: int = 0
    start_word: int = 0
    end_word: int = 0
    title: str = ""
    hook: str = ""
    score: int = 0
    reason: str = ""
    status: ClipStatus = "candidate"
    user_trimmed: bool = False
    created_at: str = field(default_factory=utcnow)
    #: Up to three on-screen headline options from detection.
    hook_options: list[str] = field(default_factory=list)
    #: The headline the render uses (an option or free text); empty = default.
    hook_text: str = ""
    #: Post text for the upload.
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    #: The line flagged for a flash-forward opening, absolute source seconds.
    teaser_start_s: float | None = None
    teaser_end_s: float | None = None
    #: Per-clip flash-forward. None follows ``export.flash_forward``.
    flash_forward: bool | None = None
    #: The clip this one is a variant of (None for an original).
    variant_of: str | None = None
    #: Zoom-pattern seed; 0 is the plain pattern, variants count up.
    motion_seed: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def headline(self) -> str:
        """The on-screen hook headline: the chosen one, else the first option, else the title."""
        for text in (self.hook_text, *self.hook_options[:1], self.title, self.hook):
            if text and text.strip():
                return text.strip()
        return ""

    @property
    def teaser(self) -> tuple[float, float] | None:
        if self.teaser_start_s is None or self.teaser_end_s is None:
            return None
        if self.teaser_end_s <= self.teaser_start_s:
            return None
        return (self.teaser_start_s, self.teaser_end_s)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Clip:
        flash = row["flash_forward"]
        return cls(
            id=row["id"],
            job_id=row["job_id"],
            start_s=row["start_s"],
            end_s=row["end_s"],
            rank=row["rank"],
            start_word=row["start_word"],
            end_word=row["end_word"],
            title=row["title"],
            hook=row["hook"],
            score=row["score"],
            reason=row["reason"],
            status=row["status"],
            user_trimmed=bool(row["user_trimmed"]),
            created_at=row["created_at"],
            hook_options=_json_list(row["hook_options"]),
            hook_text=row["hook_text"] or "",
            description=row["description"] or "",
            hashtags=_json_list(row["hashtags"]),
            teaser_start_s=row["teaser_start_s"],
            teaser_end_s=row["teaser_end_s"],
            flash_forward=None if flash is None else bool(flash),
            variant_of=row["variant_of"],
            motion_seed=row["motion_seed"] or 0,
        )


def _json_list(raw: str | None) -> list[str]:
    """A JSON list of strings from a TEXT column; anything unreadable is empty."""
    try:
        value = json.loads(raw or "[]")
    except ValueError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


@dataclass
class ClipEdit:
    clip_id: str
    edited_words: list[dict[str, Any]] | None = None
    caption_style: str = "bold_pop"
    ratio: str = "9:16"
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ClipEdit:
        raw = row["edited_words_json"]
        return cls(
            clip_id=row["clip_id"],
            edited_words=json.loads(raw) if raw else None,
            caption_style=row["caption_style"],
            ratio=row["ratio"],
            updated_at=row["updated_at"],
        )


@dataclass
class Export:
    id: str
    clip_id: str
    path: str
    ratio: str = "9:16"
    style: str = "bold_pop"
    size_bytes: int = 0
    created_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Export:
        return cls(
            id=row["id"],
            clip_id=row["clip_id"],
            path=row["path"],
            ratio=row["ratio"],
            style=row["style"],
            size_bytes=row["size_bytes"],
            created_at=row["created_at"],
        )
