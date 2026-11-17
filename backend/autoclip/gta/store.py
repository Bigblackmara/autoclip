"""GTA Mode rows and their CRUD (tables from schema migration v4).

Every function opens its own transaction through :func:`autoclip.db.connection`
and is synchronous; async callers wrap them in ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from ..db import connection
from ..db.models import new_id, utcnow

#: Stream lifecycle. ``queued`` -> the stages below -> ``done``; any stage can
#: end in ``failed`` or ``cancelled``. ``analyzed`` means judging is over and
#: the surviving clips are being downloaded and rendered.
STREAM_STATUSES = frozenset(
    {
        "queued",
        "probing",
        "downloading_audio",
        "fetching_signals",
        "transcribing",
        "building_hype",
        "judging",
        "analyzed",
        "done",
        "failed",
        "cancelled",
        # GTA Live ("Clip this live stream") — owned by gta.live, never by
        # the VOD queue.
        "live_connecting",
        "live_watching",
        "live_ended",
        "live_stopped",
    }
)
LIVE_STREAM_STATUSES = frozenset({"live_connecting", "live_watching", "live_ended", "live_stopped"})
#: Statuses that mean "a VOD queue worker should pick this stream up".
ACTIVE_STREAM_STATUSES = frozenset(
    STREAM_STATUSES - {"done", "failed", "cancelled"} - LIVE_STREAM_STATUSES
)

#: Clip lifecycle. Rejections keep their row (with the reason) so the user can
#: see why a moment was thrown out, and so the same moment is never clipped
#: twice.
CLIP_STATUSES = frozenset(
    {
        "candidate",
        "rejected",
        "approved",
        "downloading",
        "rendering",
        "ready",
        "qa_failed",
        "failed",
    }
)
#: Approved clips still owed a render.
PENDING_CLIP_STATUSES = frozenset({"approved", "downloading", "rendering"})


@dataclass
class GtaStream:
    id: str
    url: str
    platform: str = ""
    title: str = ""
    channel: str = ""
    duration_s: float = 0.0
    range_start_s: float | None = None
    range_end_s: float | None = None
    status: str = "queued"
    stage: str = ""
    progress: float = 0.0
    detail: str = ""
    error: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GtaStream:
        return cls(
            id=row["id"],
            url=row["url"],
            platform=row["platform"],
            title=row["title"],
            channel=row["channel"],
            duration_s=row["duration_s"],
            range_start_s=row["range_start_s"],
            range_end_s=row["range_end_s"],
            status=row["status"],
            stage=row["stage"],
            progress=row["progress"],
            detail=row["detail"],
            error=row["error"],
            signals=_loads(row["signals"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class GtaClip:
    id: str
    stream_id: str
    start_s: float
    end_s: float
    status: str = "candidate"
    chunk_index: int = 0
    hook_title: str = ""
    hook_quote: str = ""
    moment_type: str = ""
    summary: str = ""
    hype_score: float = 0.0
    score: int = 0
    rank: int | None = None
    verdict: dict[str, Any] = field(default_factory=dict)
    reject_reason: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    path: str = ""
    qa: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GtaClip:
        return cls(
            id=row["id"],
            stream_id=row["stream_id"],
            start_s=row["start_s"],
            end_s=row["end_s"],
            status=row["status"],
            chunk_index=row["chunk_index"],
            hook_title=row["hook_title"],
            hook_quote=row["hook_quote"],
            moment_type=row["moment_type"],
            summary=row["summary"],
            hype_score=row["hype_score"],
            score=row["score"],
            rank=row["rank"],
            verdict=_loads(row["verdict"], {}),
            reject_reason=row["reject_reason"],
            caption=row["caption"],
            hashtags=_loads(row["hashtags"], []),
            path=row["path"],
            qa=_loads(row["qa"], {}),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _loads(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        value = json.loads(text)
    except ValueError:
        return default
    return value if isinstance(value, type(default)) else default


#: Columns update_* may touch, and which of them hold JSON.
_STREAM_COLUMNS = frozenset(
    {
        "platform",
        "title",
        "channel",
        "duration_s",
        "range_start_s",
        "range_end_s",
        "status",
        "stage",
        "progress",
        "detail",
        "error",
        "signals",
    }
)
_CLIP_COLUMNS = frozenset(
    {
        "status",
        "start_s",
        "end_s",
        "chunk_index",
        "hook_title",
        "hook_quote",
        "moment_type",
        "summary",
        "hype_score",
        "score",
        "rank",
        "verdict",
        "reject_reason",
        "caption",
        "hashtags",
        "path",
        "qa",
        "error",
    }
)
_JSON_COLUMNS = frozenset({"signals", "verdict", "hashtags", "qa"})


def _encode(column: str, value: Any) -> Any:
    if column in _JSON_COLUMNS:
        return json.dumps(value)
    return value


def _assignments(fields: dict[str, Any], allowed: frozenset[str]) -> tuple[str, list[Any]]:
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown column(s): {', '.join(sorted(unknown))}")
    columns = sorted(fields)
    sql = ", ".join(f"{name} = ?" for name in columns)
    return sql, [_encode(name, fields[name]) for name in columns]


# --------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------


def create_stream(
    url: str,
    *,
    range_start_s: float | None = None,
    range_end_s: float | None = None,
) -> GtaStream:
    stream = GtaStream(id=new_id(), url=url, range_start_s=range_start_s, range_end_s=range_end_s)
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO gta_streams (id, url, range_start_s, range_end_s, status,
                                     signals, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stream.id,
                stream.url,
                stream.range_start_s,
                stream.range_end_s,
                stream.status,
                "{}",
                stream.created_at,
                stream.updated_at,
            ),
        )
    return stream


def get_stream(stream_id: str) -> GtaStream | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM gta_streams WHERE id = ?", (stream_id,)).fetchone()
    return GtaStream.from_row(row) if row else None


def list_streams(*, statuses: frozenset[str] | None = None, limit: int = 200) -> list[GtaStream]:
    with connection() as conn:
        if statuses:
            marks = ", ".join("?" for _ in statuses)
            rows = conn.execute(
                f"SELECT * FROM gta_streams WHERE status IN ({marks}) "
                "ORDER BY created_at ASC LIMIT ?",
                (*sorted(statuses), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM gta_streams ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [GtaStream.from_row(row) for row in rows]


def update_stream(stream_id: str, **fields: Any) -> None:
    if "status" in fields and fields["status"] not in STREAM_STATUSES:
        raise ValueError(f"Unknown stream status {fields['status']!r}")
    if not fields:
        return
    sql, values = _assignments(fields, _STREAM_COLUMNS)
    with connection() as conn:
        conn.execute(
            f"UPDATE gta_streams SET {sql}, updated_at = ? WHERE id = ?",
            (*values, utcnow(), stream_id),
        )


def delete_stream(stream_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM gta_streams WHERE id = ?", (stream_id,))


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


def create_clip(clip: GtaClip) -> GtaClip:
    if clip.status not in CLIP_STATUSES:
        raise ValueError(f"Unknown clip status {clip.status!r}")
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO gta_clips (id, stream_id, status, start_s, end_s, chunk_index,
                                   hook_title, hook_quote, moment_type, summary,
                                   hype_score, score, rank, verdict, reject_reason,
                                   caption, hashtags, path, qa, error,
                                   created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clip.id,
                clip.stream_id,
                clip.status,
                clip.start_s,
                clip.end_s,
                clip.chunk_index,
                clip.hook_title,
                clip.hook_quote,
                clip.moment_type,
                clip.summary,
                clip.hype_score,
                clip.score,
                clip.rank,
                json.dumps(clip.verdict),
                clip.reject_reason,
                clip.caption,
                json.dumps(clip.hashtags),
                clip.path,
                json.dumps(clip.qa),
                clip.error,
                clip.created_at,
                clip.updated_at,
            ),
        )
    return clip


def get_clip(clip_id: str) -> GtaClip | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM gta_clips WHERE id = ?", (clip_id,)).fetchone()
    return GtaClip.from_row(row) if row else None


def list_clips(
    stream_id: str | None = None,
    *,
    statuses: frozenset[str] | None = None,
    limit: int = 1000,
) -> list[GtaClip]:
    clauses: list[str] = []
    params: list[Any] = []
    if stream_id is not None:
        clauses.append("stream_id = ?")
        params.append(stream_id)
    if statuses:
        clauses.append(f"status IN ({', '.join('?' for _ in statuses)})")
        params.extend(sorted(statuses))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM gta_clips {where} ORDER BY stream_id, start_s LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [GtaClip.from_row(row) for row in rows]


def update_clip(clip_id: str, **fields: Any) -> None:
    if "status" in fields and fields["status"] not in CLIP_STATUSES:
        raise ValueError(f"Unknown clip status {fields['status']!r}")
    if not fields:
        return
    sql, values = _assignments(fields, _CLIP_COLUMNS)
    with connection() as conn:
        conn.execute(
            f"UPDATE gta_clips SET {sql}, updated_at = ? WHERE id = ?",
            (*values, utcnow(), clip_id),
        )


def delete_clips(stream_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM gta_clips WHERE stream_id = ?", (stream_id,))


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def create_report(channel_url: str, report: dict[str, Any]) -> str:
    report_id = new_id()
    with connection() as conn:
        conn.execute(
            "INSERT INTO gta_reports (id, channel_url, report, created_at) VALUES (?, ?, ?, ?)",
            (report_id, channel_url, json.dumps(report), utcnow()),
        )
    return report_id


def latest_report() -> dict[str, Any] | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM gta_reports ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "channel_url": row["channel_url"],
        "created_at": row["created_at"],
        "report": _loads(row["report"], {}),
    }


# --------------------------------------------------------------------------
# YouTube uploads (posting schedule, schema migration v5)
# --------------------------------------------------------------------------

#: ``uploading`` -> ``scheduled`` (on YouTube, private until its slot, then
#: public). ``failed`` rows are retried (bounded by ``attempts``);
#: ``interrupted`` = AutoClip stopped mid-upload, so whether YouTube got the
#: video is unknown and nothing retries it on its own; ``skipped`` = the user
#: excluded the clip from auto-posting.
UPLOAD_STATUSES = frozenset({"uploading", "scheduled", "failed", "interrupted", "skipped"})
#: Rows that own their publish slot (no other clip may take it).
SLOT_HOLDING_STATUSES = frozenset({"uploading", "scheduled", "interrupted"})


@dataclass
class GtaUpload:
    id: str
    clip_id: str
    status: str
    slot_at: str | None = None
    video_id: str = ""
    url: str = ""
    title: str = ""
    error: str | None = None
    attempts: int = 0
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GtaUpload:
        return cls(
            id=row["id"],
            clip_id=row["clip_id"],
            status=row["status"],
            slot_at=row["slot_at"],
            video_id=row["video_id"],
            url=row["url"],
            title=row["title"],
            error=row["error"],
            attempts=row["attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


_UPLOAD_COLUMNS = frozenset({"status", "slot_at", "video_id", "url", "title", "error", "attempts"})


def _check_upload_status(status: str) -> None:
    if status not in UPLOAD_STATUSES:
        raise ValueError(f"Unknown upload status {status!r}")


def create_upload(
    clip_id: str, *, status: str, slot_at: str | None = None, title: str = ""
) -> GtaUpload:
    """Insert the clip's upload row (raises sqlite3.IntegrityError if it has one)."""
    _check_upload_status(status)
    upload = GtaUpload(id=new_id(), clip_id=clip_id, status=status, slot_at=slot_at, title=title)
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO gta_uploads (id, clip_id, status, slot_at, title, attempts,
                                     created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (upload.id, clip_id, status, slot_at, title, upload.created_at, upload.updated_at),
        )
    return upload


def get_upload(upload_id: str) -> GtaUpload | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM gta_uploads WHERE id = ?", (upload_id,)).fetchone()
    return GtaUpload.from_row(row) if row else None


def upload_for_clip(clip_id: str) -> GtaUpload | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM gta_uploads WHERE clip_id = ?", (clip_id,)).fetchone()
    return GtaUpload.from_row(row) if row else None


def list_uploads(*, statuses: frozenset[str] | None = None, limit: int = 1000) -> list[GtaUpload]:
    clauses: list[str] = []
    params: list[Any] = []
    if statuses:
        clauses.append(f"status IN ({', '.join('?' for _ in statuses)})")
        params.extend(sorted(statuses))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM gta_uploads {where} "
            "ORDER BY COALESCE(slot_at, created_at) DESC, rowid DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [GtaUpload.from_row(row) for row in rows]


def update_upload(upload_id: str, **fields: Any) -> None:
    if "status" in fields:
        _check_upload_status(fields["status"])
    if not fields:
        return
    sql, values = _assignments(fields, _UPLOAD_COLUMNS)
    with connection() as conn:
        conn.execute(
            f"UPDATE gta_uploads SET {sql}, updated_at = ? WHERE id = ?",
            (*values, utcnow(), upload_id),
        )


def delete_upload(upload_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM gta_uploads WHERE id = ?", (upload_id,))
