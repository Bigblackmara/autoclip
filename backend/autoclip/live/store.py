"""Storage layer for Live Watch rows.

A9 owns this file end to end; other workers import and call these functions.
Conventions mirror ``autoclip.db.store`` exactly: every function opens its own
transaction via :func:`autoclip.db.connection` (per-thread connections, WAL,
busy timeout), so callers never manage commits. Functions are synchronous; the
async manager wraps them in ``asyncio.to_thread``.

Schema (db/schema.py migration v2): ``live_streams``, ``live_clips``,
``live_uploads`` — one column per dataclass field in :mod:`.models`. The
status columns deliberately carry NO CHECK constraints: the status vocabulary
evolves with the feature (``states.py``), so forward-compatible writes must
never be blocked at the SQL layer. Validation that matters lives in code.

CRASH RECOVERY — the manager (A8) MUST call :func:`reset_interrupted_states`
once on ``start()``, before launching any watcher task: a restart kills every
watcher mid-flight, but their rows still claim states like ``watching``.
``db.init()`` deliberately does not do this (that file is not A9's to wire),
so the manager is the integration point.

Patching conventions
--------------------
``update_stream_state`` keeps its frozen keyword signature, where ``None``
means "leave the field alone" (it is the default of every parameter). To
clear ``last_error``, rewrite the row via :func:`upsert_stream` with
``last_error=None`` — the upsert writes every column.

``update_clip`` / ``update_upload`` take ``**fields``: a field present in the
dict is written, an omitted field is untouched. For the nullable ``error``
column, passing ``error=None`` explicitly clears it to NULL; passing
``error=UNSET`` (re-exported from ``db.store``) is the same as omitting the
field, mirroring :func:`autoclip.db.store.update_job`. Attempting to clear a
NOT NULL column raises ``ValueError`` rather than a subtle IntegrityError.
Unknown field names also raise ``ValueError`` — typo protection for
dict-driven callers, far clearer than the OperationalError a typoed column
would otherwise produce.

Every write refreshes ``updated_at`` (except a completely empty patch, which
is a no-op), so the UI can detect staleness the same way it does for jobs.
"""

from __future__ import annotations

from typing import Any

from ..db import connection
from ..db.store import UNSET
from .models import LiveClipRow, LiveStreamRow, LiveUploadRow, utcnow

__all__ = [
    "UNSET",
    "create_clip",
    "create_upload",
    "delete_stream",
    "get_clip",
    "get_stream",
    "list_clips",
    "list_streams",
    "list_uploads",
    "reset_interrupted_states",
    "update_clip",
    "update_stream_state",
    "update_upload",
    "upsert_stream",
]

_Unset = type(UNSET)  # private class in db.store; typed here for isinstance checks


def _is_unset(value: Any) -> bool:
    return isinstance(value, _Unset) or value is UNSET


# --------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------


def get_stream(stream_id: str) -> LiveStreamRow | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM live_streams WHERE id = ?", (stream_id,)).fetchone()
    return LiveStreamRow.from_row(row) if row else None


def list_streams() -> list[LiveStreamRow]:
    """All streams in registration order (oldest first).

    Streams are a curated watch list, not a feed — the user expects them
    stable in the order they were added. Clips and uploads below are feeds:
    newest first.
    """
    with connection() as conn:
        rows = conn.execute("SELECT * FROM live_streams ORDER BY created_at ASC, id ASC").fetchall()
    return [LiveStreamRow.from_row(r) for r in rows]


def upsert_stream(row: LiveStreamRow) -> LiveStreamRow:
    """Insert the stream, or update every column of the existing row.

    ``created_at`` is written on insert only — a re-upsert never resets a
    stream's age. ``updated_at`` is always refreshed. Returns the row as it
    now exists in the database (not the caller's copy), so callers can rely
    on the returned timestamps.

    This is also the path that clears ``last_error`` — the frozen
    ``update_stream_state`` signature treats ``None`` as "leave alone", so
    a full-row rewrite is the clearing mechanism.
    """
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO live_streams (id, url, label, enabled, state, state_detail,
                                       last_error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                url          = excluded.url,
                label        = excluded.label,
                enabled      = excluded.enabled,
                state        = excluded.state,
                state_detail = excluded.state_detail,
                last_error   = excluded.last_error,
                updated_at   = excluded.updated_at
            """,
            (
                row.id,
                row.url,
                row.label,
                int(row.enabled),
                row.state,
                row.state_detail,
                row.last_error,
                row.created_at,
                utcnow(),
            ),
        )
    loaded = get_stream(row.id)
    assert loaded is not None  # just wrote it; a miss here means a broken database
    return loaded


def update_stream_state(
    stream_id: str,
    *,
    state: str | None = None,
    state_detail: str | None = None,
    error: str | None = None,
) -> None:
    """Patch a stream's runtime fields — the watcher's hot path.

    Semantics of the frozen signature: ``None`` (the default of every
    parameter) means "leave the field alone", so a state transition never
    accidentally wipes the recorded error. Pass ``state_detail=""`` to clear
    the detail string; to clear ``last_error`` rewrite the row with
    :func:`upsert_stream`.

    ``updated_at`` is always refreshed.
    """
    fields: dict[str, Any] = {"updated_at": utcnow()}
    if state is not None:
        fields["state"] = state
    if state_detail is not None:
        fields["state_detail"] = state_detail
    if error is not None:
        fields["last_error"] = error

    assignments = ", ".join(f"{name} = ?" for name in fields)
    with connection() as conn:
        conn.execute(
            f"UPDATE live_streams SET {assignments} WHERE id = ?",
            (*fields.values(), stream_id),
        )


def delete_stream(stream_id: str) -> None:
    """Remove a stream; its clips and uploads go with it (ON DELETE CASCADE)."""
    with connection() as conn:
        conn.execute("DELETE FROM live_streams WHERE id = ?", (stream_id,))


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------

_CLIP_NULLABLE = frozenset({"error"})
_CLIP_PATCHABLE = frozenset(
    {
        "stream_id",
        "status",
        "trigger_start_s",
        "trigger_end_s",
        "clip_start_s",
        "clip_end_s",
        "title",
        "score",
        "reason",
        "hook",
        "judge_model",
        "transcript_path",
        "rendered_path",
        "caption_style",
        "ratio",
        "error",
    }
)


def create_clip(row: LiveClipRow) -> LiveClipRow:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO live_clips (id, stream_id, status, trigger_start_s,
                                     trigger_end_s, clip_start_s, clip_end_s,
                                     title, score, reason, hook, judge_model,
                                     transcript_path, rendered_path, caption_style,
                                     ratio, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.id,
                row.stream_id,
                row.status,
                row.trigger_start_s,
                row.trigger_end_s,
                row.clip_start_s,
                row.clip_end_s,
                row.title,
                row.score,
                row.reason,
                row.hook,
                row.judge_model,
                row.transcript_path,
                row.rendered_path,
                row.caption_style,
                row.ratio,
                row.error,
                row.created_at,
                row.updated_at,
            ),
        )
    return row


def get_clip(clip_id: str) -> LiveClipRow | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM live_clips WHERE id = ?", (clip_id,)).fetchone()
    return LiveClipRow.from_row(row) if row else None


def list_clips(stream_id: str | None = None, limit: int = 200) -> list[LiveClipRow]:
    """Newest-first clips, optionally scoped to one stream."""
    query = "SELECT * FROM live_clips"
    params: list[Any] = []
    if stream_id is not None:
        query += " WHERE stream_id = ?"
        params.append(stream_id)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)
    with connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [LiveClipRow.from_row(r) for r in rows]


def update_clip(clip_id: str, **fields: Any) -> None:
    """Patch clip columns; see "Patching conventions" in the module docstring.

    A field present in ``fields`` is written; omitted fields (or fields whose
    value is :data:`UNSET`) are untouched. ``error=None`` clears the error to
    NULL; clearing any other (NOT NULL) column or naming an unknown column
    raises ``ValueError``. An empty patch is a no-op and does not touch
    ``updated_at``.
    """
    _patch_row("live_clips", clip_id, fields, _CLIP_PATCHABLE, _CLIP_NULLABLE)


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------

_UPLOAD_NULLABLE = frozenset({"error"})
_UPLOAD_PATCHABLE = frozenset(
    {"clip_id", "platform", "status", "upload_url", "platform_media_id", "error"}
)


def create_upload(row: LiveUploadRow) -> LiveUploadRow:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO live_uploads (id, clip_id, platform, status, upload_url,
                                      platform_media_id, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.id,
                row.clip_id,
                row.platform,
                row.status,
                row.upload_url,
                row.platform_media_id,
                row.error,
                row.created_at,
                row.updated_at,
            ),
        )
    return row


def list_uploads(clip_id: str | None = None, limit: int = 200) -> list[LiveUploadRow]:
    """Newest-first upload attempts, optionally scoped to one clip."""
    query = "SELECT * FROM live_uploads"
    params: list[Any] = []
    if clip_id is not None:
        query += " WHERE clip_id = ?"
        params.append(clip_id)
    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)
    with connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [LiveUploadRow.from_row(r) for r in rows]


def update_upload(upload_id: str, **fields: Any) -> None:
    """Patch upload columns; same conventions as :func:`update_clip`."""
    _patch_row("live_uploads", upload_id, fields, _UPLOAD_PATCHABLE, _UPLOAD_NULLABLE)


# --------------------------------------------------------------------------
# Shared patch machinery
# --------------------------------------------------------------------------


def _patch_row(
    table: str,
    key_value: str,
    fields: dict[str, Any],
    patchable: frozenset[str],
    nullable: frozenset[str],
) -> None:
    """Apply a partial UPDATE with None-means-clear semantics.

    Table and column names come from this module's own constants only, never
    from caller input, so the f-string interpolation is safe. Unknown or
    read-only names fail fast with ``ValueError``.
    """
    fields = dict(fields)
    if not fields:
        return

    unknown = sorted(set(fields) - patchable)
    if unknown:
        raise ValueError(f"Unknown or read-only field(s) on {table}: {', '.join(unknown)}")

    assignments: dict[str, Any] = {"updated_at": utcnow()}
    for name, value in fields.items():
        if _is_unset(value):
            continue  # explicit UNSET == omitted: leave the column alone
        if value is None:
            if name not in nullable:
                raise ValueError(f"Cannot clear non-nullable column '{name}' on {table}")
            assignments[name] = None
        else:
            assignments[name] = value

    if len(assignments) == 1:  # only updated_at: nothing to write
        return

    set_clause = ", ".join(f"{name} = ?" for name in assignments)
    with connection() as conn:
        conn.execute(
            f"UPDATE {table} SET {set_clause} WHERE id = ?",
            (*assignments.values(), key_value),
        )


# --------------------------------------------------------------------------
# Crash recovery
# --------------------------------------------------------------------------

#: States a watcher task leaves behind on a hard kill; everything except the
#: terminal states (ended, error) and user-paused (off). Mirrors
#: ``states.ACTIVE_STATES`` minus ``off``, spelled as raw strings so this
#: module keeps no import-time coupling to the states module.
RESETTABLE_STATES: tuple[str, ...] = (
    "waiting_for_live",
    "connecting",
    "watching",
    "clipping",
    "reconnecting",
)


def reset_interrupted_states() -> int:
    """Crash recovery: reset non-terminal live rows (see module docstring).

    Every stream in an active state is flipped back to ``waiting_for_live``
    with its ``state_detail`` cleared — a restart killed all watcher tasks,
    and the manager re-launches from ``waiting_for_live``. ``off``, ``ended``
    and ``error`` rows are untouched: paused-by-user is a deliberate state,
    and terminal states describe a genuinely finished stream. ``last_error``
    is intentionally preserved as a diagnostic breadcrumb of the previous
    run; clear it through :func:`upsert_stream` when appropriate.

    Returns the number of matching rows. Idempotent — a second call finds
    nothing left to reset except rows already in ``waiting_for_live``, which
    keep that state (only their stale detail is cleared).

    Requires :func:`autoclip.db.init` to have run (the manager always inits
    before starting watchers, so this is never a practical constraint).
    """
    placeholders = ", ".join("?" for _ in RESETTABLE_STATES)
    with connection() as conn:
        cursor = conn.execute(
            f"""
            UPDATE live_streams
            SET state = 'waiting_for_live',
                state_detail = '',
                updated_at = ?
            WHERE state IN ({placeholders})
            """,
            (utcnow(), *RESETTABLE_STATES),
        )
        return cursor.rowcount
