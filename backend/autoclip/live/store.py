"""Storage layer singleton for Live Watch rows.

A9 owns this file end to end (store functions + wiring into db/schema.py
migration v2). Other workers import and call these functions; the skeletons
raise NotImplementedError until A9 lands its work order.

Design notes for A9:
- Follow the conventions of ``autoclip.db.store`` (per-thread connections,
  WAL, UNSET-sentinel for nullable patch fields).
- Migration v2 appends three tables to db/schema.py (live_streams,
  live_clips, live_uploads) — do NOT touch _V1 or MIGRATIONS[0].
- CRASH RECOVERY: on db.init(), any live_streams row with a non-terminal
  state ('waiting_for_live','connecting','watching','clipping',
  'reconnecting') must be reset to 'waiting_for_live' and its state_detail
  cleared — a restart kills all watcher tasks, and the manager re-launches
  from 'waiting_for_live'.
- live_clips.status covers: detecting → queued → rendering → ready | failed;
  plus 'rejected' (LLM gate said no or rate-limited). Keep CK constraints
  OUT of the DDL for status (states evolve) — enforce in code.
"""

from __future__ import annotations

from .models import LiveClipRow, LiveStreamRow, LiveUploadRow


def get_stream(stream_id: str) -> LiveStreamRow | None:
    raise NotImplementedError("A9: implement")


def list_streams() -> list[LiveStreamRow]:
    raise NotImplementedError("A9: implement")


def upsert_stream(row: LiveStreamRow) -> LiveStreamRow:
    raise NotImplementedError("A9: implement")


def update_stream_state(
    stream_id: str,
    *,
    state: str | None = None,
    state_detail: str | None = None,
    error: str | None = None,
) -> None:
    raise NotImplementedError("A9: implement")


def delete_stream(stream_id: str) -> None:
    raise NotImplementedError("A9: implement")


def create_clip(row: LiveClipRow) -> LiveClipRow:
    raise NotImplementedError("A9: implement")


def get_clip(clip_id: str) -> LiveClipRow | None:
    raise NotImplementedError("A9: implement")


def list_clips(stream_id: str | None = None, limit: int = 200) -> list[LiveClipRow]:
    raise NotImplementedError("A9: implement")


def update_clip(clip_id: str, **fields) -> None:
    raise NotImplementedError("A9: implement")


def create_upload(row: LiveUploadRow) -> LiveUploadRow:
    raise NotImplementedError("A9: implement")


def list_uploads(clip_id: str | None = None, limit: int = 200) -> list[LiveUploadRow]:
    raise NotImplementedError("A9: implement")


def update_upload(upload_id: str, **fields) -> None:
    raise NotImplementedError("A9: implement")


def reset_interrupted_states() -> int:
    """Crash recovery: reset non-terminal live rows (see module docstring).

    Returns the number of streams reset. Called by db.init() wiring or the
    manager on startup — A9 decides placement, but it MUST run before the
    manager starts any watcher.
    """
    raise NotImplementedError("A9: implement")
