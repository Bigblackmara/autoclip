"""Retention: delete old work automatically.

With ``housekeeping.retention_days`` = N (0 = off), an hourly background
sweep deletes what was made more than N days ago, database rows and files
under the AutoClip home alike:

* normal mode: jobs (their clips, transcripts, exports, ``work/<job>`` and
  ``exports/<job>``) and sources (``media/<source>``). A source goes only
  once every job made from it is going too, so a fresh job on an old video
  keeps its video.
* GTA Mode: clips (their rendered file, when it is inside the AutoClip home)
  and, once no clip is left, the stream and its working files.

Never deleted: a job that is queued or running; a GTA stream that is being
processed or watched live; a GTA clip still rendering, uploading, or holding
a pending or scheduled upload (``uploading``, ``scheduled``, ``interrupted``,
``checking``, or ``failed`` with retries left). Those are looked at again at
the next sweep. Files outside the AutoClip home (a user-chosen GTA output
folder) are never touched. Live Watch keeps its own retention rules.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import paths
from .db import connection

log = logging.getLogger(__name__)

SWEEP_EVERY_S = 3600.0
STARTUP_DELAY_S = 60.0

BUSY_JOB_STATUSES = frozenset({"queued", "running"})
#: GTA upload rows that still owe (or may still owe) a post.
PENDING_UPLOAD_STATUSES = frozenset({"uploading", "scheduled", "interrupted", "checking"})


@dataclass
class Guards:
    """What is in use right now, beyond what the database rows say."""

    running_job_id: str | None = None
    #: GTA queue busy keys ("stream:<id>...", "clip:<stream>:<clip>...").
    gta_busy_keys: set[str] = field(default_factory=set)
    gta_uploading_clip_ids: set[str] = field(default_factory=set)
    gta_live_running: Callable[[str], bool] = lambda _stream_id: False
    #: Failed GTA uploads are retried up to this many attempts.
    gta_max_attempts: int = 3


def live_guards() -> Guards:
    """Guards from the running app's singletons."""
    from .gta.live import live_manager
    from .gta.poster import MAX_ATTEMPTS, poster
    from .gta.queue import queue as gta_queue
    from .jobs.queue import queue

    return Guards(
        running_job_id=queue.status().running_job_id,
        gta_busy_keys=set(gta_queue.busy_keys()),
        gta_uploading_clip_ids=set(poster.uploading_clip_ids()),
        gta_live_running=live_manager.is_running,
        gta_max_attempts=MAX_ATTEMPTS,
    )


@dataclass
class Plan:
    jobs: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    gta_clips: list[str] = field(default_factory=list)
    gta_streams: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.jobs or self.sources or self.gta_clips or self.gta_streams)


@dataclass
class Report:
    jobs: int = 0
    sources: int = 0
    gta_clips: int = 0
    gta_streams: int = 0
    errors: list[str] = field(default_factory=list)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _older(value: str | None, cutoff: datetime) -> bool:
    moment = parse_time(value)
    # An unreadable timestamp is never "old": deleting needs certainty.
    return moment is not None and moment < cutoff


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def select(days: int, *, now: datetime | None = None, guards: Guards | None = None) -> Plan:
    """What a sweep would delete. Reads only."""
    plan = Plan()
    if days <= 0:
        return plan
    guards = guards or Guards()
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)

    with connection() as conn:
        jobs = conn.execute("SELECT id, source_id, status, created_at FROM jobs").fetchall()
        sources = conn.execute("SELECT id, created_at FROM sources").fetchall()

        expired_jobs: set[str] = set()
        for job in jobs:
            if (
                _older(job["created_at"], cutoff)
                and job["status"] not in BUSY_JOB_STATUSES
                and job["id"] != guards.running_job_id
            ):
                expired_jobs.add(job["id"])
                plan.jobs.append(job["id"])

        jobs_by_source: dict[str, list[str]] = {}
        for job in jobs:
            jobs_by_source.setdefault(job["source_id"], []).append(job["id"])
        for source in sources:
            if _older(source["created_at"], cutoff) and all(
                job_id in expired_jobs for job_id in jobs_by_source.get(source["id"], [])
            ):
                plan.sources.append(source["id"])

        if not _table_exists(conn, "gta_streams"):
            return plan
        streams = conn.execute("SELECT id, status, created_at FROM gta_streams").fetchall()
        clips = conn.execute("SELECT id, stream_id, status, created_at FROM gta_clips").fetchall()
        uploads = conn.execute("SELECT clip_id, status, attempts FROM gta_uploads").fetchall()

    from .gta.store import ACTIVE_STREAM_STATUSES, PENDING_CLIP_STATUSES

    pending_upload_clips = {
        row["clip_id"]
        for row in uploads
        if row["status"] in PENDING_UPLOAD_STATUSES
        or (row["status"] == "failed" and (row["attempts"] or 0) < guards.gta_max_attempts)
    }
    clips_by_stream: dict[str, list[Any]] = {}
    for clip in clips:
        clips_by_stream.setdefault(clip["stream_id"], []).append(clip)

    for stream in streams:
        stream_id = stream["id"]
        busy = (
            stream["status"] in ACTIVE_STREAM_STATUSES
            or stream["status"] in ("live_connecting", "live_watching")
            or guards.gta_live_running(stream_id)
            or any(
                key.startswith((f"stream:{stream_id}", f"clip:{stream_id}:"))
                for key in guards.gta_busy_keys
            )
        )
        if busy or not _older(stream["created_at"], cutoff):
            continue
        kept = 0
        for clip in clips_by_stream.get(stream_id, []):
            if (
                _older(clip["created_at"], cutoff)
                and clip["status"] not in PENDING_CLIP_STATUSES
                and clip["id"] not in pending_upload_clips
                and clip["id"] not in guards.gta_uploading_clip_ids
            ):
                plan.gta_clips.append(clip["id"])
            else:
                kept += 1
        if kept == 0:
            plan.gta_streams.append(stream_id)
    return plan


# --------------------------------------------------------------------------
# Deleting
# --------------------------------------------------------------------------


def _inside_home(path: Path) -> bool:
    try:
        resolved = path.resolve()
        home = paths.root().resolve()
    except OSError:
        return False
    return resolved != home and resolved.is_relative_to(home)


def remove_path(path: Path | str | None) -> None:
    """Delete a file or folder, but only inside the AutoClip home."""
    if not path:
        return
    target = Path(path)
    if not target.exists() or not _inside_home(target):
        return
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            target.unlink()


def delete_job(job_id: str) -> None:
    """A normal-mode job: its rows (clips, transcript, exports cascade) and files."""
    with connection() as conn:
        export_paths = [
            row["path"]
            for row in conn.execute(
                "SELECT e.path FROM exports e JOIN clips c ON c.id = e.clip_id WHERE c.job_id = ?",
                (job_id,),
            ).fetchall()
        ]
        # Re-checked in the DELETE itself: a retry may have requeued it since.
        deleted = conn.execute(
            "DELETE FROM jobs WHERE id = ? AND status NOT IN ('queued', 'running')", (job_id,)
        ).rowcount
    if not deleted:
        return
    for export_path in export_paths:
        remove_path(export_path)
    remove_path(paths.job_work_dir(job_id))
    remove_path(paths.exports_dir() / job_id)


def delete_source(source_id: str) -> None:
    """A normal-mode source with no jobs left: its row and its media folder."""
    with connection() as conn:
        # Only once no job is left on it (a new one may have been made since).
        deleted = conn.execute(
            "DELETE FROM sources WHERE id = ? AND NOT EXISTS "
            "(SELECT 1 FROM jobs WHERE source_id = ?)",
            (source_id, source_id),
        ).rowcount
    if not deleted:
        return
    remove_path(paths.source_media_dir(source_id))


def delete_gta_clip(clip_id: str) -> None:
    with connection() as conn:
        row = conn.execute("SELECT path FROM gta_clips WHERE id = ?", (clip_id,)).fetchone()
        conn.execute("DELETE FROM gta_clips WHERE id = ?", (clip_id,))
    if row is not None:
        remove_path(row["path"])


def delete_gta_stream(stream_id: str) -> None:
    from .gta import store as gta_store
    from .gta.settings import work_root

    gta_store.delete_stream(stream_id)
    remove_path(work_root() / stream_id)


def apply(plan: Plan) -> Report:
    report = Report()
    steps: list[tuple[str, list[str], Callable[[str], None]]] = [
        ("jobs", plan.jobs, delete_job),
        ("sources", plan.sources, delete_source),
        ("gta_clips", plan.gta_clips, delete_gta_clip),
        ("gta_streams", plan.gta_streams, delete_gta_stream),
    ]
    for name, ids, delete in steps:
        for item_id in ids:
            try:
                delete(item_id)
            except Exception as exc:
                log.warning("Retention: could not delete %s %s: %s", name, item_id, exc)
                report.errors.append(f"{name} {item_id}: {exc}")
            else:
                setattr(report, name, getattr(report, name) + 1)
    return report


def sweep(days: int, *, now: datetime | None = None, guards: Guards | None = None) -> Report:
    plan = select(days, now=now, guards=guards)
    if plan.empty:
        return Report()
    report = apply(plan)
    log.info(
        "Retention (%d days): deleted %d job(s), %d source(s), %d GTA clip(s), %d GTA stream(s).",
        days,
        report.jobs,
        report.sources,
        report.gta_clips,
        report.gta_streams,
    )
    return report


# --------------------------------------------------------------------------
# Background
# --------------------------------------------------------------------------


class RetentionService:
    """Hourly sweep; idle while ``retention_days`` is 0."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.last_run_at: str | None = None
        self.last_report: Report | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="autoclip-retention")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def run_once(self) -> Report | None:
        from .config import load

        days = load().housekeeping.retention_days
        if days <= 0:
            return None
        report = await asyncio.to_thread(sweep, days, guards=live_guards())
        self.last_run_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.last_report = report
        return report

    async def _loop(self) -> None:
        await asyncio.sleep(STARTUP_DELAY_S)
        while True:
            try:
                await self.run_once()
            except Exception:
                log.exception("Retention sweep failed; trying again in an hour.")
            await asyncio.sleep(SWEEP_EVERY_S)

    def status(self) -> dict[str, Any]:
        return {
            "last_run_at": self.last_run_at,
            "last_report": asdict(self.last_report) if self.last_report else None,
        }


service = RetentionService()
