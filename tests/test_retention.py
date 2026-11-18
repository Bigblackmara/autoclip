"""Retention: which old work the hourly sweep deletes, and what it keeps."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from autoclip import config, paths, retention
from autoclip.db import connection, store
from autoclip.db.models import Clip, Export, Job, Source, new_id

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
OLD = (NOW - timedelta(days=10)).isoformat(timespec="seconds")
NEW = (NOW - timedelta(days=1)).isoformat(timespec="seconds")


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def make_source(created_at: str) -> Source:
    source_id = new_id()
    media = touch(paths.source_media_dir(source_id) / "source.mp4")
    return store.create_source(
        Source(id=source_id, type="upload", path=str(media), title="s", created_at=created_at)
    )


def make_job(source: Source, created_at: str, status: str = "done") -> Job:
    job = store.create_job(
        Job(id=new_id(), source_id=source.id, status=status, created_at=created_at)
    )
    touch(paths.job_work_dir(job.id) / "audio.wav")
    clip = Clip(id=new_id(), job_id=job.id, start_s=0.0, end_s=10.0)
    store.replace_clips(job.id, [clip])
    export_file = touch(paths.exports_dir() / job.id / "01-clip.mp4")
    store.create_export(Export(id=new_id(), clip_id=clip.id, path=str(export_file)))
    return job


def make_stream(created_at: str, status: str = "done") -> str:
    stream_id = new_id()
    with connection() as conn:
        conn.execute(
            "INSERT INTO gta_streams (id, url, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (stream_id, "https://twitch.tv/videos/1", status, created_at, created_at),
        )
    touch(paths.root() / "gta" / stream_id / "audio.m4a")
    return stream_id


def make_gta_clip(stream_id: str, created_at: str, status: str = "ready") -> str:
    clip_id = new_id()
    rendered = touch(paths.exports_dir() / "gta" / f"{clip_id}.mp4")
    with connection() as conn:
        conn.execute(
            "INSERT INTO gta_clips (id, stream_id, status, start_s, end_s, path, "
            "created_at, updated_at) VALUES (?, ?, ?, 0, 30, ?, ?, ?)",
            (clip_id, stream_id, status, str(rendered), created_at, created_at),
        )
    return clip_id


def make_upload(clip_id: str, status: str, attempts: int = 0) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO gta_uploads (id, clip_id, platform, status, attempts, created_at, "
            "updated_at) VALUES (?, ?, 'youtube', ?, ?, ?, ?)",
            (new_id(), clip_id, status, attempts, OLD, OLD),
        )


def ids(table: str) -> set[str]:
    with connection() as conn:
        return {row["id"] for row in conn.execute(f"SELECT id FROM {table}").fetchall()}


@pytest.fixture(autouse=True)
def db_ready(initialised_db: int) -> None:
    paths.ensure_layout()


def test_off_by_default_and_at_zero() -> None:
    make_job(make_source(OLD), OLD)

    assert config.Settings().housekeeping.retention_days == 0
    assert retention.select(0, now=NOW).empty


def test_old_finished_work_is_deleted_rows_and_files() -> None:
    source = make_source(OLD)
    job = make_job(source, OLD)

    report = retention.sweep(7, now=NOW)

    assert (report.jobs, report.sources) == (1, 1)
    assert ids("sources") == set() and ids("jobs") == set()
    assert ids("clips") == set() and ids("exports") == set()
    assert not paths.source_media_dir(source.id).exists()
    assert not paths.job_work_dir(job.id).exists()
    assert not (paths.exports_dir() / job.id).exists()


def test_young_work_is_kept() -> None:
    source = make_source(NEW)
    make_job(source, NEW)

    assert retention.select(7, now=NOW).empty


@pytest.mark.parametrize("status", ["queued", "running"])
def test_processing_jobs_and_their_source_are_kept(status: str) -> None:
    source = make_source(OLD)
    busy = make_job(source, OLD, status=status)

    plan = retention.select(7, now=NOW)

    assert busy.id not in plan.jobs
    assert source.id not in plan.sources


def test_the_job_the_worker_is_running_is_kept_whatever_its_row_says() -> None:
    source = make_source(OLD)
    job = make_job(source, OLD, status="done")

    plan = retention.select(7, now=NOW, guards=retention.Guards(running_job_id=job.id))

    assert plan.jobs == [] and plan.sources == []


def test_an_old_source_with_a_new_job_keeps_its_video() -> None:
    source = make_source(OLD)
    old_job = make_job(source, OLD)
    new_job = make_job(source, NEW)

    retention.sweep(7, now=NOW)

    assert ids("jobs") == {new_job.id}
    assert ids("sources") == {source.id}
    assert paths.source_media_dir(source.id).exists()
    assert not paths.job_work_dir(old_job.id).exists()


def test_a_job_requeued_after_selection_survives_the_delete() -> None:
    source = make_source(OLD)
    job = make_job(source, OLD, status="failed")
    plan = retention.select(7, now=NOW)
    store.update_job(job.id, status="queued")  # the user pressed Retry

    retention.apply(plan)

    assert ids("jobs") == {job.id}
    assert ids("sources") == {source.id}
    assert paths.job_work_dir(job.id).exists()


def test_files_outside_the_home_are_never_touched(tmp_path: Path) -> None:
    outside = touch(tmp_path / "elsewhere" / "clip.mp4")
    stream = make_stream(OLD)
    clip = make_gta_clip(stream, OLD)
    with connection() as conn:
        conn.execute("UPDATE gta_clips SET path = ? WHERE id = ?", (str(outside), clip))

    retention.sweep(7, now=NOW)

    assert ids("gta_clips") == set()
    assert outside.exists()


def test_gta_old_stream_and_clips_are_deleted() -> None:
    stream = make_stream(OLD)
    clip = make_gta_clip(stream, OLD)
    posted = make_gta_clip(stream, OLD)
    make_upload(posted, "posted")

    report = retention.sweep(7, now=NOW)

    assert (report.gta_clips, report.gta_streams) == (2, 1)
    assert ids("gta_streams") == set() and ids("gta_uploads") == set()
    assert not (paths.exports_dir() / "gta" / f"{clip}.mp4").exists()
    assert not (paths.root() / "gta" / stream).exists()


@pytest.mark.parametrize(
    ("status", "attempts"),
    [("scheduled", 0), ("uploading", 0), ("checking", 0), ("interrupted", 0), ("failed", 1)],
)
def test_gta_clips_with_a_pending_upload_are_kept(status: str, attempts: int) -> None:
    stream = make_stream(OLD)
    pending = make_gta_clip(stream, OLD)
    make_upload(pending, status, attempts)
    other = make_gta_clip(stream, OLD)

    retention.sweep(7, now=NOW)

    assert ids("gta_clips") == {pending}
    assert ids("gta_streams") == {stream}  # still holds a clip
    assert (paths.exports_dir() / "gta" / f"{pending}.mp4").exists()
    assert not (paths.exports_dir() / "gta" / f"{other}.mp4").exists()


def test_gta_failed_upload_out_of_retries_no_longer_protects_the_clip() -> None:
    stream = make_stream(OLD)
    clip = make_gta_clip(stream, OLD)
    make_upload(clip, "failed", attempts=3)

    assert clip in retention.select(7, now=NOW).gta_clips


@pytest.mark.parametrize("status", ["judging", "analyzed", "live_watching", "queued"])
def test_gta_streams_still_processing_are_kept(status: str) -> None:
    stream = make_stream(OLD, status=status)
    make_gta_clip(stream, OLD)

    plan = retention.select(7, now=NOW)

    assert plan.gta_clips == [] and plan.gta_streams == []


def test_gta_rendering_clip_and_guarded_streams_are_kept() -> None:
    rendering_stream = make_stream(OLD)
    rendering = make_gta_clip(rendering_stream, OLD, status="rendering")
    live_stream = make_stream(OLD, status="live_ended")
    make_gta_clip(live_stream, OLD)
    busy_stream = make_stream(OLD)
    make_gta_clip(busy_stream, OLD)
    uploading_stream = make_stream(OLD)
    uploading = make_gta_clip(uploading_stream, OLD)

    guards = retention.Guards(
        gta_busy_keys={f"stream:{busy_stream}"},
        gta_uploading_clip_ids={uploading},
        gta_live_running=lambda stream_id: stream_id == live_stream,
    )
    plan = retention.select(7, now=NOW, guards=guards)

    assert rendering not in plan.gta_clips and uploading not in plan.gta_clips
    assert plan.gta_streams == []
    assert plan.gta_clips == []


def test_service_reads_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(OLD)
    make_job(source, OLD)
    settings = config.Settings()
    monkeypatch.setattr(retention, "live_guards", retention.Guards)

    service = retention.RetentionService()
    config.save(settings)
    assert asyncio.run(service.run_once()) is None  # 0 = off

    settings.housekeeping.retention_days = 1
    config.save(settings)
    report = asyncio.run(service.run_once())

    assert report is not None and report.sources == 1
    assert service.status()["last_report"]["sources"] == 1


def test_settings_api_round_trips_retention_days(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.app import ENV_NO_WORKER, create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv(ENV_NO_WORKER, "1")
    with TestClient(create_app()) as client:
        body = client.put("/api/settings", json={"housekeeping": {"retention_days": 14}}).json()
        assert body["housekeeping"]["retention_days"] == 14
        assert (
            client.put("/api/settings", json={"housekeeping": {"retention_days": -1}}).status_code
            == 400
        )
        status = client.get("/api/system/retention").json()
        assert status["retention_days"] == 14
