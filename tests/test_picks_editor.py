"""Picked / joined clips together with the clip editor (wave-2 integration)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip.api import picks as picks_api
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import export, picks, runner, segments
from autoclip.pipeline.reframe.croppath import centre_crop
from autoclip.pipeline.runner import JobWorkspace
from autoclip.pipeline.segments import Span
from autoclip.pipeline.transcript import Transcript, Word
from fastapi.testclient import TestClient


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list:
    requests: list = []

    def fake_export(request, *, work_dir, settings=None, on_progress=None, cancelled=None):
        requests.append(request)
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        request.destination.write_bytes(b"mp4")
        return request.destination

    monkeypatch.setattr(picks_api, "_launch", lambda work: work())
    monkeypatch.setattr(
        runner,
        "build_crop_path",
        lambda video, *, start_s, end_s, transcript=None, config=None: centre_crop(
            1920, 1080, end_s - start_s
        ),
    )
    monkeypatch.setattr(runner, "facecam_for_clip", lambda *a, **k: None)
    monkeypatch.setattr(export, "export_clip", fake_export)
    picks._tasks.clear()
    return requests


@pytest.fixture
def job(client: TestClient) -> Job:
    source = store.create_source(
        Source(
            id=new_id(),
            type="upload",
            path="C:/media/video.mp4",
            title="Podcast",
            duration_s=120.0,
            width=1920,
            height=1080,
            fps=30.0,
        )
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
    words = [
        Word(f"w{i}" + ("." if i % 4 == 3 else ""), round(i * 0.5, 3), round(i * 0.5 + 0.4, 3))
        for i in range(240)
    ]
    Transcript(words=words, language="en").save(JobWorkspace(job.id).transcript)
    return job


def test_add_clips_writes_the_editor_columns(job) -> None:
    clip = Clip(
        id=new_id(),
        job_id=job.id,
        start_s=1,
        end_s=5,
        rank=3,
        hook_text="Opening line",
        hashtags=["#x"],
        flash_forward=True,
        motion_seed=2,
    )

    store.add_clips([clip])

    (loaded,) = store.list_clips(job.id)
    assert loaded.hook_text == "Opening line" and loaded.headline == "Opening line"
    assert loaded.hashtags == ["#x"]
    assert loaded.flash_forward is True and loaded.motion_seed == 2
    assert loaded.variant_of is None and loaded.teaser is None


def test_picked_clips_open_on_their_first_line_not_the_placeholder_title(
    client, job, rendered
) -> None:
    response = client.post(
        f"/api/jobs/{job.id}/picked-clips", json={"ranges": [{"start_s": 4.0, "end_s": 12.0}]}
    )

    assert response.status_code == 202, response.text
    (clip,) = store.list_clips(job.id)
    assert clip.title.startswith("Your pick")
    assert clip.headline == "w8 w9 w10 w11."
    (request,) = rendered
    assert request.edit is not None and request.edit.hook_text == "w8 w9 w10 w11."


def test_range_to_joined_maps_only_ranges_inside_one_part() -> None:
    spans = [Span(10.0, 14.0), Span(30.0, 33.0)]

    assert segments.range_to_joined((31.0, 32.0), spans) == (5.0, 6.0)
    assert segments.range_to_joined((13.0, 31.0), spans) is None
    assert segments.range_to_joined(None, spans) is None


def test_flash_forward_on_a_joined_clip_uses_the_joined_timeline(job, rendered) -> None:
    workspace = JobWorkspace(job.id)
    spans = [Span(4.0, 12.0), Span(40.0, 52.0)]
    clip = Clip(
        id=new_id(),
        job_id=job.id,
        start_s=4.0,
        end_s=52.0,
        rank=1,
        start_word=8,
        end_word=103,
        title="Joined",
        # The tease line sits in the second part: 44-46 s in source time,
        # 12-14 s on the joined timeline.
        teaser_start_s=44.0,
        teaser_end_s=46.0,
        flash_forward=True,
    )
    store.add_clips([clip])
    segments.save(workspace.segments(clip.id), spans)
    task = picks.start_task(job.id, [clip.id])
    source = store.get_source(job.source_id)

    picks.render(task, job, source, [clip], framing="fill")

    assert task.state == "done", task.error
    (request,) = rendered
    assert request.spans == [(4.0, 12.0), (40.0, 52.0)]
    teaser = request.edit.teaser
    assert teaser is not None and request.edit.keeps[0] == teaser
    assert 11.5 <= teaser.start <= 12.1 and teaser.end <= 14.5
    # Captions stay on the joined timeline, within the clip.
    assert all(w.start >= 0 and w.end <= 20.0 + 1e-6 for w in request.words)


def test_a_variant_of_a_joined_clip_keeps_its_parts(client, job, rendered) -> None:
    response = client.post(
        f"/api/jobs/{job.id}/picked-clips",
        json={
            "ranges": [{"start_s": 4.0, "end_s": 12.0}, {"start_s": 40.0, "end_s": 50.0}],
            "join": True,
        },
    )
    (picked,) = response.json()["clips"]

    variant = client.post(f"/api/clips/{picked['id']}/variant", json={"export": False})

    assert variant.status_code == 201, variant.text
    parts = client.get(f"/api/clips/{variant.json()['id']}/segments").json()
    assert parts["spans"] == [list(s) for s in picked["spans"]]
