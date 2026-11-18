"""The picking API, run through the real app with reframe and export faked."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from autoclip import app as app_module
from autoclip.api import picks as picks_api
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import export, picks, runner, segments
from autoclip.pipeline.reframe.croppath import centre_crop
from autoclip.pipeline.runner import JobWorkspace
from autoclip.pipeline.signals import AudienceCurve
from autoclip.pipeline.transcript import Transcript, Word
from fastapi.testclient import TestClient


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run renders inline and record what would have been exported."""
    calls: dict = {"exports": [], "reframes": []}

    def fake_crop_path(video, *, start_s, end_s, transcript=None, config=None):
        calls["reframes"].append((start_s, end_s, config.framing if config else None))
        return centre_crop(1920, 1080, end_s - start_s)

    def fake_export(request, *, work_dir, settings=None, on_progress=None, cancelled=None):
        calls["exports"].append(request)
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        request.destination.write_bytes(b"mp4")
        return request.destination

    monkeypatch.setattr(picks_api, "_launch", lambda work: work())
    monkeypatch.setattr(runner, "build_crop_path", fake_crop_path)
    monkeypatch.setattr(runner, "facecam_for_clip", lambda *a, **k: None)
    monkeypatch.setattr(export, "export_clip", fake_export)
    picks._tasks.clear()
    return calls


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
    workspace = JobWorkspace(job.id)
    Transcript(words=words).save(workspace.transcript)
    store.replace_clips(
        job.id,
        [Clip(id=new_id(), job_id=job.id, start_s=60, end_s=80, rank=1, status="exported")],
    )
    return job


def post(client: TestClient, job_id: str, ranges, **body):
    return client.post(
        f"/api/jobs/{job_id}/picked-clips",
        json={"ranges": [{"start_s": a, "end_s": b} for a, b in ranges], **body},
    )


class TestPickedClips:
    def test_makes_one_clip_per_range(self, client, job, rendered) -> None:
        response = post(client, job.id, [(4.0, 12.0), (40.0, 50.0)])

        assert response.status_code == 202, response.text
        body = response.json()
        assert len(body["clips"]) == 2
        assert all(len(c["spans"]) == 1 for c in body["clips"])

        clips = {c.id: c for c in store.list_clips(job.id)}
        new = [clips[c["id"]] for c in body["clips"]]
        assert [c.rank for c in new] == [2, 3]
        assert all(c.status == "exported" for c in new)
        assert len(rendered["exports"]) == 2
        assert all(r.spans is None for r in rendered["exports"])
        # Ordinary clips render on source time.
        assert rendered["exports"][0].start_s == new[0].start_s

        status = client.get(f"/api/jobs/{job.id}/picked-clips").json()
        assert status["state"] == "done"
        assert status["progress"] == 1.0

    def test_join_renders_one_multi_part_clip(self, client, job, rendered) -> None:
        response = post(client, job.id, [(40.0, 50.0), (4.0, 12.0)], join=True)

        assert response.status_code == 202, response.text
        (clip_out,) = response.json()["clips"]
        assert len(clip_out["spans"]) == 2
        assert clip_out["spans"][0][0] < clip_out["spans"][1][0]

        (request,) = rendered["exports"]
        spans = [segments.Span(a, b) for a, b in request.spans]
        total = segments.total(spans)
        assert request.start_s == 0.0
        assert request.end_s == pytest.approx(total)
        assert request.crop_path.duration_s == pytest.approx(total)
        # Each part was framed on its own.
        assert [(a, b) for a, b, _ in rendered["reframes"]] == [(s.start, s.end) for s in spans]
        # Captions are on the joined timeline: nothing from the gap, in order.
        starts = [w.start for w in request.words]
        assert starts == sorted(starts)
        assert all(w.start >= 0.0 and w.end <= total + 1e-6 for w in request.words)
        assert not any(w.text in {"w40", "w60"} for w in request.words)  # 20 s / 30 s: gap

        sidecar = client.get(f"/api/clips/{clip_out['id']}/segments").json()
        assert sidecar["spans"] == [list(s) for s in request.spans]
        assert sidecar["duration_s"] == pytest.approx(total, abs=1e-3)

        words = client.get(f"/api/clips/{clip_out['id']}/words").json()
        assert all(
            any(a <= (w["start"] + w["end"]) / 2 < b for a, b in request.spans) for w in words
        )

    def test_framing_reaches_the_planner(self, client, job, rendered) -> None:
        post(client, job.id, [(4.0, 12.0)], framing="split")

        assert rendered["reframes"][0][2] == "split"

    def test_invalid_ranges_are_rejected(self, client, job, rendered) -> None:
        response = post(client, job.id, [(10.0, 10.4)])

        assert response.status_code == 400
        assert "at least" in response.json()["detail"]
        assert len(store.list_clips(job.id)) == 1

    def test_ranges_past_the_video_are_rejected(self, client, job, rendered) -> None:
        assert post(client, job.id, [(500.0, 520.0)]).status_code == 400

    def test_empty_body_is_a_validation_error(self, client, job, rendered) -> None:
        assert post(client, job.id, []).status_code == 422

    def test_unknown_job(self, client, rendered) -> None:
        assert post(client, "nope", [(1, 5)]).status_code == 404

    def test_unfinished_job(self, client, job, rendered) -> None:
        store.update_job(job.id, status="running")

        assert post(client, job.id, [(4.0, 12.0)]).status_code == 409

    def test_one_render_at_a_time(self, client, job, rendered, monkeypatch) -> None:
        monkeypatch.setattr(picks_api, "_launch", lambda work: None)  # never finishes
        assert post(client, job.id, [(4.0, 12.0)]).status_code == 202

        assert post(client, job.id, [(40.0, 50.0)]).status_code == 409

    def test_a_failed_render_is_reported(self, client, job, rendered, monkeypatch) -> None:
        def boom(*a, **k):
            raise export.ExportError("ffmpeg fell over")

        monkeypatch.setattr(export, "export_clip", boom)

        assert post(client, job.id, [(4.0, 12.0)]).status_code == 202
        status = client.get(f"/api/jobs/{job.id}/picked-clips").json()
        assert status["state"] == "failed"
        assert "ffmpeg fell over" in status["error"]

    def test_status_404_before_any_pick(self, client, job, rendered) -> None:
        assert client.get(f"/api/jobs/{job.id}/picked-clips").status_code == 404


class TestAudience:
    def test_serves_the_cached_curve_and_clip_bands(self, client, job) -> None:
        curve = AudienceCurve(
            curve=np.linspace(0, 1, 120, dtype=np.float32),
            kind="audience",
            sources=["heatmap", "energy"],
            components={
                "heatmap": np.ones(120, dtype=np.float32),
                "energy": np.zeros(120, dtype=np.float32),
            },
        )
        curve.save(JobWorkspace(job.id).audience)

        body = client.get(f"/api/jobs/{job.id}/audience").json()

        assert body["kind"] == "audience"
        assert body["step_s"] == 1
        assert len(body["curve"]) == 120
        assert set(body["components"]) == {"heatmap", "energy"}
        assert body["clips"][0]["start_s"] == 60
        assert body["has_transcript"] is True

    def test_long_curves_are_max_pooled(self, client, job) -> None:
        values = np.zeros(10_000, dtype=np.float32)
        values[5_001] = 1.0
        AudienceCurve(curve=values, kind="energy", sources=["energy"]).save(
            JobWorkspace(job.id).audience
        )

        body = client.get(f"/api/jobs/{job.id}/audience").json()

        assert len(body["curve"]) <= picks_api.MAX_GRAPH_POINTS
        assert max(body["curve"]) == 1.0  # the peak survives

    def test_no_signals_gives_an_empty_curve(self, client, job) -> None:
        body = client.get(f"/api/jobs/{job.id}/audience").json()

        assert body["kind"] == "none"
        assert body["curve"] == []
        assert body["duration_s"] == 120.0

    def test_unknown_job(self, client) -> None:
        assert client.get("/api/jobs/nope/audience").status_code == 404


def test_add_clips_keeps_existing_rows(client, job) -> None:
    before = store.list_clips(job.id)

    store.add_clips([Clip(id=new_id(), job_id=job.id, start_s=1, end_s=5, rank=9)])

    after = store.list_clips(job.id)
    assert len(after) == len(before) + 1
