"""Autopilot: filters, waiting for the rewatch graph, dedupe, booking, API.

The Explore service and the job runner are fakes: no network, no jobs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip.app import create_app
from autoclip.autopilot import engine as engine_mod
from autoclip.autopilot import store
from autoclip.autopilot.adapters import Candidate, ClipInfo, Details, JobOutcome
from autoclip.autopilot.models import Autopilot, AutopilotIn, AutopilotSource
from autoclip.db import store as db_store
from autoclip.db.models import Clip, Export, Job, Source, new_id
from autoclip.publishing import posts
from autoclip.publishing import service as publishing
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _db(initialised_db: int) -> None:
    """Every test runs against a migrated temp database."""


NOW = datetime(2026, 11, 18, 8, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


def cand(vid: str, **kw) -> Candidate:
    base = {
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": f"Video {vid}",
        "views": 50_000,
        "duration_s": 1200.0,
        "upload_date": "2026-11-17",
    }
    base.update(kw)
    return Candidate(video_id=vid, **base)


class FakeDiscovery:
    def __init__(self) -> None:
        self.listing: dict[str, list[Candidate]] = {}
        self.info: dict[str, Details] = {}
        self.list_calls: list[str] = []
        self.detail_calls: list[str] = []

    def list_source(self, source: AutopilotSource, *, max_age_days: int, limit: int):
        self.list_calls.append(source.value)
        return list(self.listing.get(source.value, []))

    def details(self, url: str) -> Details:
        self.detail_calls.append(url)
        vid = url.rsplit("=", 1)[-1]
        return self.info.get(
            vid,
            Details(title=f"Video {vid}", views=50_000, duration_s=1200, upload_date="2026-11-17"),
        )


class FakeRunner:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.outcomes: dict[str, JobOutcome] = {}

    def start(self, url: str, *, max_clips: int, title: str = "") -> str:
        self.started.append(url)
        job_id = f"job{len(self.started)}"
        self.outcomes[job_id] = JobOutcome("running")
        return job_id

    def outcome(self, job_id: str) -> JobOutcome:
        return self.outcomes.get(job_id, JobOutcome("missing", "gone"))


def real_clip(tmp_path: Path, score: int, title: str = "Clip") -> str:
    """A DB clip with an exported file (so posts can be created for it)."""
    source = db_store.create_source(
        Source(id=new_id(), type="youtube", path=str(tmp_path / "s.mp4"), title="Src")
    )
    job = db_store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
    clip = db_store.create_clip(
        Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30, title=title, score=score)
    )
    path = tmp_path / f"{clip.id}.mp4"
    path.write_bytes(b"\x00" * 32)
    db_store.create_export(Export(id=new_id(), clip_id=clip.id, path=str(path)))
    return clip.id


def make_rule(**kw) -> Autopilot:
    data = {
        "name": "Podcasts",
        "sources": [{"kind": "channel", "value": "@chan"}],
        "slots": ["12:00", "18:00"],
        "timezone": "UTC",
        "platforms": ["youtube"],
        "approve_before_posting": False,
        "heatmap_wait_hours": 6,
        "clips_per_video": 2,
        "rights_confirmed": True,
    }
    data.update(kw)
    return store.create_autopilot(AutopilotIn(**data))


def make_engine(clock: Clock, disc: FakeDiscovery, runner: FakeRunner):
    return engine_mod.AutopilotEngine(
        discovery=disc, runner=runner, clock=clock, on_booked=lambda: None
    )


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Pure decisions
# --------------------------------------------------------------------------


class TestFilters:
    def rule(self, **filters) -> Autopilot:
        f = {"min_views": 10_000, "min_minutes": 5, "max_minutes": 60, "max_age_days": 7}
        f.update(filters)
        return Autopilot(
            name="x",
            sources=[{"kind": "topic", "value": "chess"}],
            filters=f,
            rights_confirmed=True,
        )

    def test_listing_skips_obvious_misses(self) -> None:
        r = self.rule()
        assert engine_mod.judge_listing(cand("a", is_live=True), r, NOW) == (
            "skipped",
            "A live stream",
        )
        assert engine_mod.judge_listing(cand("a", is_short=True), r, NOW)[1] == "A Short"
        assert "shorter" in engine_mod.judge_listing(cand("a", duration_s=120), r, NOW)[1]
        assert "longer" in engine_mod.judge_listing(cand("a", duration_s=7200), r, NOW)[1]
        old = cand("a", upload_date="2026-10-01")
        assert engine_mod.judge_listing(old, r, NOW) == ("skipped", "Older than 7 days")
        assert engine_mod.judge_listing(cand("a"), r, NOW)[0] == "waiting"
        # unknown duration/date (flat listings): wait and look closer
        unknown = cand("a", duration_s=None, upload_date=None, views=None)
        assert engine_mod.judge_listing(unknown, r, NOW)[0] == "waiting"

    def test_details_wait_for_views_and_rewatch_graph(self) -> None:
        r = self.rule()
        r.heatmap_wait_hours = 12
        video = {"first_seen": "2026-11-18T06:00:00Z", "language": ""}
        few = Details(views=900, duration_s=1200, upload_date="2026-11-17")
        status, reason = engine_mod.judge_details(video, few, r, NOW)
        assert status == "waiting" and "900 views" in reason
        graph = Details(views=20_000, duration_s=1200, has_heatmap=True)
        assert engine_mod.judge_details(video, graph, r, NOW) == (
            "ready",
            "Most replayed graph found",
        )
        no_graph = Details(views=20_000, duration_s=1200)
        status, reason = engine_mod.judge_details(video, no_graph, r, NOW)
        assert status == "waiting" and "Most replayed" in reason
        later = NOW + timedelta(hours=11)
        status, reason = engine_mod.judge_details(video, no_graph, r, later)
        assert status == "ready" and "audio and comments" in reason

    def test_details_language_and_age(self) -> None:
        r = self.rule()
        video = {"first_seen": "2026-11-18T06:00:00Z", "language": "en"}
        german = Details(views=20_000, duration_s=1200, language="de", has_heatmap=True)
        assert engine_mod.judge_details(video, german, r, NOW)[0] == "skipped"
        en_us = Details(views=20_000, duration_s=1200, language="en-us", has_heatmap=True)
        assert engine_mod.judge_details(video, en_us, r, NOW)[0] == "ready"
        old = Details(views=20_000, duration_s=1200, upload_date="2026-10-01")
        assert engine_mod.judge_details(video, old, r, NOW)[0] == "skipped"

    def test_rule_validation(self) -> None:
        with pytest.raises(ValueError):
            AutopilotIn(name="x", sources=[], rights_confirmed=True)
        with pytest.raises(ValueError):
            AutopilotIn(name="x", sources=[{"kind": "topic", "value": "a"}], slots=["25:00"])
        with pytest.raises(ValueError):
            AutopilotIn(name="x", sources=[{"kind": "topic", "value": "a"}], timezone="Mars/Base")
        with pytest.raises(ValueError):
            AutopilotIn(
                name="x",
                sources=[{"kind": "topic", "value": "a"}],
                filters={"min_minutes": 50, "max_minutes": 10},
            )
        ok = AutopilotIn(
            name="x", sources=[{"kind": "topic", "value": "a", "country": "de"}], slots=["9:5"]
        )
        assert ok.slots == ["09:05"] and ok.sources[0].wanted_language == "de"


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class TestEngine:
    def test_scan_dedupes_and_waits_for_the_graph(self) -> None:
        rule = make_rule()
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        disc.listing["@chan"] = [cand("v1"), cand("v1"), cand("short", is_short=True)]
        eng = make_engine(clock, disc, runner)
        run(eng.run_once())
        state = store.get_state(rule.id)
        assert set(state["videos"]) == {"v1", "short"}
        assert state["videos"]["short"]["status"] == "skipped"
        assert state["videos"]["v1"]["status"] == "waiting"  # no graph yet
        assert runner.started == []
        assert disc.detail_calls == ["https://www.youtube.com/watch?v=v1"]
        # Within the hour: no new scan, nothing peeked again.
        clock.advance(minutes=20)
        run(eng.run_once())
        assert disc.list_calls == ["@chan"] and len(disc.detail_calls) == 1
        # Next hour: scanned again, but v1 is not re-added; peeked again.
        clock.advance(hours=1)
        run(eng.run_once())
        assert len(disc.list_calls) == 2 and len(disc.detail_calls) == 2
        assert store.get_state(rule.id)["videos"]["v1"]["status"] == "waiting"
        # After heatmap_wait_hours it is clipped anyway.
        clock.advance(hours=6)
        run(eng.run_once())
        video = store.get_state(rule.id)["videos"]["v1"]
        assert video["status"] == "clipping" and "AI picks" in video["reason"]
        assert runner.started == ["https://www.youtube.com/watch?v=v1"]

    def test_heatmap_video_is_preferred_and_one_job_at_a_time(self) -> None:
        rule = make_rule()
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        disc.listing["@chan"] = [cand("big", views=900_000), cand("graph", views=60_000)]
        disc.info["big"] = Details(views=900_000, duration_s=1200)
        disc.info["graph"] = Details(views=60_000, duration_s=1200, has_heatmap=True)
        eng = make_engine(clock, disc, runner)
        run(eng.run_once())
        assert runner.started == ["https://www.youtube.com/watch?v=graph"]
        clock.advance(hours=10)  # "big" is ready now too, but one job at a time
        run(eng.run_once())
        assert len(runner.started) == 1
        assert store.get_state(rule.id)["videos"]["big"]["status"] == "ready"

    def test_harvest_then_book_into_next_free_slots(self, tmp_path: Path) -> None:
        rule = make_rule(platforms=["youtube", "instagram"])
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        disc.listing["@chan"] = [cand("v1")]
        disc.info["v1"] = Details(views=99_000, duration_s=1500, has_heatmap=True)
        eng = make_engine(clock, disc, runner)
        run(eng.run_once())
        ids = [real_clip(tmp_path, 90, "Best"), real_clip(tmp_path, 70), real_clip(tmp_path, 50)]
        runner.outcomes["job1"] = JobOutcome(
            "done",
            clips=[
                ClipInfo(ids[2], 50, "c", True),
                ClipInfo(ids[0], 90, "Best", True),
                ClipInfo(ids[1], 70, "b", True),
            ],
        )
        clock.advance(minutes=15)
        run(eng.run_once())
        state = store.get_state(rule.id)
        assert state["videos"]["v1"]["status"] == "done"
        booked = {c["clip_id"]: c for c in state["clips"]}
        assert set(booked) == set(ids[:2])  # the job's best clips_per_video
        assert booked[ids[0]]["slot_key"] == "2026-11-18T12:00:00Z"  # best clip first
        assert booked[ids[1]]["slot_key"] == "2026-11-18T18:00:00Z"
        ap_posts = posts.list_posts(autopilot_id=rule.id)
        assert len(ap_posts) == 4
        assert {p.platform for p in ap_posts if p.clip_id == ids[0]} == {"youtube", "instagram"}
        assert all(p.status == "queued" for p in ap_posts)
        # Booking again books nothing twice.
        clock.advance(minutes=15)
        run(eng.run_once())
        assert len(posts.list_posts(autopilot_id=rule.id)) == 4

    def test_approval_holds_clips_in_review(self, tmp_path: Path) -> None:
        rule = make_rule(approve_before_posting=True)
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        disc.listing["@chan"] = [cand("v1")]
        disc.info["v1"] = Details(views=99_000, duration_s=1500, has_heatmap=True)
        eng = make_engine(clock, disc, runner)
        run(eng.run_once())
        cid = real_clip(tmp_path, 80)
        runner.outcomes["job1"] = JobOutcome("done", clips=[ClipInfo(cid, 80, "c", True)])
        run(eng.run_once())
        assert store.get_state(rule.id)["clips"][0]["status"] == "review"
        assert posts.list_posts(autopilot_id=rule.id) == []
        eng.set_clip(rule, cid, "approve")
        run(eng.run_once())
        clip = store.get_state(rule.id)["clips"][0]
        assert clip["status"] == "booked" and clip["slot_key"] == "2026-11-18T12:00:00Z"
        # Rejecting a booked clip cancels its unsent posts and frees the slot.
        eng.set_clip(rule, cid, "reject")
        assert all(p.status == "canceled" for p in posts.list_posts(autopilot_id=rule.id))
        assert store.get_state(rule.id)["clips"][0]["status"] == "rejected"

    def test_cancel_on_schedule_frees_the_slot_for_the_next_clip(self, tmp_path: Path) -> None:
        rule = make_rule(slots=["12:00"])
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        eng = make_engine(clock, disc, runner)
        a, b = real_clip(tmp_path, 90), real_clip(tmp_path, 80)
        with store.edit_state(rule.id) as st:
            st["last_scan_at"] = "2026-11-18T07:59:00Z"
            for cid, score in ((a, 90), (b, 80)):
                st["clips"].append(
                    {"clip_id": cid, "score": score, "status": "ready", "post_ids": []}
                )
        run(eng.run_once())
        st = store.get_state(rule.id)
        assert [c["slot_key"] for c in st["clips"]] == [
            "2026-11-18T12:00:00Z",
            "2026-11-19T12:00:00Z",
        ]
        first_post = next(p for p in posts.list_posts(autopilot_id=rule.id) if p.clip_id == a)
        posts.update_post(first_post.id, status="canceled")
        run(eng.run_once())
        st = store.get_state(rule.id)
        assert st["clips"][0]["status"] == "rejected"
        # b keeps its booking (nothing moves); the freed slot stays free for new clips
        assert st["clips"][1]["status"] == "booked"

    def test_retryable_failure_keeps_the_booking(self, tmp_path: Path) -> None:
        rule = make_rule(slots=["12:00"])
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        eng = make_engine(clock, disc, runner)
        cid = real_clip(tmp_path, 90)
        with store.edit_state(rule.id) as st:
            st["last_scan_at"] = "2026-11-18T07:59:00Z"
            st["clips"].append({"clip_id": cid, "score": 90, "status": "ready", "post_ids": []})
        run(eng.run_once())
        (post,) = posts.list_posts(autopilot_id=rule.id)
        posts.update_post(post.id, status="failed", attempts=1)
        run(eng.run_once())
        assert store.get_state(rule.id)["clips"][0]["status"] == "booked"
        posts.update_post(post.id, attempts=3)
        run(eng.run_once())
        assert store.get_state(rule.id)["clips"][0]["status"] == "failed"

    def test_hand_posted_clip_is_not_posted_twice(self, tmp_path: Path) -> None:
        rule = make_rule(platforms=["youtube", "instagram"])
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        eng = make_engine(clock, disc, runner)
        cid = real_clip(tmp_path, 90)
        publishing.create_posts(cid, ["youtube"], now=NOW)  # posted by hand
        with store.edit_state(rule.id) as st:
            st["last_scan_at"] = "2026-11-18T07:59:00Z"
            st["clips"].append({"clip_id": cid, "score": 90, "status": "ready", "post_ids": []})
        run(eng.run_once())
        mine = posts.list_posts(autopilot_id=rule.id)
        assert [p.platform for p in mine] == ["instagram"]
        assert len(posts.list_posts(clip_id=cid, platform="youtube")) == 1

    def test_paused_autopilot_does_not_scan_or_book(self, tmp_path: Path) -> None:
        rule = make_rule()
        store.update_autopilot(rule.id, enabled=False)
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        disc.listing["@chan"] = [cand("v1")]
        cid = real_clip(tmp_path, 90)
        with store.edit_state(rule.id) as st:
            st["clips"].append({"clip_id": cid, "score": 90, "status": "ready", "post_ids": []})
        run(make_engine(clock, disc, runner).run_once())
        assert disc.list_calls == [] and posts.list_posts(autopilot_id=rule.id) == []
        assert store.get_state(rule.id)["message"].startswith("Paused")

    def test_enough_stock_means_no_new_clipping(self, tmp_path: Path) -> None:
        rule = make_rule(slots=["12:00"], approve_before_posting=True)
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        disc.listing["@chan"] = [cand("v1")]
        disc.info["v1"] = Details(views=99_000, duration_s=1500, has_heatmap=True)
        with store.edit_state(rule.id) as st:
            for i in range(2):
                st["clips"].append({"clip_id": f"c{i}", "status": "review", "post_ids": []})
        run(make_engine(clock, disc, runner).run_once())
        assert runner.started == []
        assert store.get_state(rule.id)["videos"]["v1"]["status"] == "ready"

    def test_other_autopilots_videos_are_skipped(self) -> None:
        first = make_rule(name="A")
        second = make_rule(name="B")
        with store.edit_state(first.id) as st:
            st["videos"]["v1"] = {"id": "v1", "status": "done"}
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()
        disc.listing["@chan"] = [cand("v1")]
        run(make_engine(clock, disc, runner).run_once(force_scan={second.id}))
        video = store.get_state(second.id)["videos"]["v1"]
        assert video["status"] == "skipped" and "“A”" in video["reason"]

    def test_failed_job_and_source_errors_are_recorded(self) -> None:
        rule = make_rule()
        clock, disc, runner = Clock(), FakeDiscovery(), FakeRunner()

        def boom(*_a, **_k):
            raise RuntimeError("yt-dlp says no")

        disc.list_source = boom  # type: ignore[method-assign]
        run(make_engine(clock, disc, runner).run_once())
        state = store.get_state(rule.id)
        assert "yt-dlp says no" in state["last_error"]
        assert state["runs"][-1]["errors"]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


class TestApi:
    payload = {
        "name": "Chess",
        "sources": [{"kind": "topic", "value": "chess", "country": "US"}],
        "slots": ["12:00"],
        "platforms": ["youtube"],
        "rights_confirmed": True,
    }

    def test_crud_pause_resume(self, client: TestClient) -> None:
        missing_rights = {**self.payload, "rights_confirmed": False}
        assert client.post("/api/autopilots", json=missing_rights).status_code == 400
        made = client.post("/api/autopilots", json=self.payload)
        assert made.status_code == 201, made.text
        ap = made.json()
        assert ap["enabled"] and ap["slots"] and ap["counts"]["stock"] == 0
        assert len(client.get("/api/autopilots").json()["autopilots"]) == 1
        assert client.post(f"/api/autopilots/{ap['id']}/pause").json()["enabled"] is False
        assert client.post(f"/api/autopilots/{ap['id']}/resume").json()["enabled"] is True
        edited = client.put(f"/api/autopilots/{ap['id']}", json={**self.payload, "name": "C2"})
        assert edited.json()["name"] == "C2" and edited.json()["created_at"] == ap["created_at"]
        assert client.post(f"/api/autopilots/{ap['id']}/run").json()["queued"]
        detail = client.get(f"/api/autopilots/{ap['id']}").json()
        assert {"runs", "videos", "clips", "posts", "plan"} <= set(detail)
        assert client.delete(f"/api/autopilots/{ap['id']}").status_code == 204
        assert client.get(f"/api/autopilots/{ap['id']}").status_code == 404

    def test_clip_actions(self, client: TestClient) -> None:
        ap = client.post("/api/autopilots", json=self.payload).json()
        with store.edit_state(ap["id"]) as st:
            st["clips"].append({"clip_id": "c1", "status": "review", "post_ids": []})
        r = client.post(f"/api/autopilots/{ap['id']}/clips/c1", json={"action": "approve"})
        assert r.status_code == 200 and r.json()["status"] == "ready"
        again = client.post(f"/api/autopilots/{ap['id']}/clips/c1", json={"action": "approve"})
        assert again.status_code == 409
        assert (
            client.post(f"/api/autopilots/{ap['id']}/clips/zz", json={"action": "reject"})
        ).status_code == 404
