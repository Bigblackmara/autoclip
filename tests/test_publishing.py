"""Normal-mode posting: JSON records, no double posts, the poster, Schedule, API.

Everything runs against a temp AutoClip home with fake uploaders: no
network, no real YouTube or Zernio calls.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Export, Job, Source, new_id
from autoclip.gta import store as gta_store
from autoclip.gta.zernio import ZernioResult
from autoclip.live.uploads.tokens import UploadApiError, UploadMeta, UploadResult
from autoclip.publishing import core, posts, remote, service
from autoclip.publishing import poster as poster_mod
from autoclip.publishing.jsonstore import JsonFile, write_atomic
from autoclip.publishing.posts import DoublePostError, StateChanged
from autoclip.publishing.settings import PublishSettings
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _db(initialised_db: int) -> None:
    """Every test runs against a migrated temp database."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

NOW = datetime(2026, 11, 18, 10, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


def make_clip(tmp_path: Path, *, title: str = "He did WHAT", exported: bool = True) -> Clip:
    source = store.create_source(
        Source(id=new_id(), type="upload", path=str(tmp_path / "src.mp4"), title="Source video")
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
    clip = store.create_clip(
        Clip(
            id=new_id(),
            job_id=job.id,
            start_s=10,
            end_s=40,
            title=title,
            score=80,
            description="The moment nobody expected",
            hashtags=["#funny", "#wow"],
            status="exported",
        )
    )
    if exported:
        path = tmp_path / f"{clip.id}.mp4"
        path.write_bytes(b"\x00" * 64)
        store.create_export(Export(id=new_id(), clip_id=clip.id, path=str(path)))
    return clip


class FakeUploader:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[Path, UploadMeta]] = []
        self.error = error

    async def __call__(self, path: Path, meta: UploadMeta) -> UploadResult:
        self.calls.append((path, meta))
        if self.error is not None:
            raise self.error
        vid = f"vid{len(self.calls)}"
        return UploadResult(platform="youtube", media_id=vid, url=f"https://youtu.be/{vid}")


class FakeZernio:
    def __init__(self, kind: str = "scheduled") -> None:
        self.kind = kind
        self.sent: list[tuple[str, datetime]] = []
        self.checks: list[tuple[str, str]] = []
        self.check_kind = "posted"
        self.deleted: list[str] = []

    def schedule_reel(self, api_key, account, path, caption, publish_at, **kw):
        self.sent.append((caption, publish_at))
        kw["on_media_url"]("https://cdn.zernio/m1.mp4")
        if self.kind in {"scheduled", "posted"}:
            kw["on_post_id"]("zp1")
            return ZernioResult(
                kind=self.kind, post_id="zp1", media_url="https://cdn.zernio/m1.mp4"
            )
        return ZernioResult(
            kind=self.kind, message="unclear", media_url="https://cdn.zernio/m1.mp4"
        )

    def check_post(self, api_key, post_id, media_url, sent_at=None):
        self.checks.append((post_id, media_url))
        return ZernioResult(kind=self.check_kind, post_id="zp1", post_url="https://ig/p/1")

    def delete_post(self, api_key, post_id):
        self.deleted.append(post_id)
        return remote.RemoteResult(True, "Removed from Zernio; it will not be posted.")


def make_poster(clock: Clock, uploader=None, ig=None, ready: str = "", lead: int = 120):
    return poster_mod.Poster(
        uploader=uploader or FakeUploader(),
        instagram=ig or FakeZernio(),
        clock=clock,
        ready_check=lambda: ready,
        instagram_ready_check=lambda: ready,
        settings_loader=lambda: PublishSettings(lead_minutes=lead),
        api_key=lambda: "zk_test",
    )


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# JSON store
# --------------------------------------------------------------------------


class TestJsonStore:
    def test_atomic_write_and_read(self, autoclip_home: Path) -> None:
        f = JsonFile("publishing", "t.json", dict)
        with f.edit() as data:
            data["a"] = 1
        assert f.read() == {"a": 1}
        assert json.loads(f.path.read_text(encoding="utf-8")) == {"a": 1}
        assert not list(f.path.parent.glob(".t.json.*.tmp"))

    def test_failed_edit_writes_nothing(self, autoclip_home: Path) -> None:
        f = JsonFile("publishing", "t.json", dict)
        with f.edit() as data:
            data["a"] = 1
        with pytest.raises(RuntimeError), f.edit() as data:
            data["a"] = 2
            raise RuntimeError("boom")
        assert f.read() == {"a": 1}

    def test_corrupt_file_is_kept_aside(self, autoclip_home: Path) -> None:
        f = JsonFile("publishing", "t.json", dict)
        f.path.parent.mkdir(parents=True, exist_ok=True)
        f.path.write_text("{not json", encoding="utf-8")
        assert f.read() == {}
        assert (f.path.parent / "t.json.corrupt-1").exists()

    def test_concurrent_edits_lose_nothing(self, autoclip_home: Path) -> None:
        f = JsonFile("publishing", "count.json", lambda: {"n": 0})

        def bump() -> None:
            for _ in range(25):
                with f.edit() as data:
                    data["n"] += 1

        threads = [threading.Thread(target=bump) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert f.read()["n"] == 100

    def test_write_atomic_replaces(self, tmp_path: Path) -> None:
        target = tmp_path / "x.json"
        write_atomic(target, [1])
        write_atomic(target, [2])
        assert json.loads(target.read_text()) == [2]


# --------------------------------------------------------------------------
# Records + never twice
# --------------------------------------------------------------------------


class TestPostRecords:
    def test_one_active_post_per_clip_and_platform(self) -> None:
        first = posts.create_post("c1", "youtube", title="t")
        with pytest.raises(DoublePostError, match="never posted twice"):
            posts.create_post("c1", "youtube")
        posts.create_post("c1", "instagram")  # other platform is fine
        posts.update_post(first.id, status="canceled")
        posts.create_post("c1", "youtube")  # cancelled: may be posted again

    def test_failed_post_frees_the_platform_but_retry_checks_again(self) -> None:
        first = posts.create_post("c1", "youtube")
        posts.update_post(first.id, status="failed")
        second = posts.create_post("c1", "youtube")
        with pytest.raises(DoublePostError):
            posts.update_post(first.id, status="queued")  # would make two active
        assert posts.get_post(second.id).status == "queued"

    def test_expect_guards_against_races(self) -> None:
        post = posts.create_post("c1", "youtube")
        posts.update_post(post.id, status="canceled")
        with pytest.raises(StateChanged):
            posts.update_post(post.id, expect=frozenset({"queued"}), status="uploading")

    def test_fields_are_whitelisted(self) -> None:
        post = posts.create_post("c1", "youtube")
        with pytest.raises(ValueError):
            posts.update_post(post.id, clip_id="other")
        with pytest.raises(ValueError):
            posts.update_post(post.id, status="bogus")

    def test_records_survive_as_json(self, autoclip_home: Path) -> None:
        post = posts.create_post("c1", "instagram", scheduled_at="2026-11-19T12:00:00Z")
        raw = json.loads((autoclip_home / "publishing" / "posts.json").read_text("utf-8"))
        assert raw["posts"][0]["id"] == post.id
        assert raw["posts"][0]["scheduled_at"] == "2026-11-19T12:00:00Z"


# --------------------------------------------------------------------------
# Creating posts for a clip
# --------------------------------------------------------------------------


class TestCreatePosts:
    def test_creates_one_per_platform_with_clip_text(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        made = service.create_posts(clip.id, ["youtube", "instagram"], now=NOW)
        assert [p.platform for p in made] == ["youtube", "instagram"]
        assert made[0].title == "He did WHAT"
        assert made[0].caption == "The moment nobody expected"
        assert made[0].hashtags == ["#funny", "#wow"]
        assert made[0].scheduled_at is None

    def test_needs_an_exported_file(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path, exported=False)
        with pytest.raises(service.PostRequestError, match="Export"):
            service.create_posts(clip.id, ["youtube"], now=NOW)
        with pytest.raises(LookupError):
            service.create_posts("nope", ["youtube"], now=NOW)

    def test_time_must_be_ahead(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        with pytest.raises(service.PostRequestError, match="minutes ahead"):
            service.create_posts(
                clip.id, ["youtube"], scheduled_at=NOW + timedelta(minutes=2), now=NOW
            )
        made = service.create_posts(
            clip.id, ["youtube"], scheduled_at=NOW + timedelta(hours=5), now=NOW
        )
        assert made[0].scheduled_at == "2026-11-18T15:00:00Z"

    def test_double_post_creates_nothing(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        service.create_posts(clip.id, ["instagram"], now=NOW)
        with pytest.raises(DoublePostError):
            service.create_posts(clip.id, ["youtube", "instagram"], now=NOW)
        assert posts.list_posts(clip_id=clip.id, platform="youtube") == []


# --------------------------------------------------------------------------
# The poster
# --------------------------------------------------------------------------


class TestPosterYouTube:
    def test_post_now_goes_public(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["youtube"], now=NOW)
        uploader = FakeUploader()
        result = run(make_poster(Clock(), uploader).run_once())
        assert result["youtube"] == "posted"
        (path, meta) = uploader.calls[0]
        assert path.is_file() and meta.privacy == "public" and meta.publish_at is None
        assert "#Shorts" in meta.description and meta.tags == ["funny", "wow"]
        saved = posts.get_post(post.id)
        assert saved.status == "posted" and saved.remote_id == "vid1"

    def test_scheduled_post_is_sent_within_lead_as_private_publish_at(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        when = NOW + timedelta(hours=3)
        (post,) = service.create_posts(clip.id, ["youtube"], scheduled_at=when, now=NOW)
        clock, uploader = Clock(), FakeUploader()
        p = make_poster(clock, uploader, lead=120)
        assert run(p.run_once())["youtube"] == "waiting"  # 3 h ahead, lead 2 h
        assert uploader.calls == []
        clock.advance(hours=1, minutes=30)
        assert run(p.run_once())["youtube"] == "scheduled"
        meta = uploader.calls[0][1]
        assert meta.privacy == "private" and meta.publish_at == when
        clock.advance(hours=2)
        run(p.run_once())
        assert posts.get_post(post.id).status == "posted"  # YouTube made it public
        assert len(uploader.calls) == 1

    def test_missed_time_is_never_back_filled(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(
            clip.id, ["youtube"], scheduled_at=NOW + timedelta(hours=1), now=NOW
        )
        clock, uploader = Clock(NOW + timedelta(hours=2)), FakeUploader()  # PC was off
        run(make_poster(clock, uploader).run_once())
        saved = posts.get_post(post.id)
        assert uploader.calls == [] and saved.status == "failed"
        assert "not posted late" in saved.error
        run(make_poster(clock, uploader).run_once())
        assert uploader.calls == []  # not retried on its own

    def test_quota_pauses_without_using_attempts(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["youtube"], now=NOW)
        clock = Clock()
        uploader = FakeUploader(UploadApiError("HTTP 403: quotaExceeded"))
        p = make_poster(clock, uploader)
        assert run(p.run_once())["youtube"] == "paused"
        saved = posts.get_post(post.id)
        assert saved.status == "failed" and saved.attempts == 0
        clock.advance(minutes=10)
        assert run(p.run_once())["youtube"] == "paused"
        assert len(uploader.calls) == 1

    def test_failures_retry_then_stop(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["youtube"], now=NOW)
        clock = Clock()
        uploader = FakeUploader(UploadApiError("HTTP 400: bad"))
        p = make_poster(clock, uploader)
        for _ in range(6):
            run(p.run_once())
            clock.advance(seconds=core.RETRY_AFTER_S + 1)
        assert len(uploader.calls) == core.MAX_ATTEMPTS
        assert posts.get_post(post.id).attempts == core.MAX_ATTEMPTS

    def test_not_ready_keeps_posts_queued(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["youtube"], now=NOW)
        p = make_poster(Clock(), ready="Connect your YouTube channel in the Posting tab.")
        assert run(p.run_once())["youtube"] == "not_ready"
        assert posts.get_post(post.id).status == "queued"
        assert "Connect" in p.states["youtube"].message

    def test_missing_file_fails_for_good(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["youtube"], now=NOW)
        for export in store.list_exports(clip.id):
            Path(export.path).unlink()
        uploader = FakeUploader()
        run(make_poster(Clock(), uploader).run_once())
        assert uploader.calls == []
        assert posts.get_post(post.id).attempts == core.MAX_ATTEMPTS

    def test_recover_marks_crashed_uploads(self) -> None:
        yt = posts.create_post("c1", "youtube")
        posts.update_post(yt.id, status="uploading")
        ig_sent = posts.create_post("c2", "instagram")
        posts.update_post(ig_sent.id, status="uploading", media_url="https://m")
        ig_not = posts.create_post("c3", "instagram")
        posts.update_post(ig_not.id, status="uploading")
        assert poster_mod.Poster.recover() == 3
        assert posts.get_post(yt.id).status == "interrupted"
        assert posts.get_post(ig_sent.id).status == "checking"
        assert posts.get_post(ig_not.id).status == "failed"


class TestPosterInstagram:
    def test_post_now_is_scheduled_shortly_then_confirmed(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["instagram"], now=NOW)
        clock, ig = Clock(), FakeZernio()
        p = make_poster(clock, ig=ig)
        assert run(p.run_once())["instagram"] == "scheduled"
        caption, when = ig.sent[0]
        assert when == NOW + timedelta(seconds=poster_mod.IG_NOW_DELAY_S)
        assert "#funny #wow" in caption
        clock.advance(minutes=20)
        run(p.run_once())
        saved = posts.get_post(post.id)
        assert saved.status == "posted" and saved.url == "https://ig/p/1"
        assert len(ig.sent) == 1

    def test_unclear_answer_is_checked_never_resent(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["instagram"], now=NOW)
        clock, ig = Clock(), FakeZernio(kind="unconfirmed")
        p = make_poster(clock, ig=ig)
        assert run(p.run_once())["instagram"] == "checking"
        clock.advance(seconds=core.IG_CHECK_EVERY_S + 1)
        ig.check_kind = "scheduled"
        run(p.run_once())
        assert posts.get_post(post.id).status == "scheduled"
        assert len(ig.sent) == 1 and len(ig.checks) == 1


class TestCancelAndRetry:
    def test_cancel_queued_and_zernio_scheduled(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        when = NOW + timedelta(hours=1)
        yt, ig_post = service.create_posts(
            clip.id, ["youtube", "instagram"], scheduled_at=when, now=NOW
        )
        ig = FakeZernio()
        p = make_poster(Clock(), FakeUploader(), ig=ig, lead=30)
        assert run(p.cancel(yt.id)).status == "canceled"  # never sent
        p2 = make_poster(Clock(), FakeUploader(), ig=ig, lead=120)
        run(p2.run_once())
        assert posts.get_post(ig_post.id).status == "scheduled"
        assert run(p2.cancel(ig_post.id)).status == "canceled"
        assert ig.deleted == ["zp1"]

    def test_youtube_scheduled_needs_manage_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(
            clip.id, ["youtube"], scheduled_at=NOW + timedelta(hours=1), now=NOW
        )
        p = make_poster(Clock())
        run(p.run_once())
        assert posts.get_post(post.id).status == "scheduled"
        monkeypatch.setattr(remote, "youtube_can_edit", lambda: False)
        with pytest.raises(poster_mod.CancelError) as info:
            run(p.cancel(post.id))
        assert info.value.studio and "YouTube Studio" in str(info.value)

        calls: list[str] = []

        async def fake_private(video_id: str, **_kw) -> str:
            calls.append(video_id)
            return "private"

        monkeypatch.setattr(remote, "youtube_can_edit", lambda: True)
        monkeypatch.setattr(remote, "youtube_make_private", fake_private)
        canceled = run(p.cancel(post.id, "private"))
        assert canceled.status == "canceled" and calls == ["vid1"]
        assert "private" in canceled.error

    def test_retry_resets_and_makes_a_past_time_now(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(
            clip.id, ["youtube"], scheduled_at=NOW + timedelta(hours=1), now=NOW
        )
        posts.update_post(post.id, status="failed", attempts=3, error="x")
        clock = Clock(NOW + timedelta(hours=2))
        retried = make_poster(clock).retry(post.id)
        assert retried.status == "queued" and retried.attempts == 0
        assert retried.scheduled_at is None

    def test_retry_refuses_a_second_active_post(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        (post,) = service.create_posts(clip.id, ["youtube"], now=NOW)
        posts.update_post(post.id, status="canceled")
        service.create_posts(clip.id, ["youtube"], now=NOW)
        with pytest.raises(DoublePostError):
            make_poster(Clock()).retry(post.id)

    def test_uploading_can_not_be_cancelled(self) -> None:
        post = posts.create_post("c1", "youtube")
        posts.update_post(post.id, status="uploading")
        with pytest.raises(poster_mod.CancelError, match="being sent"):
            run(make_poster(Clock()).cancel(post.id))


# --------------------------------------------------------------------------
# Slot maths shared with GTA Mode
# --------------------------------------------------------------------------


class TestSlots:
    def test_free_slots_skip_taken_and_too_close(self) -> None:
        now = datetime(2026, 11, 18, 11, 59, tzinfo=UTC)
        taken = {"2026-11-18T18:00:00Z"}
        got = core.free_slots(["12:00", "18:00"], now, taken, days=1, tz=UTC)
        assert [core.slot_key(m) for m in got] == ["2026-11-19T12:00:00Z", "2026-11-19T18:00:00Z"]

    def test_core_reuses_gta_rules(self) -> None:
        from autoclip.gta import poster as gta_poster

        assert core.slot_key is gta_poster.slot_key
        assert core.MAX_ATTEMPTS == gta_poster.MAX_ATTEMPTS
        assert core.is_quota_error("quotaExceeded")


# --------------------------------------------------------------------------
# Schedule aggregation
# --------------------------------------------------------------------------


def _gta_upload(tmp_path: Path, *, status: str, slot_at: str, platform: str = "youtube"):
    stream = gta_store.create_stream(f"https://www.twitch.tv/videos/{new_id()[:8]}")
    clip = gta_store.create_clip(
        gta_store.GtaClip(
            id=new_id(),
            stream_id=stream.id,
            start_s=1,
            end_s=20,
            status="ready",
            hook_title="GTA moment",
            path=str(tmp_path / "g.mp4"),
            qa={"passed": True},
        )
    )
    row = gta_store.create_upload(clip.id, status=status, platform=platform, slot_at=slot_at)
    return row


class TestSchedule:
    def test_merges_normal_autopilot_and_gta(self, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        future = NOW + timedelta(hours=5)
        (queued,) = service.create_posts(clip.id, ["youtube"], scheduled_at=future, now=NOW)
        other = make_clip(tmp_path, title="Second")
        (done,) = service.create_posts(other.id, ["instagram"], now=NOW)
        posts.update_post(done.id, status="posted", url="https://ig/p/9")
        booked = posts.create_post(
            "auto-clip", "youtube", scheduled_at="2026-11-19T12:00:00Z", autopilot_id="ap1"
        )
        _gta_upload(tmp_path, status="scheduled", slot_at="2026-11-18T21:00:00Z")
        _gta_upload(tmp_path, status="failed", slot_at="2026-11-17T21:00:00Z")

        out = service.schedule(NOW, can_edit=False)
        upcoming = [(i["source"], i["status"]) for i in out["upcoming"]]
        assert upcoming == [("normal", "queued"), ("gta", "scheduled"), ("autopilot", "queued")]
        assert [i["when"] for i in out["upcoming"]] == sorted(i["when"] for i in out["upcoming"])
        past = {(i["source"], i["status"]) for i in out["past"]}
        assert ("normal", "posted") in past and ("gta", "failed") in past
        first = out["upcoming"][0]
        assert first["id"] == queued.id and first["can_cancel"] and first["title"] == "He did WHAT"
        gta_failed = next(i for i in out["past"] if i["source"] == "gta")
        assert gta_failed["can_retry"] and not gta_failed["can_cancel"]
        assert out["counts"]["failed"] == 1
        assert any(i["id"] == booked.id for i in out["upcoming"])

    def test_youtube_scheduled_cancel_depends_on_scope(self, tmp_path: Path) -> None:
        post = posts.create_post("c", "youtube", scheduled_at="2026-11-18T15:00:00Z")
        posts.update_post(post.id, status="scheduled", remote_id="vid")
        item = service.schedule(NOW, can_edit=False)["upcoming"][0]
        assert not item["can_cancel"] and "YouTube Studio" in item["cancel_note"]
        item = service.schedule(NOW, can_edit=True)["upcoming"][0]
        assert item["can_cancel"] and item["cancel_options"] == ["private", "delete"]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    monkeypatch.setattr(remote, "youtube_can_edit", lambda: False)
    with TestClient(create_app()) as test_client:
        yield test_client


class TestApi:
    def test_post_twice_is_409_and_schedule_lists_it(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        clip = make_clip(tmp_path)
        first = client.post(
            "/api/publishing/posts", json={"clip_id": clip.id, "platforms": ["youtube"]}
        )
        assert first.status_code == 201, first.text
        body = first.json()
        assert body["posts"][0]["status"] == "queued"
        assert body["warnings"]  # not connected in a temp home
        again = client.post(
            "/api/publishing/posts", json={"clip_id": clip.id, "platforms": ["youtube"]}
        )
        assert again.status_code == 409
        sched = client.get("/api/schedule").json()
        assert sched["upcoming"][0]["clip_id"] == clip.id
        listed = client.get("/api/publishing/clips").json()
        assert listed[0]["clip_id"] == clip.id and listed[0]["active"]["youtube"]

    def test_cancel_and_retry_endpoints(self, client: TestClient, tmp_path: Path) -> None:
        clip = make_clip(tmp_path)
        post_id = client.post(
            "/api/publishing/posts", json={"clip_id": clip.id, "platforms": ["instagram"]}
        ).json()["posts"][0]["id"]
        assert client.post(f"/api/publishing/posts/{post_id}/cancel").json()["status"] == (
            "canceled"
        )
        assert client.post(f"/api/publishing/posts/{post_id}/retry").json()["status"] == "queued"
        assert client.post("/api/publishing/posts/nope/retry").status_code == 404

    def test_bad_requests(self, client: TestClient, tmp_path: Path) -> None:
        clip = make_clip(tmp_path, exported=False)
        r = client.post(
            "/api/publishing/posts", json={"clip_id": clip.id, "platforms": ["youtube"]}
        )
        assert r.status_code == 400
        r = client.post("/api/publishing/posts", json={"clip_id": "x", "platforms": ["youtube"]})
        assert r.status_code == 404

    def test_status_and_settings(self, client: TestClient) -> None:
        status = client.get("/api/publishing/status").json()
        assert set(status["ready"]) == {"youtube", "instagram", "youtube_can_edit"}
        saved = client.put(
            "/api/publishing/settings", json={"category_id": "20", "lead_minutes": 60}
        )
        assert saved.status_code == 200 and saved.json()["category_id"] == "20"
        bad = client.put("/api/publishing/settings", json={"lead_minutes": 1})
        assert bad.status_code == 422
