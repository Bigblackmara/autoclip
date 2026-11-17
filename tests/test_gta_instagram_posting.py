"""GTA Mode posting to Instagram through Zernio: poster rules + API routes."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip.app import create_app
from autoclip.db.models import new_id
from autoclip.gta import poster as poster_mod
from autoclip.gta import settings as gta_settings
from autoclip.gta import store, zernio
from autoclip.gta.settings import GtaPosting, GtaSettings
from autoclip.gta.zernio import ZernioResult
from autoclip.live.uploads.tokens import UploadMeta, UploadResult
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _db(initialised_db: int) -> None:
    """Every test runs against a migrated temp database."""


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(zernio, "load_api_key", lambda: "zk_test_key")


def _now() -> datetime:
    return datetime.now(UTC).replace(second=0, microsecond=0)


def _hhmm(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%H:%M")


def _enable(slots: list[str], *, youtube: bool = False, instagram: bool = True) -> None:
    gta_settings.save(
        GtaSettings(
            posting=GtaPosting(
                enabled=True,
                slots=slots,
                lead_minutes=120,
                youtube=youtube,
                instagram=instagram,
                instagram_account="clipgod",
            )
        )
    )


def _clip(tmp_path: Path, *, score: int = 80, title: str = "NO WAY") -> store.GtaClip:
    stream = store.create_stream(f"https://www.twitch.tv/videos/{new_id()[:8]}")
    path = tmp_path / f"{new_id()}.mp4"
    path.write_bytes(b"\x00" * 64)
    return store.create_clip(
        store.GtaClip(
            id=new_id(),
            stream_id=stream.id,
            start_s=1.0,
            end_s=20.0,
            status="ready",
            hook_title=title,
            score=score,
            caption="Chat lost it",
            hashtags=["#gta", "#GTA", "#rp", "#two words"],
            path=str(path),
            qa={"passed": True},
        )
    )


class FakeZernio:
    def __init__(self, result: ZernioResult | None = None) -> None:
        self.result = result or ZernioResult(kind="scheduled", post_id="p1")
        self.calls: list[dict] = []
        self.checks: list[tuple] = []
        self.check_result = ZernioResult(kind="posted", post_id="p1", post_url="https://ig/p1")

    def schedule_reel(
        self, api_key, account, path, caption, publish_at, *, on_media_url, on_post_id
    ):
        self.calls.append(
            {"key": api_key, "account": account, "path": path, "caption": caption, "at": publish_at}
        )
        on_media_url("https://media.zernio/x.mp4")
        if self.result.post_id:
            on_post_id(self.result.post_id)
        return self.result

    def check_post(self, api_key, post_id, media_url, sent_at):
        self.checks.append((post_id, media_url, sent_at))
        return self.check_result


class FakeUploader:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, path: Path, meta: UploadMeta) -> UploadResult:
        self.calls += 1
        return UploadResult(platform="youtube", media_id="v1", url="https://youtu.be/v1")


def _poster(fake: FakeZernio, clock, uploader=None) -> poster_mod.GtaPoster:
    return poster_mod.GtaPoster(
        uploader=uploader or FakeUploader(),
        clock=clock,
        ready_check=lambda: "",
        instagram_ready_check=lambda: "",
        instagram=fake,
        tz=UTC,
    )


@pytest.mark.usefixtures("key")
class TestInstagramPoster:
    def test_schedules_best_clip_at_zernio_for_the_slot(self, tmp_path: Path) -> None:
        _clip(tmp_path, score=50)
        best = _clip(tmp_path, score=95)
        now = _now()
        slot = now + timedelta(minutes=90)
        _enable([_hhmm(slot)])
        fake = FakeZernio()
        p = _poster(fake, lambda: now)
        assert asyncio.run(p.run_once()) == {"youtube": "off", "instagram": "scheduled"}
        (call,) = fake.calls
        assert call["path"] == best.path and call["at"] == slot and call["account"] == "clipgod"
        assert call["key"] == "zk_test_key"
        assert call["caption"] == "Chat lost it\n\n#gta #rp"
        row = store.upload_for_clip(best.id, "instagram")
        assert row.status == "scheduled" and row.video_id == "p1" and row.attempts == 1
        assert row.media_url == "https://media.zernio/x.mp4"
        assert asyncio.run(p.run_once())["instagram"] == "waiting"
        assert len(fake.calls) == 1

    def test_same_clip_goes_to_both_platforms(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=60))], youtube=True)
        fake, uploader = FakeZernio(), FakeUploader()
        p = _poster(fake, lambda: now, uploader)
        assert asyncio.run(p.run_once()) == {"youtube": "scheduled", "instagram": "scheduled"}
        assert set(store.uploads_for_clip(clip.id)) == {"youtube", "instagram"}
        plan = p.plan(gta_settings.load(), now)
        assert [(e["platform"], e["status"]) for e in plan[:2]] == [
            ("youtube", "scheduled"),
            ("instagram", "scheduled"),
        ]

    def test_unclear_answer_is_checked_never_resent(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        _clip(tmp_path, score=10)
        now = _now()
        clock = {"t": now}
        _enable([_hhmm(now + timedelta(minutes=110))])
        fake = FakeZernio(ZernioResult(kind="unconfirmed", message="HTTP 502"))
        p = _poster(fake, lambda: clock["t"])
        assert asyncio.run(p.run_once())["instagram"] == "checking"
        row = store.upload_for_clip(clip.id, "instagram")
        assert row.status == "checking" and row.media_url
        # It holds the slot: the second clip is not sent for it.
        assert asyncio.run(p.run_once())["instagram"] == "waiting"
        assert len(fake.calls) == 1
        # After the check interval Zernio is asked; it says scheduled.
        fake.check_result = ZernioResult(kind="scheduled", post_id="p9")
        clock["t"] += timedelta(minutes=6)
        asyncio.run(p.run_once())
        assert fake.checks[0][1] == "https://media.zernio/x.mp4"
        row = store.upload_for_clip(clip.id, "instagram")
        assert row.status == "scheduled" and row.video_id == "p9" and len(fake.calls) == 1

    def test_not_listed_stays_checking_with_a_hint(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        clock = {"t": now}
        _enable([_hhmm(now + timedelta(minutes=110))])
        fake = FakeZernio(ZernioResult(kind="unconfirmed"))
        p = _poster(fake, lambda: clock["t"])
        asyncio.run(p.run_once())
        fake.check_result = ZernioResult(kind="unconfirmed", not_listed=True)
        clock["t"] += timedelta(minutes=6)
        asyncio.run(p.run_once())
        row = store.upload_for_clip(clip.id, "instagram")
        assert row.status == "checking" and "press Retry" in row.error
        assert len(fake.calls) == 1

    def test_confirms_scheduled_reel_went_live_after_its_slot(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        slot = now + timedelta(minutes=90)
        clock = {"t": now}
        _enable([_hhmm(slot)])
        fake = FakeZernio()
        p = _poster(fake, lambda: clock["t"])
        asyncio.run(p.run_once())
        clock["t"] = slot + timedelta(minutes=2)
        asyncio.run(p.run_once())
        assert fake.checks == []  # too early to confirm
        clock["t"] = slot + timedelta(minutes=10)
        asyncio.run(p.run_once())
        row = store.upload_for_clip(clip.id, "instagram")
        assert row.status == "posted" and row.url == "https://ig/p1"

    def test_blocked_pauses_without_burning_attempts(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        clock = {"t": now}
        _enable([_hhmm(now + timedelta(minutes=110))])
        fake = FakeZernio(ZernioResult(kind="blocked", message="rate limit", retry_after=600))
        p = _poster(fake, lambda: clock["t"])
        assert asyncio.run(p.run_once())["instagram"] == "paused"
        row = store.upload_for_clip(clip.id, "instagram")
        assert row.status == "failed" and row.attempts == 0
        clock["t"] += timedelta(minutes=5)
        assert asyncio.run(p.run_once())["instagram"] == "paused"
        assert len(fake.calls) == 1

    def test_invalid_is_never_retried_and_failed_is(self, tmp_path: Path) -> None:
        bad = _clip(tmp_path, score=90)
        now = _now()
        clock = {"t": now}
        _enable([_hhmm(now + timedelta(minutes=110))])
        fake = FakeZernio(ZernioResult(kind="invalid", message="422"))
        p = _poster(fake, lambda: clock["t"])
        asyncio.run(p.run_once())
        assert store.upload_for_clip(bad.id, "instagram").attempts == poster_mod.MAX_ATTEMPTS
        other = _clip(tmp_path, score=10)
        fake.result = ZernioResult(kind="failed", message="nope")
        clock["t"] += timedelta(minutes=6)
        assert asyncio.run(p.run_once())["instagram"] == "failed"
        assert fake.calls[-1]["path"] == other.path
        assert store.upload_for_clip(other.id, "instagram").attempts == 1

    def test_skipped_anywhere_is_not_posted(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        store.create_upload(clip.id, status="skipped", platform="youtube")
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=60))])
        fake = FakeZernio()
        assert asyncio.run(_poster(fake, lambda: now).run_once())["instagram"] == "no_clip"

    def test_stop_mid_send_becomes_checking(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=60))])

        class Hanging(FakeZernio):
            def schedule_reel(self, *args, on_media_url, on_post_id):
                on_media_url("https://media.zernio/h.mp4")
                import time

                time.sleep(0.5)
                return ZernioResult(kind="scheduled", post_id="late")

        async def scenario() -> None:
            p = _poster(Hanging(), lambda: now)
            task = asyncio.create_task(p.run_once())
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        row = store.upload_for_clip(clip.id, "instagram")
        assert row.status == "checking" and "not be sent twice" in row.error

    def test_recover_after_crash(self, tmp_path: Path) -> None:
        sent = _clip(tmp_path)
        unsent = _clip(tmp_path)
        yt = _clip(tmp_path)
        a = store.create_upload(sent.id, status="uploading", platform="instagram", slot_at="x")
        store.update_upload(a.id, media_url="https://media.zernio/a.mp4")
        b = store.create_upload(unsent.id, status="uploading", platform="instagram", slot_at="y")
        c = store.create_upload(yt.id, status="uploading", platform="youtube", slot_at="z")
        assert poster_mod.GtaPoster.recover() == 3
        assert store.get_upload(a.id).status == "checking"
        assert store.get_upload(b.id).status == "failed"
        assert store.get_upload(c.id).status == "interrupted"

    def test_caption_limits(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        clip.caption = "x" * 3000
        clip.hashtags = [f"#t{i}" for i in range(40)]
        text = poster_mod.instagram_caption(clip)
        assert len(text) <= poster_mod.IG_CAPTION_MAX
        assert text.count("#") == 30 and text.endswith("#t29")


def test_readiness_needs_rights_then_key(fake_keyring, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.live.uploads import tokens as tk

    assert "rights" in poster_mod.instagram_readiness()
    tk.accept_rights()
    assert "Zernio" in poster_mod.instagram_readiness()
    zernio.save_api_key("zk_live_abc")
    assert poster_mod.instagram_readiness() == ""


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


class TestInstagramApi:
    def test_key_save_import_and_test(
        self, client: TestClient, fake_keyring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(zernio, "reelrepost_api_key", lambda: None)
        status = client.get("/api/gta/instagram").json()
        assert status == {"key_saved": False, "reelrepost_key_available": False}
        assert client.post("/api/gta/instagram/import-reelrepost").status_code == 404
        tested = client.post("/api/gta/instagram/test", json={"account": ""}).json()
        assert tested["ok"] is False and "API key" in tested["message"]

        monkeypatch.setattr(zernio, "reelrepost_api_key", lambda: "rr_key_123")
        assert client.get("/api/gta/instagram").json()["reelrepost_key_available"] is True
        imported = client.post("/api/gta/instagram/import-reelrepost").json()
        assert imported["key_saved"] is True and zernio.load_api_key() == "rr_key_123"
        assert "rr_key_123" not in str(imported)

        seen = {}

        def fake_test(api_key, account):
            seen.update(key=api_key, account=account)
            return True, "Connected (@clipgod)", "clipgod"

        monkeypatch.setattr(zernio, "test_connection", fake_test)
        ok = client.post("/api/gta/instagram/test", json={"account": "@clipgod"}).json()
        assert ok == {"ok": True, "message": "Connected (@clipgod)", "username": "clipgod"}
        assert seen == {"key": "rr_key_123", "account": "clipgod"}

        cleared = client.put("/api/gta/instagram/key", json={"api_key": ""}).json()
        assert cleared["key_saved"] is False

    def test_overview_has_both_platforms(self, client: TestClient, tmp_path: Path) -> None:
        _clip(tmp_path, title="BOTH")
        saved = client.put(
            "/api/gta/settings",
            json={"posting": {"enabled": True, "instagram": True, "instagram_account": "@me"}},
        )
        assert saved.status_code == 200 and saved.json()["posting"]["instagram_account"] == "me"
        overview = client.get("/api/gta/posting").json()
        assert set(overview["states"]) == {"youtube", "instagram"} and overview["running"] is False
        planned = [e for e in overview["plan"] if e["status"] == "planned"]
        assert {e["platform"] for e in planned} == {"youtube", "instagram"}
        assert all(e["title"] == "BOTH" for e in planned)
        bad = client.put("/api/gta/settings", json={"posting": {"instagram_account": "a b"}})
        assert bad.status_code == 400

    def test_checking_can_be_retried(self, client: TestClient, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        row = store.create_upload(clip.id, status="checking", platform="instagram", slot_at="x")
        assert client.post(f"/api/gta/uploads/{row.id}/retry").status_code == 200
        assert store.upload_for_clip(clip.id, "instagram") is None
