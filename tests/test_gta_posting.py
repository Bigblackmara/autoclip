"""GTA Mode YouTube posting schedule: slots, clip choice, uploads, API."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from autoclip import app as app_module
from autoclip.app import create_app
from autoclip.db.models import new_id
from autoclip.gta import poster as poster_mod
from autoclip.gta import settings as gta_settings
from autoclip.gta import store
from autoclip.gta.settings import GtaPosting, GtaSettings
from autoclip.live.uploads import tokens as tk
from autoclip.live.uploads.tokens import TokenBundle, UploadApiError, UploadMeta, UploadResult
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _db(initialised_db: int) -> None:
    """Every test runs against a migrated temp database."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC).replace(second=0, microsecond=0)


def _hhmm(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%H:%M")


def _enable(slots: list[str], lead: int = 120, **extra) -> None:
    gta_settings.save(
        GtaSettings(posting=GtaPosting(enabled=True, slots=slots, lead_minutes=lead, **extra))
    )


def _clip(
    tmp_path: Path,
    *,
    score: int = 80,
    title: str = "HE DID NOT SEE THAT",
    passed: bool = True,
    status: str = "ready",
    with_file: bool = True,
) -> store.GtaClip:
    stream = store.create_stream(f"https://www.twitch.tv/videos/{new_id()[:8]}")
    path = tmp_path / f"{new_id()}.mp4"
    if with_file:
        path.write_bytes(b"\x00" * 64)
    clip = store.GtaClip(
        id=new_id(),
        stream_id=stream.id,
        start_s=10.0,
        end_s=35.0,
        status=status,
        hook_title=title,
        score=score,
        caption="Chat could not believe it",
        hashtags=["#kaicenat", "#gta", "#kaicenat", "#<bad>"],
        path=str(path),
        qa={"passed": passed},
    )
    return store.create_clip(clip)


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


def _poster(uploader, clock, ready: str = "") -> poster_mod.GtaPoster:
    return poster_mod.GtaPoster(uploader=uploader, clock=clock, ready_check=lambda: ready, tz=UTC)


# --------------------------------------------------------------------------
# Settings + slot maths
# --------------------------------------------------------------------------


class TestSlots:
    def test_posting_settings_validate_and_normalise(self) -> None:
        p = GtaPosting(slots=["9:5", "21:00", "09:05"])
        assert p.slots == ["09:05", "21:00"] and p.enabled is False
        for bad in (["25:00"], ["noon"], [], ["12:60"]):
            with pytest.raises(ValueError):
                GtaPosting(slots=bad)
        with pytest.raises(ValueError):
            GtaPosting(lead_minutes=5)
        with pytest.raises(ValueError):
            GtaPosting(category_id="99")
        assert GtaSettings().posting.enabled is False  # never on by default

    def test_due_slot_window_and_no_backfill(self) -> None:
        now = datetime(2026, 11, 17, 10, 30, tzinfo=UTC)
        slots = ["12:00", "18:00"]
        noon = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
        assert poster_mod.due_slot(slots, 120, now, set(), tz=UTC) == noon
        assert poster_mod.due_slot(slots, 60, now, set(), tz=UTC) is None  # not yet
        # Taken noon -> 18:00 is 7.5 h away: nothing due.
        taken = {poster_mod.slot_key(noon)}
        assert poster_mod.due_slot(slots, 120, now, taken, tz=UTC) is None
        # 11:59: noon is too close, and 18:00 isn't due -> never a rushed back-fill.
        late = datetime(2026, 11, 17, 11, 59, tzinfo=UTC)
        assert poster_mod.due_slot(slots, 120, late, set(), tz=UTC) is None
        # Late evening wraps to tomorrow's first slot.
        evening = datetime(2026, 11, 17, 23, 0, tzinfo=UTC)
        tomorrow = poster_mod.slot_datetimes(["00:30"], evening, tz=UTC)[0]
        assert tomorrow == datetime(2026, 11, 18, 0, 30, tzinfo=UTC)

    def test_slot_key_roundtrip(self) -> None:
        moment = datetime(2026, 11, 17, 17, 0, tzinfo=UTC)
        assert poster_mod.slot_key(moment) == "2026-11-17T17:00:00Z"
        assert poster_mod.parse_slot_key("2026-11-17T17:00:00Z") == moment


# --------------------------------------------------------------------------
# Clip choice + metadata
# --------------------------------------------------------------------------


class TestClipChoice:
    def test_only_ready_qa_passed_clips_with_files_best_first(self, tmp_path: Path) -> None:
        good = _clip(tmp_path, score=70)
        best = _clip(tmp_path, score=95)
        _clip(tmp_path, score=99, passed=False)
        _clip(tmp_path, score=99, with_file=False)
        _clip(tmp_path, score=99, status="qa_failed")
        skipped = _clip(tmp_path, score=98)
        store.create_upload(skipped.id, status="skipped")
        order = [c.id for c, _ in poster_mod.eligible_clips(_now())]
        assert order == [best.id, good.id]

    def test_metadata_is_scheduled_private_with_clean_tags(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path, title="WAIT <WHAT>")
        publish = datetime(2026, 11, 17, 17, 0, tzinfo=UTC)
        meta = poster_mod.build_meta(clip, GtaSettings(), publish)
        assert meta.title == "WAIT WHAT" and meta.privacy == "private"
        assert meta.publish_at == publish and meta.category_id == "24"
        assert meta.tags == ["kaicenat", "gta", "bad"]
        assert meta.description.endswith("#Shorts") and "<" not in meta.description
        off = GtaSettings(posting=GtaPosting(shorts_tag=False))
        assert "#Shorts" not in poster_mod.build_meta(clip, off, publish).description

    def test_youtube_payload_schedules_publish(self) -> None:
        from autoclip.live.uploads.youtube import _metadata_payload

        plain = _metadata_payload(UploadMeta(title="t", privacy="public"))
        assert plain["status"] == {"privacyStatus": "public", "selfDeclaredMadeForKids": False}
        india = timezone(timedelta(hours=5, minutes=30))
        local = datetime(2026, 11, 17, 22, 30, tzinfo=india)
        scheduled = _metadata_payload(UploadMeta(title="t", privacy="public", publish_at=local))
        assert scheduled["status"]["privacyStatus"] == "private"
        assert scheduled["status"]["publishAt"] == "2026-11-17T17:00:00Z"
        with pytest.raises(ValueError):
            UploadMeta(title="t", publish_at=datetime(2026, 11, 17, 17, 0))  # naive


# --------------------------------------------------------------------------
# The poster
# --------------------------------------------------------------------------


class TestPoster:
    def test_off_and_not_ready_upload_nothing(self, tmp_path: Path) -> None:
        _clip(tmp_path)
        now = _now()
        uploader = FakeUploader()
        assert asyncio.run(_poster(uploader, lambda: now).run_once())["youtube"] == "off"
        _enable([_hhmm(now + timedelta(minutes=90))])
        p = _poster(uploader, lambda: now, ready="Connect your YouTube channel.")
        assert asyncio.run(p.run_once())["youtube"] == "not_ready"
        assert p.state.message == "Connect your YouTube channel."
        assert uploader.calls == [] and store.list_uploads() == []

    def test_uploads_best_clip_once_per_slot(self, tmp_path: Path) -> None:
        _clip(tmp_path, score=60)
        best = _clip(tmp_path, score=90)
        now = _now()
        slot = now + timedelta(minutes=90)
        _enable([_hhmm(slot)])
        uploader = FakeUploader()
        p = _poster(uploader, lambda: now)
        assert asyncio.run(p.run_once())["youtube"] == "scheduled"
        ((path, meta),) = uploader.calls
        assert path == Path(best.path) and meta.publish_at == slot
        row = store.upload_for_clip(best.id)
        assert row is not None and row.status == "scheduled" and row.video_id == "vid1"
        assert row.slot_at == poster_mod.slot_key(slot) and row.url.endswith("vid1")
        # Same slot again: taken, nothing more is uploaded.
        assert asyncio.run(p.run_once())["youtube"] == "waiting"
        assert len(uploader.calls) == 1
        # The plan shows the scheduled clip, then the next one queued for tomorrow.
        plan = p.plan(gta_settings.load(), now)
        assert plan[0]["status"] == "scheduled" and plan[0]["clip_id"] == best.id
        assert plan[1]["status"] == "planned"

    def test_failure_retries_after_backoff_then_gives_up(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=110))])
        clock = {"t": now}
        uploader = FakeUploader(error=UploadApiError("HTTP 500: backend error"))
        p = _poster(uploader, lambda: clock["t"])
        assert asyncio.run(p.run_once())["youtube"] == "failed"
        row = store.upload_for_clip(clip.id)
        assert row is not None and row.status == "failed" and row.attempts == 1
        # Within the back-off the clip isn't retried.
        assert asyncio.run(p.run_once())["youtube"] == "no_clip"
        for attempt in (2, 3):
            clock["t"] += timedelta(minutes=6)
            assert asyncio.run(p.run_once())["youtube"] == "failed"
            assert store.upload_for_clip(clip.id).attempts == attempt
        clock["t"] += timedelta(minutes=6)
        assert asyncio.run(p.run_once())["youtube"] == "no_clip"  # out of attempts
        assert len(uploader.calls) == 3

    def test_retry_succeeds_and_reuses_the_row(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=110))])
        clock = {"t": now}
        uploader = FakeUploader(error=UploadApiError("HTTP 503"))
        p = _poster(uploader, lambda: clock["t"])
        asyncio.run(p.run_once())
        uploader.error = None
        clock["t"] += timedelta(minutes=6)
        assert asyncio.run(p.run_once())["youtube"] == "scheduled"
        rows = store.list_uploads()
        assert len(rows) == 1 and rows[0].clip_id == clip.id and rows[0].status == "scheduled"
        assert rows[0].attempts == 2 and rows[0].error is None

    def test_quota_pauses_without_burning_attempts(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=110))])
        clock = {"t": now}
        error = UploadApiError('HTTP 403: {"error": {"errors": [{"reason": "quotaExceeded"}]}}')
        uploader = FakeUploader(error=error)
        p = _poster(uploader, lambda: clock["t"])
        assert asyncio.run(p.run_once())["youtube"] == "paused"
        assert store.upload_for_clip(clip.id).attempts == 0
        clock["t"] += timedelta(minutes=30)
        assert asyncio.run(p.run_once())["youtube"] == "paused"
        assert len(uploader.calls) == 1

    def test_connection_errors_keep_attempts(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=110))])
        uploader = FakeUploader(error=tk.NotLinkedError("not linked"))
        assert asyncio.run(_poster(uploader, lambda: now).run_once())["youtube"] == "failed"
        row = store.upload_for_clip(clip.id)
        assert row.status == "failed" and row.attempts == 0 and "connection" in row.error

    def test_stop_mid_upload_marks_interrupted_never_retried(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        now = _now()
        _enable([_hhmm(now + timedelta(minutes=90))])

        class Hanging:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def __call__(self, path, meta):
                self.started.set()
                await asyncio.Event().wait()

        async def scenario() -> None:
            hanging = Hanging()
            p = _poster(hanging, lambda: now)
            task = asyncio.create_task(p.run_once())
            await hanging.started.wait()
            assert p.state.uploading_clip_id == clip.id
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        row = store.upload_for_clip(clip.id)
        assert row.status == "interrupted" and "YouTube Studio" in row.error
        # Interrupted holds its slot and its clip: nothing else happens on its own.
        uploader = FakeUploader()
        assert asyncio.run(_poster(uploader, lambda: now).run_once())["youtube"] == "waiting"
        assert uploader.calls == []

    def test_recover_turns_crashed_uploads_into_interrupted(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        row = store.create_upload(clip.id, status="uploading", slot_at="2026-11-17T12:00:00Z")
        assert poster_mod.GtaPoster.recover() == 1
        assert store.get_upload(row.id).status == "interrupted"

    def test_readiness_reads_the_keyring_in_order(self, fake_keyring) -> None:
        assert "rights" in poster_mod.readiness()
        tk.accept_rights()
        assert "client ID" in poster_mod.readiness()
        tk.save_client_credentials("youtube", "id", "secret")
        assert "Connect" in poster_mod.readiness()
        tk.save_tokens("youtube", TokenBundle(access_token="a", refresh_token="r"))
        assert poster_mod.readiness() == ""

    def test_deleting_the_clip_removes_its_upload_row(self, tmp_path: Path) -> None:
        clip = _clip(tmp_path)
        store.create_upload(clip.id, status="scheduled", slot_at="2026-11-17T12:00:00Z")
        store.delete_stream(clip.stream_id)
        assert store.list_uploads() == []


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def _planned(client: TestClient) -> list[dict]:
    return [e for e in client.get("/api/gta/posting").json()["plan"] if e["status"] == "planned"]


class TestPostingApi:
    def test_overview_skip_unskip_and_clip_upload_field(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        first = _clip(tmp_path, score=90, title="FIRST")
        second = _clip(tmp_path, score=50, title="SECOND")
        saved = client.put(
            "/api/gta/settings", json={"posting": {"enabled": True, "slots": ["23:59", "9:00"]}}
        )
        assert saved.status_code == 200 and saved.json()["posting"]["slots"] == ["09:00", "23:59"]
        overview = client.get("/api/gta/posting").json()
        assert overview["categories"]["24"] == "Entertainment"
        assert [e["title"] for e in _planned(client)[:2]] == ["FIRST", "SECOND"]

        skip = client.post(f"/api/gta/clips/{first.id}/posting", json={"skip": True})
        assert skip.json()["uploads"]["youtube"]["status"] == "skipped"
        assert skip.json()["uploads"]["instagram"]["status"] == "skipped"
        assert _planned(client)[0]["title"] == "SECOND"
        clips = {c["id"]: c for c in client.get("/api/gta/clips").json()}
        assert clips[first.id]["uploads"]["youtube"]["status"] == "skipped"
        assert clips[second.id]["uploads"] == {}

        allow = client.post(f"/api/gta/clips/{first.id}/posting", json={"skip": False})
        assert allow.json()["uploads"] == {}
        assert client.post("/api/gta/clips/nope/posting", json={"skip": True}).status_code == 404

    def test_skip_refused_once_on_youtube_and_retry_rules(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        clip = _clip(tmp_path)
        row = store.create_upload(clip.id, status="scheduled", slot_at="2026-11-17T12:00:00Z")
        skip = client.post(f"/api/gta/clips/{clip.id}/posting", json={"skip": True})
        # On YouTube already: only Instagram (not sent yet) is skipped.
        assert skip.status_code == 200
        assert skip.json()["uploads"]["youtube"]["status"] == "scheduled"
        assert skip.json()["uploads"]["instagram"]["status"] == "skipped"
        again = client.post(f"/api/gta/clips/{clip.id}/posting", json={"skip": True})
        assert again.status_code == 200
        ig = store.upload_for_clip(clip.id, "instagram")
        store.update_upload(ig.id, status="posted")
        done = client.post(f"/api/gta/clips/{clip.id}/posting", json={"skip": True})
        assert done.status_code == 409
        assert client.post(f"/api/gta/uploads/{row.id}/retry").status_code == 409
        store.update_upload(row.id, status="interrupted")
        assert client.post(f"/api/gta/uploads/{row.id}/retry").status_code == 200
        assert store.upload_for_clip(clip.id) is None
        assert client.post("/api/gta/uploads/nope/retry").status_code == 404

    def test_bad_schedule_rejected(self, client: TestClient) -> None:
        bad = client.put("/api/gta/settings", json={"posting": {"slots": ["25:00"]}})
        assert bad.status_code == 400
        assert bad.json()["detail"] == "Slot '25:00' is not a valid time of day"
        assert client.get("/api/gta/settings").json()["posting"]["enabled"] is False

    def test_run_now_is_safe_without_a_worker(self, client: TestClient) -> None:
        assert client.post("/api/gta/posting/run").json() == {"running": False}


class TestOAuthReturn:
    def test_callback_page_returns_to_gta_only_for_known_pages(self) -> None:
        from autoclip.api import live as live_api

        assert "/gta?connected=1" in live_api._callback_page("connected", None, "/gta")
        assert "/live?connected=1" in live_api._callback_page("connected", None)
        evil = live_api._callback_page("connected", None, "https://evil.example")
        assert "evil" not in evil and "/live?connected=1" in evil

    def test_start_records_return_page(self, client: TestClient, fake_keyring) -> None:
        from autoclip.api import live as live_api

        try:
            tk.save_client_credentials("youtube", "cid", "secret")
            response = client.post("/api/live/uploads/youtube/start?return_to=/gta")
            assert response.status_code == 200
            assert "accounts.google.com" in response.json()["redirect_url"]
            assert live_api._return_pages["youtube"] == "/gta"
            # A bad state still lands on the page the user came from.
            page = client.get("/api/live/uploads/youtube/callback?code=x&state=wrong")
            assert page.status_code == 400 and "/gta" in page.text
            client.post("/api/live/uploads/youtube/start?return_to=https://evil.example")
            assert live_api._return_pages["youtube"] == "/live"
        finally:
            live_api._pending_states.clear()
            live_api._return_pages.clear()
