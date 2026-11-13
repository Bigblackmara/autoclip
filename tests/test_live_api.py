"""REST contract tests for the Live Watch API (A10).

TestClient against the real app with AUTOCLIP_NO_WORKER=1 (the manager is
therefore NOT started — the lifespan hook skips it), plus an injected fake
manager standing in for the singleton. The frozen contract under test:

- every handler 503s with "Live Watch is not running." while unstarted;
- with a started manager: status/stream endpoints speak the manager's
  snapshot dialect translated to the api.ts LiveStatus/LiveStream shapes;
- clips/uploads endpoints hit the real live store (fresh DB per test);
- upload errors map RightsNotAccepted→400, NotLinkedError→409,
  UploadApiError→502, and every refusal still persists a 'skipped' row;
- the OAuth start/callback pair round-trips the CSRF state;
- settings GET/PUT persist through config.json's ``live`` section.

No real capture, no network: the fake manager's methods record calls and
return canned snapshots; A7's uploader functions are monkeypatched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from autoclip import app as app_module
from autoclip.app import create_app
from autoclip.live import store
from autoclip.live.models import LiveClipRow, LiveStreamRow
from fastapi.testclient import TestClient

# --------------------------------------------------------------------------
# The fake manager — records calls; snapshots shaped like the real one
# --------------------------------------------------------------------------


class FakeManager:
    """Stands in for ``..live.manager.manager`` (the module singleton).

    ``started`` and ``status()``/``stream_snapshot()`` mirror the real
    manager's shapes exactly; every mutating method records the call so
    the tests can assert the router delegated with the right arguments.
    """

    def __init__(self) -> None:
        self.started = False
        self.calls: list[tuple[str, tuple, dict]] = []
        self._rows: dict[str, LiveStreamRow] = {}
        self._snapshots: dict[str, dict] = {}

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    # -- the manager's async surface ---------------------------------------
    async def add_stream(self, url: str, label: str = "", *, enabled: bool = True) -> str:
        self._record("add_stream", url, label, enabled=enabled)
        if not url.strip():
            raise ValueError("Stream URL must not be empty.")
        sid = f"stream-{len(self._rows) + 1}"
        self._rows[sid] = LiveStreamRow(id=sid, url=url, label=label, enabled=enabled)
        return sid

    async def remove_stream(self, stream_id: str) -> None:
        self._record("remove_stream", stream_id)
        if stream_id not in self._rows:
            self._runtime_error(stream_id)
        self._rows.pop(stream_id, None)
        self._snapshots.pop(stream_id, None)

    async def pause_stream(self, stream_id: str) -> None:
        self._record("pause_stream", stream_id)
        if stream_id not in self._rows:
            self._runtime_error(stream_id)
        self._rows[stream_id].enabled = False

    async def resume_stream(self, stream_id: str) -> None:
        self._record("resume_stream", stream_id)
        if stream_id not in self._rows:
            self._runtime_error(stream_id)
        self._rows[stream_id].enabled = True

    def _runtime_error(self, stream_id: str) -> None:
        raise KeyError(stream_id)

    # -- snapshots ---------------------------------------------------------
    def status(self) -> dict:
        return {
            "enabled": True,
            "running": self.started,
            "watching": 2,
            "streams": [self.stream_snapshot(sid) for sid in self._rows],
            "caps": {
                "clip_queue_max": 10,
                "clip_queue_depth": 3,
                "clip_queue_workers": 1,
                "max_concurrent_streams": 5,
                "pending_clip_bytes": 2 * 1024 * 1024,
                "pending_cap_bytes": 10 * 1024 * 1024,
            },
        }

    def stream_snapshot(self, stream_id: str) -> dict:
        if stream_id not in self._rows:
            raise KeyError(stream_id)
        row = self._rows[stream_id]
        state = self._snapshots.get(stream_id, {}).get("state", "waiting_for_live")
        snapshot: dict[str, Any] = {
            "id": stream_id,
            "url": row.url,
            "label": row.label,
            "enabled": row.enabled,
            "state": state,
            "detail": self._snapshots.get(stream_id, {}).get("detail", ""),
            "capturing": False,
        }
        if row.last_error:
            snapshot["error"] = row.last_error
        return snapshot


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def fake_manager() -> FakeManager:
    return FakeManager()


@pytest.fixture
def client(
    autoclip_home, fake_manager: FakeManager, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    import autoclip.api.live as live_api_mod
    import autoclip.live.manager as manager_mod

    monkeypatch.setattr(manager_mod, "manager", fake_manager, raising=False)
    # The router imports the singleton INSIDE each handler, so patching the
    # module attribute is what the handler re-import picks up.
    monkeypatch.setattr("autoclip.live.manager.manager", fake_manager, raising=False)
    live_api_mod._pending_states.clear()
    db_init()
    with TestClient(create_app()) as test_client:
        yield test_client


def db_init() -> None:
    from autoclip import db

    db.init()


@pytest.fixture
def started_client(
    client: TestClient, fake_manager: FakeManager
) -> Iterator[tuple[TestClient, FakeManager]]:
    fake_manager.started = True
    sid = "stream-1"
    fake_manager._rows[sid] = LiveStreamRow(id=sid, url="https://twitch.tv/x", label="X")
    # The store's own FK needs the stream row too: clips reference it in
    # sqlite (live_clips.stream_id → live_streams.id), and the fake manager
    # row above only exists in memory.
    store.upsert_stream(LiveStreamRow(id=sid, url="https://twitch.tv/x", label="X"))
    yield client, fake_manager


# --------------------------------------------------------------------------
# 503 gate: unstarted manager
# --------------------------------------------------------------------------


class TestNotRunning:
    """AUTOCLIP_NO_WORKER=1 → the lifespan never starts the manager."""

    def test_status_is_503_with_the_frozen_detail(self, client: TestClient):
        response = client.get("/api/live/status")
        assert response.status_code == 503
        assert response.json() == {"detail": "Live Watch is not running."}

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("GET", "/api/live/streams", None),
            ("POST", "/api/live/streams", {"url": "https://twitch.tv/x", "label": ""}),
            ("DELETE", "/api/live/streams/s1", None),
            ("POST", "/api/live/streams/s1/pause", None),
            ("POST", "/api/live/streams/s1/resume", None),
        ],
    )
    def test_manager_endpoints_503(
        self, client: TestClient, method: str, path: str, body: dict | None
    ):
        response = client.request(method, path, json=body)
        assert response.status_code == 503
        assert "not running" in response.json()["detail"]

    def test_store_endpoints_work_even_while_unstarted(self, client: TestClient):
        """Clips/uploads read the live store, not the manager — they keep
        working (the wall renders stale history while the engine is off)."""
        assert client.get("/api/live/clips").status_code == 200
        assert client.get("/api/live/clips/nope").status_code == 404
        assert client.get("/api/live/uploads").status_code == 200


# --------------------------------------------------------------------------
# Status + streams
# --------------------------------------------------------------------------


class TestStatus:
    def test_shape_matches_api_ts(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        body = client.get("/api/live/status").json()
        assert set(body) == {"enabled", "watching", "clipping", "caps"}
        assert set(body["caps"]) == {
            "max_concurrent_streams",
            "clip_queue_len",
            "clip_queue_max",
            "disk_used_mb",
        }
        assert body["watching"] == 2
        assert body["clipping"] == 0
        assert body["caps"]["clip_queue_len"] == 3
        assert body["caps"]["clip_queue_max"] == 10
        assert body["caps"]["disk_used_mb"] == 2.0

    def test_clipping_is_derived_from_stream_states(
        self, started_client: tuple[TestClient, FakeManager]
    ):
        client, fake = started_client
        fake._snapshots["stream-1"] = {"state": "clipping", "detail": "candidate detected"}
        body = client.get("/api/live/status").json()
        assert body["clipping"] == 1


class TestStreams:
    def test_add_returns_the_row_in_api_ts_shape(
        self, started_client: tuple[TestClient, FakeManager]
    ):
        client, fake = started_client
        response = client.post(
            "/api/live/streams", json={"url": "https://twitch.tv/y", "label": "Y"}
        )
        assert response.status_code == 201
        body = response.json()
        assert set(body) == {
            "id",
            "url",
            "label",
            "enabled",
            "state",
            "state_detail",
            "last_error",
            "created_at",
            "updated_at",
        }
        assert body["url"] == "https://twitch.tv/y"
        assert body["state"] == "waiting_for_live"
        assert fake.calls[-1][0] == "add_stream"

    def test_add_rejects_an_empty_url(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.post("/api/live/streams", json={"url": "   "})
        assert response.status_code == 400

    def test_list_uses_manager_snapshots(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        body = client.get("/api/live/streams").json()
        assert len(body) == 1
        assert body[0]["state_detail"] == ""

    def test_pause_resume_flow(self, started_client: tuple[TestClient, FakeManager]):
        client, fake = started_client
        paused = client.post("/api/live/streams/stream-1/pause").json()
        assert paused["enabled"] is False
        resumed = client.post("/api/live/streams/stream-1/resume").json()
        assert resumed["enabled"] is True
        assert [c[0] for c in fake.calls[-2:]] == ["pause_stream", "resume_stream"]

    def test_remove_is_204(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.delete("/api/live/streams/stream-1")
        assert response.status_code == 204
        assert client.get("/api/live/streams").json() == []

    def test_unknown_stream_is_404(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        assert client.delete("/api/live/streams/missing").status_code == 404
        assert client.post("/api/live/streams/missing/pause").status_code == 404
        assert client.post("/api/live/streams/missing/resume").status_code == 404


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class TestSettings:
    def test_get_returns_defaults_when_unset(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        body = client.get("/api/live/settings").json()
        assert body["enabled"] is True
        assert body["rolling_buffer"]["segment_s"] == 6.0
        assert body["workers"]["clip_queue_max"] == 10
        # The api.ts LiveSettingsData sections all present
        assert set(body) == {
            "enabled",
            "rolling_buffer",
            "rate_limit",
            "detection",
            "uploads",
            "workers",
        }

    def test_put_persists_the_whole_section(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        tree = client.get("/api/live/settings").json()
        tree["enabled"] = False
        tree["workers"]["clip_queue_max"] = 5
        tree["rolling_buffer"]["retain_minutes"] = 9.5

        response = client.put("/api/live/settings", json=tree)
        assert response.status_code == 200
        assert response.json()["enabled"] is False
        assert response.json()["workers"]["clip_queue_max"] == 5

        # Persisted: a fresh GET re-reads config.json's live section.
        again = client.get("/api/live/settings").json()
        assert again["enabled"] is False
        assert again["rolling_buffer"]["retain_minutes"] == 9.5
        assert again["workers"]["clip_queue_max"] == 5

    def test_put_rejects_an_invalid_tree(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.put("/api/live/settings", json={"enabled": "not-a-bool"})
        assert response.status_code == 400
        assert "Invalid live settings" in response.json()["detail"]

    def test_put_does_not_touch_the_rest_of_config(
        self, started_client: tuple[TestClient, FakeManager], autoclip_home: Path
    ):
        client, _ = started_client
        from autoclip import config, paths

        client.put("/api/settings", json={"whisper": {"model": "large-v3"}})  # main settings
        tree = client.get("/api/live/settings").json()
        tree["workers"]["ram_budget_gb"] = 16
        client.put("/api/live/settings", json=tree)

        raw = json.loads(paths.config_path().read_text(encoding="utf-8"))
        assert raw["whisper"]["model"] == "large-v3"  # untouched
        assert raw["live"]["workers"]["ram_budget_gb"] == 16
        assert config.load().whisper.model == "large-v3"


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


def _make_clip(status: str = "ready", title: str = "Moment", **over: Any) -> str:
    # Merge into a dict first: several tests pass id=/rendered_path= overrides
    # that would collide with the defaults as literal kwargs.
    values: dict[str, Any] = {
        "id": "clip-1",
        "stream_id": "stream-1",
        "status": status,
        "title": title,
        "score": 88,
        "hook": "you won't believe",
        "clip_start_s": 10.0,
        "clip_end_s": 45.0,
        "trigger_start_s": 30.0,
        "trigger_end_s": 33.0,
        "rendered_path": "",
    }
    values.update(over)
    row = LiveClipRow(**values)
    store.create_clip(row)
    return row.id


class TestClips:
    def test_list_and_get(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        clip_id = _make_clip()
        listed = client.get("/api/live/clips").json()
        assert len(listed) == 1
        body = client.get(f"/api/live/clips/{clip_id}").json()
        assert set(body) == {
            "id",
            "stream_id",
            "status",
            "title",
            "score",
            "reason",
            "hook",
            "trigger_start_s",
            "trigger_end_s",
            "clip_start_s",
            "clip_end_s",
            "caption_style",
            "ratio",
            "error",
            "created_at",
            "updated_at",
        }
        assert body["score"] == 88
        assert body["hook"] == "you won't believe"

    def test_limit_is_honoured(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        for i in range(5):
            _make_clip(title=f"c{i}", id=f"clip-{i}")
        assert len(client.get("/api/live/clips?limit=3").json()) == 3

    def test_missing_clip_is_404(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        assert client.get("/api/live/clips/nope").status_code == 404
        assert client.post("/api/live/clips/nope/discard").status_code == 404
        assert client.get("/api/live/clips/nope/media").status_code == 404

    def test_discard_marks_rejected_and_returns_the_row(
        self, started_client: tuple[TestClient, FakeManager]
    ):
        client, _ = started_client
        clip_id = _make_clip()
        response = client.post(f"/api/live/clips/{clip_id}/discard")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "rejected"
        assert body["reason"] == "discarded by user"
        # Persisted:
        assert store.get_clip(clip_id).status == "rejected"

    def test_media_404s_unless_ready_and_present(
        self, started_client: tuple[TestClient, FakeManager], tmp_path: Path
    ):
        client, _ = started_client
        rendering_id = _make_clip(status="rendering")
        assert client.get(f"/api/live/clips/{rendering_id}/media").status_code == 404

        media = tmp_path / "final.mp4"
        media.write_bytes(b"00" * 64)
        ready_id = _make_clip(id="clip-ready", status="ready", rendered_path=str(media))
        response = client.get(f"/api/live/clips/{ready_id}/media")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("video/mp4")

        gone_id = _make_clip(id="clip-2", status="ready", rendered_path=str(tmp_path / "nope.mp4"))
        assert client.get(f"/api/live/clips/{gone_id}/media").status_code == 404


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------


class _UploadFaker:
    """Monkeypatch target standing in for A7's upload functions."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        from autoclip.live.uploads import UploadResult

        self.result = result or UploadResult(
            platform="youtube", media_id="vid123", url="https://youtu.be/vid123"
        )
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, path: Path, meta) -> Any:
        self.calls.append((str(path), meta.title))
        if self.error is not None:
            raise self.error
        return self.result


class TestUploads:
    @pytest.fixture
    def ready_clip(self, started_client: tuple[TestClient, FakeManager]) -> Path:
        media = Path(self_media_dir()) / "final.mp4"
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"11" * 32)
        _make_clip(id="clip-ok", status="ready", rendered_path=str(media))
        return media

    def test_list_is_empty_then_not(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        assert client.get("/api/live/uploads").json() == []
        from autoclip.live.models import LiveUploadRow

        _make_clip()  # live_uploads.clip_id FKs live_clips.id
        store.create_upload(LiveUploadRow(id="u1", clip_id="clip-1", platform="youtube"))
        body = client.get("/api/live/uploads").json()
        assert body[0]["id"] == "u1"
        assert set(body[0]) == {
            "id",
            "clip_id",
            "platform",
            "status",
            "upload_url",
            "platform_media_id",
            "error",
            "created_at",
            "updated_at",
        }

    def test_upload_success_persists_a_done_row(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
        ready_clip: Path,
    ):
        client, _ = started_client
        faker = _UploadFaker()
        monkeypatch.setattr("autoclip.live.uploads.upload_youtube", faker, raising=False)
        monkeypatch.setattr("autoclip.live.uploads.youtube.upload_youtube", faker, raising=True)

        response = client.post("/api/live/clips/clip-ok/upload", json={"platform": "youtube"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "done"
        assert body["platform_media_id"] == "vid123"
        assert body["upload_url"] == "https://youtu.be/vid123"
        assert faker.calls and faker.calls[0][0] == str(ready_clip)
        # Persisted too:
        rows = store.list_uploads(clip_id="clip-ok")
        assert rows[0].status == "done"

    def test_rights_refusal_is_400_and_skips_row(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
        ready_clip: Path,
    ):
        from autoclip.live.uploads.tokens import RightsNotAccepted

        client, _ = started_client
        monkeypatch.setattr(
            "autoclip.live.uploads.upload_youtube",
            _UploadFaker(error=RightsNotAccepted("locked")),
            raising=False,
        )
        response = client.post("/api/live/clips/clip-ok/upload", json={"platform": "youtube"})
        assert response.status_code == 400
        assert "rights" in response.json()["detail"].lower()
        rows = store.list_uploads(clip_id="clip-ok")
        assert rows[0].status == "skipped"

    def test_not_linked_is_409(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
        ready_clip: Path,
    ):
        from autoclip.live.uploads.tokens import NotLinkedError

        client, _ = started_client
        monkeypatch.setattr(
            "autoclip.live.uploads.upload_instagram",
            _UploadFaker(error=NotLinkedError("instagram account is not connected")),
            raising=False,
        )
        response = client.post("/api/live/clips/clip-ok/upload", json={"platform": "instagram"})
        assert response.status_code == 409
        rows_status = [r.status for r in store.list_uploads(clip_id="clip-ok")]
        assert rows_status == ["skipped"]

    def test_platform_api_error_is_502(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
        ready_clip: Path,
    ):
        from autoclip.live.uploads.tokens import UploadApiError

        client, _ = started_client
        monkeypatch.setattr(
            "autoclip.live.uploads.upload_youtube",
            _UploadFaker(error=UploadApiError("HTTP 500: boom")),
            raising=False,
        )
        response = client.post("/api/live/clips/clip-ok/upload", json={"platform": "youtube"})
        assert response.status_code == 502
        assert [r.status for r in store.list_uploads(clip_id="clip-ok")] == ["failed"]

    def test_upload_404s_a_missing_clip(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.post("/api/live/clips/nope/upload", json={"platform": "youtube"})
        assert response.status_code == 404

    def test_upload_refuses_a_non_ready_clip(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        _make_clip(id="clip-busy", status="rendering")
        response = client.post("/api/live/clips/clip-busy/upload", json={"platform": "youtube"})
        assert response.status_code == 409

    def test_unknown_platform_is_404(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.post("/api/live/clips/clip-ok/upload", json={"platform": "tiktok"})
        assert response.status_code == 422  # Literal rejects before we see it


def self_media_dir() -> str:
    """A stable scratch dir under the test's autoclip home."""
    from autoclip import paths

    directory = paths.root() / "media_fixture"
    return str(directory)


# --------------------------------------------------------------------------
# OAuth start / callback / credentials / attestation
# --------------------------------------------------------------------------


class TestOAuthFlow:
    def test_start_returns_a_redirect_url_with_state(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
    ):
        client, _ = started_client
        captured: dict = {}

        def fake_auth_url(platform, *, redirect_uri: str, state: str | None = None):
            captured.update(platform=platform, redirect_uri=redirect_uri, state=state)
            return f"https://accounts.example/{platform}?state={state}"

        monkeypatch.setattr("autoclip.live.uploads.authorization_url", fake_auth_url, raising=False)
        response = client.post("/api/live/uploads/youtube/start")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"redirect_url"}
        assert body["redirect_url"].startswith("https://accounts.example/youtube")
        assert captured["redirect_uri"].endswith("/api/live/uploads/youtube/callback")

    def test_callback_rejects_a_bad_state(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.get(
            "/api/live/uploads/youtube/callback",
            params={"code": "abc", "state": "forged"},
        )
        assert response.status_code == 400
        assert "expired" in response.text

    def test_callback_exchanges_and_redirects(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
    ):
        client, _ = started_client
        exchanged: dict = {}

        class FakeBundle:
            account = "test"
            access_token = "t"

        async def fake_exchange(platform, code, *, redirect_uri, client=None):
            exchanged.update(platform=platform, code=code)
            return FakeBundle()

        def fake_auth_url(platform, *, redirect_uri, state=None):
            return f"https://accounts.example/{platform}?state={state}"

        monkeypatch.setattr("autoclip.live.uploads.exchange_code", fake_exchange, raising=False)
        monkeypatch.setattr("autoclip.live.uploads.authorization_url", fake_auth_url, raising=False)
        start = client.post("/api/live/uploads/instagram/start").json()
        import urllib.parse

        state = urllib.parse.parse_qs(urllib.parse.urlparse(start["redirect_url"]).query)["state"][
            0
        ]

        response = client.get(
            "/api/live/uploads/instagram/callback",
            params={"code": "xyz", "state": state},
        )
        assert response.status_code == 200
        assert "/live?connected=1" in response.text
        assert exchanged == {"platform": "instagram", "code": "xyz"}

    def test_callback_without_code_is_400(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.get("/api/live/uploads/youtube/callback")
        assert response.status_code == 400

    def test_unknown_platform_paths_are_404(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        assert client.post("/api/live/uploads/tiktok/start").status_code == 404
        assert client.get("/api/live/uploads/tiktok/status").status_code == 404


class TestCredentialsAndAttestation:
    def test_credentials_are_stored_and_never_echoed(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
        fake_keyring,
    ):
        client, _ = started_client
        saved: dict = {}

        def fake_save(platform, client_id, client_secret):
            saved.update(platform=platform, client_id=client_id, client_secret=client_secret)

        monkeypatch.setattr(
            "autoclip.live.uploads.save_client_credentials", fake_save, raising=False
        )
        response = client.post(
            "/api/live/uploads/youtube/credentials",
            json={"client_id": "my-id", "client_secret": "my-secret"},
        )
        assert response.status_code == 200
        assert response.json() == {"saved": True}
        assert saved["client_secret"] == "my-secret"
        assert "my-secret" not in response.text  # never echoed

    def test_platform_status_maps_to_a7(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
    ):
        client, _ = started_client

        async def fake_status(*, client=None):
            return {"connected": True, "account": "My Channel"}

        monkeypatch.setattr("autoclip.live.uploads.youtube_status", fake_status, raising=False)
        body = client.get("/api/live/uploads/youtube/status").json()
        assert body == {"connected": True, "account": "My Channel"}

    def test_attestation_round_trip(
        self,
        started_client: tuple[TestClient, FakeManager],
        monkeypatch: pytest.MonkeyPatch,
        fake_keyring,
    ):
        client, _ = started_client
        # GET before acceptance
        assert client.get("/api/live/uploads/attestation").json() == {
            "accepted": False,
            "accepted_at": None,
        }

        posted = client.post("/api/live/uploads/attestation", json={"accepted": True})
        assert posted.status_code == 200
        body = posted.json()
        assert body["accepted"] is True
        assert body["accepted_at"]  # an ISO date string

        # GET reflects the persisted state
        again = client.get("/api/live/uploads/attestation").json()
        assert again["accepted"] is True
        assert again["accepted_at"] == body["accepted_at"]

    def test_attestation_rejects_false(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        response = client.post("/api/live/uploads/attestation", json={"accepted": False})
        assert response.status_code == 400


# --------------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------------


def _read_sse(response, want_frames: int, timeout: float) -> str:
    """Bounded SSE read: stop early once ``want_frames`` complete frames

    (blank-line-terminated) have arrived — the feed itself never ends, so a
    plain join(iter_text()) would hang forever. Falls back to whatever
    arrived when the timeout elapses, still a valid partial body.
    """
    import time as _time

    body = ""
    frames = 0
    deadline = _time.monotonic() + timeout
    for chunk in response.iter_text():
        body += chunk
        frames = body.count("\n\n")
        if frames >= want_frames or _time.monotonic() > deadline:
            break
    return body


class TestSSE:
    def test_global_feed_emits_snapshot_then_mapped_events(
        self, started_client: tuple[TestClient, FakeManager]
    ):
        client, _ = started_client
        with client.stream("GET", "/api/live/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = _read_sse(response, want_frames=3, timeout=3.0)

        assert "event: snapshot" in body
        assert '"resync": true' in body

    def test_stream_feed_includes_clip_channel_and_state(
        self, started_client: tuple[TestClient, FakeManager]
    ):
        client, _ = started_client
        with client.stream("GET", "/api/live/streams/stream-1/events") as response:
            assert response.status_code == 200
            body = _read_sse(response, want_frames=3, timeout=3.0)
        assert "event: snapshot" in body

    def test_event_name_mapping_table(self):
        from autoclip.api.live import _frontend_event_name

        # The six names the frontend listens for are exactly what can come out.
        for source, expected in [
            ("state", "state"),
            ("clip", "clip"),
            ("clip_ready", "clip_status"),
            ("clip_detecting", "clip_status"),
            ("upload", "upload"),
            ("heartbeat", "heartbeat"),
        ]:
            assert _frontend_event_name(source) == expected
        # Unknown watcher events are dropped, not passed through.
        assert _frontend_event_name("totally_new_event") is None


class TestWireShapesMatchFrontend:
    """The response shapes must be json-identical to the api.ts interfaces.

    F1/F2/F3 coded against TypeScript types; these assertions pin the
    Python wire to the exact same field sets so a rename can't slip in.
    """

    def test_stream_fields(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        body = client.post(
            "/api/live/streams", json={"url": "https://twitch.tv/shape", "label": ""}
        ).json()
        assert set(body) == {
            "id",
            "url",
            "label",
            "enabled",
            "state",
            "state_detail",
            "last_error",
            "created_at",
            "updated_at",
        }

    def test_settings_sections(self, started_client: tuple[TestClient, FakeManager]):
        client, _ = started_client
        body = client.get("/api/live/settings").json()
        assert set(body["rolling_buffer"]) == {
            "segment_s",
            "retain_minutes",
            "max_disk_mb",
            "pre_roll_s",
            "post_roll_s",
            "judge_context_s",
        }
        assert set(body["rate_limit"]) == {
            "judge_per_minute",
            "judge_per_hour",
            "judge_min_score",
            "cooldown_s",
            "judge_per_day",
        }
        assert set(body["detection"]) == {"audio", "chat"}
        assert set(body["uploads"]) == {"youtube", "instagram"}
        assert set(body["workers"]) == {
            "max_concurrent_streams",
            "clip_queue_workers",
            "clip_queue_max",
            "cpu_budget_percent",
            "ram_budget_gb",
        }
