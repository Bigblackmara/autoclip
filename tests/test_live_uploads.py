"""Live upload subsystem: token store, OAuth plumbing, YouTube + Instagram flows.

Zero network: every httpx client is a ``MockTransport`` whose handler asserts
the documented protocol (endpoints, headers, byte counts, param names), and
every secret lives in the fake keyring fixture. A test that accidentally
reached the real internet would hit an assertion in the handler long before
it could leak anything or hang.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from autoclip import config
from autoclip.live import uploads
from autoclip.live.uploads import instagram as ig
from autoclip.live.uploads import tokens as tk

YOUTUBE_CLIENT = ("yt-client-id.apps.googleusercontent.com", "yt-client-secret")
IG_CLIENT = ("ig-app-id", "ig-app-secret")

IG_USER_ID = "17801435778192284"
IG_USERNAME = "myhandle"


@pytest.fixture
def linked_youtube(fake_keyring) -> None:
    uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
    uploads.save_tokens("youtube", tk.TokenBundle(access_token="yt-at", refresh_token="yt-rt"))


@pytest.fixture
def linked_instagram(fake_keyring) -> None:
    uploads.save_client_credentials("instagram", *IG_CLIENT)
    uploads.save_tokens("instagram", tk.TokenBundle(access_token="ig-at"))


@pytest.fixture
def attested(fake_keyring) -> None:
    uploads.accept_rights()


@pytest.fixture
def clip_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"clipbytes-" * 1000)
    return path


class RecordingTransport(httpx.MockTransport):
    """MockTransport that records every request for protocol assertions."""

    def __init__(self, handler) -> None:
        super().__init__(handler)
        self.requests: list[httpx.Request] = []

    def _record(self, request: httpx.Request) -> None:
        self.requests.append(request)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._record(request)
        return super().handle_request(request)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._record(request)
        return await super().handle_async_request(request)


def bodies_of(transport: RecordingTransport) -> list[bytes]:
    """Materialize what each recorded request put on the wire."""
    return [b"".join(part for part in request.stream) for request in transport.requests]


def mock_client(transport: httpx.BaseTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


# --------------------------------------------------------------------------
# Token storage
# --------------------------------------------------------------------------


class TestTokenStore:
    def test_bundle_round_trips_through_the_keyring(self, fake_keyring) -> None:
        bundle = tk.TokenBundle(
            access_token="at", refresh_token="rt", expires_at=1.5, account="chan"
        )
        tk.save_tokens("youtube", bundle)

        assert tk.load_tokens("youtube") == bundle

    def test_platform_keyrings_are_independent(self, fake_keyring) -> None:
        tk.save_tokens("youtube", tk.TokenBundle(access_token="yt"))
        tk.save_tokens("instagram", tk.TokenBundle(access_token="ig"))

        assert tk.load_tokens("youtube").access_token == "yt"
        assert tk.load_tokens("instagram").access_token == "ig"

    def test_secret_blob_never_lands_in_config_json(self, fake_keyring) -> None:
        tk.save_tokens("youtube", tk.TokenBundle(access_token="super-secret-token"))
        config.save(config.load())

        raw = config.paths.config_path().read_text(encoding="utf-8")
        assert "super-secret-token" not in raw
        assert "live_youtube_tokens" not in raw

    def test_missing_tokens_are_none(self, fake_keyring) -> None:
        assert tk.load_tokens("youtube") is None

    def test_corrupt_blob_raises_instead_of_looking_unlinked(self, fake_keyring) -> None:
        config.set_secret(tk.YOUTUBE_TOKENS_KEY, "{not json")

        with pytest.raises(tk.TokenStateError):
            tk.load_tokens("youtube")

    def test_blob_without_access_token_is_corrupt(self, fake_keyring) -> None:
        config.set_secret(tk.YOUTUBE_TOKENS_KEY, json.dumps({"refresh_token": "rt"}))

        with pytest.raises(tk.TokenStateError):
            tk.load_tokens("youtube")

    def test_clear_tokens_disconnects(self, fake_keyring) -> None:
        tk.save_tokens("youtube", tk.TokenBundle(access_token="at"))
        tk.clear_tokens("youtube")

        assert tk.load_tokens("youtube") is None

    def test_keyringless_machine_refuses_instead_of_downgrading(self, no_keyring) -> None:
        # config.set_secret would fall back to plaintext config.json; for
        # account-wide OAuth grants that downgrade is refused outright.
        with pytest.raises(tk.TokenStateError):
            tk.save_tokens("youtube", tk.TokenBundle(access_token="at"))
        assert tk.load_tokens("youtube") is None

    def test_client_credentials_round_trip(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)

        assert uploads.load_client_credentials("youtube") == YOUTUBE_CLIENT

    def test_client_credentials_never_touch_config_json(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
        config.save(config.load())

        raw = config.paths.config_path().read_text(encoding="utf-8")
        assert "yt-client-secret" not in raw

    def test_half_stored_client_credentials_count_as_absent(self, fake_keyring) -> None:
        config.set_secret(tk.YOUTUBE_CLIENT_ID_KEY, "only-id")

        assert uploads.load_client_credentials("youtube") is None

    def test_unknown_platform_is_rejected(self, fake_keyring) -> None:
        with pytest.raises(ValueError):
            tk.save_tokens("tiktok", tk.TokenBundle(access_token="at"))


# --------------------------------------------------------------------------
# Rights-attestation gate
# --------------------------------------------------------------------------


class TestRightsAttestation:
    def test_uploads_are_locked_by_default(self, fake_keyring) -> None:
        assert not uploads.has_rights_attestation()
        with pytest.raises(uploads.RightsNotAccepted):
            uploads.require_rights_attestation()

    def test_acceptance_unlocks_and_records_the_date(self, fake_keyring) -> None:
        stored = uploads.accept_rights()

        assert stored.startswith("accepted:")
        assert uploads.has_rights_attestation()
        uploads.require_rights_attestation()  # no raise

    def test_clear_relocks(self, fake_keyring) -> None:
        uploads.accept_rights()
        uploads.clear_rights()

        assert not uploads.has_rights_attestation()

    def test_non_accepted_values_do_not_unlock(self, fake_keyring) -> None:
        config.set_secret(tk.ATTESTATION_KEY, "declined:2026-11-13")

        assert not uploads.has_rights_attestation()
        with pytest.raises(uploads.RightsNotAccepted):
            uploads.require_rights_attestation()

    async def test_gate_blocks_youtube_before_any_network(
        self, fake_keyring, linked_youtube, clip_file
    ) -> None:
        transport = RecordingTransport(lambda request: httpx.Response(200, json={}))

        with pytest.raises(uploads.RightsNotAccepted):
            await uploads.upload_youtube(
                clip_file, uploads.UploadMeta(title="t"), client=mock_client(transport)
            )
        assert transport.requests == []

    async def test_gate_blocks_instagram_before_any_network(
        self, fake_keyring, linked_instagram, clip_file
    ) -> None:
        transport = RecordingTransport(lambda request: httpx.Response(200, json={}))

        with pytest.raises(uploads.RightsNotAccepted):
            await uploads.upload_instagram(
                clip_file, uploads.UploadMeta(title="t"), client=mock_client(transport)
            )
        assert transport.requests == []


# --------------------------------------------------------------------------
# Authorization URLs
# --------------------------------------------------------------------------


class TestAuthorizationUrls:
    def test_youtube_url_uses_offline_consent_and_upload_scope(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)

        url = uploads.authorization_url("youtube", redirect_uri="http://localhost:5173/cb")

        assert "accounts.google.com/o/oauth2/v2/auth" in url
        assert "scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutube.upload" in url
        assert "access_type=offline" in url
        assert "prompt=consent" in url
        assert "client_id=yt-client-id.apps.googleusercontent.com" in url
        assert "state=" in url

    def test_instagram_url_targets_the_meta_dialog_with_publish_scope(self, fake_keyring) -> None:
        uploads.save_client_credentials("instagram", *IG_CLIENT)

        url = uploads.authorization_url("instagram", redirect_uri="https://a.b/cb")

        assert "facebook.com/v27.0/dialog/oauth" in url
        assert "instagram_content_publish" in url

    def test_urls_fail_closed_without_client_credentials(self, fake_keyring) -> None:
        with pytest.raises(tk.TokenStateError):
            uploads.authorization_url("youtube", redirect_uri="http://localhost/cb")

    def test_explicit_state_round_trips_into_the_url(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)

        url = uploads.authorization_url(
            "youtube", redirect_uri="http://localhost/cb", state="st123"
        )

        assert "state=st123" in url


# --------------------------------------------------------------------------
# Code exchange (A10 wires the redirect; the exchange itself is ours)
# --------------------------------------------------------------------------


class TestCodeExchange:
    async def test_youtube_exchange_stores_offline_tokens(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "oauth2.googleapis.com"
            form = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
            assert form["grant_type"] == "authorization_code"
            assert form["redirect_uri"] == "http%3A%2F%2Flocalhost%2Fcb"
            return httpx.Response(
                200,
                json={
                    "access_token": "yt-at",
                    "refresh_token": "yt-rt",
                    "expires_in": 3600,
                    "scope": "https://www.googleapis.com/auth/youtube.upload",
                },
            )

        bundle = await uploads.exchange_code(
            "youtube",
            "the-code",
            redirect_uri="http://localhost/cb",
            client=mock_client(httpx.MockTransport(handler)),
        )

        assert bundle.access_token == "yt-at"
        assert bundle.refresh_token == "yt-rt"
        assert bundle.expires_at is not None and bundle.expires_at > time.time()
        stored = tk.load_tokens("youtube")
        assert stored is not None and stored.access_token == "yt-at"

    async def test_instagram_exchange_upgrades_to_long_lived_token(self, fake_keyring) -> None:
        uploads.save_client_credentials("instagram", *IG_CLIENT)

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "code=ig-code" in url:
                return httpx.Response(200, json={"access_token": "short", "expires_in": 3600})
            assert "fb_exchange_token=short" in url
            return httpx.Response(200, json={"access_token": "long", "expires_in": 5184000})

        bundle = await uploads.exchange_code(
            "instagram",
            "ig-code",
            redirect_uri="https://a.b/cb",
            client=mock_client(httpx.MockTransport(handler)),
        )

        assert bundle.access_token == "long"
        assert bundle.expires_at is not None

    async def test_exchange_failure_raises_without_leaking_the_code(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="invalid_grant for code SECRET")

        with pytest.raises(tk.UploadApiError) as info:
            await uploads.exchange_code(
                "youtube",
                "SECRET",
                redirect_uri="http://localhost/cb",
                client=mock_client(httpx.MockTransport(handler)),
            )
        assert "SECRET" not in str(info.value)

    async def test_exchange_without_stored_client_credentials_is_refused(
        self, fake_keyring
    ) -> None:
        with pytest.raises(tk.TokenStateError):
            await uploads.exchange_code(
                "youtube",
                "c",
                redirect_uri="http://localhost/cb",
                client=mock_client(httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
            )


# --------------------------------------------------------------------------
# Refresh / freshness
# --------------------------------------------------------------------------


class TestRefresh:
    async def test_youtube_refresh_rotates_access_keeps_refresh_token(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
        old = tk.TokenBundle(access_token="old-at", refresh_token="keep-rt")
        tk.save_tokens("youtube", old)

        def handler(request: httpx.Request) -> httpx.Response:
            body = request.content.decode()
            assert "grant_type=refresh_token" in body
            assert "refresh_token=keep-rt" in body
            assert "client_secret=yt-client-secret" in body
            return httpx.Response(200, json={"access_token": "new-at", "expires_in": 3600})

        bundle = await uploads.refresh_tokens(
            "youtube", old, client=mock_client(httpx.MockTransport(handler))
        )

        assert bundle.access_token == "new-at"
        assert bundle.refresh_token == "keep-rt"
        assert tk.load_tokens("youtube").access_token == "new-at"

    async def test_expired_tokens_trigger_a_refresh(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
        tk.save_tokens(
            "youtube",
            tk.TokenBundle(access_token="stale", refresh_token="rt", expires_at=time.time() - 10),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert "grant_type=refresh_token" in request.content.decode()
            return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})

        bundle = await tk.ensure_fresh("youtube", client=mock_client(httpx.MockTransport(handler)))

        assert bundle.access_token == "fresh"

    async def test_fresh_tokens_skip_the_network_entirely(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
        tk.save_tokens(
            "youtube",
            tk.TokenBundle(
                access_token="still-good",
                refresh_token="rt",
                expires_at=time.time() + 3600,
            ),
        )

        def fail(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no network allowed for fresh tokens")

        bundle = await tk.ensure_fresh("youtube", client=mock_client(httpx.MockTransport(fail)))
        assert bundle.access_token == "still-good"

    async def test_refresh_failure_keeps_last_known_good_bundle(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
        tk.save_tokens(
            "youtube",
            tk.TokenBundle(access_token="old", refresh_token="rt", expires_at=time.time() - 10),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "invalid_grant"})

        with pytest.raises(tk.TokenRefreshError):
            await tk.refresh_tokens("youtube", client=mock_client(httpx.MockTransport(handler)))
        assert tk.load_tokens("youtube").access_token == "old"

    async def test_refresh_without_a_refresh_token_is_a_state_error(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
        tk.save_tokens(
            "youtube",
            tk.TokenBundle(access_token="old", expires_at=time.time() - 10),
        )

        with pytest.raises(tk.TokenStateError):
            await tk.refresh_tokens(
                "youtube",
                client=mock_client(httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
            )

    async def test_unlinked_platform_is_not_linked(self, fake_keyring) -> None:
        with pytest.raises(tk.NotLinkedError):
            await tk.ensure_fresh("instagram", client=None)

    async def test_expired_ig_token_refreshes_via_the_exchange_grant(self, fake_keyring) -> None:
        uploads.save_client_credentials("instagram", *IG_CLIENT)
        tk.save_tokens(
            "instagram",
            tk.TokenBundle(access_token="ig-old", expires_at=time.time() - 10),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert "fb_exchange_token=ig-old" in str(request.url)
            assert request.method == "GET"
            return httpx.Response(200, json={"access_token": "ig-new", "expires_in": 5184000})

        bundle = await tk.refresh_tokens(
            "instagram", client=mock_client(httpx.MockTransport(handler))
        )
        assert bundle.access_token == "ig-new"


# --------------------------------------------------------------------------
# YouTube resumable upload
# --------------------------------------------------------------------------


class TestYouTubeUpload:
    async def test_full_flow_speaks_the_documented_protocol(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        transport = RecordingTransport(
            lambda request: (
                httpx.Response(
                    200,
                    headers={"Location": "https://upload.session/abc?token=t1"},
                )
                if request.method == "POST"
                else httpx.Response(
                    200,
                    json={"id": "vid123", "snippet": {"channelTitle": "My Channel"}},
                )
                if request.method == "PUT"
                else httpx.Response(404)
            )
        )

        result = await uploads.upload_youtube(
            clip_file,
            uploads.UploadMeta(title="Best moment", tags=["gaming"]),
            client=mock_client(transport),
        )

        assert result.platform == "youtube"
        assert result.media_id == "vid123"
        assert result.url == "https://www.youtube.com/video/vid123"
        assert result.account == "My Channel"

        init, put = transport.requests
        assert init.method == "POST"
        assert str(init.url) == (
            "https://www.googleapis.com/upload/youtube/v3/videos"
            "?uploadType=resumable&part=snippet%2Cstatus"
        )
        assert init.headers["Authorization"] == "Bearer yt-at"
        assert init.headers["X-Upload-Content-Type"] == "video/mp4"
        metadata = json.loads(init.content)
        assert metadata["snippet"]["title"] == "Best moment"
        assert metadata["snippet"]["tags"] == ["gaming"]
        assert metadata["status"]["privacyStatus"] == "private"
        assert metadata["status"]["selfDeclaredMadeForKids"] is False

        assert put.method == "PUT"
        assert put.url.host == "upload.session"
        assert put.headers["Content-Type"] == "video/mp4"
        assert bodies_of(transport)[1] == clip_file.read_bytes()

    async def test_retriable_failure_resumes_from_the_range_probe(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        size = clip_file.stat().st_size
        steps = ["503", "probe", "ok"]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200, headers={"Location": "https://upload.session/xyz?token=t1"}
                )
            step = steps.pop(0)
            if step == "503":
                return httpx.Response(503)
            if step == "probe":
                assert request.headers.get("Content-Range") == f"bytes */{size}"
                assert b"".join(request.stream) == b""
                return httpx.Response(308, headers={"Range": "bytes=0-999"})
            return httpx.Response(200, json={"id": "vid9", "snippet": {"channelTitle": "C"}})

        transport = RecordingTransport(handler)

        result = await uploads.upload_youtube(
            clip_file, uploads.UploadMeta(title="t"), client=mock_client(transport)
        )

        assert result.media_id == "vid9"
        assert [r.method for r in transport.requests] == ["POST", "PUT", "PUT", "PUT"]
        sent = bodies_of(transport)
        assert sent[1] == clip_file.read_bytes()  # failing PUT: the whole file
        assert sent[3] == clip_file.read_bytes()[1000:]  # resumed PUT: tail only
        # The first PUT carries the whole file and says so; the resumed PUT
        # MUST declare which bytes it carries (bytes off-(n-1)/n).
        first, resumed = transport.requests[1], transport.requests[3]
        assert "Content-Range" not in first.headers
        assert first.headers["Content-Length"] == str(size)
        assert resumed.headers["Content-Range"] == f"bytes 1000-{size - 1}/{size}"
        assert resumed.headers["Content-Length"] == str(size - 1000)

    async def test_probe_reporting_everything_received_fetches_the_result(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        size = clip_file.stat().st_size
        steps = ["503", "probe-complete", "final"]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200, headers={"Location": "https://upload.session/xyz?token=t1"}
                )
            step = steps.pop(0)
            if step == "503":
                return httpx.Response(503)
            if step == "probe-complete":
                return httpx.Response(200, json={"id": "ignored"})
            assert request.headers.get("Content-Range") == f"bytes */{size}"
            assert b"".join(request.stream) == b""
            return httpx.Response(200, json={"id": "vid10", "snippet": {"channelTitle": "C"}})

        transport = RecordingTransport(handler)
        result = await uploads.upload_youtube(
            clip_file, uploads.UploadMeta(title="t"), client=mock_client(transport)
        )
        assert result.media_id == "vid10"

    async def test_transport_error_after_the_bytes_landed_resumes_not_fails(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        """A read timeout on the final PUT must not record a failure.

        Google may already hold the whole file (the video exists); failing
        here made the user retry and the clip was posted twice. The open
        session is probed instead and the finished resource fetched.
        """
        size = clip_file.stat().st_size
        steps = ["timeout", "probe-complete", "final"]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200, headers={"Location": "https://upload.session/xyz?token=t1"}
                )
            step = steps.pop(0)
            if step == "timeout":
                raise httpx.ReadTimeout("read timed out", request=request)
            if step == "probe-complete":
                assert request.headers.get("Content-Range") == f"bytes */{size}"
                return httpx.Response(200, json={"id": "vid11"})
            assert request.headers.get("Content-Range") == f"bytes */{size}"
            return httpx.Response(200, json={"id": "vid11", "snippet": {"channelTitle": "C"}})

        transport = RecordingTransport(handler)
        result = await uploads.upload_youtube(
            clip_file, uploads.UploadMeta(title="t"), client=mock_client(transport)
        )
        assert result.media_id == "vid11"
        # One session only: a second POST would be a second video.
        assert [r.method for r in transport.requests] == ["POST", "PUT", "PUT", "PUT"]

    async def test_transport_errors_on_every_attempt_still_fail(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    200, headers={"Location": "https://upload.session/xyz?token=t1"}
                )
            if request.headers.get("Content-Range", "").startswith("bytes */"):
                return httpx.Response(308)  # probe: nothing stored yet
            raise httpx.ConnectError("connection reset", request=request)

        with pytest.raises(tk.UploadApiError, match="transport failed"):
            await uploads.upload_youtube(
                clip_file,
                uploads.UploadMeta(title="t"),
                client=mock_client(RecordingTransport(handler)),
            )

    async def test_non_retriable_rejection_surfaces_the_redacted_error(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": "https://upload.session/bad"})
            return httpx.Response(403, text="quotaExceeded")

        with pytest.raises(tk.UploadApiError) as info:
            await uploads.upload_youtube(
                clip_file,
                uploads.UploadMeta(title="t"),
                client=mock_client(httpx.MockTransport(handler)),
            )
        assert "quotaExceeded" in str(info.value)
        assert "yt-at" not in str(info.value)

    async def test_missing_location_header_is_a_protocol_error(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        with pytest.raises(tk.UploadApiError, match="Location"):
            await uploads.upload_youtube(
                clip_file,
                uploads.UploadMeta(title="t"),
                client=mock_client(httpx.MockTransport(handler)),
            )

    async def test_missing_file_fails_before_any_network(
        self, fake_keyring, attested, linked_youtube, tmp_path
    ) -> None:
        transport = RecordingTransport(lambda r: httpx.Response(200, json={}))

        with pytest.raises(tk.UploadApiError, match="cannot read clip file"):
            await uploads.upload_youtube(
                tmp_path / "nope.mp4",
                uploads.UploadMeta(title="t"),
                client=mock_client(transport),
            )
        assert transport.requests == []

    async def test_unlinked_account_is_not_linked(self, fake_keyring, attested, clip_file) -> None:
        with pytest.raises(tk.NotLinkedError):
            await uploads.upload_youtube(
                clip_file,
                uploads.UploadMeta(title="t"),
                client=mock_client(httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
            )

    async def test_upload_persists_the_discovered_channel(
        self, fake_keyring, attested, linked_youtube, clip_file
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": "https://upload.session/z"})
            return httpx.Response(200, json={"id": "v", "snippet": {"channelTitle": "New Title"}})

        await uploads.upload_youtube(
            clip_file,
            uploads.UploadMeta(title="t"),
            client=mock_client(httpx.MockTransport(handler)),
        )

        stored = tk.load_tokens("youtube")
        assert stored is not None and stored.account == "New Title"


class TestYouTubeStatus:
    async def test_unlinked_costs_no_network(self, fake_keyring) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no network for an unlinked account")

        status = await uploads.youtube_status(client=mock_client(httpx.MockTransport(fail)))
        assert status == {"connected": False, "account": None}

    async def test_cached_channel_avoids_the_channels_lookup(
        self, fake_keyring, linked_youtube
    ) -> None:
        tk.save_tokens(
            "youtube",
            tk.TokenBundle(access_token="yt-at", refresh_token="yt-rt", account="Cached"),
        )

        def fail(request: httpx.Request) -> httpx.Response:
            raise AssertionError("a cached account must not hit the channels endpoint")

        status = await uploads.youtube_status(client=mock_client(httpx.MockTransport(fail)))
        assert status == {"connected": True, "account": "Cached"}

    async def test_first_status_discovers_and_persists_the_channel(
        self, fake_keyring, linked_youtube
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.host == "www.googleapis.com"
            assert request.url.params["mine"] == "true"
            assert request.url.params["part"] == "snippet"
            return httpx.Response(
                200, json={"items": [{"snippet": {"title": "Discovered Channel"}}]}
            )

        status = await uploads.youtube_status(client=mock_client(httpx.MockTransport(handler)))
        assert status == {"connected": True, "account": "Discovered Channel"}
        stored = tk.load_tokens("youtube")
        assert stored is not None and stored.account == "Discovered Channel"

    async def test_dead_grant_reports_not_connected(self, fake_keyring) -> None:
        uploads.save_client_credentials("youtube", *YOUTUBE_CLIENT)
        tk.save_tokens(
            "youtube",
            tk.TokenBundle(
                access_token="dead",
                refresh_token="dead-rt",
                expires_at=time.time() - 10,
            ),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "invalid_grant"})

        status = await uploads.youtube_status(client=mock_client(httpx.MockTransport(handler)))
        assert status == {"connected": False, "account": None}


# --------------------------------------------------------------------------
# Instagram resumable Reels flow
# --------------------------------------------------------------------------


def ig_flow_handler(states: list[str], *, publish_id: str = "pub-123"):
    """Build a handler speaking the full documented IG flow against a fixed
    page/IG identity; ``states`` drains one entry per container status poll."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/v27.0/{IG_USER_ID}/media" and request.method == "POST":
            assert request.url.params["media_type"] == "REELS"
            assert request.url.params["upload_type"] == "resumable"
            return httpx.Response(
                200,
                json={
                    "id": "container-1",
                    "uri": "https://rupload.facebook.com/ig-api-upload/v27.0/container-1",
                },
            )
        if path == "/ig-api-upload/v27.0/container-1" and request.method == "POST":
            return httpx.Response(200, json={"success": True})
        if path == "/v27.0/container-1" and request.method == "GET":
            state = states.pop(0) if states else "FINISHED"
            return httpx.Response(200, json={"status_code": state})
        if path == f"/v27.0/{IG_USER_ID}/media_publish" and request.method == "POST":
            assert request.url.params["creation_id"] == "container-1"
            return httpx.Response(200, json={"id": publish_id})
        if path == "/v27.0/me/accounts" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "name": "My Page",
                            "instagram_business_account": {
                                "id": IG_USER_ID,
                                "username": IG_USERNAME,
                            },
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    return handler


class TestInstagramUpload:
    async def test_full_flow_speaks_the_documented_protocol(
        self, fake_keyring, attested, linked_instagram, clip_file, monkeypatch
    ) -> None:
        monkeypatch.setattr(ig, "STATUS_POLL_INTERVAL_S", 0.0)
        transport = RecordingTransport(ig_flow_handler(["FINISHED"]))

        result = await uploads.upload_instagram(
            clip_file,
            uploads.UploadMeta(title="Best moment", description="desc"),
            client=mock_client(transport),
        )

        assert result.platform == "instagram"
        assert result.media_id == "pub-123"
        assert result.url == "https://www.facebook.com/v27.0/pub-123"
        assert result.account == IG_USERNAME

        # resolve → container → rupload → status poll → publish
        assert [r.method for r in transport.requests] == [
            "GET",
            "POST",
            "POST",
            "GET",
            "POST",
        ]
        resolve, container, rupload, poll, publish = transport.requests

        assert str(resolve.url) == (
            "https://graph.facebook.com/v27.0/me/accounts"
            "?fields=instagram_business_account%7Bid%2Cusername%7D%2Cname"
            "&access_token=ig-at"
        )
        assert str(container.url) == (
            f"https://graph.facebook.com/v27.0/{IG_USER_ID}/media"
            "?media_type=REELS&upload_type=resumable"
            "&caption=Best+moment%0A%0Adesc&access_token=ig-at"
        )
        # The encoding round-trips: Meta's server decodes the same caption back.
        assert container.url.params["caption"] == "Best moment\n\ndesc"
        assert rupload.url.host == "rupload.facebook.com"
        assert rupload.headers["Authorization"] == "OAuth ig-at"
        assert rupload.headers["offset"] == "0"
        assert rupload.headers["file_size"] == str(clip_file.stat().st_size)
        assert bodies_of(transport)[2] == clip_file.read_bytes()
        assert "fields=status_code" in str(poll.url)
        assert str(publish.url) == (
            f"https://graph.facebook.com/v27.0/{IG_USER_ID}/media_publish"
            "?creation_id=container-1&access_token=ig-at"
        )

        # The resolved IG identity is persisted for the next call.
        stored = tk.load_tokens("instagram")
        assert stored is not None
        assert stored.account_ref == IG_USER_ID
        assert stored.account == IG_USERNAME

    async def test_container_polling_waits_until_finished(
        self, fake_keyring, attested, linked_instagram, clip_file, monkeypatch
    ) -> None:
        monkeypatch.setattr(ig, "STATUS_POLL_INTERVAL_S", 0.0)
        transport = RecordingTransport(ig_flow_handler(["IN_PROGRESS", "FINISHED"]))

        result = await uploads.upload_instagram(
            clip_file, uploads.UploadMeta(title="t"), client=mock_client(transport)
        )

        assert result.media_id == "pub-123"
        polls = [r for r in transport.requests if "fields=status_code" in str(r.url)]
        assert len(polls) == 2

    async def test_container_error_state_surfaces_as_upload_api_error(
        self, fake_keyring, attested, linked_instagram, clip_file, monkeypatch
    ) -> None:
        monkeypatch.setattr(ig, "STATUS_POLL_INTERVAL_S", 0.0)

        with pytest.raises(tk.UploadApiError, match="ERROR"):
            await uploads.upload_instagram(
                clip_file,
                uploads.UploadMeta(title="t"),
                client=mock_client(httpx.MockTransport(ig_flow_handler(["ERROR"]))),
            )

    async def test_missing_page_link_is_an_actionable_error(
        self, fake_keyring, attested, linked_instagram, clip_file
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/me/accounts" in str(request.url)
            return httpx.Response(200, json={"data": []})

        with pytest.raises(tk.UploadApiError) as info:
            await uploads.upload_instagram(
                clip_file,
                uploads.UploadMeta(title="t"),
                client=mock_client(httpx.MockTransport(handler)),
            )
        assert "professional account" in str(info.value)

    async def test_unlinked_account_is_not_linked(self, fake_keyring, attested, clip_file) -> None:
        with pytest.raises(tk.NotLinkedError):
            await uploads.upload_instagram(
                clip_file,
                uploads.UploadMeta(title="t"),
                client=mock_client(httpx.MockTransport(lambda r: httpx.Response(200, json={}))),
            )

    async def test_cached_ig_identity_skips_the_accounts_lookup(
        self, fake_keyring, attested, clip_file
    ) -> None:
        uploads.save_client_credentials("instagram", *IG_CLIENT)
        tk.save_tokens(
            "instagram",
            tk.TokenBundle(access_token="ig-at", account=IG_USERNAME, account_ref=IG_USER_ID),
        )
        transport = RecordingTransport(ig_flow_handler(["FINISHED"]))

        result = await uploads.upload_instagram(
            clip_file, uploads.UploadMeta(title="t"), client=mock_client(transport)
        )

        assert result.media_id == "pub-123"
        assert not any("/me/accounts" in str(r.url) for r in transport.requests)


class TestInstagramStatus:
    async def test_unlinked_costs_no_network(self, fake_keyring) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no network for an unlinked account")

        status = await uploads.instagram_status(client=mock_client(httpx.MockTransport(fail)))
        assert status == {"connected": False, "account": None}

    async def test_cached_username_avoids_the_graph_lookup(
        self, fake_keyring, linked_instagram
    ) -> None:
        tk.save_tokens(
            "instagram",
            tk.TokenBundle(access_token="ig-at", account="cached", account_ref="ig-9"),
        )

        def fail(request: httpx.Request) -> httpx.Response:
            raise AssertionError("a cached identity must not hit the graph")

        status = await uploads.instagram_status(client=mock_client(httpx.MockTransport(fail)))
        assert status == {"connected": True, "account": "cached"}

    async def test_status_resolves_and_persists_the_ig_identity(
        self, fake_keyring, linked_instagram
    ) -> None:
        transport = RecordingTransport(ig_flow_handler(["FINISHED"]))

        status = await uploads.instagram_status(client=mock_client(transport))

        assert status == {"connected": True, "account": IG_USERNAME}
        stored = tk.load_tokens("instagram")
        assert stored is not None
        assert stored.account_ref == IG_USER_ID
        assert stored.account == IG_USERNAME


# --------------------------------------------------------------------------
# UploadMeta + redaction helpers
# --------------------------------------------------------------------------


class TestUploadMeta:
    def test_titles_and_descriptions_are_clamped_to_platform_limits(self) -> None:
        meta = uploads.UploadMeta(title="x" * 500, description="y" * 9000)
        assert len(meta.title) == 100
        assert len(meta.description) == 5000

    def test_caption_combines_title_and_description(self) -> None:
        meta = uploads.UploadMeta(title="T", description="D")
        assert meta.caption == "T\n\nD"

    def test_caption_is_title_only_without_description(self) -> None:
        assert uploads.UploadMeta(title="T").caption == "T"

    def test_caption_respects_the_2200_char_limit(self) -> None:
        meta = uploads.UploadMeta(title="t" * 3000)
        assert len(meta.caption) <= 2200

    def test_invalid_privacy_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            uploads.UploadMeta(title="t", privacy="friends")

    def test_made_for_kids_flips_the_status_part(self) -> None:
        meta = uploads.UploadMeta(title="t", made_for_kids=True)

        assert meta.made_for_kids is True


class TestRedaction:
    def test_secrets_are_removed_from_error_text(self) -> None:
        out = tk.redact_secrets("token abc123 leaked and abc123", ("abc123",))
        assert "abc123" not in out

    def test_longest_secret_is_replaced_first(self) -> None:
        out = tk.redact_secrets("has short and shortlong", ("short", "shortlong"))
        assert "short" not in out and "shortlong" not in out

    def test_none_and_empty_values_are_ignored(self) -> None:
        assert tk.redact_secrets("text", (None, "")) == "text"

    def test_output_is_capped(self) -> None:
        assert len(tk.redact_secrets("a" * 5000, ())) == 1000


# --------------------------------------------------------------------------
# API shape (the manager + REST rely on these signatures)
# --------------------------------------------------------------------------


class TestApiSurface:
    @pytest.mark.parametrize("func", [uploads.upload_youtube, uploads.upload_instagram])
    def test_upload_functions_accept_path_meta_client(self, func) -> None:
        import inspect

        signature = inspect.signature(func)
        assert list(signature.parameters)[:2] == ["path", "meta"]
        assert "client" in signature.parameters

    @pytest.mark.parametrize("func", [uploads.youtube_status, uploads.instagram_status])
    def test_status_functions_are_client_kwarg_only(self, func) -> None:
        import inspect

        assert list(inspect.signature(func).parameters) == ["client"]
