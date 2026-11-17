"""Offline tests for the GTA Mode Zernio client (scheduled Instagram Reels).

No network: every request goes through an ``httpx.MockTransport`` that
records it and serves canned per-(method, path) responses. ``_sleep`` is
patched out so retry loops run instantly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from autoclip import config
from autoclip.gta import zernio as zz

KEY = "sk_live_supersecretkey_1234567890"
IG_ID = "6af7c9bb90ae441ff9b2c56f"
UPLOAD_URL = "https://media.zernio.com/presigned?sig=abc"
PUBLIC_URL = "https://media.zernio.com/temp/123_clip.mp4"
ACCOUNTS_PATH = "/api/v1/accounts"
PRESIGN_PATH = "/api/v1/media/presign"
POSTS_PATH = "/api/v1/posts"


def _resp(status: int = 200, body=None, headers=None) -> httpx.Response:
    return httpx.Response(status, json=body if body is not None else {}, headers=headers)


ONE_ACCOUNT = {
    "accounts": [
        {"_id": IG_ID, "platform": "instagram", "username": "gtaclips", "isActive": True},
        {"_id": "fb1", "platform": "facebook", "username": "page"},
    ]
}
PRESIGN_OK = {"uploadUrl": UPLOAD_URL, "publicUrl": PUBLIC_URL}


def _post(status: str, ig_status: str | None = None, post_id: str = "p1", **extra) -> dict:
    entry = {"platform": "instagram", "status": ig_status or status}
    entry.update(extra.pop("entry", {}))
    return {
        "_id": post_id,
        "status": status,
        "platforms": [entry],
        "mediaItems": [{"type": "video", "url": PUBLIC_URL}],
        **extra,
    }


class FakeZernio:
    """Routes by (method, url path) to a QUEUE of responses / exceptions /
    callables(request); the last item repeats once the queue runs dry."""

    def __init__(self, routes: dict | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], list] = {}
        for key, value in (routes or {}).items():
            self.routes[key] = list(value) if isinstance(value, list) else [value]

    def handler(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        queue = self.routes.get((request.method, request.url.path))
        if queue is None:
            return _resp(599, {"message": f"no route for {request.method} {request.url.path}"})
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(item) and not isinstance(item, httpx.Response):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        return item

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(zz, "_sleep", slept.append)
    return slept


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch):
    def _install(fake: FakeZernio) -> FakeZernio:
        transport = httpx.MockTransport(fake.handler)
        monkeypatch.setattr(zz, "_client", lambda: httpx.Client(transport=transport))
        return fake

    return _install


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 1000)
    return path


PUBLISH_AT = datetime(2026, 11, 17, 19, 0, tzinfo=timezone(timedelta(hours=2)))


def _routes(create=None, posts_list=None, accounts=None, presign=None, put=None) -> dict:
    routes = {
        ("GET", ACCOUNTS_PATH): accounts or _resp(200, ONE_ACCOUNT),
        ("POST", PRESIGN_PATH): presign or _resp(200, PRESIGN_OK),
        ("PUT", "/presigned"): put or _resp(200),
    }
    if create is not None:
        routes[("POST", POSTS_PATH)] = create
    if posts_list is not None:
        routes[("GET", POSTS_PATH)] = posts_list
    return routes


def _schedule(clip: Path, **kw) -> zz.ZernioResult:
    return zz.schedule_reel(KEY, kw.pop("hint", ""), clip, "caption #gta", PUBLISH_AT, **kw)


# ------------------------------------------------------------------ accounts
def test_connection_one_account_sends_bearer(install) -> None:
    fake = install(FakeZernio(_routes()))
    ok, msg, username = zz.test_connection(KEY)
    assert ok, msg
    assert username == "gtaclips" and "@gtaclips" in msg
    req = fake.requests[0]
    assert req.method == "GET" and str(req.url) == zz.ACCOUNTS_URL
    assert req.headers["Authorization"] == f"Bearer {KEY}"


def test_connection_no_instagram(install) -> None:
    install(FakeZernio(_routes(accounts=_resp(200, {"accounts": [{"platform": "facebook"}]}))))
    ok, msg, username = zz.test_connection(KEY)
    assert not ok and username == ""
    assert "No Instagram account is connected" in msg


def test_connection_several_needs_hint_and_hint_picks(install) -> None:
    accounts = {
        "accounts": [
            {"_id": "a1", "platform": "instagram", "username": "alpha"},
            {"_id": "b2", "platform": "instagram", "username": "bravo"},
            {"_id": "c3", "platform": "instagram", "username": "gone", "isActive": False},
        ]
    }
    install(FakeZernio(_routes(accounts=_resp(200, accounts))))
    ok, msg, _ = zz.test_connection(KEY)
    assert not ok and "Several Instagram accounts" in msg and "alpha" in msg
    ok, msg, username = zz.test_connection(KEY, "@Bravo")
    assert ok and username == "bravo"
    ok, msg, _ = zz.test_connection(KEY, "gone")
    assert not ok and "matching 'gone'" in msg


def test_connection_empty_key_makes_no_request(install) -> None:
    fake = install(FakeZernio(_routes()))
    ok, msg, _ = zz.test_connection("  ")
    assert not ok and "API key" in msg
    assert fake.requests == []


def test_connection_bad_key_message_is_masked(install) -> None:
    body = {"message": f"key {KEY} is invalid"}
    install(FakeZernio(_routes(accounts=_resp(401, body))))
    ok, msg, _ = zz.test_connection(KEY)
    assert not ok and "invalid or expired" in msg
    assert KEY not in msg


def test_key_echoed_in_error_is_masked_everywhere(install, clip: Path) -> None:
    body = {"message": f"Authorization: Bearer {KEY} lacks scope for {KEY}"}
    install(FakeZernio(_routes(accounts=_resp(403, body))))
    ok, msg, _ = zz.test_connection(KEY)
    assert not ok and KEY not in msg and "permission" in msg
    result = _schedule(clip)
    assert result.kind == zz.BLOCKED
    assert KEY not in result.message
    assert zz._sanitize(f"oops {KEY}", KEY) == f"oops {zz._mask(KEY)}"


def test_account_rate_limited_is_blocked(install, clip: Path) -> None:
    install(FakeZernio(_routes(accounts=_resp(429, {}, {"Retry-After": "42"}))))
    result = _schedule(clip)
    assert result.kind == zz.BLOCKED and result.retry_after == 42


def test_account_outage_is_blocked_transient(install, clip: Path, no_sleep) -> None:
    fake = install(FakeZernio(_routes(accounts=httpx.ConnectError("refused"))))
    result = _schedule(clip)
    assert result.kind == zz.BLOCKED and result.transient
    assert result.retry_after == zz.NETWORK_RETRY_WAIT
    assert len(fake.calls("GET", ACCOUNTS_PATH)) == zz.RETRY_ATTEMPTS
    assert len(no_sleep) == zz.RETRY_ATTEMPTS - 1


# ------------------------------------------------------------------ validation
def test_naive_publish_at_raises_before_any_request(install, clip: Path) -> None:
    fake = install(FakeZernio(_routes()))
    with pytest.raises(ValueError):
        zz.schedule_reel(KEY, "", clip, "c", datetime(2026, 11, 17, 17, 0))
    assert fake.requests == []


def test_invalid_files(install, tmp_path: Path, clip: Path, monkeypatch) -> None:
    fake = install(FakeZernio(_routes()))
    assert _schedule(tmp_path / "missing.mp4").kind == zz.INVALID
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert _schedule(empty).kind == zz.INVALID
    monkeypatch.setattr(zz, "REEL_MAX_BYTES", 10)
    big = _schedule(clip)
    assert big.kind == zz.INVALID and "1 GB" in big.message
    assert fake.requests == []


def test_no_key_is_blocked(install, clip: Path) -> None:
    fake = install(FakeZernio(_routes()))
    assert zz.schedule_reel("", "", clip, "c", PUBLISH_AT).kind == zz.BLOCKED
    assert fake.requests == []


def test_presign_non_https_refused(install, clip: Path) -> None:
    presign = _resp(200, {"uploadUrl": "http://evil.example/up", "publicUrl": PUBLIC_URL})
    fake = install(
        FakeZernio(_routes(create=_resp(201, {"post": _post("scheduled")}), presign=presign))
    )
    result = _schedule(clip)
    assert result.kind == zz.FAILED and "non-https" in result.message
    assert not result.transient
    assert fake.calls("POST", POSTS_PATH) == []
    assert [r for r in fake.requests if r.method == "PUT"] == []


def test_upload_link_expired_fails_without_create(install, clip: Path) -> None:
    fake = install(FakeZernio(_routes(create=_resp(201, {}), put=_resp(403))))
    result = _schedule(clip)
    assert result.kind == zz.FAILED and "expired" in result.message
    assert fake.calls("POST", POSTS_PATH) == []


# ------------------------------------------------------------------ create
def test_scheduled_create_success(install, clip: Path) -> None:
    order: list[str] = []
    fake = FakeZernio()

    def create(request: httpx.Request) -> httpx.Response:
        order.append("POST /posts")
        return _resp(201, {"post": _post("scheduled", "pending", scheduledFor="x")})

    fake.routes = FakeZernio(_routes(create=create)).routes
    install(fake)
    ids: list[str] = []
    result = _schedule(
        clip,
        on_media_url=lambda url: order.append(f"media {url}"),
        on_post_id=ids.append,
    )
    assert result.kind == zz.SCHEDULED, result
    assert result.post_id == "p1" and result.media_url == PUBLIC_URL
    assert ids == ["p1"]
    assert order == [f"media {PUBLIC_URL}", "POST /posts"]

    (create_req,) = fake.calls("POST", POSTS_PATH)
    body = json.loads(create_req.content)
    assert body["scheduledFor"] == "2026-11-17T17:00:00Z"
    assert body["timezone"] == "UTC"
    assert "publishNow" not in body
    assert body["content"] == "caption #gta"
    assert body["mediaItems"] == [{"type": "video", "url": PUBLIC_URL}]
    assert body["platforms"] == [{"platform": "instagram", "accountId": IG_ID}]
    assert create_req.headers["Authorization"] == f"Bearer {KEY}"

    (put_req,) = [r for r in fake.requests if r.method == "PUT"]
    assert str(put_req.url) == UPLOAD_URL
    assert "Authorization" not in put_req.headers
    assert put_req.headers["Content-Length"] == str(clip.stat().st_size)
    assert put_req.content == clip.read_bytes()
    presign = json.loads(fake.calls("POST", PRESIGN_PATH)[0].content)
    assert presign == {"filename": "clip.mp4", "contentType": "video/mp4"}


def test_create_published_and_failed_states(install, clip: Path) -> None:
    published = _post("published", entry={"platformPostUrl": "https://www.instagram.com/reel/XYZ/"})
    install(FakeZernio(_routes(create=_resp(201, {"post": published}))))
    result = _schedule(clip)
    assert result.kind == zz.POSTED and result.post_url == "https://www.instagram.com/reel/XYZ/"

    failed = _post("failed", entry={"error": "Video too short"})
    install(FakeZernio(_routes(create=_resp(201, {"post": failed}))))
    result = _schedule(clip)
    assert result.kind == zz.FAILED and "too short" in result.message


def test_create_unknown_status_is_unconfirmed_with_id(install, clip: Path) -> None:
    install(FakeZernio(_routes(create=_resp(201, {"post": _post("draft")}))))
    result = _schedule(clip)
    assert result.kind == zz.UNCONFIRMED and result.post_id == "p1"


def test_5xx_on_create_lookup_finds_scheduled(install, clip: Path) -> None:
    listing = _resp(200, {"posts": [_post("scheduled", post_id="found9")], "hasMore": False})
    fake = install(FakeZernio(_routes(create=_resp(502, {}), posts_list=listing)))
    ids: list[str] = []
    result = _schedule(clip, on_post_id=ids.append)
    assert result.kind == zz.SCHEDULED and result.post_id == "found9"
    assert ids == ["found9"]
    assert len(fake.calls("POST", POSTS_PATH)) == 1
    (lookup,) = fake.calls("GET", POSTS_PATH)
    params = lookup.url.params
    assert params["platform"] == "instagram" and params["limit"] == "100"
    assert params["dateFrom"] == (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.WriteTimeout("write timed out"),
        httpx.RemoteProtocolError("server disconnected"),
        httpx.ReadError("reset"),
    ],
)
def test_ambiguous_create_never_resends(install, clip: Path, exc: Exception) -> None:
    empty = _resp(200, {"posts": [], "hasMore": False})
    fake = install(FakeZernio(_routes(create=exc, posts_list=empty)))
    result = _schedule(clip)
    assert result.kind == zz.UNCONFIRMED and result.transient
    assert result.media_url == PUBLIC_URL
    assert len(fake.calls("POST", POSTS_PATH)) == 1
    assert len(fake.calls("GET", POSTS_PATH)) == zz.LOOKUP_ATTEMPTS


def test_connect_error_before_send_is_retried(install, clip: Path) -> None:
    created = _resp(201, {"post": _post("scheduled")})
    fake = install(
        FakeZernio(
            _routes(create=[httpx.ConnectError("refused"), httpx.ConnectTimeout("t"), created])
        )
    )
    result = _schedule(clip)
    assert result.kind == zz.SCHEDULED
    assert len(fake.calls("POST", POSTS_PATH)) == 3


def test_connect_error_every_time_is_failed_transient(install, clip: Path) -> None:
    fake = install(FakeZernio(_routes(create=httpx.ConnectError("refused"))))
    result = _schedule(clip)
    assert result.kind == zz.FAILED and result.transient
    assert len(fake.calls("POST", POSTS_PATH)) == zz.RETRY_ATTEMPTS
    assert fake.calls("GET", POSTS_PATH) == []


def test_429_on_create_is_blocked(install, clip: Path) -> None:
    fake = install(FakeZernio(_routes(create=_resp(429, {}, {"Retry-After": "120"}))))
    result = _schedule(clip)
    assert result.kind == zz.BLOCKED and result.retry_after == 120
    assert len(fake.calls("POST", POSTS_PATH)) == 1
    install(FakeZernio(_routes(create=_resp(402, {"code": "insufficient_balance"}))))
    result = _schedule(clip)
    assert result.kind == zz.BLOCKED and result.retry_after == zz.DEFAULT_RATE_LIMIT_WAIT
    assert "balance" in result.message


def test_422_on_create_is_invalid(install, clip: Path) -> None:
    install(FakeZernio(_routes(create=_resp(422, {"message": "caption too long"}))))
    result = _schedule(clip)
    assert result.kind == zz.INVALID and "caption too long" in result.message


def test_409_and_missing_id_are_looked_up(install, clip: Path) -> None:
    listing = _resp(200, {"posts": [_post("scheduled", post_id="q7")]})
    fake = install(FakeZernio(_routes(create=_resp(409, {}), posts_list=listing)))
    result = _schedule(clip)
    assert result.kind == zz.SCHEDULED and result.post_id == "q7"
    assert len(fake.calls("POST", POSTS_PATH)) == 1

    no_id = _post("scheduled", post_id="")
    fake = install(FakeZernio(_routes(create=_resp(201, {"post": no_id}), posts_list=listing)))
    result = _schedule(clip)
    assert result.kind == zz.SCHEDULED and result.post_id == "q7"
    assert len(fake.calls("POST", POSTS_PATH)) == 1


def test_create_401_is_blocked(install, clip: Path) -> None:
    install(FakeZernio(_routes(create=_resp(401, {"message": KEY}))))
    result = _schedule(clip)
    assert result.kind == zz.BLOCKED and KEY not in result.message


# ------------------------------------------------------------------ check_post
def test_check_post_by_id_states(install) -> None:
    url = "https://www.instagram.com/reel/ABC/"
    cases = [
        (_post("published", entry={"platformPostUrl": url}), zz.POSTED),
        (_post("failed"), zz.FAILED),
        (_post("cancelled"), zz.FAILED),
        (_post("scheduled"), zz.SCHEDULED),
        (_post("publishing"), zz.SCHEDULED),
        (_post("draft"), zz.UNCONFIRMED),
    ]
    for post, kind in cases:
        install(FakeZernio({("GET", f"{POSTS_PATH}/p1"): _resp(200, {"post": post})}))
        result = zz.check_post(KEY, post_id="p1")
        assert result.kind == kind, (post["status"], result)
        assert result.post_id == "p1"
        if kind == zz.POSTED:
            assert result.post_url == url
        assert not result.not_listed


def test_check_post_id_404_falls_back_to_media_lookup(install) -> None:
    listing = _resp(200, {"posts": [_post("published", post_id="real1")]})
    fake = install(
        FakeZernio({("GET", f"{POSTS_PATH}/p1"): _resp(404), ("GET", POSTS_PATH): listing})
    )
    sent = datetime.now(UTC) - timedelta(days=7)
    result = zz.check_post(KEY, post_id="p1", media_url=PUBLIC_URL, sent_at=sent)
    assert result.kind == zz.POSTED and result.post_id == "real1"
    expected = (sent - timedelta(days=1)).strftime("%Y-%m-%d")
    assert fake.calls("GET", POSTS_PATH)[0].url.params["dateFrom"] == expected


def test_check_post_not_listed_only_after_complete_search(install) -> None:
    install(FakeZernio({("GET", POSTS_PATH): _resp(200, {"posts": [], "hasMore": False})}))
    result = zz.check_post(KEY, media_url=PUBLIC_URL)
    assert result.kind == zz.UNCONFIRMED and result.not_listed

    install(FakeZernio({("GET", f"{POSTS_PATH}/p1"): _resp(404)}))
    result = zz.check_post(KEY, post_id="p1")
    assert result.kind == zz.UNCONFIRMED and result.not_listed


def test_check_post_pages_through_the_list(install) -> None:
    other = _post("published", post_id="other")
    other["mediaItems"] = [{"type": "video", "url": "https://media.zernio.com/x.mp4"}]

    def listing(request: httpx.Request) -> httpx.Response:
        page = request.url.params.get("page", "1")
        if page == "1":
            return _resp(200, {"posts": [other], "pagination": {"page": 1, "pages": 2}})
        return _resp(200, {"posts": [_post("scheduled", post_id="p2")], "pagination": {"pages": 2}})

    fake = install(FakeZernio({("GET", POSTS_PATH): listing}))
    result = zz.check_post(KEY, media_url=PUBLIC_URL)
    assert result.kind == zz.SCHEDULED and result.post_id == "p2"
    assert len(fake.calls("GET", POSTS_PATH)) == 2


def test_check_post_incomplete_search_is_not_not_listed(install, monkeypatch) -> None:
    other = _post("published", post_id="o")
    other["mediaItems"] = []
    counter = iter(range(1000))

    def endless(request: httpx.Request) -> httpx.Response:
        page = dict(other, _id=f"o{next(counter)}")
        return _resp(200, {"posts": [page], "hasMore": True})

    monkeypatch.setattr(zz, "LOOKUP_MAX_PAGES", 3)
    fake = install(FakeZernio({("GET", POSTS_PATH): endless}))
    result = zz.check_post(KEY, media_url=PUBLIC_URL)
    assert result.kind == zz.UNCONFIRMED and not result.not_listed
    assert len(fake.calls("GET", POSTS_PATH)) == 3

    # a server that ignores ``page`` (same page twice) is "no answer" too
    same = _resp(200, {"posts": [other], "hasMore": True})
    install(FakeZernio({("GET", POSTS_PATH): same}))
    result = zz.check_post(KEY, media_url=PUBLIC_URL)
    assert result.kind == zz.UNCONFIRMED and not result.not_listed

    # an outage on the list
    install(FakeZernio({("GET", POSTS_PATH): _resp(500)}))
    result = zz.check_post(KEY, media_url=PUBLIC_URL)
    assert result.kind == zz.UNCONFIRMED and not result.not_listed and result.transient

    # an unreadable list
    install(FakeZernio({("GET", POSTS_PATH): _resp(200, {"items": []})}))
    result = zz.check_post(KEY, media_url=PUBLIC_URL)
    assert result.kind == zz.UNCONFIRMED and not result.not_listed


def test_check_post_unreadable_post_is_not_not_listed(install) -> None:
    install(FakeZernio({("GET", f"{POSTS_PATH}/p1"): httpx.Response(200, text="<html>")}))
    result = zz.check_post(KEY, post_id="p1")
    assert result.kind == zz.UNCONFIRMED and not result.not_listed


def test_check_post_blocked_and_inputs(install) -> None:
    install(FakeZernio({("GET", f"{POSTS_PATH}/p1"): _resp(401, {"message": KEY})}))
    result = zz.check_post(KEY, post_id="p1")
    assert result.kind == zz.BLOCKED and KEY not in result.message

    fake = install(FakeZernio({("GET", POSTS_PATH): _resp(429, {}, {"Retry-After": "30"})}))
    result = zz.check_post(KEY, media_url=PUBLIC_URL)
    assert result.kind == zz.BLOCKED and result.retry_after == 30
    assert len(fake.calls("GET", POSTS_PATH)) == 1  # a 429 stops the paging at once

    assert zz.check_post("", post_id="p1").kind == zz.BLOCKED
    assert zz.check_post(KEY).kind == zz.INVALID


# ------------------------------------------------------------------ state + helpers
def test_post_state_scheduled_rules() -> None:
    assert zz._post_state({"status": "scheduled"})[0] == "scheduled"
    assert zz._post_state(_post("scheduled", "pending"))[0] == "scheduled"
    assert zz._post_state(_post("", "scheduled"))[0] == "scheduled"
    assert zz._post_state(_post("scheduled", "publishing"))[0] == "waiting"
    assert zz._post_state(_post("partial", "published"))[0] == "published"
    assert zz._post_state(_post("scheduled", "failed"))[0] == "failed"
    assert zz._post_state({})[0] == "unknown"


def test_lookup_date_from() -> None:
    # 00:10 at +05:00 is 19:10 UTC the day before; minus the one-day margin
    local = (datetime.now(UTC) - timedelta(days=10)).replace(hour=0, minute=10, tzinfo=None)
    sent = local.replace(tzinfo=timezone(timedelta(hours=5)))
    assert zz.lookup_date_from(sent) == (local - timedelta(days=2)).strftime("%Y-%m-%d")
    future = datetime.now(UTC) + timedelta(days=30)
    assert zz.lookup_date_from(future) == (datetime.now(UTC) - timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )


# ------------------------------------------------------------------ secrets
def test_save_and_load_api_key(fake_keyring) -> None:
    assert zz.load_api_key() is None
    zz.save_api_key(f"  {KEY}\n")
    assert zz.load_api_key() == KEY
    assert fake_keyring.store[(config.KEYRING_SERVICE, zz.API_KEY_SECRET)] == KEY
    zz.save_api_key("   ")
    assert zz.load_api_key() is None


def test_save_api_key_without_keyring_refuses_plaintext(no_keyring) -> None:
    with pytest.raises(zz.ZernioError, match="keyring"):
        zz.save_api_key(KEY)
    assert zz.load_api_key() is None
    assert KEY not in json.dumps(config.load()._fallback_secrets)


def test_reelrepost_api_key(monkeypatch) -> None:
    import keyring

    seen: list[tuple[str, str]] = []

    def get_password(service: str, key: str) -> str:
        seen.append((service, key))
        return "sk_from_reelrepost"

    monkeypatch.setattr(keyring, "get_password", get_password)
    assert zz.reelrepost_api_key() == "sk_from_reelrepost"
    assert seen == [(zz.REELREPOST_SERVICE, zz.REELREPOST_KEY)]

    def boom(service: str, key: str) -> str:
        raise RuntimeError("no backend")

    monkeypatch.setattr(keyring, "get_password", boom)
    assert zz.reelrepost_api_key() is None
