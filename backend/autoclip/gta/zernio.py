"""GTA Mode Instagram poster: schedules Reels through the Zernio API.

Ported from ReelRepost's battle-tested ``uploader_zernio`` client, with one
big difference: AutoClip SCHEDULES each Reel for its slot (``scheduledFor``)
instead of publishing it at once, so Zernio itself publishes it on time
whether or not AutoClip is still running.

Endpoints (https://zernio.com/api/v1):

1. GET  /accounts       -> the connected Instagram account's ``_id``
2. POST /media/presign  -> {uploadUrl, publicUrl}; then PUT the bytes to
                           uploadUrl (pre-signed, no auth header)
3. POST /posts          -> {content, mediaItems, platforms, scheduledFor,
                           timezone: "UTC"}
4. GET  /posts/{id}     -> the post and its per-platform status
5. GET  /posts?platform=instagram&dateFrom=...&limit=100[&page=N]
                        -> finds a post by its (unique) media URL

Never posting twice
-------------------
Zernio documents no idempotency key for POST /posts, so the create call is
sent ONCE. It is only re-sent when the request provably never left
(``httpx.ConnectError`` / ``ConnectTimeout`` / ``PoolTimeout``: DNS failure,
connection refused, no route). Every ambiguous result (read/write timeout,
connection dropped after sending, 5xx, an answer without a post id, 409) is
resolved by looking the post up by its media URL; failing that it is
reported as ``unconfirmed`` together with what is known, and the caller
keeps asking ``check_post``. A lookup only says "not listed" after a
COMPLETE search: every page since a day before the send time (UTC). A
search it could not finish (an error on any page, more than
``LOOKUP_MAX_PAGES`` pages, an unreadable list, a 429) is "no answer".

Result kinds (``ZernioResult.kind``)
------------------------------------
``scheduled``   Zernio holds the post and will publish it (or is publishing it).
``posted``      live on Instagram (``post_url`` set when Zernio reports it).
``failed``      nothing is (or will be) published; safe to retry later.
``unconfirmed`` it may exist: never re-post, call ``check_post`` instead.
``blocked``     the ACCOUNT needs attention (key, balance, permission,
                Instagram connection, rate limit); ``retry_after`` in seconds.
``invalid``     this clip can never be posted as-is.

The API key is only ever sent in the Authorization header and is masked out
of every returned message.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .. import config

API_KEY_SECRET = "gta_zernio_api_key"
REELREPOST_SERVICE = "ReelRepost"
REELREPOST_KEY = "zernio_api_key"

BASE_URL = "https://zernio.com/api/v1"
ACCOUNTS_URL = f"{BASE_URL}/accounts"
PRESIGN_URL = f"{BASE_URL}/media/presign"
POSTS_URL = f"{BASE_URL}/posts"

REQUEST_TIMEOUT = 60.0
CREATE_TIMEOUT = 120.0
UPLOAD_TIMEOUT = 900.0  # video PUT: slow connections, big files
RETRY_ATTEMPTS = 3  # for safe (read-only / not-yet-sent) requests
RETRY_BASE_DELAY = 10.0
RETRY_MAX_DELAY = 60.0
LOOKUP_ATTEMPTS = 3  # list lookups after an ambiguous create
LOOKUP_DELAY = 15.0
LOOKUP_DAYS = 2  # how far back a post is looked up when the send time is unknown
LOOKUP_MARGIN = timedelta(days=1)  # dateFrom = (send time in UTC - this).date()
LOOKUP_PAGE_SIZE = 100  # Zernio's maximum page size
LOOKUP_MAX_PAGES = 20  # beyond this many pages a "not listed" is not trusted
VIDEO_CONTENT_TYPE = "video/mp4"
REEL_MAX_BYTES = 1024 * 1024 * 1024
DEFAULT_RATE_LIMIT_WAIT = 15 * 60  # seconds
NETWORK_RETRY_WAIT = 10 * 60  # seconds
MAX_JSON_BYTES = 2_000_000
MAX_MESSAGE_CHARS = 500

SCHEDULED = "scheduled"
POSTED = "posted"
FAILED = "failed"
UNCONFIRMED = "unconfirmed"
BLOCKED = "blocked"
INVALID = "invalid"

#: platform/post statuses meaning "Zernio has it and is still working on it"
_WAITING = frozenset({"pending", "publishing", "processing", "queued", "in_progress"})
_FAILED_STATES = frozenset({"failed", "cancelled", "canceled"})


@dataclass
class ZernioResult:
    kind: str  # scheduled | posted | failed | unconfirmed | blocked | invalid
    message: str = ""
    post_id: str = ""
    post_url: str = ""  # live reel URL when posted
    media_url: str = ""  # the uploaded public media URL, when known
    retry_after: float = 0.0  # seconds, for blocked
    transient: bool = False  # outage / network, not this clip's fault
    not_listed: bool = False  # check_post: a COMPLETE search found no such post


class ZernioError(Exception):
    """A local problem with the Zernio setup (e.g. no keyring for the key)."""


class _RateLimited(Exception):
    """HTTP 429 rate limit / 402 empty balance; never retried here."""

    def __init__(self, status_code: int, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


class _PhaseFailure(Exception):
    """A phase failed after exhausting retries, or fatally (0 = network/outage)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class _Ambiguous(Exception):
    """The create call may or may not have reached Zernio."""


# ------------------------------------------------------------------ secrets
def load_api_key() -> str | None:
    """The Zernio API key from AutoClip's keyring slot, or None."""
    return config.get_secret(API_KEY_SECRET) or None


def save_api_key(key: str) -> None:
    """Store (or, when empty, delete) the key; never in plaintext config."""
    key = (key or "").strip()
    if not key:
        config.delete_secret(API_KEY_SECRET)
        return
    if not config.set_secret(API_KEY_SECRET, key):
        config.delete_secret(API_KEY_SECRET)
        raise ZernioError(
            "no usable OS keyring backend found; the Zernio API key is never "
            "written to config.json (on headless Linux install a keyring "
            "backend such as secretstorage)"
        )


def reelrepost_api_key() -> str | None:
    """The key ReelRepost saved on this PC (for one-click import), or None."""
    try:
        import keyring

        value = keyring.get_password(REELREPOST_SERVICE, REELREPOST_KEY)
    except Exception:
        return None
    return value or None


# ------------------------------------------------------------------ plumbing
def _client() -> httpx.Client:
    """Client factory; tests monkeypatch this with an httpx.MockTransport."""
    return httpx.Client()


def _sleep(seconds: float) -> None:
    """Sleep indirection so tests can run retry loops instantly."""
    time.sleep(seconds)


def _mask(s: str) -> str:
    """Mask a secret: first 4 chars + ellipsis + last 2 chars."""
    if not s:
        return ""
    if len(s) <= 6:
        return s[:1] + "…"
    return f"{s[:4]}…{s[-2:]}"


_BEARER_RE = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)


def _sanitize(message: str, key: str) -> str:
    """Ensure a user-facing string never contains the raw API key."""
    message = str(message or "")
    if key and key in message:
        message = message.replace(key, _mask(key))
    message = _BEARER_RE.sub(r"\1***", message)
    return message[:MAX_MESSAGE_CHARS]


def _done(api_key: str, kind: str, message: str = "", **extra) -> ZernioResult:
    return ZernioResult(kind=kind, message=_sanitize(message, api_key), **extra)


def _safe_json(resp: httpx.Response) -> dict:
    try:
        declared = int(resp.headers.get("Content-Length") or 0)
    except (TypeError, ValueError):
        declared = 0
    if declared > MAX_JSON_BYTES:
        return {}
    try:
        data = resp.json()
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _retry_after(resp: httpx.Response) -> float:
    value = resp.headers.get("Retry-After")
    try:
        return max(0.0, float(value)) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


def _explain(body: dict, status_code: int) -> str:
    """Map a Zernio error body to plain, actionable language."""
    code = str(body.get("code") or "").strip()
    message = str(body.get("message") or body.get("error") or "").strip()
    low = message.lower()
    if status_code == 401:
        return (
            "Zernio rejected the API key (invalid or expired). Copy a fresh key "
            "from zernio.com (Dashboard -> API Keys) into GTA Mode's posting settings."
        )
    if status_code == 402:
        return (
            "Your Zernio balance is empty for this call. Open zernio.com -> "
            "Billing, or stay on the free tier (first 2 connected accounts)."
        )
    if status_code == 403 or code == "PLATFORM_BETA_RESTRICTED":
        return (
            "Zernio refused this call - a permission is missing for your key or "
            f"account. ({message or 'HTTP 403'})"
        )
    if status_code == 404:
        return (
            "Zernio could not find that account or post. Your Instagram may "
            "have been disconnected - reconnect it at zernio.com, then press "
            "Test connection."
        )
    if status_code == 409:
        return (
            "Zernio is still busy with a previous request for this post; it will finish on its own."
        )
    if status_code == 422:
        return f"Zernio did not accept this post. ({message or 'HTTP 422'})"
    if status_code == 429 or "rate" in low or "quota" in low:
        return (
            "A Zernio rate limit was reached (free tier: 60 requests/minute; "
            "Instagram: 100 posts/day). AutoClip will wait and continue."
        )
    if code == "PLATFORM_DISABLED" or status_code == 503:
        return "Zernio reports Instagram is temporarily unavailable."
    if message:
        return f"Zernio error (HTTP {status_code}): {message}"
    return f"Zernio returned HTTP {status_code}."


def _rate_limited(resp: httpx.Response) -> bool:
    if resp.status_code in (429, 402):
        return True
    code = str(_safe_json(resp).get("code") or "")
    return code in ("rate_limit_exceeded", "insufficient_balance")


def _not_sent(exc: BaseException) -> bool:
    """True when a request provably never reached the server."""
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))


def _backoff(delay: float) -> float:
    _sleep(delay)
    return min(delay * 2, RETRY_MAX_DELAY)


def _request(
    method: str,
    url: str,
    api_key: str,
    *,
    json_body: dict | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> httpx.Response:
    """A safe (read-only or harmless-to-repeat) request, retried on 5xx and
    network errors. Raises _RateLimited on 429/402 (no retries) and
    _PhaseFailure(0) when the retries run out."""
    delay = RETRY_BASE_DELAY
    last_error = "unknown error"
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with _client() as client:
                resp = client.request(
                    method, url, json=json_body, headers=_headers(api_key), timeout=timeout
                )
        except httpx.HTTPError:
            last_error = "Network error reaching Zernio."
        else:
            if _rate_limited(resp):
                raise _RateLimited(
                    resp.status_code,
                    _explain(_safe_json(resp), resp.status_code),
                    _retry_after(resp),
                )
            if 500 <= resp.status_code <= 599:
                last_error = f"Zernio server error (HTTP {resp.status_code})."
            else:
                return resp
        if attempt < RETRY_ATTEMPTS:
            delay = _backoff(delay)
    raise _PhaseFailure(0, last_error)


def _send_create(api_key: str, body: dict) -> httpx.Response:
    """POST /posts exactly once (re-sent only if it provably never left).

    Returns the response, or raises _Ambiguous / _RateLimited /
    _PhaseFailure (definitely not created).
    """
    delay = RETRY_BASE_DELAY
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with _client() as client:
                resp = client.request(
                    "POST", POSTS_URL, json=body, headers=_headers(api_key), timeout=CREATE_TIMEOUT
                )
        except httpx.HTTPError as exc:
            if not _not_sent(exc):
                raise _Ambiguous("the connection dropped after the post was sent") from exc
            if attempt < RETRY_ATTEMPTS:
                delay = _backoff(delay)
                continue
            raise _PhaseFailure(0, "Could not reach Zernio (no internet?).") from exc
        if _rate_limited(resp):
            raise _RateLimited(
                resp.status_code, _explain(_safe_json(resp), resp.status_code), _retry_after(resp)
            )
        if resp.status_code == 503 and str(_safe_json(resp).get("code") or "") == (
            "PLATFORM_DISABLED"
        ):
            raise _PhaseFailure(503, _explain(_safe_json(resp), 503))
        if 500 <= resp.status_code <= 599:
            raise _Ambiguous(f"Zernio answered HTTP {resp.status_code}")
        return resp
    raise _Ambiguous("no answer from Zernio")  # pragma: no cover - loop always returns/raises


# ------------------------------------------------------------------ accounts
def _accounts(api_key: str) -> list[dict]:
    """GET /accounts -> the raw accounts list (may be empty)."""
    resp = _request("GET", ACCOUNTS_URL, api_key)
    if resp.status_code != 200:
        raise _PhaseFailure(resp.status_code, _explain(_safe_json(resp), resp.status_code))
    accounts = _safe_json(resp).get("accounts")
    return accounts if isinstance(accounts, list) else []


def _pick_instagram_id(api_key: str, hint: str) -> tuple[str, str]:
    """Resolve (accountId, username) of the Instagram account to post with."""
    hint = (hint or "").strip().lstrip("@").lower()
    instas = [
        a
        for a in _accounts(api_key)
        if isinstance(a, dict)
        and str(a.get("platform") or "").lower() == "instagram"
        and a.get("isActive", True)
    ]
    if hint:
        for a in instas:
            hay = " ".join(str(a.get(k) or "") for k in ("username", "_id", "displayName")).lower()
            if hint in hay:
                return str(a.get("_id") or ""), str(a.get("username") or "")
        raise _PhaseFailure(
            404,
            f"No active Instagram account matching '{hint}' is connected to "
            "Zernio. Check zernio.com -> Accounts, or leave the account field "
            "empty for automatic selection.",
        )
    if not instas:
        raise _PhaseFailure(
            404,
            "No Instagram account is connected to Zernio. Open zernio.com, "
            "connect your Instagram (approve all permissions), then press "
            "Test connection.",
        )
    if len(instas) > 1:
        names = ", ".join(str(a.get("username") or "?") for a in instas[:5])
        raise _PhaseFailure(
            403,
            f"Several Instagram accounts are connected ({names}). Enter the "
            "username to post with in GTA Mode's posting settings.",
        )
    a = instas[0]
    return str(a.get("_id") or ""), str(a.get("username") or "")


# ------------------------------------------------------------------ media
def _is_https(url: str) -> bool:
    return urlsplit(url).scheme.lower() == "https"


def _presign(api_key: str, video_path: Path) -> tuple[str, str]:
    """POST /media/presign -> (uploadUrl, publicUrl)."""
    body = {"filename": video_path.name, "contentType": VIDEO_CONTENT_TYPE}
    resp = _request("POST", PRESIGN_URL, api_key, json_body=body)
    if resp.status_code not in (200, 201):
        raise _PhaseFailure(resp.status_code, _explain(_safe_json(resp), resp.status_code))
    data = _safe_json(resp)
    upload_url = str(data.get("uploadUrl") or "")
    public_url = str(data.get("publicUrl") or data.get("mediaUrl") or "")
    if not upload_url or not public_url:
        raise _PhaseFailure(0, "Zernio's media response was missing the upload link.")
    if not (_is_https(upload_url) and _is_https(public_url)):
        raise _PhaseFailure(-1, "Zernio's media response had a non-https link; refusing it.")
    return upload_url, public_url


def _put_media(upload_url: str, video_path: Path, size: int) -> None:
    """PUT the video bytes to the presigned URL (streamed; retried)."""
    delay = RETRY_BASE_DELAY
    last_error = "unknown error"
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            with video_path.open("rb") as fh, _client() as client:
                resp = client.put(
                    upload_url,
                    content=fh,
                    headers={"Content-Type": VIDEO_CONTENT_TYPE, "Content-Length": str(size)},
                    timeout=UPLOAD_TIMEOUT,
                )
        except httpx.HTTPError:
            last_error = "Network error while uploading the video to Zernio."
        except OSError as exc:
            raise _PhaseFailure(-1, f"The video file can't be read: {video_path} ({exc})") from exc
        else:
            if resp.status_code in (200, 201, 204):
                return
            if resp.status_code in (403, 410):
                raise _PhaseFailure(
                    resp.status_code,
                    "Zernio's media upload link was refused or expired. The reel "
                    "was not posted; it can be retried with a fresh link.",
                )
            if 500 <= resp.status_code <= 599:
                last_error = f"Zernio media server error (HTTP {resp.status_code})."
            else:
                raise _PhaseFailure(
                    resp.status_code, f"Zernio media upload failed (HTTP {resp.status_code})."
                )
        if attempt < RETRY_ATTEMPTS:
            delay = _backoff(delay)
    raise _PhaseFailure(0, last_error)


# ------------------------------------------------------------------ posts
def _instagram_entry(post: dict) -> dict | None:
    for p in post.get("platforms") or []:
        if isinstance(p, dict) and str(p.get("platform") or "").lower() == "instagram":
            return p
    return None


def _post_state(post: dict) -> tuple[str, str, str]:
    """(state, post_url, error); state is published | failed | scheduled |
    waiting (Zernio is working on it) | unknown (e.g. a draft)."""
    entry = _instagram_entry(post) or {}
    entry_status = str(entry.get("status") or "").lower()
    top = str(post.get("status") or "").lower()
    for status in (entry_status, top):
        if status == "published":
            return "published", str(entry.get("platformPostUrl") or ""), ""
        if status in _FAILED_STATES:
            error = str(
                entry.get("error")
                or entry.get("message")
                or post.get("message")
                or "Instagram did not accept the reel."
            )
            return "failed", "", error
    if top == "scheduled" and entry_status in ("", "scheduled", "pending"):
        return "scheduled", "", ""
    if entry_status == "scheduled" and top in ("", "scheduled", "pending"):
        return "scheduled", "", ""
    if entry_status in _WAITING or top in _WAITING or "scheduled" in (top, entry_status):
        return "waiting", "", ""
    return "unknown", "", ""


def _get_post(api_key: str, post_id: str) -> dict | None:
    """GET /posts/{id} -> the post dict, None when Zernio has no such post."""
    resp = _request("GET", f"{POSTS_URL}/{post_id}", api_key)
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise _PhaseFailure(resp.status_code, _explain(_safe_json(resp), resp.status_code))
    data = _safe_json(resp)
    post = data.get("post") if isinstance(data.get("post"), dict) else data
    if not post:
        # an unreadable answer is "no answer", never "no such post"
        raise _PhaseFailure(0, "Zernio's answer about the post could not be read.")
    return post


def lookup_date_from(sent_at: datetime | None = None, days: int = LOOKUP_DAYS) -> str:
    """The ``dateFrom`` (a UTC date) for looking a post up by its media URL.

    With the send time: the UTC day BEFORE it, so neither midnight nor the
    PC's time zone can put the post's creation date before the filter.
    Without it: ``days`` back from now (UTC). A naive ``sent_at`` is local.
    """
    now = datetime.now(UTC)
    if sent_at is None:
        start = now - timedelta(days=days)
    else:
        start = min(sent_at.astimezone(UTC), now) - LOOKUP_MARGIN
    return start.strftime("%Y-%m-%d")


def _post_time(post: dict) -> datetime | None:
    """The newest of a listed post's timestamps (aware UTC), if any parse."""
    times = []
    for key in ("createdAt", "scheduledFor", "publishedAt", "updatedAt"):
        value = post.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            when = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        times.append(when if when.tzinfo else when.replace(tzinfo=UTC))
    return max(times) if times else None


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _more_pages(data: dict, page: int, seen: int, count: int) -> bool:
    """True unless the list says (or shows) that this was its last page."""
    if count == 0:
        return False
    info = data.get("pagination")
    info = info if isinstance(info, dict) else {}
    has_more = info.get("hasMore", data.get("hasMore"))
    pages = _as_int(info.get("pages", info.get("totalPages")))
    total = _as_int(info.get("total"))
    if has_more is True:
        return True
    if pages is not None and page < pages:
        return True
    if total is not None and seen < total:
        return True
    if has_more is None and pages is None and total is None:
        return count >= LOOKUP_PAGE_SIZE  # a full page: there may be more
    return False


def _all_older(posts: list, cutoff: datetime) -> bool:
    """A newest-first page whose oldest post predates ``cutoff``: the pages
    after it can only be older still (trusted only when the order shows)."""
    times = [_post_time(p) for p in posts if isinstance(p, dict)]
    if not times or any(t is None for t in times):
        return False
    ordered = all(a >= b for a, b in pairwise(times))  # type: ignore[operator]
    return ordered and times[-1] < cutoff  # type: ignore[operator]


def _has_media(post: dict, media_url: str) -> bool:
    return any(
        isinstance(item, dict) and str(item.get("url") or "") == media_url
        for item in post.get("mediaItems") or []
    )


def _find_post_by_media(
    api_key: str, media_url: str, sent_at: datetime | None = None
) -> dict | None:
    """Look the post up by its (unique) uploaded media URL.

    Returns the post, or None only after a complete search of every page
    since ``lookup_date_from(sent_at)``. Raises _PhaseFailure (never with
    status 404) or _RateLimited when the search could not be completed:
    that is "no answer", not "not listed".
    """
    if not media_url:
        return None
    since = lookup_date_from(sent_at)
    cutoff = datetime.fromisoformat(since).replace(tzinfo=UTC)
    base = f"{POSTS_URL}?platform=instagram&dateFrom={since}&limit={LOOKUP_PAGE_SIZE}"
    seen_pages: set[tuple[str, ...]] = set()
    seen = 0
    for page in range(1, LOOKUP_MAX_PAGES + 1):
        url = base if page == 1 else f"{base}&page={page}"
        resp = _request("GET", url, api_key)  # a 429 stops the paging at once
        if resp.status_code != 200:
            status = 0 if resp.status_code == 404 else resp.status_code
            raise _PhaseFailure(status, _explain(_safe_json(resp), resp.status_code))
        data = _safe_json(resp)
        posts = data.get("posts")
        if not isinstance(posts, list):
            raise _PhaseFailure(0, "Zernio's list of posts could not be read.")
        ids = tuple(str(p.get("_id") or "") for p in posts if isinstance(p, dict))
        if ids and ids in seen_pages:
            # the server ignores ``page``: the rest of the list is unknown
            raise _PhaseFailure(0, "Zernio returned the same page of posts twice.")
        seen_pages.add(ids)
        seen += len(posts)
        for post in posts:
            if isinstance(post, dict) and _has_media(post, media_url):
                return post
        if not _more_pages(data, page, seen, len(posts)) or _all_older(posts, cutoff):
            return None
    raise _PhaseFailure(
        0,
        f"Zernio lists more than {LOOKUP_MAX_PAGES * LOOKUP_PAGE_SIZE} recent posts; "
        "AutoClip could not look through all of them.",
    )


def _from_post(
    api_key: str,
    post: dict,
    post_id: str,
    media_url: str,
    *,
    waiting_message: str,
) -> ZernioResult:
    """Map a post Zernio returned/listed to a result."""
    state, url, error = _post_state(post)
    if state == "published":
        return _done(api_key, POSTED, "", post_id=post_id, post_url=url, media_url=media_url)
    if state == "failed":
        return _done(api_key, FAILED, error, post_id=post_id, media_url=media_url)
    if state in ("scheduled", "waiting"):
        if state == "scheduled":
            when = str(post.get("scheduledFor") or "").strip()
            message = f"Scheduled on Instagram via Zernio{f' for {when}' if when else ''}."
            kind = SCHEDULED
        else:
            message, kind = waiting_message, SCHEDULED
        return _done(api_key, kind, message, post_id=post_id, media_url=media_url)
    status = str(post.get("status") or "unknown")
    return _done(
        api_key,
        UNCONFIRMED,
        f"Zernio holds the post but reports status '{status}'; AutoClip keeps checking.",
        post_id=post_id,
        media_url=media_url,
    )


def _resolve_ambiguous(
    api_key: str,
    media_url: str,
    reason: str,
    sent_at: datetime,
    on_post_id: Callable[[str], None],
    transient: bool = False,
) -> ZernioResult:
    """After an ambiguous create: find the post by media URL, else unconfirmed."""
    for attempt in range(LOOKUP_ATTEMPTS):
        if attempt:
            _sleep(LOOKUP_DELAY)
        try:
            post = _find_post_by_media(api_key, media_url, sent_at=sent_at)
        except (_RateLimited, _PhaseFailure):
            post = None
        if post is not None:
            post_id = str(post.get("_id") or "")
            if post_id:
                on_post_id(post_id)
            return _from_post(
                api_key,
                post,
                post_id,
                media_url,
                waiting_message="Zernio has the post and is publishing it.",
            )
    return _done(
        api_key,
        UNCONFIRMED,
        f"Zernio's answer was unclear ({reason}). AutoClip is checking whether "
        "the reel was scheduled and will not post it again meanwhile.",
        media_url=media_url,
        transient=transient,
    )


def _iso_utc(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _account_problem(status_code: int) -> bool:
    return status_code in (401, 402, 403, 404)


def _outage(status_code: int) -> bool:
    """_PhaseFailure codes that mean 'Zernio/network down' (0 = retries on
    network errors or 5xx ran out)."""
    return status_code == 0 or status_code >= 500


def _blocked_rate(api_key: str, exc: _RateLimited, **extra) -> ZernioResult:
    return _done(
        api_key,
        BLOCKED,
        exc.message,
        retry_after=exc.retry_after or DEFAULT_RATE_LIMIT_WAIT,
        **extra,
    )


# ------------------------------------------------------------------ public API
def test_connection(api_key: str, account_hint: str = "") -> tuple[bool, str, str]:
    """Verify the key and that Instagram is connected (read-only GET /accounts).

    Returns (ok, message, username).
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Enter and save your Zernio API key first.", ""
    try:
        account_id, username = _pick_instagram_id(api_key, account_hint)
    except _RateLimited as exc:
        return False, _sanitize(exc.message, api_key), ""
    except _PhaseFailure as exc:
        return False, _sanitize(exc.message, api_key), ""
    if not account_id:
        return (
            False,
            "Your Instagram is connected at Zernio, but the account id was missing "
            "from the response. Reconnect it at zernio.com.",
            "",
        )
    return True, _sanitize(f"Connected to Instagram (@{username}) via Zernio.", api_key), username


test_connection.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


def schedule_reel(
    api_key: str,
    account_hint: str,
    video_path: Path | str,
    caption: str,
    publish_at: datetime,
    *,
    on_media_url: Callable[[str], None] | None = None,
    on_post_id: Callable[[str], None] | None = None,
) -> ZernioResult:
    """Upload ``video_path`` and schedule it as a Reel for ``publish_at``.

    ``on_media_url(url)`` is called right BEFORE the create call and
    ``on_post_id(id)`` as soon as Zernio's post id is known, so the caller
    can record them and recover safely if AutoClip stops mid-post.
    """
    if publish_at.tzinfo is None or publish_at.utcoffset() is None:
        raise ValueError("publish_at must be timezone-aware")
    api_key = (api_key or "").strip()
    remember_media = on_media_url or (lambda _url: None)
    remember_id = on_post_id or (lambda _id: None)

    if not api_key:
        return _done(api_key, BLOCKED, "No Zernio API key is saved. Add it in GTA Mode settings.")
    path = Path(video_path) if str(video_path or "") else None
    if path is None or not path.is_file():
        return _done(api_key, INVALID, f"The video file is missing: {path or '(empty)'}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        return _done(api_key, INVALID, f"The video file can't be read: {path} ({exc})")
    if size == 0:
        return _done(api_key, INVALID, f"The video file is empty: {path}")
    if size > REEL_MAX_BYTES:
        return _done(
            api_key,
            INVALID,
            f"The video is {size // (1024 * 1024)} MB - above Instagram's 1 GB reel "
            "limit. Trim or recompress it first.",
        )

    # phase 1: which Instagram account
    try:
        account_id, _username = _pick_instagram_id(api_key, account_hint)
    except _RateLimited as exc:
        return _blocked_rate(api_key, exc)
    except _PhaseFailure as exc:
        if exc.status_code == 0:
            # nothing was uploaded yet: an outage, not this clip's fault
            return _done(
                api_key,
                BLOCKED,
                "Can't reach Zernio right now (no internet?). AutoClip will try again.",
                retry_after=NETWORK_RETRY_WAIT,
                transient=True,
            )
        kind = BLOCKED if _account_problem(exc.status_code) else FAILED
        return _done(api_key, kind, exc.message)
    if not account_id:
        return _done(
            api_key,
            BLOCKED,
            "Zernio listed the Instagram account without an id. Reconnect it at zernio.com.",
        )

    # phase 2: upload the video (nothing is scheduled yet: failures are safe)
    try:
        upload_url, public_url = _presign(api_key, path)
        _put_media(upload_url, path, size)
    except _RateLimited as exc:
        return _blocked_rate(api_key, exc)
    except _PhaseFailure as exc:
        if exc.status_code in (401, 402):
            return _done(api_key, BLOCKED, exc.message)
        return _done(api_key, FAILED, exc.message, transient=_outage(exc.status_code))

    # phase 3: create the scheduled post - sent once, never blindly repeated
    body = {
        "content": caption or "",
        "mediaItems": [{"type": "video", "url": public_url}],
        "platforms": [{"platform": "instagram", "accountId": account_id}],
        "scheduledFor": _iso_utc(publish_at),
        "timezone": "UTC",
    }
    remember_media(public_url)
    sent_at = datetime.now(UTC)
    try:
        resp = _send_create(api_key, body)
    except _RateLimited as exc:
        return _blocked_rate(api_key, exc, media_url=public_url)
    except _PhaseFailure as exc:
        # provably never created (no connection / platform disabled)
        return _done(
            api_key,
            FAILED,
            exc.message,
            media_url=public_url,
            transient=_outage(exc.status_code),
        )
    except _Ambiguous as exc:
        # a 5xx answer or a dropped connection: maybe created, so it is
        # looked up, never re-sent - and it is an outage signal too
        return _resolve_ambiguous(
            api_key, public_url, str(exc), sent_at, remember_id, transient=True
        )

    if resp.status_code not in (200, 201, 207):
        message = _explain(_safe_json(resp), resp.status_code)
        if resp.status_code == 409:
            # "already being processed": it exists; find it instead of re-posting
            return _resolve_ambiguous(api_key, public_url, "HTTP 409", sent_at, remember_id)
        if resp.status_code == 422:
            return _done(api_key, INVALID, message, media_url=public_url)
        kind = BLOCKED if _account_problem(resp.status_code) else FAILED
        return _done(api_key, kind, message, media_url=public_url)

    data = _safe_json(resp)
    raw_post = data.get("post")
    post: dict = raw_post if isinstance(raw_post, dict) else {}
    post_id = str(post.get("_id") or "")
    if post_id:
        remember_id(post_id)
    state, url, error = _post_state(post)
    if state == "published":
        return _done(api_key, POSTED, "", post_id=post_id, post_url=url, media_url=public_url)
    if state == "failed":
        return _done(api_key, FAILED, error, post_id=post_id, media_url=public_url)
    if not post_id:
        return _resolve_ambiguous(
            api_key, public_url, "no post id in the answer", sent_at, remember_id
        )
    return _from_post(
        api_key,
        post,
        post_id,
        public_url,
        waiting_message="Zernio has the post and is publishing it.",
    )


def check_post(
    api_key: str,
    post_id: str = "",
    media_url: str = "",
    sent_at: datetime | None = None,
) -> ZernioResult:
    """Ask Zernio what became of a post (scheduled earlier, or unconfirmed).

    posted / failed / scheduled (still waiting at Zernio) / unconfirmed
    (couldn't tell; ``not_listed`` only after a COMPLETE search found no such
    post) / blocked (bad key, rate limit). ``sent_at`` - when the create
    call went out - anchors the media-URL lookup; without it only the last
    ``LOOKUP_DAYS`` days are searched.
    """
    api_key = (api_key or "").strip()
    post_id = (post_id or "").strip()
    media_url = (media_url or "").strip()
    if not api_key:
        return _done(api_key, BLOCKED, "No Zernio API key is saved. Add it in GTA Mode settings.")
    if not post_id and not media_url:
        return _done(api_key, INVALID, "Nothing to check: no Zernio post id or media URL.")
    try:
        post = _get_post(api_key, post_id) if post_id else None
        if post is None and media_url:
            post = _find_post_by_media(api_key, media_url, sent_at=sent_at)
    except _RateLimited as exc:
        return _blocked_rate(api_key, exc, post_id=post_id, media_url=media_url)
    except _PhaseFailure as exc:
        if exc.status_code == 401:
            return _done(api_key, BLOCKED, exc.message, post_id=post_id, media_url=media_url)
        return _done(
            api_key,
            UNCONFIRMED,
            exc.message,
            post_id=post_id,
            media_url=media_url,
            transient=_outage(exc.status_code),
        )
    if post is None:
        return _done(
            api_key,
            UNCONFIRMED,
            "Zernio does not list this post (yet).",
            post_id=post_id,
            media_url=media_url,
            not_listed=True,
        )
    found_id = str(post.get("_id") or post_id)
    return _from_post(
        api_key,
        post,
        found_id,
        media_url,
        waiting_message="Still waiting at Zernio to be published.",
    )
