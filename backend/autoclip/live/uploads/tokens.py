"""OAuth storage + the Google/Meta HTTP-OAuth plumbing for Live uploads.

Everything secret lives in the OS keyring under service ``autoclip`` (used
through ``autoclip.config.set_secret``/``get_secret``), never in config.json:
config files get synced, screenshotted, and pasted into bug reports at a rate
API keys never are, and an OAuth token is a full account grant. The
plaintext-fallback that ``config.set_secret`` has for provider API keys is
therefore explicitly refused here — if no keyring backend is usable we raise
instead of quietly downgrading the security of an account-wide credential.

Client id/secret are the user's own (their Google Cloud / Meta app). They are
stored with :func:`save_client_credentials` before any authorization URL can
be built — A10's REST layer wires a settings endpoint to those helpers.

No new dependencies: token refresh and the code exchange are plain ``httpx``
form POSTs/GETs (google-api-python-client is banned by the work order; it
would also stock our test suite's ``MockTransport`` seams behind three
client-API layers).

The rights-attestation gate (spec hard rule): uploads refuse to run until the
keyring key ``live_rights_attestation`` holds a string starting with
``accepted:``. The UI's acceptance endpoint (A10) writes it via
:func:`accept_rights`. The gate string itself is not a secret, so it MAY fall
back to config storage on keyringless machines — tokens and client secrets
may not.
"""

from __future__ import annotations

import json
import logging
import secrets as _secrets
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlencode

import httpx

from ... import config

log = logging.getLogger(__name__)

PlatformName = Literal["youtube", "instagram"]
PLATFORM_YOUTUBE: PlatformName = "youtube"
PLATFORM_INSTAGRAM: PlatformName = "instagram"


# --------------------------------------------------------------------------
# Keyring keys (service is always config.KEYRING_SERVICE == "autoclip")
# --------------------------------------------------------------------------

YOUTUBE_TOKENS_KEY = "live_youtube_tokens"
INSTAGRAM_TOKENS_KEY = "live_instagram_tokens"
YOUTUBE_CLIENT_ID_KEY = "live_youtube_client_id"
YOUTUBE_CLIENT_SECRET_KEY = "live_youtube_client_secret"
INSTAGRAM_CLIENT_ID_KEY = "live_instagram_client_id"
INSTAGRAM_CLIENT_SECRET_KEY = "live_instagram_client_secret"

#: Rights-attestation flag: uploads refuse to run unless this keyring entry
#: starts with ``accepted:``. The manager and REST layer code against
#: :data:`ATTESTATION_KEY` and the :func:`accept_rights` helpers.
ATTESTATION_KEY = "live_rights_attestation"

_ATTESTATION_PREFIX = "accepted:"


# --------------------------------------------------------------------------
# OAuth endpoints / scopes
# --------------------------------------------------------------------------

YOUTUBE_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
#: Facebook-Login permission set for IG content publishing; the resumable
#: rupload host requires the Facebook Login product, not Instagram Login.
INSTAGRAM_SCOPES = "instagram_content_publish,pages_show_list"

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
META_DIALOG_URL = "https://www.facebook.com/{version}/dialog/oauth"
META_TOKEN_URL = "https://graph.facebook.com/{version}/oauth/access_token"

#: Graph API version pin — bump this one constant when the platform forces a
#: newer one; every module in this package builds URLs from it.
GRAPH_API_VERSION = "v27.0"

DEFAULT_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)

_TOKEN_KEYS: dict[str, str] = {
    PLATFORM_YOUTUBE: YOUTUBE_TOKENS_KEY,
    PLATFORM_INSTAGRAM: INSTAGRAM_TOKENS_KEY,
}
_CLIENT_KEYS: dict[str, tuple[str, str]] = {
    PLATFORM_YOUTUBE: (YOUTUBE_CLIENT_ID_KEY, YOUTUBE_CLIENT_SECRET_KEY),
    PLATFORM_INSTAGRAM: (INSTAGRAM_CLIENT_ID_KEY, INSTAGRAM_CLIENT_SECRET_KEY),
}


# --------------------------------------------------------------------------
# Errors — one hierarchy the manager/REST can catch at the right altitude
# --------------------------------------------------------------------------


class UploadError(Exception):
    """Base for every failure an upload attempt can raise."""


class RightsNotAccepted(UploadError):
    """The rights-attestation gate blocked the upload (spec hard rule).

    The user must accept the legal notice in the UI (A10's endpoint writes
    the flag) before anything is posted to either platform.
    """


class NotLinkedError(UploadError):
    """No OAuth tokens are stored for the platform (account not connected)."""


class TokenStateError(UploadError):
    """Stored OAuth state is unusable (corrupt blob, missing refresh token,
    missing client credentials, or no keyring backend to store secrets)."""


class OAuthExchangeError(UploadError):
    """The authorization-code exchange failed (A10's connect flow)."""


class TokenRefreshError(UploadError):
    """The refresh-token exchange was rejected — the user must reconnect."""


class UploadApiError(UploadError):
    """A platform API call failed mid-upload (transport error or >=400)."""


# --------------------------------------------------------------------------
# Small data classes shared by both uploaders
# --------------------------------------------------------------------------


@dataclass
class TokenBundle:
    """One platform's OAuth grant as stored (as JSON) in the keyring.

    ``account`` is a display name (YouTube channel title, IG username) and
    ``account_ref`` the platform-side id API paths are built from (IG
    professional-user id; unused for YouTube). Both are refreshed lazily by
    the status calls and saved back.
    """

    access_token: str
    refresh_token: str | None = None
    #: Unix epoch seconds; None means "no expiry information — assume valid".
    expires_at: float | None = None
    scope: str = ""
    account: str | None = None
    account_ref: str | None = None

    def is_expired(self, margin_s: float = 60.0) -> bool:
        if self.expires_at is None:
            return False
        return self.expires_at - margin_s <= time.time()

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> TokenBundle:
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or not data.get("access_token"):
                raise ValueError("blob is not a token object")
            return cls(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                expires_at=(
                    float(data["expires_at"]) if data.get("expires_at") is not None else None
                ),
                scope=str(data.get("scope") or ""),
                account=data.get("account"),
                account_ref=data.get("account_ref"),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            # Never include `raw` in the message — it IS the credential.
            raise TokenStateError(f"stored token blob is corrupt ({type(exc).__name__})") from exc


@dataclass
class UploadMeta:
    """The user-visible bits the manager composes from a live clip row.

    ``privacy`` applies to YouTube (Instagram has no per-post privacy);
    ``made_for_kids`` maps to ``status.selfDeclaredMadeForKids``. Titles are
    clamped to the platform limits here so the API never rejects a clip for a
    length rule we can enforce locally.
    """

    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    privacy: str = "private"
    #: YouTube category "20" = Gaming, the natural home for stream clips.
    category_id: str = "20"
    made_for_kids: bool = False
    #: YouTube scheduled publishing: the video uploads private and YouTube
    #: makes it public at this (timezone-aware) moment. None = publish per
    #: ``privacy`` right away.
    publish_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.privacy not in ("private", "unlisted", "public"):
            raise ValueError(f"privacy must be private/unlisted/public, got {self.privacy!r}")
        if self.publish_at is not None and self.publish_at.tzinfo is None:
            raise ValueError("publish_at must be timezone-aware")
        self.title = self.title[:100]  # YouTube's hard title limit
        self.description = self.description[:5000]  # YouTube's description limit

    @property
    def caption(self) -> str:
        """Instagram caption (2200-char platform limit, enforced here)."""
        caption = self.title if not self.description else f"{self.title}\n\n{self.description}"
        return caption[:2200]


@dataclass(frozen=True)
class UploadResult:
    """A finished upload — maps 1:1 onto a ``live_uploads`` row's outcome
    fields (``upload_url``/``platform_media_id``) without knowing about the
    store."""

    platform: PlatformName
    media_id: str
    url: str
    account: str | None = None


# --------------------------------------------------------------------------
# Keyring storage
# --------------------------------------------------------------------------


def _check_platform(platform: str) -> str:
    if platform not in _TOKEN_KEYS:
        raise ValueError(f"unknown platform {platform!r}")
    return platform


def _store_strict(key: str, value: str) -> None:
    """Store a credential, refusing the config.json fallback.

    ``config.set_secret`` falls back to a plaintext config entry when no
    keyring backend exists — fine for LLM API keys, not fine for account-wide
    OAuth grants. Undo the disk write and fail loudly instead.
    """
    if not config.set_secret(key, value):
        config.delete_secret(key)
        raise TokenStateError(
            "no usable OS keyring backend found; OAuth credentials are never "
            "written to config.json (on headless Linux install a keyring "
            "backend such as secretstorage)"
        )


def save_tokens(platform: PlatformName, bundle: TokenBundle) -> None:
    """Persist a platform's token bundle to the keyring."""
    _check_platform(platform)
    _store_strict(_TOKEN_KEYS[platform], bundle.to_json())


def load_tokens(platform: PlatformName) -> TokenBundle | None:
    """Load a platform's token bundle, or None if the account is not linked.

    Raises TokenStateError if a stored blob exists but cannot be parsed —
    silent None here would hide a keyring corruption behind "not connected".
    """
    _check_platform(platform)
    raw = config.get_secret(_TOKEN_KEYS[platform])
    if raw is None:
        return None
    return TokenBundle.from_json(raw)


def clear_tokens(platform: PlatformName) -> None:
    """Remove a platform's token blob (user disconnect / logout)."""
    _check_platform(platform)
    config.delete_secret(_TOKEN_KEYS[platform])


def save_client_credentials(platform: PlatformName, client_id: str, client_secret: str) -> None:
    """Store the user's own OAuth app credentials for ``platform``.

    Required before :func:`authorization_url` / :func:`exchange_code` /
    refresh can work. The secret half is treated exactly like a token.
    """
    _check_platform(platform)
    id_key, secret_key = _CLIENT_KEYS[platform]
    _store_strict(id_key, client_id)
    _store_strict(secret_key, client_secret)


def load_client_credentials(platform: PlatformName) -> tuple[str, str] | None:
    """Return ``(client_id, client_secret)`` or None when unset."""
    _check_platform(platform)
    id_key, secret_key = _CLIENT_KEYS[platform]
    client_id = config.get_secret(id_key)
    client_secret = config.get_secret(secret_key)
    if not client_id or not client_secret:
        return None
    return client_id, client_secret


# --------------------------------------------------------------------------
# Rights-attestation gate
# --------------------------------------------------------------------------


def accept_rights() -> str:
    """Record that the user accepted the rights notice (UI acceptance endpoint).

    Returns the stored attestation string, ``accepted:<iso-date>``.
    """
    value = f"{_ATTESTATION_PREFIX}{datetime.now(UTC).isoformat(timespec='seconds')}"
    # Not a secret: the config fallback is acceptable if no keyring is present.
    config.set_secret(ATTESTATION_KEY, value)
    return value


def rights_attestation() -> str | None:
    """The stored attestation string, or None if never accepted."""
    return config.get_secret(ATTESTATION_KEY)


def has_rights_attestation() -> bool:
    """Whether uploads are unlocked (attestation present and well-formed)."""
    stored = rights_attestation()
    return bool(stored) and stored.startswith(_ATTESTATION_PREFIX)


def clear_rights() -> None:
    """Forget the acceptance (used if the user re-reads/re-revokes the notice)."""
    config.delete_secret(ATTESTATION_KEY)


def require_rights_attestation() -> None:
    """Gate every upload: raise :class:`RightsNotAccepted` unless unlocked."""
    if not has_rights_attestation():
        raise RightsNotAccepted(
            "uploads are locked: accept the content-rights notice in the Live "
            "settings first (keyring flag 'live_rights_attestation')"
        )


# --------------------------------------------------------------------------
# HTTP plumbing shared with the uploader modules
# --------------------------------------------------------------------------


def redact_secrets(text: str, secrets: Iterable[str | None]) -> str:
    """Replace every secret occurrence with a placeholder.

    Errors from platform APIs routinely echo parts of the request; without
    this, a 400 from Meta could put the access token in an exception that
    then prints in a terminal or a log file.
    """
    out = text
    for value in sorted((s for s in secrets if s), key=len, reverse=True):
        out = out.replace(value, "«redacted»")
    return out[:1000]  # cap error text bodies at a sane length


@asynccontextmanager
async def api_client(client: httpx.AsyncClient | None = None) -> AsyncIterator[httpx.AsyncClient]:
    """Yield the caller's client, or open a configured throwaway one.

    Callers pass their own client so tests can inject ``MockTransport``; when
    none is given a fresh client is created only inside this context, so a
    cache-friendly path (tokens already fresh) never touches the network.
    """
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as own:
        yield own


async def api_request(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
    data: dict | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
    redact: Iterable[str | None] = (),
) -> httpx.Response:
    """One API call, with every failure mode converted to UploadApiError.

    httpx exception messages embed the full request URL and body; wrapping
    here guarantees secrets never reach an exception string.
    """
    try:
        response = await http.request(
            method, url, params=params, json=json, data=data, content=content, headers=headers
        )
    except httpx.HTTPError as exc:
        host = httpx.URL(url).host
        raise UploadApiError(f"{method} {host} failed: {type(exc).__name__}") from exc
    if response.status_code >= 400:
        body = response.text[:500]
        message = redact_secrets(f"HTTP {response.status_code}: {body}", redact)
        log.debug("live upload API error: %s", message)
        raise UploadApiError(message)
    return response


# --------------------------------------------------------------------------
# Authorization URL / code exchange / refresh
# --------------------------------------------------------------------------


def authorization_url(
    platform: PlatformName,
    *,
    redirect_uri: str,
    state: str | None = None,
    client_id: str | None = None,
) -> str:
    """Build the consent page URL the user's browser is pointed at (A10 flow).

    ``state`` is generated when not supplied (CSRF binding is A10's choice of
    session storage; the parameter is there so it can enforce round-tripping).
    """
    _check_platform(platform)
    if client_id is None:
        stored = load_client_credentials(platform)
        if stored is None:
            raise TokenStateError(
                f"client id/secret for {platform} are not stored; save them via "
                "save_client_credentials before starting the OAuth flow"
            )
        client_id = stored[0]
    state = state if state is not None else _secrets.token_urlsafe(16)

    if platform == PLATFORM_YOUTUBE:
        # access_type=offline + prompt=consent: Google only issues the
        # refresh token on first consent, so consent is forced every time —
        # otherwise the second connect silently loses offline access.
        query = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": YOUTUBE_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{GOOGLE_AUTH_URL}?{urlencode(query)}"

    query = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": INSTAGRAM_SCOPES,
        "state": state,
    }
    return META_DIALOG_URL.format(version=GRAPH_API_VERSION) + f"?{urlencode(query)}"


async def exchange_code(
    platform: PlatformName,
    code: str,
    *,
    redirect_uri: str,
    client: httpx.AsyncClient | None = None,
) -> TokenBundle:
    """Exchange the OAuth redirect's ``code`` for stored tokens (A10 calls this
    from its callback endpoint; refresh + upload are ours).

    YouTube returns a refresh token (offline access). Meta returns a
    short-lived token that is immediately upgraded to a long-lived (~60 day)
    one via the documented fb_exchange_token grant.
    """
    _check_platform(platform)
    creds = load_client_credentials(platform)
    if creds is None:
        raise TokenStateError(
            f"client id/secret for {platform} are not stored; cannot exchange code"
        )
    client_id, client_secret = creds
    redact = (code, client_secret)

    async with api_client(client) as http:
        if platform == PLATFORM_YOUTUBE:
            response = await api_request(
                http,
                "POST",
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                redact=redact,
            )
            try:
                payload = response.json()
                access_token = payload["access_token"]
                expires_in = float(payload["expires_in"])
            except (ValueError, KeyError) as exc:
                raise OAuthExchangeError("token endpoint returned a malformed response") from exc
            bundle = TokenBundle(
                access_token=access_token,
                # Present on first consent; per Google docs may be omitted on
                # re-consent for an already-authorized user — keep None then.
                refresh_token=payload.get("refresh_token"),
                expires_at=time.time() + expires_in,
                scope=str(payload.get("scope") or YOUTUBE_SCOPE),
            )
        else:
            token_url = META_TOKEN_URL.format(version=GRAPH_API_VERSION)
            response = await api_request(
                http,
                "GET",
                token_url,
                params={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "code": code,
                },
                redact=redact,
            )
            try:
                short = response.json()["access_token"]
            except (ValueError, KeyError) as exc:
                raise OAuthExchangeError("token endpoint returned a malformed response") from exc
            short_expiry = response.json().get("expires_in")

            # Upgrade to a long-lived token (~60 days). If the upgrade fails
            # we keep the short-lived one and warn: the account still works,
            # it just silently expires in ~an hour without this heads-up.
            access_token, expires_in = short, float(short_expiry or 0) or None
            try:
                response = await api_request(
                    http,
                    "GET",
                    token_url,
                    params={
                        "grant_type": "fb_exchange_token",
                        "fb_exchange_token": short,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                    redact=redact,
                )
                payload = response.json()
                access_token = payload["access_token"]
                expires_in = float(payload["expires_in"])
            except UploadApiError as exc:
                log.warning("Instagram long-lived token upgrade failed: %s", exc)

            bundle = TokenBundle(
                access_token=access_token,
                expires_at=time.time() + expires_in if expires_in else None,
                scope=INSTAGRAM_SCOPES,
            )

    save_tokens(platform, bundle)
    return bundle


async def refresh_tokens(
    platform: PlatformName,
    bundle: TokenBundle | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> TokenBundle:
    """Refresh a stale access token, persisting the new bundle.

    On failure the previously stored bundle is left untouched, so a rejected
    refresh (revoked grant, rotated secret) never destroys the last known
    good state.
    """
    _check_platform(platform)
    if bundle is None:
        bundle = load_tokens(platform)
        if bundle is None:
            raise NotLinkedError(f"{platform} account is not connected")
    creds = load_client_credentials(platform)
    if creds is None:
        raise TokenStateError(
            f"client id/secret for {platform} are not stored; cannot refresh tokens"
        )
    client_id, client_secret = creds
    redact = (bundle.access_token, bundle.refresh_token, client_secret)

    async with api_client(client) as http:
        # A rejected refresh (revoked grant, rotated secret) is a distinct
        # failure from a generic API hiccup: the caller must tell the user to
        # reconnect, not retry.
        try:
            if platform == PLATFORM_YOUTUBE:
                if not bundle.refresh_token:
                    raise TokenStateError(
                        f"{platform} tokens expired and no refresh token is stored — "
                        "reconnect the account in settings"
                    )
                response = await api_request(
                    http,
                    "POST",
                    GOOGLE_TOKEN_URL,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "grant_type": "refresh_token",
                        "refresh_token": bundle.refresh_token,
                    },
                    redact=redact,
                )
                try:
                    payload = response.json()
                    access_token = payload["access_token"]
                    expires_in = float(payload["expires_in"])
                except (ValueError, KeyError) as exc:
                    raise TokenRefreshError(
                        "refresh response was malformed; reconnect the account"
                    ) from exc
                new_bundle = TokenBundle(
                    access_token=access_token,
                    # Google's refresh response carries no new refresh token.
                    refresh_token=bundle.refresh_token,
                    expires_at=time.time() + expires_in,
                    scope=str(payload.get("scope") or bundle.scope),
                    account=bundle.account,
                    account_ref=bundle.account_ref,
                )
            else:
                token_url = META_TOKEN_URL.format(version=GRAPH_API_VERSION)
                response = await api_request(
                    http,
                    "GET",
                    token_url,
                    params={
                        "grant_type": "fb_exchange_token",
                        "fb_exchange_token": bundle.access_token,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                    redact=redact,
                )
                try:
                    payload = response.json()
                    access_token = payload["access_token"]
                    expires_in = float(payload["expires_in"])
                except (ValueError, KeyError) as exc:
                    raise TokenRefreshError(
                        "refresh response was malformed; reconnect the account"
                    ) from exc
                new_bundle = TokenBundle(
                    access_token=access_token,
                    expires_at=time.time() + expires_in,
                    scope=bundle.scope,
                    account=bundle.account,
                    account_ref=bundle.account_ref,
                )
        except UploadApiError as exc:
            raise TokenRefreshError(
                f"{platform} rejected the token refresh; reconnect the account ({exc})"
            ) from exc

    save_tokens(platform, new_bundle)
    return new_bundle


async def ensure_fresh(
    platform: PlatformName,
    bundle: TokenBundle | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    margin_s: float = 60.0,
) -> TokenBundle:
    """Return usable tokens, refreshing first when close to expiry.

    Unexpired tokens are returned with no network activity at all — the
    status calls lean on this to stay cheap when polling from the UI.
    """
    _check_platform(platform)
    if bundle is None:
        bundle = load_tokens(platform)
        if bundle is None:
            raise NotLinkedError(f"{platform} account is not connected")
    if not bundle.is_expired(margin_s):
        return bundle
    return await refresh_tokens(platform, bundle, client=client)
