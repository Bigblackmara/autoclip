"""Instagram Reels uploads via the Graph API resumable protocol — official API only.

The documented flow (Instagram content publishing with Facebook Login —
Instagram Login does NOT expose the resumable rupload host):

1. ``POST /{ig_user_id}/media?media_type=REELS&upload_type=resumable`` with the
   caption → a container id plus an ``rupload.facebook.com/ig-api-upload``
   upload URI.
2. ``POST`` the raw file bytes to that URI with the ``offset: 0`` and
   ``file_size`` headers. This single call carries the phases older SDKs
   called ``start``/``append`` — the current protocol uploads the whole file
   in one POST and the endpoint reads exactly these fields.
3. Poll ``GET /{container_id}?fields=status_code`` until ``FINISHED``
   (Meta transcodes asynchronously — a container id proves nothing).
4. ``POST /{ig_user_id}/media_publish`` with ``creation_id=<container_id>``.

Real-world use needs an IG professional account linked to a Facebook page
plus Meta app review for ``instagram_content_publish`` — out of scope here;
the flow is coded as documented, and every call is mockable via httpx
``MockTransport`` in tests.

Common setup phase, implemented here rather than in tokens.py: the container
endpoint needs the IG professional **user id** the media belong to. The OAuth
grant alone doesn't carry it (Meta's response has no id without explicit
fields), so the first status()/upload() call resolves it via
``GET /me/accounts?fields=instagram_business_account{id,username}`` and
caches id + username in the token bundle (they survive token refresh —
refresh_tokens preserves those fields).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import httpx

from . import tokens as tk
from .tokens import (
    TokenBundle,
    UploadApiError,
    UploadMeta,
    UploadResult,
    api_request,
    ensure_fresh,
    require_rights_attestation,
)

log = logging.getLogger(__name__)

GRAPH_URL = "https://graph.facebook.com/{version}"

#: Poll cadence for the container status check (Meta recommends a few seconds
#: between polls; faster just burns rate limit).
STATUS_POLL_INTERVAL_S = 2.0
STATUS_POLL_TIMEOUT_S = 300.0


def _graph(path: str) -> str:
    return GRAPH_URL.format(version=tk.GRAPH_API_VERSION) + path


def _media_path(bundle: TokenBundle) -> str:
    if not bundle.account_ref:
        raise UploadApiError(
            "no Instagram user id stored — connect the account (loads the IG "
            "professional user id) before uploading"
        )
    return f"/{bundle.account_ref}/media"


def _file_size(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError as exc:
        raise UploadApiError(f"cannot read clip file: {exc.strerror or exc}") from exc


def _auth_header(bundle: TokenBundle) -> dict[str, str]:
    # rupload authenticates the user token in the OAuth bearer form.
    return {"Authorization": f"OAuth {bundle.access_token}"}


def _params(bundle: TokenBundle, **params) -> dict:
    return {**params, "access_token": bundle.access_token}


async def _resolve_account(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
) -> TokenBundle:
    """Resolve + cache the id/username of the IG professional account.

    One-time per connection (and after token refresh — bundle identity is
    retained through it; refresh_tokens preserves account/account_ref fields):
    ``GET /me/accounts?fields=instagram_business_account{id,username}``.
    """
    response = await api_request(
        http,
        "GET",
        _graph("/me/accounts"),
        params=_params(bundle, fields="instagram_business_account{id,username},name"),
        redact=(bundle.access_token,),
    )
    data = response.json().get("data") or []
    ig = None
    for page in data:
        candidate = page.get("instagram_business_account") or None
        if candidate and candidate.get("id"):
            ig = candidate
            break
    if not ig:
        raise UploadApiError(
            "no Instagram professional account linked to this Facebook page set — "
            "link the IG account to a page in Meta business settings first"
        )
    updated = TokenBundle(
        access_token=bundle.access_token,
        refresh_token=bundle.refresh_token,
        expires_at=bundle.expires_at,
        scope=bundle.scope,
        account=str(ig.get("username") or bundle.account or ""),
        account_ref=str(ig["id"]),
    )
    tk.save_tokens(tk.PLATFORM_INSTAGRAM, updated)
    return updated


async def _touch(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
) -> TokenBundle:
    """Load-and-resolve: cached id when present, resolve it when not."""
    if bundle.account_ref:
        return bundle
    return await _resolve_account(http, bundle)


async def _create_container(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
    meta: UploadMeta,
) -> tuple[str, str]:
    """Step 1: open the resumable session, return ``(container_id, upload_uri)``."""
    response = await api_request(
        http,
        "POST",
        _graph(_media_path(bundle)),
        params=_params(
            bundle,
            media_type="REELS",
            upload_type="resumable",
            caption=meta.caption,
        ),
        redact=(bundle.access_token,),
    )
    payload = response.json()
    container_id = str(payload.get("id") or "")
    upload_uri = str(payload.get("uri") or "")
    if not container_id or not upload_uri:
        raise UploadApiError("container response had no id/uri pair")
    return container_id, upload_uri


async def _upload_bytes(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
    upload_uri: str,
    path: Path,
) -> None:
    """Step 2: POST the clip's bytes to the rupload host (offset 0, file_size).

    Streams the file in 1 MiB pieces — the body never lands in RAM — and the
    handle is closed by the generator's ``finally`` even when the transport
    errors mid-stream.
    """
    size = _file_size(path)
    file_handle = open(path, "rb")  # noqa: SIM115 — closed in _agen's finally

    async def _agen():
        try:
            while chunk := file_handle.read(1024 * 1024):
                yield chunk
        finally:
            file_handle.close()

    try:
        response = await http.post(
            upload_uri,
            content=_agen(),
            headers={
                **_auth_header(bundle),
                "offset": "0",
                "file_size": str(size),
            },
        )
    except httpx.HTTPError as exc:
        raise UploadApiError(f"upload transport failed: {type(exc).__name__}") from exc
    if response.status_code >= 400:
        message = tk.redact_secrets(
            f"HTTP {response.status_code}: {response.text[:500]}", (bundle.access_token,)
        )
        raise UploadApiError(message)
    try:
        confirmed = bool(response.json().get("success", False))
    except ValueError:
        raise UploadApiError("rupload returned a non-JSON response") from None
    if not confirmed:
        raise UploadApiError("rupload host did not confirm the byte upload")


async def _wait_container(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
    container_id: str,
) -> None:
    """Step 3: poll the container until Meta finishes transcode (or errors)."""
    deadline = time.monotonic() + STATUS_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        response = await api_request(
            http,
            "GET",
            _graph(f"/{container_id}"),
            params=_params(bundle, fields="status_code"),
            redact=(bundle.access_token,),
        )
        status = str(response.json().get("status_code") or "")
        if status == "FINISHED":
            return
        if status in ("ERROR", "EXPIRED"):
            raise UploadApiError(f"Instagram container processing failed ({status})")
        await asyncio.sleep(STATUS_POLL_INTERVAL_S)
    raise UploadApiError(f"Instagram container processing timed out after {STATUS_POLL_TIMEOUT_S}s")


async def _publish(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
    container_id: str,
) -> str:
    """Step 4: publish the finished container, return the published media id."""
    response = await api_request(
        http,
        "POST",
        _graph(_media_path(bundle) + "_publish"),
        params=_params(bundle, creation_id=container_id),
        redact=(bundle.access_token,),
    )
    media_id = str(response.json().get("id") or "")
    if not media_id:
        raise UploadApiError("media_publish response had no id")
    return media_id


async def upload_instagram(
    path: Path | str,
    meta: UploadMeta,
    *,
    client: httpx.AsyncClient | None = None,
) -> UploadResult:
    """Upload the clip at ``path`` as a Reel to the connected IG account.

    Same gate order as YouTube: attestation, then token, then network. Raises
    UploadApiError when no IG professional account is reachable through the
    stored page set (the most common real-world failure — the token is fine,
    the page link is missing).
    """
    require_rights_attestation()
    file_path = Path(path)
    bundle = await ensure_fresh(tk.PLATFORM_INSTAGRAM, client=client)

    async with tk.api_client(client) as http:
        bundle = await _touch(http, bundle)
        container_id, upload_uri = await _create_container(http, bundle, meta)
        await _upload_bytes(http, bundle, upload_uri, file_path)
        await _wait_container(http, bundle, container_id)
        media_id = await _publish(http, bundle, container_id)

    return UploadResult(
        platform=tk.PLATFORM_INSTAGRAM,
        # Graph's media id shares the reels namespace; the permalink is in
        # the media edge, one extra call — the manager persists the id (the
        # canonical object URL), UI links through it.
        media_id=media_id,
        url=f"https://www.facebook.com/{tk.GRAPH_API_VERSION}/{media_id}",
        account=bundle.account,
    )


async def instagram_status(*, client: httpx.AsyncClient | None = None) -> dict:
    """``{connected, account}`` for the REST layer.

    Mirror of youtube_status: unlinked costs no network; linked accounts reuse
    the cached IG username and only call the Graph when it's unknown. Resolves
    (and stores) the IG user id + username on first call after connect.
    """
    bundle = tk.load_tokens(tk.PLATFORM_INSTAGRAM)
    if bundle is None:
        return {"connected": False, "account": None}
    try:
        fresh = await ensure_fresh(tk.PLATFORM_INSTAGRAM, bundle, client=client)
    except tk.UploadError as exc:
        log.warning("Instagram status: token unusable (%s)", exc)
        return {"connected": False, "account": None}
    if fresh.account and fresh.account_ref:
        return {"connected": True, "account": fresh.account}

    async with tk.api_client(client) as http:
        try:
            resolved = await _resolve_account(http, fresh)
        except UploadApiError as exc:
            log.warning("Instagram account lookup failed: %s", exc)
            return {"connected": False, "account": None}
    return {"connected": True, "account": resolved.account}
