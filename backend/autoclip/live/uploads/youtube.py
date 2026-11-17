"""YouTube uploads via the Data API v3 resumable protocol — official API only.

Why resumable and not ``uploadType=media``: live clips are tens of megabytes
rendered to disk, and a home upload link dropping at 90 % used to mean a full
re-run for nothing. The documented two-step flow costs one extra round trip
and gives byte-range resumption with plain ``httpx``:

1. ``POST /upload/youtube/v3/videos?uploadType=resumable&part=snippet,status``
   with the metadata JSON → the session URI arrives in the ``Location`` header.
2. ``PUT`` the file bytes to that URI. We send one request for the whole file
   (Google accepts single-shot PUTs within the media limit; chunk-splitting
   only pays off on chronically bad links). What IS implemented is the
   documented recovery: when a PUT fails with a retriable status we send the
   empty-PUT status probe (``Content-Range: bytes */TOTAL``), read the
   server's ``Range`` header, and the retry streams only the missing tail.

Scope is exactly ``youtube.upload`` — it grants upload and none of the read
scopes, which keeps the consent screen minimal. google-api-python-client is
forbidden by the work order and would hide this ~40-line protocol behind a
client abstraction (plus its own dependency tree).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC
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

UPLOAD_SESSION_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
CHANNEL_INFO_URL = "https://www.googleapis.com/youtube/v3/channels"
YOUTUBE_VIDEO_URL = "https://www.youtube.com/video/{id}"

#: HTTP statuses Google documents as retriable for resumable uploads.
RETRIABLE_STATUS = frozenset({500, 502, 503, 504})

CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _ResumableSession:
    """One open resumable session: where to PUT, and how much is left."""

    session_uri: str
    path: Path
    size: int
    content_type: str = "video/mp4"


def _auth_header(bundle: TokenBundle) -> dict[str, str]:
    return {"Authorization": f"Bearer {bundle.access_token}"}


def _metadata_payload(meta: UploadMeta) -> dict:
    """The ``part=snippet,status`` resource posted at session init."""
    snippet: dict = {
        "title": meta.title,
        "description": meta.description,
        "categoryId": meta.category_id,
    }
    if meta.tags:
        snippet["tags"] = meta.tags
    return {
        "snippet": snippet,
        "status": {
            "privacyStatus": meta.privacy,
            "selfDeclaredMadeForKids": meta.made_for_kids,
            # Scheduled publishing requires private (YouTube's rule): the
            # video stays private until publishAt, then goes public.
            **(
                {
                    "privacyStatus": "private",
                    "publishAt": meta.publish_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                if meta.publish_at is not None
                else {}
            ),
        },
    }


def _file_payload(session: _ResumableSession, offset: int):
    """A lazy async stream of the session file's bytes, from ``offset`` onward.

    Streaming rather than ``read_bytes()``: clips are tens of MB and the point
    of the resumable protocol is not buffering them in RAM. The ``offset``
    parameter is what makes a resumed retry send only the un-sent tail.
    """

    async def _aiter():
        with open(session.path, "rb") as fh:
            if offset:
                fh.seek(offset)
            while chunk := fh.read(CHUNK_BYTES):
                yield chunk

    return _aiter()


def _file_size(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError as exc:
        raise UploadApiError(f"cannot read clip file: {exc.strerror or exc}") from exc


async def _open_session(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
    session: _ResumableSession,
    meta: UploadMeta,
) -> str:
    """Step 1: initiate the session, return the URI to PUT bytes to."""
    response = await api_request(
        http,
        "POST",
        UPLOAD_SESSION_URL,
        params={"uploadType": "resumable", "part": "snippet,status"},
        json=_metadata_payload(meta),
        headers={
            **_auth_header(bundle),
            "X-Upload-Content-Type": session.content_type,
            "X-Upload-Content-Length": str(session.size),
        },
        redact=(bundle.access_token,),
    )
    session_uri = response.headers.get("Location")
    if not session_uri:
        raise UploadApiError("YouTube session init returned no Location header")
    return session_uri


async def _probe_offset(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
    session: _ResumableSession,
) -> int:
    """Ask the server how many bytes it already holds (the 308 status probe).

    Returns 0 when the probe fails — restarting the whole file is correct,
    just slower, so a broken probe never blocks the upload.
    """
    try:
        response = await http.put(
            session.session_uri,
            content=b"",
            headers={
                **_auth_header(bundle),
                "Content-Range": f"bytes */{session.size}",
            },
        )
    except httpx.HTTPError:
        return 0
    if response.status_code == 308:
        # "bytes=0-999999" → first unsent byte is 999999 + 1.
        try:
            return int(response.headers.get("Range", "").rsplit("-", 1)[1]) + 1
        except (ValueError, IndexError):
            return 0
    if response.status_code < 400:
        return session.size  # upload already complete server-side
    return 0


async def _put_file(
    http: httpx.AsyncClient,
    bundle: TokenBundle,
    session: _ResumableSession,
    *,
    max_retries: int = 2,
) -> dict:
    """Step 2: PUT the bytes; on retriable failures resume from the probe.

    Retriable statuses (5xx) keep the session alive; anything else is a real
    API rejection (bad metadata, quota, revoked token) and raises with the
    redacted body so the manager can surface it on the upload row.
    """
    offset = 0
    last_error: UploadApiError | None = None
    for attempt in range(max_retries + 1):
        headers = {
            **_auth_header(bundle),
            "Content-Type": session.content_type,
        }
        if offset >= session.size:
            # The server already holds every byte: an empty status PUT
            # returns the finished resource.
            headers["Content-Range"] = f"bytes */{session.size}"
            headers["Content-Length"] = "0"
            content = b""
        else:
            # A resumed PUT MUST say which bytes it carries; without
            # Content-Range the server reads the tail as the whole file.
            headers["Content-Length"] = str(session.size - offset)
            if offset > 0:
                headers["Content-Range"] = f"bytes {offset}-{session.size - 1}/{session.size}"
            content = _file_payload(session, offset)
        try:
            response = await http.put(session.session_uri, content=content, headers=headers)
        except httpx.HTTPError as exc:
            # A dropped connection or a read timeout can hit AFTER Google
            # stored every byte (the video exists). Failing here recorded
            # the upload as failed and a retry posted the clip twice; the
            # session is still open, so ask it what it holds and resume.
            last_error = UploadApiError(f"upload transport failed: {type(exc).__name__}")
            last_error.__cause__ = exc
            if attempt >= max_retries:
                break
            offset = await _probe_offset(http, bundle, session)
            log.info(
                "YouTube resumable retry %d/%d after %s, from byte %d/%d",
                attempt + 1,
                max_retries,
                type(exc).__name__,
                offset,
                session.size,
            )
            continue
        if response.status_code < 400:
            try:
                return response.json()
            except ValueError as exc:
                raise UploadApiError("upload response was not valid JSON") from exc
        if response.status_code not in RETRIABLE_STATUS:
            message = tk.redact_secrets(
                f"HTTP {response.status_code}: {response.text[:500]}", (bundle.access_token,)
            )
            raise UploadApiError(message)
        last_error = UploadApiError(f"upload interrupted (HTTP {response.status_code})")
        offset = await _probe_offset(http, bundle, session)
        log.info(
            "YouTube resumable retry %d/%d from byte %d/%d",
            attempt + 1,
            max_retries,
            offset,
            session.size,
        )
    raise last_error or UploadApiError("upload failed after retries")


def _remember_account(bundle: TokenBundle, account: str | None) -> None:
    """Persist a discovered account name back into the stored bundle.

    Saves the status/upload path a ``channels`` lookup on every call; the
    title only changes when the user renames their channel.
    """
    if not account or account == bundle.account:
        return
    tk.save_tokens(
        tk.PLATFORM_YOUTUBE,
        TokenBundle(
            access_token=bundle.access_token,
            refresh_token=bundle.refresh_token,
            expires_at=bundle.expires_at,
            scope=bundle.scope,
            account=account,
            account_ref=bundle.account_ref,
        ),
    )


async def upload_youtube(
    path: Path | str,
    meta: UploadMeta,
    *,
    client: httpx.AsyncClient | None = None,
) -> UploadResult:
    """Upload the clip at ``path`` to the connected YouTube channel.

    Gate order matters: the rights attestation is checked before any token or
    network activity, so an unattested machine can never even learn whether
    an account is linked.
    """
    require_rights_attestation()
    file_path = Path(path)
    bundle = await ensure_fresh(tk.PLATFORM_YOUTUBE, client=client)

    async with tk.api_client(client) as http:
        session = _ResumableSession(session_uri="", path=file_path, size=_file_size(file_path))
        session = _ResumableSession(
            session_uri=await _open_session(http, bundle, session, meta),
            path=session.path,
            size=session.size,
        )
        payload = await _put_file(http, bundle, session)

    media_id = str(payload.get("id") or "")
    if not media_id:
        raise UploadApiError("YouTube upload response had no video id")
    account = (payload.get("snippet") or {}).get("channelTitle") or bundle.account
    _remember_account(bundle, account)
    return UploadResult(
        platform=tk.PLATFORM_YOUTUBE,
        media_id=media_id,
        url=YOUTUBE_VIDEO_URL.format(id=media_id),
        account=account,
    )


async def youtube_status(*, client: httpx.AsyncClient | None = None) -> dict:
    """``{connected, account}`` for the REST layer.

    Unlinked accounts cost no network. Linked ones check token freshness
    (refreshing if stale) and use the cached channel title when present; the
    one-time ``channels?mine=true`` lookup discovers it and is persisted back.
    """
    bundle = tk.load_tokens(tk.PLATFORM_YOUTUBE)
    if bundle is None:
        return {"connected": False, "account": None}
    try:
        fresh = await ensure_fresh(tk.PLATFORM_YOUTUBE, bundle, client=client)
    except tk.UploadError as exc:
        # A dead grant reports "not connected" — the UI's connect button is
        # the remedy, and a raising status endpoint would just break the
        # settings page for a reconnectable account.
        log.warning("YouTube status: token unusable (%s)", exc)
        return {"connected": False, "account": None}
    if fresh.account:
        return {"connected": True, "account": fresh.account}

    async with tk.api_client(client) as http:
        try:
            response = await api_request(
                http,
                "GET",
                CHANNEL_INFO_URL,
                params={"part": "snippet", "mine": "true"},
                headers=_auth_header(fresh),
                redact=(fresh.access_token,),
            )
        except UploadApiError as exc:
            log.warning("YouTube channel lookup failed: %s", exc)
            return {"connected": True, "account": None}
        items = response.json().get("items") or []
        account = (items[0].get("snippet") or {}).get("title") if items else None
    _remember_account(fresh, account)
    return {"connected": True, "account": account}
