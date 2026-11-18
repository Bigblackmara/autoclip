"""Withdrawing a post the platform already holds.

* Instagram (Zernio): ``DELETE /posts/{id}`` removes a scheduled post.
* YouTube: a scheduled video is private with ``publishAt``. Unscheduling
  (keep it private, drop ``publishAt``) or deleting it needs the manage
  scope (``youtube``); an upload-only grant can do neither, and the user is
  told to use YouTube Studio or reconnect.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..gta import zernio
from ..live.uploads import tokens as tk
from ..live.uploads.tokens import UploadApiError, api_request, ensure_fresh
from ..live.uploads.youtube import VIDEOS_URL, can_edit

STUDIO_NOTE = (
    "This YouTube connection only allows uploads, so AutoClip can't unschedule the video. "
    "Open YouTube Studio > Content, open the video and change its visibility (or delete it) "
    "there - or press Reconnect in GTA Mode's Posting tab and allow managing your YouTube "
    "account, then try again here."
)


@dataclass
class RemoteResult:
    ok: bool
    message: str = ""


# -------------------------------------------------------------- Instagram
def delete_zernio_post(api_key: str, post_id: str) -> RemoteResult:
    """Delete a post at Zernio (a scheduled reel is then never published)."""
    api_key = (api_key or "").strip()
    if not api_key:
        return RemoteResult(False, "No Zernio API key is saved.")
    if not post_id:
        return RemoteResult(False, "Zernio never confirmed a post id for this reel.")
    try:
        resp = zernio._request("DELETE", f"{zernio.POSTS_URL}/{post_id}", api_key)
    except zernio._RateLimited as exc:
        return RemoteResult(False, zernio._sanitize(exc.message, api_key))
    except zernio._PhaseFailure as exc:
        return RemoteResult(False, zernio._sanitize(exc.message, api_key))
    if resp.status_code in (200, 202, 204):
        return RemoteResult(True, "Removed from Zernio; it will not be posted.")
    if resp.status_code == 404:
        return RemoteResult(True, "Zernio no longer has this post.")
    message = zernio._explain(zernio._safe_json(resp), resp.status_code)
    return RemoteResult(False, zernio._sanitize(message, api_key))


# -------------------------------------------------------------- YouTube
def youtube_can_edit() -> bool:
    """Whether the stored YouTube grant includes the manage scope."""
    try:
        return can_edit(tk.load_tokens(tk.PLATFORM_YOUTUBE))
    except tk.UploadError:
        return False


async def youtube_make_private(video_id: str, *, client: httpx.AsyncClient | None = None) -> str:
    """Drop a scheduled video's ``publishAt`` so it stays private; returns its privacy."""
    bundle = await ensure_fresh(tk.PLATFORM_YOUTUBE, client=client)
    if not can_edit(bundle):
        raise UploadApiError(STUDIO_NOTE)
    auth = {"Authorization": f"Bearer {bundle.access_token}"}
    async with tk.api_client(client) as http:
        listed = await api_request(
            http,
            "GET",
            VIDEOS_URL,
            params={"part": "status", "id": video_id},
            headers=auth,
            redact=(bundle.access_token,),
        )
        items = listed.json().get("items") or []
        if not items:
            raise UploadApiError(f"Video {video_id} was not found on the channel.")
        current = items[0].get("status") or {}
        if current.get("privacyStatus") != "private":
            raise UploadApiError(
                f"Video {video_id} is already {current.get('privacyStatus')}; "
                "change it in YouTube Studio."
            )
        # Everything writable is sent back unchanged (YouTube resets omitted
        # fields); publishAt is left out, which unschedules the video.
        keep = (
            "embeddable",
            "license",
            "publicStatsViewable",
            "selfDeclaredMadeForKids",
            "containsSyntheticMedia",
        )
        status = {k: current[k] for k in keep if k in current}
        status["privacyStatus"] = "private"
        response = await api_request(
            http,
            "PUT",
            VIDEOS_URL,
            params={"part": "status"},
            json={"id": video_id, "status": status},
            headers=auth,
            redact=(bundle.access_token,),
        )
    return str((response.json().get("status") or {}).get("privacyStatus") or "private")


async def youtube_delete(video_id: str, *, client: httpx.AsyncClient | None = None) -> None:
    """Delete the video from the channel (manage scope required)."""
    bundle = await ensure_fresh(tk.PLATFORM_YOUTUBE, client=client)
    if not can_edit(bundle):
        raise UploadApiError(STUDIO_NOTE)
    async with tk.api_client(client) as http:
        await api_request(
            http,
            "DELETE",
            VIDEOS_URL,
            params={"id": video_id},
            headers={"Authorization": f"Bearer {bundle.access_token}"},
            redact=(bundle.access_token,),
        )
