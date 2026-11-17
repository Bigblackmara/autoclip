"""Moving a scheduled YouTube video's publish time (videos.update, status part)."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime

import httpx
import pytest
from autoclip.live.uploads import tokens as tk
from autoclip.live.uploads import youtube as yt


def _bundle(scope: str) -> tk.TokenBundle:
    return tk.TokenBundle(
        access_token="acc", refresh_token="ref", expires_at=time.time() + 3600, scope=scope
    )


def test_scope_asks_for_upload_and_manage() -> None:
    assert tk.YOUTUBE_SCOPE.split() == [tk.YOUTUBE_UPLOAD_SCOPE, tk.YOUTUBE_MANAGE_SCOPE]


def test_upload_only_grant_must_reconnect(fake_keyring) -> None:
    tk.save_tokens("youtube", _bundle(tk.YOUTUBE_UPLOAD_SCOPE))
    with pytest.raises(tk.UploadApiError, match="Reconnect"):
        asyncio.run(yt.reschedule_youtube("vid", datetime(2026, 11, 18, 1, tzinfo=UTC)))


def test_moves_publish_at_and_keeps_other_status_fields(fake_keyring) -> None:
    tk.save_tokens("youtube", _bundle(tk.YOUTUBE_SCOPE))
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "status": {
                                "privacyStatus": "private",
                                "publishAt": "2026-11-17T17:32:00Z",
                                "selfDeclaredMadeForKids": False,
                                "license": "youtube",
                                "uploadStatus": "processed",
                            }
                        }
                    ]
                },
            )
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json={"status": body["status"]})

    async def run() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await yt.reschedule_youtube(
                "vid1", datetime(2026, 11, 18, 0, 10, tzinfo=UTC), client=client
            )

    assert asyncio.run(run()) == "2026-11-18T00:10:00Z"
    (body,) = sent
    assert body["id"] == "vid1"
    assert body["status"] == {
        "privacyStatus": "private",
        "publishAt": "2026-11-18T00:10:00Z",
        "selfDeclaredMadeForKids": False,
        "license": "youtube",
    }


def test_refuses_a_video_that_is_already_public(fake_keyring) -> None:
    tk.save_tokens("youtube", _bundle(tk.YOUTUBE_SCOPE))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json={"items": [{"status": {"privacyStatus": "public"}}]})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await yt.reschedule_youtube("v", datetime(2026, 11, 18, tzinfo=UTC), client=client)

    with pytest.raises(tk.UploadApiError, match="already public"):
        asyncio.run(run())
