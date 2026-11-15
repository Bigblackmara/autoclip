"""yt-dlp gets a JavaScript runtime, and network blocks are explained.

YouTube now serves storyboard images only unless yt-dlp can solve its player
challenge with a JS runtime. yt-dlp enables only Deno by default, so a machine
with just Node fails every download unless the runtime is named explicitly.
"""

from __future__ import annotations

import pytest
from autoclip.config import IngestSettings
from autoclip.pipeline import ingest


def test_node_is_offered_when_it_is_the_only_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        ingest.shutil, "which", lambda name: "C:/node.exe" if name == "node" else None
    )
    assert ingest.js_runtime_options() == {"js_runtimes": {"node": {}}}


def test_no_runtime_leaves_yt_dlp_defaults(monkeypatch) -> None:
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    assert ingest.js_runtime_options() == {}


@pytest.mark.parametrize(
    "raw",
    [
        "ERROR: [youtube] abc: HTTP Error 429: Too Many Requests",
        "WARNING: Only images are available for download.",
        "ERROR: [youtube] abc: No video formats found!",
    ],
)
def test_network_blocks_point_at_the_vpn(raw: str) -> None:
    error = ingest._translate_ytdlp_error(Exception(raw), IngestSettings())
    assert "refused to serve" in str(error)
    assert "VPN" in error.hint


def test_bot_check_still_asks_for_cookies() -> None:
    error = ingest._translate_ytdlp_error(
        Exception("Sign in to confirm you're not a bot"), IngestSettings()
    )
    assert "bot check" in str(error)
    assert "cookies_from_browser" in error.hint
