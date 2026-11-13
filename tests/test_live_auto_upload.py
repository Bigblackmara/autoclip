"""Auto-upload integration: manager._maybe_auto_upload.

Proves the wave-3 wiring: a clip that reaches "ready" with
settings.live.uploads.<platform>.auto_upload on produces a persisted
live_uploads row; with auto_upload off (the default) nothing happens.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from autoclip.live.manager import LiveManager
from autoclip.live.models import LiveClipRow, LiveUploadRow
from autoclip.live.settings import LiveSettings
from autoclip.live.watcher import ClipWorkItem


@dataclass
class _UploadResult:
    url: str = "https://youtu.be/fake123"
    media_id: str = "fake123"


class _FakeUploads:
    """Stands in for the uploads package via sys.modules patching."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def upload_youtube(self, path: Any, meta: Any) -> _UploadResult:
        self.calls.append(("youtube", str(path)))
        return _UploadResult()

    async def upload_instagram(self, path: Any, meta: Any) -> _UploadResult:
        self.calls.append(("instagram", str(path)))
        return _UploadResult()


class _FakeStore:
    def __init__(self) -> None:
        self.uploads: dict[str, LiveUploadRow] = {}

    def create_upload(self, row: LiveUploadRow) -> LiveUploadRow:
        self.uploads[row.id] = row
        return row

    def update_upload(self, upload_id: str, **fields: Any) -> None:
        row = self.uploads[upload_id]
        for key, value in fields.items():
            setattr(row, key, value)

    def get_clip(self, clip_id: str) -> LiveClipRow:
        return LiveClipRow(
            id=clip_id, stream_id="s1", status="ready", rendered_path="D:/fake/clip.mp4"
        )


@dataclass
class _Judgement:
    score: int = 90
    title: str = "Judged title"
    hook: str = ""
    reason: str = ""
    start_word_offset: int | None = None
    end_word_offset: int | None = None


@pytest.fixture()
def fake_upload_modules(monkeypatch: pytest.MonkeyPatch) -> _FakeUploads:
    """Install standalone fake upload modules.

    The manager imports ``from .uploads import youtube`` — which executes the
    real ``uploads/__init__.py`` and re-exports many names from each module.
    Standalone fakes must therefore provide EVERY name ``__init__`` imports
    from that module, or the package import itself blows up.
    """
    fake = _FakeUploads()

    def install(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    async def _no_status() -> dict:
        return {"connected": False, "account": ""}

    install(
        "autoclip.live.uploads.youtube",
        upload_youtube=fake.upload_youtube,
        youtube_status=_no_status,
    )
    install(
        "autoclip.live.uploads.instagram",
        upload_instagram=fake.upload_instagram,
        instagram_status=_no_status,
    )
    tokens_mod = install(
        "autoclip.live.uploads.tokens",
        UploadMeta=lambda **kw: types.SimpleNamespace(**kw),
        ATTESTATION_KEY="live_rights_attestation",
        RightsNotAccepted=type("RightsNotAccepted", (Exception,), {}),
        UploadError=type("UploadError", (Exception,), {}),
        UploadResult=_UploadResult,
        accept_rights=lambda: "accepted:now",
        authorization_url=lambda *a, **k: "https://example.invalid",
        clear_rights=lambda: None,
        clear_tokens=lambda *a: None,
        exchange_code=lambda *a, **k: None,
        has_rights_attestation=lambda: False,
        load_client_credentials=lambda *a: None,
        load_tokens=lambda *a: None,
        refresh_tokens=lambda *a, **k: None,
        require_rights_attestation=lambda: None,
        rights_attestation=lambda: "",
        save_client_credentials=lambda *a: None,
        save_tokens=lambda *a: None,
    )

    # In a suite run the REAL uploads package is already imported, and
    # ``from .uploads import youtube`` resolves through the package's
    # cached submodule attributes — not sys.modules. Patch both paths.
    # (raising=False: the package holds no youtube attr until first import)
    import autoclip.live.uploads as _uploads_pkg

    monkeypatch.setattr(
        _uploads_pkg, "youtube", sys.modules["autoclip.live.uploads.youtube"], raising=False
    )
    monkeypatch.setattr(
        _uploads_pkg, "instagram", sys.modules["autoclip.live.uploads.instagram"], raising=False
    )
    monkeypatch.setattr(_uploads_pkg, "tokens", tokens_mod, raising=False)
    return fake


def _clip_row() -> LiveClipRow:
    return _FakeStore().get_clip("c1")


def _item(clip_id: str = "c1") -> ClipWorkItem:
    return ClipWorkItem(
        stream_id="s1",
        clip_id=clip_id,
        label="test stream",
        source_path=Path("D:/fake/judge_window.mp4"),
        clip_start_s=0.0,
        clip_end_s=10.0,
        trigger_start_s=0.0,
        trigger_end_s=10.0,
        judgement=_Judgement(),
    )


def _manager(settings: LiveSettings) -> LiveManager:
    mgr = LiveManager(settings=settings)
    mgr._store = _FakeStore()  # noqa: SLF001 - test seam
    return mgr


@pytest.mark.asyncio
async def test_auto_upload_opt_in_persists_done_row(fake_upload_modules: _FakeUploads) -> None:
    settings = LiveSettings()
    settings.uploads.youtube.auto_upload = True
    mgr = _manager(settings)
    await mgr._maybe_auto_upload(_item(), _clip_row(), "Test clip")  # noqa: SLF001
    assert fake_upload_modules.calls == [("youtube", "D:/fake/clip.mp4")]
    rows = list(mgr._store.uploads.values())  # noqa: SLF001
    assert len(rows) == 1
    assert rows[0].status == "done"
    assert rows[0].upload_url == "https://youtu.be/fake123"


@pytest.mark.asyncio
async def test_auto_upload_default_off_uploads_nothing(fake_upload_modules: _FakeUploads) -> None:
    mgr = _manager(LiveSettings())  # auto_upload defaults False
    await mgr._maybe_auto_upload(_item(), _clip_row(), "Test clip")  # noqa: SLF001
    assert fake_upload_modules.calls == []
    assert mgr._store.uploads == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_refusal_records_skipped_not_failed(fake_upload_modules: _FakeUploads) -> None:
    settings = LiveSettings()
    settings.uploads.youtube.auto_upload = True

    class RightsNotAccepted(Exception):
        pass

    async def refuse(path: Any, meta: Any) -> Any:
        raise RightsNotAccepted("accept the rights notice first")

    sys.modules["autoclip.live.uploads.youtube"].upload_youtube = refuse  # type: ignore[attr-defined]
    mgr = _manager(settings)
    await mgr._maybe_auto_upload(_item(), _clip_row(), "Test clip")  # noqa: SLF001
    rows = list(mgr._store.uploads.values())  # noqa: SLF001
    assert len(rows) == 1
    assert rows[0].status == "skipped"
    assert "rights" in (rows[0].error or "").lower()


@pytest.mark.asyncio
async def test_token_error_records_failed_row(fake_upload_modules: _FakeUploads) -> None:
    settings = LiveSettings()
    settings.uploads.youtube.auto_upload = True

    async def boom(path: Any, meta: Any) -> Any:
        raise RuntimeError("TokenRefreshError: refresh failed")

    sys.modules["autoclip.live.uploads.youtube"].upload_youtube = boom  # type: ignore[attr-defined]
    mgr = _manager(settings)
    await mgr._maybe_auto_upload(_item(), _clip_row(), "Test clip")  # noqa: SLF001
    rows = list(mgr._store.uploads.values())  # noqa: SLF001
    assert rows[0].status == "failed"
    assert "refresh failed" in (rows[0].error or "")


@pytest.mark.asyncio
async def test_instagram_opt_in_uses_instagram(fake_upload_modules: _FakeUploads) -> None:
    settings = LiveSettings()
    settings.uploads.instagram.auto_upload = True
    mgr = _manager(settings)
    await mgr._maybe_auto_upload(_item(), _clip_row(), "Whistle king")  # noqa: SLF001
    assert [p for p, _ in fake_upload_modules.calls] == ["instagram"]
    rows = list(mgr._store.uploads.values())  # noqa: SLF001
    assert rows[0].status == "done"
