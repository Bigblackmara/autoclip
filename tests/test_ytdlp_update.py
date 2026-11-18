"""yt-dlp self-update: version decisions, the failure-triggered retry, status.

PyPI and pip are always fakes here: no network, no real installs.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from autoclip import ytdlp_update
from autoclip.config import IngestSettings
from autoclip.pipeline import ffmpeg, ingest
from autoclip.ytdlp_update import Updater

T0 = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


class Fakes:
    """A fake PyPI + pip + installed version, wired into an :class:`Updater`."""

    def __init__(self, installed: str | None, latest: str | Exception, pip_ok: bool = True):
        self.installed = installed
        self.latest = latest
        self.pip_ok = pip_ok
        self.pip_calls = 0
        self.pypi_calls = 0
        self.now = T0

    def fetch(self) -> str:
        self.pypi_calls += 1
        if isinstance(self.latest, Exception):
            raise self.latest
        return self.latest

    def upgrade(self) -> tuple[bool, str]:
        self.pip_calls += 1
        if self.pip_ok and isinstance(self.latest, str):
            self.installed = self.latest
            return True, "Successfully installed yt-dlp"
        return False, "ERROR: Could not install packages\nnetwork down"

    def updater(self) -> Updater:
        return Updater(
            fetch_latest=self.fetch,
            upgrade=self.upgrade,
            installed=lambda: self.installed,
            now=lambda: self.now,
        )


@pytest.mark.parametrize(
    ("candidate", "current", "newer"),
    [
        ("2026.11.1", "2026.10.26", True),
        ("2026.10.26.1", "2026.10.26", True),
        ("2026.10.26", "2026.10.26", False),
        ("2026.9.30", "2026.10.26", False),
        ("2026.10.26", None, True),
        (None, "2026.10.26", False),
        ("2026.11.1.dev0", "2026.10.26", True),
    ],
)
def test_is_newer(candidate: str | None, current: str | None, newer: bool) -> None:
    assert ytdlp_update.is_newer(candidate, current) is newer


def test_up_to_date_does_not_run_pip() -> None:
    fakes = Fakes("2026.10.26", "2026.10.26")

    state = fakes.updater().check(reason="startup")

    assert state.last_result == "up_to_date"
    assert fakes.pip_calls == 0
    assert ytdlp_update.load_state().last_check_at == T0.isoformat(timespec="seconds")


def test_newer_on_pypi_runs_pip_and_records_it() -> None:
    fakes = Fakes("2026.10.26", "2026.11.15")

    state = fakes.updater().check(reason="scheduled")

    assert fakes.pip_calls == 1
    assert state.last_result == "updated"
    assert state.installed_version == "2026.11.15"
    assert state.last_update_at is not None
    assert "2026.10.26 -> 2026.11.15" in state.last_message


def test_pip_failure_is_recorded_not_raised() -> None:
    fakes = Fakes("2026.10.26", "2026.11.15", pip_ok=False)

    state = fakes.updater().check(reason="scheduled")

    assert state.last_result == "failed"
    assert "network down" in state.last_message
    assert state.installed_version == "2026.10.26"


def test_pypi_unreachable_skips_pip() -> None:
    fakes = Fakes("2026.10.26", httpx.ConnectError("no route"))

    state = fakes.updater().check(reason="startup")

    assert state.last_result == "failed"
    assert "PyPI" in state.last_message
    assert fakes.pip_calls == 0


def test_fetch_latest_version_reads_pypi_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ytdlp_update.PYPI_URL
        return httpx.Response(200, json={"info": {"version": "2026.11.15"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    assert ytdlp_update.fetch_latest_version(client) == "2026.11.15"


def test_upgrade_command_prefers_pip_and_falls_back_to_uv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        def __init__(self, code: int) -> None:
            self.returncode = code

    command = ytdlp_update.upgrade_command(lambda *a, **k: Result(0))
    assert command is not None and command[1:5] == ["-m", "pip", "install", "--upgrade"]

    monkeypatch.setattr(ytdlp_update.shutil, "which", lambda name: "/bin/uv")
    command = ytdlp_update.upgrade_command(lambda *a, **k: Result(1))
    assert command is not None and command[:3] == ["/bin/uv", "pip", "install"]

    monkeypatch.setattr(ytdlp_update.shutil, "which", lambda name: None)
    assert ytdlp_update.upgrade_command(lambda *a, **k: Result(1)) is None


@pytest.mark.parametrize(
    "message",
    [
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot",
        "ERROR: [youtube] abc: nsig extraction failed: Some formats may be missing",
        "ERROR: [youtube] abc: Unable to extract uploader id",
        "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    ],
)
def test_outdated_errors_trigger_one_update_and_say_retry(message: str) -> None:
    fakes = Fakes("2026.10.26", "2026.11.15")
    updater = fakes.updater()

    assert updater.update_for_failure(message, failed_version="2026.10.26") is True
    assert fakes.pip_calls == 1
    assert ytdlp_update.load_state().last_reason == "download_failure"


def test_unrelated_errors_never_update() -> None:
    fakes = Fakes("2026.10.26", "2026.11.15")

    assert fakes.updater().update_for_failure("Private video", "2026.10.26") is False
    assert fakes.pypi_calls == 0


def test_no_retry_when_nothing_newer_exists() -> None:
    fakes = Fakes("2026.10.26", "2026.10.26")

    assert fakes.updater().update_for_failure("nsig extraction failed", "2026.10.26") is False
    assert fakes.pypi_calls == 1 and fakes.pip_calls == 0


def test_failure_checks_are_rate_limited() -> None:
    fakes = Fakes("2026.10.26", "2026.10.26")
    updater = fakes.updater()

    updater.update_for_failure("nsig extraction failed", "2026.10.26")
    fakes.now = T0 + timedelta(minutes=10)
    updater.update_for_failure("nsig extraction failed", "2026.10.26")
    assert fakes.pypi_calls == 1

    fakes.now = T0 + timedelta(hours=2)
    updater.update_for_failure("nsig extraction failed", "2026.10.26")
    assert fakes.pypi_calls == 2


def test_status_reports_restart_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    fakes = Fakes("2026.11.15", "2026.11.15")
    monkeypatch.setattr(ytdlp_update, "loaded_version", lambda: "2026.10.26")

    status = fakes.updater().status(auto_update=True)

    assert status["installed_version"] == "2026.11.15"
    assert status["loaded_version"] == "2026.10.26"
    assert status["restart_pending"] is True


# --------------------------------------------------------------------------
# ingest_youtube: update once, retry once in a fresh interpreter
# --------------------------------------------------------------------------


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(path: Path) -> ffmpeg.MediaInfo:
        return ffmpeg.MediaInfo(path=path, duration_s=60.0, has_video=True, has_audio=True)

    monkeypatch.setattr(ingest, "_probe_and_validate", probe)


def _write_media(target_dir: Path) -> None:
    (target_dir / "source.mp4").write_bytes(b"video")


def test_outdated_failure_updates_and_retries_in_a_subprocess(
    monkeypatch: pytest.MonkeyPatch, fake_probe: None
) -> None:
    calls: list[str] = []
    versions = {"installed": "2026.10.26"}

    def in_process(url, settings, target_dir, on_progress=None):
        calls.append("in_process")
        raise ingest.YtdlpFailure("ERROR: nsig extraction failed", download_error=True)

    def in_subprocess(url, settings, target_dir, on_progress=None):
        calls.append("subprocess")
        _write_media(target_dir)
        return {"title": "Fresh", "uploader": "Chan", "duration": 60}

    def update_for_failure(message, failed_version):
        calls.append(f"update:{failed_version}")
        versions["installed"] = "2026.11.15"
        return True

    monkeypatch.setattr(ingest, "download_in_process", in_process)
    monkeypatch.setattr(ingest, "download_in_subprocess", in_subprocess)
    monkeypatch.setattr(ytdlp_update, "stale_in_process", lambda: False)
    monkeypatch.setattr(ytdlp_update, "loaded_version", lambda: "2026.10.26")
    monkeypatch.setattr(ytdlp_update, "installed_version", lambda: versions["installed"])
    monkeypatch.setattr(ytdlp_update.updater, "update_for_failure", update_for_failure)

    source = ingest.ingest_youtube("https://youtu.be/abc", IngestSettings())

    assert calls == ["in_process", "update:2026.10.26", "subprocess"]
    assert source.title == "Fresh" and source.channel == "Chan"


def test_no_retry_when_auto_update_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    def in_process(url, settings, target_dir, on_progress=None):
        raise ingest.YtdlpFailure("ERROR: nsig extraction failed", download_error=True)

    def never(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("must not update")

    monkeypatch.setattr(ingest, "download_in_process", in_process)
    monkeypatch.setattr(ytdlp_update, "stale_in_process", lambda: False)
    monkeypatch.setattr(ytdlp_update.updater, "update_for_failure", never)

    with pytest.raises(ingest.IngestError, match="yt-dlp could not download"):
        ingest.ingest_youtube("https://youtu.be/abc", IngestSettings(ytdlp_auto_update=False))


def test_retry_failure_is_translated_once(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url, settings, target_dir, on_progress=None):
        raise ingest.YtdlpFailure("ERROR: Sign in to confirm you're not a bot", download_error=True)

    monkeypatch.setattr(ingest, "download_in_process", fail)
    monkeypatch.setattr(ingest, "download_in_subprocess", fail)
    monkeypatch.setattr(ytdlp_update, "stale_in_process", lambda: False)
    monkeypatch.setattr(ytdlp_update.updater, "update_for_failure", lambda m, v: True)

    with pytest.raises(ingest.IngestError, match="bot check"):
        ingest.ingest_youtube("https://youtu.be/abc", IngestSettings())


def test_after_an_update_downloads_go_to_the_subprocess(
    monkeypatch: pytest.MonkeyPatch, fake_probe: None
) -> None:
    used: list[str] = []

    def in_subprocess(url, settings, target_dir, on_progress=None):
        used.append("subprocess")
        _write_media(target_dir)
        return {"title": "T"}

    monkeypatch.setattr(ingest, "download_in_subprocess", in_subprocess)
    monkeypatch.setattr(ytdlp_update, "stale_in_process", lambda: True)

    ingest.ingest_youtube("https://youtu.be/abc", IngestSettings())

    assert used == ["subprocess"]


class RecordingStdin(io.StringIO):
    """Keeps what was written after close(), as a real pipe's reader would."""

    written = ""

    def close(self) -> None:
        self.written = self.getvalue()
        super().close()


class FakeProcess:
    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self.stdin = RecordingStdin()
        self.stdout = iter(f"{line}\n" for line in lines)
        self.returncode = returncode
        self.payload: dict[str, Any] | None = None

    def wait(self) -> int:
        return self.returncode


def test_subprocess_protocol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    started: dict[str, Any] = {}
    process = FakeProcess(
        [
            "[youtube] chatter",
            f"{ingest.WORKER_PROGRESS}0.5000",
            ingest.WORKER_RESULT + json.dumps({"ok": True, "metadata": {"title": "X"}}),
        ]
    )

    def popen(command, **kwargs):
        started["command"] = command
        started["env"] = kwargs["env"]
        return process

    monkeypatch.setattr(ingest, "_popen", popen)
    progress: list[float] = []

    metadata = ingest.download_in_subprocess(
        "https://youtu.be/abc", IngestSettings(), tmp_path, progress.append
    )

    assert metadata == {"title": "X"}
    assert progress == [0.5]
    assert started["command"][1:] == ["-m", "autoclip.pipeline.ytdlp_worker"]
    package_root = str(Path(ingest.__file__).resolve().parents[2])
    assert started["env"]["PYTHONPATH"].split(os.pathsep)[0] == package_root
    assert json.loads(process.stdin.written)["url"] == "https://youtu.be/abc"


def test_subprocess_failure_and_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    failed = FakeProcess(
        [ingest.WORKER_RESULT + json.dumps({"ok": False, "message": "boom", "download_error": 1})],
        1,
    )
    monkeypatch.setattr(ingest, "_popen", lambda command, **kwargs: failed)
    with pytest.raises(ingest.YtdlpFailure, match="boom") as caught:
        ingest.download_in_subprocess("u", IngestSettings(), tmp_path)
    assert caught.value.download_error is True

    crashed = FakeProcess(["Traceback (most recent call last):", "ImportError: x"], 1)
    monkeypatch.setattr(ingest, "_popen", lambda command, **kwargs: crashed)
    with pytest.raises(ingest.YtdlpFailure, match="crashed") as caught:
        ingest.download_in_subprocess("u", IngestSettings(), tmp_path)
    assert caught.value.download_error is False


def test_worker_main_runs_the_in_process_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoclip.pipeline import ytdlp_worker

    def in_process(url, settings, target_dir, on_progress=None):
        on_progress(0.25)
        return {"title": url}

    monkeypatch.setattr(ingest, "download_in_process", in_process)
    payload = {"url": "u1", "target_dir": str(tmp_path), "settings": IngestSettings().model_dump()}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    assert ytdlp_worker.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"{ingest.WORKER_PROGRESS}0.2500"
    assert json.loads(out[1][len(ingest.WORKER_RESULT) :]) == {
        "ok": True,
        "metadata": {"title": "u1"},
    }


def test_status_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.app import ENV_NO_WORKER, create_app
    from fastapi.testclient import TestClient

    fakes = Fakes("2026.10.26", "2026.11.15")
    monkeypatch.setattr(ytdlp_update, "updater", fakes.updater())
    monkeypatch.setenv(ENV_NO_WORKER, "1")
    with TestClient(create_app()) as client:
        before = client.get("/api/system/ytdlp").json()
        after = client.post("/api/system/ytdlp/check").json()

    assert before["auto_update"] is True and before["last_check_at"] is None
    assert after["last_result"] == "updated"
    assert after["installed_version"] == "2026.11.15"
    assert after["latest_version"] == "2026.11.15"


def test_service_skips_checks_while_auto_update_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from autoclip import config

    settings = config.Settings()
    settings.ingest.ytdlp_auto_update = False
    config.save(settings)
    fakes = Fakes("2026.10.26", "2026.11.15")
    monkeypatch.setattr(ytdlp_update, "STARTUP_DELAY_S", 0)

    service = ytdlp_update.UpdaterService(fakes.updater())

    async def run() -> None:
        await service.start()
        await asyncio.sleep(0.2)
        await service.stop()

    asyncio.run(run())
    assert fakes.pypi_calls == 0

    settings.ingest.ytdlp_auto_update = True
    config.save(settings)
    asyncio.run(run())
    assert fakes.pypi_calls == 1 and fakes.pip_calls == 1


def test_run_upgrade_reports_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def runner(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "pip 25", "")
        assert kwargs["timeout"] == ytdlp_update.PIP_TIMEOUT_S
        return subprocess.CompletedProcess(command, 1, "", "ERROR: no network")

    ok, output = ytdlp_update.run_upgrade(runner)

    assert ok is False and "no network" in output
