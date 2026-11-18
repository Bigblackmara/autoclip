"""Health checks with fake probes (no subprocesses, no network)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoclip import app as app_module
from autoclip import config, doctor, system
from autoclip.app import create_app
from autoclip.cli import app as cli_app
from autoclip.system import FFmpegInfo, GPUInfo, OptionalDeps, SystemReport
from fastapi.testclient import TestClient
from typer.testing import CliRunner


def good_report(**ff_changes) -> SystemReport:
    ff = FFmpegInfo(
        found=True,
        path="/usr/bin/ffmpeg",
        version="7.1",
        has_libass=True,
        has_fontconfig=True,
        has_libx264=True,
        has_nvenc=True,
        nvenc_works=True,
        filters={"crop", "scale", "ass", "concat", "loudnorm", "drawtext"},
        ffprobe_found=True,
    )
    for key, value in ff_changes.items():
        setattr(ff, key, value)
    return SystemReport(
        python_version="3.11.9",
        python_ok=True,
        platform="Test OS",
        ffmpeg=ff,
        gpu=GPUInfo(accel="cuda", name="RTX 4090", ctranslate2_cuda=True),
        deps=OptionalDeps(
            keyring_backend=True, faster_whisper=True, mediapipe=True, scenedetect=True
        ),
    )


def make_probes(tmp_path: Path, **changes) -> doctor.Probes:
    settings = config.Settings()
    db = tmp_path / "autoclip.db"
    base = doctor.Probes(
        report=lambda: good_report(),
        which=lambda name: "/usr/bin/node" if name == "node" else None,
        module_available=lambda name: True,
        ytdlp_version=lambda: "2026.11.01",
        ytdlp_loaded=lambda: None,
        settings=lambda: settings,
        get_secret=lambda name, s: "sk-test",
        youtube_state=lambda: {"rights": True, "client": True, "linked": True, "can_edit": True},
        zernio_key=lambda: "zk",
        zernio_live=lambda key: (True, "Connected to Instagram (@me) via Zernio."),
        provider_live=lambda name: (True, "ok"),
        disk_usage=lambda path: SimpleNamespace(free=500 * doctor.GB),
        home=lambda: tmp_path,
        db_path=lambda: db,
        db_version=lambda path: 7,
        schema_version=lambda: 7,
        today=lambda: date(2026, 11, 18),
    )
    return replace(base, **changes)


def by_id(checks: list[doctor.Check]) -> dict[str, doctor.Check]:
    return {c.id: c for c in checks}


class TestChecks:
    def test_healthy_machine_is_all_ok(self, tmp_path: Path) -> None:
        checks = doctor.run_checks(probes=make_probes(tmp_path))
        bad = [c for c in checks if c.status != "ok"]
        assert bad == []
        ids = set(by_id(checks))
        assert {
            "ffmpeg",
            "ffprobe",
            "libass",
            "drawtext",
            "nvenc",
            "ytdlp",
            "js_runtime",
            "llm",
            "keyring",
            "youtube",
            "zernio",
            "disk",
            "db",
            "cuda",
        } <= ids
        assert doctor.summary(checks)["status"] == "ok"
        assert "not contacted" in by_id(checks)["llm"].detail

    def test_video_tool_problems(self, tmp_path: Path) -> None:
        p = make_probes(tmp_path, report=lambda: good_report(found=False))
        c = by_id(doctor.run_checks(probes=p))
        assert c["ffmpeg"].status == "fail" and c["ffmpeg"].fix
        assert "libass" not in c  # nothing more to say without ffmpeg
        report = good_report(has_libass=False, has_nvenc=True, nvenc_works=False)
        report.ffmpeg.filters.discard("drawtext")
        c = by_id(doctor.run_checks(probes=make_probes(tmp_path, report=lambda: report)))
        assert c["libass"].status == "fail"
        assert c["drawtext"].status == "warn"
        assert c["nvenc"].status == "warn" and "driver" in c["nvenc"].fix

    def test_downloads(self, tmp_path: Path) -> None:
        p = make_probes(
            tmp_path,
            ytdlp_version=lambda: "2026.08.01",
            which=lambda name: None,
            module_available=lambda name: False,
        )
        c = by_id(doctor.run_checks(probes=p))
        assert c["ytdlp"].status == "warn" and "update-ytdlp" in c["ytdlp"].fix
        assert c["js_runtime"].status == "warn" and "Node.js" in c["js_runtime"].fix
        assert c["ytdlp_ejs"].status == "warn"
        c = by_id(doctor.run_checks(probes=make_probes(tmp_path, ytdlp_version=lambda: None)))
        assert c["ytdlp"].status == "fail"
        p = make_probes(tmp_path, ytdlp_loaded=lambda: "2026.10.01")
        assert "Restart" in by_id(doctor.run_checks(probes=p))["ytdlp"].fix

    def test_llm_config_and_live(self, tmp_path: Path) -> None:
        c = by_id(doctor.run_checks(probes=make_probes(tmp_path, get_secret=lambda n, s: None)))
        assert c["llm"].status == "fail" and "set-secret" in c["llm"].fix
        calls: list[str] = []

        def live(name: str) -> tuple[bool, str]:
            calls.append(name)
            return False, "401 unauthorized"

        assert doctor.run_checks(probes=make_probes(tmp_path, provider_live=live)) and calls == []
        c = by_id(doctor.run_checks(live=True, probes=make_probes(tmp_path, provider_live=live)))
        assert calls == ["anthropic"] and c["llm"].status == "fail"

    def test_secrets_and_connections(self, tmp_path: Path) -> None:
        report = good_report()
        report.deps.keyring_backend = False
        p = make_probes(
            tmp_path,
            report=lambda: report,
            youtube_state=lambda: {
                "rights": True,
                "client": True,
                "linked": True,
                "can_edit": False,
                "account": "My Channel",
            },
            zernio_key=lambda: "",
        )
        c = by_id(doctor.run_checks(probes=p))
        assert c["keyring"].status == "warn"
        assert c["youtube"].status == "warn" and "upload-only" in c["youtube"].detail
        assert c["zernio"].status == "warn" and c["zernio"].fix
        p = make_probes(tmp_path, youtube_state=lambda: {"rights": False})
        assert "rights notice" in by_id(doctor.run_checks(probes=p))["youtube"].detail
        p = make_probes(tmp_path, zernio_live=lambda key: (False, "Zernio rejected the API key"))
        assert by_id(doctor.run_checks(live=True, probes=p))["zernio"].status == "fail"

    def test_storage_and_gpu(self, tmp_path: Path) -> None:
        p = make_probes(tmp_path, disk_usage=lambda path: SimpleNamespace(free=1 * doctor.GB))
        assert by_id(doctor.run_checks(probes=p))["disk"].status == "fail"
        p = make_probes(tmp_path, disk_usage=lambda path: SimpleNamespace(free=5 * doctor.GB))
        assert by_id(doctor.run_checks(probes=p))["disk"].status == "warn"
        for version, status in ((None, "warn"), (5, "warn"), (9, "fail"), (-1, "fail")):
            p = make_probes(tmp_path, db_version=lambda path, v=version: v)
            assert by_id(doctor.run_checks(probes=p))["db"].status == status
        report = good_report()
        report.gpu = GPUInfo(name="GTX 1060", cuda_libraries_missing=["cuBLAS"])
        c = by_id(doctor.run_checks(probes=make_probes(tmp_path, report=lambda: report)))
        assert c["cuda"].status == "warn" and "cuBLAS" in c["cuda"].detail
        report.gpu = GPUInfo()
        c = by_id(doctor.run_checks(probes=make_probes(tmp_path, report=lambda: report)))
        assert c["cuda"].status == "ok"

    def test_real_db_version_reader(self, initialised_db: int, tmp_path: Path) -> None:
        from autoclip import paths

        assert doctor._db_version(paths.db_path()) == initialised_db
        assert doctor._db_version(tmp_path / "missing.db") is None

    def test_report_shape(self, tmp_path: Path) -> None:
        out = doctor.doctor_report(probes=make_probes(tmp_path, zernio_key=lambda: ""))
        assert out["summary"] == {
            "status": "warn",
            "ok": len(out["checks"]) - 1,
            "warn": 1,
            "fail": 0,
        }
        assert all(
            {"id", "group", "label", "status", "detail", "fix"} <= set(c) for c in out["checks"]
        )


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def test_api_doctor(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[bool] = []

    original = doctor.doctor_report

    def fake(*, live: bool = False):
        seen.append(live)
        return original(live=live, probes=make_probes(tmp_path))

    monkeypatch.setattr(doctor, "doctor_report", fake)
    body = client.get("/api/system/doctor").json()
    assert body["summary"]["status"] == "ok" and body["checks"]
    client.get("/api/system/doctor?live=true")
    assert seen == [False, True]


def test_cli_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(system, "refresh", lambda: good_report())
    original = doctor.run_checks

    def fake_checks(live: bool = False, probes=None):  # noqa: ARG001
        return original(live=live, probes=make_probes(tmp_path))

    monkeypatch.setattr(doctor, "run_checks", fake_checks)
    result = CliRunner().invoke(cli_app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["summary"]["status"] == "ok"
    table = CliRunner().invoke(cli_app, ["doctor"])
    assert "Downloads, AI, posting, storage" in table.output
