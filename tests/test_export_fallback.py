"""A GPU-encoder failure at render time falls back to the CPU encoder.

NVENC can pass the start-up probe and still fail mid-render (encoder sessions
or GPU memory taken by other work, e.g. Live Watch renders plus a batch job).
The CPU encoder gives the same quality, only slower, so the export retries
there instead of failing the job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from autoclip.config import ExportSettings
from autoclip.pipeline import captions, export, ffmpeg
from autoclip.pipeline.reframe.croppath import centre_crop


def _request(tmp_path: Path) -> export.ExportRequest:
    return export.ExportRequest(
        source=tmp_path / "src.mp4",
        destination=tmp_path / "out" / "clip.mp4",
        start_s=0.0,
        end_s=4.0,
        crop_path=centre_crop(1280, 720, 4.0),
        words=[],
        style=captions.get_style("bold_pop"),
    )


@pytest.fixture
def fake_encoders(monkeypatch):
    def encoder_args(settings):
        if settings.prefer_hardware_encoder:
            return ["-c:v", "h264_nvenc"]
        return ["-c:v", "libx264"]

    monkeypatch.setattr(export, "encoder_args", encoder_args)


def _fake_run(calls: list, fail_when: str):
    def run(args, **_kwargs):
        calls.append(list(args))
        if fail_when in args:
            raise ffmpeg.FFmpegError(
                "Task finished with error code: -12 (Cannot allocate memory)",
                command=list(args),
            )
        Path(args[-1]).write_bytes(b"mp4")

    return run


def test_gpu_failure_retries_on_the_cpu(tmp_path, monkeypatch, fake_encoders) -> None:
    calls: list = []
    monkeypatch.setattr(export.ffmpeg, "run", _fake_run(calls, "h264_nvenc"))

    out = export.export_clip(
        _request(tmp_path),
        work_dir=tmp_path / "work",
        settings=ExportSettings(prefer_hardware_encoder=True),
    )

    assert out.read_bytes() == b"mp4"
    assert ["h264_nvenc" in c for c in calls] == [True, False]
    assert "libx264" in calls[1]


def test_cpu_failure_is_reported_without_a_retry(tmp_path, monkeypatch, fake_encoders) -> None:
    calls: list = []
    monkeypatch.setattr(export.ffmpeg, "run", _fake_run(calls, "libx264"))

    with pytest.raises(export.ExportError, match="Could not render clip.mp4"):
        export.export_clip(
            _request(tmp_path),
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False),
        )
    assert len(calls) == 1


def test_cancelled_render_is_not_retried(tmp_path, monkeypatch, fake_encoders) -> None:
    calls: list = []
    monkeypatch.setattr(export.ffmpeg, "run", _fake_run(calls, "h264_nvenc"))

    with pytest.raises(export.ExportError):
        export.export_clip(
            _request(tmp_path),
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=True),
            cancelled=lambda: True,
        )
    assert len(calls) == 1
