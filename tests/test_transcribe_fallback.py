"""GPU problems must degrade transcription to the CPU, not fail the job.

The real failure this guards against: CTranslate2 loads the Whisper model on
CUDA fine, then the first encode dies with ``Library cublas64_12.dll is not
found or cannot be loaded`` — five percent into an hour-long job.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoclip import system
from autoclip.config import WhisperSettings
from autoclip.pipeline import transcribe as transcribe_mod

CUBLAS_ERROR = "Library cublas64_12.dll is not found or cannot be loaded"


def _segments():
    words = [
        SimpleNamespace(word=" hello", start=0.0, end=0.4),
        SimpleNamespace(word=" world.", start=0.4, end=0.9),
    ]
    return [SimpleNamespace(text=" hello world.", start=0.0, end=0.9, words=words)]


def _fake_model(fail_on: dict[str, str], created: list[tuple[str, str]]):
    """A WhisperModel stand-in. ``fail_on`` maps device -> 'load' | 'encode'."""

    class FakeWhisperModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            created.append((device, compute_type))
            self.device = device
            if fail_on.get(device) == "load":
                raise RuntimeError(CUBLAS_ERROR)

        def transcribe(self, audio: str, **_kwargs):
            if fail_on.get(self.device) == "encode":
                raise RuntimeError(CUBLAS_ERROR)
            return iter(_segments()), SimpleNamespace(duration=0.9, language="en")

    return FakeWhisperModel


@pytest.fixture
def on_cuda(monkeypatch):
    monkeypatch.setattr(transcribe_mod, "resolve_compute", lambda _s: ("cuda", "float16"))
    monkeypatch.setattr(transcribe_mod, "_cpu_compute_type", lambda: "int8")


def _use(monkeypatch, model_cls) -> None:
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", model_cls)


class TestGPUFallback:
    @pytest.mark.parametrize("phase", ["load", "encode"])
    def test_cuda_runtime_failure_retries_on_cpu(self, monkeypatch, on_cuda, phase) -> None:
        created: list[tuple[str, str]] = []
        _use(monkeypatch, _fake_model({"cuda": phase}, created))
        progress: list[float] = []

        result = transcribe_mod.transcribe(
            Path("audio.wav"), WhisperSettings(), on_progress=progress.append
        )

        assert [w.text for w in result.words] == ["hello", "world."]
        assert created == [("cuda", "float16"), ("cpu", "int8")]
        assert progress[-1] == 1.0

    def test_cpu_failure_after_fallback_is_reported(self, monkeypatch, on_cuda) -> None:
        created: list[tuple[str, str]] = []
        _use(monkeypatch, _fake_model({"cuda": "encode", "cpu": "load"}, created))

        with pytest.raises(transcribe_mod.TranscriptionError):
            transcribe_mod.transcribe(Path("audio.wav"), WhisperSettings())

    def test_non_gpu_errors_are_not_swallowed(self, monkeypatch, on_cuda) -> None:
        class BrokenAudio:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def transcribe(self, *_args, **_kwargs):
                raise ValueError("Invalid data found when processing input")

        _use(monkeypatch, BrokenAudio)

        with pytest.raises(ValueError, match="Invalid data"):
            transcribe_mod.transcribe(Path("audio.wav"), WhisperSettings())

    def test_cpu_jobs_never_retry(self, monkeypatch) -> None:
        monkeypatch.setattr(transcribe_mod, "resolve_compute", lambda _s: ("cpu", "int8"))
        created: list[tuple[str, str]] = []
        _use(monkeypatch, _fake_model({"cpu": "encode"}, created))

        with pytest.raises(RuntimeError, match="cublas"):
            transcribe_mod.transcribe(Path("audio.wav"), WhisperSettings())
        assert created == [("cpu", "int8")]


class TestCudaLibraryProbe:
    def test_unloadable_libraries_are_named(self, monkeypatch) -> None:
        import ctypes

        def refuse(*_args, **_kwargs):
            raise OSError("not found")

        monkeypatch.setattr(ctypes, "CDLL", refuse)

        assert system.missing_cuda_libraries() == ["cuBLAS", "cuDNN"]

    def test_loadable_libraries_report_nothing_missing(self, monkeypatch) -> None:
        import ctypes

        monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: object())

        assert system.missing_cuda_libraries() == []

    def test_gpu_without_runtime_libraries_is_not_offered(self, monkeypatch) -> None:
        if sys.platform == "darwin":
            pytest.skip("no CUDA path on macOS")
        fake_ct2 = SimpleNamespace(
            get_cuda_device_count=lambda: 1,
            get_supported_compute_types=lambda device: {"int8", "float32"},
        )
        monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
        monkeypatch.setattr(system.shutil, "which", lambda _name: None)
        monkeypatch.setattr(system, "missing_cuda_libraries", lambda: ["cuBLAS"])

        info = system.probe_gpu()

        assert info.accel == "cpu"
        assert not info.ctranslate2_cuda
        assert info.cuda_libraries_missing == ["cuBLAS"]
        assert info.device == "cpu"
