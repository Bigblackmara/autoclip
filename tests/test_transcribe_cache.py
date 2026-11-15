"""The opt-in Whisper model cache and Live Watch's shared GPU resources.

Live Watch transcribes many short windows: loading a model per call costs
seconds and a fresh VRAM allocation, and a judge transcription racing a
render's transcription would load two models side by side. These tests pin
the cache semantics, its interaction with the CPU fallback, and that the
judge path honours the user's WhisperSettings.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoclip import config
from autoclip.config import WhisperSettings
from autoclip.pipeline import transcribe as transcribe_mod

CUBLAS_ERROR = "Library cublas64_12.dll is not found or cannot be loaded"


def _segments():
    words = [SimpleNamespace(word=" hi", start=0.0, end=0.4)]
    return [SimpleNamespace(text=" hi", start=0.0, end=0.4, words=words)]


def _model_class(created: list[tuple[str, str, str]], fail_on: dict[str, str] | None = None):
    fail_on = fail_on or {}

    class FakeWhisperModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            created.append((model, device, compute_type))
            self.device = device
            if fail_on.get(device) == "load":
                raise RuntimeError(CUBLAS_ERROR)

        def transcribe(self, audio: str, **_kwargs):
            if fail_on.get(self.device) == "encode":
                raise RuntimeError(CUBLAS_ERROR)
            return iter(_segments()), SimpleNamespace(duration=0.4, language="en")

    return FakeWhisperModel


@pytest.fixture
def fake_whisper(monkeypatch):
    import faster_whisper

    created: list[tuple[str, str, str]] = []

    def install(fail_on: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
        monkeypatch.setattr(faster_whisper, "WhisperModel", _model_class(created, fail_on))
        return created

    monkeypatch.setattr(transcribe_mod, "resolve_compute", lambda _s: ("cpu", "int8"))
    return install


class TestModelCache:
    def test_without_a_cache_every_call_loads(self, fake_whisper) -> None:
        created = fake_whisper()
        transcribe_mod.transcribe(Path("a.wav"), WhisperSettings())
        transcribe_mod.transcribe(Path("a.wav"), WhisperSettings())
        assert len(created) == 2  # the batch pipeline's behaviour is unchanged

    def test_a_cache_loads_once(self, fake_whisper) -> None:
        created = fake_whisper()
        cache = transcribe_mod.WhisperModelCache()
        for _ in range(3):
            transcribe_mod.transcribe(Path("a.wav"), WhisperSettings(), model_cache=cache)
        assert len(created) == 1
        assert len(cache) == 1

    def test_a_different_model_replaces_the_resident_one(self, fake_whisper) -> None:
        created = fake_whisper()
        cache = transcribe_mod.WhisperModelCache(max_models=1)
        transcribe_mod.transcribe(Path("a.wav"), WhisperSettings(model="small"), model_cache=cache)
        transcribe_mod.transcribe(Path("a.wav"), WhisperSettings(model="base"), model_cache=cache)
        transcribe_mod.transcribe(Path("a.wav"), WhisperSettings(model="small"), model_cache=cache)
        assert [c[0] for c in created] == ["small", "base", "small"]
        assert len(cache) == 1

    def test_gpu_runtime_failure_evicts_the_cuda_model(self, fake_whisper, monkeypatch) -> None:
        created = fake_whisper({"cuda": "encode"})
        monkeypatch.setattr(transcribe_mod, "resolve_compute", lambda _s: ("cuda", "float16"))
        monkeypatch.setattr(transcribe_mod, "_cpu_compute_type", lambda: "int8")
        cache = transcribe_mod.WhisperModelCache(max_models=2)

        result = transcribe_mod.transcribe(Path("a.wav"), WhisperSettings(), model_cache=cache)
        assert [w.text for w in result.words] == ["hi"]  # CPU fallback kept
        assert [c[1] for c in created] == ["cuda", "cpu"]
        assert len(cache) == 1  # only the working CPU model stays resident

        transcribe_mod.transcribe(Path("a.wav"), WhisperSettings(), model_cache=cache)
        assert [c[1] for c in created] == ["cuda", "cpu", "cuda"]  # GPU retried, not reused


class TestLiveSharedResources:
    def test_judge_transcription_uses_the_users_whisper_settings(
        self, fake_whisper, monkeypatch
    ) -> None:
        from autoclip.live import gpu

        created = fake_whisper()
        monkeypatch.setattr(gpu, "MODEL_CACHE", transcribe_mod.WhisperModelCache())
        settings = config.load()
        settings.whisper.model = "medium"
        settings.whisper.language = "de"
        config.save(settings)

        seen: list[WhisperSettings] = []
        real = transcribe_mod.transcribe

        def spy(audio, whisper_settings=None, **kwargs):
            seen.append(whisper_settings)
            return real(audio, whisper_settings, **kwargs)

        monkeypatch.setattr(transcribe_mod, "transcribe", spy)
        from autoclip.live.watcher import _default_transcribe

        _default_transcribe(Path("judge_audio.wav"))
        _default_transcribe(Path("judge_audio.wav"))

        assert [s.model for s in seen] == ["medium", "medium"]
        assert seen[0].language == "de"
        assert len(created) == 1  # one resident model across judge calls

    def test_gpu_slot_serializes_judge_and_render_work(self, monkeypatch) -> None:
        from autoclip.live import gpu

        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_stage(audio, settings, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return SimpleNamespace(words=[])

        threads = [
            threading.Thread(
                target=gpu.transcribe,
                args=(Path(f"{i}.wav"), WhisperSettings()),
                kwargs={"transcribe_fn": fake_stage},
            )
            for i in range(4)
        ]
        with gpu.gpu_slot():  # a render's encode holds the slot first
            for t in threads:
                t.start()
            time.sleep(0.1)
            assert active == 0  # nobody got past the held slot
        for t in threads:
            t.join(timeout=5)
        assert peak == 1
