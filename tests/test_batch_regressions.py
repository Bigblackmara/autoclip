"""Regression tests for batch-pipeline bugs that failed, hung, or lost work.

Each test names the user-visible failure it guards against. None of them touch
the network, a GPU, a real API key, or the user's real AutoClip home.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoclip import paths
from autoclip.config import ExportSettings
from autoclip.db import store
from autoclip.db.models import Clip, Export, Job, Source, new_id
from autoclip.jobs.queue import JobQueue
from autoclip.pipeline import captions, export, ffmpeg, highlights, prepare
from autoclip.pipeline.reframe.croppath import CropPath, centre_crop
from autoclip.pipeline.runner import JobCancelled, PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig, ProviderError, TranscriptWindow
from autoclip.providers.anthropic_provider import AnthropicProvider
from autoclip.providers.base import extract_json_object
from autoclip.providers.gemini_provider import GeminiProvider
from autoclip.providers.openai_provider import OpenAIProvider

VALID = '{"clips":[{"start_word_index":10,"end_word_index":50,"title":"T","score":80}]}'


@pytest.fixture
def window() -> TranscriptWindow:
    return TranscriptWindow(text="[0]hello [1]world", first_word=0, last_word=100)


@pytest.fixture
def source(initialised_db) -> Source:
    return store.create_source(
        Source(
            id=new_id(),
            type="upload",
            path="C:/media/video.mp4",
            title="Test source",
            duration_s=600.0,
            width=1920,
            height=1080,
        )
    )


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class TestAnthropicRequestShape:
    """Current Claude models 400 on assistant prefill, and the API version the
    installed SDK sends 400s on ``temperature`` — so every window failed."""

    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        calls: list[dict] = []

        async def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking="..."),
                    SimpleNamespace(type="text", text="Here you go:\n" + VALID),
                ]
            )

        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        monkeypatch.setattr(AnthropicProvider, "_client", lambda self: client)
        return calls

    async def test_sends_no_prefill_and_no_sampling_parameters(
        self, calls: list[dict], window: TranscriptWindow
    ) -> None:
        provider = AnthropicProvider("claude-sonnet-5", api_key="test")

        result = await provider.detect_highlights(window, DetectionConfig())

        assert len(result.clips) == 1
        request = calls[0]
        assert request["messages"][-1]["role"] == "user"
        assert all(m["role"] != "assistant" for m in request["messages"])
        assert "temperature" not in request


class TestOpenAIRejectedParameters:
    """o4-mini (a suggested model) rejects any non-default temperature; the old
    retry only dropped response_format, so every request failed."""

    @staticmethod
    def fake_client(rejects: set[str], calls: list[dict]):
        async def create(**kwargs):
            calls.append(kwargs)
            if "temperature" in rejects and "temperature" in kwargs:
                raise RuntimeError(
                    "Error code: 400 - Unsupported value: 'temperature' does not support "
                    "0.3 with this model. Only the default (1) value is supported."
                )
            if "response_format" in rejects and "response_format" in kwargs:
                raise RuntimeError("Error code: 400 - response_format is not supported")
            message = SimpleNamespace(content=VALID)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    async def test_reasoning_model_retries_without_temperature(
        self, monkeypatch: pytest.MonkeyPatch, window: TranscriptWindow
    ) -> None:
        calls: list[dict] = []
        client = self.fake_client({"temperature"}, calls)
        monkeypatch.setattr(OpenAIProvider, "_client", lambda self: client)

        result = await OpenAIProvider("o4-mini", api_key="test").detect_highlights(
            window, DetectionConfig()
        )

        assert len(result.clips) == 1
        assert "temperature" not in calls[-1]
        # JSON mode survives: only the rejected parameter is dropped.
        assert "response_format" in calls[-1]

    async def test_vendor_rejecting_both_parameters_still_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, window: TranscriptWindow
    ) -> None:
        calls: list[dict] = []
        client = self.fake_client({"temperature", "response_format"}, calls)
        monkeypatch.setattr(OpenAIProvider, "_client", lambda self: client)

        result = await OpenAIProvider("x", api_key="test").detect_highlights(
            window, DetectionConfig()
        )

        assert len(result.clips) == 1
        assert len(calls) == 3

    async def test_unrelated_errors_are_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, window: TranscriptWindow
    ) -> None:
        calls: list[dict] = []

        async def create(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("Error code: 401 - invalid api key")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(OpenAIProvider, "_client", lambda self: client)
        # Not the real stored numes/Groq keys: those would answer instead.
        monkeypatch.setattr("autoclip.providers.openai_provider.fast_endpoints", lambda: [])

        with pytest.raises(ProviderError, match="rejected the API key"):
            await OpenAIProvider("x", api_key="test").detect_highlights(window, DetectionConfig())
        assert len(calls) == 1


class TestGeminiTimeout:
    def test_client_has_a_request_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """google-genai defaults to no timeout; a stalled call hung the queue."""
        from google import genai

        captured: dict = {}
        monkeypatch.setattr(genai, "Client", lambda **kwargs: captured.update(kwargs))

        GeminiProvider(api_key="test")._client()

        options = captured["http_options"]
        assert options.timeout and options.timeout > 0
        assert options.retry_options.attempts > 1


class TestJsonExtractionAfterBracedProse:
    def test_braces_in_leading_prose_do_not_hide_the_object(self) -> None:
        raw = "Each clip is {start, end}. Result:\n" + VALID

        assert len(extract_json_object(raw)["clips"]) == 1

    def test_truncated_reply_still_fails_instead_of_returning_an_inner_clip(self) -> None:
        # A first complete inner clip object must not pass as "no clips".
        raw = '{"clips":[{"start_word_index":1,"end_word_index":5},{"start_word_'

        with pytest.raises(ValueError):
            extract_json_object(raw)


# --------------------------------------------------------------------------
# Highlight detection
# --------------------------------------------------------------------------


def _transcript(seconds: int = 120) -> Transcript:
    words = [
        Word(text="word." if i % 10 == 9 else "word", start=i * 0.5, end=i * 0.5 + 0.4)
        for i in range(seconds * 2)
    ]
    return Transcript(words=words)


class _FailingProvider:
    name = "failing"

    async def detect_highlights(self, window, config):
        raise ProviderError("Anthropic rejected the API key.", provider="anthropic")


class TestEveryWindowFailing:
    async def test_the_provider_error_surfaces_not_no_clips_found(self) -> None:
        """An invalid key used to be reported as "No clips were found"."""
        with pytest.raises(ProviderError, match="rejected the API key"):
            await highlights.detect(
                _transcript(), _FailingProvider(), DetectionConfig(), job_id="job"
            )


# --------------------------------------------------------------------------
# Job settings, cancellation, retry
# --------------------------------------------------------------------------


class TestPerJobSettings:
    def test_runner_uses_the_jobs_overrides(self, source: Source) -> None:
        """The queue built runners from global settings, ignoring the job's."""
        job = Job(
            id=new_id(),
            source_id=source.id,
            settings={
                "clips": {"min_duration_s": 5.0, "max_duration_s": 30.0, "max_clips": 2},
                "export": {"ratio": "1:1", "caption_style": "boxed"},
                "whisper": {"model": "tiny", "diarization": False},
            },
        )

        runner = PipelineRunner(job, source)

        assert runner.settings.clips.max_clips == 2
        assert runner.settings.clips.max_duration_s == 30.0
        assert runner.settings.export.ratio == "1:1"
        assert runner.settings.export.caption_style == "boxed"
        assert runner.settings.whisper.model == "tiny"

    def test_a_job_without_a_snapshot_uses_current_settings(self) -> None:
        from autoclip.pipeline.runner import settings_for_job

        settings = settings_for_job(Job(id=new_id(), source_id="s"))

        assert settings.clips.max_clips == 10


class TestCancellation:
    def test_cancel_during_ffmpeg_marks_the_job_cancelled(
        self, source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ffmpeg.Cancelled / TranscriptionError used to mark the job "failed"."""
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        cancelled = {"flag": False}

        def extract(*args, **kwargs):
            cancelled["flag"] = True
            raise ffmpeg.Cancelled("Render cancelled.")

        monkeypatch.setattr(prepare, "extract_audio", extract)
        runner = PipelineRunner(job, source, is_cancelled=lambda: cancelled["flag"])

        import asyncio

        with pytest.raises(JobCancelled):
            asyncio.run(runner.run())
        assert store.get_job(job.id).status == "cancelled"

    def test_a_job_cancelled_while_queued_can_be_retried(self, source: Source) -> None:
        """The stale cancel flag cancelled the retried job the instant it started."""
        queue = JobQueue()
        job = store.create_job(Job(id=new_id(), source_id=source.id))

        assert queue.cancel(job.id)
        assert store.get_job(job.id).status == "cancelled"

        queue.requeued(job.id)

        assert not queue.is_cancelled(job.id)

    def test_retry_route_clears_the_cancel_flag(
        self, autoclip_home, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoclip import app as app_module
        from autoclip.app import create_app
        from autoclip.jobs.queue import queue
        from fastapi.testclient import TestClient

        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        with TestClient(create_app()) as client:
            src = store.create_source(
                Source(id=new_id(), type="upload", path="C:/m/v.mp4", duration_s=60.0)
            )
            job = store.create_job(Job(id=new_id(), source_id=src.id))
            assert client.post(f"/api/jobs/{job.id}/cancel").status_code == 200
            assert client.post(f"/api/jobs/{job.id}/retry").status_code == 200

            assert not queue.is_cancelled(job.id)


# --------------------------------------------------------------------------
# Export stage
# --------------------------------------------------------------------------


class TestExportStage:
    @pytest.fixture
    def runner(self, source: Source) -> PipelineRunner:
        from autoclip.config import Settings

        job = store.create_job(Job(id=new_id(), source_id=source.id))
        return PipelineRunner(job, source, settings=Settings())

    @pytest.fixture
    def rendered(self, monkeypatch: pytest.MonkeyPatch) -> list[export.ExportRequest]:
        rendered: list[export.ExportRequest] = []

        def fake_export(request, **kwargs):
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            request.destination.write_bytes(b"mp4")
            rendered.append(request)
            return request.destination

        monkeypatch.setattr(export, "export_clip", fake_export)
        return rendered

    def _clips(self, job_id: str, titles: list[str]) -> list[Clip]:
        clips = [
            Clip(
                id=new_id(),
                job_id=job_id,
                rank=rank,
                start_s=rank * 10.0,
                end_s=rank * 10.0 + 5.0,
                title=title,
            )
            for rank, title in enumerate(titles, start=1)
        ]
        store.replace_clips(job_id, clips)
        return clips

    def test_clips_with_the_same_title_get_separate_files(
        self, runner: PipelineRunner, rendered: list
    ) -> None:
        """The second clip overwrote the first one's file."""
        clips = self._clips(runner.job.id, ["Wait... what?!", "wait what", "🔥🔥", "🎉"])
        crops = {c.id: centre_crop(1920, 1080, 5.0) for c in clips}

        runner._stage_export(clips, _transcript(), crops)

        destinations = {r.destination for r in rendered}
        assert len(destinations) == 4
        assert all(d.exists() for d in destinations)

    def test_retry_skips_clips_already_exported(
        self, runner: PipelineRunner, rendered: list
    ) -> None:
        """A retry re-rendered every clip and doubled its export records."""
        clips = self._clips(runner.job.id, ["First", "Second"])
        done = paths.exports_dir() / runner.job.id / "01-first_9x16.mp4"
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_bytes(b"mp4")
        store.create_export(Export(id=new_id(), clip_id=clips[0].id, path=str(done)))
        store.update_clip(clips[0].id, status="exported")
        clips = store.list_clips(runner.job.id)
        crops = {c.id: centre_crop(1920, 1080, 5.0) for c in clips}

        runner._stage_export(clips, _transcript(), crops)

        assert [r.destination.name for r in rendered] == ["02-second_9x16.mp4"]
        assert len(store.list_exports(clips[0].id)) == 1

    def test_audio_only_source_is_flagged_to_the_renderer(
        self, runner: PipelineRunner, rendered: list
    ) -> None:
        runner.source.has_video = False
        clips = self._clips(runner.job.id, ["Podcast"])

        runner._stage_export(
            clips, _transcript(), {c.id: centre_crop(1080, 1920, 5.0) for c in clips}
        )

        assert rendered[0].has_video is False


@pytest.mark.slow
class TestAudioOnlyRender:
    def test_audio_only_source_renders_a_captioned_vertical_clip(self, tmp_path: Path) -> None:
        """The filtergraph read [0:v], which an audio file does not have."""
        audio = tmp_path / "podcast.m4a"
        subprocess.run(
            [
                ffmpeg.ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=6",
                "-c:a",
                "aac",
                str(audio),
            ],
            check=True,
            capture_output=True,
        )
        destination = tmp_path / "out.mp4"
        words = [Word(text=f"w{i}", start=1.0 + i * 0.4, end=1.3 + i * 0.4) for i in range(8)]
        request = export.ExportRequest(
            source=audio,
            destination=destination,
            start_s=1.0,
            end_s=4.0,
            crop_path=centre_crop(1080, 1920, 3.0),
            words=words,
            style=captions.get_style("bold_pop"),
            has_video=False,
        )

        export.export_clip(
            request,
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False),
        )

        info = ffmpeg.probe(destination)
        assert info.has_video and info.has_audio
        assert (info.width, info.height) == (1080, 1920)
        assert info.duration_s == pytest.approx(3.0, abs=0.35)


# --------------------------------------------------------------------------
# Prepare / ffmpeg
# --------------------------------------------------------------------------


class TestAtomicAudioExtraction:
    def test_a_failed_extraction_leaves_no_reusable_audio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truncated audio.wav was reused on retry and only part was clipped."""

        def failing_run(args, **kwargs):
            Path(args[-1]).write_bytes(b"RIFF partial")
            raise ffmpeg.FFmpegError("No space left on device", command=["ffmpeg"])

        monkeypatch.setattr(ffmpeg, "run", failing_run)
        destination = tmp_path / "audio.wav"

        with pytest.raises(ffmpeg.FFmpegError):
            prepare.extract_audio(tmp_path / "in.mp4", destination)

        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []

    def test_success_lands_at_the_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict = {}

        def ok_run(args, **kwargs):
            seen.update(kwargs)
            Path(args[-1]).write_bytes(b"RIFF whole")

        monkeypatch.setattr(ffmpeg, "run", ok_run)
        cancel = lambda: False  # noqa: E731

        prepare.extract_audio(tmp_path / "in.mp4", tmp_path / "audio.wav", cancelled=cancel)

        assert (tmp_path / "audio.wav").read_bytes() == b"RIFF whole"
        assert seen["cancelled"] is cancel


_NOISY_FFMPEG = r"""
import sys
# Far more stderr than any OS pipe buffer holds, as a damaged source produces.
for i in range(40000):
    sys.stderr.write("[h264 @ 0x0] error while decoding MB %d\n" % i)
sys.stderr.flush()
if "-progress" in sys.argv:
    sys.stdout.write("out_time_us=500000\nprogress=continue\n")
    sys.stdout.write("out_time_us=1000000\nprogress=end\n")
    sys.stdout.flush()
"""


@pytest.fixture
def noisy_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in a fake ffmpeg that floods stderr before finishing."""
    script = tmp_path / "noisy_ffmpeg.py"
    script.write_text(_NOISY_FFMPEG, encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "ffmpeg.bat"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = tmp_path / "ffmpeg"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)
    monkeypatch.setattr(ffmpeg, "ffmpeg_path", lambda: str(launcher))


def _finishes(target, timeout_s: float = 60.0) -> bool:
    errors: list[BaseException] = []

    def body() -> None:
        try:
            target()
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if errors:
        raise errors[0]
    return not worker.is_alive()


class TestStderrFlood:
    """Unread stderr filled the pipe and hung the job at a fixed percentage."""

    def test_progress_pump_does_not_deadlock(self, noisy_ffmpeg) -> None:
        progress: list[float] = []

        assert _finishes(
            lambda: ffmpeg.run(
                ["-i", "in.mp4", "out.mp4"], total_duration_s=1.0, on_progress=progress.append
            )
        )
        assert progress[-1] == 1.0

    def test_cancellable_wait_does_not_deadlock(self, noisy_ffmpeg) -> None:
        assert _finishes(lambda: ffmpeg.run(["-i", "in.mp4", "out.mp4"], cancelled=lambda: False))


# --------------------------------------------------------------------------
# Re-export API
# --------------------------------------------------------------------------


class TestReexportCropPath:
    """The cached crop path was reused regardless of ratio or a later trim."""

    @pytest.fixture
    def clip_and_cache(self, source: Source):
        from autoclip.pipeline.runner import JobWorkspace

        job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
        clip = Clip(id=new_id(), job_id=job.id, start_s=10.0, end_s=40.0, rank=1)
        workspace = JobWorkspace(job.id)
        return workspace, clip

    def test_matching_cache_is_reused(self, source: Source, clip_and_cache) -> None:
        from autoclip.api.clips import _crop_path_for

        workspace, clip = clip_and_cache
        cached = centre_crop(1920, 1080, 30.0)
        cached.segments[0].keyframes[0].x = 123.0  # distinguishable from a fresh centre
        cached.save(workspace.crop_path(clip.id))

        path = _crop_path_for(workspace, clip, source, "9:16")

        assert path.segments[0].keyframes[0].x == 123.0

    def test_cache_for_another_ratio_is_not_stretched(self, source: Source, clip_and_cache) -> None:
        from autoclip.api.clips import _crop_path_for

        workspace, clip = clip_and_cache
        # The job ran at 1:1, so the cache holds square windows.
        centre_crop(1920, 1080, 30.0, aspect_w=1, aspect_h=1).save(workspace.crop_path(clip.id))

        path = _crop_path_for(workspace, clip, source, "9:16")

        segment = path.segments[0]
        assert segment.width / segment.height == pytest.approx(9 / 16, rel=0.02)

    def test_cache_from_before_a_trim_is_not_reused(self, source: Source, clip_and_cache) -> None:
        from autoclip.api.clips import _crop_path_for

        workspace, clip = clip_and_cache
        centre_crop(1920, 1080, 30.0).save(workspace.crop_path(clip.id))
        clip.end_s = 30.0  # user trimmed 30 s down to 20 s

        path = _crop_path_for(workspace, clip, source, "9:16")

        assert isinstance(path, CropPath)
        assert path.duration_s == pytest.approx(20.0)
