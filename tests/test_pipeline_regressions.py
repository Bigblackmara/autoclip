"""Regression tests for the second round of batch-pipeline bugs.

Each test names the user-visible failure it guards against. None of them touch
the network, a GPU, a real API key, or the user's real AutoClip home.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pysubs2
import pytest
from autoclip.config import ExportSettings
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline import captions, export, ffmpeg, highlights
from autoclip.pipeline import runner as runner_module
from autoclip.pipeline.reframe.croppath import centre_crop
from autoclip.pipeline.runner import JobCancelled, PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import (
    DetectionConfig,
    LLMProvider,
    ProviderError,
    ProviderStatus,
    TranscriptWindow,
    ollama_provider,
)
from autoclip.providers.gemini_provider import GeminiProvider
from autoclip.providers.ollama_provider import OllamaProvider

BS = "\\"


def _words(*texts: str) -> list[Word]:
    return [Word(text=t, start=i * 0.5, end=i * 0.5 + 0.4) for i, t in enumerate(texts)]


def _transcript(seconds: int = 120) -> Transcript:
    return Transcript(
        words=[Word(text="word", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(seconds * 2)]
    )


# --------------------------------------------------------------------------
# Captions: transcript text must render literally
# --------------------------------------------------------------------------


def _our_tags_removed(text: str) -> str:
    """Event text minus the override blocks AutoClip itself emits."""
    for tag in pysubs2.SSAEvent.OVERRIDE_SEQUENCE.findall(text):
        if tag.startswith(("{\\c", "{\\kf", "{\\r")):
            text = text.replace(tag, "", 1)
    return text


def _libass_unsafe(text: str) -> list[str]:
    """Sequences libass would interpret in what should be literal text."""
    found: list[str] = []
    for i, char in enumerate(text):
        if char == "{" and (i == 0 or text[i - 1] != BS):
            found.append("unescaped {")
        if char == BS and i + 1 < len(text) and text[i + 1] in "nNh":
            found.append(BS + text[i + 1])
    return found


class TestCaptionEscaping:
    """Braces were parsed as override blocks and \\N / \\n / \\h as breaks."""

    @pytest.mark.parametrize("style_key", list(captions.PRESETS))
    def test_markup_in_words_is_literal_in_every_style(self, style_key: str) -> None:
        words = _words("{laughs}", "a" + BS + "Nb", "c" + BS + "nd", "e" + BS + "hf", "end}")
        subs = captions.build_ass(words, captions.get_style(style_key), width=1080, height=1920)

        for event in subs.events:
            literal = _our_tags_removed(event.text)
            assert _libass_unsafe(literal) == [], event.text

    def test_trailing_backslash_does_not_escape_the_highlight_tag(self) -> None:
        """``word\\`` + ``{\\c...}`` became ``\\{`` — the tag was drawn as text."""
        subs = captions.build_ass(
            _words("back" + BS, "next"),
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
        )

        # Event 0 highlights "BACK\": its closing {\r} directly follows the
        # backslash, and "\{" would make libass draw "{\r}" as text.
        first = subs.events[0].text
        assert first.startswith("{" + BS + "c")
        assert BS + "{" + BS + "r}" not in first
        assert "BACK" + BS + "\u2060{" + BS + "r}" in first

    def test_all_caps_does_not_turn_a_soft_break_into_a_hard_one(self) -> None:
        text = captions.escape_ass_text(("a" + BS + "nb").upper())
        assert BS + "N" not in text

    def test_real_newlines_cannot_split_the_dialogue_line(self, tmp_path: Path) -> None:
        path = captions.write_ass(
            tmp_path / "c.ass",
            _words("one\ntwo", "three\r\nfour"),
            captions.get_style("clean_lower"),
            width=1080,
            height=1920,
        )
        reloaded = pysubs2.load(str(path))

        assert [e.text for e in reloaded.events] == ["one two three four"]

    def test_srt_sidecar_keeps_braces_and_backslashes(self, tmp_path: Path) -> None:
        path = captions.write_srt(tmp_path / "c.srt", _words("{laughs}", "a" + BS + "Nb"))
        text = path.read_text(encoding="utf-8").replace("\u2060", "")

        assert "{laughs} a" + BS + "Nb" in text

    def test_libass_draws_escaped_text_literally(self, tmp_path: Path) -> None:
        r"""Checks the escapes against the real renderer the export uses.

        An unclosed ``{`` is drawn literally by libass, so it is the reference
        for what a brace looks like; ``\{`` must draw the same pixels, and an
        escaped ``\N`` must stay on one line.
        """
        shutil.copytree(captions.FONT_DIR, tmp_path / "fonts")

        def render(text: str) -> bytes:
            ass = tmp_path / "t.ass"
            subs = pysubs2.SSAFile()
            subs.info["PlayResX"], subs.info["PlayResY"] = "640", "240"
            style = pysubs2.SSAStyle(fontname="Anton", fontsize=60, outline=0, shadow=0)
            style.alignment = pysubs2.Alignment.MIDDLE_CENTER
            subs.styles["Default"] = style
            subs.events.append(pysubs2.SSAEvent(start=0, end=5000, text=text))
            subs.save(str(ass), encoding="utf-8")
            done = subprocess.run(
                [
                    ffmpeg.ffmpeg_path(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=640x240:d=1",
                    "-vf",
                    "ass=filename=t.ass:fontsdir=fonts",
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "-",
                ],
                capture_output=True,
                check=True,
                cwd=tmp_path,
            )
            return done.stdout

        def lit_rows(frame: bytes) -> int:
            return sum(1 for y in range(240) if max(frame[y * 640 : (y + 1) * 640]) > 128)

        assert render(captions.escape_ass_text("AB{CD")) == render("AB{CD")
        assert render(captions.escape_ass_text("AB}CD")) == render("AB}CD")
        one_line = lit_rows(render("ABCD"))
        assert lit_rows(render(captions.escape_ass_text("AB" + BS + "NCD"))) <= one_line + 2
        assert lit_rows(render("AB" + BS + "NCD")) > one_line * 1.5  # the bug, for contrast


# --------------------------------------------------------------------------
# Export: no partial MP4 in the exports folder
# --------------------------------------------------------------------------


def _request(tmp_path: Path) -> export.ExportRequest:
    return export.ExportRequest(
        source=tmp_path / "src.mp4",
        destination=tmp_path / "exports" / "clip.mp4",
        start_s=0.0,
        end_s=4.0,
        crop_path=centre_crop(1280, 720, 4.0),
        words=[],
        style=captions.get_style("bold_pop"),
    )


@pytest.fixture
def fake_encoders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        export,
        "encoder_args",
        lambda s: ["-c:v", "h264_nvenc" if s.prefer_hardware_encoder else "libx264"],
    )


def _scripted_run(outcomes: dict[str, BaseException | bytes], outputs: list[Path]):
    """ffmpeg stand-in: writes a partial file, then fails or finishes per encoder."""

    def run(args, **_kwargs):
        out = Path(args[-1])
        outputs.append(out)
        encoder = args[args.index("-c:v") + 1]
        outcome = outcomes[encoder]
        out.write_bytes(b"half-written moov-less mp4")
        if isinstance(outcome, BaseException):
            raise outcome
        out.write_bytes(outcome)

    return run


class TestAtomicExport:
    """A failed or cancelled render left a truncated MP4 in the exports folder."""

    @pytest.mark.parametrize(
        "failure",
        [
            ffmpeg.FFmpegError("ffmpeg exited with code 1.", command=["ffmpeg"]),
            ffmpeg.Cancelled("Render cancelled."),
        ],
    )
    def test_failure_leaves_no_file_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_encoders, failure
    ) -> None:
        outputs: list[Path] = []
        monkeypatch.setattr(export.ffmpeg, "run", _scripted_run({"libx264": failure}, outputs))
        request = _request(tmp_path)

        with pytest.raises((export.ExportError, ffmpeg.Cancelled)):
            export.export_clip(
                request,
                work_dir=tmp_path / "work",
                settings=ExportSettings(prefer_hardware_encoder=False),
            )

        assert outputs and outputs[0] != request.destination
        assert outputs[0].parent == request.destination.parent
        assert list(request.destination.parent.iterdir()) == []

    def test_failed_reexport_keeps_the_previous_good_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_encoders
    ) -> None:
        request = _request(tmp_path)
        request.destination.parent.mkdir(parents=True)
        request.destination.write_bytes(b"the good render")
        error = ffmpeg.FFmpegError("boom", command=["ffmpeg"])
        monkeypatch.setattr(export.ffmpeg, "run", _scripted_run({"libx264": error}, []))

        with pytest.raises(export.ExportError):
            export.export_clip(
                request,
                work_dir=tmp_path / "work",
                settings=ExportSettings(prefer_hardware_encoder=False),
            )

        assert request.destination.read_bytes() == b"the good render"
        assert [p.name for p in request.destination.parent.iterdir()] == ["clip.mp4"]

    def test_gpu_failure_then_cpu_success_lands_only_the_final_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_encoders
    ) -> None:
        outcomes = {
            "h264_nvenc": ffmpeg.FFmpegError("out of memory", command=["ffmpeg"]),
            "libx264": b"cpu render",
        }
        monkeypatch.setattr(export.ffmpeg, "run", _scripted_run(outcomes, []))
        request = _request(tmp_path)

        out = export.export_clip(
            request,
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=True),
        )

        assert out == request.destination
        assert out.read_bytes() == b"cpu render"
        assert [p.name for p in out.parent.iterdir()] == ["clip.mp4"]

    def test_empty_output_is_an_error_and_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_encoders
    ) -> None:
        monkeypatch.setattr(export.ffmpeg, "run", _scripted_run({"libx264": b""}, []))
        request = _request(tmp_path)

        with pytest.raises(export.ExportError, match="is empty"):
            export.export_clip(
                request,
                work_dir=tmp_path / "work",
                settings=ExportSettings(prefer_hardware_encoder=False),
            )
        assert list(request.destination.parent.iterdir()) == []


# --------------------------------------------------------------------------
# Provider validation: one bad item must not sink the window
# --------------------------------------------------------------------------


class _Scripted(LLMProvider):
    name = "scripted"
    requires_key = False

    def __init__(self, responses: list[str]) -> None:
        super().__init__("test-model")
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.prompts.append(user)
        return self.responses.pop(0)

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


@pytest.fixture
def window() -> TranscriptWindow:
    return TranscriptWindow(text="[0]hello [1]world", first_word=0, last_word=100)


GOOD = {"start_word_index": 10, "end_word_index": 50, "title": "Good", "score": 80}


class TestPartialValidation:
    async def test_bad_items_are_dropped_and_good_ones_kept(
        self, window: TranscriptWindow, caplog: pytest.LogCaptureFixture
    ) -> None:
        reply = {
            "clips": [
                GOOD,
                {"start_word_index": -5, "end_word_index": 20},
                {"end_word_index": 30, "title": "no start"},
                "not an object",
                {**GOOD, "start_word_index": 60, "end_word_index": 90, "title": "Also good"},
            ]
        }
        provider = _Scripted([json.dumps(reply)])

        result = await provider.detect_highlights(window, DetectionConfig())

        assert [c.title for c in result.clips] == ["Good", "Also good"]
        assert len(provider.prompts) == 1  # no retry needed
        assert "dropped 3 malformed clip item(s) of 5" in caplog.text

    async def test_all_items_bad_still_retries_with_the_errors(
        self, window: TranscriptWindow
    ) -> None:
        bad = json.dumps({"clips": [{"start_word_index": "soon", "end_word_index": 3}]})
        provider = _Scripted([bad, json.dumps({"clips": [GOOD]})])

        result = await provider.detect_highlights(window, DetectionConfig())

        assert len(result.clips) == 1
        assert len(provider.prompts) == 2
        assert "clips[0]" in provider.prompts[1]
        assert "start_word_index" in provider.prompts[1]

    async def test_all_items_bad_twice_raises(self, window: TranscriptWindow) -> None:
        bad = json.dumps({"clips": [{"title": "no indices"}]})
        provider = _Scripted([bad, bad])

        with pytest.raises(ProviderError, match="malformed clip data twice"):
            await provider.detect_highlights(window, DetectionConfig())

    async def test_clips_that_is_not_a_list_retries(self, window: TranscriptWindow) -> None:
        provider = _Scripted(['{"clips": {"oops": 1}}', json.dumps({"clips": [GOOD]})])

        result = await provider.detect_highlights(window, DetectionConfig())

        assert len(result.clips) == 1
        assert len(provider.prompts) == 2


# --------------------------------------------------------------------------
# Ollama: the prompt must fit the context window
# --------------------------------------------------------------------------


class TestOllamaContext:
    """Ollama's small default context silently cut the front of the prompt."""

    def test_context_is_sized_to_the_prompt_and_clamped(self) -> None:
        assert ollama_provider.context_tokens("s", "u") == 8192
        # 40 000 chars ~ 11 400 tokens + reply room -> next power of two.
        assert ollama_provider.context_tokens("x" * 10_000, "y" * 30_000) == 16384
        assert ollama_provider.context_tokens("", "z" * 1_000_000) == 32768

    async def test_request_carries_num_ctx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"response": '{"clips": []}'})

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            ollama_provider.httpx,
            "AsyncClient",
            lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        )
        user = "[1]word " * 4000  # ~32 000 chars: well past a 4096-token default

        raw = await OllamaProvider("llama3.1:8b")._complete("system", user, DetectionConfig())

        assert raw == '{"clips": []}'
        options = sent[0]["options"]
        assert options["num_ctx"] == 16384
        assert options["num_ctx"] * 3.5 > len(user)
        assert options["temperature"] == DetectionConfig().temperature


# --------------------------------------------------------------------------
# Cancellation during highlight detection
# --------------------------------------------------------------------------


class _HangingProvider(LLMProvider):
    """A provider whose request never returns, like a stalled hosted call."""

    name = "hanging"
    requires_key = False

    def __init__(self) -> None:
        super().__init__("test-model")
        self.started = 0
        self.cancelled = 0

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.started += 1
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return '{"clips": []}'

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


def _flag_after(seconds: float):
    deadline = time.monotonic() + seconds
    return lambda: time.monotonic() >= deadline


class TestCancelDuringDetection:
    """Cancel waited out every in-flight LLM call before the job stopped."""

    async def test_in_flight_windows_are_cancelled_promptly(self) -> None:
        provider = _HangingProvider()
        started = time.monotonic()

        with pytest.raises(highlights.DetectionCancelled):
            await highlights.detect(
                _transcript(seconds=30 * 60),  # several windows
                provider,
                DetectionConfig(),
                job_id="job",
                cancelled=_flag_after(0.2),
            )

        assert time.monotonic() - started < 5
        assert provider.started >= 1
        assert provider.cancelled == provider.started

    async def test_without_a_cancel_flag_detection_completes(self) -> None:
        provider = _Scripted(['{"clips": []}'])

        with pytest.raises(highlights.HighlightError, match="No clips were found"):
            await highlights.detect(
                _transcript(), provider, DetectionConfig(), job_id="job", cancelled=lambda: False
            )

    def test_runner_marks_the_job_cancelled(
        self, initialised_db, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = store.create_source(
            Source(id=new_id(), type="upload", path="C:/media/v.mp4", title="t", duration_s=600)
        )
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        monkeypatch.setattr(runner_module, "build_provider", lambda *a, **k: _HangingProvider())
        runner = PipelineRunner(job, source, is_cancelled=_flag_after(0.2))
        runner._stage_prepare = lambda: Path("audio.wav")
        runner._stage_transcribe = lambda audio: _transcript()
        runner._load_or_detect_silences = lambda audio: []
        started = time.monotonic()

        with pytest.raises(JobCancelled):
            asyncio.run(runner.run())

        assert time.monotonic() - started < 5
        assert store.get_job(job.id).status == "cancelled"


class TestGeminiCancelIsNotJoined:
    """asyncio.run joined the default executor, so a cancelled Gemini call
    still held the job until the blocking request finished."""

    def test_cancelled_call_does_not_block_loop_shutdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()

        def generate_content(**_kwargs):
            release.wait(10)
            return SimpleNamespace(text='{"clips": []}')

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        monkeypatch.setattr(GeminiProvider, "_client", lambda self: client)
        provider = GeminiProvider("gemini-2.0-flash", api_key="test")

        async def cancel_soon() -> None:
            task = asyncio.create_task(provider._complete("s", "u", DetectionConfig()))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        started = time.monotonic()
        try:
            asyncio.run(cancel_soon())
            assert time.monotonic() - started < 5
        finally:
            release.set()
