"""Live clip production — orchestration of reused pipeline stages.

produce() is deliberately thin glue: its whole value is the *order* of the
reused pipeline stages, the pin/unpin discipline around the buffer extraction,
and the error contract every stage failure funnels into. Those are exactly what
these tests pin down — with the heavy stages stubbed, because what matters here
is the choreography, not whether Whisper transcribes.

One test runs the remux against a real ffmpeg binary (synthetic testsrc+sine
MPEG-TS segments, ``-c copy`` container remux only — no x264, so it stays in
the fast suite): the remux is the one genuinely live-specific ffmpeg
invocation, and it is where concatenated ``-reset_timestamps`` segment quirks
would surface if the flags were wrong.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from autoclip import paths
from autoclip.config import Settings
from autoclip.live import clip_factory
from autoclip.live.buffer import SEGMENT_S
from autoclip.live.models import LiveClipRow
from autoclip.pipeline import export as export_mod
from autoclip.pipeline import ffmpeg as ffmpeg_mod
from autoclip.pipeline.reframe.croppath import CropPath
from autoclip.pipeline.transcript import Transcript
from autoclip.pipeline.transcript import Word as TWord

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeBuffer:
    """A RingBuffer stand-in: extractions are marker bytes on disk.

    Implements just enough of the frozen contract (extract / pin_oldest /
    unpin) for produce() to be exercised end to end without A1's bodies.
    """

    def __init__(self, fail_extract: bool = False) -> None:
        self.fail_extract = fail_extract
        self.extract_calls: list[tuple[float, float, Path]] = []
        self.pins: list[int] = []
        self.unpins: list[int | None] = []

    def extract(self, start_s: float, end_s: float, target: Path) -> Path:
        self.extract_calls.append((start_s, end_s, target))
        if self.fail_extract:
            raise ValueError("epoch mixing refused")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"FAKE_TS_" + b"x" * 64)
        return target

    def pin_oldest(self, index: int) -> None:
        self.pins.append(index)

    def unpin(self, index: int | None = None) -> None:
        self.unpins.append(index)


@dataclass
class _MediaInfo:
    duration_s: float = 30.0
    has_video: bool = True
    has_audio: bool = True


def _fake_probe(path: Path) -> _MediaInfo:
    """Stand-in for ffprobe in stubbed runs: probe of any written file succeeds."""
    return _MediaInfo()


def make_row(clip_id: str = "clip1", **overrides: Any) -> LiveClipRow:
    defaults: dict[str, Any] = {
        "id": clip_id,
        "stream_id": "stream1",
        "trigger_start_s": 0.0,
        "trigger_end_s": 10.0,
        "clip_start_s": 0.0,
        "clip_end_s": 30.0,
        "title": "Best Moment Ever",
        "score": 82,
        "caption_style": "bold_pop",
        "ratio": "9:16",
    }
    defaults.update(overrides)
    return LiveClipRow(**defaults)


def make_transcript() -> Transcript:
    return Transcript(
        words=[
            TWord(text="hello", start=0.2, end=0.6),
            TWord(text="world", start=0.8, end=1.2),
        ]
    )


def make_crop() -> CropPath:
    return CropPath(source_width=1280, source_height=720, segments=[])


@dataclass
class StageSpy:
    """Records every pipeline stage invocation produce() makes."""

    extract_audio_calls: list[tuple[Path, Path]] = field(default_factory=list)
    transcribe_calls: list[Path] = field(default_factory=list)
    reframe_calls: list[dict[str, Any]] = field(default_factory=list)
    export_calls: list[export_mod.ExportRequest] = field(default_factory=list)
    export_settings: list[Any] = field(default_factory=list)
    remux_calls: list[tuple[Path, Path]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    transcript: Transcript = field(default_factory=make_transcript)

    def install(self, monkeypatch: pytest.MonkeyPatch, *, real_probe: bool = False) -> StageSpy:
        """Stub every stage. ``real_probe=True`` leaves ffprobe live (real-remux test)."""

        def fake_extract_audio(source, destination, *, duration_s=None, on_progress=None):
            self.order.append("audio")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"RIFF")
            self.extract_audio_calls.append((Path(source), Path(destination)))
            return destination

        def fake_transcribe(audio, settings, *, duration_s=None, on_progress=None, **kwargs):
            self.order.append("transcribe")
            if on_progress:
                on_progress(0.5)
                on_progress(1.0)
            self.transcribe_calls.append(Path(audio))
            return self.transcript

        def fake_build_crop_path(video, *, start_s, end_s, transcript=None, config=None):
            self.order.append("reframe")
            self.reframe_calls.append(
                {
                    "start_s": start_s,
                    "end_s": end_s,
                    "aspect": (config.aspect_w, config.aspect_h),
                }
            )
            return make_crop()

        def fake_export_clip(request, *, work_dir, settings=None, on_progress=None, **kwargs):
            self.order.append("export")
            self.export_calls.append(request)
            self.export_settings.append(settings)
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            request.destination.write_bytes(b"FAKE_MP4")
            request.destination.with_suffix(".srt").write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nhello world\n", encoding="utf-8"
            )
            return request.destination

        def fake_remux(raw: Path, destination: Path, clip_id: str) -> None:
            self.order.append("remux")
            self.remux_calls.append((Path(raw), Path(destination)))
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"FAKE_REMUXED_MP4")

        monkeypatch.setattr(clip_factory.prepare, "extract_audio", fake_extract_audio)
        monkeypatch.setattr(clip_factory.transcribe, "transcribe", fake_transcribe)
        monkeypatch.setattr(clip_factory, "build_crop_path", fake_build_crop_path)
        monkeypatch.setattr(clip_factory.export, "export_clip", fake_export_clip)
        monkeypatch.setattr(clip_factory, "_remux", fake_remux)
        monkeypatch.setattr(clip_factory, "_probe_reusable", lambda path: None)
        if not real_probe:
            monkeypatch.setattr(clip_factory.ffmpeg, "probe", _fake_probe)
        return self


def run_produce(
    monkeypatch: pytest.MonkeyPatch,
    row: LiveClipRow | None = None,
    *,
    buffer: FakeBuffer | None = None,
    on_progress=None,
) -> tuple[clip_factory.ProduceResult, StageSpy, FakeBuffer]:
    spy = StageSpy().install(monkeypatch)
    buf = buffer or FakeBuffer()
    result = clip_factory.produce(buf, row or make_row(), Settings(), on_progress)
    return result, spy, buf


def install_failure(monkeypatch: pytest.MonkeyPatch, target: str, exc: Exception) -> StageSpy:
    """Install stubs with exactly one stage raising, so failure tags can be asserted."""
    spy = StageSpy().install(monkeypatch)

    def boom(*args, **kwargs):
        raise exc

    if target == "extract_audio":
        monkeypatch.setattr(clip_factory.prepare, "extract_audio", boom)
    elif target == "transcribe":
        monkeypatch.setattr(clip_factory.transcribe, "transcribe", boom)
    elif target == "build_crop_path":
        monkeypatch.setattr(clip_factory, "build_crop_path", boom)
    elif target == "export_clip":
        monkeypatch.setattr(clip_factory.export, "export_clip", boom)
    elif target == "_remux":
        monkeypatch.setattr(clip_factory, "_remux", boom)
    else:
        raise AssertionError(f"unknown failure target {target}")
    return spy


# --------------------------------------------------------------------------
# Contract: signature and result shape
# --------------------------------------------------------------------------


class TestContract:
    def test_produce_signature_matches_manager_expectation(self) -> None:
        """produce(buffer, clip_row, settings, on_progress=None) -> ProduceResult — frozen shape."""
        signature = inspect.signature(clip_factory.produce)
        assert list(signature.parameters) == ["buffer", "clip_row", "settings", "on_progress"]
        assert signature.parameters["on_progress"].default is None

    def test_result_carries_paths_and_title(self, monkeypatch) -> None:
        result, _, _ = run_produce(monkeypatch)
        work = paths.root() / "live" / "clips" / "clip1"

        assert isinstance(result, clip_factory.ProduceResult)
        assert result.rendered_path.parent == work
        assert result.srt_path.parent == work
        assert result.transcript_path.parent == work
        assert result.transcript_path.name == "transcript.json"
        assert result.title == "Best Moment Ever"

    def test_stage_weights_sum_to_one(self) -> None:
        assert pytest.approx(sum(clip_factory.STAGE_WEIGHTS.values())) == 1.0

    def test_stage_order_matches_weights_keys(self) -> None:
        assert tuple(clip_factory.STAGE_WEIGHTS) == clip_factory.STAGE_ORDER


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


class TestOrchestration:
    def test_stages_run_in_order(self, monkeypatch) -> None:
        _, spy, _ = run_produce(monkeypatch)

        assert spy.order == ["remux", "audio", "transcribe", "reframe", "export"]

    def test_window_is_extracted_with_pin_discipline(self, monkeypatch) -> None:
        row = make_row(clip_start_s=12.0, clip_end_s=42.0)
        _, spy, buf = run_produce(monkeypatch, row)

        assert len(buf.extract_calls) == 1
        start_s, end_s, target = buf.extract_calls[0]
        assert start_s == 12.0
        assert end_s == 42.0
        assert target == paths.root() / "live" / "clips" / "clip1" / "raw.ts"

        # The pin covers the first segment of the window and is released after.
        assert buf.pins == [int(12.0 // SEGMENT_S)]
        assert buf.unpins == [int(12.0 // SEGMENT_S)]

    def test_remux_reads_the_raw_extract_writes_normalized_mp4(self, monkeypatch) -> None:
        _, spy, _ = run_produce(monkeypatch)
        work = paths.root() / "live" / "clips" / "clip1"

        assert spy.remux_calls == [(work / "raw.ts", work / "normalized.mp4")]

    def test_extract_audio_and_transcribe_receive_the_right_files(self, monkeypatch) -> None:
        _, spy, _ = run_produce(monkeypatch)
        work = paths.root() / "live" / "clips" / "clip1"

        assert spy.extract_audio_calls == [(work / "normalized.mp4", work / "audio.wav")]
        assert spy.transcribe_calls == [work / "audio.wav"]

    def test_reframe_uses_9_16_when_the_row_says_9_16(self, monkeypatch) -> None:
        _, spy, _ = run_produce(monkeypatch)

        assert spy.reframe_calls[0]["aspect"] == (9, 16)

    def test_reframe_aspect_follows_the_row_ratio(self, monkeypatch) -> None:
        row = make_row(ratio="1:1")
        _, spy, _ = run_produce(monkeypatch, row)

        assert spy.reframe_calls[0]["aspect"] == (1, 1)

    def test_export_request_is_built_from_row_and_settings(self, monkeypatch) -> None:
        row = make_row()
        _, spy, _ = run_produce(monkeypatch, row)
        request = spy.export_calls[0]
        work = paths.root() / "live" / "clips" / "clip1"

        assert request.source == work / "normalized.mp4"
        assert request.destination.parent == work
        assert request.destination.name == "best-moment-ever_9x16.mp4"
        assert request.ratio == "9:16"
        assert request.style is clip_factory.captions.get_style("bold_pop")
        # Clip-relative timeline: the extraction already starts at clip_start_s.
        assert request.start_s == 0.0
        assert request.end_s == request.duration_s > 0
        assert [w.text for w in request.words] == ["hello", "world"]
        assert request.crop_path == make_crop()

    def test_srt_sidecar_is_forced_for_live_clips(self, monkeypatch) -> None:
        """Live clips always carry the .srt the gallery and uploads read."""
        _, spy, _ = run_produce(monkeypatch)

        assert spy.export_settings[0].write_srt is True
        # The user's global export settings object must not be mutated.
        assert Settings().export.write_srt is False

    def test_rendered_clip_name_uses_title_slug_and_ratio(self, monkeypatch) -> None:
        result, _, _ = run_produce(monkeypatch)

        assert result.rendered_path.name == "best-moment-ever_9x16.mp4"
        assert result.srt_path.name == "best-moment-ever_9x16.srt"

    def test_empty_title_falls_back_to_clip_id(self, monkeypatch) -> None:
        row = make_row(title="")
        result, _, _ = run_produce(monkeypatch, row)

        assert result.rendered_path.name.startswith("clip-clip1")
        assert result.title == ""

    def test_transcript_is_persisted_into_the_clip_directory(self, monkeypatch) -> None:
        result, _, _ = run_produce(monkeypatch)

        assert (
            json.loads(result.transcript_path.read_text(encoding="utf-8"))["words"][0]["text"]
            == "hello"
        )


# --------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------


class TestProgress:
    def test_progress_starts_at_zero_and_ends_at_one(self, monkeypatch) -> None:
        events: list[float] = []
        run_produce(monkeypatch, on_progress=lambda stage, fraction: events.append(fraction))

        assert events[0] == 0.0
        assert events[-1] == 1.0
        assert all(0.0 <= value <= 1.0 for value in events)

    def test_progress_is_monotonic_across_stage_boundaries(self, monkeypatch) -> None:
        events: list[float] = []
        run_produce(monkeypatch, on_progress=lambda stage, fraction: events.append(fraction))

        for before, after in zip(events, events[1:], strict=False):
            assert after >= before - 1e-9

    def test_all_stages_announce_themselves(self, monkeypatch) -> None:
        stages: list[str] = []
        run_produce(monkeypatch, on_progress=lambda stage, _: stages.append(stage))

        for stage in clip_factory.STAGE_ORDER:
            assert stage in stages

    def test_nested_stage_progress_is_offset_by_completed_stages(self, monkeypatch) -> None:
        """Whisper at 50% is less than reframe at 0% — overall never moves backwards."""
        events: list[tuple[str, float]] = []
        run_produce(monkeypatch, on_progress=lambda s, f: events.append((s, f)))

        half_transcribe = (
            clip_factory.STAGE_WEIGHTS["extract"]
            + clip_factory.STAGE_WEIGHTS["remux"]
            + clip_factory.STAGE_WEIGHTS["audio"]
            + 0.5 * clip_factory.STAGE_WEIGHTS["transcribe"]
        )
        assert any(
            stage == "transcribe" and pytest.approx(fraction) == half_transcribe
            for stage, fraction in events
        )


# --------------------------------------------------------------------------
# Resume (idempotence)
# --------------------------------------------------------------------------


class TestResume:
    def test_rerun_skips_extract_for_the_same_window(self, monkeypatch) -> None:
        row = make_row()
        run_produce(monkeypatch, row)

        buffer = FakeBuffer()
        StageSpy().install(monkeypatch)
        clip_factory.produce(buffer, row, Settings())

        assert buffer.extract_calls == []
        assert buffer.pins == []

    def test_rerun_reuses_remux_transcript_and_crop(self, monkeypatch) -> None:
        row = make_row()
        run_produce(monkeypatch, row)

        # Second run: normalized.mp4 probes clean, transcript and crop files
        # exist — only the export re-runs.
        spy = StageSpy().install(monkeypatch)
        monkeypatch.setattr(clip_factory, "_probe_reusable", lambda path: _MediaInfo())
        clip_factory.produce(FakeBuffer(), row, Settings())

        assert spy.remux_calls == []
        assert spy.transcribe_calls == []
        assert spy.reframe_calls == []
        assert len(spy.export_calls) == 1

    def test_corrupt_transcript_cache_is_rebuilt(self, monkeypatch) -> None:
        row = make_row()
        run_produce(monkeypatch, row)
        work = paths.root() / "live" / "clips" / "clip1"
        (work / "transcript.json").write_text("{ not json", encoding="utf-8")

        spy = StageSpy().install(monkeypatch)
        monkeypatch.setattr(clip_factory, "_probe_reusable", lambda path: _MediaInfo())
        clip_factory.produce(FakeBuffer(), row, Settings())

        assert len(spy.transcribe_calls) == 1

    def test_revised_window_reextracts_and_rewrites_the_manifest(self, monkeypatch) -> None:
        row = make_row(clip_start_s=10.0, clip_end_s=40.0)
        run_produce(monkeypatch, row)

        revised = make_row(clip_start_s=12.0, clip_end_s=44.0)
        buffer = FakeBuffer()
        spy = StageSpy().install(monkeypatch)
        clip_factory.produce(buffer, revised, Settings())

        assert len(buffer.extract_calls) == 1
        assert buffer.extract_calls[0][0] == 12.0
        assert buffer.extract_calls[0][1] == 44.0
        # Stale artifacts from the old window are wiped, not mixed in.
        work = paths.root() / "live" / "clips" / "clip1"
        assert json.loads((work / "window.json").read_text(encoding="utf-8")) == {
            "clip_start_s": 12.0,
            "clip_end_s": 44.0,
        }
        assert len(spy.export_calls) == 1

    def test_garbage_manifest_triggers_fresh_extraction(self, monkeypatch) -> None:
        row = make_row()
        run_produce(monkeypatch, row)
        work = paths.root() / "live" / "clips" / "clip1"
        (work / "window.json").write_text('"just a string"', encoding="utf-8")

        buffer = FakeBuffer()
        StageSpy().install(monkeypatch)
        clip_factory.produce(buffer, row, Settings())

        assert len(buffer.extract_calls) == 1


# --------------------------------------------------------------------------
# Failures → ClipProductionError with stage context
# --------------------------------------------------------------------------


class TestFailures:
    def test_extract_failure_is_tagged(self, monkeypatch) -> None:
        buffer = FakeBuffer(fail_extract=True)

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(buffer, make_row(), Settings())

        assert excinfo.value.stage == "extract"
        assert "clip1" in str(excinfo.value)

    def test_extract_writing_an_empty_file_fails(self, monkeypatch) -> None:
        class EmptyExtractBuffer(FakeBuffer):
            def extract(self, start_s, end_s, target):
                self.extract_calls.append((start_s, end_s, target))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"")
                return target

        buffer = EmptyExtractBuffer()

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(buffer, make_row(), Settings())

        assert excinfo.value.stage == "extract"
        # The pin is released even on the empty-file failure path.
        assert buffer.pins == buffer.unpins

    def test_pin_is_released_even_when_extract_raises(self, monkeypatch) -> None:
        buffer = FakeBuffer(fail_extract=True)

        with pytest.raises(clip_factory.ClipProductionError):
            clip_factory.produce(buffer, make_row(), Settings())

        assert buffer.pins == buffer.unpins

    def test_remux_failure_is_tagged(self, monkeypatch) -> None:
        spy = StageSpy().install(monkeypatch)

        def failing_remux(*args, **kwargs):
            spy.order.append("remux")
            raise ffmpeg_mod.FFmpegError("boom", command=["ffmpeg"])

        monkeypatch.setattr(clip_factory, "_remux", failing_remux)

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(), Settings())

        assert excinfo.value.stage == "remux"
        assert "boom" in str(excinfo.value)
        assert spy.order == ["remux"]

    def test_missing_audio_track_in_remux_output_fails(self, monkeypatch) -> None:
        """No audio → no transcript and no audible clip: a hard error, not a warning."""

        def fake_remux(raw, destination, clip_id):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"FAKE")

        monkeypatch.setattr(clip_factory, "_remux", fake_remux)
        monkeypatch.setattr(
            ffmpeg_mod,
            "probe",
            lambda path: ffmpeg_mod.MediaInfo(
                path=Path(path), duration_s=30.0, has_video=True, has_audio=False
            ),
        )

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(), Settings())

        assert excinfo.value.stage == "remux"
        assert "audio" in str(excinfo.value)

    def test_transcribe_failure_is_tagged(self, monkeypatch) -> None:
        install_failure(monkeypatch, "transcribe", RuntimeError("whisper exploded"))

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(), Settings())

        assert excinfo.value.stage == "transcribe"
        assert "whisper exploded" in str(excinfo.value)

    def test_reframe_failure_is_tagged(self, monkeypatch) -> None:
        install_failure(monkeypatch, "build_crop_path", ValueError("no video stream"))

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(), Settings())

        assert excinfo.value.stage == "reframe"

    def test_export_failure_is_tagged(self, monkeypatch) -> None:
        install_failure(monkeypatch, "export_clip", export_mod.ExportError("encoder refused"))

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(), Settings())

        assert excinfo.value.stage == "export"
        assert "encoder refused" in str(excinfo.value)

    def test_export_omitting_the_srt_sidecar_fails(self, monkeypatch) -> None:
        """The .srt sidecar is contractual for live clips: its absence is an error."""

        def export_without_srt(request, *, work_dir, settings=None, on_progress=None, **kwargs):
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            request.destination.write_bytes(b"FAKE_MP4")
            return request.destination

        # Install the spy first, then override just export_clip, so the spy's
        # own export stub doesn't replace the sidecar-omitting one.
        StageSpy().install(monkeypatch)
        monkeypatch.setattr(clip_factory.export, "export_clip", export_without_srt)

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(), Settings())

        assert excinfo.value.stage == "export"
        assert "sidecar" in str(excinfo.value)


# --------------------------------------------------------------------------
# Validation (fail before any disk work)
# --------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        ("start", "end"),
        [(0.0, 0.0), (5.0, 5.0), (5.0, 4.0)],
    )
    def test_empty_or_reversed_window_fails(self, monkeypatch, start: float, end: float) -> None:
        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(
                FakeBuffer(), make_row(clip_start_s=start, clip_end_s=end), Settings()
            )

        assert excinfo.value.stage == "validate"
        assert not (paths.root() / "live" / "clips" / "clip1").exists()

    def test_negative_window_start_fails(self, monkeypatch) -> None:
        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(
                FakeBuffer(), make_row(clip_start_s=-1.0, clip_end_s=10.0), Settings()
            )

        assert excinfo.value.stage == "validate"

    def test_unknown_caption_style_fails(self, monkeypatch) -> None:
        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(caption_style="nope"), Settings())

        assert excinfo.value.stage == "validate"
        assert "style" in str(excinfo.value)

    def test_unknown_ratio_fails(self, monkeypatch) -> None:
        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(ratio="4:5"), Settings())

        assert excinfo.value.stage == "validate"
        assert "ratio" in str(excinfo.value)

    def test_row_without_an_id_fails(self, monkeypatch) -> None:
        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(FakeBuffer(), make_row(id=""), Settings())

        assert excinfo.value.stage == "validate"


# --------------------------------------------------------------------------
# The one real-ffmpeg test
# --------------------------------------------------------------------------


class TestRealRemux:
    """The remux runs against a real ffmpeg; every other stage stays stubbed.

    Segments are generated with the same muxer settings the capture layer uses
    (segment muxer into ``.ts``, ``-reset_timestamps 1``), byte-concatenated
    the way A1's ``buffer.extract`` does, and remuxed with ``-c copy``. Only
    the remux is real — transcribe/reframe/export are stubbed for speed, so
    this needs no x264 encode and stays out of the slow suite.
    """

    @pytest.fixture(scope="class")
    def synthetic_extract(self, tmp_path_factory) -> bytes:
        """~6s testsrc+sine MPEG-TS segments, concatenated like an extraction."""
        seg_dir = tmp_path_factory.mktemp("segs")
        subprocess.run(
            [
                ffmpeg_mod.ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=15:duration=6.2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=6.2",
                "-c:v",
                "mpeg2video",
                "-q:v",
                "5",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                # The artefact the remux exists to absorb: every segment
                # restarts its PTS at zero.
                "-f",
                "segment",
                "-segment_time",
                "3",
                "-reset_timestamps",
                "1",
                "-segment_start_number",
                "0",
                str(seg_dir / "seg_%06d.ts"),
            ],
            check=True,
            capture_output=True,
        )
        segments = sorted(seg_dir.glob("seg_*.ts"))
        assert len(segments) >= 2, "expected at least two segments to concatenate"
        return b"".join(segment.read_bytes() for segment in segments)

    def test_remux_preserves_streams_and_duration(self, monkeypatch, synthetic_extract) -> None:
        row = make_row(id="realtest", clip_start_s=0.0, clip_end_s=6.2)

        class RealExtractBuffer(FakeBuffer):
            def extract(self, start_s, end_s, target):
                self.extract_calls.append((start_s, end_s, target))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(synthetic_extract)
                return target

        # real_probe=True keeps ffprobe live for the post-remux codec checks;
        # everything else stays stubbed. Capture the production remux BEFORE
        # install() replaces it, and record its invocation manually.
        real_remux = clip_factory._remux
        spy = StageSpy().install(monkeypatch, real_probe=True)

        def real_remux_recording(raw, destination, clip_id):
            spy.order.append("remux")
            return real_remux(raw, destination, clip_id)

        monkeypatch.setattr(clip_factory, "_remux", real_remux_recording)

        result = clip_factory.produce(RealExtractBuffer(), row, Settings())  # type: ignore[arg-type]

        # The real remux ran; the stubbed stages ran after it, in order.
        assert "remux" in spy.order
        assert spy.order == ["remux", "audio", "transcribe", "reframe", "export"]

        normalized = paths.root() / "live" / "clips" / "realtest" / clip_factory.NORMALIZED_NAME
        assert normalized.is_file()
        info = ffmpeg_mod.probe(normalized)
        assert info.has_video and info.has_audio
        # Container-only: codecs must survive a -c copy remux un-re-encoded.
        assert info.video_codec == "mpeg2video"
        assert info.audio_codec == "aac"
        assert info.duration_s == pytest.approx(6.2, abs=0.6)
        assert normalized.suffix == ".mp4"
        assert result.rendered_path.parent == normalized.parent

    def test_remux_of_garbage_fails_cleanly(self, monkeypatch) -> None:
        """A truncated/corrupt extraction must fail as a tagged remux error."""

        class GarbageBuffer(FakeBuffer):
            def extract(self, start_s, end_s, target):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"\x47")  # single bogus sync byte
                return target

        real_remux = clip_factory._remux
        spy = StageSpy().install(monkeypatch, real_probe=True)

        def real_remux_recording(raw, destination, clip_id):
            spy.order.append("remux")
            return real_remux(raw, destination, clip_id)

        monkeypatch.setattr(clip_factory, "_remux", real_remux_recording)

        with pytest.raises(clip_factory.ClipProductionError) as excinfo:
            clip_factory.produce(GarbageBuffer(), make_row(), Settings())  # type: ignore[arg-type]

        assert excinfo.value.stage == "remux"
        assert spy.order == ["remux"]
