"""Smoke renders against real ffmpeg: speaker split, faceless fit, joined clips.

These prove the new filtergraphs parse and produce the right frame size and
length; the planner and timeline logic have their own unit tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from autoclip.config import ExportSettings
from autoclip.pipeline import captions, edit, export, ffmpeg, segments
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropRect,
    CropSegment,
    Strategy,
    centre_crop,
    fit_segment,
    join_paths,
)
from autoclip.pipeline.segments import Span
from autoclip.pipeline.transcript import Word

pytestmark = pytest.mark.slow

SW, SH = 1280, 720
DURATION = 12.0


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "podcast.mp4"
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
            f"testsrc2=size={SW}x{SH}:rate=30:duration={DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=330:duration={DURATION}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def settings() -> ExportSettings:
    # CPU encode keeps the test independent of the machine's GPU.
    return ExportSettings(prefer_hardware_encoder=False, crf=30)


def split_segment(start: float, end: float) -> CropSegment:
    return CropSegment(
        start_s=start,
        end_s=end,
        width=404,
        height=720,
        keyframes=[CropKeyframe(start, 438.0, 0.0)],
        strategy=Strategy.SPLIT,
        panes=[CropRect(40, 80, 448, 398), CropRect(780, 90, 448, 398)],
    )


def words_between(start: float, end: float) -> list[Word]:
    out = []
    t = start + 0.2
    i = 0
    while t + 0.4 < end:
        out.append(Word(f"word{i}", round(t, 3), round(t + 0.35, 3)))
        t += 0.5
        i += 1
    return out


def test_speaker_split_fit_and_crop_in_one_clip(source_video, tmp_path) -> None:
    """A shot each of split, faceless fit and crop, captioned, with edits on."""
    crop_path = CropPath(
        SW,
        SH,
        [
            split_segment(0.0, 2.0),
            fit_segment(2.0, 4.0, SW, SH),
            CropSegment(
                4.0,
                6.0,
                404,
                720,
                keyframes=[CropKeyframe(4.0, 400.0, 0.0)],
                strategy=Strategy.TRACK,
            ),
        ],
    )
    words = words_between(2.0, 8.0)
    request = export.ExportRequest(
        source=source_video,
        destination=tmp_path / "split.mp4",
        start_s=2.0,
        end_s=8.0,
        crop_path=crop_path,
        words=words,
        style=captions.get_style("bold_pop"),
        edit=edit.EditPlan(hook_text="TWO SPEAKERS", keywords=["word3"], two_pass_loudnorm=False),
    )

    export.export_clip(request, work_dir=tmp_path / "work", settings=settings())

    info = ffmpeg.probe(request.destination)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration_s == pytest.approx(6.0, abs=0.35)
    assert info.has_audio


def test_joined_clip_renders_its_parts_back_to_back(source_video, tmp_path) -> None:
    spans = [Span(1.0, 3.5), Span(7.0, 10.0)]
    total = segments.total(spans)
    words = segments.words_on_timeline(words_between(1.0, 3.5) + words_between(7.0, 10.0), spans)
    crop_path = join_paths(
        [centre_crop(SW, SH, spans[0].duration), CropPath(SW, SH, [split_segment(0.0, 3.0)])]
    )
    # A pause cut on the joined timeline, across the join, plus two-pass
    # loudness (which measures the joined audio with its own inputs).
    plan = edit.EditPlan(
        keeps=[edit.Keep(0.0, 2.0), edit.Keep(2.6, total)],
        hook_text="JOINED",
        two_pass_loudnorm=True,
    )
    request = export.ExportRequest(
        source=source_video,
        destination=tmp_path / "joined.mp4",
        start_s=0.0,
        end_s=total,
        crop_path=crop_path,
        words=words,
        style=captions.get_style("bold_pop"),
        edit=plan,
        spans=segments.as_pairs(spans),
    )

    export.export_clip(request, work_dir=tmp_path / "work", settings=settings())

    info = ffmpeg.probe(request.destination)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration_s == pytest.approx(total - 0.6, abs=0.35)
    assert info.has_audio


def test_joined_clip_without_edits_renders(source_video, tmp_path) -> None:
    """The plain export path (no edit plan) maps the joined audio itself."""
    spans = [Span(0.5, 2.5), Span(9.0, 11.0)]
    request = export.ExportRequest(
        source=source_video,
        destination=tmp_path / "plain-joined.mp4",
        start_s=0.0,
        end_s=4.0,
        crop_path=centre_crop(SW, SH, 4.0),
        words=segments.words_on_timeline(words_between(0.5, 11.0), spans),
        style=captions.get_style("bold_pop"),
        spans=segments.as_pairs(spans),
    )

    export.export_clip(request, work_dir=tmp_path / "work", settings=settings())

    info = ffmpeg.probe(request.destination)
    assert info.duration_s == pytest.approx(4.0, abs=0.35)
    assert info.has_audio
