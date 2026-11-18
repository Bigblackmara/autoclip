"""Framing planner: two-speaker split screen and blurred fit for faceless shots.

Decisions are tested on synthetic face tracks, so no model or video is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from autoclip.pipeline import ffmpeg
from autoclip.pipeline import reframe as reframe_module
from autoclip.pipeline.reframe import ReframeConfig, Strategy, build_crop_path, plan_shot_layout
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    centre_crop,
    join_paths,
    split_at,
    target_crop_size,
)
from autoclip.pipeline.reframe.faces import FaceObservation
from autoclip.pipeline.reframe.scenes import Shot
from autoclip.pipeline.reframe.tracker import build_tracks

SW, SH = 1920, 1080
CROP_W, CROP_H = target_crop_size(SW, SH, 9, 16)
FPS = 5.0


def face(t: float, cx: float, *, width: float = 220, talking: bool = True) -> FaceObservation:
    """A face at ``cx`` whose mouth opens and closes when ``talking``."""
    mar = (0.05 + 0.25 * ((int(t * FPS) % 3) == 0)) if talking else 0.05
    return FaceObservation(
        t=round(t, 3),
        cx=cx,
        cy=480,
        width=width,
        height=width * 1.3,
        eye_y=440,
        mar=mar,
    )


def two_shot(
    duration: float = 6.0,
    *,
    left: float = 520,
    right: float = 1400,
    right_width: float = 220,
    right_talks: bool = True,
    right_presence: float = 1.0,
) -> list[FaceObservation]:
    out: list[FaceObservation] = []
    samples = int(duration * FPS)
    for i in range(samples):
        t = i / FPS
        out.append(face(t, left))
        if i < samples * right_presence:
            out.append(face(t, right, width=right_width, talking=right_talks))
    return out


def plan(observations, *, framing: str = "auto", shot: Shot | None = None, speaker_map=None):
    shot = shot or Shot(0.0, 6.0)
    return plan_shot_layout(
        shot,
        observations=observations,
        tracks=build_tracks(observations),
        speaker_map=speaker_map or {},
        source_w=SW,
        source_h=SH,
        crop_w=CROP_W,
        crop_h=CROP_H,
        config=ReframeConfig(framing=framing, sample_fps=FPS),
    )


class TestTwoSpeakerSplit:
    def test_two_talking_people_far_apart_split(self) -> None:
        segment = plan(two_shot())

        assert segment is not None
        assert segment.strategy is Strategy.SPLIT
        assert segment.is_split
        top, bottom = segment.panes
        # Left person on top, right person below, each centred in their pane.
        assert top.x + top.width / 2 == pytest.approx(520, abs=2)
        assert bottom.x + bottom.width / 2 == pytest.approx(1400, abs=2)

    def test_panes_have_the_half_frame_aspect_and_stay_in_the_source(self) -> None:
        segment = plan(two_shot(left=150, right=1800))

        assert segment is not None
        for pane in segment.panes:
            assert pane.width / pane.height == pytest.approx(9 / 8, rel=0.02)
            assert pane.width % 2 == 0 and pane.height % 2 == 0
            assert 0 <= pane.x <= SW - pane.width
            assert 0 <= pane.y <= SH - pane.height

    def test_panes_are_steady_for_the_whole_shot(self) -> None:
        segment = plan(two_shot())

        assert segment is not None
        assert segment.start_s == 0.0 and segment.end_s == 6.0
        assert len(segment.keyframes) == 1

    def test_the_nominal_crop_keeps_the_target_ratio(self) -> None:
        # Used for non-vertical ratios and by the review preview.
        segment = plan(two_shot())

        assert segment is not None
        assert (segment.width, segment.height) == (CROP_W, CROP_H)

    def test_people_close_enough_for_one_crop_do_not_split(self) -> None:
        # 400 px apart fits inside a 608 px wide 9:16 crop.
        assert plan(two_shot(left=760, right=1160)) is None

    def test_a_silent_listener_does_not_earn_half_the_frame(self) -> None:
        assert plan(two_shot(right_talks=False)) is None

    def test_a_small_background_face_does_not_split(self) -> None:
        assert plan(two_shot(right_width=80)) is None

    def test_a_face_seen_only_briefly_does_not_split(self) -> None:
        assert plan(two_shot(right_presence=0.3)) is None

    def test_short_shots_never_split(self) -> None:
        observations = [o for o in two_shot() if o.t < 1.0]

        assert plan(observations, shot=Shot(0.0, 1.0)) is None

    def test_diarized_speakers_decide_who_talks(self) -> None:
        observations = two_shot(right_talks=False)
        tracks = build_tracks(observations)
        left = min(tracks, key=lambda t: t.mean_cx)
        right = max(tracks, key=lambda t: t.mean_cx)

        # Both mapped to speakers: split even though one mouth barely moves.
        both = plan_shot_layout(
            Shot(0.0, 6.0),
            observations=observations,
            tracks=tracks,
            speaker_map={"A": left, "B": right},
            source_w=SW,
            source_h=SH,
            crop_w=CROP_W,
            crop_h=CROP_H,
            config=ReframeConfig(framing="auto", sample_fps=FPS),
        )
        one = plan_shot_layout(
            Shot(0.0, 6.0),
            observations=observations,
            tracks=tracks,
            speaker_map={"A": left},
            source_w=SW,
            source_h=SH,
            crop_w=CROP_W,
            crop_h=CROP_H,
            config=ReframeConfig(framing="auto", sample_fps=FPS),
        )

        assert both is not None and both.is_split
        assert one is None

    def test_forced_split_relaxes_distance_and_speech(self) -> None:
        segment = plan(two_shot(left=760, right=1160, right_talks=False), framing="split")

        assert segment is not None and segment.is_split

    def test_forced_split_still_needs_two_distinct_faces(self) -> None:
        assert plan(two_shot(left=900, right=1000), framing="split") is None

    def test_square_output_never_splits(self) -> None:
        observations = two_shot()
        crop_w, crop_h = target_crop_size(SW, SH, 1, 1)
        segment = plan_shot_layout(
            Shot(0.0, 6.0),
            observations=observations,
            tracks=build_tracks(observations),
            speaker_map={},
            source_w=SW,
            source_h=SH,
            crop_w=crop_w,
            crop_h=crop_h,
            config=ReframeConfig(aspect_w=1, aspect_h=1, framing="auto", sample_fps=FPS),
        )

        assert segment is None

    def test_a_single_speaker_is_left_to_the_ordinary_crop(self) -> None:
        observations = [face(i / FPS, 960) for i in range(30)]

        assert plan(observations) is None


class TestFaceless:
    def test_a_shot_without_faces_is_fitted(self) -> None:
        segment = plan([], shot=Shot(2.0, 8.0))

        assert segment is not None
        assert segment.fit
        assert (segment.start_s, segment.end_s) == (2.0, 8.0)
        assert (segment.width, segment.height) == (SW, SH)

    def test_tiny_faces_do_not_count_as_a_presenter(self) -> None:
        observations = [face(i / FPS, 960, width=30) for i in range(30)]

        segment = plan(observations)

        assert segment is not None and segment.fit

    def test_a_rare_face_still_counts_as_faceless(self) -> None:
        observations = [face(i / FPS, 960) for i in range(3)]  # 3 of 30 samples

        segment = plan(observations)

        assert segment is not None and segment.fit


# --------------------------------------------------------------------------
# build_crop_path with the framing modes (face detection and shots faked)
# --------------------------------------------------------------------------


@pytest.fixture
def fake_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A fake 1920x1080 video whose faces and shots the test sets."""
    state: dict = {"observations": [], "shots": [Shot(0.0, 6.0)]}

    monkeypatch.setattr(ffmpeg, "probe", lambda path: _Info())
    monkeypatch.setattr(reframe_module, "sample_faces", lambda *a, **k: list(state["observations"]))
    monkeypatch.setattr(reframe_module, "detect_shots", lambda *a, **k: list(state["shots"]))
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    state["video"] = video
    return state


class _Info:
    has_video = True
    width = SW
    height = SH
    duration_s = 6.0


def _build(state, framing: str) -> CropPath:
    return build_crop_path(
        state["video"],
        start_s=0.0,
        end_s=6.0,
        config=ReframeConfig(framing=framing, sample_fps=FPS),
    )


class TestBuildCropPathFraming:
    def test_fill_keeps_the_original_centre_crop_for_faceless_video(self, fake_video) -> None:
        path = _build(fake_video, "fill")

        assert len(path.segments) == 1
        assert not path.segments[0].fit
        assert path.segments[0].strategy is Strategy.GENERAL

    def test_auto_fits_faceless_video(self, fake_video) -> None:
        path = _build(fake_video, "auto")

        assert len(path.segments) == 1 and path.segments[0].fit

    def test_fit_framing_fits_everything(self, fake_video) -> None:
        fake_video["observations"] = [face(i / FPS, 960) for i in range(30)]

        path = _build(fake_video, "fit")

        assert all(s.fit for s in path.segments)

    def test_auto_splits_a_two_shot_and_crops_a_single(self, fake_video) -> None:
        fake_video["shots"] = [Shot(0.0, 3.0), Shot(3.0, 6.0)]
        two = list(two_shot(duration=3.0))
        single = [face(3.0 + i / FPS, 960) for i in range(15)]
        fake_video["observations"] = two + single

        path = _build(fake_video, "auto")

        assert [s.strategy for s in path.segments] == [Strategy.SPLIT, Strategy.TRACK]
        assert path.segments[0].end_s == path.segments[1].start_s == 3.0
        assert path.split_share == pytest.approx(0.5)

    def test_fill_never_splits(self, fake_video) -> None:
        fake_video["observations"] = two_shot()

        path = _build(fake_video, "fill")

        assert not path.has_split

    def test_split_panes_survive_a_save_and_load(self, fake_video, tmp_path: Path) -> None:
        fake_video["observations"] = two_shot()
        path = _build(fake_video, "auto")

        loaded = CropPath.load(path.save(tmp_path / "crop.json"))

        assert loaded.segments[0].panes == path.segments[0].panes
        assert loaded.segments[0].strategy is Strategy.SPLIT

    def test_live_watch_default_is_unchanged(self) -> None:
        assert ReframeConfig().framing == "fill"


class TestJoinedPaths:
    def test_join_paths_shifts_each_part(self) -> None:
        first = centre_crop(SW, SH, 4.0)
        second = centre_crop(SW, SH, 6.0)

        joined = join_paths([first, second])

        assert [(s.start_s, s.end_s) for s in joined.segments] == [(0.0, 4.0), (4.0, 10.0)]
        assert joined.segments[1].keyframes[0].t == 4.0

    def test_split_at_cuts_a_spanning_segment_at_the_joins(self) -> None:
        moving = CropPath(
            SW,
            SH,
            [
                CropSegment(
                    0.0,
                    10.0,
                    CROP_W,
                    CROP_H,
                    keyframes=[CropKeyframe(0.0, 0.0, 0.0), CropKeyframe(10.0, 100.0, 0.0)],
                    strategy=Strategy.TRACK,
                )
            ],
        )

        cut = split_at(moving, [4.0])

        assert [(s.start_s, s.end_s) for s in cut.segments] == [(0.0, 4.0), (4.0, 10.0)]
        # The second piece starts where the pan had got to.
        assert cut.segments[1].keyframes[0].t == 4.0
        assert cut.segments[1].keyframes[0].x == pytest.approx(40.0)
