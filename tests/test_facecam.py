"""Facecam detection and the facecam + gameplay split layout.

Detection is exercised on synthetic data — fake detector outputs and
generated greyscale frames — so these tests need neither MediaPipe nor a
video. One slow test renders a real split clip with ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from autoclip.config import ExportSettings, Settings
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import captions, export, ffmpeg, runner
from autoclip.pipeline.reframe import facecam as fcm
from autoclip.pipeline.reframe.croppath import CropKeyframe, CropPath, CropSegment, centre_crop
from autoclip.pipeline.reframe.facecam import FaceBox, Facecam
from autoclip.pipeline.transcript import Word

SW, SH = 1920, 1080
#: The synthetic overlay: bottom-left corner, 16:9, like a typical stream.
CAM = (0, 816, 470, 264)


def _cam_face(jitter: float = 0.0) -> FaceBox:
    return FaceBox(left=200 + jitter, top=880 + jitter / 2, width=64, height=72)


def _gameplay_frames(count: int = 16, *, seed: int = 7) -> np.ndarray:
    """Analysis-size frames: changing 'gameplay' with a static bordered cam.

    The cam interior is a fixed texture (the streamer's room) with a
    persistent inner line, and the cam has a bright frame around it — the
    things the border search has to tell apart.
    """
    rng = np.random.default_rng(seed)
    scale = SW / fcm.ANALYSIS_WIDTH
    h, w = round(SH / scale), fcm.ANALYSIS_WIDTH
    x0, y0, cw, ch = (round(v / scale) for v in CAM)
    room = rng.uniform(40, 120, size=(ch, cw)).astype(np.float32)
    room[ch // 2, :] = 230.0  # a shelf: a persistent line inside the cam
    frames = []
    for _ in range(count):
        # Smooth, changing "game" content: coarse noise scaled up.
        coarse = rng.uniform(0, 255, size=(9, 16)).astype(np.float32)
        frame = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        frame[y0 : y0 + ch, x0 : x0 + cw] = room
        frame[y0, x0 : x0 + cw] = 250.0
        frame[y0 : y0 + ch, x0 + cw - 1] = 250.0
        frames.append(frame)
    return np.stack(frames)


def _facecam(**overrides) -> Facecam:
    values = {
        "x": CAM[0],
        "y": CAM[1],
        "width": CAM[2],
        "height": CAM[3],
        "source_width": SW,
        "source_height": SH,
        "face_left": 200.0,
        "face_top": 880.0,
        "face_width": 64.0,
        "face_height": 72.0,
        "presence": 0.9,
        "method": "border",
    }
    values.update(overrides)
    return Facecam(**values)


# --------------------------------------------------------------------------
# Stable faces
# --------------------------------------------------------------------------


class TestStableFaces:
    def test_a_small_face_that_stays_put_is_stable(self) -> None:
        detections = [[_cam_face(jitter=i % 3)] for i in range(10)]

        clusters = fcm.stable_face_clusters(detections, SW, SH)

        assert len(clusters) == 1
        assert clusters[0].presence == pytest.approx(1.0)

    def test_dropouts_are_tolerated(self) -> None:
        # Streamers turn away; a face seen in 6 of 10 samples is still a cam.
        detections = [[_cam_face()] if i % 5 not in (1, 3) else [] for i in range(10)]

        assert fcm.stable_face_clusters(detections, SW, SH)

    def test_a_face_crossing_the_frame_is_not_stable(self) -> None:
        detections = [
            [FaceBox(left=100 + i * 170, top=400, width=60, height=70)] for i in range(10)
        ]

        assert fcm.stable_face_clusters(detections, SW, SH) == []

    def test_a_large_face_is_a_talking_head_not_a_cam(self) -> None:
        detections = [[FaceBox(left=800, top=300, width=320, height=380)] for _ in range(10)]

        assert fcm.stable_face_clusters(detections, SW, SH) == []

    def test_a_rarely_seen_face_is_not_stable(self) -> None:
        detections = [[_cam_face()] if i < 2 else [] for i in range(10)]

        assert fcm.stable_face_clusters(detections, SW, SH) == []


# --------------------------------------------------------------------------
# Locating the overlay rectangle
# --------------------------------------------------------------------------


class TestLocateFacecam:
    def test_finds_the_overlay_border(self) -> None:
        frames = _gameplay_frames()
        detections = [[_cam_face(jitter=i % 4)] for i in range(len(frames))]

        cam = fcm.locate_facecam(detections, frames, SW, SH)

        assert cam is not None
        assert cam.method == "border"
        # Within two analysis pixels (16 source px) of the true rectangle,
        # and never past it into the gameplay.
        tolerance = 2 * SW / fcm.ANALYSIS_WIDTH
        assert cam.x == pytest.approx(CAM[0], abs=tolerance)
        assert cam.y == pytest.approx(CAM[1], abs=tolerance)
        assert cam.x + cam.width == pytest.approx(CAM[0] + CAM[2], abs=tolerance)
        assert cam.y + cam.height == pytest.approx(CAM[1] + CAM[3], abs=tolerance)
        assert cam.x + cam.width <= CAM[0] + CAM[2]
        assert cam.y >= CAM[1]
        # Not the smaller rectangle closed by the shelf line inside the cam.
        assert cam.height > CAM[3] * 0.75
        assert cam.is_overlay

    def test_dimensions_are_even(self) -> None:
        frames = _gameplay_frames()
        detections = [[_cam_face()] for _ in range(len(frames))]

        cam = fcm.locate_facecam(detections, frames, SW, SH)

        assert cam is not None
        assert cam.x % 2 == cam.y % 2 == cam.width % 2 == cam.height % 2 == 0

    def test_estimates_a_rect_around_the_face_without_frames(self) -> None:
        detections = [[_cam_face(jitter=i % 4)] for i in range(10)]

        cam = fcm.locate_facecam(detections, None, SW, SH)

        assert cam is not None
        assert cam.method == "estimate"
        face = _cam_face()
        assert cam.x <= face.left and cam.x + cam.width >= face.right
        assert cam.y <= face.top and cam.y + cam.height >= face.bottom
        assert cam.x + cam.width <= SW and cam.y + cam.height <= SH
        assert cam.area_ratio < fcm.MAX_CAM_AREA

    def test_no_faces_means_no_facecam(self) -> None:
        assert fcm.locate_facecam([[] for _ in range(10)], _gameplay_frames(10), SW, SH) is None

    def test_moving_face_means_no_facecam(self) -> None:
        detections = [
            [FaceBox(left=100 + i * 170, top=400, width=60, height=70)] for i in range(10)
        ]

        assert fcm.locate_facecam(detections, None, SW, SH) is None

    def test_talking_head_means_no_facecam(self) -> None:
        detections = [[FaceBox(left=760, top=260, width=300, height=360)] for _ in range(10)]

        assert fcm.locate_facecam(detections, None, SW, SH) is None

    def test_the_docked_cam_beats_a_face_in_the_middle(self) -> None:
        # A lobby character standing still is a stable small face too.
        character = FaceBox(left=930, top=280, width=58, height=64)
        detections = [[_cam_face(), character] for _ in range(10)]

        cam = fcm.locate_facecam(detections, None, SW, SH)

        assert cam is not None
        assert cam.face_left == pytest.approx(200, abs=1)

    def test_the_biggest_cam_wins_a_squad_stream(self) -> None:
        teammate = FaceBox(left=1700, top=60, width=80, height=90)
        streamer = FaceBox(left=180, top=860, width=110, height=120)
        detections = [[teammate, streamer] for _ in range(10)]

        cam = fcm.locate_facecam(detections, None, SW, SH)

        assert cam is not None
        assert cam.face_left == pytest.approx(180, abs=1)


class TestDetectFacecam:
    def test_runs_the_reader_and_detector_over_the_clip(self) -> None:
        seen_times: list[float] = []

        def reader(video, times):
            seen_times.extend(times)
            return [np.zeros((SH, SW, 3), dtype=np.uint8) for _ in times]

        cam = fcm.detect_facecam(
            Path("stream.mp4"),
            start_s=100.0,
            end_s=130.0,
            source_width=SW,
            source_height=SH,
            reader=reader,
            detector=lambda frame: [_cam_face()],
        )

        assert cam is not None
        assert all(100.0 <= t < 130.0 for t in seen_times)
        assert fcm.MIN_SAMPLES <= len(seen_times) <= fcm.MAX_SAMPLES

    def test_unreadable_video_gives_none(self) -> None:
        cam = fcm.detect_facecam(
            Path("missing.mp4"),
            start_s=0.0,
            end_s=10.0,
            source_width=SW,
            source_height=SH,
            reader=lambda video, times: [],
            detector=lambda frame: [_cam_face()],
        )

        assert cam is None


class TestDetectionCache:
    def test_round_trips_a_facecam(self, tmp_path: Path) -> None:
        path = fcm.save_detection(tmp_path / "facecam" / "clip.json", _facecam())

        cached, cam = fcm.load_detection(path)

        assert cached is True
        assert cam == _facecam()

    def test_caches_no_facecam_too(self, tmp_path: Path) -> None:
        path = fcm.save_detection(tmp_path / "clip.json", None)

        assert fcm.load_detection(path) == (True, None)

    def test_missing_or_corrupt_cache_is_a_miss(self, tmp_path: Path) -> None:
        assert fcm.load_detection(tmp_path / "nope.json") == (False, None)
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert fcm.load_detection(bad) == (False, None)


# --------------------------------------------------------------------------
# Layout resolution and geometry
# --------------------------------------------------------------------------


class TestResolveLayout:
    def test_auto_splits_for_an_overlay(self) -> None:
        assert export.resolve_layout("auto", _facecam(), "9:16") == "split"

    def test_auto_crops_without_a_facecam(self) -> None:
        assert export.resolve_layout("auto", None, "9:16") == "crop"

    def test_auto_crops_for_a_centred_face(self) -> None:
        centred = _facecam(x=700, y=300, face_left=930.0, face_top=380.0)
        assert export.resolve_layout("auto", centred, "9:16") == "crop"

    def test_split_forces_any_detected_cam(self) -> None:
        centred = _facecam(x=700, y=300, method="estimate")
        assert export.resolve_layout("split", centred, "9:16") == "split"

    def test_crop_never_splits(self) -> None:
        assert export.resolve_layout("crop", _facecam(), "9:16") == "crop"

    @pytest.mark.parametrize("ratio", ["1:1", "16:9"])
    def test_other_ratios_fall_back_to_crop(self, ratio: str) -> None:
        assert export.resolve_layout("split", _facecam(), ratio) == "crop"


class TestSplitGeometry:
    def test_panel_follows_the_cam_aspect(self) -> None:
        geometry = export.split_geometry(_facecam(), None, 1080, 1920)

        # 470x264 -> 1080 wide is ~607 tall: inside the 30-42 % band.
        assert geometry.top_h == pytest.approx(1080 * 264 / 470, abs=2)
        assert geometry.top_h % 2 == 0

    def test_cam_crop_is_never_stretched(self) -> None:
        for width, height in [(470, 264), (300, 300), (600, 200), (320, 240)]:
            cam = _facecam(width=width, height=height, y=SH - height)
            geometry = export.split_geometry(cam, None, 1080, 1920)
            _, _, cw, ch = geometry.cam

            assert cw / ch == pytest.approx(1080 / geometry.top_h, rel=0.02)
            assert 0.30 * 1920 <= geometry.top_h <= 0.42 * 1920 + 2

    def test_cam_crop_stays_inside_the_cam(self) -> None:
        cam = _facecam(width=600, height=200, y=SH - 200)
        x, y, w, h = export.split_geometry(cam, None, 1080, 1920).cam

        assert cam.x <= x and x + w <= cam.x + cam.width
        assert cam.y <= y and y + h <= cam.y + cam.height

    def test_gameplay_fills_the_rest_at_full_height(self) -> None:
        geometry = export.split_geometry(_facecam(), None, 1080, 1920)
        gx, gy, gw, gh = geometry.game

        assert (gy, gh) == (0, SH)
        assert gw / gh == pytest.approx(1080 / (1920 - geometry.top_h), rel=0.01)
        assert gx % 2 == 0 and gw % 2 == 0

    def test_gameplay_is_centred_by_default(self) -> None:
        gx, _, gw, _ = export.split_geometry(_facecam(), None, 1080, 1920).game

        assert gx + gw / 2 == pytest.approx(SW / 2, abs=2)

    def test_gameplay_follows_the_crop_path_but_not_onto_the_cam(self) -> None:
        # A crop path that tracked the webcam face is ignored; one framing
        # the action on the right is followed.
        on_cam = centre_crop(SW, SH, 10.0)
        on_cam.segments[0].keyframes = [CropKeyframe(0.0, 0.0, 0.0)]
        right = CropPath(
            SW,
            SH,
            [CropSegment(0.0, 10.0, 608, 1080, keyframes=[CropKeyframe(0.0, 1200.0, 0.0)])],
        )
        cam = _facecam()

        tracked = export.split_geometry(cam, on_cam, 1080, 1920).game
        followed = export.split_geometry(cam, right, 1080, 1920).game

        assert tracked[0] + tracked[2] / 2 == pytest.approx(SW / 2, abs=2)
        assert followed[0] + followed[2] / 2 == pytest.approx(1504, abs=40)

    def test_gameplay_moves_clear_of_a_wide_cam(self) -> None:
        cam = _facecam(width=700, height=394, y=SH - 394)
        gx, _, gw, _ = export.split_geometry(cam, None, 1080, 1920).game

        assert gx >= cam.x + cam.width - 1
        assert gx + gw <= SW


class TestSplitFiltergraph:
    def _request(self, **kwargs) -> export.ExportRequest:
        values = {
            "source": Path("in.mp4"),
            "destination": Path("out.mp4"),
            "start_s": 0.0,
            "end_s": 5.0,
            "crop_path": centre_crop(SW, SH, 5.0),
            "words": [],
            "style": captions.get_style("bold_pop"),
            "layout": "split",
            "facecam": _facecam(),
        }
        values.update(kwargs)
        return export.ExportRequest(**values)

    def test_stacks_the_cam_over_the_gameplay(self) -> None:
        request = self._request()
        geometry = export.split_geometry(request.facecam, request.crop_path, 1080, 1920)

        graph = export.build_video_filtergraph(request, subtitle_name="captions.ass")

        cx, cy, cw, ch = geometry.cam
        assert "[0:v]split=2[camsrc][gamesrc]" in graph
        assert f"crop={cw}:{ch}:{cx}:{cy},scale=1080:{geometry.top_h}" in graph
        assert f"scale=1080:{1920 - geometry.top_h}" in graph
        assert "vstack=inputs=2" in graph
        assert graph.endswith("[vstack]ass=filename=captions.ass:fontsdir=fonts[vout]")

    def test_without_captions_it_passes_through(self) -> None:
        graph = export.build_video_filtergraph(self._request(), subtitle_name=None)

        assert graph.endswith("[vstack]null[vout]")

    def test_crop_layout_keeps_the_crop_graph(self) -> None:
        graph = export.build_video_filtergraph(self._request(layout="crop"), subtitle_name=None)

        assert "vstack" not in graph

    def test_square_output_falls_back_to_crop(self) -> None:
        graph = export.build_video_filtergraph(self._request(ratio="1:1"), subtitle_name=None)

        assert "vstack" not in graph

    def test_split_without_a_facecam_falls_back_to_crop(self) -> None:
        graph = export.build_video_filtergraph(self._request(facecam=None), subtitle_name=None)

        assert "vstack" not in graph


class TestSplitCaptions:
    def test_margin_keeps_presets_below_the_cam(self) -> None:
        top_h = 760
        for style in captions.PRESETS.values():
            margin = export.split_caption_margin(style, 1920, top_h)
            # Bottom of the caption block plus three lines stays under the seam.
            block_top = 1920 * (1 - margin - 3 * style.size_ratio * 1.25)
            assert block_top >= top_h
            assert margin <= style.margin_v_ratio

    def test_tall_panel_pulls_captions_down(self) -> None:
        style = captions.get_style("bold_pop")

        assert export.split_caption_margin(style, 1920, 1300) < style.margin_v_ratio

    def test_write_ass_applies_the_margin_override(self, tmp_path: Path) -> None:
        words = [Word(text="hello", start=0.0, end=0.5)]
        style = captions.get_style("bold_pop")

        default = captions.build_ass(words, style, width=1080, height=1920)
        moved = captions.build_ass(words, style, width=1080, height=1920, margin_v_ratio=0.1)

        assert default.styles[captions.STYLE_NAME].marginv == round(1920 * style.margin_v_ratio)
        assert moved.styles[captions.STYLE_NAME].marginv == 192


# --------------------------------------------------------------------------
# Settings, runner and API plumbing
# --------------------------------------------------------------------------


class TestSettingsPlumbing:
    def test_layout_defaults_to_auto(self) -> None:
        assert ExportSettings().layout == "auto"

    def test_layout_is_job_scoped(self, initialised_db) -> None:
        assert ("export", "layout") in runner.JOB_SCOPED_SETTINGS
        source = store.create_source(
            Source(id=new_id(), type="upload", path="C:/m/v.mp4", duration_s=60.0)
        )
        job = store.create_job(
            Job(id=new_id(), source_id=source.id, settings={"export": {"layout": "crop"}})
        )

        assert runner.settings_for_job(job).export.layout == "crop"

    def test_wants_facecam(self) -> None:
        assert runner.wants_facecam("auto", "9:16", True)
        assert runner.wants_facecam("split", "9:16", True)
        assert not runner.wants_facecam("crop", "9:16", True)
        assert not runner.wants_facecam("auto", "1:1", True)
        assert not runner.wants_facecam("auto", "9:16", False)


class TestRunnerFacecam:
    @pytest.fixture
    def setup(self, initialised_db):
        source = store.create_source(
            Source(
                id=new_id(),
                type="upload",
                path="C:/m/stream.mp4",
                duration_s=600.0,
                width=SW,
                height=SH,
            )
        )
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        clip = Clip(id=new_id(), job_id=job.id, rank=1, start_s=10.0, end_s=40.0)
        store.replace_clips(job.id, [clip])
        return source, job, clip

    def test_detection_is_cached_per_clip(self, setup, monkeypatch) -> None:
        source, job, clip = setup
        calls: list[tuple[float, float]] = []

        def fake_detect(video, *, start_s, end_s, source_width, source_height):
            calls.append((start_s, end_s))
            return _facecam()

        monkeypatch.setattr(runner, "detect_facecam", fake_detect)
        workspace = runner.JobWorkspace(job.id)

        first = runner.facecam_for_clip(workspace, clip, source)
        second = runner.facecam_for_clip(workspace, clip, source)

        assert first == second == _facecam()
        assert calls == [(10.0, 40.0)]
        assert workspace.facecam(clip.id).exists()

    def test_a_detection_failure_keeps_the_crop_and_is_not_cached(self, setup, monkeypatch) -> None:
        source, job, clip = setup

        def broken(*args, **kwargs):
            raise RuntimeError("model download failed")

        monkeypatch.setattr(runner, "detect_facecam", broken)
        workspace = runner.JobWorkspace(job.id)

        assert runner.facecam_for_clip(workspace, clip, source) is None
        assert not workspace.facecam(clip.id).exists()

    def test_export_stage_renders_split_for_a_detected_cam(self, setup, monkeypatch) -> None:
        source, job, clip = setup
        monkeypatch.setattr(runner, "detect_facecam", lambda *a, **k: _facecam())
        rendered: list[export.ExportRequest] = []

        def fake_export(request, **kwargs):
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            request.destination.write_bytes(b"mp4")
            rendered.append(request)
            return request.destination

        monkeypatch.setattr(export, "export_clip", fake_export)
        pipeline = runner.PipelineRunner(job, source, settings=Settings())

        from autoclip.pipeline.transcript import Transcript

        transcript = Transcript(words=[Word(text="hi", start=10.0, end=10.5)], language="en")
        pipeline._stage_export([clip], transcript, {clip.id: centre_crop(SW, SH, 30.0)})

        assert rendered[0].layout == "split"
        assert rendered[0].facecam == _facecam()

    def test_crop_layout_skips_detection(self, setup, monkeypatch) -> None:
        source, job, clip = setup

        def boom(*args, **kwargs):
            raise AssertionError("facecam detection must not run for the crop layout")

        monkeypatch.setattr(runner, "detect_facecam", boom)
        rendered: list[export.ExportRequest] = []

        def fake_export(request, **kwargs):
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            request.destination.write_bytes(b"mp4")
            rendered.append(request)
            return request.destination

        monkeypatch.setattr(export, "export_clip", fake_export)
        settings = Settings()
        settings.export.layout = "crop"
        pipeline = runner.PipelineRunner(job, source, settings=settings)

        from autoclip.pipeline.transcript import Transcript

        transcript = Transcript(words=[Word(text="hi", start=10.0, end=10.5)], language="en")
        pipeline._stage_export([clip], transcript, {clip.id: centre_crop(SW, SH, 30.0)})

        assert rendered[0].layout == "crop"
        assert rendered[0].facecam is None


class TestApiPlumbing:
    @pytest.fixture
    def client(self, autoclip_home, monkeypatch):
        from autoclip import app as app_module
        from autoclip.app import create_app
        from fastapi.testclient import TestClient

        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        with TestClient(create_app()) as test_client:
            yield test_client

    @pytest.fixture
    def source(self, client) -> Source:
        return store.create_source(
            Source(
                id=new_id(),
                type="upload",
                path="C:/media/stream.mp4",
                title="Stream",
                duration_s=600.0,
                width=SW,
                height=SH,
            )
        )

    def test_job_create_accepts_a_layout(self, client, source) -> None:
        response = client.post(
            "/api/jobs", json={"source_id": source.id, "settings": {"layout": "split"}}
        )

        assert response.status_code == 201
        job = store.get_job(response.json()["id"])
        assert job.settings["export"]["layout"] == "split"
        assert runner.settings_for_job(job).export.layout == "split"

    def test_job_create_rejects_an_unknown_layout(self, client, source) -> None:
        response = client.post(
            "/api/jobs", json={"source_id": source.id, "settings": {"layout": "sideways"}}
        )

        assert response.status_code == 422

    def test_settings_round_trip_the_layout(self, client) -> None:
        assert client.get("/api/settings").json()["export"]["layout"] == "auto"

        response = client.put("/api/settings", json={"export": {"layout": "crop"}})

        assert response.status_code == 200
        assert response.json()["export"]["layout"] == "crop"
        assert client.put("/api/settings", json={"export": {"layout": "x"}}).status_code == 400

    def test_clips_report_the_jobs_layout(self, client, source) -> None:
        job = store.create_job(
            Job(
                id=new_id(),
                source_id=source.id,
                status="done",
                settings={"export": {"layout": "split"}},
            )
        )
        store.replace_clips(
            job.id, [Clip(id=new_id(), job_id=job.id, rank=1, start_s=0.0, end_s=30.0)]
        )

        clip = client.get(f"/api/jobs/{job.id}/clips").json()[0]

        assert clip["layout"] == "split"

    def test_re_export_uses_the_requested_layout(self, client, source, monkeypatch) -> None:
        from autoclip.api import clips as clips_api

        job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
        clip = Clip(id=new_id(), job_id=job.id, rank=1, start_s=0.0, end_s=30.0, title="T")
        store.replace_clips(job.id, [clip])
        transcript_path = runner.JobWorkspace(job.id).transcript
        from autoclip.pipeline.transcript import Transcript

        Transcript(words=[Word(text="hi", start=1.0, end=1.4)], language="en").save(transcript_path)
        monkeypatch.setattr(runner, "detect_facecam", lambda *a, **k: _facecam())
        rendered: list[export.ExportRequest] = []

        def fake_export(request, **kwargs):
            request.destination.parent.mkdir(parents=True, exist_ok=True)
            request.destination.write_bytes(b"mp4")
            rendered.append(request)
            return request.destination

        monkeypatch.setattr(clips_api.export_module, "export_clip", fake_export)

        split = client.post(
            f"/api/clips/{clip.id}/export",
            json={"ratio": "9:16", "style": "bold_pop", "layout": "split"},
        )
        cropped = client.post(
            f"/api/clips/{clip.id}/export",
            json={"ratio": "9:16", "style": "bold_pop", "layout": "crop"},
        )
        square = client.post(
            f"/api/clips/{clip.id}/export",
            json={"ratio": "1:1", "style": "bold_pop", "layout": "split"},
        )

        assert split.status_code == cropped.status_code == square.status_code == 201
        assert [r.layout for r in rendered] == ["split", "crop", "crop"]
        assert rendered[0].facecam == _facecam()


# --------------------------------------------------------------------------
# Real render
# --------------------------------------------------------------------------


@pytest.mark.slow
class TestSplitRender:
    def test_renders_a_vertical_split_clip(self, tmp_path: Path) -> None:
        """testsrc2 'gameplay' with a drawn 'cam' box, rendered on the CPU."""
        source = tmp_path / "stream.mp4"
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
                "testsrc2=size=1280x720:rate=30:duration=4",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=4",
                "-vf",
                "drawbox=x=0:y=540:w=320:h=180:color=red@1:t=fill",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        cam = Facecam(
            x=0,
            y=540,
            width=320,
            height=180,
            source_width=1280,
            source_height=720,
            face_left=140.0,
            face_top=580.0,
            face_width=40.0,
            face_height=44.0,
            presence=1.0,
        )
        destination = tmp_path / "split.mp4"
        request = export.ExportRequest(
            source=source,
            destination=destination,
            start_s=0.5,
            end_s=3.5,
            crop_path=centre_crop(1280, 720, 3.0),
            words=[
                Word(text=w, start=0.6 + i * 0.5, end=1.0 + i * 0.5)
                for i, w in enumerate(["split", "layout", "works", "fine"])
            ],
            style=captions.get_style("bold_pop"),
            layout="split",
            facecam=cam,
        )

        export.export_clip(
            request,
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False),
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert info.has_audio
        assert not destination.with_name("split.partial.mp4").exists()

        # The top panel is the red "cam" box, scaled up.
        top_h = export.split_geometry(cam, request.crop_path, 1080, 1920).top_h
        frame = subprocess.run(
            [
                ffmpeg.ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "1.0",
                "-i",
                str(destination),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:",
            ],
            check=True,
            capture_output=True,
        ).stdout
        image = np.frombuffer(frame, dtype=np.uint8).reshape(1920, 1080, 3)
        panel = image[20 : top_h - 20, 20:-20].reshape(-1, 3).mean(axis=0)
        assert panel[0] > 180 and panel[1] < 80 and panel[2] < 80
