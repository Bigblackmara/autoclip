"""Clip-editor features ported from CrestClip.

Hook options (detection, persistence, the review API, the render), variants,
the flash-forward opening (plan timings, caption retiming, hook placement)
and the caption-language modes (the Devanagari -> Hinglish romaniser and
the LLM translation pass with its retiming).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from autoclip import db
from autoclip.config import ExportSettings, Settings
from autoclip.db import store
from autoclip.db.models import Clip, ClipEdit, Job, Source, new_id
from autoclip.pipeline import captions, edit, export, highlights, language
from autoclip.pipeline.edit import Keep, Zoom
from autoclip.pipeline.prepare import Silence
from autoclip.pipeline.reframe.croppath import centre_crop
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DETECTION_PROMPT, detection_config
from autoclip.providers.base import (
    ClipCandidate,
    DetectionConfig,
    LLMProvider,
    ProviderStatus,
    TranscriptWindow,
)


def _w(text: str, start: float, end: float | None = None) -> Word:
    return Word(text=text, start=start, end=start + 0.4 if end is None else end)


def _speech(duration: float, *, start: float = 0.0, every: float = 0.5) -> list[Word]:
    """A word every ``every`` seconds, a sentence end every 8 words."""
    words: list[Word] = []
    t = start + 0.1
    while t + 0.4 < start + duration:
        index = len(words)
        text = f"word{index}" + ("." if index % 8 == 7 else "")
        words.append(_w(text, t))
        t += every
    return words


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------

V7_COLUMNS = {
    "hook_options",
    "hook_text",
    "description",
    "hashtags",
    "teaser_start_s",
    "teaser_end_s",
    "flash_forward",
    "variant_of",
    "motion_seed",
}


def _clip_columns() -> set[str]:
    with db.connection() as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(clips)").fetchall()}


class TestMigration:
    def test_fresh_database_has_the_editor_columns(self, initialised_db) -> None:
        assert db.SCHEMA_VERSION == 7
        assert _clip_columns() >= V7_COLUMNS

    def test_a_v6_database_upgrades_and_keeps_its_clips(self, autoclip_home: Path) -> None:
        from autoclip.db import schema

        with db.connection() as conn:
            for migration in schema.MIGRATIONS[:6]:
                with conn:
                    migration(conn)
            conn.execute("PRAGMA user_version = 6")
            conn.execute(
                "INSERT INTO sources (id, type, path, title, created_at) "
                "VALUES ('s', 'upload', 'p', 't', 'now')"
            )
            conn.execute(
                "INSERT INTO jobs (id, source_id, status, created_at, updated_at) "
                "VALUES ('j', 's', 'done', 'now', 'now')"
            )
            conn.execute(
                "INSERT INTO clips (id, job_id, start_s, end_s, title, created_at) "
                "VALUES ('c', 'j', 1, 30, 'Old clip', 'now')"
            )

        assert db.init() == 7
        assert db.init() == 7  # idempotent
        clip = store.get_clip("c")
        assert clip is not None and clip.title == "Old clip"
        assert clip.hook_options == [] and clip.hashtags == [] and clip.flash_forward is None
        assert clip.headline == "Old clip" and clip.motion_seed == 0 and clip.teaser is None

    def test_rerunning_v7_on_a_migrated_database_is_a_no_op(self, initialised_db) -> None:
        from autoclip.db import schema

        with db.connection() as conn, conn:
            schema._migration_v7(conn)
        assert _clip_columns() >= V7_COLUMNS

    def test_a_database_without_a_clips_table_is_left_alone(self) -> None:
        from autoclip.db import schema

        conn = sqlite3.connect(":memory:")
        schema._migration_v7(conn)
        conn.commit()


# --------------------------------------------------------------------------
# Hook options: detection -> clip -> database
# --------------------------------------------------------------------------


class TestCandidateParsing:
    def test_hooks_hashtags_and_teaser_are_cleaned(self) -> None:
        candidate = ClipCandidate.model_validate(
            {
                "start_word_index": 10,
                "end_word_index": 90,
                "hooks": ["  The number  that kills ", "the number that kills", "", "Two", "Three"],
                "hashtags": "#Money  business, #money #side-hustle",
                "teaser_start_word_index": "40",
                "teaser_end_word_index": 46.0,
                "description": None,
            }
        )

        assert candidate.hooks == ["The number that kills", "Two", "Three"]
        assert candidate.hashtags == ["#Money", "#business", "#sidehustle"]
        assert (candidate.teaser_start_word_index, candidate.teaser_end_word_index) == (40, 46)
        assert candidate.description == ""

    def test_minus_one_means_no_teaser_and_old_replies_still_parse(self) -> None:
        none = ClipCandidate.model_validate(
            {
                "start_word_index": 1,
                "end_word_index": 9,
                "teaser_start_word_index": -1,
                "teaser_end_word_index": -1,
            }
        )
        old = ClipCandidate.model_validate({"start_word_index": 1, "end_word_index": 9})

        assert none.teaser_start_word_index is None
        assert old.hooks == [] and old.hashtags == [] and old.teaser_end_word_index is None

    def test_a_teaser_outside_the_clip_is_dropped_by_the_parser(self) -> None:
        class Echo(LLMProvider):
            name = "echo"
            requires_key = False

            async def _complete(self, system, user, config):  # pragma: no cover - unused
                return ""

            async def health_check(self):  # pragma: no cover - unused
                return ProviderStatus(name="echo", available=True)

        raw = json.dumps(
            {
                "clips": [
                    {
                        "start_word_index": 10,
                        "end_word_index": 50,
                        "teaser_start_word_index": 60,
                        "teaser_end_word_index": 64,
                    },
                    {
                        "start_word_index": 10,
                        "end_word_index": 50,
                        "teaser_start_word_index": 30,
                        "teaser_end_word_index": 34,
                    },
                ]
            }
        )
        window = TranscriptWindow(text="", first_word=0, last_word=100)

        clips = Echo("m")._parse(raw, window).clips

        assert clips[0].teaser_start_word_index is None
        assert clips[1].teaser_start_word_index == 30

    def test_detection_config_uses_the_v3_prompt_and_a_language_rule(self) -> None:
        settings = Settings()
        settings.export.caption_language = "english"

        config = detection_config(settings)

        assert config.prompt_version == DETECTION_PROMPT == "highlight_v3"
        assert "in English" in config.writing_rule
        prompt = (Path(highlights.__file__).parents[1] / "prompts" / "highlight_v3.txt").read_text(
            encoding="utf-8"
        )
        assert "`hooks`" in prompt and "teaser_start_word_index" in prompt
        assert "## Audience signals" in prompt  # v3 keeps v2's signal handling


class TestBuildClips:
    def test_hooks_post_text_and_teaser_land_on_the_clip(self, initialised_db) -> None:
        words = [_w(f"w{i}" + ("." if i % 10 == 9 else ""), i * 0.5) for i in range(120)]
        transcript = Transcript(words=words)
        candidate = ClipCandidate(
            start_word_index=10,
            end_word_index=69,
            title="Title",
            hook="w10",
            hooks=["First hook", "Second hook", "Third hook"],
            teaser_start_word_index=50,
            teaser_end_word_index=54,
            description="Watch this.",
            hashtags=["#one", "#two"],
            score=80,
        )
        config = DetectionConfig(min_duration_s=5.0, max_duration_s=60.0, max_clips=5)

        (clip,) = highlights.build_clips(transcript, [candidate], config, job_id="j")

        assert clip.hook_options == ["First hook", "Second hook", "Third hook"]
        assert clip.hook_text == "First hook" == clip.headline
        assert clip.description == "Watch this." and clip.hashtags == ["#one", "#two"]
        assert clip.teaser == (25.0, 27.4)

        source = store.create_source(Source(id="s", type="upload", path="p"))
        job = store.create_job(Job(id="j", source_id=source.id))
        store.replace_clips(job.id, [clip])
        (loaded,) = store.list_clips(job.id)
        assert loaded.hook_options == clip.hook_options and loaded.teaser == clip.teaser
        assert loaded.hashtags == ["#one", "#two"] and loaded.flash_forward is None

    def test_a_teaser_outside_the_refined_clip_is_not_kept(self) -> None:
        words = [_w(f"w{i}", i * 0.5) for i in range(40)]
        candidate = ClipCandidate(
            start_word_index=0,
            end_word_index=39,
            teaser_start_word_index=1,
            teaser_end_word_index=2,
        )

        assert highlights.teaser_times(Transcript(words=words), candidate, 5.0, 19.0) == (
            None,
            None,
        )


# --------------------------------------------------------------------------
# Review API: hook choice, post text, flash-forward toggle, variants
# --------------------------------------------------------------------------


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch):
    from autoclip import app as app_module
    from autoclip.app import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def _job_with_clip(**values) -> tuple[Job, Clip]:
    source = store.create_source(
        Source(
            id=new_id(), type="upload", path="C:/m/v.mp4", duration_s=120.0, width=1920, height=1080
        )
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
    base = {
        "id": new_id(),
        "job_id": job.id,
        "rank": 1,
        "start_s": 10.0,
        "end_s": 30.0,
        "start_word": 20,
        "end_word": 59,
        "title": "Big win",
        "hook_options": ["Hook A", "Hook B", "Hook C"],
        "hook_text": "Hook A",
        "description": "Desc",
        "hashtags": ["#a"],
        "teaser_start_s": 22.0,
        "teaser_end_s": 24.0,
    }
    base.update(values)
    clip = Clip(**base)
    store.replace_clips(job.id, [clip])
    return job, clip


def _capture_exports(monkeypatch) -> list[export.ExportRequest]:
    from autoclip.api import clips as clips_api

    rendered: list[export.ExportRequest] = []

    def fake_export(request, **kwargs):
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        request.destination.write_bytes(b"mp4")
        rendered.append(request)
        return request.destination

    monkeypatch.setattr(clips_api.export_module, "export_clip", fake_export)
    return rendered


class TestReviewApi:
    def test_clip_out_lists_hooks_post_text_and_flash_state(self, client) -> None:
        _, clip = _job_with_clip()

        body = client.get(f"/api/clips/{clip.id}").json()

        assert body["hook_options"] == ["Hook A", "Hook B", "Hook C"]
        assert body["headline"] == "Hook A" and body["hook_text"] == "Hook A"
        assert body["description"] == "Desc" and body["hashtags"] == ["#a"]
        assert body["flash_forward"] is None and body["flash_forward_default"] is False
        assert body["has_teaser"] is True and body["variant_of"] is None

    def test_pick_a_hook_write_one_and_edit_post_text(self, client) -> None:
        _, clip = _job_with_clip()

        picked = client.patch(f"/api/clips/{clip.id}", json={"hook_text": "Hook C"}).json()
        written = client.patch(
            f"/api/clips/{clip.id}",
            json={
                "hook_text": "  My own   hook ",
                "description": " New caption ",
                "hashtags": ["money", "#Side Hustle", "#money"],
            },
        ).json()

        assert picked["headline"] == "Hook C"
        assert written["hook_text"] == "My own hook" == written["headline"]
        assert written["description"] == "New caption"
        assert written["hashtags"] == ["#money", "#SideHustle"]
        assert store.get_clip(clip.id).hook_text == "My own hook"

    def test_flash_forward_toggle_is_tri_state(self, client) -> None:
        _, clip = _job_with_clip()

        on = client.patch(f"/api/clips/{clip.id}", json={"flash_forward": True}).json()
        untouched = client.patch(f"/api/clips/{clip.id}", json={"title": "Renamed"}).json()
        cleared = client.patch(f"/api/clips/{clip.id}", json={"flash_forward": None}).json()

        assert on["flash_forward"] is True
        assert untouched["flash_forward"] is True
        assert cleared["flash_forward"] is None

    def test_export_renders_the_chosen_hook_and_the_flash_forward(
        self, client, monkeypatch
    ) -> None:
        from autoclip.pipeline.runner import JobWorkspace

        rendered = _capture_exports(monkeypatch)
        job, clip = _job_with_clip()
        Transcript(words=_speech(40.0), language="en").save(JobWorkspace(job.id).transcript)
        client.patch(f"/api/clips/{clip.id}", json={"hook_text": "Hook B", "flash_forward": True})

        response = client.post(f"/api/clips/{clip.id}/export", json={"ratio": "9:16"})

        assert response.status_code == 201
        (request,) = rendered
        assert request.edit is not None
        assert request.edit.hook_text == "Hook B"
        assert request.edit.teaser is not None and request.edit.keeps[0] == request.edit.teaser
        assert 11.8 <= request.edit.teaser.start <= 12.0  # 22 s absolute, clip from 10 s
        assert request.edit.hook_start_s == pytest.approx(request.edit.teaser.duration)


class TestVariants:
    def test_variant_cycles_style_hook_and_zoom_pattern_and_exports(
        self, client, monkeypatch
    ) -> None:
        from autoclip.pipeline.runner import JobWorkspace

        rendered = _capture_exports(monkeypatch)
        job, clip = _job_with_clip(flash_forward=True)
        workspace = JobWorkspace(job.id)
        Transcript(words=_speech(40.0), language="en").save(workspace.transcript)
        centre_crop(1920, 1080, 20.0).save(workspace.crop_path(clip.id))
        store.upsert_clip_edit(ClipEdit(clip_id=clip.id, caption_style="karaoke_fill"))

        response = client.post(f"/api/clips/{clip.id}/variant", json={})

        assert response.status_code == 201
        body = response.json()
        assert body["id"] != clip.id and body["variant_of"] == clip.id
        assert body["caption_style"] == "clean_lower"  # the preset after karaoke_fill
        assert body["headline"] == "Hook B" and body["motion_seed"] == 1
        assert body["title"] == "Big win (variant 1)" and body["status"] == "exported"
        assert body["start_s"] == clip.start_s and body["end_s"] == clip.end_s
        assert body["flash_forward"] is True and body["hashtags"] == ["#a"]
        assert workspace.crop_path(body["id"]).exists(), "the framing is reused"
        (request,) = rendered
        assert request.style.key == "clean_lower"
        assert request.edit is not None and request.edit.hook_text == "Hook B"
        assert request.destination.name == "01-big-win-variant-1_9x16.mp4"
        listed = client.get(f"/api/jobs/{job.id}/clips").json()
        assert [c["id"] for c in listed] == [clip.id, body["id"]]

    def test_a_variant_of_a_variant_counts_on_from_the_original(self, client, monkeypatch) -> None:
        _capture_exports(monkeypatch)
        _, clip = _job_with_clip()

        first = client.post(f"/api/clips/{clip.id}/variant", json={"export": False}).json()
        second = client.post(f"/api/clips/{first['id']}/variant", json={"export": False}).json()

        assert first["status"] == "kept" and not first["exports"]
        assert second["variant_of"] == clip.id
        assert second["title"] == "Big win (variant 2)"
        assert second["motion_seed"] == 2 and second["headline"] == "Hook C"
        assert second["caption_style"] == captions_after(first["caption_style"])

    def test_variant_of_a_missing_clip_is_404(self, client) -> None:
        assert client.post("/api/clips/nope/variant", json={}).status_code == 404

    def test_hook_cycle_without_options_keeps_the_headline(self) -> None:
        from autoclip.api.clips import next_hook, next_style

        clip = Clip(id="c", job_id="j", start_s=0, end_s=1, title="T")
        assert next_hook(clip) == "" and clip.headline == "T"
        keys = list(captions.PRESETS)
        assert next_style(keys[-1]) == keys[0]
        assert next_style("unknown") == keys[0]


def captions_after(key: str) -> str:
    keys = list(captions.PRESETS)
    return keys[(keys.index(key) + 1) % len(keys)]


class TestVariantZooms:
    def test_each_seed_gets_its_own_pattern(self) -> None:
        words = _speech(20.0)
        patterns = {seed: edit.plan_zooms(words, 20.0, "subtle", (), seed) for seed in range(4)}

        assert patterns[0] == edit.plan_zooms(words, 20.0, "subtle")  # seed 0 unchanged
        assert len({tuple(p) for p in patterns.values()}) == 4
        assert patterns[1][0].z > 1.0  # odd seeds start punched in

    def test_motion_off_variants_get_a_steady_punch_in(self) -> None:
        assert edit.plan_zooms(_speech(10.0), 10.0, "off", (), 0) == []
        assert edit.plan_zooms(_speech(10.0), 10.0, "off", (), 1) == [Zoom(0.0, 1.05)]


# --------------------------------------------------------------------------
# Flash-forward
# --------------------------------------------------------------------------


class TestFlashForward:
    SETTINGS = ExportSettings(pace="off", motion="subtle", keyword_highlight=False)

    def test_plan_plays_the_tease_first_then_the_clip_from_the_start(self) -> None:
        words = _speech(20.0, start=100.0)

        plan = edit.plan_for_clip(
            self.SETTINGS,
            start_s=100.0,
            end_s=120.0,
            words=words,
            title="Hook",
            flash_forward=True,
            teaser=(112.0, 114.0),
        )

        assert plan.teaser == Keep(11.9, 14.1667)  # 0.1 s lead-in, +0.15 s, frame-snapped
        assert plan.keeps == [plan.teaser, Keep(0.0, 20.0)]
        lead = plan.teaser.duration
        assert plan.output_duration(20.0) == pytest.approx(20.0 + lead)
        assert plan.hook_start_s == pytest.approx(lead)
        assert plan.zooms[0] == Zoom(0.0, edit.TEASER_ZOOM)
        assert plan.zooms[1].t == pytest.approx(lead, abs=1e-3)
        assert all(z.t >= lead - 1e-3 for z in plan.zooms[1:])
        assert not plan.intro_zoom and "flash-forward" in plan.summary()

    def test_captions_are_retimed_tease_first(self) -> None:
        words = _speech(20.0, start=100.0)
        plan = edit.plan_for_clip(
            self.SETTINGS,
            start_s=100.0,
            end_s=120.0,
            words=words,
            flash_forward=True,
            teaser=(112.0, 114.0),
        )
        relative = [Word(w.text, w.start - 100.0, w.end - 100.0) for w in words]

        out = edit.retime_words(relative, plan.kept(20.0))

        lead = plan.teaser.duration
        tease = [w for w in out if w.start < lead]
        expected = [w.text for w in relative if 11.9 <= (w.start + w.end) / 2 < plan.teaser.end]
        assert [w.text for w in tease] == expected
        body = out[len(tease) :]
        assert body[0].text == relative[0].text
        assert body[0].start == pytest.approx(lead + relative[0].start, abs=1e-3)
        assert body[-1].end == pytest.approx(lead + relative[-1].end, abs=1e-3)

    def test_the_tease_ends_in_a_measured_pause(self) -> None:
        assert edit.fit_teaser((12.0, 13.5), 20.0, [(13.2, 13.9)]) == Keep(11.9, 13.3)

    def test_unfit_teases_are_refused(self) -> None:
        assert edit.fit_teaser((1.0, 2.5), 20.0) is None  # the opening line
        assert edit.fit_teaser((10.0, 10.3), 20.0) is None  # too short
        assert edit.fit_teaser((10.0, 16.0), 20.0) is None  # too long

    def test_heuristic_pick_when_the_model_flagged_none(self) -> None:
        words = [
            _w("so", 0.2),
            _w("we", 0.7),
            _w("started", 1.2),
            _w("small.", 1.7),
            _w("then", 3.0),
            _w("it", 3.5),
            _w("grew.", 4.0),
            _w("and", 5.5),
            _w("honestly", 6.0),
            _w("nothing", 6.5),
            _w("happened.", 7.0),
            _w("we", 9.0),
            _w("made", 9.5),
            _w("$40,000", 10.0),
            _w("overnight!", 10.5),
            _w("after", 12.5),
            _w("that", 13.0),
            _w("we", 13.5),
            _w("rested.", 14.0),
        ]

        teaser = edit.pick_teaser(words, 15.0)

        assert teaser is not None
        assert teaser.start == pytest.approx(8.9) and teaser.end == pytest.approx(11.05)

    def test_fallback_heuristic_in_the_plan(self) -> None:
        words = _speech(20.0, start=100.0)

        plan = edit.plan_for_clip(
            self.SETTINGS, start_s=100.0, end_s=120.0, words=words, flash_forward=True
        )

        assert plan.teaser is not None and plan.teaser.start >= edit.TEASER_LEAD_S - 0.1

    def test_global_default_and_per_clip_override(self) -> None:
        words = _speech(20.0, start=100.0)
        on = self.SETTINGS.model_copy(update={"flash_forward": True})
        kwargs = {"start_s": 100.0, "end_s": 120.0, "words": words, "teaser": (112.0, 114.0)}

        assert edit.plan_for_clip(self.SETTINGS, **kwargs).teaser is None  # default off
        assert edit.plan_for_clip(on, **kwargs).teaser is not None
        assert edit.plan_for_clip(on, flash_forward=False, **kwargs).teaser is None

    def test_with_pause_cuts_the_tease_comes_before_the_cut_clip(self) -> None:
        words = [w for w in _speech(20.0, start=100.0) if not 104.0 <= w.start < 106.0]
        settings = self.SETTINGS.model_copy(update={"pace": "natural"})

        plan = edit.plan_for_clip(
            settings,
            start_s=100.0,
            end_s=120.0,
            words=words,
            silences=[Silence(104.0, 106.0)],
            flash_forward=True,
            teaser=(112.0, 114.0),
        )

        assert plan.keeps is not None and len(plan.keeps) == 3
        assert plan.keeps[0] == plan.teaser and plan.keeps[1].start == 0.0
        assert "1 pause cuts" in plan.summary()

    def test_export_places_the_hook_after_the_tease(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            export.ffmpeg, "run", lambda args, **kw: Path(args[-1]).write_bytes(b"x")
        )
        monkeypatch.setattr(export.edit_module, "measure_loudnorm", lambda *a, **k: None)
        words = _speech(20.0, start=100.0)
        plan = edit.plan_for_clip(
            ExportSettings(pace="off", motion="off"),
            start_s=100.0,
            end_s=120.0,
            words=words,
            title="The hook",
            flash_forward=True,
            teaser=(112.0, 114.0),
        )
        request = export.ExportRequest(
            source=tmp_path / "src.mp4",
            destination=tmp_path / "out" / "clip.mp4",
            start_s=100.0,
            end_s=120.0,
            crop_path=centre_crop(1920, 1080, 20.0),
            words=words,
            style=captions.get_style("bold_pop"),
            edit=plan,
        )

        graph = export.build_video_filtergraph(request, subtitle_name="c.ass")
        export.export_clip(
            request,
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False),
        )

        assert "[ek0]trim=start=11.9000:end=14.1667" in graph
        assert "[ek1]trim=start=0.0000:end=20.0000" in graph
        ass = (tmp_path / "work" / "clip" / "captions.ass").read_text(encoding="utf-8")
        hook_line = next(
            line
            for line in ass.splitlines()
            if line.startswith("Dialogue:") and ",AutoClipHook," in line
        )
        assert hook_line.startswith("Dialogue: 1,0:00:02.27,0:00:05.47")


# --------------------------------------------------------------------------
# Caption language: romaniser
# --------------------------------------------------------------------------


class TestHinglish:
    @pytest.mark.parametrize(
        ("devanagari", "hinglish"),
        [
            ("क्या बात है", "kya baat hai"),
            ("मैं घर जा रहा हूँ", "main ghar ja raha hoon"),
            ("समझना", "samajhna"),
            ("\u095bिंदगी", "zindagi"),  # precomposed ज़
            ("\u091c\u093cिंदगी", "zindagi"),  # ज + nukta
            ("दोस्त", "dost"),
            ("मित्र", "mitra"),
            ("सत्य", "satya"),
            ("वाला", "wala"),
            ("विश्वास", "vishwas"),
            ("छोड़ दो", "chhod do"),
            ("नहीं", "nahin"),
            ("काम", "kaam"),
            ("बीवी", "biwi"),
            ("मालिक", "malik"),
            ("कपड़े", "kapde"),
            ("अंबानी", "ambani"),
            ("दुःख", "duhkh"),
            ("१२३ रुपये।", "123 rupye."),
            ("hello दुनिया 2026", "hello duniya 2026"),
        ],
    )
    def test_chat_spelling(self, devanagari: str, hinglish: str) -> None:
        assert language.to_hinglish(devanagari) == hinglish

    def test_other_text_is_untouched(self) -> None:
        for text in ("plain English", "தமிழ்", "😀 #tag", ""):
            assert language.to_hinglish(text) == text

    def test_as_spoken_romanises_hindi_but_keeps_marathi(self) -> None:
        words = [_w("क्या", 0.0), _w("बात", 0.5)]

        hindi = language.localise(words, "as_spoken", language="hi", hook="ये सच है?")
        unknown = language.localise(words, "as_spoken", language="")
        marathi = language.localise(words, "as_spoken", language="mr")

        assert [w.text for w in hindi.words] == ["kya", "baat"]
        assert hindi.hook == "ye sach hai?" and hindi.applied == ["hinglish"]
        assert [w.text for w in unknown.words] == ["kya", "baat"]
        assert [w.text for w in marathi.words] == ["क्या", "बात"] and not marathi.applied
        assert [(w.start, w.end) for w in hindi.words] == [(w.start, w.end) for w in words]

    def test_roman_forces_devanagari_to_latin_without_a_model(self) -> None:
        words = [_w("क्या", 0.0)]

        out = language.localise(words, "roman", language="mr")

        assert [w.text for w in out.words] == ["kya"]


# --------------------------------------------------------------------------
# Caption language: the LLM pass
# --------------------------------------------------------------------------


class FakeTranslator(LLMProvider):
    """Answers caption_language prompts with canned translations, line for line."""

    name = "fake"
    requires_key = False

    def __init__(self, table: dict[str, str] | None = None, *, broken: bool = False) -> None:
        super().__init__("fake-model")
        self.table = table or {}
        self.broken = broken
        self.calls: list[tuple[str, str]] = []

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.calls.append((config.prompt_version, user))
        if self.broken:
            return '{"lines": ["one", "two", "three"]}'
        body = user.split("\n\nYour previous")[0]
        payload = json.loads(body[body.rindex('{"lines"') :])
        return json.dumps(
            {"lines": [self.table.get(line, line.upper()) for line in payload["lines"]]}
        )

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


SPANISH = [
    _w("hola", 1.0, 1.4),
    _w("amigos.", 1.5, 2.0),
    _w("esto", 3.0, 3.3),
    _w("es", 3.4, 3.5),
    _w("increíble!", 3.6, 4.4),
]


class TestTranslation:
    def test_caption_lines_split_at_sentences_and_pauses(self) -> None:
        assert language.caption_lines(SPANISH) == [(0, 1), (2, 4)]

    def test_spread_keeps_the_line_span_and_weights_by_length(self) -> None:
        out = language.spread_words("hi there friend", 2.0, 5.0)

        assert out[0].start == 2.0 and out[-1].end == 5.0
        assert [w.text for w in out] == ["hi", "there", "friend"]
        assert out[0].duration < out[2].duration
        assert all(a.end == b.start for a, b in zip(out, out[1:], strict=False))

    def test_english_translates_lines_hook_and_description_keeping_timings(
        self, tmp_path: Path
    ) -> None:
        provider = FakeTranslator(
            {
                "hola amigos.": "hello my friends.",
                "esto es increíble!": "this is amazing!",
                "¡Mira esto!": "Look at this!",
                "Un video genial.": "A great video.",
            }
        )
        converter = language.provider_converter(provider, language="es", cache_dir=tmp_path)

        out = language.localise(
            SPANISH,
            "english",
            language="es",
            hook="¡Mira esto!",
            description="Un video genial.",
            converter=converter,
        )

        assert [w.text for w in out.words] == ["hello", "my", "friends.", "this", "is", "amazing!"]
        assert out.words[0].start == 1.0 and out.words[2].end == 2.0
        assert out.words[3].start == 3.0 and out.words[-1].end == 4.4
        assert out.hook == "Look at this!" and out.description == "A great video."
        assert out.applied == ["translated"]
        ((version, user),) = provider.calls
        assert version == language.PROMPT_VERSION and "Return exactly 4 lines" in user

        again = language.localise(
            SPANISH,
            "english",
            language="es",
            hook="¡Mira esto!",
            description="Un video genial.",
            converter=converter,
        )
        assert len(provider.calls) == 1, "the second export reuses the cached answer"
        assert [w.text for w in again.words] == [w.text for w in out.words]

    def test_english_source_is_never_sent(self) -> None:
        provider = FakeTranslator()
        converter = language.provider_converter(provider)

        out = language.localise([_w("hello", 0.0)], "english", language="en", converter=converter)
        ascii_unknown = language.localise([_w("hello", 0.0)], "english", converter=converter)

        assert out.words[0].text == "hello" and ascii_unknown.words[0].text == "hello"
        assert provider.calls == []

    def test_a_broken_reply_falls_back_to_the_spoken_words(self) -> None:
        provider = FakeTranslator(broken=True)
        converter = language.provider_converter(provider)
        hindi = [_w("क्या", 0.0), _w("बात", 0.5)]

        out = language.localise(hindi, "english", language="hi", converter=converter)

        assert [w.text for w in out.words] == ["kya", "baat"]  # rule-based Hinglish
        assert len(provider.calls) == 2  # one repair retry

    def test_roman_sends_only_non_latin_lines_to_the_model(self) -> None:
        provider = FakeTranslator({"வணக்கம் நண்பர்களே.": "vanakkam nanbargale."})
        converter = language.provider_converter(provider)
        words = [_w("வணக்கம்", 0.0), _w("நண்பர்களே.", 0.5), _w("okay", 2.0), _w("bye.", 2.4)]

        out = language.localise(words, "roman", language="ta", converter=converter)

        assert [w.text for w in out.words] == ["vanakkam", "nanbargale.", "okay", "bye."]
        assert out.words[2:] == words[2:]
        ((_, user),) = provider.calls
        assert "okay" not in user and "Do NOT translate" in user

    def test_runs_inside_a_running_event_loop(self) -> None:
        provider = FakeTranslator({"hola amigos.": "hello friends."})
        converter = language.provider_converter(provider)

        async def inside() -> list[str]:
            return converter(["hola amigos."], "english")

        assert asyncio.run(inside()) == ["hello friends."]

    def test_api_export_translates_the_burned_captions(self, client, monkeypatch) -> None:
        from autoclip import providers
        from autoclip.pipeline.runner import JobWorkspace

        rendered = _capture_exports(monkeypatch)
        provider = FakeTranslator({"Hook A": "Hook in English"})
        monkeypatch.setattr(providers, "build_provider", lambda *a, **k: provider)
        job, clip = _job_with_clip(start_s=0.0, end_s=6.0, teaser_start_s=None, teaser_end_s=None)
        Transcript(words=SPANISH, language="es").save(JobWorkspace(job.id).transcript)
        store.update_clip(clip.id, end_word=len(SPANISH) - 1, start_word=0)
        client.put("/api/settings", json={"export": {"caption_language": "english"}})

        response = client.post(f"/api/clips/{clip.id}/export", json={"ratio": "9:16"})

        assert response.status_code == 201
        (request,) = rendered
        assert [w.text for w in request.words] == ["HOLA", "AMIGOS.", "ESTO", "ES", "INCREÍBLE!"]
        assert request.edit is not None and request.edit.hook_text == "Hook in English"
        assert store.get_clip(clip.id).description == "DESC"
        assert request.words[-1].end == SPANISH[-1].end


# --------------------------------------------------------------------------
# Real render
# --------------------------------------------------------------------------


@pytest.mark.slow
class TestFlashForwardRender:
    def test_tease_then_cut_clip_renders_at_the_planned_length(self, tmp_path: Path) -> None:
        import subprocess

        from autoclip.pipeline import ffmpeg, prepare

        ff = ffmpeg.ffmpeg_path()
        src = tmp_path / "src.mp4"
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=12",
             "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
             "-af", "volume='if(between(t,3.0,4.5),0,1)':eval=frame",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)],
            check=True,
        )  # fmt: skip
        settings = ExportSettings(prefer_hardware_encoder=False, motion="subtle")
        words = [
            _w(f"word{i}", t)
            for i, t in enumerate([0.2, 0.8, 1.4, 2.0, 2.5, 4.7, 5.3, 6.0, 7.0, 8.0, 9.0, 10.0])
        ]
        plan = edit.plan_for_clip(
            settings,
            start_s=0.0,
            end_s=12.0,
            words=words,
            silences=prepare.detect_silences(src),
            title="A hook",
            flash_forward=True,
            teaser=(8.0, 9.4),
        )
        assert plan.teaser is not None and plan.keeps is not None and len(plan.keeps) == 3
        dest = tmp_path / "out" / "ff.mp4"

        export.export_clip(
            export.ExportRequest(
                source=src,
                destination=dest,
                start_s=0.0,
                end_s=12.0,
                crop_path=centre_crop(1280, 720, 12.0),
                words=words,
                style=captions.get_style("bold_pop"),
                edit=plan,
            ),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        info = ffmpeg.probe(dest)
        assert info.has_audio and info.height == 1920
        assert info.duration_s == pytest.approx(plan.output_duration(12.0), abs=0.1)
        assert info.duration_s > 12.0  # the tease plays on top of the (cut) clip
