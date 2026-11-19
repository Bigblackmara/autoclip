"""GTA Mode, wave 3: flash-forward, hook options, variants, caption language,
picks on the Hype Curve."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pysubs2
import pytest
from autoclip import app as app_module
from autoclip import db
from autoclip.app import create_app
from autoclip.config import ExportSettings
from autoclip.db.models import new_id
from autoclip.gta import edits, hype, judging, pipeline, qa, render, store
from autoclip.gta import poster as poster_mod
from autoclip.gta import settings as gta_settings
from autoclip.gta.settings import GtaPosting, GtaSettings, parse_section
from autoclip.pipeline import ffmpeg
from autoclip.pipeline.transcript import Transcript, Word
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _db(initialised_db: int) -> None:
    """Every test runs against a migrated temp database."""


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def _words(text: str, start: float, step: float = 0.5) -> list[Word]:
    out, t = [], start
    for token in text.split():
        out.append(Word(text=token, start=t, end=t + step * 0.85))
        t += step
    return out


class FakeModel:
    """A scripted provider: returns queued responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    async def __call__(self, system: str, user: str, config) -> str:
        self.prompts.append((system, user))
        return self.responses.pop(0) if self.responses else '{"verdicts": []}'


def _verdict(cid: str, **extra) -> dict:
    return {
        "id": cid,
        "scroll_stop": {"answer": "yes", "why": "scream"},
        "context_free": {"answer": "yes", "why": "clear"},
        "payoff": {"answer": "yes", "why": "lands"},
        "score": 80,
        "hook_titles": ["He robbed the WRONG bank", "POV: the vault explodes", "Is this legal?"],
        "peak_quote": "NO WAY he blew up the vault",
        "reaction_text": "HE DID NOT SEE THAT",
        "caption": "The heist went sideways.",
        "hashtags": ["#gta", "#heist", "#shorts"],
        "moment_type": "freakout",
        **extra,
    }


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [ffmpeg.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error", *args], check=True
    )


def _make_section(path: Path, duration: float = 22.0) -> None:
    """16:9 video; a quiet tone that gets LOUD at 14-16 s (the peak)."""
    _ffmpeg(
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={duration}",
        "-af", "volume=volume='if(between(t,14,16),1.0,0.12)':eval=frame",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    )  # fmt: skip


def _speech(offset: float = 0.0) -> list[Word]:
    words = (
        _words("okay we go in through the back of the bank now", 1.0)
        + _words("NO WAY he blew up the vault", 14.0, 0.3)
        + _words("chat that was absolutely insane", 17.0)
    )
    return [Word(w.text, w.start + offset, w.end + offset) for w in words]


# --------------------------------------------------------------------------
# Settings + migration v8 + store
# --------------------------------------------------------------------------


class TestSettingsAndStore:
    def test_defaults_and_validation(self) -> None:
        s = GtaSettings()
        assert s.flash_forward is False and s.flash_forward_s == 2.0
        assert s.caption_language == "as_spoken"
        with pytest.raises(ValueError):
            GtaSettings(flash_forward_s=5.0)
        with pytest.raises(ValueError):
            GtaSettings(caption_language="klingon")
        assert parse_section({"caption_language": "klingon"}).caption_language == "as_spoken"

    def test_settings_api_roundtrip(self, client: TestClient) -> None:
        saved = client.put(
            "/api/gta/settings", json={"flash_forward": True, "caption_language": "english"}
        )
        assert saved.status_code == 200, saved.text
        assert saved.json()["flash_forward"] is True
        assert saved.json()["caption_language"] == "english"
        bad = client.put("/api/gta/settings", json={"caption_language": "nope"})
        assert bad.status_code == 400

    def test_migration_v8_columns(self) -> None:
        assert db.SCHEMA_VERSION == 8
        with db.connection() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(gta_clips)").fetchall()}
        assert {
            "hook_options",
            "teaser_start_s",
            "teaser_end_s",
            "flash_forward",
            "variant_of",
            "motion_seed",
            "caption_style",
            "localised",
        } <= cols

    def test_new_clip_fields_roundtrip(self) -> None:
        stream = store.create_stream("https://youtu.be/abc")
        clip = store.create_clip(
            store.GtaClip(
                id="c1",
                stream_id=stream.id,
                start_s=1.0,
                end_s=20.0,
                hook_options=["a", "b", "c"],
                teaser_start_s=12.0,
                teaser_end_s=14.0,
                flash_forward=False,
                caption_style="neon",
                localised={"caption": "x"},
            )
        )
        got = store.get_clip(clip.id)
        assert got is not None and got.hook_options == ["a", "b", "c"]
        assert got.flash_forward is False and got.teaser_end_s == 14.0
        assert got.family_id == "c1" and got.caption_style == "neon"
        store.update_clip("c1", flash_forward=None, variant_of="root", motion_seed=2)
        got = store.get_clip("c1")
        assert got.flash_forward is None and got.family_id == "root" and got.motion_seed == 2


# --------------------------------------------------------------------------
# Judge v2: three hooks + the peak line
# --------------------------------------------------------------------------


class TestJudgeV2:
    def _candidate(self, transcript: Transcript) -> judging.Candidate:
        first, last = transcript.indices_in_range(0.0, 20.0)
        return judging.Candidate("c0", 0, first, last, 0.5, 20.0, "okay", "other", "", 60)

    def test_final_round_keeps_three_hooks_and_anchors_the_peak(self) -> None:
        transcript = Transcript(words=_speech())
        candidate = self._candidate(transcript)
        model = FakeModel([json.dumps({"verdicts": [_verdict("c0")]})])
        asyncio.run(
            judging.final_round(
                transcript,
                [candidate],
                complete=model,
                settings=GtaSettings(min_clip_s=5, max_clip_s=30),
                streamer="S",
                title="T",
            )
        )
        assert judging.FINAL_PROMPT == "gta_final_v2"
        assert "hook_titles" in model.prompts[0][0] and "peak_quote" in model.prompts[0][0]
        assert candidate.status == "approved"
        assert candidate.hook_options == [
            "He robbed the WRONG bank",
            "POV: the vault explodes",
            "Is this legal?",
        ]
        assert candidate.hook_title == "He robbed the WRONG bank"
        assert candidate.teaser_start_s == pytest.approx(14.0)
        assert candidate.teaser_end_s == pytest.approx(16.055)

    def test_v1_answer_still_parses(self) -> None:
        transcript = Transcript(words=_speech())
        candidate = self._candidate(transcript)
        old = _verdict("c0", hook_title="One title only")
        del old["hook_titles"], old["peak_quote"]
        asyncio.run(
            judging.final_round(
                transcript,
                [candidate],
                complete=FakeModel([json.dumps({"verdicts": [old]})]),
                settings=GtaSettings(min_clip_s=5, max_clip_s=30),
                streamer="",
                title="",
            )
        )
        assert candidate.hook_options == ["One title only"]
        assert candidate.teaser_start_s is None

    def test_describe_never_rejects(self) -> None:
        transcript = Transcript(words=_speech())
        candidate = self._candidate(transcript)
        bad = _verdict("c0", payoff={"answer": "no", "why": "meh"})
        asyncio.run(
            judging.describe(
                transcript,
                [candidate],
                complete=FakeModel([json.dumps({"verdicts": [bad]})]),
                settings=GtaSettings(),
                streamer="S",
                title="T",
            )
        )
        assert candidate.status == "candidate" and candidate.hook_title
        assert candidate.caption == "The heist went sideways." and candidate.score == 80

    def test_find_quote_skips_the_opening(self) -> None:
        transcript = Transcript(words=_words("wow wow okay then later wow wow", 0.0))
        # "wow wow" at 0.0 is the opening (no tease); the one at 2.5 s is used.
        assert judging.find_quote(transcript, 0.0, 10.0, "Wow, wow!") == pytest.approx((2.5, 3.425))
        assert judging.find_quote(transcript, 0.0, 10.0, "nothing like it") is None
        assert judging.find_quote(transcript, 0.0, 10.0, "") is None


# --------------------------------------------------------------------------
# Flash-forward planning + render
# --------------------------------------------------------------------------


class TestFlashForwardPlan:
    def _levels(self, seconds: float, loud: tuple[float, float]) -> np.ndarray:
        hop = 0.05
        n = int(seconds / hop)
        levels = np.full(n, -40.0, dtype=np.float32)
        a, b = int(loud[0] / hop), int(loud[1] / hop)
        levels[a:b] = -6.0
        return levels

    def test_quoted_line_wins(self) -> None:
        words = _speech()
        keep = render.plan_teaser(
            words, 0.5, 20.0, levels=None, length_s=2.0, span=(14.0, 15.6), peak_s=3.0
        )
        assert keep is not None and keep.start <= 14.0 and keep.end >= 15.6

    def test_hype_peak_refined_to_the_loudest_beat(self) -> None:
        levels = self._levels(20.0, (14.0, 16.0))
        keep = render.plan_teaser(
            _speech(), 0.0, 20.0, levels=levels, length_s=2.0, span=None, peak_s=12.5
        )
        assert keep is not None
        assert keep.start < 15.0 < keep.end  # pulled onto the loud beat
        assert render.TEASER_MIN_S <= keep.end - keep.start <= render.TEASER_MAX_S

    def test_never_the_opening_and_never_on_a_short_clip(self) -> None:
        levels = self._levels(20.0, (0.0, 1.5))
        keep = render.plan_teaser([], 0.0, 20.0, levels=levels, length_s=2.0)
        # The loudest beat is the opening itself: the tease moves past it.
        assert keep is None or keep.start >= render.TEASER_LEAD_S
        assert render.plan_teaser([], 0.0, 5.0, levels=levels, length_s=2.0) is None

    def test_filtergraph_punches_in_on_several_spans(self) -> None:
        graph = render.build_filtergraph(
            [render.Keep(10.0, 12.0), render.Keep(0.0, 20.0)],
            seek_s=0.0,
            subtitle_name=None,
            fonts_name="fonts",
            audio_filter="anull",
            punch_in=[(0.0, 2.0, 1.3), (8.0, 9.5, 1.3)],
        )
        assert "between(t,0.000,2.000)+between(t,8.000,9.500)" in graph
        assert "concat=n=2" in graph

    def test_hook_title_waits_for_the_tease(self, tmp_path: Path) -> None:
        path = render.write_subtitles(
            tmp_path / "c.ass",
            _words("hello there", 0.2),
            style_key="gta",
            hook_title="Big hook",
            title_s=3.0,
            duration_s=20.0,
            title_start_s=2.0,
        )
        hook = [e for e in pysubs2.load(str(path)).events if e.style == render.TITLE_STYLE]
        assert hook and hook[0].start == 2000 and hook[0].end == 5000


class TestFlashForwardRender:
    def _render(self, tmp_path: Path, *, max_s: float = 58.0, flash: bool = True):
        section = tmp_path / "section.mp4"
        if not section.exists():
            _make_section(section)
        destination = tmp_path / f"out-{flash}-{max_s}.mp4"
        result = render.render(
            render.RenderJob(
                section=section,
                section_offset_s=100.0,
                start_s=100.5,
                end_s=120.0,
                words=_speech(100.0),
                hook_title="He robbed the WRONG bank",
                destination=destination,
                work_dir=tmp_path / f"work-{flash}-{max_s}",
                extras=render.Extras(flash_forward=flash, flash_forward_s=2.0, motion="subtle"),
                peak_s=115.0,
            ),
            style_key="gta",
            title_s=3.0,
            dead_air_min_s=0.6,
            dead_air_keep_s=0.12,
            max_duration_s=max_s,
            loudness_lufs=-14.0,
            true_peak_db=-1.0,
            export_settings=ExportSettings(prefer_hardware_encoder=False),
        )
        return section, destination, result

    def test_opens_on_the_peak_then_plays_from_the_start(self, tmp_path: Path) -> None:
        section, destination, result = self._render(tmp_path)
        assert "flash-forward" in result.edits and "punch-in" in result.edits
        assert 1.0 <= result.teaser_s <= 3.6
        tease = result.keeps[0]
        assert tease.start < 15.0 < tease.end  # section-local: the loud 14-16 s beat
        assert result.keeps[1].start == pytest.approx(0.5, abs=0.05)  # then from the start
        info = ffmpeg.probe(destination)
        assert info.duration_s == pytest.approx(result.duration_s, abs=0.25)
        ass = pysubs2.load(str(tmp_path / "work-True-58.0" / "captions.ass"))
        hook = [e for e in ass.events if e.style == render.TITLE_STYLE]
        assert hook[0].start == pytest.approx(result.teaser_s * 1000, abs=10)  # ASS: centiseconds
        report = qa.run_qa(
            destination,
            section=section,
            section_start_s=0.5,
            section_end_s=20.0,
            target_lufs=-14.0,
            tolerance_lu=1.0,
            true_peak_db=-1.0,
            face_min_ratio=0.5,
            face_ratio_fn=lambda *_: 1.0,
            edits=result.edits,
        )
        assert report.passed, report.to_dict()

    def test_tease_counts_towards_the_length_cap(self, tmp_path: Path) -> None:
        _, destination, result = self._render(tmp_path, max_s=12.0)
        assert result.teaser_s > 0
        assert result.duration_s <= 12.0 + 1e-6
        assert ffmpeg.probe(destination).duration_s < 12.4


# --------------------------------------------------------------------------
# stage_render: flash-forward from the Hype Curve, caption language, variants
# --------------------------------------------------------------------------


def _vod_with_section(tmp_path: Path, *, status: str = "done") -> tuple[store.GtaStream, Path]:
    """A finished VOD stream with its transcript, Hype Curve and one section."""
    stream = store.create_stream("https://youtu.be/abcdefghijk")
    store.update_stream(
        stream.id, status=status, duration_s=600.0, channel="Streamer", platform="youtube"
    )
    ws = pipeline.Workspace(stream.id)
    Transcript(words=_speech(100.0), language="en").save(ws.transcript)
    score = np.zeros(600, dtype=np.float32)
    score[114:117] = 90.0
    hype.HypeCurve(
        score=score, markers=[hype.Marker(t=115.0, score=90.0, reasons=["scream"])], available=[]
    ).save(ws.hype)
    return store.get_stream(stream.id), ws.sections_dir


def _ready_clip(stream: store.GtaStream, **extra) -> store.GtaClip:
    clip = store.GtaClip(
        id=new_id(),
        stream_id=stream.id,
        start_s=100.5,
        end_s=120.0,
        status="ready",
        hook_title="He robbed the WRONG bank",
        hook_options=["He robbed the WRONG bank", "POV: the vault explodes", "Is this legal?"],
        score=80,
        caption="The heist went sideways.",
        hashtags=["#gta"],
        qa={"passed": True},
        **extra,
    )
    return store.create_clip(clip)


class TestStageRender:
    def test_flash_forward_from_the_hype_curve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream, sections = _vod_with_section(tmp_path)
        clip = _ready_clip(stream, flash_forward=True)
        sections.mkdir(parents=True, exist_ok=True)
        _make_section(sections / f"{clip.id}.mp4")
        (sections / f"{clip.id}.offset.json").write_text(
            json.dumps({"section": f"{clip.id}.mp4", "start_s": 100.0}), encoding="utf-8"
        )
        monkeypatch.setattr(qa, "face_ratio", lambda *a, **k: 0.9)
        monkeypatch.setattr(pipeline, "load_settings", lambda: _app_settings())
        monkeypatch.setattr(pipeline, "output_root", lambda: tmp_path / "exports")
        out = pipeline.stage_render(clip, stream, _reporter(stream), GtaSettings())
        assert out["qa"]["passed"], out["qa"]
        assert 1.0 <= out["qa"]["flash_forward_s"] <= 3.6
        assert (
            Path(out["path"])
            .with_suffix(".txt")
            .read_text(encoding="utf-8")
            .startswith("The heist went sideways.")
        )

    def test_variant_shares_the_section_and_renders_its_own_look(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream, sections = _vod_with_section(tmp_path)
        original = _ready_clip(stream)
        sections.mkdir(parents=True, exist_ok=True)
        _make_section(sections / f"{original.id}.mp4")
        (sections / f"{original.id}.offset.json").write_text(
            json.dumps({"section": f"{original.id}.mp4", "start_s": 100.0}), encoding="utf-8"
        )
        variant = edits.make_variant(original.id, GtaSettings())
        assert pipeline.section_stem(variant) == original.id
        # The section stage finds the shared file instead of downloading.
        seen: list[str] = []
        monkeypatch.setattr(
            pipeline.fetch,
            "download_section",
            lambda url, directory, stem, *a, **k: seen.append(stem),
        )
        pipeline.stage_section(variant, stream, _reporter(stream), GtaSettings())
        assert seen == [original.id]
        monkeypatch.setattr(qa, "face_ratio", lambda *a, **k: 0.9)
        monkeypatch.setattr(pipeline, "load_settings", lambda: _app_settings())
        monkeypatch.setattr(pipeline, "output_root", lambda: tmp_path / "exports")
        # Hook drawn by the subtitles (no static caption box), so it shows in the .ass.
        out = pipeline.stage_render(
            variant, stream, _reporter(stream), GtaSettings(static_caption=False)
        )
        assert out["qa"]["passed"], out["qa"]
        ass = (pipeline.Workspace(stream.id).render_dir(variant.id) / "captions.ass").read_text(
            encoding="utf-8"
        )
        assert "POV: THE VAULT EXPLODES" in ass  # the next hook option

    def test_caption_language_roman_rewrites_hindi(self, tmp_path: Path) -> None:
        stream, _ = _vod_with_section(tmp_path)
        clip = _ready_clip(stream)
        store.update_clip(clip.id, hook_title="क्या बात है", caption="क्या बात है भाई")
        clip = store.get_clip(clip.id)
        words = _words("क्या बात है", 101.0)
        display, hook = pipeline.localise_clip(
            clip,
            words,
            GtaSettings(caption_language="as_spoken"),
            language="hi",
            cache_dir=tmp_path / "lang",
        )
        assert display is not None and all(w.text.isascii() for w in display)
        assert hook.isascii() and hook
        fresh = store.get_clip(clip.id)
        assert fresh.localised["mode"] == "as_spoken" and fresh.localised["caption"].isascii()
        assert pipeline.post_text(fresh).startswith(fresh.localised["caption"])
        assert poster_mod.post_title(fresh) == hook
        assert poster_mod.instagram_caption(fresh).startswith(fresh.localised["caption"])

    def test_english_without_a_provider_never_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream, _ = _vod_with_section(tmp_path)
        clip = _ready_clip(stream)
        display, hook = pipeline.localise_clip(
            clip,
            _speech(100.0),
            GtaSettings(caption_language="english"),
            language="en",
            cache_dir=tmp_path / "lang",
        )
        assert display is None and hook == clip.hook_title


def _app_settings():
    from autoclip.config import Settings

    settings = Settings()
    settings.export.prefer_hardware_encoder = False
    return settings


def _reporter(stream: store.GtaStream) -> pipeline.Reporter:
    return pipeline.Reporter(stream.id, lambda: False)


# --------------------------------------------------------------------------
# Variants + hook choice (API)
# --------------------------------------------------------------------------


class TestVariantsAndHooks:
    def test_variant_rows_cycle_looks_hooks_and_seeds(self, tmp_path: Path) -> None:
        stream, _ = _vod_with_section(tmp_path)
        original = _ready_clip(stream)
        first = edits.make_variant(original.id, GtaSettings())
        assert first.variant_of == original.id and first.status == "approved"
        assert first.hook_title == "POV: the vault explodes"
        assert first.caption_style == "pop" and first.motion_seed == 1
        assert store.get_stream(stream.id).status == "analyzed"  # the queue picks it up
        store.update_clip(first.id, status="ready")
        # A variant of a variant still belongs to the original's family.
        second = edits.make_variant(first.id, GtaSettings())
        assert second.variant_of == original.id
        assert second.hook_title == "Is this legal?"
        assert second.caption_style == "neon" and second.motion_seed == 2
        assert [c.id for c in edits.family(original)] == [original.id, first.id, second.id]

    def test_variant_refused_for_unfinished_and_live_clips(self, tmp_path: Path) -> None:
        stream, _ = _vod_with_section(tmp_path)
        pending = _ready_clip(stream)
        store.update_clip(pending.id, status="rendering")
        with pytest.raises(edits.EditError):
            edits.make_variant(pending.id)
        live = store.create_stream("https://www.twitch.tv/somebody")
        store.update_stream(live.id, status="live_watching", signals={"mode": "live"})
        clip = _ready_clip(store.get_stream(live.id))
        with pytest.raises(edits.EditError):
            edits.make_variant(clip.id)

    def test_api_pick_hook_flash_and_variant(self, client: TestClient, tmp_path: Path) -> None:
        stream, _ = _vod_with_section(tmp_path)
        clip = _ready_clip(stream)
        listed = client.get(f"/api/gta/streams/{stream.id}").json()["clips"][0]
        assert listed["hook_options"][2] == "Is this legal?" and listed["source"] == "judge"

        changed = client.patch(
            f"/api/gta/clips/{clip.id}",
            json={"hook_title": "Is this legal?", "flash_forward": True},
        )
        assert changed.status_code == 200, changed.text
        body = changed.json()
        assert body["hook_title"] == "Is this legal?" and body["flash_forward"] is True
        assert body["status"] == "rendering"
        # Busy now: another edit waits for the render.
        assert (
            client.patch(f"/api/gta/clips/{clip.id}", json={"flash_forward": None}).status_code
            == 409
        )
        store.update_clip(clip.id, status="ready")
        back = client.patch(f"/api/gta/clips/{clip.id}", json={"flash_forward": None})
        assert back.status_code == 200 and back.json()["flash_forward"] is None
        assert client.patch(f"/api/gta/clips/{clip.id}", json={}).status_code == 400
        store.update_clip(clip.id, status="ready")

        made = client.post(f"/api/gta/clips/{clip.id}/variant")
        assert made.status_code == 201, made.text
        assert made.json()["variant_of"] == clip.id and made.json()["source"] == "variant"
        assert client.post("/api/gta/clips/nope/variant").status_code == 404


# --------------------------------------------------------------------------
# Poster: a variant never shares a day with its family on one platform
# --------------------------------------------------------------------------


def _postable(tmp_path: Path, stream: store.GtaStream, **extra) -> store.GtaClip:
    path = tmp_path / f"{new_id()}.mp4"
    path.write_bytes(b"\x00" * 64)
    clip = store.GtaClip(
        id=new_id(),
        stream_id=stream.id,
        start_s=10.0,
        end_s=35.0,
        status="ready",
        hook_title="Title",
        path=str(path),
        qa={"passed": True},
        **extra,
    )
    return store.create_clip(clip)


class TestPosterFamilies:
    def test_variant_waits_for_another_day_on_the_same_platform(self, tmp_path: Path) -> None:
        stream = store.create_stream("https://youtu.be/x")
        original = _postable(tmp_path, stream, score=90)
        variant = _postable(tmp_path, stream, score=85, variant_of=original.id)
        other = _postable(tmp_path, stream, score=10)
        now = datetime(2026, 11, 18, 8, 0, tzinfo=UTC)
        noon = now.replace(hour=12)
        store.create_upload(
            original.id, status="scheduled", platform="youtube", slot_at=poster_mod.slot_key(noon)
        )
        evening = now.replace(hour=17)
        ids = [c.id for c, _ in poster_mod.eligible_clips(now, "youtube", slot=evening, tz=UTC)]
        assert variant.id not in ids and other.id in ids
        tomorrow = evening + timedelta(days=1)
        ids = [c.id for c, _ in poster_mod.eligible_clips(now, "youtube", slot=tomorrow, tz=UTC)]
        assert ids[0] == variant.id
        # Instagram is another platform: the variant may go there today.
        ids = [c.id for c, _ in poster_mod.eligible_clips(now, "instagram", slot=evening, tz=UTC)]
        assert ids[0] == original.id and variant.id in ids

    def test_plan_never_puts_a_family_twice_on_one_day(self, tmp_path: Path) -> None:
        stream = store.create_stream("https://youtu.be/y")
        original = _postable(tmp_path, stream, score=90)
        variant = _postable(tmp_path, stream, score=85, variant_of=original.id)
        other = _postable(tmp_path, stream, score=10)
        gta_settings.save(GtaSettings(posting=GtaPosting(enabled=True, slots=["12:00", "17:00"])))
        now = datetime(2026, 11, 18, 8, 0, tzinfo=UTC)
        p = poster_mod.GtaPoster(clock=lambda: now, ready_check=lambda: "", tz=UTC)
        plan = [e for e in p.plan(gta_settings.load(), now) if e["platform"] == "youtube"]
        by_day: dict[str, list[str]] = {}
        for entry in plan:
            if entry["clip_id"]:
                by_day.setdefault(entry["slot_at"][:10], []).append(entry["clip_id"])
        assert by_day["2026-11-18"] == [original.id, other.id]
        assert by_day["2026-11-19"][0] == variant.id


# --------------------------------------------------------------------------
# Picks on the Hype Curve
# --------------------------------------------------------------------------


class TestPicks:
    def test_hype_graph(self, client: TestClient, tmp_path: Path) -> None:
        stream, _ = _vod_with_section(tmp_path)
        _ready_clip(stream)
        graph = client.get(f"/api/gta/streams/{stream.id}/hype").json()
        assert graph["problem"] == "" and graph["duration_s"] == 600.0
        assert len(graph["curve"]) == 600 and max(graph["curve"]) == pytest.approx(0.9)
        assert graph["hotspots"][0]["peak_s"] == 115.0
        assert graph["clips"][0]["status"] == "ready"
        assert client.get("/api/gta/streams/nope/hype").status_code == 404

    def test_picks_with_the_judge_prompt(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream, _ = _vod_with_section(tmp_path)
        model = FakeModel(
            [json.dumps({"verdicts": [_verdict("p0", payoff={"answer": "no", "why": "x"})]})]
        )
        monkeypatch.setattr(pipeline, "_provider_complete", lambda: model)
        made = client.post(
            f"/api/gta/streams/{stream.id}/picks",
            json={"ranges": [{"start_s": 101.2, "end_s": 119.9}]},
        )
        assert made.status_code == 201, made.text
        (clip,) = made.json()["clips"]
        assert clip["status"] == "approved" and clip["source"] == "pick"
        assert clip["hook_title"] == "He robbed the WRONG bank" and len(clip["hook_options"]) == 3
        assert clip["start_s"] == pytest.approx(101.0)  # widened to the whole word
        assert clip["teaser_start_s"] == pytest.approx(114.0)
        assert "hook_titles" in model.prompts[0][0]
        assert store.get_stream(stream.id).status == "analyzed"

    def test_picks_without_a_provider_get_plain_titles(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream, _ = _vod_with_section(tmp_path, status="failed")

        def no_provider():
            raise RuntimeError("no API key")

        monkeypatch.setattr(pipeline, "_provider_complete", no_provider)
        made = client.post(
            f"/api/gta/streams/{stream.id}/picks",
            json={"ranges": [{"start_s": 100.0, "end_s": 112.0}]},
        )
        assert made.status_code == 201, made.text
        (clip,) = made.json()["clips"]
        assert clip["hook_title"] == "okay we go in through the"
        assert store.get_stream(stream.id).status == "analyzed"

    @pytest.mark.parametrize(
        ("ranges", "needle"),
        [
            ([(100.0, 102.0)], "at least"),
            ([(100.0, 200.0)], "too long"),
            ([(100.0, 120.0), (110.0, 130.0)], "overlap"),
        ],
    )
    def test_bad_ranges(self, client: TestClient, tmp_path: Path, ranges, needle) -> None:
        stream, _ = _vod_with_section(tmp_path)
        bad = client.post(
            f"/api/gta/streams/{stream.id}/picks",
            json={"ranges": [{"start_s": a, "end_s": b} for a, b in ranges]},
        )
        assert bad.status_code == 400 and needle in bad.json()["detail"]

    def test_running_and_live_streams_refuse_picks(self, client: TestClient, tmp_path) -> None:
        stream, _ = _vod_with_section(tmp_path, status="judging")
        r = client.post(
            f"/api/gta/streams/{stream.id}/picks",
            json={"ranges": [{"start_s": 100.0, "end_s": 112.0}]},
        )
        assert r.status_code == 409 and "still being analysed" in r.json()["detail"]
