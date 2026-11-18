"""GTA Mode: key words, caption looks, frame motion, comment signal, captions-first."""

from __future__ import annotations

import json
import re
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pysubs2
import pytest
from autoclip import app as app_module
from autoclip.app import create_app
from autoclip.gta import captions as gta_captions
from autoclip.gta import fetch, hype, pipeline, render, store
from autoclip.gta.settings import CAPTION_STYLES, GtaSettings, parse_section
from autoclip.pipeline import captions
from autoclip.pipeline.signals import Comment
from autoclip.pipeline.transcript import Transcript, Word
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _db(initialised_db: int) -> None:
    """Every GTA test runs against a migrated temp database."""


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def _words(text: str, start: float, step: float = 0.4) -> list[Word]:
    out = []
    t = start
    for token in text.split():
        out.append(Word(text=token, start=t, end=t + step * 0.8))
        t += step
    return out


SPEECH = (
    "bro the police helicopter just landed on my roof and stole the lamborghini "
    "nobody believes this but forty seven cops chased me across vinewood hills "
    "then the mechanic exploded the garage with twenty grenades absolutely insane"
)

#: Bottom of the uncropped 16:9 stream frame in the 1080x1920 canvas.
FRAME_BOTTOM_16x9 = (1920 + 608) // 2


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self) -> None:
        s = GtaSettings()
        assert s.caption_style == "gta" and s.keyword_highlight is True
        assert s.motion == "subtle" and s.comment_signal is True
        assert s.youtube_captions is False

    def test_legacy_style_and_validation(self) -> None:
        assert GtaSettings(caption_style="bold_pop").caption_style == "gta"
        assert GtaSettings(caption_style="NEON").caption_style == "neon"
        assert GtaSettings(caption_style="karaoke_fill").caption_style == "karaoke_fill"
        with pytest.raises(ValueError):
            GtaSettings(caption_style="boxed")
        with pytest.raises(ValueError):
            GtaSettings(motion="wild")
        # A bad saved value never breaks startup: defaults instead.
        assert parse_section({"motion": "wild"}).motion == "subtle"

    def test_api_roundtrip(self, client) -> None:
        saved = client.put(
            "/api/gta/settings",
            json={"caption_style": "comic", "motion": "dynamic", "youtube_captions": True},
        ).json()
        assert saved["caption_style"] == "comic" and saved["motion"] == "dynamic"
        assert saved["youtube_captions"] is True and saved["keyword_highlight"] is True
        assert client.put("/api/gta/settings", json={"motion": "x"}).status_code == 400


# --------------------------------------------------------------------------
# Captions: key words + looks
# --------------------------------------------------------------------------


def _caption_events(subs: pysubs2.SSAFile) -> list[pysubs2.SSAEvent]:
    return [e for e in subs.events if e.style == captions.STYLE_NAME]


class TestCaptions:
    def test_keywords_are_picked_and_drawn(self, tmp_path: Path) -> None:
        words = _words(SPEECH, 0.0)
        keywords = gta_captions.clip_keywords(words, True)
        assert 3 <= len(keywords) <= 8
        assert gta_captions.clip_keywords(words, False) == []
        path = render.write_subtitles(
            tmp_path / "c.ass",
            words,
            style_key="gta",
            hook_title="Hook",
            title_s=3.0,
            duration_s=words[-1].end,
            keywords=keywords,
        )
        style = captions.get_style("bold_pop")
        emphasis = captions.ass_colour_override(style.emphasis)
        events = _caption_events(pysubs2.load(str(path)))
        marked = [e for e in events if emphasis in e.text]
        assert marked, "no key word was emphasised"
        for event in events:
            # At most one key word per caption group (so per line).
            assert event.text.count(emphasis) <= 1

    @pytest.mark.parametrize("look", sorted(CAPTION_STYLES))
    def test_every_look_stays_in_the_bottom_band(self, tmp_path: Path, look: str) -> None:
        words = _words(SPEECH + " unbelievably extraordinary catastrophically", 0.0)
        keywords = gta_captions.clip_keywords(words, True)
        path = render.write_subtitles(
            tmp_path / f"{look}.ass",
            words,
            style_key=look,
            hook_title="Title",
            title_s=3.0,
            duration_s=words[-1].end,
            keywords=keywords,
        )
        subs = pysubs2.load(str(path))
        style = gta_captions.resolve_style(look)
        ass_style = subs.styles[captions.STYLE_NAME]
        assert ass_style.fontname == style.font
        assert ass_style.marginv == render.CAPTION_MARGIN_PX >= render.SAFE_BOTTOM_PX
        assert ass_style.alignment == pysubs2.Alignment.BOTTOM_CENTER
        # Rebuild the groups exactly as written and check each fits 2 lines.
        groups = gta_captions.fit_groups(
            captions.group_words(words, max_words=style.max_words),
            style,
            width=render.OUT_W,
            height=render.OUT_H,
            keywords=keywords,
        )
        lines = max(
            gta_captions.line_count(
                g.words, style, width=render.OUT_W, height=render.OUT_H, keywords=keywords
            )
            for g in groups
        )
        assert lines <= gta_captions.MAX_LINES
        # Two lines of the biggest drawn word still end above the bottom safe
        # zone and below the stream frame.
        grow = max(style.scale_percent, style.emphasis_scale) / 100
        block = 2 * gta_captions.font_size_px(style, render.OUT_H) * grow
        bottom = render.OUT_H - ass_style.marginv
        assert bottom <= render.OUT_H - render.SAFE_BOTTOM_PX
        assert bottom - block >= FRAME_BOTTOM_16x9
        # The hook title keeps its own style in the top band.
        hook = next(e for e in subs.events if e.style == render.TITLE_STYLE)
        y = int(re.search(r"\\pos\(\d+,(\d+)\)", hook.text).group(1))
        assert y - 100 > render.SAFE_TOP_PX

    def test_long_group_is_split_to_two_lines(self) -> None:
        style = gta_captions.resolve_style("comic")
        wide = _words("unbelievably extraordinary catastrophically magnificently", 0.0)
        group = captions.CaptionGroup(words=wide)
        assert gta_captions.line_count(wide, style, width=1080, height=1920) > 2
        split = gta_captions.fit_groups([group], style, width=1080, height=1920)
        assert len(split) >= 2
        assert [w for g in split for w in g.words] == wide


# --------------------------------------------------------------------------
# Motion
# --------------------------------------------------------------------------


def _chains(graph: str) -> dict[str, str]:
    """Filter chains by their first input label."""
    out = {}
    for chain in graph.split(";"):
        match = re.match(r"\[([^\]]+)\]", chain)
        if match:
            out.setdefault(match.group(1), chain)
    return out


class TestMotion:
    def test_zoom_only_on_the_frame_layer(self) -> None:
        motion = render.FrameMotion(
            zooms=(render.shared_edit.Zoom(0.0, 1.0), render.shared_edit.Zoom(4.5, 1.08)),
            intro=False,
            width=1080,
            height=608,
        )
        graph = render.build_filtergraph(
            [render.Keep(10.0, 22.0)],
            seek_s=9.0,
            subtitle_name="captions.ass",
            fonts_name="fonts",
            audio_filter="anull",
            punch_in=(5.0, 6.5, 1.3),
            motion=motion,
        )
        assert graph.count("zoompan") == 1
        chains = _chains(graph)
        assert "zoompan" in chains["fgsrc"] and "s=1080x608" in chains["fgsrc"]
        assert "if(lt(it,4.500),1.000,1.080)" in chains["fgsrc"]
        assert "zoompan" not in chains["bgsrc"] and "boxblur" in chains["bgsrc"]
        assert "zoompan" not in chains["zsrc"]
        # Text is burned after every layer is composed, so it never moves.
        final = next(c for c in graph.split(";") if "ass=filename" in c)
        assert "zoompan" not in final
        assert graph.index("zoompan") < graph.index("ass=filename")

    def test_no_motion_keeps_the_plain_fit(self) -> None:
        graph = render.build_filtergraph(
            [render.Keep(0.0, 10.0)],
            seek_s=0.0,
            subtitle_name=None,
            fonts_name="fonts",
            audio_filter="anull",
        )
        assert "zoompan" not in graph and "force_original_aspect_ratio=decrease" in graph

    def test_fitted_size(self) -> None:
        assert render.fitted_size(1920, 1080) == (1080, 608)
        assert render.fitted_size(1280, 720) == (1080, 608)
        assert render.fitted_size(1080, 1920) == (1080, 1920)
        assert render.fitted_size(None, 720) is None

    def test_plan_motion_levels(self, tmp_path: Path) -> None:
        words = _words(SPEECH.replace("roof and", "roof. and"), 0.0, 0.5)
        duration = words[-1].end + 0.5
        keeps = [render.Keep(0.0, duration)]
        assert render.plan_motion(words, duration, keeps, "off", tmp_path / "x.mp4") is None
        assert render.plan_motion(words, 3.0, keeps, "subtle", tmp_path / "x.mp4") is None
        # An unreadable section never fails the render: just no motion.
        assert render.plan_motion(words, duration, keeps, "dynamic", tmp_path / "x.mp4") is None


# --------------------------------------------------------------------------
# Comment timestamps in the Hype Curve
# --------------------------------------------------------------------------


def _loudness(seconds: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (-30.0 + rng.normal(0, 0.5, seconds)).astype(np.float32)


class TestCommentSignal:
    COMMENTS = [  # noqa: RUF012 - canned data
        Comment("10:00 LMAOOO the helicopter", likes=900),
        Comment("bro at 10:02 i died", likes=120),
        Comment("10:01 clip of the year", likes=40),
        Comment("9:58 what was that", likes=5),
        Comment("great stream", likes=3000),
    ]

    def test_comments_become_a_weighted_signal_with_a_marker(self) -> None:
        seconds = 1800
        curve = hype.build_curve(seconds, loudness_db=_loudness(seconds), comments=self.COMMENTS)
        assert "comments" in curve.available and "audio" in curve.available
        assert float(curve.components["comments"].max()) == pytest.approx(1.0)
        near = [m for m in curve.markers if 595 <= m.t <= 630]
        assert near, curve.markers
        assert any("comment timestamp" in r for r in near[0].reasons)
        assert curve.peak(595, 630) > curve.peak(1200, 1300)

    def test_too_few_mentions_is_no_signal(self) -> None:
        curve = hype.build_curve(1800, loudness_db=_loudness(1800), comments=self.COMMENTS[:2])
        assert "comments" not in curve.available
        assert "comments" not in hype.build_curve(1800, comments=None).available

    def test_comments_are_cached_and_never_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream = store.create_stream("https://www.youtube.com/watch?v=abc")
        stream.platform = "youtube"
        ws = pipeline.Workspace(stream.id)

        def boom(url: str) -> list:
            raise RuntimeError("network down")

        monkeypatch.setattr(pipeline, "_fetch_comments", boom)
        assert pipeline.load_comments(ws, stream, fetch_missing=True) == []
        assert not ws.comments.exists()  # a failure is not cached

        calls: list[str] = []

        def canned(url: str) -> list:
            calls.append(url)
            return self.COMMENTS

        monkeypatch.setattr(pipeline, "_fetch_comments", canned)
        assert len(pipeline.load_comments(ws, stream, fetch_missing=True)) == 5
        assert len(pipeline.load_comments(ws, stream, fetch_missing=True)) == 5
        assert len(calls) == 1  # cached after the first read
        assert len(pipeline.load_comments(ws, stream, fetch_missing=False)) == 5
        stream.platform = "twitch"
        assert pipeline.load_comments(ws, stream, fetch_missing=True) == []


# --------------------------------------------------------------------------
# YouTube captions as the transcript
# --------------------------------------------------------------------------


def _json3(start_s: float, end_s: float, step_s: float = 0.4) -> dict:
    """Automatic-caption json3 with one word every ``step_s`` in a range."""
    events = []
    t = start_s
    i = 0
    while t < end_s:
        segs = [{"utf8": (" " if k else "") + f"word{i + k}", "tOffsetMs": round(k * step_s * 1000)}
                for k in range(5)]  # fmt: skip
        events.append({"tStartMs": round(t * 1000), "dDurationMs": round(5 * step_s * 1000),
                       "segs": segs})  # fmt: skip
        t += 5 * step_s
        i += 5
    return {"wireMagic": "pb3", "events": events}


def _silent_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))


class TestCaptionTranscript:
    def test_probe_picks_the_spoken_language_track(self) -> None:
        info = {
            "id": "v",
            "duration": 600,
            "language": "en",
            "automatic_captions": {
                "en-orig": [{"ext": "json3", "url": "https://example.invalid/en.json3"}],
                "de": [{"ext": "json3", "url": "https://example.invalid/de.json3"}],
            },
        }
        result = fetch.probe_from_info(info, "https://www.youtube.com/watch?v=v")
        assert result.caption_track == {
            "lang": "en",
            "kind": "auto",
            "url": "https://example.invalid/en.json3",
        }
        assert fetch.probe_from_info(info, "https://www.twitch.tv/videos/1").caption_track == {}

    def test_caption_part_doubts(self) -> None:
        words = _words(" ".join(["go"] * 200), 0.0, 0.5)  # 0-100 s, 120 wpm
        good = Transcript(words=words, language="en", model="youtube-auto:en")
        piece = pipeline.caption_part(good, 0.0, 100.0)
        assert piece is not None and piece.source == "youtube_captions"
        assert piece.words[0].start == pytest.approx(0.0)
        # Too sparse for real speech, or not covering the part: Whisper.
        sparse = Transcript(words=_words(" ".join(["go"] * 25), 0.0, 20.0))
        assert pipeline.caption_part(sparse, 0.0, 600.0) is None
        assert pipeline.caption_part(good, 0.0, 400.0) is None  # ends 300 s early
        assert pipeline.caption_part(None, 0.0, 100.0) is None

    def _setup(self, monkeypatch, autoclip_home: Path, *, captions_on: bool):
        from autoclip.gta import settings as gta_settings

        settings = GtaSettings(youtube_captions=captions_on)
        gta_settings.save(settings)
        monkeypatch.setattr(fetch, "PART_S", 120)
        stream = store.create_stream("https://www.youtube.com/watch?v=abc")
        store.update_stream(stream.id, platform="youtube", duration_s=240.0)
        stream = store.get_stream(stream.id)
        ws = pipeline.Workspace(stream.id)
        _silent_wav(ws.parts_dir / "part_0000.wav", 120)
        _silent_wav(ws.parts_dir / "part_0001.wav", 120)
        # Captions only cover the first part: the second goes to Whisper.
        ws.captions_dir.mkdir(parents=True, exist_ok=True)
        (ws.captions_dir / "captions.auto.en.json3").write_text(json.dumps(_json3(1.0, 119.0)))
        whisper_calls: list[str] = []

        def fake_transcribe(path, *_args, **_kwargs):
            whisper_calls.append(Path(path).name)
            return Transcript(words=_words("whisper heard this", 5.0), language="en", model="small")

        monkeypatch.setattr(pipeline.transcribe, "transcribe", fake_transcribe)
        report = pipeline.Reporter(stream.id, lambda: False)
        return stream, ws, settings, report, whisper_calls

    def test_captions_skip_whisper_per_part(self, monkeypatch, autoclip_home: Path) -> None:
        stream, ws, settings, report, calls = self._setup(
            monkeypatch, autoclip_home, captions_on=True
        )
        pipeline.stage_transcribe(stream, report, settings)
        assert calls == ["part_0001.wav"]
        transcript = Transcript.load(ws.transcript)
        assert transcript.words[0].text == "word0"
        assert any(
            w.text == "whisper" and w.start == pytest.approx(125.0) for w in transcript.words
        )
        assert store.get_stream(stream.id).signals["caption_parts"] == "1/2"

    def test_setting_off_uses_whisper_everywhere(self, monkeypatch, autoclip_home: Path) -> None:
        stream, ws, settings, report, calls = self._setup(
            monkeypatch, autoclip_home, captions_on=False
        )
        pipeline.stage_transcribe(stream, report, settings)
        assert calls == ["part_0000.wav", "part_0001.wav"]
        assert all(not w.text.startswith("word") for w in Transcript.load(ws.transcript).words)
