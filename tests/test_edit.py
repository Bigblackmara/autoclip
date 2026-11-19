"""Short-form editing for normal-mode exports (pipeline/edit.py).

Pace cuts, re-timed captions, punch-in zooms, key words, hook headline,
music, watermark, forced fps and two-pass loudness — the plan maths, the
filtergraph strings, the ASS output, the settings, and one real render.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from autoclip.config import ExportSettings
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import captions, edit, export, ffmpeg
from autoclip.pipeline.edit import Keep, Zoom
from autoclip.pipeline.prepare import Silence
from autoclip.pipeline.reframe.croppath import centre_crop
from autoclip.pipeline.transcript import Transcript, Word
from pydantic import ValidationError


def _w(text: str, start: float, end: float | None = None) -> Word:
    return Word(text=text, start=start, end=start + 0.4 if end is None else end)


def _say(sentence: str) -> list[Word]:
    """One word per second."""
    return [_w(text, float(i)) for i, text in enumerate(sentence.split())]


def _talk(duration: float, *, gaps: list[tuple[float, float]] = ()) -> list[Word]:
    """A word every 0.5 s, none inside ``gaps``."""
    words: list[Word] = []
    t = 0.1
    while t + 0.4 < duration:
        if not any(a - 0.4 < t < b for a, b in gaps):
            words.append(_w(f"word{len(words)}", t))
        t += 0.5
    return words


# --------------------------------------------------------------------------
# Pace
# --------------------------------------------------------------------------


class TestPlanPace:
    def test_off_and_short_clips_are_never_cut(self) -> None:
        silences = [(3.0, 5.0)]
        assert edit.plan_pace(10.0, silences, [], "off") == [Keep(0.0, 10.0)]
        assert edit.plan_pace(3.5, [(1.0, 2.5)], [], "natural") == [Keep(0.0, 3.5)]

    def test_natural_shortens_a_long_pause_leaving_breath_each_side(self) -> None:
        words = _talk(12.0, gaps=[(4.0, 6.0)])

        keeps = edit.plan_pace(12.0, [(4.0, 6.0)], words, "natural")

        assert keeps == [Keep(0.0, 4.18), Keep(5.82, 12.0)]

    def test_natural_keeps_short_pauses_fast_cuts_them(self) -> None:
        silences = [(4.0, 4.5)]  # 0.5 s: under natural's 0.6 s, over fast's 0.35 s
        words = _talk(12.0, gaps=silences)

        assert edit.plan_pace(12.0, silences, words, "natural") == [Keep(0.0, 12.0)]
        assert edit.plan_pace(12.0, silences, words, "fast") == [Keep(0.0, 4.1), Keep(4.4, 12.0)]

    def test_leading_dead_air_and_long_tail_are_trimmed(self) -> None:
        silences = [(0.0, 1.5), (9.0, 12.0)]
        words = _talk(9.0, gaps=[(0.0, 1.5)])

        keeps = edit.plan_pace(12.0, silences, words, "natural")

        assert keeps == [Keep(1.32, 9.7)]

    def test_a_silence_over_a_spoken_word_is_not_cut(self) -> None:
        """silencedetect can mistake quiet speech for silence."""
        words = [*_talk(4.0), _w("whisper", 5.0, 5.6), *[_w("x", 7.0 + i * 0.5) for i in range(8)]]

        keeps = edit.plan_pace(12.0, [(4.0, 6.8)], words, "natural")

        assert keeps == [Keep(0.0, 12.0)]

    def test_a_lone_filler_between_two_pauses_goes_with_them(self) -> None:
        words = [*_talk(4.0), _w("um,", 4.45, 4.75), *[_w("x", 5.6 + i * 0.5) for i in range(12)]]
        silences = [(3.95, 4.4), (4.8, 5.5)]  # each under 0.6 s on its own

        keeps = edit.plan_pace(12.0, silences, words, "natural")

        assert keeps == [Keep(0.0, 4.13), Keep(5.32, 12.0)]

    def test_gives_up_when_most_of_the_clip_would_go(self) -> None:
        silences = [(1.0, 4.0), (5.0, 9.0)]

        assert edit.plan_pace(10.0, silences, [], "fast") == [Keep(0.0, 10.0)]

    def test_silences_in_is_clip_relative(self) -> None:
        silences = [Silence(5.0, 6.0), Silence(19.0, 21.0), Silence(40.0, 41.0)]

        assert edit.silences_in(silences, 10.0, 30.0) == [(9.0, 11.0)]


class TestKeepMaths:
    def test_snap_puts_cuts_on_the_frame_grid(self) -> None:
        keeps = edit.snap_keeps([Keep(0.0, 4.18), Keep(5.82, 12.0)], 12.0, 30)

        for keep in keeps:
            for t in (keep.start, keep.end):
                assert abs(t * 30 - round(t * 30)) < 1e-3
        assert keeps[0] == Keep(0.0, 4.1667)
        assert keeps[1].start == pytest.approx(5.8333, abs=1e-4)

    def test_snap_merges_touching_keeps_and_drops_slivers(self) -> None:
        keeps = edit.snap_keeps([Keep(0.0, 2.001), Keep(2.005, 4.0), Keep(4.5, 4.55)], 5.0)

        assert keeps == [Keep(0.0, 4.0)]

    def test_cut_times_are_output_times(self) -> None:
        assert edit.cut_times([Keep(0.0, 2.0), Keep(5.0, 7.0), Keep(9.0, 10.0)]) == [2.0, 4.0]
        assert edit.cut_times([Keep(0.0, 9.0)]) == []

    def test_retime_moves_words_after_a_cut_and_drops_cut_words(self) -> None:
        keeps = [Keep(0.0, 2.0), Keep(5.0, 7.0)]
        words = [_w("a", 0.5, 1.0), _w("gone", 3.0, 3.5), _w("b", 5.5, 6.0), _w("edge", 1.7, 2.2)]

        out = edit.retime_words(words, keeps)

        assert [(w.text, w.start, w.end) for w in out] == [
            ("a", 0.5, 1.0),
            ("edge", 1.7, 2.0),
            ("b", 2.5, 3.0),
        ]

    def test_gta_shares_the_keep_type_and_remap(self) -> None:
        from autoclip.gta import render

        assert render.Keep is edit.Keep
        assert render.remap_words is edit.remap_words
        assert render.loudnorm_filter is edit.loudnorm_filter


# --------------------------------------------------------------------------
# Motion
# --------------------------------------------------------------------------


def _sentences(duration: float) -> list[Word]:
    """Four-word sentences, a 0.3 s pause after each."""
    words: list[Word] = []
    t = 0.0
    i = 0
    while t + 0.3 < duration:
        text = f"w{i}." if i % 4 == 3 else f"w{i}"
        words.append(_w(text, t, t + 0.3))
        t += 0.6 if i % 4 == 3 else 0.35
        i += 1
    return words


class TestPlanZooms:
    def test_subtle_zooms_at_pauses_every_few_seconds(self) -> None:
        zooms = edit.plan_zooms(_sentences(20.0), 20.0, "subtle")

        assert zooms[0] == Zoom(0.0, 1.0)
        assert {z.z for z in zooms} == {1.0, 1.08}
        # The first punch-in may come early; after that, one per ~4.5 s at most.
        gaps = [b.t - a.t for a, b in zip(zooms[1:], zooms[2:], strict=False)]
        assert len(zooms) >= 4
        assert all(g >= 4.5 - 1e-6 for g in gaps)

    def test_dynamic_zooms_more_often_and_deeper(self) -> None:
        subtle = edit.plan_zooms(_sentences(20.0), 20.0, "subtle")
        dynamic = edit.plan_zooms(_sentences(20.0), 20.0, "dynamic")

        assert len(dynamic) > len(subtle)
        assert max(z.z for z in dynamic) == 1.2

    def test_every_jump_cut_gets_a_zoom_change_even_with_motion_off(self) -> None:
        assert edit.plan_zooms(_sentences(20.0), 20.0, "off") == []

        zooms = edit.plan_zooms(_sentences(20.0), 20.0, "off", [5.0, 11.0])

        assert zooms == [Zoom(0.0, 1.0), Zoom(5.0, edit.CUT_ZOOM), Zoom(11.0, 1.0)]

    def test_cuts_are_forced_zoom_events_with_motion_on(self) -> None:
        zooms = edit.plan_zooms(_sentences(20.0), 20.0, "subtle", [7.77])

        assert 7.77 in [z.t for z in zooms]

    def test_short_clips_do_not_zoom(self) -> None:
        assert edit.plan_zooms(_sentences(3.0), 3.0, "dynamic", [1.5]) == []

    def test_zoom_expression(self) -> None:
        zooms = [Zoom(0.0, 1.0), Zoom(2.5, 1.08), Zoom(7.0, 1.0)]

        assert edit.zoom_expression(zooms) == "if(lt(it,2.500),1.000,if(lt(it,7.000),1.080,1.000))"
        assert edit.zoom_expression(zooms, intro=True).startswith(
            "if(lt(it,0.5),1.18-0.18*(it/0.5)*(2-it/0.5),if(lt(it,2.500)"
        )
        assert edit.zoom_expression([]) == "1"


# --------------------------------------------------------------------------
# Key words
# --------------------------------------------------------------------------


class TestKeywords:
    def test_numbers_then_longest_content_words(self) -> None:
        words = _say(
            "So basically we grew the agency to 10,000 customers because retention "
            "actually matters more than you think"
        )

        assert edit.pick_keywords(words) == ["10000", "customers", "retention"]

    def test_between_three_and_eight_and_deterministic(self) -> None:
        vocabulary = [
            "rocket",
            "planet",
            "harbour",
            "violin",
            "meadow",
            "lantern",
            "copper",
            "winter",
            "garden",
            "marble",
        ]
        words = [_w(vocabulary[i % len(vocabulary)], i) for i in range(200)]

        picked = edit.pick_keywords(words)

        assert len(picked) == 8
        assert picked == edit.pick_keywords(list(words))

    def test_stopwords_fillers_and_contractions_are_never_picked(self) -> None:
        words = [
            _w(t, i)
            for i, t in enumerate(["don't", "really", "umm", "because", "should've", "the"])
        ]

        assert edit.pick_keywords(words) == []

    def test_one_candidate_per_stem(self) -> None:
        words = [_w(t, i) for i, t in enumerate(["failed", "failure", "failing"])]

        assert edit.pick_keywords(words) == ["failed"]

    def test_keyword_of_matches_endings_and_punctuation(self) -> None:
        keywords = ["fail", "retention", "10000"]

        assert edit.keyword_of("FAILED!", keywords) == "fail"
        assert edit.keyword_of("retention,", keywords) == "retention"
        assert edit.keyword_of("10,000", keywords) == "10000"
        assert edit.keyword_of("a", keywords) is None


# --------------------------------------------------------------------------
# Captions: key words and hook
# --------------------------------------------------------------------------


class TestCaptionExtras:
    def _words(self) -> list[Word]:
        texts = [
            "the",
            "agency",
            "grew",
            "fast.",
            "the",
            "agency",
            "grew",
            "again.",
            "agency",
            "agency",
        ]
        return [_w(t, i * 0.5, i * 0.5 + 0.4) for i, t in enumerate(texts)]

    def test_key_word_drawn_in_emphasis_colour_once_per_line_and_twice_per_clip(self) -> None:
        style = captions.get_style("clean_lower")  # static: one event per line
        subs = captions.build_ass(
            self._words(), style, width=1080, height=1920, keywords=["agency"]
        )
        emph = f"\\c{captions.ass_colour_override(style.emphasis)}"

        lines = [e.text for e in subs.events]
        assert [line.count(emph) for line in lines] == [1, 1, 0]

    def test_per_word_style_keeps_emphasis_while_spoken(self) -> None:
        style = captions.get_style("pop")
        subs = captions.build_ass(
            [_w("big", 0.0), _w("money", 0.5)], style, width=1080, height=1920, keywords=["money"]
        )
        emph = f"\\c{captions.ass_colour_override(style.emphasis)}\\fscx124\\fscy124"

        assert all(f"{emph}}}MONEY" in e.text for e in subs.events)

    def test_karaoke_key_word(self) -> None:
        style = captions.get_style("karaoke_fill")
        subs = captions.build_ass(
            [_w("big", 0.0), _w("money", 0.5)], style, width=1080, height=1920, keywords=["money"]
        )

        assert "\\kf" in subs.events[0].text
        assert f"\\c{captions.ass_colour_override(style.emphasis)}" in subs.events[0].text

    def test_no_keywords_leaves_output_unchanged(self) -> None:
        style = captions.get_style("bold_pop")
        words = self._words()

        plain = captions.build_ass(words, style, width=1080, height=1920)
        empty = captions.build_ass(words, style, width=1080, height=1920, keywords=[])

        assert [e.text for e in plain.events] == [e.text for e in empty.events]

    def test_hook_headline_for_the_first_seconds(self) -> None:
        subs = captions.build_ass(
            [],
            captions.get_style("bold_pop"),
            width=1080,
            height=1920,
            hook_text="Why {most} agencies die",
            hook_top_px=300,
            duration_s=20.0,
        )

        (hook,) = subs.events
        assert hook.style == captions.HOOK_STYLE_NAME
        assert (hook.start, hook.end) == (0, 3200)
        assert hook.text.startswith("{\\an8\\pos(540,300)\\fad(120,220)}Why \\{most\\}")
        assert subs.styles[captions.HOOK_STYLE_NAME].borderstyle == 3

    def test_hook_never_outlasts_a_short_clip(self) -> None:
        subs = captions.build_ass(
            [], captions.get_style("pop"), width=1080, height=1920, hook_text="Hi", duration_s=2.5
        )

        assert subs.events[0].end == 2300

    @pytest.mark.parametrize("key", ["pop", "neon", "comic", "minimal"])
    def test_new_presets_render_with_their_own_fonts(self, key: str) -> None:
        style = captions.get_style(key)

        assert style.font_path.is_file()
        subs = captions.build_ass(self._words(), style, width=1080, height=1920, keywords=["grew"])
        assert subs.events and subs.styles[captions.STYLE_NAME].fontname == style.font

    def test_neon_glows(self) -> None:
        subs = captions.build_ass(
            [_w("a", 0.0), _w("b", 0.5)], captions.get_style("neon"), width=1080, height=1920
        )

        assert all(e.text.startswith("{\\blur4}") for e in subs.events)
        assert "\\3c" in subs.events[0].text and "{\\r\\blur4}" in subs.events[0].text


# --------------------------------------------------------------------------
# Music and watermark
# --------------------------------------------------------------------------


class TestMusic:
    def test_resolves_a_file_or_a_stable_pick_from_a_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "music"
        folder.mkdir()
        for name in ("a.mp3", "b.wav", "c.m4a", "notes.txt"):
            (folder / name).write_bytes(b"x")

        assert edit.resolve_music(str(folder / "a.mp3")) == folder / "a.mp3"
        picks = {edit.resolve_music(str(folder), seed=f"clip{i}") for i in range(30)}
        assert picks == {folder / "a.mp3", folder / "b.wav", folder / "c.m4a"}
        assert edit.resolve_music(str(folder), "x") == edit.resolve_music(str(folder), "x")

    def test_missing_or_empty_is_no_music(self, tmp_path: Path) -> None:
        assert edit.resolve_music("") is None
        assert edit.resolve_music(str(tmp_path / "nope.mp3")) is None
        assert edit.resolve_music(str(tmp_path)) is None

    def test_ducked_mix(self) -> None:
        graph = edit.music_chain(2, 20.0, 0.3, duck=True, speech="[esp]", out_label="[emix]")

        assert graph.startswith("[2:a]aformat=")
        assert "atrim=0:20.000" in graph and "volume=0.300" in graph
        assert "afade=t=in:st=0:d=0.8" in graph and "afade=t=out:st=18.200:d=1.800" in graph
        assert "[esp]asplit=2[sp1][sp2]" in graph
        assert "[mus][sp2]sidechaincompress=" in graph
        assert graph.endswith(
            "[sp1][mduck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[emix]"
        )

    def test_unducked_and_music_only(self) -> None:
        plain = edit.music_chain(1, 10.0, 0.5, duck=False, speech="[esp]", out_label="[emix]")
        alone = edit.music_chain(1, 10.0, 0.5, duck=True, speech=None, out_label="[emix]")

        assert "sidechaincompress" not in plain and "[esp][mus]amix" in plain
        assert alone.endswith("[mus]anull[emix]")


class TestWatermark:
    @pytest.mark.parametrize(
        ("corner", "x", "y"),
        [
            ("top_left", "43", "144"),
            ("top_right", "W-w-43", "144"),
            ("bottom_left", "43", "H-h-307"),
            ("bottom_right", "W-w-43", "H-h-307"),
        ],
    )
    def test_corners_on_vertical(self, corner: str, x: str, y: str) -> None:
        mark = edit.Watermark(path=Path("logo.png"), corner=corner, opacity=0.5)

        graph = edit.watermark_filter(mark, 2, "[v]", "[w]", 1080, 1920)

        assert graph.startswith(
            "[2:v]scale=205:134:force_original_aspect_ratio=decrease,"
            "format=rgba,colorchannelmixer=aa=0.500[wm];"
        )
        assert graph.endswith(f"[v][wm]overlay={x}:{y}:format=auto[w]")

    def test_off_missing_image_and_empty_text_are_no_watermark(self, tmp_path: Path) -> None:
        assert edit.resolve_watermark(ExportSettings()) is None
        missing = ExportSettings(watermark="image", watermark_image=str(tmp_path / "x.png"))
        assert edit.resolve_watermark(missing) is None
        assert edit.resolve_watermark(ExportSettings(watermark="text")) is None

    def test_image_and_text_logo(self, tmp_path: Path) -> None:
        png = tmp_path / "logo.png"
        png.write_bytes(b"png")
        image = edit.resolve_watermark(
            ExportSettings(
                watermark="image", watermark_image=str(png), watermark_corner="bottom_left"
            )
        )
        text = edit.resolve_watermark(
            ExportSettings(watermark="text", watermark_text="My Brand"), cache_dir=tmp_path / "c"
        )

        assert image == edit.Watermark(path=png, corner="bottom_left", opacity=0.85)
        assert text is not None and text.path.is_file() and text.path.suffix == ".png"

    def test_hook_moves_below_a_top_corner_logo(self) -> None:
        top = edit.Watermark(path=Path("l.png"), corner="top_left")
        bottom = edit.Watermark(path=Path("l.png"), corner="bottom_left")

        plain = edit.hook_top_px(1080, 1920, split=False, watermark=None)
        assert plain == 300
        assert edit.hook_top_px(1080, 1920, split=False, watermark=bottom) == plain
        assert edit.hook_top_px(1080, 1920, split=False, watermark=top) == 144 + 134 + 29
        assert edit.hook_top_px(1080, 1920, split=True, watermark=None) == 230


# --------------------------------------------------------------------------
# Loudness
# --------------------------------------------------------------------------


class TestLoudness:
    STDERR = """[Parsed_loudnorm_0 @ 0x1]
{
\t"input_i" : "-23.51",
\t"input_tp" : "-6.02",
\t"input_lra" : "3.10",
\t"input_thresh" : "-33.80",
\t"output_i" : "-14.02",
\t"target_offset" : "0.02"
}"""

    def test_parse_and_linear_second_pass(self) -> None:
        measured = edit.parse_loudnorm(self.STDERR)

        assert measured == {
            "input_i": "-23.51",
            "input_tp": "-6.02",
            "input_lra": "3.10",
            "input_thresh": "-33.80",
            "target_offset": "0.02",
        }
        # +9.51 dB would put the -6.02 dBTP peak at +3.49, over the -1.5 ceiling:
        # loudnorm would drop to its dynamic mode (loud, crunchy opening), so
        # the gain is applied evenly and only the peaks are limited.
        assert edit.loudnorm_filter(measured, target=-14.0, true_peak=-1.5) == (
            "volume=9.51dB,alimiter=limit=0.7943:attack=5:release=80:level=false:latency=true"
        )
        # With headroom for the gain, the linear second pass stays.
        roomy = {**measured, "input_tp": "-14.00"}
        assert edit.loudnorm_filter(roomy, target=-14.0, true_peak=-1.5) == (
            "loudnorm=I=-14.0:TP=-1.5:LRA=11:measured_I=-23.51:measured_TP=-14.00"
            ":measured_LRA=3.10:measured_thresh=-33.80:offset=0.02:linear=true"
        )

    def test_silence_or_garbage_falls_back_to_one_pass(self) -> None:
        assert edit.parse_loudnorm(self.STDERR.replace("-23.51", "-inf")) is None
        assert edit.parse_loudnorm(self.STDERR.replace("-23.51", "-80.0")) is None
        assert edit.parse_loudnorm("nothing here") is None
        assert edit.loudnorm_filter(None, target=-14.0, true_peak=-1.5) == (
            "loudnorm=I=-14.0:TP=-1.5:LRA=11"
        )


# --------------------------------------------------------------------------
# Settings and the plan
# --------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self) -> None:
        s = ExportSettings()

        assert (s.pace, s.motion, s.hook_title, s.keyword_highlight) == (
            "natural",
            "subtle",
            True,
            True,
        )
        assert not s.music_enabled and s.watermark == "off"
        assert s.fps == 30 and s.two_pass_loudnorm and s.loudness_lufs == -14.0

    @pytest.mark.parametrize(
        "bad",
        [
            {"pace": "slow"},
            {"motion": "wild"},
            {"music_volume": 300},
            {"music_volume": -1},
            {"watermark": "logo"},
            {"watermark_corner": "middle"},
            {"watermark_opacity": 0.0},
            {"fps": 5},
        ],
    )
    def test_invalid_values_are_rejected(self, bad: dict) -> None:
        with pytest.raises(ValidationError):
            ExportSettings(**bad)

    def test_api_round_trip_and_rejection(self, autoclip_home, monkeypatch) -> None:
        from autoclip import app as app_module
        from autoclip.app import create_app
        from fastapi.testclient import TestClient

        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        with TestClient(create_app()) as client:
            ok = client.put(
                "/api/settings",
                json={"export": {"pace": "fast", "motion": "off", "watermark_corner": "top_left"}},
            )
            bad = client.put("/api/settings", json={"export": {"pace": "slow"}})

            assert ok.status_code == 200
            body = ok.json()["export"]
            assert (body["pace"], body["motion"], body["watermark_corner"]) == (
                "fast",
                "off",
                "top_left",
            )
            assert body["hook_title"] is True and body["fps"] == 30
            assert bad.status_code == 400


class TestPlanForClip:
    WORDS = [
        _w(t, 100.0 + i * 0.5)
        for i, t in enumerate(["this", "changed", "everything", "for", "the", "business"])
    ]

    def test_defaults_cut_zoom_hook_and_keywords(self) -> None:
        words = [*_talk(4.0), *[_w(f"later{i}", 6.0 + i * 0.5) for i in range(12)]]
        words = [Word(w.text, w.start + 100.0, w.end + 100.0) for w in words]

        plan = edit.plan_for_clip(
            ExportSettings(),
            start_s=100.0,
            end_s=112.0,
            words=words,
            silences=[Silence(104.0, 105.9)],
            title="  Why most agencies   die ",
            seed="c1",
        )

        assert plan.keeps is not None and len(plan.keeps) == 2
        assert plan.output_duration(12.0) < 12.0
        assert any(abs(z.t - plan.keeps[0].duration) < 1e-3 for z in plan.zooms)
        assert plan.hook_text == "Why most agencies die"
        assert plan.keywords
        assert plan.fps == 30 and plan.two_pass_loudnorm
        assert plan.music is None and plan.watermark is None

    def test_everything_off(self) -> None:
        settings = ExportSettings(
            pace="off", motion="off", hook_title=False, keyword_highlight=False
        )

        plan = edit.plan_for_clip(
            settings, start_s=100.0, end_s=112.0, words=self.WORDS, silences=[Silence(104, 106)]
        )

        assert plan.keeps is None and plan.zooms == [] and not plan.intro_zoom
        assert plan.hook_text == "" and plan.keywords == []
        assert plan.summary() == []

    def test_no_motion_where_zooms_would_break_the_layout(self) -> None:
        plan = edit.plan_for_clip(
            ExportSettings(motion="dynamic"),
            start_s=0.0,
            end_s=20.0,
            words=_sentences(20.0),
            allow_motion=False,
        )

        assert plan.zooms == [] and not plan.intro_zoom

    def test_music_volume_is_a_gain(self, tmp_path: Path) -> None:
        track = tmp_path / "t.mp3"
        track.write_bytes(b"x")
        settings = ExportSettings(music_enabled=True, music_path=str(track), music_volume=40)

        plan = edit.plan_for_clip(settings, start_s=0.0, end_s=10.0, words=[])

        assert plan.music == track and plan.music_volume == pytest.approx(0.4)


# --------------------------------------------------------------------------
# Export filtergraph
# --------------------------------------------------------------------------


def _request(tmp_path: Path, plan: edit.EditPlan | None, **values) -> export.ExportRequest:
    base = {
        "source": tmp_path / "src.mp4",
        "destination": tmp_path / "out" / "clip.mp4",
        "start_s": 10.0,
        "end_s": 22.0,
        "crop_path": centre_crop(1280, 720, 12.0),
        "words": [_w("hello", 10.5), _w("world", 18.0)],
        "style": captions.get_style("bold_pop"),
        "edit": plan,
    }
    base.update(values)
    return export.ExportRequest(**base)


PLAN = edit.EditPlan(
    keeps=[Keep(0.0, 4.0), Keep(5.0, 9.0), Keep(10.0, 12.0)],
    zooms=[Zoom(0.0, 1.0), Zoom(4.0, 1.08)],
    hook_text="Hook",
    watermark=edit.Watermark(path=Path("logo.png")),
    music=Path("song.mp3"),
)


class TestEditFiltergraph:
    def test_without_an_edit_the_graph_is_unchanged(self, tmp_path: Path) -> None:
        graph = export.build_video_filtergraph(_request(tmp_path, None), subtitle_name="c.ass")

        assert "fps=" not in graph and "zoompan" not in graph and "[ek" not in graph
        assert graph.endswith("ass=filename=c.ass:fontsdir=fonts[vout]")

    def test_cuts_zooms_watermark_then_captions(self, tmp_path: Path) -> None:
        graph = export.build_video_filtergraph(_request(tmp_path, PLAN), subtitle_name="c.ass")

        assert "[v0]fps=30,split=3[ek0][ek1][ek2]" in graph
        assert "[ek1]trim=start=5.0000:end=9.0000,setpts=PTS-STARTPTS[ec1]" in graph
        assert "[ec0][ec1][ec2]concat=n=3:v=1:a=0[ecut]" in graph
        assert "[ecut]zoompan=z='if(lt(it,4.000),1.000,1.080)'" in graph
        assert ":d=1:s=1080x1920:fps=30,setsar=1[ezoom]" in graph
        assert "[1:v]scale=205:134" in graph and "[ezoom][wm]overlay=W-w-43:144" in graph
        assert graph.endswith("[ewm]ass=filename=c.ass:fontsdir=fonts[vout]")

    def test_uncut_edit_still_forces_the_frame_rate(self, tmp_path: Path) -> None:
        plan = edit.EditPlan(fps=24)

        graph = export.build_video_filtergraph(_request(tmp_path, plan), subtitle_name=None)

        assert graph.endswith("[v0]fps=24[efps];[efps]null[vout]")

    def test_split_layout_and_audio_only_never_zoom(self, tmp_path: Path) -> None:
        audio_only = _request(tmp_path, PLAN, has_video=False)

        graph = export.build_video_filtergraph(audio_only, subtitle_name=None)

        assert "zoompan" not in graph
        assert "[2:v]scale=205:134" in graph  # background is input 1, logo input 2
        assert export.input_layout(audio_only) == export.InputLayout(
            background=1, watermark=2, music=3
        )

    def test_audio_graph_cuts_mixes_and_normalises(self, tmp_path: Path) -> None:
        request = _request(tmp_path, PLAN)

        graph = export.build_edit_audio_graph(request, tail="loudnorm=I=-14", music_index=2)

        assert graph is not None
        assert graph.startswith("[0:a]asplit=3[ea0][ea1][ea2];")
        assert (
            "[ea1]atrim=start=5.0000:end=9.0000,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.012"
            in graph
        )
        assert "[eac0][eac1][eac2]concat=n=3:v=0:a=1," in graph
        assert "[2:a]aformat=" in graph and "atrim=0:10.000" in graph
        assert "sidechaincompress" in graph
        assert graph.endswith("[emix]loudnorm=I=-14[aout]")

    def test_audio_graph_without_audio_or_music_is_none(self, tmp_path: Path) -> None:
        silent = _request(tmp_path, edit.EditPlan(), has_audio=False)
        music_only = _request(tmp_path, edit.EditPlan(music=Path("m.mp3")), has_audio=False)

        assert export.build_edit_audio_graph(silent, tail="anull", music_index=None) is None
        graph = export.build_edit_audio_graph(music_only, tail="anull", music_index=1)
        assert graph is not None and "[0:a]" not in graph and "[mus]anull[emix]" in graph


class TestEditExportArgs:
    def test_render_arguments(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[list[str]] = []
        measured: list[tuple] = []

        def fake_run(args, **kwargs):
            calls.append([*args, f"total={kwargs['total_duration_s']:.3f}"])
            Path(args[-1]).write_bytes(b"mp4")

        def fake_measure(inputs, graph, **kwargs):
            measured.append((list(inputs), graph))
            return {
                "input_i": "-20.0",
                "input_tp": "-12.0",
                "input_lra": "4.0",
                "input_thresh": "-30.0",
                "target_offset": "0.1",
            }

        monkeypatch.setattr(export.ffmpeg, "run", fake_run)
        monkeypatch.setattr(export.edit_module, "measure_loudnorm", fake_measure)
        logo = tmp_path / "logo.png"
        logo.write_bytes(b"png")
        plan = edit.EditPlan(
            keeps=PLAN.keeps,
            zooms=PLAN.zooms,
            hook_text="Hook",
            keywords=["world"],
            watermark=edit.Watermark(path=logo),
            music=tmp_path / "song.mp3",
        )

        export.export_clip(
            _request(tmp_path, plan),
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False, write_srt=True),
        )

        (args,) = calls
        assert args[args.index("-i", 6) + 1] == str(logo)
        loop = args.index("-stream_loop")
        assert args[loop : loop + 4] == ["-stream_loop", "-1", "-i", str(tmp_path / "song.mp3")]
        assert "-af" not in args and args[args.index("[aout]") - 1] == "-map"
        assert args[args.index("-r") + 1] == "30"
        assert args[args.index("-t", 6) + 1] == "10.0000"
        assert args[-1] == "total=10.000"
        graph = args[args.index("-filter_complex") + 1]
        assert "measured_I=-20.0" in graph and "linear=true" in graph
        # measured over the same cut + music mix, with the music as input 1
        (inputs, measure_graph) = measured[0]
        assert inputs[-4:] == ["-stream_loop", "-1", "-i", str(tmp_path / "song.mp3")]
        assert "[1:a]aformat=" in measure_graph and "print_format=json[aout]" in measure_graph

        ass = (tmp_path / "work" / "clip" / "captions.ass").read_text(encoding="utf-8")
        # "world" was at 18.0 s absolute = 8.0 s clip time; after the 1 s cut at 4-5 s, 7.0 s.
        assert "0:00:07.00" in ass and "AutoClipHook" in ass
        assert f"\\c{captions.ass_colour_override('#7CFF55')}" in ass
        srt = (tmp_path / "out" / "clip.srt").read_text(encoding="utf-8")
        assert "00:00:07,000" in srt

    def test_one_pass_when_two_pass_is_off(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(
            export.ffmpeg,
            "run",
            lambda args, **kw: (calls.append(list(args)), Path(args[-1]).write_bytes(b"x")),
        )
        monkeypatch.setattr(
            export.edit_module,
            "measure_loudnorm",
            lambda *a, **k: pytest.fail("measured despite two_pass_loudnorm=False"),
        )

        export.export_clip(
            _request(tmp_path, edit.EditPlan(two_pass_loudnorm=False)),
            work_dir=tmp_path / "work",
            settings=ExportSettings(prefer_hardware_encoder=False),
        )

        graph = calls[0][calls[0].index("-filter_complex") + 1]
        assert "[esp]loudnorm=I=-14.0:TP=-1.5:LRA=11,aresample=48000[aout]" in graph


# --------------------------------------------------------------------------
# Runner and re-export wiring
# --------------------------------------------------------------------------


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


def _capture(monkeypatch, target) -> list[export.ExportRequest]:
    rendered: list[export.ExportRequest] = []

    def fake_export(request, **kwargs):
        request.destination.parent.mkdir(parents=True, exist_ok=True)
        request.destination.write_bytes(b"mp4")
        rendered.append(request)
        return request.destination

    monkeypatch.setattr(target, "export_clip", fake_export)
    return rendered


class TestWiring:
    def test_runner_cuts_on_the_cached_silences(self, source: Source, monkeypatch) -> None:
        from autoclip.config import Settings
        from autoclip.pipeline.runner import PipelineRunner

        rendered = _capture(monkeypatch, export)
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        runner = PipelineRunner(job, source, settings=Settings())
        runner.workspace.silences.write_text(
            json.dumps([{"start": 14.0, "end": 16.0}]), encoding="utf-8"
        )
        clip = Clip(id=new_id(), job_id=job.id, rank=1, start_s=10.0, end_s=22.0, title="Big win")
        store.replace_clips(job.id, [clip])
        words = [w for w in _talk(22.0, gaps=[(14.0, 16.0)]) if w.start >= 10.0]

        runner._stage_export(
            [clip], Transcript(words=words), {clip.id: centre_crop(1920, 1080, 12.0)}
        )

        (request,) = rendered
        assert request.edit is not None
        assert request.edit.keeps == [Keep(0.0, 4.1667), Keep(5.8333, 12.0)]
        assert request.edit.hook_text == "Big win"
        assert request.has_audio is True

    def test_re_export_uses_the_same_plan(self, autoclip_home, monkeypatch) -> None:
        from autoclip import app as app_module
        from autoclip.api import clips as clips_api
        from autoclip.app import create_app
        from autoclip.pipeline.runner import JobWorkspace
        from fastapi.testclient import TestClient

        monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
        rendered = _capture(monkeypatch, clips_api.export_module)
        with TestClient(create_app()) as client:
            src = store.create_source(
                Source(
                    id=new_id(),
                    type="upload",
                    path="C:/m/v.mp4",
                    duration_s=60.0,
                    width=1920,
                    height=1080,
                )
            )
            job = store.create_job(Job(id=new_id(), source_id=src.id, status="done"))
            clip = Clip(id=new_id(), job_id=job.id, rank=1, start_s=10.0, end_s=22.0, title="T")
            store.replace_clips(job.id, [clip])
            workspace = JobWorkspace(job.id)
            Transcript(words=_talk(22.0, gaps=[(14.0, 16.0)]), language="en").save(
                workspace.transcript
            )
            workspace.silences.write_text(json.dumps([{"start": 14.0, "end": 16.0}]), "utf-8")
            client.put("/api/settings", json={"export": {"motion": "dynamic"}})

            response = client.post(
                f"/api/clips/{clip.id}/export", json={"ratio": "9:16", "style": "pop"}
            )

        assert response.status_code == 201
        (request,) = rendered
        assert request.edit is not None and request.edit.keeps is not None
        assert len(request.edit.keeps) == 2 and request.edit.intro_zoom
        assert request.edit.hook_text == "T"


# --------------------------------------------------------------------------
# Real render
# --------------------------------------------------------------------------


@pytest.mark.slow
class TestEditedRender:
    def test_cut_zoomed_music_watermarked_clip_renders(self, tmp_path: Path) -> None:
        ff = ffmpeg.ffmpeg_path()
        src = tmp_path / "src.mp4"
        song = tmp_path / "song.wav"
        # Speech-like tone with a 1.5 s silence at 3.0-4.5 s.
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25:duration=10",
             "-f", "lavfi", "-i", "sine=frequency=300:duration=10",
             "-af", "volume='if(between(t,3.0,4.5),0,1)':eval=frame",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)],
            check=True,
        )  # fmt: skip
        subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "sine=frequency=660:duration=3", str(song)],
            check=True,
        )  # fmt: skip
        from autoclip.pipeline import prepare

        settings = ExportSettings(
            prefer_hardware_encoder=False,
            motion="dynamic",
            music_enabled=True,
            music_path=str(song),
            watermark="text",
            watermark_text="Test Logo",
        )
        words = [
            _w(f"word{i}", t)
            for i, t in enumerate([0.2, 0.8, 1.4, 2.0, 2.5, 4.7, 5.3, 6.0, 7.0, 8.0])
        ]
        plan = edit.plan_for_clip(
            settings,
            start_s=0.0,
            end_s=10.0,
            words=words,
            silences=prepare.detect_silences(src),
            title="A hook",
            seed="s",
        )
        assert plan.keeps is not None and len(plan.keeps) == 2
        dest = tmp_path / "out" / "edited.mp4"

        export.export_clip(
            _request(
                tmp_path,
                plan,
                source=src,
                destination=dest,
                start_s=0.0,
                end_s=10.0,
                crop_path=centre_crop(1280, 720, 10.0),
                words=words,
                style=captions.get_style("neon"),
            ),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        info = ffmpeg.probe(dest)
        assert info.width == 1080 and info.height == 1920
        assert info.fps == pytest.approx(30.0)
        assert info.has_audio
        assert info.duration_s == pytest.approx(plan.output_duration(10.0), abs=0.1)
        assert info.duration_s < 9.0
        loud = subprocess.run(
            [ff, "-hide_banner", "-nostdin", "-i", str(dest), "-af",
             "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        measured = edit.parse_loudnorm(loud.stderr)
        assert measured is not None
        assert float(measured["input_i"]) == pytest.approx(-14.0, abs=1.0)
