"""GTA Mode: store, settings, signals, Hype Curve, judging, render, QA, queue, API."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from autoclip import app as app_module
from autoclip.app import create_app
from autoclip.config import ExportSettings
from autoclip.gta import align, fetch, hype, judging, qa, render, signals, store
from autoclip.gta import settings as gta_settings
from autoclip.gta.gate import HeavyGate
from autoclip.gta.settings import GtaBias, GtaSettings
from autoclip.pipeline import ffmpeg
from autoclip.pipeline.transcript import Transcript, Word
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _db(initialised_db: int) -> None:
    """Every GTA test runs against a migrated temp database."""


def _words(text: str, start: float, step: float = 0.4) -> list[Word]:
    out = []
    t = start
    for token in text.split():
        out.append(Word(text=token, start=t, end=t + step * 0.8))
        t += step
    return out


# --------------------------------------------------------------------------
# Store + settings
# --------------------------------------------------------------------------


class TestStore:
    def test_stream_roundtrip_and_status_validation(self) -> None:
        stream = store.create_stream("https://www.twitch.tv/videos/123", range_start_s=60.0)
        store.update_stream(stream.id, status="probing", signals={"a": 1}, title="T")
        got = store.get_stream(stream.id)
        assert got is not None
        assert got.status == "probing" and got.signals == {"a": 1} and got.title == "T"
        assert got.range_start_s == 60.0 and got.range_end_s is None
        with pytest.raises(ValueError):
            store.update_stream(stream.id, status="bogus")
        with pytest.raises(ValueError):
            store.update_stream(stream.id, nonsense=1)

    def test_active_statuses_exclude_live_and_terminal(self) -> None:
        for status in ("done", "failed", "cancelled", "live_watching", "live_stopped"):
            assert status not in store.ACTIVE_STREAM_STATUSES
        assert "judging" in store.ACTIVE_STREAM_STATUSES

    def test_clip_json_columns_and_cascade(self) -> None:
        stream = store.create_stream("https://youtu.be/x")
        clip = store.create_clip(
            store.GtaClip(
                id="c1",
                stream_id=stream.id,
                start_s=1.0,
                end_s=20.0,
                status="approved",
                hashtags=["#a", "#b"],
                verdict={"payoff": {"answer": "yes"}},
                qa={"passed": True},
            )
        )
        got = store.get_clip(clip.id)
        assert got is not None and got.hashtags == ["#a", "#b"] and got.qa == {"passed": True}
        approved = store.list_clips(stream.id, statuses=frozenset({"approved"}))
        assert [c.id for c in approved] == ["c1"]
        store.delete_stream(stream.id)
        assert store.get_clip("c1") is None

    def test_reports(self) -> None:
        assert store.latest_report() is None
        store.create_report("https://youtube.com/@x", {"count": 3})
        report = store.latest_report()
        assert report is not None and report["report"] == {"count": 3}


class TestSettings:
    def test_defaults_and_roundtrip_keep_other_sections(self, autoclip_home: Path) -> None:
        config = autoclip_home / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"live": {"enabled": False}, "active_provider": "x"}))
        settings = gta_settings.load()
        assert settings.chunk_minutes == 12.0 and settings.max_clip_s < 60
        settings.max_clips_per_stream = 4
        gta_settings.save(settings)
        raw = json.loads(config.read_text())
        assert raw["live"] == {"enabled": False} and raw["active_provider"] == "x"
        assert gta_settings.load().max_clips_per_stream == 4

    def test_invalid_section_falls_back(self) -> None:
        fallback = gta_settings.parse_section({"max_clip_s": 75})
        assert fallback.max_clip_s == GtaSettings().max_clip_s
        assert gta_settings.parse_section("junk").chunk_minutes == 12.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            GtaSettings(max_clip_s=60)
        with pytest.raises(ValueError):
            GtaSettings(min_clip_s=30, max_clip_s=20)


# --------------------------------------------------------------------------
# Fetch helpers (no network)
# --------------------------------------------------------------------------


class TestFetch:
    @pytest.mark.parametrize(
        ("url", "platform"),
        [
            ("https://www.youtube.com/watch?v=abc", "youtube"),
            ("https://youtu.be/abc", "youtube"),
            ("https://www.twitch.tv/videos/2924282282", "twitch"),
            ("https://kick.com/video/abc", "kick"),
            ("https://example.com/x", "other"),
        ],
    )
    def test_detect_platform(self, url: str, platform: str) -> None:
        assert fetch.detect_platform(url) == platform

    def test_supported_url(self) -> None:
        assert fetch.is_supported_url("https://www.twitch.tv/videos/1")
        assert not fetch.is_supported_url("ftp://twitch.tv/videos/1")
        assert not fetch.is_supported_url("not a url")

    def test_parse_heatmap_and_probe(self) -> None:
        info = {
            "id": "v",
            "title": "Big stream",
            "channel": "Streamer",
            "uploader_id": "streamer",
            "duration": 3600,
            "live_status": "was_live",
            "heatmap": [
                {"start_time": 10, "end_time": 20, "value": 0.9},
                {"start_time": 0, "end_time": 10, "value": 2},
                {"bad": 1},
            ],
            "subtitles": {"live_chat": [{}]},
        }
        result = fetch.probe_from_info(info, "https://www.youtube.com/watch?v=v")
        assert result.heatmap == [(0.0, 10.0, 1.0), (10.0, 20.0, 0.9)]
        assert result.has_live_chat_replay and not result.is_live
        assert result.duration_s == 3600 and result.channel == "Streamer"
        live = fetch.probe_from_info({"is_live": True}, "https://twitch.tv/x")
        assert live.is_live

    def test_twitch_vod_id(self) -> None:
        assert fetch.twitch_vod_id("https://www.twitch.tv/videos/2924282282?t=1h") == "2924282282"
        assert fetch.twitch_vod_id("https://www.twitch.tv/kaicenat") is None

    def test_find_media_ignores_partials(self, tmp_path: Path) -> None:
        (tmp_path / "c1.mp4.part").write_bytes(b"x" * 10)
        (tmp_path / "c1.offset.json").write_text("{}")
        assert fetch._find_media(tmp_path, "c1") is None
        (tmp_path / "c1.mp4").write_bytes(b"x" * 5)
        assert fetch._find_media(tmp_path, "c1") == tmp_path / "c1.mp4"


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


class TestSignals:
    def test_youtube_chat_line(self) -> None:
        renderer = {
            "message": {"runs": [{"text": "CLIP IT "}, {"emoji": {"shortcuts": [":skull:"]}}]}
        }
        line = json.dumps(
            {
                "replayChatItemAction": {
                    "videoOffsetTimeMsec": "61500",
                    "actions": [
                        {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": renderer}}}
                    ],
                }
            }
        )
        messages = signals.parse_youtube_chat_line(line)
        assert messages == [signals.ChatMessage(t=61.5, text="CLIP IT :skull:")]
        assert signals.parse_youtube_chat_line("not json") == []
        assert signals.parse_youtube_chat_line(json.dumps({"other": 1})) == []

    def test_keyword_hits(self) -> None:
        assert signals.keyword_hits("clip that LMAO") == (1, 1, 0)
        assert signals.keyword_hits("???????")[2] == 1
        assert signals.keyword_hits("what?")[2] == 0
        assert signals.keyword_hits("💀💀💀")[1] == 3

    def test_chat_series_applies_lag(self) -> None:
        series = signals.chat_series(
            [signals.ChatMessage(16.2, "clip"), signals.ChatMessage(2.0, "early")], 30
        )
        lagged = int(16.2 - signals.CHAT_LAG_S)
        assert series.rate[lagged] == 1 and series.clip[lagged] == 1
        assert series.rate.sum() == 1  # the early message lands before t=0 and is dropped

    def test_wav_loudness_per_second(self, tmp_path: Path) -> None:
        rate = 16000
        t = np.arange(rate * 3) / rate
        samples = np.zeros_like(t)
        samples[rate : 2 * rate] = 0.5 * np.sin(2 * np.pi * 440 * t[rate : 2 * rate])
        path = tmp_path / "a.wav"
        _write_wav(path, samples)
        db = signals.wav_loudness_per_second(path)
        assert len(db) == 3
        assert db[0] == signals.SILENCE_DB and db[2] == signals.SILENCE_DB
        assert -12 < db[1] < -6  # RMS of a 0.5 sine is about -9 dBFS

    def test_loudness_for_parts_places_parts_by_index(self, tmp_path: Path) -> None:
        a = tmp_path / "part_0000.wav"
        b = tmp_path / "part_0001.wav"
        _write_wav(a, np.zeros(16000 * 2))
        _write_wav(b, np.full(16000 * 2, 0.25))
        out = signals.loudness_for_parts([a, b], part_s=3.0)  # part a is 1 s short
        assert len(out) == 5
        assert out[2] == signals.SILENCE_DB and out[3] > -15

    def test_series_helpers(self) -> None:
        heat = signals.heatmap_series([(2.0, 4.0, 0.8)], 6)
        assert list(heat) == pytest.approx([0, 0, 0.8, 0.8, 0, 0])
        clips = signals.clips_series([signals.ViewerClip(1.0, 2.0, 0)], 5)
        assert clips[1] == pytest.approx(1.0) and clips[3] == 0

    def test_twitch_viewer_clips_filters_by_vod(self) -> None:
        def get(url: str, params: dict, headers: dict) -> dict:
            assert headers["Authorization"] == "Bearer tok"
            if url.endswith("/users"):
                return {"data": [{"id": "42"}]}
            if url.endswith("/videos"):
                return {"data": [{"created_at": "2026-11-01T00:00:00Z"}]}
            return {
                "data": [
                    {"video_id": "999", "vod_offset": 120, "duration": 30, "view_count": 50},
                    {"video_id": "111", "vod_offset": 5, "duration": 30, "view_count": 50},
                    {"video_id": "999", "vod_offset": None},
                ],
                "pagination": {},
            }

        clips = signals.twitch_viewer_clips(
            "999",
            "streamer",
            credentials=("id", "secret"),
            http_get=get,
            http_post=lambda url, params: {"access_token": "tok"},
        )
        assert [c.offset_s for c in clips] == [120.0]


# --------------------------------------------------------------------------
# Hype Curve
# --------------------------------------------------------------------------


class TestHype:
    def _loudness(self, seconds: int, spike_at: int, spike_db: float = -8.0) -> np.ndarray:
        rng = np.random.default_rng(1)
        db = -30 + rng.normal(0, 1.0, seconds).astype(np.float32)
        db[spike_at : spike_at + 4] = spike_db
        return db

    def test_stream_edges_are_not_spikes(self) -> None:
        curve = hype.build_curve(1200, loudness_db=self._loudness(1200, 600))
        assert curve.score[0] < 10 and curve.score[-1] < 10
        assert all(10 < m.t < 1190 for m in curve.markers)

    def test_audio_spike_becomes_a_marker(self) -> None:
        curve = hype.build_curve(1200, loudness_db=self._loudness(1200, 600))
        assert curve.available == ["audio"]
        assert any(abs(m.t - 601) <= 4 for m in curve.markers)
        marker = min(curve.markers, key=lambda m: abs(m.t - 601))
        assert any("louder" in r for r in marker.reasons)
        assert curve.peak(595, 610) > curve.peak(100, 200)

    def test_chat_and_audio_agree(self) -> None:
        seconds = 1200
        messages = [signals.ChatMessage(t=float(t), text="hi") for t in range(0, seconds, 3)]
        messages += [signals.ChatMessage(t=606 + i * 0.05, text="CLIP IT LMAO") for i in range(200)]
        chat = signals.chat_series(messages, seconds)
        mild = self._loudness(seconds, 600, spike_db=-26.0)
        both = hype.build_curve(seconds, loudness_db=mild, chat=chat)
        audio_only = hype.build_curve(seconds, loudness_db=mild)
        assert set(both.available) == {"audio", "chat", "keywords"}
        assert both.peak(595, 615) > audio_only.peak(595, 615)
        reasons = " ".join(r for m in both.markers_between(590, 620) for r in m.reasons)
        assert "chat" in reasons and '"clip"' in reasons

    def test_no_signals(self) -> None:
        curve = hype.build_curve(100)
        assert curve.available == [] and curve.markers == [] and float(curve.score.max()) == 0

    def test_save_load(self, tmp_path: Path) -> None:
        curve = hype.build_curve(900, loudness_db=self._loudness(900, 450))
        curve.save(tmp_path / "h.json")
        loaded = hype.HypeCurve.load(tmp_path / "h.json")
        assert len(loaded.score) == 900 and len(loaded.markers) == len(curve.markers)


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------


def _long_transcript(minutes: int = 30) -> Transcript:
    words: list[Word] = []
    t = 0.0
    i = 0
    while t < minutes * 60:
        text = f"word{i}." if i % 12 == 11 else f"word{i}"
        words.append(Word(text=text, start=t, end=t + 0.3))
        t += 0.5
        i += 1
    return Transcript(words=words)


class FakeModel:
    """A scripted provider: returns queued responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    async def __call__(self, system: str, user: str, config) -> str:
        self.prompts.append((system, user))
        return self.responses.pop(0) if self.responses else '{"moments": []}'


class TestChunks:
    def test_exact_twelve_minute_chunks(self) -> None:
        transcript = _long_transcript(30)
        chunks = judging.build_chunks(transcript, chunk_s=720, tail_s=60)
        assert [(c.start_s, c.end_s) for c in chunks[:2]] == [(0, 720), (720, 1440)]
        # The last chunk runs from its boundary to the last spoken word.
        assert chunks[2].start_s == 1440 and chunks[2].end_s == pytest.approx(1800, abs=1)
        first, second = chunks[0], chunks[1]
        assert transcript.words[first.last_word].start < 720
        assert transcript.words[second.first_word].start >= 720
        assert first.tail_word > first.last_word
        assert transcript.words[first.tail_word].start < 780

    def test_range_start(self) -> None:
        chunks = judging.build_chunks(_long_transcript(30), chunk_s=720, tail_s=0, start_s=600)
        assert chunks[0].start_s == 600 and chunks[0].end_s == 1320

    def test_render_marks_tail(self) -> None:
        transcript = _long_transcript(2)
        chunk = judging.build_chunks(transcript, chunk_s=60, tail_s=10)[0]
        text = judging.render_chunk(transcript, chunk)
        assert "TAIL (read-only" in text and text.startswith("[0]word0")


class TestFiller:
    @pytest.mark.parametrize(
        ("text", "filler"),
        [
            ("uh what is that", "uh"),
            ("So yeah this happened", "so"),
            ("and then he", "and"),
            ("I mean whatever", "i mean"),
            ("WAIT what is that", None),
            ("Bro no way", None),
        ],
    )
    def test_opens_on_filler(self, text: str, filler: str | None) -> None:
        transcript = Transcript(words=_words(text, 0))
        assert judging.opens_on_filler(transcript, 0) == filler


def _chunk_response(moments: list[dict]) -> str:
    return json.dumps({"moments": moments})


def _final_response(verdicts: list[dict]) -> str:
    return json.dumps({"verdicts": verdicts})


def _yes(cid: str, score: int = 80, title: str = "He found WHAT") -> dict:
    return {
        "id": cid,
        "scroll_stop": {"answer": "yes", "why": "scream"},
        "context_free": {"answer": "yes", "why": "clear"},
        "payoff": {"answer": "yes", "why": "lands"},
        "score": score,
        "hook_title": title,
        "caption": "Nobody expected this.",
        "hashtags": ["#Streamer", "gta", "#gta", "#shorts!"],
        "moment_type": "freakout",
    }


class TestJudging:
    def _setup(self):
        text = (
            "uh so yeah anyway. WAIT what is that thing in the water. "
            "No way no way no way it is a shark. I can not believe it. "
            "Chat that was insane. okay moving on now. " * 3
        )
        transcript = Transcript(words=_words(text, 0.0, step=0.6))
        chunks = judging.build_chunks(transcript, chunk_s=720, tail_s=60)
        return transcript, chunks

    def _find(self, transcript, chunks, model, settings=None):
        return asyncio.run(
            judging.find_candidates(
                transcript,
                chunks,
                complete=model,
                settings=settings or GtaSettings(min_clip_s=5, max_clip_s=30),
                curve=None,
                silences=[],
                streamer="Streamer",
                title="Stream",
            )
        )

    def _index(self, transcript, text: str, nth: int = 0) -> int:
        return [i for i, w in enumerate(transcript.words) if w.text == text][nth]

    def test_phase1_rejects_filler_and_context(self) -> None:
        transcript, chunks = self._setup()
        wait = self._index(transcript, "WAIT")
        model = FakeModel(
            [
                _chunk_response(
                    [
                        {"start_word": 0, "end_word": 30, "hook_quote": "uh so yeah"},
                        {
                            "start_word": wait,
                            "end_word": wait + 22,
                            "hook_quote": "WAIT what is that",
                            "needs_prior_context": True,
                        },
                        {
                            "start_word": wait,
                            "end_word": wait + 22,
                            "hook_quote": "WAIT what is that",
                            "moment_type": "Crazy Discovery",
                        },
                        {"start_word": 99999, "end_word": 100000},
                    ]
                )
            ]
        )
        found = self._find(transcript, chunks, model)
        assert len(found) == 3  # the out-of-chunk proposal is dropped outright
        assert found[0].status == "rejected" and "filler" in found[0].reject_reason
        assert found[1].status == "rejected" and "prior context" in found[1].reject_reason
        assert found[2].status == "candidate" and found[2].moment_type == "crazy_discovery"
        assert transcript.words[found[2].start_word].text == "WAIT"
        assert 5 <= found[2].end_s - found[2].start_s <= 30
        system, user = model.prompts[0]
        assert "{min_s}" not in system and "HYPE MARKERS" in user

    def test_hook_quote_moves_start_off_filler(self) -> None:
        transcript, chunks = self._setup()
        wait = self._index(transcript, "WAIT")
        moment = {"start_word": wait - 1, "end_word": wait + 22, "hook_quote": "WAIT what is that"}
        found = self._find(transcript, chunks, FakeModel([_chunk_response([moment])]))
        assert found[0].status == "candidate" and found[0].start_word == wait

    def test_final_round_one_no_rejects(self) -> None:
        transcript, chunks = self._setup()
        wait = self._index(transcript, "WAIT")
        second = self._index(transcript, "WAIT", 1)
        model = FakeModel(
            [
                _chunk_response(
                    [
                        {"start_word": wait, "end_word": wait + 22, "hook_quote": "WAIT what"},
                        {"start_word": second, "end_word": second + 22, "hook_quote": "WAIT what"},
                    ]
                ),
            ]
        )
        found = self._find(transcript, chunks, model)
        ids = [c.id for c in found]
        bad = _yes(ids[1])
        bad["payoff"] = {"answer": "no", "why": "fizzles"}
        model.responses = [_final_response([_yes(ids[0]), bad])]
        result = asyncio.run(
            judging.final_round(
                transcript,
                found,
                complete=model,
                settings=GtaSettings(min_clip_s=5, max_clip_s=30),
                streamer="Streamer",
                title="Stream",
            )
        )
        assert result[0].status == "approved"
        assert result[0].hashtags == ["#streamer", "#gta", "#shorts"]
        assert result[0].hook_title == "He found WHAT"
        assert result[1].status == "rejected" and "no payoff" in result[1].reject_reason

    def test_head_to_head_dedupes_and_caps(self) -> None:
        settings = GtaSettings(min_clip_s=5, max_clip_s=30, max_clips_per_stream=1)
        a = judging.Candidate("a", 0, 0, 1, 10.0, 30.0, "", "freakout", "", 50)
        b = judging.Candidate("b", 0, 0, 1, 12.0, 32.0, "", "freakout", "", 50)
        c = judging.Candidate("c", 0, 0, 1, 100.0, 120.0, "", "freakout", "", 50)
        for cand, score in ((a, 90), (b, 70), (c, 60)):
            cand.status = "approved"
            cand.rank_score = score
            cand.hook_title = cand.id
        judging._head_to_head([a, b, c], settings)
        assert a.status == "approved"
        assert b.status == "rejected" and "Same moment" in b.reject_reason
        assert c.status == "rejected" and "head-to-head" in c.reject_reason

    def test_missing_verdict_and_malformed_retry(self) -> None:
        transcript, chunks = self._setup()
        wait = self._index(transcript, "WAIT")
        moment = {"start_word": wait, "end_word": wait + 22, "hook_quote": "WAIT"}
        model = FakeModel(["this is not json", _chunk_response([moment])])
        found = self._find(transcript, chunks, model)
        assert len(found) == 1 and len(model.prompts) == 2  # one repair retry
        model.responses = [_final_response([])]
        asyncio.run(
            judging.final_round(
                transcript, found, complete=model, settings=GtaSettings(), streamer="", title=""
            )
        )
        assert found[0].status == "rejected" and "no verdict" in found[0].reject_reason

    def test_twice_malformed_raises(self) -> None:
        from autoclip.providers.base import ProviderError

        transcript, chunks = self._setup()
        with pytest.raises(ProviderError):
            self._find(transcript, chunks, FakeModel(["nope", "still nope"]))

    def test_bias_multiplier_is_bounded(self) -> None:
        bias = GtaBias(moment_types={"freakout": 5.0, "hot_take": 0.1}, streamers={"Kai": 2.0})
        assert judging.bias_multiplier(bias, "freakout", "kai") == 1.6
        assert judging.bias_multiplier(bias, "hot_take", "x") == 0.6
        assert judging.bias_multiplier(GtaBias(), "other", "x") == 1.0


# --------------------------------------------------------------------------
# Render (pure parts)
# --------------------------------------------------------------------------


class TestPlanKeeps:
    def test_quiet_gap_without_words_is_cut(self) -> None:
        words = _words("a b c d e", 0.0, 0.4) + _words("f g h", 5.0, 0.4)
        levels = np.full(int(8 / 0.05), -20.0, dtype=np.float32)
        levels[int(2.2 / 0.05) : int(4.9 / 0.05)] = -70.0
        keeps = render.plan_keeps(words, 0.0, 8.0, levels=levels, min_gap_s=0.6, keep_s=0.12)
        assert len(keeps) == 2
        assert render.output_duration(keeps) < 8.0 - 2.0

    def test_loud_gap_without_words_is_kept(self) -> None:
        """A scream Whisper didn't transcribe is the clip — never cut it."""
        words = _words("a b c", 0.0, 0.4) + _words("d e", 5.0, 0.4)
        levels = np.full(int(8 / 0.05), -20.0, dtype=np.float32)
        keeps = render.plan_keeps(words, 0.0, 8.0, levels=levels)
        assert keeps == [render.Keep(0.0, 8.0)]

    def test_no_levels_keeps_everything(self) -> None:
        keeps = render.plan_keeps(_words("a", 0.0), 0.0, 5.0, levels=None)
        assert keeps == [render.Keep(0.0, 5.0)]

    def test_remap_words(self) -> None:
        keeps = [render.Keep(0.0, 2.0), render.Keep(5.0, 7.0)]
        words = [Word("a", 0.5, 0.9), Word("b", 5.5, 6.0), Word("gone", 3.0, 3.5)]
        out = render.remap_words(words, keeps)
        assert [(w.text, round(w.start, 2)) for w in out] == [("a", 0.5), ("b", 2.5)]

    def test_cap_duration(self) -> None:
        keeps = [render.Keep(0, 30), render.Keep(40, 70)]
        capped = render._cap_duration(keeps, 58.0)
        assert render.output_duration(capped) == pytest.approx(58.0)

    def test_filtergraph_shape(self) -> None:
        graph = render.build_filtergraph(
            [render.Keep(10.0, 12.0), render.Keep(13.0, 15.0)],
            seek_s=9.0,
            subtitle_name="captions.ass",
            fonts_name="fonts",
            audio_filter="loudnorm=I=-14",
        )
        assert "trim=start=1.0000:end=3.0000" in graph
        assert "concat=n=2:v=1:a=1" in graph
        assert "force_original_aspect_ratio=decrease" in graph  # uncropped frame
        assert "boxblur" in graph and "ass=filename=captions.ass" in graph
        assert graph.endswith("[acat]loudnorm=I=-14[aout]")


class TestQaHelpers:
    def test_edge_trims(self) -> None:
        assert qa.edge_trims([(0.0, 0.4)], 30.0) == (0.45, 0.0)
        assert qa.edge_trims([(29.5, 30.0)], 30.0) == (0.0, pytest.approx(0.55))
        assert qa.edge_trims([(10.0, 10.5)], 30.0) is None


class TestGate:
    def test_live_jumps_the_queue(self) -> None:
        import threading

        gate = HeavyGate()
        order: list[str] = []
        release = threading.Event()

        def holder() -> None:
            with gate.hold():
                release.wait(5)
                order.append("first")

        def batch() -> None:
            with gate.hold():
                order.append("batch")

        def live() -> None:
            with gate.hold(live=True):
                order.append("live")

        t1 = threading.Thread(target=holder)
        t1.start()
        time.sleep(0.1)
        t2 = threading.Thread(target=batch)
        t2.start()
        time.sleep(0.1)
        t3 = threading.Thread(target=live)
        t3.start()
        time.sleep(0.1)
        release.set()
        for t in (t1, t2, t3):
            t.join(5)
        assert order == ["first", "live", "batch"]


# --------------------------------------------------------------------------
# Align
# --------------------------------------------------------------------------


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [ffmpeg.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


class TestAlign:
    def test_best_lag_finds_offset(self) -> None:
        rng = np.random.default_rng(3)
        reference = rng.random(3000).astype(np.float32)
        probe = reference[1234:1734].copy()
        lag, score = align.best_lag(reference, probe)
        assert lag == 1234 and score > 0.99

    def test_section_offset_from_real_audio(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(5)
        rate = 16000
        # 60 s of "stream" audio: noise bursts of varying loudness.
        env = np.repeat(rng.random(600) ** 3, rate // 10)
        stream_audio = (rng.normal(0, 1, len(env)) * env * 0.3).astype(np.float32)
        part = tmp_path / "parts" / "part_0000.wav"
        _write_wav(part, stream_audio)
        # The "downloaded section" really starts at 23.4 s (26 s was requested).
        section = tmp_path / "section.m4a"
        _ffmpeg("-ss", "23.4", "-t", "20", "-i", str(part), "-c:a", "aac", str(section))
        start, confidence = align.section_offset(
            section, expected_start_s=26.0, parts=[part], part_s=1800, range_start_s=0.0
        )
        assert confidence > 0.8
        assert start == pytest.approx(23.4, abs=0.05)

    def test_silence_falls_back(self, tmp_path: Path) -> None:
        part = tmp_path / "part_0000.wav"
        _write_wav(part, np.zeros(16000 * 30))
        section = tmp_path / "s.wav"
        _write_wav(section, np.zeros(16000 * 10))
        start, confidence = align.section_offset(
            section, expected_start_s=5.0, parts=[part], part_s=1800, range_start_s=0.0
        )
        assert start == 5.0 and confidence < align.MIN_CORRELATION


# --------------------------------------------------------------------------
# Real render + QA (ffmpeg)
# --------------------------------------------------------------------------


def _make_section(path: Path, *, duration: float = 24.0, black: tuple[float, float] | None = None):
    """16:9 test video; a tone with a 4 s dead-air gap at 8-12 s."""
    video = f"testsrc2=size=1280x720:rate=30:duration={duration}"
    if black:
        a, b = black
        video += f",drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='between(t,{a},{b})'"
    _ffmpeg(
        "-f", "lavfi", "-i", video,
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={duration}",
        "-af", "volume=volume='if(between(t,8,12),0,0.4)':eval=frame",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    )  # fmt: skip


def _speech_words() -> list[Word]:
    return _words("this is the moment it happens right now look", 1.0, 0.7) + _words(
        "and then the payoff lands hard here", 12.5, 0.7
    )


def _render(section: Path, destination: Path, work: Path, offset: float, start: float, end: float):
    words = [Word(w.text, w.start + offset, w.end + offset) for w in _speech_words()]
    return render.render(
        render.RenderJob(
            section=section,
            section_offset_s=offset,
            start_s=start,
            end_s=end,
            words=words,
            hook_title="He did WHAT",
            destination=destination,
            work_dir=work,
        ),
        style_key="bold_pop",
        title_s=3.0,
        dead_air_min_s=0.6,
        dead_air_keep_s=0.12,
        max_duration_s=58.0,
        loudness_lufs=-14.0,
        true_peak_db=-1.0,
        export_settings=ExportSettings(prefer_hardware_encoder=False),
    )


def _qa(destination: Path, section: Path, a: float, b: float, face: float):
    return qa.run_qa(
        destination,
        section=section,
        section_start_s=a,
        section_end_s=b,
        target_lufs=-14.0,
        tolerance_lu=1.0,
        true_peak_db=-1.0,
        face_min_ratio=0.5,
        face_ratio_fn=lambda *_: face,
    )


class TestRealRender:
    def test_render_uncropped_with_dead_air_cut(self, tmp_path: Path) -> None:
        section = tmp_path / "section.mp4"
        _make_section(section)
        destination = tmp_path / "out" / "clip.mp4"
        # The section starts at stream time 100 s.
        result = _render(section, destination, tmp_path / "work", 100.0, 100.5, 118.0)
        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert result.cut_s > 2.5  # most of the 4 s gap is gone
        assert info.duration_s == pytest.approx(result.duration_s, abs=0.25)
        ass = (tmp_path / "work" / "captions.ass").read_text(encoding="utf-8")
        assert "GtaHook" in ass and "HE DID WHAT" in ass
        assert not list((tmp_path / "out").glob("*.partial.mp4"))

        report = _qa(destination, section, 0.5, 18.0, face=1.0)
        assert report.length.ok, report.length.detail
        assert report.loudness.ok, report.loudness.detail
        assert report.black_frames.ok, report.black_frames.detail
        assert report.facecam.ok and report.passed

    def test_qa_catches_black_frames_and_missing_face(self, tmp_path: Path) -> None:
        section = tmp_path / "section.mp4"
        _make_section(section, black=(5.0, 6.0))
        destination = tmp_path / "clip.mp4"
        _render(section, destination, tmp_path / "work", 0.0, 1.0, 18.0)
        report = _qa(destination, section, 1.0, 18.0, face=0.1)
        assert not report.black_frames.ok and report.black_spans
        assert not report.facecam.ok and "10%" in report.facecam.detail
        assert not report.passed

    def test_fix_loudness(self, tmp_path: Path) -> None:
        clip = tmp_path / "quiet.mp4"
        _ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=8",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
            "-af", "volume=0.02", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip),
        )  # fmt: skip
        before, _ = qa.measure_loudness(clip)
        assert before is not None and before < -25
        qa.fix_loudness(clip, before, -14.0, -1.0)
        after, peak = qa.measure_loudness(clip)
        assert after == pytest.approx(-14.0, abs=1.0)
        assert peak is not None and peak <= -0.5


# --------------------------------------------------------------------------
# Queue (stages faked) + API
# --------------------------------------------------------------------------


@pytest.fixture
def fake_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from autoclip.gta import pipeline

    calls: list[str] = []

    def probe(stream, report):
        calls.append("probe")
        store.update_stream(stream.id, duration_s=600.0, channel="Streamer", title="T")

    def judge(stream, report, settings, complete=None):
        calls.append("judge")
        for suffix, status, start in (("a", "approved", 10), ("r", "rejected", 50)):
            store.create_clip(
                store.GtaClip(
                    id=f"{stream.id}-{suffix}",
                    stream_id=stream.id,
                    start_s=start,
                    end_s=start + 20,
                    status=status,
                )
            )

    def render_clip(clip, stream, report, settings):
        calls.append("render")
        path = tmp_path / f"{clip.id}.mp4"
        path.write_bytes(b"mp4")
        return {"path": str(path), "qa": {"passed": True}}

    monkeypatch.setattr(pipeline, "stage_probe", probe)
    monkeypatch.setattr(pipeline, "stage_audio", lambda s, r, st: calls.append("audio"))
    monkeypatch.setattr(pipeline, "stage_signals", lambda s, r: calls.append("signals"))
    monkeypatch.setattr(pipeline, "stage_transcribe", lambda s, r: calls.append("transcribe"))
    monkeypatch.setattr(pipeline, "stage_hype", lambda s, r: calls.append("hype"))
    monkeypatch.setattr(pipeline, "stage_judge", judge)
    monkeypatch.setattr(pipeline, "stage_section", lambda c, s, r, st: calls.append("section"))
    monkeypatch.setattr(pipeline, "stage_render", render_clip)
    return calls


async def _wait_for(predicate, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await asyncio.to_thread(predicate):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


class TestQueue:
    async def test_streams_run_every_stage_then_clips(self, fake_stages) -> None:
        from autoclip.gta.queue import GtaQueue

        q = GtaQueue()
        await q.start()
        try:
            a = store.create_stream("https://www.twitch.tv/videos/1")
            b = store.create_stream("https://www.twitch.tv/videos/2")
            q.notify()
            await _wait_for(lambda: all(store.get_stream(s.id).status == "done" for s in (a, b)))
        finally:
            await q.stop()
        for s in (a, b):
            clips = {c.id: c for c in store.list_clips(s.id)}
            assert clips[f"{s.id}-a"].status == "ready"
            assert clips[f"{s.id}-r"].status == "rejected"
            assert "1 clip(s) ready" in store.get_stream(s.id).detail
        assert fake_stages.count("transcribe") == 2 and fake_stages.count("render") == 2

    async def test_failure_then_retry_resumes_at_failed_stage(
        self, fake_stages, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoclip.gta import pipeline
        from autoclip.gta.queue import GtaQueue
        from autoclip.pipeline.ingest import IngestError

        state = {"fail": True}

        def flaky(stream, report):
            if state["fail"]:
                raise IngestError("network down", hint="try again")
            fake_stages.append("signals")

        monkeypatch.setattr(pipeline, "stage_signals", flaky)
        q = GtaQueue()
        await q.start()
        try:
            s = store.create_stream("https://youtu.be/abc")
            q.notify()
            await _wait_for(lambda: store.get_stream(s.id).status == "failed")
            failed = store.get_stream(s.id)
            assert failed.stage == "fetching_signals" and "try again" in failed.error
            state["fail"] = False
            store.update_stream(s.id, status=failed.stage, error=None)
            q.notify()
            await _wait_for(lambda: store.get_stream(s.id).status == "done")
        finally:
            await q.stop()
        assert fake_stages.count("probe") == 1  # earlier stages did not run again

    async def test_cancel_stops_a_running_stage(self, fake_stages, monkeypatch) -> None:
        from autoclip.gta import pipeline
        from autoclip.gta.pipeline import StreamCancelled
        from autoclip.gta.queue import GtaQueue

        def slow(stream, report, settings):
            for _ in range(250):
                if report.cancelled():
                    raise StreamCancelled("x")
                time.sleep(0.02)

        monkeypatch.setattr(pipeline, "stage_audio", slow)
        q = GtaQueue()
        await q.start()
        try:
            s = store.create_stream("https://youtu.be/abc")
            q.notify()
            await _wait_for(lambda: store.get_stream(s.id).status == "downloading_audio")
            await asyncio.sleep(0.2)
            store.update_stream(s.id, status="cancelled")
            q.cancel(s.id)
            await _wait_for(lambda: not q.busy_keys())
        finally:
            await q.stop()
        assert store.get_stream(s.id).status == "cancelled"
        assert "signals" not in fake_stages


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


class TestApi:
    def test_add_streams_validates_and_dedupes(self, client: TestClient) -> None:
        urls = [
            "https://www.twitch.tv/videos/1",
            " https://www.twitch.tv/videos/1 ",
            "https://example.com/x",
            "https://youtu.be/abc",
        ]
        response = client.post("/api/gta/streams", json={"urls": urls})
        assert response.status_code == 201
        body = response.json()
        assert [s["url"] for s in body["added"]] == [urls[0], urls[3]]
        assert body["added"][0]["platform"] == "twitch"
        assert body["skipped"][0]["url"] == "https://example.com/x"
        again = client.post("/api/gta/streams", json={"urls": ["https://youtu.be/abc"]}).json()
        assert again["added"] == [] and "Already" in again["skipped"][0]["reason"]
        assert len(client.get("/api/gta/streams").json()) == 2

    def test_bad_range(self, client: TestClient) -> None:
        response = client.post(
            "/api/gta/streams",
            json={"urls": ["https://youtu.be/a"], "range_start_s": 50, "range_end_s": 10},
        )
        assert response.status_code == 400

    def test_cancel_retry_delete(self, client: TestClient) -> None:
        added = client.post("/api/gta/streams", json={"urls": ["https://youtu.be/a"]}).json()
        sid = added["added"][0]["id"]
        assert client.delete(f"/api/gta/streams/{sid}").status_code == 409  # still active
        assert client.post(f"/api/gta/streams/{sid}/retry").status_code == 409
        assert client.post(f"/api/gta/streams/{sid}/cancel").json()["status"] == "cancelled"
        assert client.post(f"/api/gta/streams/{sid}/retry").json()["status"] == "queued"
        client.post(f"/api/gta/streams/{sid}/cancel")
        assert client.delete(f"/api/gta/streams/{sid}").status_code == 204
        assert client.get(f"/api/gta/streams/{sid}").status_code == 404

    def test_clips_and_video(self, client: TestClient, tmp_path: Path) -> None:
        stream = store.create_stream("https://youtu.be/v")
        video = tmp_path / "c.mp4"
        video.write_bytes(b"data")
        store.create_clip(
            store.GtaClip(
                id="clip1",
                stream_id=stream.id,
                start_s=0,
                end_s=20,
                status="ready",
                caption="Wild.",
                hashtags=["#a", "#b"],
                path=str(video),
            )
        )
        clips = client.get("/api/gta/clips", params={"status": "ready"}).json()
        assert clips[0]["post_text"] == "Wild.\n\n#a #b" and clips[0]["has_video"]
        assert client.get("/api/gta/clips/clip1/video").content == b"data"
        assert client.get("/api/gta/clips/nope/video").status_code == 404

    def test_settings_roundtrip_and_validation(self, client: TestClient) -> None:
        current = client.get("/api/gta/settings").json()
        assert current["chunk_minutes"] == 12 and current["twitch_credentials_set"] is False
        saved = client.put("/api/gta/settings", json={"max_clips_per_stream": 3}).json()
        assert saved["max_clips_per_stream"] == 3
        assert client.put("/api/gta/settings", json={"max_clip_s": 90}).status_code == 400
        # The learned bias can't be overwritten from the form.
        client.put("/api/gta/settings", json={"bias": {"summary": "hacked"}})
        assert client.get("/api/gta/settings").json()["bias"]["summary"] == ""

    def test_twitch_credentials(self, client: TestClient, fake_keyring) -> None:
        half = client.put("/api/gta/twitch-credentials", json={"client_id": "x"})
        assert half.status_code == 400
        ok = client.put(
            "/api/gta/twitch-credentials", json={"client_id": "id", "client_secret": "sec"}
        )
        assert ok.json() == {"twitch_credentials_set": True}
        assert client.get("/api/gta/settings").json()["twitch_credentials_set"] is True
        client.put("/api/gta/twitch-credentials", json={})
        assert client.get("/api/gta/settings").json()["twitch_credentials_set"] is False


# --------------------------------------------------------------------------
# Feedback loop
# --------------------------------------------------------------------------


class TestFeedback:
    def test_analyse_and_bias(self) -> None:
        from autoclip.gta import feedback

        shorts = [
            feedback.PostedShort(f"s{i}", 1000, moment_type="freakout", streamer="Kai")
            for i in range(4)
        ]
        shorts += [
            feedback.PostedShort(f"h{i}", 100, moment_type="hot_take", streamer="Other")
            for i in range(4)
        ]
        shorts.append(feedback.PostedShort("lonely", 900, moment_type="funny_fail"))
        analysis = feedback.analyse(shorts)
        assert analysis["count"] == 9
        assert analysis["moment_types"]["freakout"]["vs_channel"] > 1
        bias = feedback.bias_from(analysis, "https://youtube.com/@me")
        assert "funny_fail" not in bias.moment_types  # one Short can't steer selection
        assert bias.moment_types["freakout"] > 1 > bias.moment_types["hot_take"]
        assert "freakout" in bias.summary and bias.streamers["Kai"] > 1

    def test_match_to_clips(self) -> None:
        from autoclip.gta import feedback

        stream = store.create_stream("https://youtu.be/q")
        clip = store.GtaClip(
            id="m1",
            stream_id=stream.id,
            start_s=0,
            end_s=1,
            hook_title="He found a SHARK",
            moment_type="crazy_discovery",
        )
        shorts = [
            feedback.PostedShort("He found a shark!! #shorts", 10),
            feedback.PostedShort("unrelated", 5),
        ]
        feedback.match_to_clips(shorts, [clip], {stream.id: "Kai"})
        assert shorts[0].moment_type == "crazy_discovery" and shorts[0].streamer == "Kai"
        assert shorts[1].moment_type == ""

    def test_look_at_whats_working(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from autoclip.gta import feedback

        monkeypatch.setattr(
            feedback,
            "list_shorts",
            lambda url: [feedback.PostedShort(f"t{i}", 100 * (i + 1)) for i in range(6)],
        )
        labels = [
            {"index": i, "moment_type": "freakout" if i >= 3 else "hot_take", "streamer": "Kai"}
            for i in range(6)
        ]
        model = FakeModel([json.dumps({"labels": labels})])
        report = feedback.look_at_whats_working("https://youtube.com/@me", complete=model)
        assert report["moment_types"]["freakout"]["vs_channel"] > 1
        assert gta_settings.load().bias.moment_types["freakout"] > 1
        assert store.latest_report()["report"]["count"] == 6

    def test_shorts_url(self) -> None:
        from autoclip.gta import feedback

        base = "https://www.youtube.com/@me"
        assert feedback.shorts_url(base + "/") == base + "/shorts"
        assert feedback.shorts_url(base + "/shorts") == base + "/shorts"


# --------------------------------------------------------------------------
# GTA Live: listen trigger + evaluate -> judge -> render (synthetic buffer)
# --------------------------------------------------------------------------


class _FakeBuffer:
    """A ring buffer whose 'stream' is one local file starting at stream time 0."""

    def __init__(self, path: Path, length_s: float) -> None:
        self.path = path
        self.length_s = length_s

    def covered_until_s(self) -> float:
        return self.length_s

    def extract_window(self, start_s, end_s, target, *, anchor_s=None):
        from autoclip.live.buffer import ExtractedWindow

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.path.read_bytes())
        return ExtractedWindow(
            path=target, start_s=0.0, end_s=self.length_s, start_index=0, end_index=0
        )


class TestLive:
    def test_listen_fires_on_a_loud_moment(self, tmp_path: Path) -> None:
        from autoclip.gta.live import GtaLiveWatch

        section = tmp_path / "s.mp4"
        _make_section(section, duration=24.0)
        stream = store.create_stream("https://www.youtube.com/watch?v=live")
        watch = GtaLiveWatch(stream.id, stream.url)
        watch._buffer = _FakeBuffer(section, 24.0)
        # A quiet history, then the last interval (the tone) is far louder.
        for t in range(-200, 0):
            watch._levels.append((float(t), -45.0))
        fired, why = watch._listen(0.0, 6.0)
        assert fired and "lost it" in why

    def test_evaluate_judges_and_renders(self, tmp_path: Path, monkeypatch) -> None:
        from autoclip.gta import live
        from autoclip.pipeline import transcribe as transcribe_mod

        section = tmp_path / "s.mp4"
        _make_section(section, duration=24.0)
        stream = store.create_stream("https://www.twitch.tv/streamer")
        store.update_stream(
            stream.id, status="live_watching", channel="Streamer", signals={"mode": "live"}
        )

        words = _words("WAIT what is that thing right there no way it is real", 1.0, 0.6)
        words += _words("I can not believe it chat that was insane.", 12.5, 0.6)
        monkeypatch.setattr(transcribe_mod, "transcribe", lambda *a, **k: Transcript(words=words))

        class Model(FakeModel):
            async def __call__(self, system, user, config):
                self.prompts.append((system, user))
                if "verdicts" in system:
                    cid = user.split("### id: ")[1].split("\n")[0]
                    return _final_response([_yes(cid, title="He saw WHAT")])
                return _chunk_response(
                    [
                        {
                            "start_word": 0,
                            "end_word": len(words) - 1,
                            "hook_quote": "WAIT what is that",
                        }
                    ]
                )

        model = Model([])
        monkeypatch.setattr(live, "_provider_complete", lambda: model)
        monkeypatch.setattr(qa, "face_ratio", lambda *a, **k: 0.9)

        watch = live.GtaLiveWatch(stream.id, stream.url)
        watch._buffer = _FakeBuffer(section, 24.0)
        settings = GtaSettings(min_clip_s=5)
        watch._evaluate(18.0, "test", settings, triggered_at=time.time())

        clips = store.list_clips(stream.id)
        assert len(clips) == 1, [c.reject_reason for c in clips]
        clip = clips[0]
        assert clip.status == "ready", (clip.error, clip.qa)
        assert clip.hook_title == "He saw WHAT"
        assert Path(clip.path).exists() and Path(clip.path).with_suffix(".txt").exists()
        assert clip.qa["turnaround_ok"] is True and clip.qa["passed"] is True
        info = ffmpeg.probe(Path(clip.path))
        assert (info.width, info.height) == (1080, 1920)


# --------------------------------------------------------------------------
# Delivery spec: own-value edits, watermark, safe zones, format QA
# --------------------------------------------------------------------------


def _make_logo(path: Path) -> None:
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=0xFF00FF@1.0:s=200x200:d=1,format=rgba",
        "-frames:v", "1", str(path),
    )  # fmt: skip


class TestDeliverySpec:
    def test_settings_lengths_and_watermark_validation(self) -> None:
        s = GtaSettings()
        assert (s.min_clip_s, s.target_clip_s) == (15.0, 30.0) and s.max_clip_s < 60
        with pytest.raises(ValueError):
            GtaSettings(target_clip_s=59)
        with pytest.raises(ValueError):
            GtaSettings(watermark_corner="bottom_right")
        with pytest.raises(ValueError):
            GtaSettings(punch_in_zoom=3)

    def test_key_moment_maps_through_cuts(self) -> None:
        hop = 0.05
        levels = np.full(int(10 / hop), -40.0, dtype=np.float32)
        levels[int(7.0 / hop) : int(8.0 / hop)] = -5.0  # loud beat at 7-8 s (local)
        keeps = [render.Keep(0.0, 3.0), render.Keep(5.0, 10.0)]  # 3-5 s cut out
        moment = render.key_moment(levels, keeps, 0.0, hop_s=hop)
        assert moment == pytest.approx(3.0 + 2.0, abs=0.06)  # 7 s local -> 5 s output
        a, b = render.punch_in_window(moment, 8.0, 1.5)
        assert a == pytest.approx(4.8, abs=0.07) and b == pytest.approx(6.3, abs=0.07)
        assert render.punch_in_window(7.9, 8.0, 1.5)[1] <= 7.9

    def test_text_respects_safe_zones(self, tmp_path: Path) -> None:
        import pysubs2

        path = render.write_subtitles(
            tmp_path / "c.ass",
            _words("one two three four five six", 0.0),
            style_key="bold_pop",
            hook_title="Hook",
            title_s=3.0,
            duration_s=10.0,
            reaction_text="NO WAY",
            reaction_window=(1.0, 2.0),
        )
        subs = pysubs2.load(str(path))
        from autoclip.pipeline import captions

        caption_style = subs.styles[captions.STYLE_NAME]
        assert caption_style.marginv >= render.SAFE_BOTTOM_PX
        assert render.TEXT_CENTRE_Y - 150 > render.SAFE_TOP_PX
        reaction = next(e for e in subs.events if e.style == "GtaReaction")
        assert reaction.start >= 3000  # never on top of the hook title

    def test_render_with_punch_in_reaction_and_logo(self, tmp_path: Path) -> None:
        section = tmp_path / "section.mp4"
        _make_section(section)
        logo = tmp_path / "logo.png"
        _make_logo(logo)
        destination = tmp_path / "clip.mp4"
        extras = render.Extras(
            reaction_text="HE DID NOT SEE THAT",
            watermark=render.Watermark(path=logo, corner="top_right", width_px=120),
        )
        result = render.render(
            render.RenderJob(
                section=section,
                section_offset_s=0.0,
                start_s=0.5,
                end_s=20.0,
                words=_speech_words(),
                hook_title="Title",
                destination=destination,
                work_dir=tmp_path / "work",
                extras=extras,
            ),
            style_key="bold_pop",
            title_s=3.0,
            dead_air_min_s=0.6,
            dead_air_keep_s=0.12,
            max_duration_s=58.0,
            loudness_lufs=-14.0,
            true_peak_db=-1.0,
            export_settings=ExportSettings(prefer_hardware_encoder=False),
        )
        assert result.edits == ["punch-in", "reaction text"] and result.watermarked
        # The logo's magenta sits top-right, just under the 150 px safe zone.
        frame = tmp_path / "f.png"
        _ffmpeg("-ss", "1", "-i", str(destination), "-frames:v", "1", str(frame))
        import cv2

        img = cv2.imread(str(frame))
        y = render.LOGO_TOP_PX + 30
        b, g, r = (int(v) for v in img[y, 1080 - render.LOGO_SIDE_PX - 60])
        assert r > 180 and b > 180 and g < 120, (b, g, r)
        assert not np.any(
            (img[:150, :, 2] > 200) & (img[:150, :, 0] > 200) & (img[:150, :, 1] < 80)
        )

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
            watermarked=result.watermarked,
            target_s=30.0,
        )
        assert report.format.ok, report.format.detail
        assert report.own_value.ok and report.watermark.ok and report.passed

    def test_qa_flags_wrong_format_and_missing_edits(self, tmp_path: Path) -> None:
        clip = tmp_path / "small.mp4"
        _ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(clip),
        )  # fmt: skip
        report = qa.run_qa(
            clip,
            section=clip,
            section_start_s=0,
            section_end_s=6,
            target_lufs=-14.0,
            tolerance_lu=50.0,
            true_peak_db=10.0,
            face_min_ratio=0.0,
            face_ratio_fn=lambda *_: 1.0,
            edits=[],
            watermarked=False,
            target_s=4.0,
        )
        assert not report.format.ok and "640x360" in report.format.detail
        assert not report.own_value.ok and not report.watermark.ok
        assert report.notes and "target" in report.notes[0]
        assert not report.passed

    def test_reaction_text_reaches_the_verdict(self) -> None:
        c = judging.Candidate("x", 0, 0, 1, 0.0, 20.0, "", "other", "", 50)
        verdict = judging._Verdict.model_validate(
            {**_yes("x"), "reaction_text": "#wow CHAT WAS RIGHT"}
        )
        judging._apply_verdict(c, verdict, GtaSettings(), "Streamer")
        assert c.status == "approved" and c.verdict["reaction_text"] == "CHAT WAS RIGHT"
