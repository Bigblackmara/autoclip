"""YouTube's own captions as the transcript, and the ingest sidecars.

json3 documents and info dicts are small hand-made fixtures; the yt-dlp
object at ingest is a fake, so nothing touches the network.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from autoclip.config import Settings
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline import ingest, signals, transcribe
from autoclip.pipeline import youtube_captions as yc
from autoclip.pipeline.runner import PipelineRunner

WORDS = [
    "nobody",
    "tells",
    "you",
    "this",
    "about",
    "scaling",
    "a",
    "company",
    "until",
    "it",
    "is",
    "far",
    "too",
    "late",
]


def auto_json3(count: int = 60, *, start_ms: int = 1000, step_ms: int = 400) -> dict:
    """Automatic captions: one seg per word, events of five words, word offsets."""
    events = []
    t = start_ms
    for first in range(0, count, 5):
        segs = []
        for i in range(first, min(count, first + 5)):
            word = WORDS[i % len(WORDS)]
            segs.append(
                {"utf8": (" " if i > first else "") + word, "tOffsetMs": (i - first) * step_ms}
            )
        events.append({"tStartMs": t, "dDurationMs": 5 * step_ms, "wWinId": 1, "segs": segs})
        events.append({"tStartMs": t + 5 * step_ms, "aAppend": 1, "segs": [{"utf8": "\n"}]})
        t += 5 * step_ms
    return {"wireMagic": "pb3", "events": events}


def manual_json3(lines: int = 10) -> dict:
    """Manual captions: one seg per line, no word offsets, punctuation."""
    events = [
        {
            "tStartMs": 2000 + i * 3000,
            "dDurationMs": 2800,
            "segs": [{"utf8": f"This is line number {i}, and it ends here."}],
        }
        for i in range(lines)
    ]
    events.insert(0, {"tStartMs": 0, "dDurationMs": 1500, "segs": [{"utf8": "[Music]"}]})
    return {"events": events}


def _formats(lang: str) -> list[dict]:
    return [
        {"ext": "vtt", "url": f"https://example.invalid/{lang}.vtt"},
        {"ext": "json3", "url": f"https://example.invalid/{lang}.json3"},
    ]


# --------------------------------------------------------------------------
# Track choice
# --------------------------------------------------------------------------


class TestPickTrack:
    def test_manual_track_in_the_spoken_language_wins(self) -> None:
        info = {
            "language": "en",
            "subtitles": {"en-US": _formats("en-US"), "de": _formats("de")},
            "automatic_captions": {"en-orig": _formats("en-orig")},
        }
        track = yc.pick_track(info)

        assert (track.kind, track.lang) == ("manual", "en-US")
        assert track.url.endswith("en-US.json3")

    def test_automatic_original_track_when_no_manual(self) -> None:
        info = {
            "language": "hi",
            "subtitles": {"live_chat": _formats("live_chat")},
            "automatic_captions": {"en": _formats("en"), "hi-orig": _formats("hi-orig")},
        }
        track = yc.pick_track(info)

        assert (track.kind, track.lang) == ("auto", "hi")

    def test_never_a_machine_translation(self) -> None:
        """Without an -orig track only the video's own language is accepted."""
        info = {"language": "es", "automatic_captions": {"en": _formats("en")}}

        assert yc.pick_track(info) is None

    def test_plain_auto_track_of_the_stated_language(self) -> None:
        info = {"language": "es", "automatic_captions": {"es": _formats("es")}}

        assert yc.pick_track(info).lang == "es"

    def test_orig_track_disagreeing_with_the_language_is_doubt(self) -> None:
        info = {"language": "fr", "automatic_captions": {"en-orig": _formats("en-orig")}}

        assert yc.pick_track(info) is None

    def test_unknown_language_never_uses_manual_tracks(self) -> None:
        info = {"subtitles": {"en": _formats("en")}}

        assert yc.pick_track(info) is None

    def test_forced_whisper_language_must_match(self) -> None:
        info = {"language": "en", "automatic_captions": {"en-orig": _formats("en-orig")}}

        assert yc.pick_track(info, language="de") is None
        assert yc.pick_track(info, language="en").lang == "en"

    def test_track_without_json3_is_skipped(self) -> None:
        info = {"language": "en", "subtitles": {"en": [{"ext": "vtt", "url": "x"}]}}

        assert yc.pick_track(info) is None
        assert yc.pick_track(None) is None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


class TestParseJson3:
    def test_automatic_words_get_their_own_start_times(self) -> None:
        words = yc.parse_json3(auto_json3(10))

        assert [w.text for w in words] == WORDS[:10]
        assert words[0].start == pytest.approx(1.0)
        assert words[1].start == pytest.approx(1.4)
        assert words[5].start == pytest.approx(3.0)
        for a, b in zip(words, words[1:], strict=False):
            assert a.start < b.start
            assert a.end <= b.start + 1e-9
            assert a.end - a.start <= yc.MAX_WORD_S + 1e-9

    def test_manual_lines_are_spread_over_the_line_and_tags_dropped(self) -> None:
        words = yc.parse_json3(manual_json3(2))

        assert "[Music]" not in [w.text for w in words]
        assert words[0].text == "This"
        assert words[0].start == pytest.approx(2.0)
        first_line = [w for w in words if w.start < 5.0]
        assert first_line[-1].text == "here."
        assert first_line[-1].end <= 4.8 + 1e-6

    def test_non_json3_raises(self) -> None:
        with pytest.raises(ValueError):
            yc.parse_json3({"nope": []})


class TestToTranscript:
    def test_good_automatic_captions_become_a_transcript(self) -> None:
        transcript = yc.to_transcript(auto_json3(60), lang="en", kind="auto", duration_s=30.0)

        assert transcript is not None
        assert transcript.source == "youtube_captions"
        assert transcript.model == "youtube-auto:en"
        assert transcript.language == "en"
        assert len(transcript.words) == 60
        assert transcript.segments
        assert transcript.segments[0].first_word == 0
        assert transcript.segments[-1].last_word == 59

    def test_too_few_words_fall_back(self) -> None:
        assert yc.to_transcript(auto_json3(5), lang="en", kind="auto") is None

    def test_sparse_captions_fall_back(self) -> None:
        """60 words over an hour is a lyrics stub, not the video's speech."""
        assert yc.to_transcript(auto_json3(60), lang="en", kind="auto", duration_s=3600) is None

    def test_captions_running_past_the_end_fall_back(self) -> None:
        assert yc.to_transcript(auto_json3(60), lang="en", kind="auto", duration_s=10) is None

    def test_manual_captions_keep_punctuation_for_sentence_ends(self) -> None:
        transcript = yc.to_transcript(manual_json3(10), lang="en-US", kind="manual")

        assert transcript.language == "en"
        assert transcript.sentence_end_indices()


class TestLoadForSource:
    def test_reads_the_saved_track(self, tmp_path: Path) -> None:
        (tmp_path / "captions.auto.en.json3").write_text(json.dumps(auto_json3(60)), "utf-8")

        transcript = yc.load_for_source(tmp_path, duration_s=30)

        assert transcript is not None and len(transcript.words) == 60

    def test_unreadable_file_and_no_file_give_none(self, tmp_path: Path) -> None:
        assert yc.load_for_source(tmp_path) is None
        (tmp_path / "captions.auto.en.json3").write_text("{broken", "utf-8")
        assert yc.load_for_source(tmp_path) is None

    def test_language_forced_elsewhere_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "captions.auto.en.json3").write_text(json.dumps(auto_json3(60)), "utf-8")

        assert yc.load_for_source(tmp_path, language="de") is None


# --------------------------------------------------------------------------
# Ingest sidecars
# --------------------------------------------------------------------------


class _FakeYdl:
    def __init__(self, payloads: dict[str, bytes] | None = None, fail: bool = False) -> None:
        self.payloads = payloads or {}
        self.fail = fail
        self.opened: list[str] = []

    def sanitize_info(self, info: dict) -> dict:
        return dict(info)

    def urlopen(self, url: str):
        self.opened.append(url)
        if self.fail:
            raise OSError("HTTP Error 429")
        return io.BytesIO(self.payloads[url])


def _metadata() -> dict:
    return {
        "id": "abc",
        "title": "t",
        "language": "en",
        "duration": 30,
        "formats": [{"url": "huge"}],
        "heatmap": [{"start_time": 0, "end_time": 3, "value": 1.0}],
        "automatic_captions": {"en-orig": _formats("en-orig"), "de": _formats("de")},
    }


class TestSaveSidecars:
    def test_writes_info_without_formats(self, tmp_path: Path) -> None:
        ingest.save_sidecars(_FakeYdl(), _metadata(), tmp_path, captions=False)

        info = json.loads((tmp_path / signals.INFO_FILE).read_text("utf-8"))
        assert info["heatmap"]
        assert "formats" not in info
        assert not list(tmp_path.glob("captions.*"))

    def test_fetches_the_spoken_language_track(self, tmp_path: Path) -> None:
        url = "https://example.invalid/en-orig.json3"
        ydl = _FakeYdl({url: json.dumps(auto_json3(30)).encode()})

        ingest.save_sidecars(ydl, _metadata(), tmp_path, captions=True)

        assert ydl.opened == [url]
        assert (tmp_path / "captions.auto.en.json3").exists()

    def test_caption_failures_never_fail_the_ingest(self, tmp_path: Path) -> None:
        ingest.save_sidecars(_FakeYdl(fail=True), _metadata(), tmp_path, captions=True)

        assert (tmp_path / signals.INFO_FILE).exists()
        assert not list(tmp_path.glob("captions.*"))

    def test_garbage_caption_payload_is_not_saved(self, tmp_path: Path) -> None:
        url = "https://example.invalid/en-orig.json3"
        ingest.save_sidecars(_FakeYdl({url: b"<html>"}), _metadata(), tmp_path, captions=True)

        assert not list(tmp_path.glob("captions.*"))

    def test_odd_metadata_is_ignored(self, tmp_path: Path) -> None:
        ingest.save_sidecars(_FakeYdl(), None, tmp_path, captions=True)

        assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------
# Runner: captions first, Whisper on doubt
# --------------------------------------------------------------------------


class TestTranscribeStage:
    def _runner(self, tmp_path: Path, *, prefer: bool, kind: str = "youtube"):
        media = tmp_path / "media"
        media.mkdir()
        (media / "captions.auto.en.json3").write_text(json.dumps(auto_json3(60)), "utf-8")
        source = store.create_source(
            Source(
                id=new_id(),
                type=kind,
                path=str(media / "source.mp4"),
                url="https://youtu.be/abc" if kind == "youtube" else None,
                duration_s=30.0,
            )
        )
        job = store.create_job(Job(id=new_id(), source_id=source.id))
        settings = Settings()
        settings.ingest.prefer_youtube_captions = prefer
        return PipelineRunner(job, source, settings=settings), job

    def test_good_captions_skip_whisper(
        self, initialised_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def whisper(*args, **kwargs):
            raise AssertionError("Whisper must not run when captions are good")

        monkeypatch.setattr(transcribe, "transcribe", whisper)
        runner, job = self._runner(tmp_path, prefer=True)

        transcript = runner._stage_transcribe(Path("audio.wav"))

        assert transcript.source == "youtube_captions"
        assert len(transcript.words) == 60
        assert store.get_transcript(job.id).source == "youtube_captions"
        assert runner.workspace.transcript.exists()

    @pytest.mark.parametrize(("prefer", "kind"), [(False, "youtube"), (True, "upload")])
    def test_whisper_runs_when_captions_are_not_wanted(
        self, initialised_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefer, kind
    ) -> None:
        from autoclip.pipeline.transcript import Transcript, Word

        calls: list[Path] = []

        def whisper(audio, *args, **kwargs):
            calls.append(audio)
            return Transcript(words=[Word("hi", 0.0, 0.5)], model="tiny")

        monkeypatch.setattr(transcribe, "transcribe", whisper)
        runner, _ = self._runner(tmp_path, prefer=prefer, kind=kind)

        transcript = runner._stage_transcribe(Path("audio.wav"))

        assert calls and transcript.source == "whisper"

    def test_bad_captions_fall_back_to_whisper(
        self, initialised_db, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autoclip.pipeline.transcript import Transcript, Word

        monkeypatch.setattr(
            transcribe, "transcribe", lambda *a, **k: Transcript(words=[Word("hi", 0.0, 0.5)])
        )
        runner, _ = self._runner(tmp_path, prefer=True)
        runner.source.duration_s = 3600.0  # 60 words in an hour: not the speech

        assert runner._stage_transcribe(Path("audio.wav")).source == "whisper"
