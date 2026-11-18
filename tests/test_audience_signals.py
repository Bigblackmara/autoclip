"""Audience signals: heatmap, comment timestamps, chapters, audio energy.

All inputs are small fakes (an info json, a comment list, a loudness
profile); the comment fetcher is injected, so nothing touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from autoclip.pipeline import signals
from autoclip.pipeline.signals import AudienceCurve, Comment, build_curve

DURATION = 600.0


def _heatmap(hot_start: float, hot_end: float, *, bins: int = 100) -> list[tuple]:
    """A flat 0.2 heatmap with one 1.0 peak over ``[hot_start, hot_end)``."""
    step = DURATION / bins
    out = []
    for i in range(bins):
        start, end = i * step, (i + 1) * step
        value = 1.0 if start < hot_end and end > hot_start else 0.2
        out.append((start, end, value))
    return out


def _info(**extra) -> dict:
    step = DURATION / 100
    heatmap = [
        {"start_time": i * step, "end_time": (i + 1) * step, "value": 1.0 if 40 <= i < 43 else 0.2}
        for i in range(100)
    ]
    return {
        "id": "abc",
        "duration": DURATION,
        "heatmap": heatmap,
        "chapters": [
            {"start_time": 0, "end_time": 200, "title": "Intro"},
            {"start_time": 200, "end_time": 600, "title": "The big reveal"},
        ],
        **extra,
    }


# --------------------------------------------------------------------------
# Comments
# --------------------------------------------------------------------------


class TestCommentTimestamps:
    def test_reads_minutes_and_hours(self) -> None:
        assert signals.comment_timestamps("4:20 was insane", 7200) == [260]
        assert signals.comment_timestamps("at 1:02:03 lol", 7200) == [3723]

    def test_ignores_impossible_and_out_of_range_times(self) -> None:
        assert signals.comment_timestamps("4:75 nope", 600) == []
        assert signals.comment_timestamps("15:00 is past the end", 600) == []
        assert signals.comment_timestamps("0:00 start", 600) == []

    def test_ignores_ratios_and_long_numbers(self) -> None:
        assert signals.comment_timestamps("ratio 16:9:1 and 123:45", 9000) == []

    def test_nearby_mentions_in_one_comment_count_once(self) -> None:
        assert signals.comment_timestamps("4:10 ... 4:12 omg, then 8:00", 600) == [250, 480]

    def test_parse_comments_tolerates_junk(self) -> None:
        parsed = signals.parse_comments(
            [
                {"text": "a", "like_count": "7"},
                {"text": ""},
                "junk",
                {"text": "b", "like_count": None},
            ]
        )
        assert [(c.text, c.likes) for c in parsed] == [("a", 7), ("b", 0)]


# --------------------------------------------------------------------------
# Curve
# --------------------------------------------------------------------------


class TestBuildCurve:
    def test_heatmap_makes_an_audience_curve_peaking_on_the_replayed_part(self) -> None:
        curve = build_curve(DURATION, heatmap=_heatmap(300, 330))

        assert curve.kind == "audience"
        assert curve.sources == ["heatmap"]
        assert curve.seconds == 600
        assert curve.hotspots, "the replayed stretch should be a hotspot"
        top = curve.hotspots[0]
        assert 290 <= top.peak_s <= 335
        assert any("most replayed" in r for r in top.reasons)

    def test_the_opening_of_the_heatmap_is_damped(self) -> None:
        heat = [(0.0, 6.0, 1.0)] + [(6.0 * i, 6.0 * (i + 1), 0.5) for i in range(1, 100)]
        curve = build_curve(DURATION, heatmap=heat)

        # Everyone starts at 0:00; that is not a replay.
        assert curve.curve[0] < curve.curve[100]

    def test_comment_timestamps_alone_make_an_audience_curve(self) -> None:
        comments = [
            Comment("5:00 bro 💀", likes=900),
            Comment("the part at 5:02 lmao", likes=40),
            Comment("5:01 cannot breathe", likes=3),
            Comment("first", likes=0),
        ]
        curve = build_curve(DURATION, comments=comments)

        assert curve.kind == "audience"
        assert curve.sources == ["comments"]
        peak = int(np.argmax(curve.curve))
        # Mentions are spread a few seconds past the timestamp.
        assert 300 <= peak <= 312
        assert "3 comment timestamps" in curve.hotspots[0].reasons[0]

    def test_too_few_comment_timestamps_are_ignored(self) -> None:
        curve = build_curve(DURATION, comments=[Comment("5:00"), Comment("6:00")])

        assert curve.kind == "none"
        assert curve.range_score(0, 30) is None

    def test_likes_weigh_a_mention(self) -> None:
        comments = [Comment("2:00", likes=50_000)] + [Comment("7:00", likes=0)] * 2
        curve = build_curve(DURATION, comments=comments)

        assert curve.curve[125] > curve.curve[425]

    def test_loudness_alone_makes_an_energy_curve(self) -> None:
        loud = np.full(600, -30.0, dtype=np.float32)
        loud[400:406] = -8.0  # a shout well above the baseline
        curve = build_curve(DURATION, loudness_db=loud)

        assert curve.kind == "energy"
        assert curve.sources == ["energy"]
        assert 398 <= int(np.argmax(curve.curve)) <= 407

    def test_silence_is_not_energy(self) -> None:
        curve = build_curve(DURATION, loudness_db=np.full(600, signals.SILENCE_DB))

        assert curve.kind == "none"

    def test_viewer_data_outweighs_loudness(self) -> None:
        loud = np.full(600, -30.0, dtype=np.float32)
        loud[100:110] = -5.0
        curve = build_curve(DURATION, heatmap=_heatmap(300, 330), loudness_db=loud)

        assert curve.kind == "audience"
        assert set(curve.sources) == {"heatmap", "energy"}
        assert curve.range_score(300, 330) > curve.range_score(95, 125)

    def test_nothing_gives_an_empty_curve(self) -> None:
        curve = build_curve(DURATION)

        assert curve.kind == "none"
        assert not curve.has_data
        assert curve.hotspots == []


class TestRangeScore:
    def test_hot_span_ranks_near_the_top_and_cold_span_low(self) -> None:
        curve = build_curve(DURATION, heatmap=_heatmap(300, 330))

        assert curve.range_score(295, 335) >= 90
        assert curve.range_score(100, 140) <= 60

    def test_a_flat_curve_ranks_everything_in_the_middle(self) -> None:
        curve = AudienceCurve(curve=np.full(100, 0.5, dtype=np.float32), kind="energy")

        assert curve.range_score(10, 20) == 50

    def test_span_past_the_end_is_none(self) -> None:
        curve = build_curve(DURATION, heatmap=_heatmap(300, 330))

        assert curve.range_score(700, 730) is None


class TestPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        curve = build_curve(
            DURATION,
            heatmap=_heatmap(300, 330),
            chapters=[signals.Chapter(0.0, 200.0, "Intro")],
        )
        path = curve.save(tmp_path / "audience.json")
        loaded = AudienceCurve.load(path)

        assert loaded.kind == curve.kind
        assert loaded.sources == curve.sources
        assert loaded.chapters == curve.chapters
        assert [h.peak_s for h in loaded.hotspots] == [h.peak_s for h in curve.hotspots]
        assert loaded.range_score(300, 330) == curve.range_score(300, 330)

    def test_unknown_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "audience.json"
        path.write_text(json.dumps({"version": 99, "curve": []}), encoding="utf-8")

        with pytest.raises(ValueError):
            AudienceCurve.load(path)


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


class TestReaders:
    def test_chapters_from_info(self) -> None:
        chapters = signals.parse_chapters(_info())

        assert [(c.start_s, c.title) for c in chapters] == [
            (0.0, "Intro"),
            (200.0, "The big reveal"),
        ]

    def test_bad_chapters_are_skipped(self) -> None:
        info = {
            "chapters": [{"start_time": "x"}, {"start_time": 5, "end_time": 2, "title": "t"}, 3]
        }

        assert signals.parse_chapters(info) == []
        assert signals.parse_chapters(None) == []

    def test_missing_or_broken_info_json_is_none(self, tmp_path: Path) -> None:
        assert signals.load_info(tmp_path / "nope.json") is None
        broken = tmp_path / "info.json"
        broken.write_text("{not json", encoding="utf-8")
        assert signals.load_info(broken) is None

    def test_loudness_per_second_averages_energy(self) -> None:
        values = [-20.0] * 10 + [-120.0] * 10 + [-10.0] * 5
        per_second = signals.loudness_per_second(values, 0.1)

        assert len(per_second) == 3
        assert per_second[0] == pytest.approx(-20.0, abs=0.01)
        assert per_second[1] == signals.SILENCE_DB
        assert per_second[2] == pytest.approx(-10.0, abs=0.01)


# --------------------------------------------------------------------------
# build_for_source
# --------------------------------------------------------------------------


class TestBuildForSource:
    def _source_dir(self, tmp_path: Path, info: dict | None = None) -> Path:
        directory = tmp_path / "media"
        directory.mkdir()
        if info is not None:
            (directory / signals.INFO_FILE).write_text(json.dumps(info), encoding="utf-8")
        return directory

    def test_uses_heatmap_chapters_and_comments(self, tmp_path: Path) -> None:
        directory = self._source_dir(tmp_path, _info())
        calls: list[str] = []

        def fetch(url: str) -> list[Comment]:
            calls.append(url)
            return [Comment("4:05 omg", 10), Comment("4:06!!", 2), Comment("4:04 lol", 1)]

        curve = signals.build_for_source(
            source_dir=directory,
            url="https://youtu.be/abc",
            is_youtube=True,
            duration_s=DURATION,
            fetch=fetch,
        )

        assert calls == ["https://youtu.be/abc"]
        assert curve.kind == "audience"
        assert curve.sources == ["heatmap", "comments"]
        assert [c.title for c in curve.chapters] == ["Intro", "The big reveal"]
        assert (directory / signals.COMMENTS_FILE).exists()

    def test_cached_comments_are_not_fetched_again(self, tmp_path: Path) -> None:
        directory = self._source_dir(tmp_path, _info())
        (directory / signals.COMMENTS_FILE).write_text(
            json.dumps([{"text": "1:00", "like_count": 1}] * 3), encoding="utf-8"
        )

        def fetch(url: str) -> list[Comment]:
            raise AssertionError("must use the cache")

        curve = signals.build_for_source(
            source_dir=directory, url="u", is_youtube=True, duration_s=DURATION, fetch=fetch
        )

        assert "comments" in curve.sources

    def test_a_failing_comment_fetch_is_not_fatal_or_cached(self, tmp_path: Path) -> None:
        directory = self._source_dir(tmp_path, _info())

        def fetch(url: str) -> list[Comment]:
            raise RuntimeError("HTTP Error 429: Too Many Requests")

        curve = signals.build_for_source(
            source_dir=directory, url="u", is_youtube=True, duration_s=DURATION, fetch=fetch
        )

        assert curve.sources == ["heatmap"]
        assert not (directory / signals.COMMENTS_FILE).exists()

    def test_comments_can_be_switched_off(self, tmp_path: Path) -> None:
        directory = self._source_dir(tmp_path, _info())

        def fetch(url: str) -> list[Comment]:
            raise AssertionError("comments are off")

        signals.build_for_source(
            source_dir=directory,
            url="u",
            is_youtube=True,
            duration_s=DURATION,
            use_comments=False,
            fetch=fetch,
        )

    def test_an_upload_falls_back_to_audio_energy(self, tmp_path: Path) -> None:
        directory = self._source_dir(tmp_path, _info())  # ignored: not YouTube
        values = [-30.0] * 6000
        values[3000:3050] = [-6.0] * 50

        curve = signals.build_for_source(
            source_dir=directory,
            url=None,
            is_youtube=False,
            duration_s=DURATION,
            loudness_values=values,
            loudness_step_s=0.1,
        )

        assert curve.kind == "energy"
        assert curve.chapters == []

    def test_energy_can_be_switched_off(self, tmp_path: Path) -> None:
        curve = signals.build_for_source(
            source_dir=self._source_dir(tmp_path),
            url=None,
            is_youtube=False,
            duration_s=DURATION,
            loudness_values=[-30.0] * 6000,
            use_energy=False,
        )

        assert curve.kind == "none"

    def test_duration_falls_back_to_the_info_json(self, tmp_path: Path) -> None:
        curve = signals.build_for_source(
            source_dir=self._source_dir(tmp_path, _info()), url=None, is_youtube=True, duration_s=0
        )

        assert curve.seconds == 600
