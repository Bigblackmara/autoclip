"""Picking parts on the rewatch graph: validation, snapping, joined timelines."""

from __future__ import annotations

import pytest
from autoclip.pipeline import captions, edit, export, picks, segments
from autoclip.pipeline.prepare import Silence
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropRect,
    CropSegment,
    Strategy,
    centre_crop,
    fit_segment,
)
from autoclip.pipeline.segments import Span
from autoclip.pipeline.transcript import Transcript, Word


def sentence_transcript() -> Transcript:
    """Two-second sentences of four words, from 0 s to 120 s.

    Word ``i`` runs ``[i*0.5, i*0.5 + 0.4]``; every fourth word ends a sentence.
    """
    words = []
    for i in range(240):
        text = f"w{i}" + ("." if i % 4 == 3 else "")
        words.append(Word(text=text, start=round(i * 0.5, 3), end=round(i * 0.5 + 0.4, 3)))
    return Transcript(words=words)


class TestCleanRanges:
    def test_sorts_and_merges_overlaps(self) -> None:
        spans = picks.clean_ranges([(50, 60), (10, 20), (15, 25)], duration_s=100)

        assert spans == [Span(10, 25), Span(50, 60)]

    def test_swapped_edges_are_accepted(self) -> None:
        assert picks.clean_ranges([(20, 10)], duration_s=100) == [Span(10, 20)]

    def test_clamps_to_the_video(self) -> None:
        assert picks.clean_ranges([(-5, 3), (95, 130)], duration_s=100) == [
            Span(0, 3),
            Span(95, 100),
        ]

    @pytest.mark.parametrize(
        "ranges",
        [
            [],
            [(10, 10.5)],
            [(150, 160)],
            [(-10, -2)],
            [(float("nan"), 10)],
            [(0, float("inf"))],
            [(i * 2, i * 2 + 1.5) for i in range(picks.MAX_PICKS + 1)],
        ],
        ids=["empty", "too-short", "past-end", "before-start", "nan", "inf", "too-many"],
    )
    def test_rejects_invalid_ranges(self, ranges) -> None:
        with pytest.raises(picks.PickError):
            picks.clean_ranges(ranges, duration_s=100)


class TestGroup:
    def test_one_clip_per_range(self) -> None:
        groups = picks.group([Span(0, 10), Span(20, 30)], join=False)

        assert groups == [[Span(0, 10)], [Span(20, 30)]]

    def test_join_makes_one_clip(self) -> None:
        groups = picks.group([Span(0, 10), Span(20, 30)], join=True)

        assert groups == [[Span(0, 10), Span(20, 30)]]

    def test_each_clip_is_capped(self) -> None:
        groups = picks.group([Span(0, 400)], join=False)

        assert groups == [[Span(0, picks.MAX_CLIP_S)]]

    def test_a_joined_clip_is_capped_in_total(self) -> None:
        groups = picks.group([Span(0, 100), Span(200, 300), Span(400, 450)], join=True)

        assert segments.total(groups[0]) == pytest.approx(picks.MAX_CLIP_S)
        assert groups[0][-1] == Span(200, 280)


class TestSnapping:
    def test_edges_snap_to_sentences_nearby(self) -> None:
        transcript = sentence_transcript()
        # Picked from mid-sentence (word 5 at 2.5 s) to mid-sentence (word 18).
        snapped, start_word, end_word = picks.snap_span(transcript, Span(2.4, 9.3))

        assert start_word == 4  # sentence start (after "w3.")
        assert end_word == 19  # sentence end "w19."
        assert snapped.start <= transcript.words[4].start
        assert snapped.end >= transcript.words[19].end

    def test_never_starts_mid_word(self) -> None:
        transcript = sentence_transcript()
        # 10.1 s is inside word 20 (10.0-10.4); its midpoint is before the pick.
        snapped, start_word, _ = picks.snap_span(transcript, Span(10.25, 30.0))

        assert transcript.words[start_word].start >= 10.25 - picks.SNAP_WINDOW_S
        assert snapped.start <= transcript.words[start_word].start

    def test_far_sentence_edges_are_not_used(self) -> None:
        words = [Word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(200)]
        words[199] = Word("end.", 99.5, 99.9)
        transcript = Transcript(words=words)

        _, start_word, end_word = picks.snap_span(transcript, Span(20.0, 40.0))

        # No punctuation within the window: the user's edges stand.
        assert words[start_word].start == pytest.approx(20.0, abs=0.5)
        assert words[end_word].end == pytest.approx(40.0, abs=0.6)

    def test_edges_settle_into_silences(self) -> None:
        transcript = sentence_transcript()
        silences = [Silence(1.9, 2.0), Silence(9.9, 10.0)]

        snapped, _, _ = picks.snap_span(transcript, Span(2.0, 9.9), silences=silences)

        assert 1.9 <= snapped.start <= 2.0
        assert 9.9 <= snapped.end <= 10.0

    def test_a_range_without_words_is_left_alone(self) -> None:
        assert picks.snap_span(sentence_transcript(), Span(130.0, 140.0)) is None


class TestPlan:
    def test_makes_one_clip_per_range(self) -> None:
        clips = picks.plan(
            sentence_transcript(), [(4.0, 12.0), (40.0, 50.0)], duration_s=120, join=False
        )

        assert len(clips) == 2
        assert all(not c.is_joined for c in clips)
        assert clips[0].start_word == 8 and clips[0].end_word == 23

    def test_joins_ranges_in_source_order(self) -> None:
        clips = picks.plan(
            sentence_transcript(), [(40.0, 50.0), (4.0, 12.0)], duration_s=120, join=True
        )

        assert len(clips) == 1
        clip = clips[0]
        assert clip.is_joined
        assert [s.start < 20 for s in clip.spans] == [True, False]
        assert clip.duration_s < clip.end_s - clip.start_s

    def test_ranges_that_snap_together_merge(self) -> None:
        clips = picks.plan(
            sentence_transcript(), [(4.0, 9.8), (10.1, 16.0)], duration_s=120, join=True
        )

        assert len(clips) == 1 and len(clips[0].spans) == 1

    def test_works_without_a_transcript(self) -> None:
        clips = picks.plan(Transcript(), [(4.0, 12.0)], duration_s=120, join=False)

        assert clips[0].spans == [Span(4.0, 12.0)]

    def test_clip_rows_rank_after_existing_ones(self) -> None:
        planned = picks.plan(
            sentence_transcript(), [(4.0, 12.0), (40.0, 50.0)], duration_s=120, join=False
        )

        rows = picks.build_clips("job1", planned, first_rank=7)

        assert [r.rank for r in rows] == [7, 8]
        assert rows[0].title == f"Your pick #1 · {picks.clock(planned[0].start_s)}"
        assert rows[0].score == 50  # no audience curve
        assert rows[0].start_s == planned[0].start_s


class TestJoinedTimeline:
    spans = [Span(10.0, 14.0), Span(30.0, 33.0)]

    def test_offsets_and_total(self) -> None:
        assert segments.offsets(self.spans) == [0.0, 4.0]
        assert segments.total(self.spans) == 7.0

    def test_time_maps_both_ways(self) -> None:
        assert segments.to_joined(31.0, self.spans) == 5.0
        assert segments.to_joined(20.0, self.spans) is None
        assert segments.to_source(5.0, self.spans) == 31.0
        assert segments.to_source(1.0, self.spans) == 11.0

    def test_caption_words_follow_the_joins(self) -> None:
        words = [
            Word("a", 10.5, 11.0),
            Word("gap", 20.0, 20.5),  # between the parts: dropped
            Word("b", 13.6, 14.2),  # straddles the first part's end: clipped
            Word("cut", 13.9, 14.5),  # its middle is past the end: dropped
            Word("c", 30.2, 30.6),
        ]

        timeline = segments.words_on_timeline(words, self.spans)

        assert [(w.text, w.start, w.end) for w in timeline] == [
            ("a", 0.5, 1.0),
            ("b", 3.6, 4.0),
            ("c", 4.2, 4.6),
        ]

    def test_words_in_keeps_source_times(self) -> None:
        words = [Word("a", 10.5, 11.0), Word("gap", 20.0, 20.5), Word("c", 30.2, 30.6)]

        assert [w.text for w in segments.words_in(words, self.spans)] == ["a", "c"]

    def test_silences_are_cut_at_the_joins(self) -> None:
        silences = [Silence(13.5, 31.0), Silence(40.0, 41.0)]

        moved = segments.silences_on_timeline(silences, self.spans)

        assert moved == [Silence(3.5, 4.0), Silence(4.0, 5.0)]

    def test_pause_cuts_and_captions_stay_in_sync(self) -> None:
        """Edit keeps on the joined timeline re-time captions consistently."""
        words = [Word("a", 10.5, 11.0), Word("c", 31.0, 31.4)]
        timeline = segments.words_on_timeline(words, self.spans)
        keeps = [edit.Keep(0.0, 2.0), edit.Keep(4.5, 7.0)]  # cut 2.0-4.5

        out = edit.retime_words(timeline, keeps)

        # "c" at joined 5.0 lands 0.5 s into the second keep, after 2.0 s kept.
        assert [(w.text, w.start) for w in out] == [("a", 0.5), ("c", 2.5)]

    def test_clip_to_handles_a_trimmed_envelope(self) -> None:
        assert segments.clip_to(self.spans, 12.0, 31.0) == [Span(12.0, 14.0), Span(30.0, 31.0)]

    def test_sidecar_round_trips(self, tmp_path) -> None:
        path = segments.save(tmp_path / "seg" / "clip.json", self.spans)

        assert segments.load(path) == self.spans
        assert segments.load(tmp_path / "missing.json") is None

    def test_unreadable_sidecar_is_ignored(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{nope", encoding="utf-8")

        assert segments.load(path) is None


# --------------------------------------------------------------------------
# Filtergraph construction (no ffmpeg needed)
# --------------------------------------------------------------------------


def _request(**overrides) -> export.ExportRequest:
    base = {
        "source": __import__("pathlib").Path("source.mp4"),
        "destination": __import__("pathlib").Path("out.mp4"),
        "start_s": 0.0,
        "end_s": 7.0,
        "crop_path": centre_crop(1920, 1080, 7.0),
        "words": [],
        "style": captions.get_style("bold_pop"),
    }
    base.update(overrides)
    return export.ExportRequest(**base)


class TestJoinedFiltergraph:
    spans = [(10.0, 14.0), (30.0, 33.0)]

    def test_each_part_is_its_own_seeked_input(self) -> None:
        args = export.source_inputs(_request(spans=self.spans))

        assert args == [
            "-ss",
            "10.0000",
            "-t",
            "4.0000",
            "-i",
            "source.mp4",
            "-ss",
            "30.0000",
            "-t",
            "3.0000",
            "-i",
            "source.mp4",
        ]

    def test_crop_segments_are_trimmed_from_their_own_part(self) -> None:
        graph = export.build_video_filtergraph(_request(spans=self.spans), subtitle_name=None)

        assert "[0:v]trim=start=0.0000:end=4.0000" in graph
        assert "[1:v]trim=start=0.0000:end=3.0000" in graph
        assert "concat=n=2:v=1:a=0[vcat]" in graph

    def test_audio_joins_with_fades(self) -> None:
        parts, label = export.joined_audio(_request(spans=self.spans))

        assert label == "[ajoin]"
        assert parts[0].startswith("[0:a]atrim=duration=4.0000")
        assert "afade=t=out" in parts[1]
        assert parts[-1] == "[ja0][ja1]concat=n=2:v=0:a=1[ajoin]"

    def test_extra_inputs_come_after_the_parts(self) -> None:
        request = _request(spans=self.spans, has_video=False)

        assert export.input_layout(request).background == 2
        assert export.build_video_filtergraph(request, subtitle_name=None).startswith("[2:v]")

    def test_edit_audio_reads_the_joined_speech(self) -> None:
        plan = edit.EditPlan(
            keeps=[edit.Keep(0.0, 2.0), edit.Keep(4.5, 7.0)], two_pass_loudnorm=False
        )
        request = _request(spans=self.spans, edit=plan)

        graph = export.build_edit_audio_graph(request, tail="anull", music_index=None)

        assert "[ajoin]asplit=2" in graph
        assert "[0:a]asplit" not in graph

    def test_plain_clips_are_unchanged(self) -> None:
        request = _request()

        assert export.source_inputs(request)[:2] == ["-ss", "0.0000"]
        assert export.joined_audio(request) == ([], "[0:a]")
        assert "trim" not in export.build_video_filtergraph(request, subtitle_name=None)


class TestSpeakerSplitFiltergraph:
    def _split_path(self) -> CropPath:
        return CropPath(
            1920,
            1080,
            [
                CropSegment(
                    0.0,
                    7.0,
                    608,
                    1080,
                    keyframes=[CropKeyframe(0.0, 656.0, 0.0)],
                    strategy=Strategy.SPLIT,
                    panes=[CropRect(100, 50, 704, 626), CropRect(1100, 60, 704, 626)],
                )
            ],
        )

    def test_panes_are_cropped_and_stacked(self) -> None:
        graph = export.build_video_filtergraph(
            _request(crop_path=self._split_path()), subtitle_name=None
        )

        assert "crop=704:626:100:50,scale=1080:960" in graph
        assert "crop=704:626:1100:60,scale=1080:960" in graph
        assert "vstack=inputs=2" in graph

    def test_square_output_uses_the_nominal_crop(self) -> None:
        graph = export.build_video_filtergraph(
            _request(crop_path=self._split_path(), ratio="1:1"), subtitle_name=None
        )

        assert "vstack" not in graph
        assert "crop=w=608:h=1080" in graph

    def test_faceless_fit_is_a_blurred_fill(self) -> None:
        path = CropPath(1920, 1080, [fit_segment(0.0, 7.0, 1920, 1080)])

        graph = export.build_video_filtergraph(_request(crop_path=path), subtitle_name=None)

        assert "boxblur" in graph and "overlay" in graph

    def test_no_punch_in_zooms_on_a_split(self) -> None:
        plan = edit.EditPlan(zooms=[edit.Zoom(0.0, 1.1)], two_pass_loudnorm=False)
        request = _request(crop_path=self._split_path(), edit=plan)

        graph = export.build_video_filtergraph(request, subtitle_name=None)

        assert "zoompan" not in graph and "[ezoom]" not in graph

    def test_captions_sit_on_the_seam(self) -> None:
        style = captions.get_style("bold_pop")

        margin = export.speakers_caption_margin(style)

        assert 0.3 < margin < 0.5
