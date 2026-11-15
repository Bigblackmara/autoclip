"""Pause-aware sentence breaks for transcripts Whisper left unpunctuated.

With Whisper "small" on a 2h44m IShowSpeed stream (job 52d42055d02d4a07),
52 of 599 40-word windows had no sentence-ending punctuation, so snapping had
nothing to land on. The word lists below mirror stretches of that transcript
(timestamps in each docstring); gaps are 0 s between words, as 72% of that
stream's are, with the measured pauses inserted where the speaker stopped.
"""

from __future__ import annotations

import pytest
from autoclip.db.models import Clip
from autoclip.pipeline import boundaries, highlights
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig

WORD_S = 0.3


def words_from(
    text: str, *, pauses: dict[str, float] | None = None, gap: float = 0.0
) -> list[Word]:
    """Tokens of ``text`` back to back; ``pauses`` adds silence after a ``token@n`` marker.

    A token written ``now@`` gets the pause keyed ``"now@"`` after it, and is
    stored as plain ``now``, so markers never leak into the transcript text.
    """
    words: list[Word] = []
    t = 0.0
    for token in text.split():
        words.append(Word(text=token.rstrip("@"), start=t, end=t + WORD_S))
        t += WORD_S + gap + (pauses or {}).get(token, 0.0)
    return words


def chatter(n: int) -> str:
    """Unpunctuated streamer talk with no function words, ``n`` tokens long."""
    return " ".join(f"yo{i}" for i in range(n))


def at(transcript: Transcript, text: str, *, after: int = 0) -> int:
    return next(i for i, w in enumerate(transcript.words) if i >= after and w.text == text)


# --------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------


class TestPauseThreshold:
    def test_fast_speech_gets_the_floor(self) -> None:
        transcript = Transcript(words=words_from(chatter(50)))
        assert boundaries.pause_threshold(transcript) == boundaries.PAUSE_FLOOR_S

    def test_a_pausing_speaker_raises_it_to_their_p90(self) -> None:
        """Every fifth gap is 1.0 s: 0.8 s pauses are that speaker's rhythm, not breaks."""
        tokens = [f"yo{i}" + ("@" if i % 5 == 4 else "") for i in range(100)]
        transcript = Transcript(
            words=words_from(" ".join(tokens), pauses={t: 1.0 for t in tokens if t.endswith("@")})
        )

        assert boundaries.pause_threshold(transcript) == pytest.approx(1.0)

    def test_it_is_capped(self) -> None:
        transcript = Transcript(words=words_from(chatter(30), gap=3.0))
        assert boundaries.pause_threshold(transcript) == boundaries.PAUSE_CEILING_S

    def test_a_single_word_gets_the_floor(self) -> None:
        transcript = Transcript(words=words_from("yo"))
        assert boundaries.pause_threshold(transcript) == boundaries.PAUSE_FLOOR_S


# --------------------------------------------------------------------------
# Which pauses count
# --------------------------------------------------------------------------


class TestSentenceBreaks:
    def test_a_long_pause_in_unpunctuated_talk_is_a_break(self) -> None:
        transcript = Transcript(
            words=words_from(f"{chatter(10)} right now@ chat {chatter(10)}", pauses={"now@": 2.98})
        )
        breaks = boundaries.SentenceBreaks(transcript)

        assert breaks.soft == {at(transcript, "now")}
        assert breaks.ends_at(at(transcript, "now"))
        assert breaks.starts_at(at(transcript, "chat"))
        assert not breaks.ends_at(at(transcript, "now"), soft=False)

    def test_a_pause_after_a_function_word_is_a_hesitation(self) -> None:
        """ "I went to the ... store": the thought is still open."""
        transcript = Transcript(
            words=words_from(
                f"{chatter(10)} I went to the@ store {chatter(10)}", pauses={"the@": 1.2}
            )
        )
        assert boundaries.SentenceBreaks(transcript).soft == frozenset()

    def test_a_short_gap_is_not_a_break(self) -> None:
        transcript = Transcript(
            words=words_from(f"{chatter(10)} right now@ chat {chatter(10)}", pauses={"now@": 0.4})
        )
        assert boundaries.SentenceBreaks(transcript).soft == frozenset()

    def test_punctuation_nearby_suppresses_the_pause(self) -> None:
        """Pause three words before "500." — the ASR already marked this thought's end."""
        transcript = Transcript(
            words=words_from(
                f"{chatter(10)} they'll@ appreciate that 500. I was {chatter(10)}",
                pauses={"they'll@": 1.0},
            )
        )
        breaks = boundaries.SentenceBreaks(transcript)

        assert breaks.soft == frozenset()
        assert breaks.ends_at(at(transcript, "500."))

    def test_breaks_are_deterministic(self) -> None:
        text = f"{chatter(10)} now@ {chatter(10)} bro@ {chatter(10)}"
        pauses = {"now@": 1.0, "bro@": 0.7}
        first = boundaries.SentenceBreaks(Transcript(words=words_from(text, pauses=pauses)))
        second = boundaries.SentenceBreaks(Transcript(words=words_from(text, pauses=pauses)))

        assert first.soft == second.soft and first.threshold_s == second.threshold_s


# --------------------------------------------------------------------------
# Snapping
# --------------------------------------------------------------------------


class TestPauseSnapping:
    def test_end_lands_on_the_longest_pause_not_a_nearer_hesitation(self) -> None:
        """#24, 32:06 — "this real | talk bro I'm up right now | chat let me know chat".

        Without punctuation the end stayed wherever the model put it. From
        "talk", the 0.67 s hesitation inside "real talk" is nearer than the
        2.98 s stop after "now", but the longer pause is where the thought ends.
        """
        transcript = Transcript(
            words=words_from(
                f"{chatter(14)} bro I do this real@ talk bro I'm up right now@ chat@ let me "
                f"know chat when my skin comes out bro {chatter(14)}",
                pauses={"real@": 0.67, "now@": 2.98, "chat@": 0.67},
            )
        )
        talk = at(transcript, "talk")

        assert boundaries.snap_end_to_sentence(transcript, talk) == at(transcript, "now")

    def test_start_lands_after_a_long_pause(self) -> None:
        """#24, 31:04 — "were you uh chat | this dude thinks he's better than me bro" (9.3 s)."""
        transcript = Transcript(
            words=words_from(
                f"{chatter(14)} you gonna bot were you uh chat@ this dude thinks he's better "
                f"than me bro {chatter(14)}",
                pauses={"chat@": 9.31},
            )
        )
        thinks = at(transcript, "thinks")

        assert boundaries.snap_start_to_sentence(transcript, thinks) == at(transcript, "this")

    def test_punctuation_in_reach_beats_a_nearer_pause(self) -> None:
        """#10, 24:47 — "...same user What | All he say was they'll appreciate that 500."

        The 1.0 s pause after "What" is far enough from "500." to count as a
        break, but the punctuated end is inside the search window, so it wins.
        """
        transcript = Transcript(
            words=words_from(
                f"{chatter(14)} I am live in twitch same user What@ All he say was they'll "
                f"appreciate that 500. I was the only {chatter(14)}",
                pauses={"What@": 1.0},
            )
        )
        breaks = boundaries.SentenceBreaks(transcript)
        assert at(transcript, "What") in breaks.soft

        end = boundaries.snap_end_to_sentence(transcript, at(transcript, "What"), breaks)

        assert end == at(transcript, "500.")

    def test_no_break_in_reach_leaves_the_index(self) -> None:
        transcript = Transcript(words=words_from(chatter(60)))
        assert boundaries.snap_end_to_sentence(transcript, 30) == 30
        assert boundaries.snap_start_to_sentence(transcript, 30) == 30


# --------------------------------------------------------------------------
# Through refine() and overlap trimming
# --------------------------------------------------------------------------


class TestPausesInRefine:
    def test_unpunctuated_end_snaps_to_the_pause(self) -> None:
        """#26, 43:52 — "what do you say bro for sure wait | speed speed speed speed"."""
        transcript = Transcript(
            words=words_from(
                f"{chatter(80)} chat what do you say bro for sure wait@ speed speed speed "
                f"speed aren't you from {chatter(20)}",
                pauses={"wait@": 1.09},
            )
        )

        result = boundaries.refine(transcript, 0, at(transcript, "aren't"), min_duration_s=5)

        assert result is not None
        assert transcript.words[result.end_word].text == "wait"
        # The fallback tail stays inside the pause, short of the next word.
        assert result.end_s <= transcript.words[result.end_word + 1].start

    def test_over_long_clip_walks_back_to_a_pause(self) -> None:
        """No punctuation anywhere: the clamp used to stop on whichever word fit."""
        transcript = Transcript(
            words=words_from(f"{chatter(40)} done@ {chatter(80)}", pauses={"done@": 1.5})
        )
        done = at(transcript, "done")

        result = boundaries.refine(
            transcript, 0, len(transcript.words) - 1, min_duration_s=5, max_duration_s=20
        )

        assert result is not None
        assert result.end_word == done

    def test_explicit_break_map_matches_the_default(self) -> None:
        transcript = Transcript(
            words=words_from(f"{chatter(80)} wait@ {chatter(20)}", pauses={"wait@": 1.0})
        )
        breaks = boundaries.SentenceBreaks(transcript)

        a = boundaries.refine(transcript, 0, 78, min_duration_s=5, breaks=breaks)
        b = boundaries.refine(transcript, 0, 78, min_duration_s=5)

        assert a == b
        assert transcript.words[a.end_word].text == "wait"


def clip(id_: str, start_word: int, end_word: int, transcript: Transcript, score: int) -> Clip:
    start_s, end_s = transcript.time_range(start_word, end_word)
    return Clip(
        id=id_,
        job_id="job",
        start_s=start_s,
        end_s=end_s,
        start_word=start_word,
        end_word=end_word,
        title=f"title {id_}",
        score=score,
    )


class TestPausesInOverlaps:
    def test_weaker_clip_is_trimmed_at_a_pause_instead_of_dropped(self) -> None:
        """Unpunctuated overlap: with punctuation-only edges the weaker clip had nowhere to cut."""
        transcript = Transcript(
            words=words_from(f"{chatter(110)} gg@ {chatter(89)}", pauses={"gg@": 1.2})
        )
        gg = at(transcript, "gg")
        strong = clip("strong", 0, 99, transcript, score=80)
        weak = clip("weak", 80, len(transcript.words) - 1, transcript, score=60)
        config = DetectionConfig(min_duration_s=20, max_duration_s=40)

        result = highlights.resolve_overlaps([strong, weak], transcript, config)

        assert [c.id for c in result] == ["strong", "weak"]
        assert result[1].start_word == gg + 1
        assert result[1].start_s >= strong.end_s
