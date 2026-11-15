"""Boundary regressions built from a real job: a 2h44m IShowSpeed Fortnite stream.

Every word list below mirrors a clip edge AutoClip actually produced on that
job (clip rank and timestamp in each docstring). The fixes they pin:

* the model's verbatim ``hook`` decides the start when it is near the index;
* leading connectives/hesitations are stripped;
* unpunctuated ends never stop on a function word;
* a loud reaction right after the last word is kept, bounded by max length,
  while loud *new talk* after it is not;
* partially overlapping clips are merged (or trimmed) instead of both shipping.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from autoclip.db.models import Clip
from autoclip.pipeline import boundaries, highlights, prepare
from autoclip.pipeline.prepare import Loudness
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import ClipCandidate, DetectionConfig

WORD_S = 0.3
GAP_S = 0.05


def words_from(
    text: str, *, start: float = 0.0, pauses: dict[int, float] | None = None
) -> list[Word]:
    """Tokens of ``text`` at a steady pace; ``pauses`` adds silence before an index."""
    words: list[Word] = []
    t = start
    for index, token in enumerate(text.split()):
        t += (pauses or {}).get(index, 0.0)
        words.append(Word(text=token, start=t, end=t + WORD_S))
        t += WORD_S + GAP_S
    return words


def index_of(transcript: Transcript, token: str, *, after: int = 0) -> int:
    return next(i for i, w in enumerate(transcript.words) if i >= after and w.text == token)


def filler(n: int) -> str:
    """A run of complete sentences to pad a clip past the minimum length."""
    return " ".join(f"Line{i} goes here." for i in range(n))


def flat_loudness(duration_s: float, level: float) -> Loudness:
    return Loudness(values=[level] * int(duration_s / 0.1 + 1))


def set_loudness(loudness: Loudness, start: float, end: float, level: float) -> None:
    for t, _ in loudness.frames(start, end):
        loudness.values[round((t + Loudness.WINDOW_S / 2) / loudness.step_s) - 1] = level


# --------------------------------------------------------------------------
# Starts
# --------------------------------------------------------------------------


class TestHookAnchoredStart:
    def test_connective_before_the_hook_is_not_pulled_in(self) -> None:
        """#19, 1:23:42 — "Yeah. And Speed, what was your favorite 2K?" opened on "And"."""
        transcript = Transcript(
            words=words_from(
                f"in Tilted Towers. Yeah. And Speed, what was your favorite 2K? {filler(12)}"
            )
        )
        speed = index_of(transcript, "Speed,")

        result = boundaries.refine(
            transcript,
            speed,
            len(transcript.words) - 1,
            min_duration_s=5,
            hook="Speed, what was your favorite 2K?",
        )

        assert result is not None
        assert transcript.words[result.start_word].text == "Speed,"

    def test_connective_is_stripped_even_without_a_hook(self) -> None:
        """Same clip with no hook: the sentence snap lands on "And", which is then dropped."""
        transcript = Transcript(
            words=words_from(
                f"in Tilted Towers. Yeah. And Speed, what was your favorite 2K? {filler(12)}"
            )
        )

        result = boundaries.refine(
            transcript, index_of(transcript, "Speed,"), len(transcript.words) - 1, min_duration_s=5
        )

        assert result is not None
        assert transcript.words[result.start_word].text == "Speed,"

    def test_reply_preamble_before_the_hook_is_cut(self) -> None:
        """#02, 1:34:11 — "But you know what, JC, when we played..." (hook began at "JC,")."""
        transcript = Transcript(
            words=words_from(
                f"that was, uh, okay. But you know what, JC, when we played, like, you spoke "
                f"to me, your words really hit me hard. {filler(12)}"
            )
        )
        but = index_of(transcript, "But")

        result = boundaries.refine(
            transcript,
            but,
            len(transcript.words) - 1,
            min_duration_s=5,
            hook="JC, when we played, like, you spoke to me, your words really hit me hard.",
        )

        assert result is not None
        assert transcript.words[result.start_word].text == "JC,"
        assert (
            result.start_s >= transcript.words[result.start_word].start - boundaries.FALLBACK_LEAD_S
        )

    def test_hook_eleven_words_into_the_sentence_still_anchors(self) -> None:
        """#37, 0:14:21 — opened 4 s early, on "I really love that speech I did bro, but yeah"."""
        transcript = Transcript(
            words=words_from(
                "Yes indeed chat. I really love that speech I did bro, but yeah, yeah, "
                f"I didn't understand what I said bro. {filler(12)}"
            )
        )
        sentence_start = index_of(transcript, "I")

        result = boundaries.refine(
            transcript,
            sentence_start,
            len(transcript.words) - 1,
            min_duration_s=5,
            hook="I didn't understand what I said",
        )

        assert result is not None
        assert (
            transcript.text_between(result.start_word, result.start_word + 2)
            == "I didn't understand"
        )

    def test_hook_matching_ignores_asr_tokenisation(self) -> None:
        """#05/#30 — Whisper writes "100 ,000" and "y 'all"; the model writes "100,000", "y'all"."""
        transcript = Transcript(
            words=words_from(
                f"on me, though? Wait, so you really unboxed almost 100 ,000 chests? {filler(12)}"
            )
        )

        anchored = boundaries.anchor_to_hook(
            transcript, 0, "Wait, so you really unboxed almost 100,000 chests?"
        )

        assert anchored == index_of(transcript, "Wait,")

    def test_unknown_hook_falls_back_to_the_sentence_snap(self) -> None:
        transcript = Transcript(words=words_from(f"Before this. Real start here now. {filler(12)}"))

        result = boundaries.refine(
            transcript,
            index_of(transcript, "here"),
            len(transcript.words) - 1,
            min_duration_s=5,
            hook="words the model invented",
        )

        assert result is not None
        assert transcript.words[result.start_word].text == "Real"

    def test_short_hooks_are_too_ambiguous_to_anchor(self) -> None:
        transcript = Transcript(words=words_from("bro bro bro yo bro"))

        assert boundaries.anchor_to_hook(transcript, 2, "bro") is None

    def test_at_most_three_fillers_are_stripped(self) -> None:
        transcript = Transcript(words=words_from("so and um yeah okay the point is clear."))

        assert boundaries.strip_leading_fillers(transcript, 0, len(transcript.words) - 1) == 3

    def test_stripping_never_consumes_the_whole_clip(self) -> None:
        transcript = Transcript(words=words_from("so and"))

        assert boundaries.strip_leading_fillers(transcript, 0, 1) == 1


# --------------------------------------------------------------------------
# Ends
# --------------------------------------------------------------------------


class TestUnpunctuatedEnds:
    def test_padding_never_reaches_into_the_next_word(self) -> None:
        """#14, 0:35:00 — "mute her you heard bro" ran 0.35 s into "the acne" (0.00 s gap)."""
        words = [
            Word(text=token, start=i * 0.3, end=(i + 1) * 0.3)
            for i, token in enumerate(
                ("I don't want to hear your girlfriend bro mute her you heard bro " * 8).split()
                + ["the", "acne", "to", "him", "on", "some", "acne", "see", "bro"] * 3
            )
        ]
        transcript = Transcript(words=words)
        start, end = 20, index_of(transcript, "the") - 1

        result = boundaries.refine(transcript, start, end, min_duration_s=5)

        assert result is not None
        assert (result.start_word, result.end_word) == (start, end)
        assert result.end_s <= transcript.words[end + 1].start
        assert result.start_s >= transcript.words[start - 1].end

    def test_end_backs_off_a_dangling_function_word(self) -> None:
        """#24, 0:32:11 — unpunctuated Whisper text, the clip ended on "...right now chat let"."""
        transcript = Transcript(
            words=words_from(
                "I do this real talk bro I'm up right now chat let me know chat when my skin"
            )
        )
        let = index_of(transcript, "let")

        assert transcript.words[boundaries.shed_dangling_end(transcript, 0, let)].text == "chat"

    def test_end_is_left_alone_when_everything_nearby_dangles(self) -> None:
        transcript = Transcript(words=words_from("yes the of to a the"))

        assert boundaries.shed_dangling_end(transcript, 0, 5) == 5

    def test_punctuated_function_word_is_a_valid_end(self) -> None:
        """#38 ends "Spot on come on." — the full stop makes "on." a real end."""
        transcript = Transcript(words=words_from("I like this one Spot on come on."))

        assert boundaries.shed_dangling_end(transcript, 0, 7) == 7


class TestReactionTail:
    @staticmethod
    def stream(after: str) -> Transcript:
        # ~25 s of speech, the punchline, then whatever follows it.
        return Transcript(
            words=words_from(f"{filler(18)} We did it, baby. We did it. {after}", pauses=None)
        )

    def punchline_end(self, transcript: Transcript) -> int:
        first = index_of(transcript, "We")
        return index_of(transcript, "it.", after=index_of(transcript, "We", after=first + 1))

    def test_loud_reaction_after_the_last_word_is_kept(self) -> None:
        """#18, 1:01:18 — cut on "We did it." with a 12+ LU shout starting right after."""
        transcript = self.stream("Oh, my days. Now let me tell you about the next game we play.")
        end = self.punchline_end(transcript)
        last_end = transcript.words[end].end
        loudness = flat_loudness(transcript.duration + 5, -22.0)
        set_loudness(loudness, last_end + 0.2, last_end + 1.6, -9.0)

        result = boundaries.refine(
            transcript, 0, end, min_duration_s=5, max_duration_s=90, loudness=loudness
        )

        assert result is not None
        # "Oh, my days." is a short exclamation inside the reaction: taken in.
        assert transcript.words[result.end_word].text == "days."
        assert result.end_s >= transcript.words[result.end_word].end
        # The longer sentence after it is new talk: the cut lands before it.
        now = index_of(transcript, "Now")
        assert result.end_s <= transcript.words[now].start

    def test_loud_new_talk_is_not_a_reaction(self) -> None:
        """#03, 2:05:00 — loud, but it's "There's one more chest here, bro" — a new topic."""
        transcript = self.stream("There's one more chest here, bro. Now you know it.")
        end = self.punchline_end(transcript)
        last_end = transcript.words[end].end
        loudness = flat_loudness(transcript.duration + 5, -22.0)
        set_loudness(loudness, last_end + 0.2, last_end + 3.0, -9.0)

        result = boundaries.refine(
            transcript, 0, end, min_duration_s=5, max_duration_s=90, loudness=loudness
        )

        assert result is not None
        assert result.end_word == end
        assert result.end_s <= transcript.words[end + 1].start

    def test_wordless_scream_runs_until_it_settles(self) -> None:
        """#15, 1:06:28 — screaming in a transcript gap: hold until loudness drops."""
        words = self.stream("").words
        words += words_from("Okay next round.", start=words[-1].end + 4.0)
        transcript = Transcript(words=words)
        end = self.punchline_end(transcript)
        last_end = transcript.words[end].end
        loudness = flat_loudness(transcript.duration + 5, -22.0)
        set_loudness(loudness, last_end, last_end + 4.0, -40.0)
        set_loudness(loudness, last_end + 0.3, last_end + 2.5, -8.0)

        result = boundaries.refine(
            transcript, 0, end, min_duration_s=5, max_duration_s=90, loudness=loudness
        )

        assert result is not None
        assert result.end_word == end
        assert last_end + 2.0 <= result.end_s <= last_end + 3.2

    def test_speech_level_after_the_end_changes_nothing(self) -> None:
        transcript = self.stream("Oh, my days.")
        end = self.punchline_end(transcript)
        loudness = flat_loudness(transcript.duration + 5, -22.0)

        with_profile = boundaries.refine(transcript, 0, end, min_duration_s=5, loudness=loudness)
        without = boundaries.refine(transcript, 0, end, min_duration_s=5)

        assert with_profile == without

    def test_late_reaction_does_not_belong_to_the_clip(self) -> None:
        transcript = self.stream("")
        end = self.punchline_end(transcript)
        last_end = transcript.words[end].end
        loudness = flat_loudness(transcript.duration + 8, -22.0)
        set_loudness(loudness, last_end + boundaries.REACTION_ONSET_S + 0.5, last_end + 4, -8.0)

        assert boundaries.reaction_tail(transcript, 0, end, loudness, latest_end_s=1e9) is None

    def test_reaction_never_breaks_the_maximum_length(self) -> None:
        transcript = self.stream("")
        end = self.punchline_end(transcript)
        last_end = transcript.words[end].end
        loudness = flat_loudness(transcript.duration + 8, -22.0)
        set_loudness(loudness, last_end, last_end + 8, -8.0)
        max_duration = last_end - transcript.words[0].start + 1.0

        result = boundaries.refine(
            transcript, 0, end, min_duration_s=5, max_duration_s=max_duration, loudness=loudness
        )

        assert result is not None
        assert result.duration_s <= max_duration + 1e-9

    def test_quiet_floor_stops_hum_counting_as_a_reaction(self) -> None:
        """A whispered clip at -50 LUFS followed by -40 LUFS room tone is no reaction."""
        transcript = self.stream("")
        end = self.punchline_end(transcript)
        last_end = transcript.words[end].end
        loudness = flat_loudness(transcript.duration + 5, -50.0)
        set_loudness(loudness, last_end, last_end + 3, -40.0)

        assert boundaries.reaction_tail(transcript, 0, end, loudness, latest_end_s=1e9) is None


# --------------------------------------------------------------------------
# Overlaps
# --------------------------------------------------------------------------


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


class TestOverlaps:
    @pytest.fixture
    def transcript(self) -> Transcript:
        return Transcript(words=words_from(filler(60)))  # 180 words, ~63 s

    def test_overlapping_pair_merges_when_the_union_fits(self, transcript: Transcript) -> None:
        """#28 + #32, 1:25:24 — padded to 20 s, #28 swallowed #32's opening line (3.4 s shared)."""
        a = clip("awards", 0, 59, transcript, score=71)
        b = clip("backflip", 50, 119, transcript, score=70)

        merged = highlights.resolve_overlaps([a, b], transcript, DetectionConfig(max_duration_s=90))

        assert len(merged) == 1
        assert merged[0].title == "title awards"
        assert (merged[0].start_word, merged[0].end_word) == (0, 119)
        assert merged[0].start_s == a.start_s and merged[0].end_s == b.end_s

    def test_weaker_clip_is_trimmed_when_the_union_is_too_long(
        self, transcript: Transcript
    ) -> None:
        """#21 + #25, 1:57:48 — 13.9 s shared; here the union would break the ceiling."""
        a = clip("strong", 0, 59, transcript, score=73)
        b = clip("weak", 45, 149, transcript, score=72)
        config = DetectionConfig(min_duration_s=10, max_duration_s=40)

        result = highlights.resolve_overlaps([a, b], transcript, config)

        assert [c.id for c in result] == ["strong", "weak"]
        weak = result[1]
        assert weak.start_s >= a.end_s
        assert transcript.words[weak.start_word - 1].ends_sentence

    def test_weaker_clip_is_dropped_when_trimming_leaves_too_little(
        self, transcript: Transcript
    ) -> None:
        a = clip("strong", 0, 59, transcript, score=80)
        b = clip("weak", 40, 70, transcript, score=60)
        config = DetectionConfig(min_duration_s=20, max_duration_s=22)

        assert [c.id for c in highlights.resolve_overlaps([a, b], transcript, config)] == ["strong"]

    def test_touching_clips_are_left_alone(self, transcript: Transcript) -> None:
        a = clip("a", 0, 59, transcript, score=80)
        b = clip("b", 60, 119, transcript, score=70)
        b.start_s -= highlights.OVERLAP_TOLERANCE_S / 2  # silence padding crossing over

        result = highlights.resolve_overlaps([a, b], transcript, DetectionConfig())

        assert {c.id for c in result} == {"a", "b"}

    def test_build_clips_uses_the_hook_and_merges_overlaps(self, transcript: Transcript) -> None:
        candidates = [
            ClipCandidate(start_word_index=0, end_word_index=59, hook="Line1 goes here.", score=80),
            ClipCandidate(start_word_index=50, end_word_index=119, score=70),
        ]

        clips = highlights.build_clips(
            transcript, candidates, DetectionConfig(min_duration_s=5), job_id="job"
        )

        assert len(clips) == 1
        assert transcript.words[clips[0].start_word].text == "Line1"


# --------------------------------------------------------------------------
# Loudness profile plumbing
# --------------------------------------------------------------------------


class TestLoudnessProfile:
    def test_parse_handles_silence_markers(self) -> None:
        text = "\n".join(
            ["frame:0", "lavfi.r128.M=-120.691", "lavfi.r128.M=nan", "lavfi.r128.M=-18.5"]
            + ["lavfi.r128.M=-inf"]
        )

        assert prepare.parse_loudness(text).values == [-120.0, -120.0, -18.5, -120.0]

    def test_frames_report_window_centres(self) -> None:
        loudness = Loudness(values=[float(i) for i in range(30)])

        frames = loudness.frames(1.0, 1.2)

        assert [round(t, 2) for t, _ in frames] == [1.0, 1.1, 1.2]
        # The window centred at 1.0 s is the one that ends at 1.2 s: index 11.
        assert frames[0][1] == 11.0

    def test_round_trips_through_json(self) -> None:
        loudness = Loudness(values=[-20.04, -8.0])

        restored = Loudness.from_dict(json.loads(json.dumps(loudness.to_dict())))

        assert restored.values == [-20.0, -8.0] and restored.step_s == 0.1

    def test_failed_measurement_gives_an_empty_profile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(prepare.ffmpeg, "ffmpeg_path", lambda: "ffmpeg")
        monkeypatch.setattr(
            prepare.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="boom"),
        )

        assert not prepare.measure_loudness(tmp_path / "audio.wav")

    def test_missing_ffmpeg_gives_an_empty_profile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        def missing() -> str:
            raise prepare.ffmpeg.FFmpegError("ffmpeg was not found", command=["ffmpeg"])

        monkeypatch.setattr(prepare.ffmpeg, "ffmpeg_path", missing)

        assert not prepare.measure_loudness(tmp_path / "audio.wav")


class TestRunnerLoudnessCache:
    def test_profile_is_measured_once_and_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        from autoclip.pipeline.runner import JobWorkspace, PipelineRunner

        calls: list[object] = []

        def measure(audio):
            calls.append(audio)
            return Loudness(values=[-20.0, -8.0])

        monkeypatch.setattr(prepare, "measure_loudness", measure)
        fake = SimpleNamespace(workspace=JobWorkspace("job-loudness"))

        first = PipelineRunner._load_or_measure_loudness(fake, "audio.wav")
        second = PipelineRunner._load_or_measure_loudness(fake, "audio.wav")

        assert first.values == second.values == [-20.0, -8.0]
        assert len(calls) == 1

    def test_failed_measurement_is_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        from autoclip.pipeline.runner import JobWorkspace, PipelineRunner

        monkeypatch.setattr(prepare, "measure_loudness", lambda audio: Loudness(values=[]))
        fake = SimpleNamespace(workspace=JobWorkspace("job-no-ffmpeg"))

        assert not PipelineRunner._load_or_measure_loudness(fake, "audio.wav")
        assert not fake.workspace.loudness.exists()
