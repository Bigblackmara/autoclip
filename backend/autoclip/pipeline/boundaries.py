"""Clip boundary refinement.

An LLM proposes a word range. Turning that into a cut that sounds deliberate
takes these passes:

1. **Start** — open on the model's quoted ``hook`` when it can be found near
   the proposed start (the hook is verbatim, the index is a guess); otherwise
   snap to the nearest sentence start. Either way, drop leading connectives
   and hesitations ("and", "so", "but", "um"): a clip that opens on one sounds
   like a reply to something the viewer never heard.
2. **End** — snap to a sentence end; in unpunctuated stretches, never stop on
   a function word ("the", "my", "let").
3. **Duration clamp** — pull the end back (or push it out) to land inside the
   configured length range, always stopping on a sentence boundary.
4. **Silence alignment** — nudge the final seconds into a measured audio trough.
5. **Reaction tail** — when the audio right after the last word gets much
   louder than the clip's own speech (a laugh, a scream, "bro!"), run the end
   on until that reaction settles, bounded by the maximum length.

Step 4 is what a fixed ±0.25s pad can't do. Padding by a constant clips breaths
and plosives, because the gap before a word varies with how the speaker breathes.
Cutting inside actual silence is where a human editor would put the blade.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass

from .prepare import Loudness, Silence
from .transcript import Transcript

log = logging.getLogger(__name__)

#: How many words either side of the model's guess we'll search for a boundary.
SENTENCE_SEARCH_WORDS = 12

#: Fallback padding when no silence is measurable near the boundary.
FALLBACK_LEAD_S = 0.25
FALLBACK_TAIL_S = 0.35

#: How far from a boundary we'll look for a usable silence.
SILENCE_SEARCH_RADIUS_S = 0.75

#: How far into the silence before the first word the clip should start.
#: Enough to catch the breath before speech without a dead beat at the front.
SILENCE_LEAD_S = 0.12
#: How long to hold after the last word before cutting.
SILENCE_TAIL_S = 0.28

#: Never cut closer than this to the edge of a detected silence — silence
#: detection has hysteresis, and the true edge sits slightly inside.
SILENCE_MARGIN_S = 0.04

#: Weak first words: connectives that hang off something said before the clip,
#: and hesitation sounds. Stripped from the front of a clip.
LEADING_FILLERS = frozenset(
    {
        "and", "but", "so", "or", "because", "cause", "then", "also", "plus",
        "anyway", "anyways", "um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah",
        "hmm", "mm", "yeah", "yep", "okay", "ok",
    }
)  # fmt: skip

#: Never strip more than this many words; past that the model clearly meant it.
MAX_LEADING_FILLERS = 3

#: Function words a thought cannot end on. Only consulted when the word carries
#: no punctuation, i.e. in stretches the ASR left unpunctuated.
DANGLING_ENDS = frozenset(
    {
        "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "from",
        "about", "into", "by", "and", "but", "or", "so", "because", "if", "that",
        "my", "your", "his", "her", "their", "our", "its", "let", "i", "i'm",
        "we're", "you're", "is", "are", "was", "were", "gonna", "wanna", "gotta",
        "just", "be", "will", "would", "can", "could",
    }
)  # fmt: skip

#: How far back an unpunctuated end may retreat to shed dangling words.
MAX_DANGLING_RETREAT = 4

#: Characters of the hook compared against the transcript: enough to be
#: unambiguous, short enough to survive the model paraphrasing the tail.
HOOK_MATCH_CHARS = 24
#: Hooks shorter than this (after normalising) are too ambiguous to anchor on.
HOOK_MIN_CHARS = 8

#: The reaction must start this soon after the last word to belong to the clip.
REACTION_ONSET_S = 1.5
#: ...and be this much louder than the clip's own speech.
REACTION_RISE_LU = 6.0
#: Nothing quieter than this counts as a reaction, however quiet the speech was.
REACTION_FLOOR_LUFS = -30.0
#: Longest tail a reaction may add after the last word.
REACTION_MAX_TAIL_S = 4.0
#: Loudness back under speech level for this long means the reaction is over.
REACTION_SETTLE_S = 0.5
#: Sentences spoken during the tail longer than this are new talk, not reaction.
REACTION_MAX_SENTENCE_WORDS = 4
#: When the cut moves before a word of new talk, leave this much air before it.
REACTION_LEAD_GAP_S = 0.05
#: A tail that adds less than this after the last word isn't worth taking.
REACTION_MIN_GAIN_S = 0.3
#: Loudness at or below this is digital silence, not speech, for the baseline.
SPEECH_MIN_LUFS = -70.0


@dataclass
class Boundary:
    start_s: float
    end_s: float
    start_word: int
    end_word: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def refine(
    transcript: Transcript,
    start_word: int,
    end_word: int,
    *,
    silences: list[Silence] | None = None,
    min_duration_s: float = 20.0,
    max_duration_s: float = 90.0,
    hook: str = "",
    loudness: Loudness | None = None,
) -> Boundary | None:
    """Refine a proposed word range into a cuttable boundary.

    Returns None when the range cannot be made to satisfy the duration
    constraints — a candidate that would need to be butchered is better dropped.

    ``hook`` is the model's verbatim quote of the opening words; found near
    ``start_word`` it decides the start. ``loudness`` enables the reaction tail.

    Preconditions:
        transcript has word-level timings; start_word and end_word are valid
        indices with start_word < end_word.
    """
    if not transcript.words:
        return None

    last_index = len(transcript.words) - 1
    start_word = max(0, min(start_word, last_index))
    end_word = max(start_word, min(end_word, last_index))

    anchored = anchor_to_hook(transcript, start_word, hook) if hook else None
    if anchored is not None and anchored < end_word:
        start_word = anchored
    else:
        start_word = snap_start_to_sentence(transcript, start_word)
    end_word = snap_end_to_sentence(transcript, end_word)
    start_word = strip_leading_fillers(transcript, start_word, end_word)
    end_word = shed_dangling_end(transcript, start_word, end_word)

    if end_word <= start_word:
        return None

    end_word = _clamp_duration(transcript, start_word, end_word, min_duration_s, max_duration_s)
    if end_word is None:
        return None

    start_s, end_s = transcript.time_range(start_word, end_word)
    start_s = max(0.0, align_start(start_s, silences or []))
    end_s = align_end(end_s, silences or [])

    if loudness:
        tail = reaction_tail(
            transcript, start_word, end_word, loudness, latest_end_s=start_s + max_duration_s
        )
        if tail is not None:
            end_word, reaction_end = tail
            end_s = max(end_s, reaction_end)

    start_s, end_s = _keep_out_of_neighbours(transcript, start_word, end_word, start_s, end_s)

    # Alignment can only ever widen the clip, so re-check the ceiling.
    if end_s - start_s > max_duration_s:
        end_s = start_s + max_duration_s

    return Boundary(start_s=start_s, end_s=end_s, start_word=start_word, end_word=end_word)


def _keep_out_of_neighbours(
    transcript: Transcript, start_word: int, end_word: int, start_s: float, end_s: float
) -> tuple[float, float]:
    """Stop the padding from reaching into the words either side of the clip.

    Without a measured silence the fixed 0.25 s lead / 0.35 s tail ran straight
    into the neighbouring word wherever speech is continuous — on the Speed
    stream every unpunctuated clip end (gap 0.00 s) closed on the first
    syllable of the next sentence.
    """
    words = transcript.words
    if start_word > 0:
        start_s = max(start_s, min(words[start_word - 1].end, words[start_word].start))
    if end_word + 1 < len(words):
        end_s = min(end_s, max(words[end_word + 1].start, words[end_word].end))
    return start_s, end_s


# --------------------------------------------------------------------------
# Hook anchoring and edge clean-up
# --------------------------------------------------------------------------


def _letters(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.lower())


def _bare(text: str) -> str:
    return re.sub(r"[^\w']", "", text.lower()).strip("'")


def anchor_to_hook(transcript: Transcript, index: int, hook: str) -> int | None:
    """Find the word where the model's verbatim ``hook`` begins, near ``index``.

    The prompt asks for the hook to quote the clip's opening words exactly. On
    a real stream the sentence snap pulled starts back past it onto "But you
    know what, JC," or "I really love that speech I did bro, but yeah, yeah,":
    the model's start was right and the snap made it worse.

    Matching is on letters and digits only, so ASR tokenisation ("y 'all",
    "100 ,000") and punctuation differences don't break it. The nearest match
    wins, earlier on a tie.
    """
    key = _letters(hook)[:HOOK_MATCH_CHARS]
    if len(key) < HOOK_MIN_CHARS:
        return None
    for offset in range(SENTENCE_SEARCH_WORDS + 1):
        for candidate in (index - offset, index + offset):
            if 0 <= candidate < len(transcript.words) and _hook_starts_at(
                transcript, candidate, key
            ):
                return candidate
    return None


def _hook_starts_at(transcript: Transcript, index: int, key: str) -> bool:
    first = _letters(transcript.words[index].text)
    if not first or not key.startswith(first):
        return False
    text = ""
    for word in transcript.words[index:]:
        text += _letters(word.text)
        if len(text) >= len(key):
            break
    return text.startswith(key)


def strip_leading_fillers(transcript: Transcript, start_word: int, end_word: int) -> int:
    """Skip connectives and hesitations at the very front of a clip."""
    for _ in range(MAX_LEADING_FILLERS):
        word = transcript.words[start_word]
        if start_word >= end_word or _bare(word.text) not in LEADING_FILLERS:
            break
        start_word += 1
    return start_word


def shed_dangling_end(transcript: Transcript, start_word: int, end_word: int) -> int:
    """Back an unpunctuated end off trailing function words ("...now chat let").

    Leaves the end alone when every nearby word is a function word: shedding
    the whole tail would gut the clip for a cosmetic gain.
    """
    words = transcript.words
    for candidate in range(end_word, max(start_word, end_word - MAX_DANGLING_RETREAT) - 1, -1):
        if candidate <= start_word:
            break
        if words[candidate].ends_clause or _bare(words[candidate].text) not in DANGLING_ENDS:
            return candidate
    return end_word


# --------------------------------------------------------------------------
# Sentence snapping
# --------------------------------------------------------------------------


def snap_start_to_sentence(transcript: Transcript, index: int) -> int:
    """Move ``index`` to the first word of the nearest sentence.

    A word starts a sentence if the word before it ended one. Index 0 always
    qualifies. Ties prefer moving earlier, since starting slightly wide is
    recoverable while clipping the hook is not.
    """
    if index <= 0:
        return 0

    for offset in range(SENTENCE_SEARCH_WORDS + 1):
        for candidate in (index - offset, index + offset):
            if candidate <= 0:
                return 0
            if candidate >= len(transcript.words):
                continue
            if transcript.words[candidate - 1].ends_sentence:
                return candidate

    return index


def snap_end_to_sentence(transcript: Transcript, index: int) -> int:
    """Move ``index`` to the last word of the nearest sentence.

    Ties prefer moving later — ending on the complete thought rather than
    truncating it.
    """
    last_index = len(transcript.words) - 1
    if index >= last_index:
        return last_index

    for offset in range(SENTENCE_SEARCH_WORDS + 1):
        for candidate in (index + offset, index - offset):
            if candidate >= last_index:
                return last_index
            if candidate < 0:
                continue
            if transcript.words[candidate].ends_sentence:
                return candidate

    return index


def _clamp_duration(
    transcript: Transcript,
    start_word: int,
    end_word: int,
    min_duration_s: float,
    max_duration_s: float,
) -> int | None:
    """Adjust ``end_word`` so the clip fits the duration range.

    Returns the adjusted end index, or None if no sentence boundary produces a
    valid length.
    """
    start_s = transcript.words[start_word].start

    def duration_at(index: int) -> float:
        return transcript.words[index].end - start_s

    if duration_at(end_word) > max_duration_s:
        # Walk back to the last sentence end that fits.
        for candidate in range(end_word, start_word, -1):
            if (
                duration_at(candidate) <= max_duration_s
                and transcript.words[candidate].ends_sentence
            ):
                end_word = candidate
                break
        else:
            # No sentence boundary fits — the speaker is running long without
            # punctuation. Take the last word that fits rather than dropping it.
            for candidate in range(end_word, start_word, -1):
                if duration_at(candidate) <= max_duration_s:
                    end_word = candidate
                    break
            else:
                return None

    if duration_at(end_word) < min_duration_s:
        last_index = len(transcript.words) - 1
        for candidate in range(end_word + 1, last_index + 1):
            if duration_at(candidate) > max_duration_s:
                break
            if transcript.words[candidate].ends_sentence:
                end_word = candidate
                if duration_at(end_word) >= min_duration_s:
                    break

    duration = duration_at(end_word)
    if duration < min_duration_s or duration > max_duration_s:
        return None

    return end_word


# --------------------------------------------------------------------------
# Silence alignment
# --------------------------------------------------------------------------


def align_start(start_s: float, silences: list[Silence]) -> float:
    """Pull the clip start back into the silence preceding the first word."""
    silence = _silence_ending_near(start_s, silences)
    if silence is None:
        return max(0.0, start_s - FALLBACK_LEAD_S)

    # A silence shorter than twice the margin can't accommodate it; clamping to
    # the silence's own end keeps the cut out of the preceding word, which is
    # the point of aligning at all.
    earliest = min(silence.start + SILENCE_MARGIN_S, silence.end)
    preferred = silence.end - SILENCE_LEAD_S
    return max(0.0, max(earliest, min(preferred, start_s)))


def align_end(end_s: float, silences: list[Silence]) -> float:
    """Push the clip end into the silence following the last word."""
    silence = _silence_starting_near(end_s, silences)
    if silence is None:
        return end_s + FALLBACK_TAIL_S

    latest = max(silence.end - SILENCE_MARGIN_S, silence.start)
    preferred = silence.start + SILENCE_TAIL_S
    return max(end_s, min(preferred, latest))


def _silence_ending_near(t: float, silences: list[Silence]) -> Silence | None:
    """Find the silence whose end sits closest to ``t``, within the radius."""
    best: Silence | None = None
    best_distance = SILENCE_SEARCH_RADIUS_S
    for silence in silences:
        distance = abs(silence.end - t)
        if distance <= best_distance:
            best_distance = distance
            best = silence
    return best


def _silence_starting_near(t: float, silences: list[Silence]) -> Silence | None:
    """Find the silence whose start sits closest to ``t``, within the radius."""
    best: Silence | None = None
    best_distance = SILENCE_SEARCH_RADIUS_S
    for silence in silences:
        distance = abs(silence.start - t)
        if distance <= best_distance:
            best_distance = distance
            best = silence
    return best


# --------------------------------------------------------------------------
# Reaction tail
# --------------------------------------------------------------------------


def speech_level(
    loudness: Loudness, transcript: Transcript, start_word: int, end_word: int
) -> float | None:
    """Median loudness while the words in the range are being spoken."""
    levels: list[float] = []
    for word in transcript.slice(start_word, end_word):
        levels.extend(v for _, v in loudness.frames(word.start, word.end) if v > SPEECH_MIN_LUFS)
    return statistics.median(levels) if levels else None


def reaction_tail(
    transcript: Transcript,
    start_word: int,
    end_word: int,
    loudness: Loudness,
    *,
    latest_end_s: float,
) -> tuple[int, float] | None:
    """Extend the end through a loud reaction that starts right after it.

    On a real IShowSpeed stream 11 of 38 clips stopped within 1.5 s of a jump
    of 6-25 LU over the clip's own speech: the shout after "We did it, baby",
    "Brother! Brother!" after the skin line. Those reactions are the payoff.

    Returns ``(end_word, end_s)`` for the extended cut, or None when nothing
    loud follows. The cut lands where loudness settles back under speech
    level, or at the quietest moment of the last second when the reaction
    outlasts :data:`REACTION_MAX_TAIL_S`; never inside a word, never past
    ``latest_end_s``.
    """
    words = transcript.words
    last_end = words[end_word].end
    baseline = speech_level(loudness, transcript, start_word, end_word)
    if baseline is None:
        return None
    threshold = max(baseline + REACTION_RISE_LU, REACTION_FLOOR_LUFS)

    limit = min(last_end + REACTION_MAX_TAIL_S, latest_end_s)
    if limit <= last_end:
        return None

    # Only windows centred after the last word, so the word itself (still
    # inside a 400 ms window) can't pose as the reaction.
    onset_frames = loudness.frames(last_end + Loudness.WINDOW_S / 2, last_end + REACTION_ONSET_S)
    onset = next((t for t, v in onset_frames if v >= threshold), None)
    if onset is None or onset >= limit:
        return None

    cut: float | None = None
    quiet_since: float | None = None
    for t, v in loudness.frames(onset, limit):
        if v >= baseline:
            quiet_since = None
            continue
        if quiet_since is None:
            quiet_since = t
        if t - quiet_since >= REACTION_SETTLE_S:
            cut = quiet_since
            break
    if cut is None:
        # Still loud at the limit: cut at the quietest moment of the last second.
        last_second = loudness.frames(max(onset, limit - 1.0), limit)
        cut = min(last_second, key=lambda f: (f[1], -f[0]))[0] if last_second else limit
    cut = min(max(cut, onset), limit)

    new_end, cut = _fit_tail_words(transcript, end_word, cut, limit)
    if cut - last_end < REACTION_MIN_GAIN_S:
        return None
    log.debug("Reaction after word %d: end moved %.2fs -> %.2fs.", end_word, last_end, cut)
    return new_end, cut


def _fit_tail_words(
    transcript: Transcript, end_word: int, cut: float, limit: float
) -> tuple[int, float]:
    """Settle which words spoken during the reaction tail belong to the clip.

    Streamers keep talking through a reaction, and loud talk is not a
    reaction. Short complete exclamations ("Brother! Brother!", "Oh, my
    days.") are taken in; the first longer or unfinished sentence is new
    talk, so the cut moves to just before it. Never stops inside a word.
    """
    words = transcript.words
    new_end = end_word
    sentence: list[int] = []
    for index in range(end_word + 1, len(words)):
        word = words[index]
        if word.start >= cut:
            break
        sentence.append(index)
        if len(sentence) > REACTION_MAX_SENTENCE_WORDS or word.end > limit:
            return new_end, max(words[new_end].end, words[sentence[0]].start - REACTION_LEAD_GAP_S)
        if word.ends_sentence:
            new_end = index
            cut = max(cut, word.end)
            sentence = []
    if sentence:
        # A sentence was still going at the cut: end before it starts.
        cut = max(words[new_end].end, words[sentence[0]].start - REACTION_LEAD_GAP_S)
    return new_end, min(cut, limit)
