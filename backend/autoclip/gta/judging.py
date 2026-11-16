"""GTA Mode judging: 12-minute chunks -> Phase 1 filter -> Phase 2 final round.

1. The transcript is read in chunks of exactly ``chunk_minutes`` (plus a
   read-only tail so a payoff just past the seam can be judged), with the
   Hype Curve's markers for that chunk in the prompt.
2. Phase 1 throws a proposal out immediately when it needs prior context
   (the model's own flag) or when the thought starts on filler — checked
   here on the actual words, not trusted to the model.
3. Phase 2: every survivor is judged as the exact clip a stranger would see
   against three yes/no tests (scroll-stop in 2 s, context-free, payoff). One
   "no" and it is out. Survivors go head-to-head on score (nudged by what the
   feedback loop learned works) and the same moment is never kept twice.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from ..pipeline import boundaries
from ..pipeline.boundaries import SentenceBreaks
from ..pipeline.prepare import Silence
from ..pipeline.transcript import Transcript
from ..providers.base import DetectionConfig, ProviderError, extract_json_object, load_prompt
from .hype import HypeCurve
from .settings import GtaBias, GtaSettings

log = logging.getLogger(__name__)

CHUNK_PROMPT = "gta_chunk_v1"
FINAL_PROMPT = "gta_final_v1"

MOMENT_TYPES = ("first_reaction", "crazy_discovery", "freakout", "funny_fail", "hot_take", "other")

#: Hesitations and hanging connectives a clip may never open on.
FILLER_WORDS = frozenset(
    {
        "uh", "uhh", "uhm", "um", "umm", "erm", "er", "ah", "ahh", "hmm", "mm", "mhm",
        "so", "and", "but", "yeah", "yea", "like", "anyway", "anyways", "okay", "ok",
        "alright", "well", "or", "because", "cause", "then",
    }
)  # fmt: skip
#: Two-word openers that are filler even when the first word alone isn't.
FILLER_PHRASES = frozenset(
    {"i mean", "you know", "so yeah", "yeah so", "and uh", "so uh", "like uh", "okay so"}
)

#: Candidates per Phase 2 call (each carries its full clip text).
FINAL_BATCH = 6
#: Two kept clips sharing more than this much stream time are the same moment.
SAME_MOMENT_OVERLAP_S = 4.0

#: Completion function: (system, user, config) -> raw model text.
Complete = Callable[[str, str, DetectionConfig], Awaitable[str]]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    index: int
    start_s: float
    end_s: float
    first_word: int
    #: Last word that may START a clip.
    last_word: int
    #: Last word of the read-only tail (>= last_word).
    tail_word: int


def build_chunks(
    transcript: Transcript,
    *,
    chunk_s: float,
    tail_s: float,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[Chunk]:
    """Split the transcript into back-to-back chunks of exactly ``chunk_s``.

    Chunks are laid on the stream clock (0-12 min, 12-24 min, ...) from
    ``start_s``; a chunk with no speech is skipped, not merged.
    """
    words = transcript.words
    if not words:
        return []
    stop = end_s if end_s is not None else words[-1].end
    chunks: list[Chunk] = []
    t = start_s
    index = 0
    while t < stop:
        c_end = min(t + chunk_s, stop)
        first = _first_word_at_or_after(transcript, t)
        if first is not None and words[first].start < c_end:
            last = _last_word_starting_before(transcript, c_end)
            tail = _last_word_starting_before(transcript, min(c_end + tail_s, stop + tail_s))
            if last is not None and last >= first:
                chunks.append(
                    Chunk(
                        index=index,
                        start_s=t,
                        end_s=c_end,
                        first_word=first,
                        last_word=last,
                        tail_word=max(last, tail if tail is not None else last),
                    )
                )
        t += chunk_s
        index += 1
    return chunks


def _first_word_at_or_after(transcript: Transcript, t: float) -> int | None:
    for i in range(transcript.index_at_time(t), len(transcript.words)):
        if transcript.words[i].start >= t:
            return i
    return None


def _last_word_starting_before(transcript: Transcript, t: float) -> int | None:
    words = transcript.words
    i = min(transcript.index_at_time(t), len(words) - 1)
    while i >= 0 and words[i].start >= t:
        i -= 1
    return i if i >= 0 else None


def render_chunk(transcript: Transcript, chunk: Chunk) -> str:
    parts: list[str] = []
    for index in range(chunk.first_word, chunk.tail_word + 1):
        if index == chunk.last_word + 1:
            parts.append("\n\nTAIL (read-only — clips may end here but not start here):\n")
        parts.append(f"[{index}]{transcript.words[index].text}")
    return " ".join(parts).strip()


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def render_markers(curve: HypeCurve | None, chunk: Chunk, transcript: Transcript) -> str:
    if curve is None:
        return "(no hype signals available for this stream)"
    markers = curve.markers_between(chunk.start_s, chunk.end_s)
    if not markers:
        return "(no hype markers in this chunk)"
    lines = []
    for marker in markers:
        word = transcript.index_at_time(marker.t)
        reasons = "; ".join(marker.reasons) or "combined signals"
        lines.append(
            f"- {_clock(marker.t)} (near word [{word}]) hype {marker.score:.0f}/100: {reasons}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Model output
# --------------------------------------------------------------------------


class _Moment(BaseModel):
    start_word: int
    end_word: int
    hook_quote: str = ""
    moment_type: str = "other"
    why: str = ""
    needs_prior_context: bool = False
    starts_with_filler: bool = False
    confidence: int = 50

    @field_validator("hook_quote", "why", "moment_type", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("needs_prior_context", "starts_with_filler", mode="before")
    @classmethod
    def _flag(cls, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return bool(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _score(cls, value: Any) -> int:
        try:
            return max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            return 50


class _ChunkOut(BaseModel):
    moments: list[_Moment] = []


class _Answer(BaseModel):
    answer: str = "no"
    why: str = ""

    @field_validator("answer", mode="before")
    @classmethod
    def _yes_no(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        return "yes" if text in {"yes", "y", "true", "1"} else "no"

    @field_validator("why", mode="before")
    @classmethod
    def _why(cls, value: Any) -> str:
        return "" if value is None else str(value)


class _Verdict(BaseModel):
    id: str
    scroll_stop: _Answer = _Answer()
    context_free: _Answer = _Answer()
    payoff: _Answer = _Answer()
    score: int = 0
    hook_title: str = ""
    caption: str = ""
    hashtags: list[str] = []
    moment_type: str = "other"

    @field_validator("id", "hook_title", "caption", "moment_type", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("score", mode="before")
    @classmethod
    def _score(cls, value: Any) -> int:
        try:
            return max(0, min(100, int(float(value))))
        except (TypeError, ValueError):
            return 0

    @field_validator("scroll_stop", "context_free", "payoff", mode="before")
    @classmethod
    def _answer(cls, value: Any) -> Any:
        # Accept a bare "yes"/"no" as well as {"answer": ..., "why": ...}.
        if isinstance(value, str | bool):
            return {"answer": str(value)}
        return value

    @field_validator("hashtags", mode="before")
    @classmethod
    def _tags(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = value.replace(",", " ").split()
        return [str(v) for v in value or []]


class _FinalOut(BaseModel):
    verdicts: list[_Verdict] = []


async def _ask(
    complete: Complete,
    system: str,
    user: str,
    config: DetectionConfig,
    model: type[BaseModel],
) -> BaseModel:
    """One provider call with one repair retry on malformed JSON."""
    raw = await complete(system, user, config)
    try:
        return model.model_validate(extract_json_object(raw))
    except (ValidationError, ValueError) as first_error:
        repair = (
            f"{user}\n\nYour previous response did not match the required schema.\n"
            f"Validation error:\n{first_error}\n\n"
            "Respond again with ONLY the corrected JSON object. No prose, no markdown fences."
        )
        raw = await complete(system, repair, config)
        try:
            return model.model_validate(extract_json_object(raw))
        except (ValidationError, ValueError) as second_error:
            raise ProviderError(
                "The AI judge returned malformed data twice.",
                hint=f"Last validation error: {second_error}",
            ) from second_error


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    id: str
    chunk_index: int
    start_word: int
    end_word: int
    start_s: float
    end_s: float
    hook_quote: str
    moment_type: str
    why: str
    confidence: int
    hype_score: float = 0.0
    status: str = "candidate"  # candidate | rejected | approved
    reject_reason: str = ""
    verdict: dict[str, Any] = field(default_factory=dict)
    score: int = 0
    rank_score: float = 0.0
    hook_title: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.status = "rejected"
        self.reject_reason = reason


def _normalise(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def opens_on_filler(transcript: Transcript, start_word: int) -> str | None:
    """The filler a clip starting at ``start_word`` opens on, else None."""
    words = transcript.words
    if not 0 <= start_word < len(words):
        return None
    first = _normalise(words[start_word].text)
    if first in FILLER_WORDS:
        return first
    if start_word + 1 < len(words):
        pair = f"{first} {_normalise(words[start_word + 1].text)}"
        if pair in FILLER_PHRASES:
            return pair
    return None


def _clean_type(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("-", "_")
    return value if value in MOMENT_TYPES else "other"


async def find_candidates(
    transcript: Transcript,
    chunks: list[Chunk],
    *,
    complete: Complete,
    settings: GtaSettings,
    curve: HypeCurve | None,
    silences: list[Silence],
    streamer: str,
    title: str,
    on_chunk: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    done_chunks: dict[int, list[Candidate]] | None = None,
    on_chunk_result: Callable[[int, list[Candidate]], None] | None = None,
) -> list[Candidate]:
    """Read every chunk and return all proposals, Phase 1 verdicts applied.

    ``done_chunks`` carries results from a previous (interrupted) run so a
    resumed stream never pays for a chunk twice; ``on_chunk_result`` lets the
    caller persist each chunk as soon as it is judged.
    """
    system = load_prompt(CHUNK_PROMPT)
    system = system.replace("{min_s}", f"{settings.min_clip_s:.0f}").replace(
        "{max_s}", f"{settings.max_clip_s:.0f}"
    )
    config = DetectionConfig(prompt_version=CHUNK_PROMPT, temperature=0.3)
    breaks = SentenceBreaks(transcript)
    done = done_chunks or {}
    out: list[Candidate] = []

    for position, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise JudgingCancelled("Cancelled.")
        if chunk.index in done:
            out.extend(done[chunk.index])
            if on_chunk:
                on_chunk(position + 1, len(chunks))
            continue
        user = (
            f"STREAMER: {streamer or 'unknown'}\n"
            f"STREAM: {title or 'untitled'}\n"
            f"CHUNK: {_clock(chunk.start_s)}-{_clock(chunk.end_s)} "
            f"(chunk {position + 1} of {len(chunks)})\n\n"
            f"HYPE MARKERS:\n{render_markers(curve, chunk, transcript)}\n\n"
            f"TRANSCRIPT:\n{render_chunk(transcript, chunk)}"
        )
        parsed = await _ask(complete, system, user, config, _ChunkOut)
        assert isinstance(parsed, _ChunkOut)
        found: list[Candidate] = []
        for number, moment in enumerate(parsed.moments[:5]):
            candidate = _phase1(
                transcript,
                chunk,
                moment,
                settings=settings,
                silences=silences,
                breaks=breaks,
                curve=curve,
                candidate_id=f"c{chunk.index}-{number}",
            )
            if candidate is not None:
                found.append(candidate)
        if on_chunk_result:
            on_chunk_result(chunk.index, found)
        out.extend(found)
        if on_chunk:
            on_chunk(position + 1, len(chunks))
    return out


class JudgingCancelled(RuntimeError):
    pass


def _phase1(
    transcript: Transcript,
    chunk: Chunk,
    moment: _Moment,
    *,
    settings: GtaSettings,
    silences: list[Silence],
    breaks: SentenceBreaks,
    curve: HypeCurve | None,
    candidate_id: str,
) -> Candidate | None:
    """Turn one proposal into a Candidate with its Phase 1 verdict.

    Returns None only for proposals too malformed to even record (indices
    outside the chunk). Rejected candidates are returned with their reason.
    """
    start = moment.start_word
    end = moment.end_word
    if not chunk.first_word <= start <= chunk.last_word or end <= start:
        log.info("Dropping a proposal outside its chunk (%s-%s).", start, end)
        return None
    end = min(end, chunk.tail_word)

    # Where the clip really starts: the hook quote is verbatim, the index a guess.
    anchored = boundaries.anchor_to_hook(transcript, start, moment.hook_quote)
    if anchored is not None and chunk.first_word <= anchored < end:
        start = anchored

    start_s, end_s = transcript.time_range(start, end)
    candidate = Candidate(
        id=candidate_id,
        chunk_index=chunk.index,
        start_word=start,
        end_word=end,
        start_s=start_s,
        end_s=end_s,
        hook_quote=moment.hook_quote,
        moment_type=_clean_type(moment.moment_type),
        why=moment.why,
        confidence=moment.confidence,
        hype_score=curve.peak(start_s, end_s) if curve is not None else 0.0,
    )

    filler = opens_on_filler(transcript, start)
    if filler is not None:
        candidate.reject(f'Phase 1: starts with filler ("{filler}")')
        return candidate
    if moment.starts_with_filler:
        candidate.reject("Phase 1: starts with filler (flagged by the judge)")
        return candidate
    if moment.needs_prior_context:
        candidate.reject("Phase 1: needs prior context to understand")
        return candidate

    # Tighten the END to a sentence boundary and the length window; the
    # start is final (it passed the filler check as it is).
    end = boundaries.snap_end_to_sentence(transcript, end, breaks)
    end = boundaries.shed_dangling_end(transcript, start, end)
    if end <= start:
        candidate.reject("Phase 1: no complete thought")
        return candidate
    clamped = boundaries._clamp_duration(  # noqa: SLF001 - shared boundary rules
        transcript, start, end, settings.min_clip_s, settings.max_clip_s, breaks
    )
    if clamped is None:
        candidate.reject(
            f"Phase 1: can't be cut to {settings.min_clip_s:.0f}-{settings.max_clip_s:.0f} s"
        )
        return candidate
    end = clamped
    start_s, end_s = transcript.time_range(start, end)
    start_s = max(0.0, boundaries.align_start(start_s, silences))
    end_s = boundaries.align_end(end_s, silences)
    candidate.end_word = end
    candidate.start_s = start_s
    candidate.end_s = end_s
    if curve is not None:
        candidate.hype_score = curve.peak(start_s, end_s)
    return candidate


# --------------------------------------------------------------------------
# Phase 2
# --------------------------------------------------------------------------


def _clip_text(transcript: Transcript, candidate: Candidate) -> str:
    words = transcript.slice(candidate.start_word, candidate.end_word)
    lines: list[str] = []
    for word in words:
        lines.append(word.text)
    return " ".join(lines)


def _first_two_seconds(transcript: Transcript, candidate: Candidate) -> str:
    words = transcript.slice(candidate.start_word, candidate.end_word)
    limit = candidate.start_s + 2.0
    return " ".join(w.text for w in words if w.start < limit)


def _bias_note(bias: GtaBias) -> str:
    if not bias.summary and not bias.moment_types and not bias.streamers:
        return ""
    lines = [
        "WHAT IS WORKING ON THE CHANNEL (from real view counts; use as a tie-breaker "
        "in score, never to pass a clip that fails a test):"
    ]
    if bias.summary:
        lines.append(bias.summary)
    for name, mult in sorted(bias.moment_types.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {name}: {mult:.2f}x the channel average")
    return "\n".join(lines)


def bias_multiplier(bias: GtaBias, moment_type: str, streamer: str) -> float:
    """How much the feedback loop nudges a clip's head-to-head score."""
    mult = 1.0
    if moment_type in bias.moment_types:
        mult *= bias.moment_types[moment_type]
    key = streamer.strip().lower()
    for name, value in bias.streamers.items():
        if name.strip().lower() == key:
            mult *= value
            break
    # Dampened and bounded: data steers selection, it never overrides judging.
    return max(0.6, min(1.6, 1.0 + 0.5 * (mult - 1.0)))


async def final_round(
    transcript: Transcript,
    candidates: list[Candidate],
    *,
    complete: Complete,
    settings: GtaSettings,
    streamer: str,
    title: str,
    curve: HypeCurve | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[Candidate]:
    """Phase 2 on every Phase 1 survivor; returns ALL candidates, updated."""
    system = load_prompt(FINAL_PROMPT)
    config = DetectionConfig(prompt_version=FINAL_PROMPT, temperature=0.2)
    alive = [c for c in candidates if c.status == "candidate"]
    note = _bias_note(settings.bias)

    for offset in range(0, len(alive), FINAL_BATCH):
        if cancelled is not None and cancelled():
            raise JudgingCancelled("Cancelled.")
        batch = alive[offset : offset + FINAL_BATCH]
        blocks = []
        for c in batch:
            reasons = ""
            if curve is not None:
                hot = curve.markers_between(c.start_s, c.end_s)
                if hot:
                    reasons = "; ".join(r for m in hot for r in m.reasons)
            blocks.append(
                f"### id: {c.id}\n"
                f"length: {c.end_s - c.start_s:.0f} s\n"
                f"first 2 seconds: {_first_two_seconds(transcript, c)}\n"
                f"what the finder saw: {c.why}\n"
                f"live signals during the clip: {reasons or 'none recorded'}\n"
                f"FULL CLIP TRANSCRIPT: {_clip_text(transcript, c)}"
            )
        user = (
            f"STREAMER: {streamer or 'unknown'}\nSTREAM: {title or 'untitled'}\n"
            + (f"\n{note}\n" if note else "")
            + "\nCANDIDATES:\n\n"
            + "\n\n".join(blocks)
        )
        parsed = await _ask(complete, system, user, config, _FinalOut)
        assert isinstance(parsed, _FinalOut)
        by_id = {v.id.strip(): v for v in parsed.verdicts}
        for c in batch:
            verdict = by_id.get(c.id)
            if verdict is None:
                c.reject("Phase 2: the judge gave no verdict")
                continue
            _apply_verdict(c, verdict, settings, streamer)

    _head_to_head(candidates, settings)
    return candidates


def _apply_verdict(c: Candidate, verdict: _Verdict, settings: GtaSettings, streamer: str) -> None:
    c.verdict = {
        "scroll_stop": verdict.scroll_stop.model_dump(),
        "context_free": verdict.context_free.model_dump(),
        "payoff": verdict.payoff.model_dump(),
    }
    c.score = verdict.score
    c.hook_title = _clean_title(verdict.hook_title)
    c.caption = verdict.caption.strip()
    c.hashtags = clean_hashtags(verdict.hashtags)
    if verdict.moment_type:
        c.moment_type = _clean_type(verdict.moment_type)
    failed = [
        label
        for label, answer in (
            ("won't stop the scroll in 2 s", verdict.scroll_stop),
            ("needs outside context", verdict.context_free),
            ("no payoff inside the clip", verdict.payoff),
        )
        if answer.answer != "yes"
    ]
    if failed:
        c.reject("Phase 2: " + "; ".join(failed))
        return
    if not c.hook_title:
        c.reject("Phase 2: no usable hook title")
        return
    c.status = "approved"
    c.rank_score = c.score * bias_multiplier(settings.bias, c.moment_type, streamer)


def _head_to_head(candidates: list[Candidate], settings: GtaSettings) -> None:
    """Rank approved clips, drop repeats of the same moment, cap the count."""
    approved = sorted(
        (c for c in candidates if c.status == "approved"),
        key=lambda c: (-c.rank_score, -c.hype_score, c.start_s),
    )
    kept: list[Candidate] = []
    for c in approved:
        twin = next((k for k in kept if _overlap(k, c) > SAME_MOMENT_OVERLAP_S), None)
        if twin is not None:
            c.reject(f"Same moment as a stronger clip ({twin.hook_title})")
            continue
        if len(kept) >= settings.max_clips_per_stream:
            c.reject("Lost head-to-head (stream's clip limit reached)")
            continue
        kept.append(c)


def _overlap(a: Candidate, b: Candidate) -> float:
    return max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))


def _clean_title(title: str) -> str:
    title = re.sub(r"#\w+", "", title).strip().strip("\"'“”")
    title = re.sub(r"\s+", " ", title)
    words = title.split()
    return " ".join(words[:9])


def clean_hashtags(tags: list[str]) -> list[str]:
    out: list[str] = []
    for tag in tags:
        body = re.sub(r"[^\w]", "", tag.lstrip("#").lower())
        if body and f"#{body}" not in out:
            out.append(f"#{body}")
    return out[:8]
