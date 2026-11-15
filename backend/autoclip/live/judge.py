"""LLM gate for live candidate windows.

Contract:
- Input: a text rendering of the candidate window (words with indexes, in the
  style of highlights.render_window_text) plus cheap-signal context.
- Uses the STANDARD provider layer (providers.build_provider + detection_config)
  with prompt ``live_judge_v1`` (file owned by A6).
- Output: LiveJudgement (below). One retry with validation feedback, exactly
  like providers.base.LLMProvider.detect_highlights.
- Rate limits from settings live.rate_limit are enforced by the manager, NOT
  here — judge.py stays pure.

Why this gate exists: the cheap detectors measure arousal — how loud the room
got, how fast chat is moving — not meaning. A jump-scare scream is a 2.0-strength
audio spike with nothing worth publishing inside it, and a hype train erupts
over visuals a transcript will never show. The judge is the only step in the
chain that reads the content, so it is the difference between a highlight reel
and a folder of noise. It is also the most expensive step, which is why it runs
last, on windows the cheap triggers already believe in — and why a window with
no transcribed speech is rejected locally without spending a call on it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..pipeline.highlights import render_window_text
from ..pipeline.transcript import Transcript
from ..providers import ProviderError, build_provider, detection_config
from ..providers.base import extract_json_object, load_prompt
from .signals import SignalKind

log = logging.getLogger(__name__)

#: Prompt file stem in ``autoclip/prompts/`` — versioned like highlight_v1 so
#: the gate prompt can be iterated on without touching code.
PROMPT_VERSION = "live_judge_v1"

#: What each cheap signal kind measures, shown to the model beside the strength
#: so it knows what the number means — and, crucially, what it cannot mean.
SIGNAL_DETAILS: dict[str, str] = {
    str(SignalKind.AUDIO_ENERGY): (
        "stream audio rose above its own rolling baseline (RMS energy; strength "
        "is the z-score as a multiple of the spike threshold)"
    ),
    str(SignalKind.CHAT_RATE): (
        "chat message velocity rose above its own rolling baseline (strength is "
        "the z-score as a multiple of the rate threshold)"
    ),
    str(SignalKind.CHAT_HYPE): (
        "hype-train-level chat velocity — hundreds of messages per minute; always strength 2.0"
    ),
}

#: Short scale legend embedded in every user prompt, next to the signal values
#: it explains. The full scale definition lives in the system prompt; the user
#: prompt only needs the numbers to be readable.
_STRENGTH_SCALE_NOTE = (
    "How to read strength: each detector is normalized to its own 0.0-2.0 scale. "
    "1.0 is a clear spike on that detector, 2.0 is extreme for it (for example a "
    "hype train), and anything below 0.4 is treated as noise. Several detectors "
    "corroborating each other is what earned this window a judgement."
)


@dataclass
class LiveJudgement:
    """The model's verdict on one candidate window."""

    score: int  # 0-100 virality score
    title: str = ""
    hook: str = ""
    reason: str = ""
    #: Trimming advice in words relative to the window's first word — used to
    #: tighten a window before render, optional (may be None).
    start_word_offset: int | None = None
    end_word_offset: int | None = None
    #: Raw peak strengths of the cheap signals that triggered this window.
    signals: dict[str, float] = field(default_factory=dict)


class LiveJudgeOut(BaseModel):
    """One judge verdict as returned by the model, before window-clamping.

    Coercion mirrors :class:`~autoclip.providers.base.ClipCandidate`: models emit
    nulls, floats, and 0-1 fractions no matter what the schema says, and a
    tolerable answer should not cost a repair retry.
    """

    score: int = Field(default=50, ge=0, le=100)
    title: str = ""
    hook: str = ""
    reason: str = ""
    start_word_offset: int | None = None
    end_word_offset: int | None = None

    @field_validator("title", "hook", "reason", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return "" if value is None else str(value)

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, value: Any) -> int:
        """Accept floats and 0-1 fractions, which models emit despite the schema.

        Only a NON-INTEGER value strictly between 0 and 1 reads as a fraction
        (0.87 → 87). An integer ``1`` (or ``1.0``) is a score of 1, not 100:
        scaling it would turn a flat rejection into a perfect pass.
        """
        if value is None or isinstance(value, bool):
            return 50
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 50
        if not math.isfinite(number):
            return 50
        if 0.0 < number < 1.0 and not number.is_integer():
            number *= 100
        return max(0, min(100, round(number)))

    @field_validator("start_word_offset", "end_word_offset", mode="before")
    @classmethod
    def _coerce_offset(cls, value: Any) -> int | None:
        """Nulls and junk mean "no advice" — coercing them to 0 would be a lie."""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return max(0, round(number))


def _signal_entry(kind: str, value: Any) -> tuple[float, str]:
    """Extract ``(strength, detail)`` from one signals-mapping value.

    The frozen signature takes plain peak strengths, but the manager may hand
    over :class:`~autoclip.live.signals.DetectorSignal` objects straight from
    the aggregator; their ``detail`` text says what actually happened, which
    makes a far better prompt than a bare number.
    """
    strength = getattr(value, "strength", value)
    try:
        strength_f = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid strength for signal '{kind}': {value!r}") from exc
    if not math.isfinite(strength_f):
        raise ValueError(f"Invalid strength for signal '{kind}': {value!r}")

    detail = str(getattr(value, "detail", "") or "")
    if not detail:
        detail = SIGNAL_DETAILS.get(kind, f"cheap detector '{kind}' fired")
    return strength_f, detail


def _collect_signals(signals: dict | None) -> dict[str, tuple[float, str]]:
    """Normalize the signals mapping to ``kind -> (strength, detail)``.

    Keys are stringified so :class:`SignalKind` enum members end up as their
    plain ``"audio_energy"``-style values on :attr:`LiveJudgement.signals`,
    which is what the store layer wants to serialize.
    """
    collected: dict[str, tuple[float, str]] = {}
    for kind, value in (signals or {}).items():
        key = str(kind)
        collected[key] = _signal_entry(key, value)
    return collected


def render_judge_prompt(
    stream_label: str,
    words: list,
    signals: dict[str, float] | None = None,
) -> str:
    """Build the user message for one candidate window.

    The transcript rendering is deliberately the same ``[index]word`` format
    as the highlights pipeline (reused from it, not reimplemented): the model
    never has to derive an index, only copy one, which is what keeps cheap
    local models usable here too.
    """
    transcript = Transcript(words=list(words))
    body = render_window_text(transcript, 0, max(0, len(words) - 1))

    label = " ".join(stream_label.split()) or "(unlabeled stream)"
    collected = _collect_signals(signals)
    if collected:
        # Strongest evidence first: the model reads the block top-down and the
        # tail is where small-context models start skimming.
        ordered = sorted(collected.items(), key=lambda item: (-item[1][0], item[0]))
        signals_block = "\n".join(
            f"- {kind}: {strength:.2f} — {detail}" for kind, (strength, detail) in ordered
        )
    else:
        signals_block = "(none — this window was submitted without cheap-signal evidence)"

    return (
        f'Below is a candidate window from live stream "{label}". The cheap '
        "detectors fired on this moment; judge whether the CONTENT justifies "
        "clipping it.\n\n"
        "Every word is tagged with its offset index as [offset]word, relative to "
        "this window's first word: [0] is the first word of the window. Give any "
        "trim advice using these same offsets.\n\n"
        "SIGNALS — peak cheap-detector strengths for this window:\n"
        f"{signals_block}\n\n"
        f"{_STRENGTH_SCALE_NOTE}\n\n"
        "Transcript of the window:\n"
        f"---\n{body}\n---\n\n"
        "Respond with ONLY the JSON object described in the system message. No "
        "prose, no markdown fences."
    )


def _parse(raw: str) -> LiveJudgeOut:
    """Validate one raw model response into a :class:`LiveJudgeOut`."""
    payload = extract_json_object(raw)
    if "score" not in payload:
        # Every other field defaults harmlessly; the score is the whole point of
        # the gate, and inventing one would hide a model that ignored the task.
        raise ValueError("The response must contain a 'score' field.")
    return LiveJudgeOut.model_validate(payload)


def _clamp_offset(value: int | None, last_index: int) -> int | None:
    """Clamp a word offset into ``[0, last_index]``; ``None`` passes through."""
    if value is None:
        return None
    return max(0, min(int(value), last_index))


class JudgeBudgetExhausted(RuntimeError):
    """The caller's judge budget refused a call; no LLM request was made."""


def _spend(before_call: Callable[[], bool] | None, what: str) -> None:
    """Ask the caller's budget for one paid call; raise when it says no."""
    if before_call is not None and not before_call():
        raise JudgeBudgetExhausted(f"judge budget exhausted before the {what}")


async def judge_window(
    stream_label: str,
    words: list,  # list[Transcript.Word]
    signals: dict[str, float],
    *,
    provider=None,  # providers.LLMProvider
    temperature: float = 0.2,
    before_call: Callable[[], bool] | None = None,
) -> LiveJudgement:
    """Render the window, call the provider once (with one repair retry).

    ``provider`` is injectable so tests (and the manager) can pass a fake or a
    pre-built instance; when omitted, the ACTIVE provider is built via
    :func:`autoclip.providers.build_provider` — its model and API key come from
    settings/keyring and are never handled or logged here.

    ``before_call`` is consulted before EVERY provider request — the repair
    retry is a second paid call and must fit the budget like the first.
    It returns False to refuse (:class:`JudgeBudgetExhausted` is raised,
    nothing is sent); returning True records the call.

    Word offsets in the verdict are relative to the window's first word
    (``[0]``) and clamped to the window; ``None`` on an offset means "no trim
    advice for that edge". The input signals are echoed on the judgement as
    peak strengths so the store can record what evidence fired.
    """
    collected = _collect_signals(signals)
    strengths = {kind: strength for kind, (strength, _) in collected.items()}

    if not words:
        # Nothing was transcribed: silence, music, or dead air — live ASR fails
        # this way constantly. No model can rescue an empty window, and burning
        # a rate-limit token on one would be waste. Score 0 and let the manager
        # mark the row rejected like any other sub-threshold verdict.
        return LiveJudgement(
            score=0,
            reason="No speech was transcribed in this window.",
            signals=strengths,
        )

    if provider is None:
        provider = build_provider()

    system = load_prompt(PROMPT_VERSION)
    user = render_judge_prompt(stream_label, words, signals)

    config = detection_config()
    config.prompt_version = PROMPT_VERSION
    config.temperature = temperature

    name = provider.name or type(provider).__name__
    _spend(before_call, "judge call")
    raw = await provider._complete(system, user, config)
    try:
        out = _parse(raw)
    except (ValidationError, ValueError) as first_error:
        log.warning("%s returned an invalid live judgement; retrying with feedback.", name)
        repair = (
            f"{user}\n\n"
            "Your previous response did not match the required schema.\n"
            f"Validation error:\n{first_error}\n\n"
            "Respond again with ONLY the corrected JSON object. No prose, no "
            "markdown fences."
        )
        _spend(before_call, "repair retry")
        raw = await provider._complete(system, repair, config)
        try:
            out = _parse(raw)
        except (ValidationError, ValueError) as second_error:
            raise ProviderError(
                f"{name} returned malformed judgement data twice.",
                provider=name,
                hint=(
                    f"Last validation error: {second_error}\n\n"
                    "Smaller local models struggle with strict JSON. Try a larger "
                    "model, or switch to a hosted provider."
                ),
            ) from second_error

    last_index = len(words) - 1
    start = _clamp_offset(out.start_word_offset, last_index)
    end = _clamp_offset(out.end_word_offset, last_index)
    if start is not None and end is not None and end < start:
        # Reversed trim advice would tighten the window into an empty range —
        # worse than no advice at all — so drop the pair rather than pass it on.
        start = end = None

    log.info("Judge scored a %d-word window via %s at %d.", len(words), name, out.score)
    return LiveJudgement(
        score=out.score,
        title=out.title,
        hook=out.hook,
        reason=out.reason,
        start_word_offset=start,
        end_word_offset=end,
        signals=strengths,
    )
