"""LLM gate for live candidate windows (skeleton — A6 owns bodies).

Contract:
- Input: a text rendering of the candidate window (words with indexes, in the
  style of highlights.render_window_text) plus cheap-signal context.
- Uses the STANDARD provider layer (providers.build_provider + detection_config)
  with prompt ``live_judge_v1`` (file owned by A6).
- Output: LiveJudgement (below). One retry with validation feedback, exactly
  like providers.base.LLMProvider.detect_highlights.
- Rate limits from settings live.rate_limit are enforced by the manager, NOT
  here — judge.py stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


async def judge_window(
    stream_label: str,
    words: list,  # list[Transcript.Word]
    signals: dict[str, float],
    *,
    provider=None,  # providers.LLMProvider
    temperature: float = 0.2,
) -> LiveJudgement:
    """Render the window, call the provider once (with one repair retry)."""
    raise NotImplementedError("A6: implement")
