"""The judge — a second, separate LLM pass over the detected clips.

The call that picks clips also scores them, and it rates its own picks
generously. A second call that did not choose them asks three yes/no
questions of every candidate, as the exact clip a stranger would see (the
template is GTA Mode's Phase 2, and CrestClip's blind judge):

1. scroll_stop — does the opening ~2 seconds stop the scroll?
2. context_free — does it make sense with zero outside context?
3. payoff — does something land inside the clip?

One "no" drops the candidate. The judge only ever *removes* clips, and it
fails open: a provider error, malformed JSON twice, or a missing verdict
keeps the candidates it concerned. ``min_keep`` stops it from emptying a job
— the best-scoring rejects are kept, marked as such, to reach it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from ..db.models import Clip
from ..providers.base import DetectionConfig, ProviderError, extract_json_object, load_prompt
from .transcript import Transcript

log = logging.getLogger(__name__)

PROMPT_VERSION = "highlight_judge_v1"

#: Candidates per judge call (each carries its full transcript).
BATCH_SIZE = 6

#: Seconds of opening text shown as "first 2 seconds".
OPENING_S = 2.0

TESTS: tuple[tuple[str, str], ...] = (
    ("scroll_stop", "weak hook in the first 2 s"),
    ("context_free", "needs outside context"),
    ("payoff", "no payoff inside the clip"),
)

Complete = Callable[[str, str, DetectionConfig], Awaitable[str]]


class _Answer(BaseModel):
    answer: str = "yes"
    why: str = ""

    @field_validator("answer", mode="before")
    @classmethod
    def _yes_no(cls, value: Any) -> str:
        text = str(value if value is not None else "").strip().lower()
        if text in {"no", "n", "false", "0"}:
            return "no"
        return "yes" if text in {"yes", "y", "true", "1"} else "unknown"

    @field_validator("why", mode="before")
    @classmethod
    def _why(cls, value: Any) -> str:
        return "" if value is None else str(value)


class _Verdict(BaseModel):
    id: str
    scroll_stop: _Answer = _Answer(answer="unknown")
    context_free: _Answer = _Answer(answer="unknown")
    payoff: _Answer = _Answer(answer="unknown")

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("scroll_stop", "context_free", "payoff", mode="before")
    @classmethod
    def _answer(cls, value: Any) -> Any:
        # Accept a bare "yes"/"no" (or a bool) as well as {"answer", "why"}.
        if isinstance(value, str | bool):
            return {"answer": str(value)}
        return value

    def failed(self) -> list[str]:
        """Labels of the tests answered with a clear "no"."""
        return [label for name, label in TESTS if getattr(self, name).answer == "no"]


class _Out(BaseModel):
    verdicts: list[_Verdict] = []


@dataclass
class JudgeEntry:
    clip_id: str
    title: str
    start_s: float
    end_s: float
    score: int
    #: "passed" | "rejected" | "rescued" | "unjudged"
    outcome: str
    failed: list[str] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)
    #: Why a candidate passed without a verdict ("judge failed", "no verdict").
    note: str = ""


def opening_text(transcript: Transcript, clip: Clip, seconds: float = OPENING_S) -> str:
    words = transcript.slice(clip.start_word, clip.end_word)
    limit = clip.start_s + seconds
    opening = [w.text for w in words if w.start < limit]
    return " ".join(opening or [w.text for w in words[:4]])


def clip_text(transcript: Transcript, clip: Clip) -> str:
    return transcript.text_between(clip.start_word, clip.end_word)


def render_batch(transcript: Transcript, batch: list[tuple[str, Clip]], title: str = "") -> str:
    blocks = [
        f"### id: {key}\n"
        f"length: {clip.duration_s:.0f} s\n"
        f"first 2 seconds: {opening_text(transcript, clip)}\n"
        f"FULL CLIP TRANSCRIPT: {clip_text(transcript, clip)}"
        for key, clip in batch
    ]
    head = f"VIDEO: {title}\n\n" if title else ""
    return (
        f"{head}CANDIDATES:\n\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with ONLY the JSON object. No prose, no markdown fences."
    )


async def _ask(complete: Complete, system: str, user: str, config: DetectionConfig) -> _Out:
    """One judge call with one repair retry on malformed JSON."""
    raw = await complete(system, user, config)
    try:
        return _Out.model_validate(_verdicts_payload(raw))
    except (ValidationError, ValueError) as first_error:
        repair = (
            f"{user}\n\nYour previous response did not match the required schema.\n"
            f"Validation error:\n{first_error}\n\n"
            "Respond again with ONLY the corrected JSON object. No prose, no markdown fences."
        )
        raw = await complete(system, repair, config)
        try:
            return _Out.model_validate(_verdicts_payload(raw))
        except (ValidationError, ValueError) as second_error:
            raise ProviderError(
                "The clip judge returned malformed data twice.",
                hint=f"Last validation error: {second_error}",
            ) from second_error


def _verdicts_payload(raw: str) -> dict[str, Any]:
    payload = extract_json_object(raw)
    # extract_json_object wraps a bare list as {"clips": [...]}.
    if "verdicts" not in payload and isinstance(payload.get("clips"), list):
        payload = {"verdicts": payload["clips"]}
    return payload


class ClipJudge:
    """Callable judge: ``await judge(clips, transcript)`` returns the survivors.

    After a run, :attr:`entries` holds one record per candidate and
    :attr:`errors` the batch failures (each of which kept its candidates).
    """

    def __init__(
        self,
        complete: Complete,
        *,
        max_candidates: int = 20,
        min_keep: int = 1,
        title: str = "",
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.complete = complete
        self.max_candidates = max(1, max_candidates)
        self.min_keep = max(0, min_keep)
        self.title = title
        self.batch_size = max(1, batch_size)
        self.entries: list[JudgeEntry] = []
        self.errors: list[str] = []

    def report(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "prompt": PROMPT_VERSION,
            "errors": self.errors,
            "entries": [asdict(e) for e in self.entries],
        }

    async def __call__(self, clips: list[Clip], transcript: Transcript) -> list[Clip]:
        self.entries = []
        self.errors = []
        ordered = sorted(clips, key=lambda c: c.score, reverse=True)
        pool = ordered[: self.max_candidates]
        for clip in ordered[len(pool) :]:
            # Below the pool: weaker than every judged candidate, and untested.
            self.entries.append(self._entry(clip, "unjudged"))
        if not pool:
            return []

        system = load_prompt(PROMPT_VERSION)
        config = DetectionConfig(prompt_version=PROMPT_VERSION, temperature=0.2)
        passed: list[Clip] = []
        rejected: list[tuple[Clip, _Verdict]] = []

        for offset in range(0, len(pool), self.batch_size):
            batch = [
                (f"c{offset + i + 1}", c)
                for i, c in enumerate(pool[offset : offset + self.batch_size])
            ]
            try:
                out = await _ask(
                    self.complete, system, render_batch(transcript, batch, self.title), config
                )
            except Exception as exc:
                # Fail open: the judge may only remove clips it actually judged.
                log.warning("Clip judge failed on a batch (%s); keeping its candidates.", exc)
                self.errors.append(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
                for _, clip in batch:
                    passed.append(clip)
                    self.entries.append(self._entry(clip, "passed", note="judge failed"))
                continue
            verdicts = {v.id: v for v in out.verdicts}
            for key, clip in batch:
                verdict = verdicts.get(key) or verdicts.get(clip.id)
                if verdict is None:
                    passed.append(clip)
                    self.entries.append(self._entry(clip, "passed", note="no verdict"))
                    continue
                failed = verdict.failed()
                if failed:
                    rejected.append((clip, verdict))
                    self.entries.append(self._entry(clip, "rejected", verdict=verdict))
                else:
                    passed.append(clip)
                    self.entries.append(self._entry(clip, "passed", verdict=verdict))

        if len(passed) < self.min_keep and rejected:
            # Fewest failed tests first, then the higher score.
            rejected.sort(key=lambda item: (len(item[1].failed()), -item[0].score))
            for clip, verdict in rejected[: self.min_keep - len(passed)]:
                clip.reason = (
                    f"{clip.reason} [judge: kept to reach the minimum; "
                    f"{'; '.join(verdict.failed())}]"
                ).strip()
                passed.append(clip)
                for entry in self.entries:
                    if entry.clip_id == clip.id:
                        entry.outcome = "rescued"

        for clip, verdict in rejected:
            if clip not in passed:
                log.info("Judge dropped %r: %s.", clip.title, "; ".join(verdict.failed()))
        log.info("Judge kept %d of %d candidate(s).", len(passed), len(pool))
        passed.sort(key=lambda c: c.score, reverse=True)
        return passed

    @staticmethod
    def _entry(
        clip: Clip, outcome: str, *, verdict: _Verdict | None = None, note: str = ""
    ) -> JudgeEntry:
        return JudgeEntry(
            clip_id=clip.id,
            title=clip.title,
            start_s=round(clip.start_s, 2),
            end_s=round(clip.end_s, 2),
            score=clip.score,
            outcome=outcome,
            failed=verdict.failed() if verdict else [],
            verdict=verdict.model_dump(exclude={"id"}) if verdict else {},
            note=note,
        )
