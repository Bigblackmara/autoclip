"""Highlight detection — transcript in, ranked clips out.

The transcript is cut into overlapping windows so that long videos fit inside
small context windows, each window is scored by the configured provider, and the
resulting candidates are merged, deduplicated, boundary-refined, and ranked.

Overlap matters: a clip straddling a window edge would otherwise be seen only in
halves by both windows and proposed by neither. The overlap costs tokens and
buys back the clips that live on the seams.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..db.models import Clip, new_id
from ..providers import ClipCandidate, DetectionConfig, LLMProvider, TranscriptWindow
from . import boundaries
from .prepare import Loudness, Silence
from .signals import AudienceCurve, clock
from .transcript import Transcript

log = logging.getLogger(__name__)

#: Window length and overlap in seconds of speech.
WINDOW_S = 8 * 60
OVERLAP_S = 60

#: Two candidates covering this much of the same words are the same clip.
DEDUPE_IOU = 0.4

#: Clips sharing no more than this much time merely touch: silence padding on
#: both sides of a boundary can legitimately cross.
OVERLAP_TOLERANCE_S = 0.5

#: Hosted providers tolerate parallel windows; a local model is already
#: saturating the GPU, so extra concurrency only adds contention.
HOSTED_CONCURRENCY = 3
LOCAL_CONCURRENCY = 1


#: How often an in-flight detection checks the job's cancel flag, in seconds.
CANCEL_POLL_S = 0.25


class HighlightError(RuntimeError):
    """Highlight detection produced nothing usable."""


class DetectionCancelled(RuntimeError):
    """The job was cancelled while windows were still being scored."""


#: A judge pass: candidates (best first) in, survivors out. See
#: :class:`autoclip.pipeline.judge.ClipJudge`.
Judge = Callable[[list[Clip], Transcript], Awaitable[list[Clip]]]


@dataclass
class AudienceOptions:
    """How the audience curve steers detection (see ``config.DetectionSettings``)."""

    curve: AudienceCurve
    #: Final-score share of the curve when it holds viewer data.
    audience_weight: float = 0.4
    #: Final-score share when the curve is audio energy alone.
    energy_weight: float = 0.12
    #: Hotspots listed per window in the prompt; 0 disables hints.
    hints_per_window: int = 4

    @property
    def weight(self) -> float:
        return self.audience_weight if self.curve.kind == "audience" else self.energy_weight


def build_windows(
    transcript: Transcript,
    *,
    window_s: float = WINDOW_S,
    overlap_s: float = OVERLAP_S,
) -> list[TranscriptWindow]:
    """Split a transcript into overlapping windows.

    Preconditions:
        overlap_s is less than window_s, otherwise windows would never advance.
    """
    if not transcript.words:
        return []
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be smaller than window_s")

    windows: list[TranscriptWindow] = []
    total_words = len(transcript.words)
    cursor = 0

    while cursor < total_words:
        window_start_time = transcript.words[cursor].start
        window_end_time = window_start_time + window_s

        last = cursor
        while last + 1 < total_words and transcript.words[last + 1].end <= window_end_time:
            last += 1

        windows.append(_make_window(transcript, cursor, last))

        if last >= total_words - 1:
            break

        # Step forward by (window - overlap), measured in time then converted
        # back to a word index so overlap stays constant regardless of pace.
        next_time = transcript.words[last].end - overlap_s
        next_cursor = transcript.index_at_time(next_time)
        cursor = max(cursor + 1, next_cursor)

    return windows


def _make_window(transcript: Transcript, first: int, last: int) -> TranscriptWindow:
    start_s, end_s = transcript.time_range(first, last)
    words = transcript.slice(first, last)
    speakers = sorted({w.speaker for w in words if w.speaker})
    return TranscriptWindow(
        text=render_window_text(transcript, first, last),
        first_word=first,
        last_word=last,
        duration_s=end_s - start_s,
        speakers=speakers,
    )


def render_window_text(transcript: Transcript, first: int, last: int) -> str:
    """Render words as ``[index]word`` so the model can cite exact positions.

    Tagging every word is verbose, but it's the reason the model can't
    miscount — it never has to derive an index, only copy one.
    """
    parts: list[str] = []
    current_speaker: str | None = None

    for index in range(first, min(last + 1, len(transcript.words))):
        word = transcript.words[index]
        if word.speaker and word.speaker != current_speaker:
            current_speaker = word.speaker
            parts.append(f"\n<{word.speaker}>")
        parts.append(f"[{index}]{word.text}")

    return " ".join(parts).strip()


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


async def detect(
    transcript: Transcript,
    provider: LLMProvider,
    config: DetectionConfig,
    *,
    job_id: str,
    silences: list[Silence] | None = None,
    loudness: Loudness | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    audience: AudienceOptions | None = None,
    judge: Judge | None = None,
) -> list[Clip]:
    """Run detection across the whole transcript and return ranked clips.

    ``cancelled`` is polled while windows are in flight; once it returns True
    the outstanding provider calls are cancelled and DetectionCancelled is
    raised.

    ``audience`` adds each window's hotspots to its prompt and blends the
    curve into every clip's score. ``judge`` runs on the refined clips
    before the count is capped, so a dropped clip makes room for the next.
    """
    windows = build_windows(transcript)
    if not windows:
        raise HighlightError("The transcript is empty, so there is nothing to clip.")
    if audience is not None and audience.curve.has_data and audience.hints_per_window > 0:
        for window in windows:
            window.hints = audience_hints(
                audience.curve, transcript, window, limit=audience.hints_per_window
            )

    log.info("Detecting highlights across %d window(s) with %s.", len(windows), provider.name)

    concurrency = LOCAL_CONCURRENCY if provider.name == "ollama" else HOSTED_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()
    errors: list[Exception] = []

    async def run_window(window: TranscriptWindow) -> list[ClipCandidate]:
        nonlocal completed
        async with semaphore:
            try:
                result = await provider.detect_highlights(window, config)
                candidates = result.clips
            except Exception as exc:
                # One bad window shouldn't lose the whole video's other windows.
                log.warning(
                    "Window %d-%d failed (%s); continuing with the remaining windows.",
                    window.first_word,
                    window.last_word,
                    exc,
                )
                errors.append(exc)
                candidates = []
            async with lock:
                completed += 1
                if on_progress:
                    on_progress(completed / len(windows))
            return candidates

    gathered = asyncio.gather(*(run_window(w) for w in windows))
    if cancelled is None:
        results = await gathered
    else:
        results = await _unless_cancelled(gathered, cancelled)
    candidates = [c for group in results for c in group]

    if not candidates and len(errors) == len(windows):
        # Every window failed: that is a provider problem (bad key, unknown
        # model, unreachable server), not a video without highlights. Saying
        # "no clips were found" here sent users tuning prompts and models when
        # the fix was in settings.
        raise errors[0]

    if not candidates:
        raise HighlightError(
            "No clips were found. This can mean the video genuinely has no "
            "self-contained highlights, or that the model struggled with the "
            "transcript — try a larger model or a different provider."
        )

    log.info("Providers proposed %d raw candidates.", len(candidates))
    clips = build_clips(
        transcript,
        candidates,
        config,
        job_id=job_id,
        silences=silences or [],
        loudness=loudness,
        audience=audience,
        truncate=judge is None,
    )
    if judge is None:
        return clips

    judged = asyncio.ensure_future(judge(clips, transcript))
    clips = await (judged if cancelled is None else _unless_cancelled(judged, cancelled))
    return rank_clips(clips, config.max_clips)


async def _unless_cancelled(
    gathered: asyncio.Future[list[list[ClipCandidate]]], cancelled: Callable[[], bool]
) -> list[list[ClipCandidate]]:
    """Await ``gathered``, cancelling it as soon as ``cancelled()`` is True.

    The runner only checked its cancel flag between stages, so a cancel
    during detection waited out every window — minutes per window on a local
    model, up to the ten-minute request timeouts on a stalled hosted one —
    before the job stopped.
    """
    while True:
        done, _ = await asyncio.wait({gathered}, timeout=CANCEL_POLL_S)
        if done:
            return gathered.result()
        if cancelled():
            gathered.cancel()
            # Let every window task unwind (closing its HTTP connection)
            # before reporting, so nothing is left running on this loop.
            with contextlib.suppress(asyncio.CancelledError):
                await gathered
            raise DetectionCancelled("Highlight detection was cancelled.")


def build_clips(
    transcript: Transcript,
    candidates: list[ClipCandidate],
    config: DetectionConfig,
    *,
    job_id: str,
    silences: list[Silence] | None = None,
    loudness: Loudness | None = None,
    audience: AudienceOptions | None = None,
    truncate: bool = True,
) -> list[Clip]:
    """Deduplicate, refine, rank, and truncate raw candidates into clips.

    With ``audience``, each clip's score becomes the AI score blended with
    the curve over the clip's final span (:func:`blend_score`) before the
    overlap passes and the ranking use it. ``truncate=False`` keeps every
    survivor, ranked, for a judge pass to cut down.
    """
    deduped = dedupe(candidates)
    log.info("%d candidates remain after dedupe.", len(deduped))
    breaks = boundaries.SentenceBreaks(transcript)

    clips: list[Clip] = []
    for candidate in deduped:
        boundary = boundaries.refine(
            transcript,
            candidate.start_word_index,
            candidate.end_word_index,
            silences=silences or [],
            min_duration_s=config.min_duration_s,
            max_duration_s=config.max_duration_s,
            hook=candidate.hook,
            loudness=loudness,
            breaks=breaks,
        )
        if boundary is None:
            log.debug(
                "Dropped candidate %d-%d: no valid boundary within the duration range.",
                candidate.start_word_index,
                candidate.end_word_index,
            )
            continue

        clips.append(
            Clip(
                id=new_id(),
                job_id=job_id,
                start_s=boundary.start_s,
                end_s=boundary.end_s,
                start_word=boundary.start_word,
                end_word=boundary.end_word,
                title=candidate.title.strip()
                or transcript.text_between(
                    boundary.start_word, min(boundary.start_word + 8, boundary.end_word)
                ),
                hook=candidate.hook.strip(),
                score=candidate.score,
                reason=candidate.reason.strip(),
                hook_options=list(candidate.hooks),
                hook_text=candidate.hooks[0] if candidate.hooks else "",
                description=candidate.description.strip(),
                hashtags=list(candidate.hashtags),
            )
        )
        clips[-1].teaser_start_s, clips[-1].teaser_end_s = teaser_times(
            transcript, candidate, boundary.start_s, boundary.end_s
        )
        if audience is not None:
            blend_score(clips[-1], audience)

    # Refinement can move edges enough that two survivors now overlap.
    clips = _dedupe_clips(clips)
    clips = resolve_overlaps(clips, transcript, config, breaks=breaks)
    if not truncate:
        return rank_clips(clips, None)
    clips = rank_clips(clips, config.max_clips)
    log.info("Produced %d final clips.", len(clips))
    return clips


def teaser_times(
    transcript: Transcript, candidate: ClipCandidate, start_s: float, end_s: float
) -> tuple[float | None, float | None]:
    """The model's flash-forward line in source seconds, if it lies inside the clip.

    The line is checked against the clip's refined edges; the final fit (a
    pause to end on, 0.8-3.6 s long) happens when the edit is planned, since
    a later trim can move the edges again.
    """
    first, last = candidate.teaser_start_word_index, candidate.teaser_end_word_index
    words = transcript.words
    if first is None or last is None or not (0 <= first <= last < len(words)):
        return None, None
    a, b = words[first].start, words[last].end
    if a < start_s - 0.05 or b > end_s + 0.05 or b <= a:
        return None, None
    return round(a, 3), round(b, 3)


def rank_clips(clips: list[Clip], max_clips: int | None) -> list[Clip]:
    """Sort by score, keep the best ``max_clips`` (all when None), number them."""
    ordered = sorted(clips, key=lambda c: c.score, reverse=True)
    if max_clips is not None:
        ordered = ordered[:max_clips]
    for rank, clip in enumerate(ordered, start=1):
        clip.rank = rank
    return ordered


# --------------------------------------------------------------------------
# Audience signals
# --------------------------------------------------------------------------


def blend_score(clip: Clip, audience: AudienceOptions) -> None:
    """Blend the audience curve into ``clip.score``; note the parts in its reason.

    ``score = (1 - w) * AI + w * audience``, where ``audience`` is how hot the
    clip's span is against the rest of the video (0..100) and ``w`` is the
    audience weight (or the smaller energy weight when the curve holds only
    loudness). Without curve data the clip keeps its AI score. The Clip row
    has no score-component columns, so the parts go into ``reason``.
    """
    curve = audience.curve
    value = curve.range_score(clip.start_s, clip.end_s) if curve.has_data else None
    weight = audience.weight
    if value is None or weight <= 0:
        return
    ai = clip.score
    clip.score = max(0, min(100, round((1.0 - weight) * ai + weight * value)))
    note = f"[score {clip.score} = AI {ai} x{1 - weight:.2f} + {curve.label} {value} x{weight:.2f}]"
    clip.reason = f"{clip.reason} {note}".strip()


def audience_hints(
    curve: AudienceCurve, transcript: Transcript, window: TranscriptWindow, *, limit: int
) -> list[str]:
    """Prompt lines for the hotspots and chapters inside one window."""
    if not transcript.words:
        return []
    start_s, end_s = transcript.time_range(window.first_word, window.last_word)
    lines: list[str] = []
    strength = "weak signal" if curve.kind == "energy" else "viewer signal"
    for spot in curve.hotspots_between(start_s, end_s)[:limit]:
        first = max(window.first_word, transcript.index_at_time(spot.start_s))
        last = max(first, min(window.last_word, transcript.index_at_time(spot.end_s)))
        lines.append(
            f"- {clock(spot.start_s)}-{clock(spot.end_s)}, around words [{first}]-[{last}] "
            f"({strength}, strength {round(spot.value * 100)}): {'; '.join(spot.reasons)}"
        )
    chapters = curve.chapters_between(start_s, end_s)
    if chapters:
        listed = ", ".join(
            f'{clock(c.start_s)} "{c.title}" (from word [{transcript.index_at_time(c.start_s)}])'
            for c in chapters[:6]
        )
        lines.append(f"- Chapters starting here: {listed}")
    return lines


def dedupe(
    candidates: list[ClipCandidate], *, iou_threshold: float = DEDUPE_IOU
) -> list[ClipCandidate]:
    """Greedily keep the highest-scoring candidate from each overlapping cluster.

    Overlapping windows mean the same moment is often proposed two or three
    times, sometimes with slightly different edges. Highest score wins.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[ClipCandidate] = []

    for candidate in ordered:
        if any(
            _iou(
                candidate.start_word_index,
                candidate.end_word_index,
                existing.start_word_index,
                existing.end_word_index,
            )
            > iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)

    return kept


def _dedupe_clips(clips: list[Clip], *, iou_threshold: float = DEDUPE_IOU) -> list[Clip]:
    ordered = sorted(clips, key=lambda c: c.score, reverse=True)
    kept: list[Clip] = []
    for clip in ordered:
        if any(
            _iou(clip.start_word, clip.end_word, other.start_word, other.end_word) > iou_threshold
            for other in kept
        ):
            continue
        kept.append(clip)
    return kept


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection over union of two inclusive index ranges."""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    if intersection == 0:
        return 0.0
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - intersection
    return intersection / union if union else 0.0


def resolve_overlaps(
    clips: list[Clip],
    transcript: Transcript,
    config: DetectionConfig,
    *,
    breaks: boundaries.SentenceBreaks | None = None,
) -> list[Clip]:
    """Remove the partial overlaps that the IoU dedupe lets through.

    On a real IShowSpeed stream two pairs survived dedupe while sharing 3.4 s
    and 13.9 s of audio, so a viewer of both watched the same bit twice. Two
    overlapping clips are one moment: they merge when the union still fits
    ``max_duration_s`` (the better-scoring clip keeps its title). Otherwise the
    weaker clip gives up the shared stretch at a sentence boundary, and is
    dropped if what remains is shorter than ``min_duration_s``. Sentence
    boundaries include soft pause breaks (:class:`boundaries.SentenceBreaks`).
    """
    if breaks is None:
        breaks = boundaries.SentenceBreaks(transcript)
    remaining = sorted(clips, key=lambda c: c.score, reverse=True)
    kept: list[Clip] = []
    while remaining:
        clip = remaining.pop(0)
        other = next((k for k in kept if _overlap_s(clip, k) > OVERLAP_TOLERANCE_S), None)
        if other is None:
            kept.append(clip)
            continue
        union_start = min(clip.start_s, other.start_s)
        union_end = max(clip.end_s, other.end_s)
        if union_end - union_start <= config.max_duration_s:
            # ``other`` scored at least as high, so it absorbs ``clip``. It is
            # re-queued so the grown clip is checked against the rest of kept.
            kept.remove(other)
            other.start_s, other.end_s = union_start, union_end
            other.start_word = min(clip.start_word, other.start_word)
            other.end_word = max(clip.end_word, other.end_word)
            remaining.insert(0, other)
            continue
        trimmed = _trim_off(clip, other, transcript, config.min_duration_s, breaks)
        if trimmed is None:
            log.debug("Dropped clip %s: it overlaps %s and cannot be trimmed.", clip.id, other.id)
        else:
            remaining.insert(0, trimmed)
    return kept


def _overlap_s(a: Clip, b: Clip) -> float:
    return min(a.end_s, b.end_s) - max(a.start_s, b.start_s)


def _trim_off(
    clip: Clip,
    keeper: Clip,
    transcript: Transcript,
    min_duration_s: float,
    breaks: boundaries.SentenceBreaks,
) -> Clip | None:
    """Shorten ``clip`` on a sentence edge so it no longer overlaps ``keeper``."""
    words = transcript.words
    if clip.start_s >= keeper.start_s:
        # The clip runs on past the keeper's end: start it after the keeper.
        first = next(
            (
                i
                for i in range(keeper.end_word + 1, clip.end_word)
                if breaks.starts_at(i) and words[i].start >= keeper.end_s
            ),
            None,
        )
        if first is None:
            return None
        clip.start_word = first
        clip.start_s = max(keeper.end_s, words[first].start - boundaries.FALLBACK_LEAD_S)
    else:
        # The clip leads into the keeper: end it before the keeper starts.
        last = next(
            (
                i
                for i in range(keeper.start_word - 1, clip.start_word, -1)
                if breaks.ends_at(i) and words[i].end <= keeper.start_s
            ),
            None,
        )
        if last is None:
            return None
        clip.end_word = last
        clip.end_s = min(keeper.start_s, words[last].end + boundaries.FALLBACK_TAIL_S)
    if clip.end_s - clip.start_s < min_duration_s:
        return None
    return clip
