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
from collections.abc import Callable

from ..db.models import Clip, new_id
from ..providers import ClipCandidate, DetectionConfig, LLMProvider, TranscriptWindow
from . import boundaries
from .prepare import Loudness, Silence
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
) -> list[Clip]:
    """Run detection across the whole transcript and return ranked clips.

    ``cancelled`` is polled while windows are in flight; once it returns True
    the outstanding provider calls are cancelled and DetectionCancelled is
    raised.
    """
    windows = build_windows(transcript)
    if not windows:
        raise HighlightError("The transcript is empty, so there is nothing to clip.")

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
    return build_clips(
        transcript, candidates, config, job_id=job_id, silences=silences or [], loudness=loudness
    )


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
) -> list[Clip]:
    """Deduplicate, refine, rank, and truncate raw candidates into clips."""
    deduped = dedupe(candidates)
    log.info("%d candidates remain after dedupe.", len(deduped))

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
            )
        )

    # Refinement can move edges enough that two survivors now overlap.
    clips = _dedupe_clips(clips)
    clips = resolve_overlaps(clips, transcript, config)
    clips.sort(key=lambda c: c.score, reverse=True)
    clips = clips[: config.max_clips]

    for rank, clip in enumerate(clips, start=1):
        clip.rank = rank

    log.info("Produced %d final clips.", len(clips))
    return clips


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
    clips: list[Clip], transcript: Transcript, config: DetectionConfig
) -> list[Clip]:
    """Remove the partial overlaps that the IoU dedupe lets through.

    On a real IShowSpeed stream two pairs survived dedupe while sharing 3.4 s
    and 13.9 s of audio, so a viewer of both watched the same bit twice. Two
    overlapping clips are one moment: they merge when the union still fits
    ``max_duration_s`` (the better-scoring clip keeps its title). Otherwise the
    weaker clip gives up the shared stretch at a sentence boundary, and is
    dropped if what remains is shorter than ``min_duration_s``.
    """
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
        trimmed = _trim_off(clip, other, transcript, config.min_duration_s)
        if trimmed is None:
            log.debug("Dropped clip %s: it overlaps %s and cannot be trimmed.", clip.id, other.id)
        else:
            remaining.insert(0, trimmed)
    return kept


def _overlap_s(a: Clip, b: Clip) -> float:
    return min(a.end_s, b.end_s) - max(a.start_s, b.start_s)


def _trim_off(
    clip: Clip, keeper: Clip, transcript: Transcript, min_duration_s: float
) -> Clip | None:
    """Shorten ``clip`` on a sentence edge so it no longer overlaps ``keeper``."""
    words = transcript.words
    if clip.start_s >= keeper.start_s:
        # The clip runs on past the keeper's end: start it after the keeper.
        first = next(
            (
                i
                for i in range(keeper.end_word + 1, clip.end_word)
                if words[i - 1].ends_sentence and words[i].start >= keeper.end_s
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
                if words[i].ends_sentence and words[i].end <= keeper.start_s
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
