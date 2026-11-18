"""GTA Mode clip edits after judging: hook choice, re-render, variants, picks.

* **Hook choice / re-render** — the judge writes three hook titles; the user
  picks one (or flips the clip's flash-forward) and the clip renders again
  through the normal render + QA path.
* **Variants** — "Make variant" on a ready clip adds a NEW clip row for the
  same moment with the next caption look, the next hook option and its own
  zoom pattern. It shares the original's downloaded section. The poster
  treats it as its own clip but never posts two of one family on the same
  platform on the same day.
* **Picks** — ranges the user drags on a stream's Hype Curve become clips:
  no reject phase, but the judge prompt still writes the hook titles,
  caption and hashtags when an AI provider is available (else a plain
  title from the clip's opening words).

The queue renders clips only while their stream is ``analyzed``; a finished
(``done``) stream is moved back there so it picks the new work up, and goes
back to ``done`` by itself when the clips are finished.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace

import numpy as np

from ..db.models import new_id
from ..pipeline.transcript import Transcript
from . import judging, store
from .hype import HypeCurve
from .settings import CAPTION_STYLES, VARIANT_LOOKS, GtaSettings
from .settings import load as load_settings
from .store import GtaClip, GtaStream

log = logging.getLogger(__name__)

#: Clips that can be rendered again (or varied): finished or failed ones.
RERENDERABLE = frozenset({"ready", "qa_failed", "failed"})
#: Streams whose clips the queue may be woken for.
WAKEABLE = frozenset({"done", "analyzed"})
#: A pick shorter than this can't pass QA (5 s minimum).
MIN_PICK_S = 5.0
#: Dead air is cut before the length cap, so a pick may run a bit past it.
PICK_SLACK_S = 30.0
MAX_PICKS = 12
#: Words of the clip's opening used as a plain title without an AI judge.
PLAIN_TITLE_WORDS = 6
#: Most points sent for the Hype Curve graph (longer curves are max-pooled).
GRAPH_POINTS = 2000


class EditError(ValueError):
    """A request the clip or stream can't take; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def _stream(stream_id: str) -> GtaStream:
    stream = store.get_stream(stream_id)
    if stream is None:
        raise EditError("Stream not found.", 404)
    return stream


def _clip(clip_id: str) -> GtaClip:
    clip = store.get_clip(clip_id)
    if clip is None:
        raise EditError("Clip not found.", 404)
    return clip


def _vod_only(stream: GtaStream) -> None:
    if stream.signals.get("mode") == "live":
        raise EditError(
            "GTA Live clips are cut from the live buffer and can't be rendered again here."
        )


def wake_stream(stream: GtaStream) -> None:
    """Let the queue render the stream's pending clips again."""
    if stream.status == "done":
        store.update_stream(stream.id, status="analyzed", detail="Rendering edited clips")
    from .queue import queue

    queue.notify()


def _not_uploading(clip_id: str) -> None:
    from .poster import poster

    if clip_id in poster.uploading_clip_ids():
        raise EditError("This clip is uploading right now; try again shortly.")


# --------------------------------------------------------------------------
# Hook choice + re-render
# --------------------------------------------------------------------------


def rerender(
    clip_id: str,
    *,
    hook_title: str | None = None,
    flash_forward: bool | None = None,
    flash_forward_set: bool = False,
) -> GtaClip:
    """Change the hook title and/or the flash-forward, then render again.

    ``flash_forward_set`` says the caller sent ``flash_forward`` (None then
    means "follow the GTA setting").
    """
    clip = _clip(clip_id)
    stream = _stream(clip.stream_id)
    _vod_only(stream)
    if stream.status not in WAKEABLE:
        raise EditError("Wait until the stream has finished judging.")
    if clip.status not in RERENDERABLE:
        raise EditError("Only finished clips can be rendered again.")
    _not_uploading(clip.id)
    fields: dict = {}
    if hook_title is not None:
        title = judging._clean_title(hook_title)  # noqa: SLF001 - same cleaning as the judge
        if not title:
            raise EditError("The hook title can't be empty.", 400)
        fields["hook_title"] = title
        # The old hook's caption-language copy is stale until the re-render.
        fields["localised"] = {k: v for k, v in clip.localised.items() if k != "hook"}
        if title not in clip.hook_options:
            # A hand-written hook joins the options, so a variant can reuse it.
            fields["hook_options"] = [*clip.hook_options, title][-6:]
    if flash_forward_set:
        fields["flash_forward"] = flash_forward
    store.update_clip(clip.id, status="rendering", error=None, **fields)
    wake_stream(stream)
    updated = store.get_clip(clip.id)
    assert updated is not None
    return updated


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------


def family(clip: GtaClip) -> list[GtaClip]:
    """The original and all its variants, oldest first."""
    root = clip.family_id
    members = [c for c in store.list_clips(clip.stream_id) if c.family_id == root]
    return sorted(members, key=lambda c: (c.variant_of is not None, c.created_at))


def next_look(used: list[str], base: str) -> str:
    """The next caption look after ``base`` that the family hasn't used."""
    order = list(VARIANT_LOOKS)
    start = order.index(base) if base in order else -1
    rotation = [order[(start + 1 + i) % len(order)] for i in range(len(order))]
    fresh = [look for look in rotation if look not in used]
    return (fresh or rotation)[0]


def next_hook(options: list[str], used: list[str], fallback: str) -> str:
    """The first hook option no family member uses yet, else the next in turn."""
    pool = [o for o in options if o] or [fallback]
    fresh = [o for o in pool if o not in used]
    if fresh:
        return fresh[0]
    return pool[len(used) % len(pool)]


def make_variant(clip_id: str, settings: GtaSettings | None = None) -> GtaClip:
    """A new clip row re-cutting ``clip_id``'s moment (rendered by the queue)."""
    settings = settings or load_settings()
    clip = _clip(clip_id)
    stream = _stream(clip.stream_id)
    _vod_only(stream)
    if clip.status != "ready":
        raise EditError("Only a finished clip that passed QA can get a variant.")
    if stream.status not in WAKEABLE:
        raise EditError("Wait until the stream has finished judging.")
    members = family(clip)
    original = next((c for c in members if c.id == clip.family_id), clip)

    def look(c: GtaClip) -> str:
        return c.caption_style if c.caption_style in CAPTION_STYLES else settings.caption_style

    looks = [look(c) for c in members]
    options = original.hook_options or [original.hook_title]
    variant = GtaClip(
        id=new_id(),
        stream_id=clip.stream_id,
        start_s=original.start_s,
        end_s=original.end_s,
        status="approved",  # the queue finds the shared section and renders
        chunk_index=original.chunk_index,
        hook_title=next_hook(options, [c.hook_title for c in members], original.hook_title),
        hook_quote=original.hook_quote,
        moment_type=original.moment_type,
        summary=original.summary,
        hype_score=original.hype_score,
        score=original.score,
        rank=None,
        verdict={**original.verdict, "variant": len(members)},
        caption=original.caption,
        hashtags=list(original.hashtags),
        hook_options=list(options),
        teaser_start_s=original.teaser_start_s,
        teaser_end_s=original.teaser_end_s,
        flash_forward=original.flash_forward,
        variant_of=original.id,
        motion_seed=max(c.motion_seed for c in members) + 1,
        caption_style=next_look(looks, look(original)),
    )
    store.create_clip(variant)
    wake_stream(stream)
    return variant


# --------------------------------------------------------------------------
# Hype Curve graph + picks
# --------------------------------------------------------------------------


def _workspace(stream: GtaStream):
    from .pipeline import Workspace

    return Workspace(stream.id)


def pick_problem(stream: GtaStream) -> str:
    """Why clips can't be picked on this stream yet ('' = they can)."""
    if stream.signals.get("mode") == "live":
        return "GTA Live streams are clipped as they happen; pick on a VOD instead."
    if stream.status in store.ACTIVE_STREAM_STATUSES and stream.status != "analyzed":
        return "The stream is still being analysed."
    ws = _workspace(stream)
    if not ws.hype.exists():
        return "This stream has no Hype Curve yet."
    if not ws.transcript.exists():
        return "This stream has no transcript yet."
    return ""


def hype_graph(stream_id: str) -> dict:
    """The stream's Hype Curve, pooled for the graph, with its markers and clips."""
    stream = _stream(stream_id)
    ws = _workspace(stream)
    duration = float(stream.duration_s or 0.0)
    out: dict = {
        "duration_s": duration,
        "step_s": 1.0,
        "curve": [],
        "available": [],
        "hotspots": [],
        "clips": [],
        "range_start_s": stream.range_start_s or 0.0,
        "range_end_s": stream.range_end_s or duration,
        "max_pick_s": round(load_settings().max_clip_s + PICK_SLACK_S, 1),
        "min_pick_s": MIN_PICK_S,
        "problem": pick_problem(stream),
    }
    if not ws.hype.exists():
        return out
    curve = HypeCurve.load(ws.hype)
    score = np.clip(curve.score.astype(np.float64) / 100.0, 0.0, 1.0)
    duration = max(duration, float(len(score)))
    step = max(1, int(np.ceil(len(score) / GRAPH_POINTS)))
    if step > 1:
        pad = (-len(score)) % step
        padded = np.concatenate([score, np.zeros(pad)]) if pad else score
        score = padded.reshape(-1, step).max(axis=1)
    out.update(
        duration_s=duration,
        step_s=float(step),
        curve=[round(float(v), 3) for v in score],
        available=curve.available,
        hotspots=[
            {
                "start_s": max(0.0, m.t - 10),
                "end_s": m.t + 10,
                "peak_s": m.t,
                "value": round(min(1.0, m.score / 100.0), 3),
                "reasons": m.reasons,
            }
            for m in curve.markers
        ],
    )
    if not out["range_end_s"]:
        out["range_end_s"] = duration
    clips = [c for c in store.list_clips(stream.id) if c.status not in {"rejected", "candidate"}]
    out["clips"] = [
        {
            "id": c.id,
            "title": c.hook_title,
            "rank": c.rank or 0,
            "start_s": c.start_s,
            "end_s": c.end_s,
            "status": c.status,
            "spans": [],
        }
        for c in clips
    ]
    return out


@dataclass(frozen=True)
class PickRange:
    start_s: float
    end_s: float


def clean_picks(
    ranges: list[tuple[float, float]], *, lo: float, hi: float, max_s: float
) -> list[PickRange]:
    """Validated, sorted, non-overlapping ranges inside the analysed range."""
    if not ranges:
        raise EditError("Drag at least one range on the Hype Curve.", 400)
    if len(ranges) > MAX_PICKS:
        raise EditError(f"Pick at most {MAX_PICKS} ranges at once.", 400)
    out: list[PickRange] = []
    for a, b in sorted(ranges):
        a, b = max(lo, float(a)), min(hi, float(b))
        if b - a < MIN_PICK_S:
            raise EditError(f"Each range must be at least {MIN_PICK_S:.0f} s long.", 400)
        if b - a > max_s:
            raise EditError(
                f"A range of {b - a:.0f} s is too long for a Short (at most {max_s:.0f} s).", 400
            )
        if out and a < out[-1].end_s:
            raise EditError("The ranges overlap; drag them apart.", 400)
        out.append(PickRange(a, b))
    return out


def _snap(transcript: Transcript, a: float, b: float) -> tuple[float, float]:
    """Widen a range to whole words: never start or end mid-word."""
    for w in transcript.words:
        if w.start < a < w.end:
            a = w.start
        if w.start < b < w.end:
            b = w.end
        if w.start > b:
            break
    return round(a, 3), round(b, 3)


def plain_title(transcript: Transcript, a: float, b: float, streamer: str) -> str:
    """A title without an AI judge: the clip's opening words."""
    first, stop = transcript.indices_in_range(a, b)  # half-open
    words = [w.text.strip() for w in transcript.words[first:stop]][:PLAIN_TITLE_WORDS]
    text = judging._clean_title(" ".join(words), max_words=PLAIN_TITLE_WORDS)  # noqa: SLF001
    text = text.rstrip(",;:-")
    if text:
        return text
    return f"{streamer or 'Stream'} at {judging._clock(a)}"  # noqa: SLF001


def make_picks(
    stream_id: str,
    ranges: list[tuple[float, float]],
    *,
    complete=None,
    settings: GtaSettings | None = None,
) -> list[GtaClip]:
    """Clips from ranges dragged on the Hype Curve (rendered by the queue)."""
    settings = settings or load_settings()
    stream = _stream(stream_id)
    problem = pick_problem(stream)
    if problem:
        raise EditError(problem)
    ws = _workspace(stream)
    transcript = Transcript.load(ws.transcript)
    curve = HypeCurve.load(ws.hype)
    lo = stream.range_start_s or 0.0
    hi = stream.range_end_s or stream.duration_s or float(curve.seconds)
    picks = clean_picks(ranges, lo=lo, hi=hi, max_s=settings.max_clip_s + PICK_SLACK_S)
    streamer = stream.channel or "the streamer"

    candidates: list[judging.Candidate] = []
    for number, pick in enumerate(picks):
        a, b = _snap(transcript, pick.start_s, pick.end_s)
        first, stop = transcript.indices_in_range(a, b)  # half-open
        candidates.append(
            judging.Candidate(
                id=f"p{number}",
                chunk_index=-1,
                start_word=first,
                end_word=max(first, stop - 1),
                start_s=a,
                end_s=b,
                hook_quote="",
                moment_type="other",
                why="The creator picked this range on the Hype Curve.",
                confidence=100,
                hype_score=curve.peak(a, b),
                verdict={"source": "pick"},
            )
        )

    described = False
    if transcript.words:
        try:
            if complete is None:
                from .pipeline import _provider_complete

                complete = _provider_complete()
            asyncio.run(
                judging.describe(
                    transcript,
                    candidates,
                    complete=complete,
                    settings=settings,
                    streamer=streamer,
                    title=stream.title,
                    curve=curve,
                )
            )
            described = True
        except Exception as exc:  # no provider / no key / bad answer: plain titles
            log.info("Picked GTA clips get plain titles (%s).", exc)

    created: list[GtaClip] = []
    for c in candidates:
        title = c.hook_title or plain_title(transcript, c.start_s, c.end_s, streamer)
        clip = GtaClip(
            id=new_id(),
            stream_id=stream.id,
            start_s=c.start_s,
            end_s=c.end_s,
            status="approved",
            chunk_index=-1,
            hook_title=title,
            moment_type=c.moment_type,
            summary=c.why,
            hype_score=round(c.hype_score, 1),
            score=c.score,
            verdict={**c.verdict, "source": "pick", "described": described and bool(c.hook_title)},
            caption=c.caption,
            hashtags=c.hashtags,
            hook_options=c.hook_options or [title],
            teaser_start_s=c.teaser_start_s,
            teaser_end_s=c.teaser_end_s,
        )
        store.create_clip(clip)
        created.append(clip)
    if stream.status not in WAKEABLE:
        # A stream whose judging failed (no AI provider...) still has its
        # Hype Curve and transcript: its picks render all the same.
        store.update_stream(stream.id, status="analyzed", error=None, detail="Rendering picks")
        stream = replace(stream, status="analyzed")
    wake_stream(stream)
    return created
