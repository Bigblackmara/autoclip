"""YouTube's own captions as the transcript ("captions first", from CrestClip).

YouTube already ran speech recognition on most videos, and its automatic
captions say when every word starts — to within about a tenth of a second.
When they exist in the language actually spoken, Whisper has nothing to add
for finding clips, and skipping it saves most of a job's wall time.

Two kinds of track are used, both as json3:

* **manual** — uploaded by the creator: human text with punctuation, but timed
  per caption line, so word times are spread over the line by length.
* **automatic** — YouTube's ASR in the original language (the ``xx-orig``
  track; the other automatic tracks are machine translations of it). Word
  timed, lower-case, unpunctuated.

The rule throughout is "fall back to Whisper on any doubt": an unknown
language, a track in another language than the one forced in settings, a
track too sparse to be the real speech, or anything that does not parse.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transcript import Segment, Transcript, Word

log = logging.getLogger(__name__)

#: Transcript.source / DB transcript source for this path.
SOURCE = "youtube_captions"

#: Caption files saved next to a source: ``captions.<kind>.<lang>.json3``.
FILE_GLOB = "captions.*.json3"

#: Below this many words, or this many words per minute of video, the track
#: is not the video's speech (a music video, a lyrics-only track, a stub).
MIN_WORDS = 20
MIN_WORDS_PER_MINUTE = 20.0

#: Only starts are given for automatic words: a word lasts until the next
#: one starts, capped so it never runs through a pause.
MAX_WORD_S = 0.8
#: Manual words are spread over their caption line; cap them a little longer.
MAX_SPREAD_WORD_S = 1.5
MIN_WORD_S = 0.05

#: A pause this long, or a sentence end, closes a segment.
SEGMENT_GAP_S = 0.6
SEGMENT_MAX_WORDS = 16

#: ``[Music]``, ``[Laughter]``, ``(applause)``, music notes, speaker arrows.
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|[♪♫]+|>>")


@dataclass
class CaptionTrack:
    lang: str
    kind: str  # "manual" | "auto"
    url: str

    @property
    def filename(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", self.lang)
        return f"captions.{self.kind}.{safe}.json3"


def _base(lang: str) -> str:
    return lang.split("-")[0].lower()


def _json3_url(formats: Any) -> str | None:
    if not isinstance(formats, list):
        return None
    for fmt in formats:
        if isinstance(fmt, dict) and fmt.get("ext") == "json3" and fmt.get("url"):
            return str(fmt["url"])
    return None


def pick_track(info: dict[str, Any] | None, *, language: str = "") -> CaptionTrack | None:
    """Choose the caption track in the video's own language, or None.

    Manual captions win (human text and punctuation); automatic captions are
    used only in their ``-orig`` (spoken-language) form, or the plain track
    of the video's stated language. ``language`` is the Whisper language
    forced in settings; a track in any other language is refused.
    """
    if not isinstance(info, dict):
        return None
    spoken = str(info.get("language") or "").strip()
    forced = language.strip().lower()

    def allowed(lang: str) -> bool:
        return not forced or _base(lang) == forced

    manual = info.get("subtitles") if isinstance(info.get("subtitles"), dict) else {}
    if spoken:
        for lang in sorted(manual, key=lambda k: (k != spoken, k)):
            if lang == "live_chat" or _base(lang) != _base(spoken) or not allowed(lang):
                continue
            url = _json3_url(manual[lang])
            if url:
                return CaptionTrack(lang=lang, kind="manual", url=url)

    auto = info.get("automatic_captions")
    auto = auto if isinstance(auto, dict) else {}
    originals = [k for k in auto if k.endswith("-orig")]
    for lang in originals:
        base = lang[: -len("-orig")]
        if spoken and _base(base) != _base(spoken):
            continue  # doubt: the "spoken" track disagrees with the video's language
        if not allowed(base):
            continue
        url = _json3_url(auto[lang])
        if url:
            return CaptionTrack(lang=base, kind="auto", url=url)
    if not originals and spoken and spoken in auto and allowed(spoken):
        url = _json3_url(auto[spoken])
        if url:
            return CaptionTrack(lang=spoken, kind="auto", url=url)
    return None


# --------------------------------------------------------------------------
# json3 -> words
# --------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return _NON_SPEECH.sub(" ", text).split()


def parse_json3(data: Any) -> list[Word]:
    """Timed words from a json3 caption document.

    Each segment ("seg") of an event runs from its offset to the next seg's
    offset (or the event's end); the words inside a seg share that span by
    character length. Automatic captions have one word per seg, so this is
    exact for them; manual captions have one seg per line, so their words
    are spread over the line.
    """
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise ValueError("not a json3 caption document")

    words: list[tuple[float, float, str, bool]] = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("segs"), list):
            continue
        try:
            start = float(event.get("tStartMs") or 0) / 1000.0
            end = start + float(event.get("dDurationMs") or 0) / 1000.0
        except (TypeError, ValueError):
            continue
        segs = [s for s in event["segs"] if isinstance(s, dict)]
        offsets: list[float] = []
        for seg in segs:
            try:
                offsets.append(start + float(seg.get("tOffsetMs") or 0) / 1000.0)
            except (TypeError, ValueError):
                offsets.append(start)
        for index, seg in enumerate(segs):
            tokens = _tokens(str(seg.get("utf8") or ""))
            if not tokens:
                continue
            seg_start = offsets[index]
            seg_end = offsets[index + 1] if index + 1 < len(segs) else max(end, seg_start)
            spread = len(tokens) > 1
            total = sum(len(t) for t in tokens) or 1
            cursor = seg_start
            for token in tokens:
                share = (seg_end - seg_start) * len(token) / total
                words.append((cursor, cursor + share, token, spread))
                cursor += share

    words.sort(key=lambda w: w[0])
    out: list[Word] = []
    for i, (start, end, text, spread) in enumerate(words):
        if out and out[-1].text == text and abs(out[-1].start - start) < 1e-3:
            continue  # a repeated event
        next_start = words[i + 1][0] if i + 1 < len(words) else float("inf")
        cap = MAX_SPREAD_WORD_S if spread else MAX_WORD_S
        stop = min(end if end > start else start + cap, next_start, start + cap)
        out.append(
            Word(text=text, start=round(start, 3), end=round(max(start + MIN_WORD_S, stop), 3))
        )
    # Starts are sorted; keep every end at or before the next start.
    for a, b in zip(out, out[1:], strict=False):
        if a.end > b.start:
            a.end = max(a.start, b.start)
    return out


def build_segments(words: list[Word]) -> list[Segment]:
    segments: list[Segment] = []
    first = 0
    for i, word in enumerate(words):
        last = i == len(words) - 1
        gap = words[i + 1].start - word.end if not last else 0.0
        if last or word.ends_sentence or gap >= SEGMENT_GAP_S or i - first + 1 >= SEGMENT_MAX_WORDS:
            chunk = words[first : i + 1]
            segments.append(
                Segment(
                    text=" ".join(w.text for w in chunk),
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    first_word=first,
                    last_word=i,
                )
            )
            first = i + 1
    return segments


def to_transcript(data: Any, *, lang: str, kind: str, duration_s: float = 0.0) -> Transcript | None:
    """A Transcript from a json3 document, or None when it fails the checks."""
    try:
        words = parse_json3(data)
    except ValueError as exc:
        log.info("YouTube captions unusable (%s); using Whisper.", exc)
        return None
    if len(words) < MIN_WORDS:
        log.info("YouTube captions have only %d words; using Whisper.", len(words))
        return None
    if duration_s > 0:
        per_minute = len(words) / (duration_s / 60.0)
        if per_minute < MIN_WORDS_PER_MINUTE:
            log.info("YouTube captions too sparse (%.0f words/min); using Whisper.", per_minute)
            return None
        if words[-1].start > duration_s + 5.0:
            log.info("YouTube captions run past the video's end; using Whisper.")
            return None
    return Transcript(
        words=words,
        segments=build_segments(words),
        language=_base(lang),
        model=f"youtube-{kind}:{lang}",
        source=SOURCE,
    )


def load_for_source(
    source_dir: Path, *, duration_s: float = 0.0, language: str = ""
) -> Transcript | None:
    """The saved caption track next to a source as a Transcript, or None."""
    files = sorted(Path(source_dir).glob(FILE_GLOB))
    # Manual before automatic, matching pick_track.
    files.sort(key=lambda p: (".manual." not in p.name, p.name))
    for path in files:
        parts = path.name.split(".")
        if len(parts) != 4:
            continue
        _, kind, lang, _ = parts
        if language and _base(lang) != language.strip().lower():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.info("Unreadable caption file %s; skipping it.", path.name)
            continue
        transcript = to_transcript(data, lang=lang, kind=kind, duration_s=duration_s)
        if transcript is not None:
            return transcript
    return None
