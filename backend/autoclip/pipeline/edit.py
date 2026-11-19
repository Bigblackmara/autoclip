"""Short-form editing on top of a normal-mode export.

What a human short-form editor does after the cut is chosen, applied in the
same ffmpeg pass as the crop and captions:

* **Pace** — long pauses are shortened (a lone "um" between two pauses goes
  too). Cuts land only on silences measured from the audio; transcript word
  timings are estimates and are used only to recognise fillers and to refuse a
  cut that would swallow a word.
* **Motion** — punch-in zooms at natural pauses, plus a zoom change at every
  jump cut so the jump hides in the picture.
* **Hook headline** — the chosen hook (else the clip's title) on screen for
  the first ~3 seconds.
* **Flash-forward** — optionally, the clip's punchiest 1-3 s line plays
  first, punched in, then the clip from the start. It is the first keep of
  the plan, so captions, zooms and the audio follow it automatically.
* **Key words** — 3-8 important words drawn bigger in an accent colour.
* **Music** — a user-supplied track (or one picked from a folder), looped,
  faded and ducked under speech.
* **Watermark** — a text logo or PNG in a chosen corner.
* **Loudness** — two-pass loudnorm over the exact audio that will be rendered.

Times in this module are clip-relative (0 = clip start) unless stated.
Ported from CrestClip's pace/render/captions modules and generalised from GTA
Mode's renderer, which now imports the shared pieces from here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import ffmpeg
from .transcript import Word

if TYPE_CHECKING:
    from ..config import ExportSettings

log = logging.getLogger(__name__)

FPS = 30

# --------------------------------------------------------------------------
# Keeps (spans of the clip that survive the cut)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Keep:
    """A kept span, in the caller's time base (clip-relative for exports)."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def output_duration(keeps: Sequence[Keep]) -> float:
    return sum(k.duration for k in keeps)


def remap_words(words: list[Word], keeps: list[Keep]) -> list[Word]:
    """Word timings on the output timeline; a word overlapping a keep is clipped to it."""
    out: list[Word] = []
    offsets: list[float] = []
    total = 0.0
    for keep in keeps:
        offsets.append(total)
        total += keep.duration
    for word in words:
        for keep, offset in zip(keeps, offsets, strict=True):
            if word.end <= keep.start or word.start >= keep.end:
                continue
            start = offset + max(word.start, keep.start) - keep.start
            end = offset + min(word.end, keep.end) - keep.start
            if end > start:
                out.append(Word(text=word.text, start=start, end=end, speaker=word.speaker))
            break
    return out


def retime_words(words: Sequence[Word], keeps: Sequence[Keep]) -> list[Word]:
    """Words re-timed onto the edited clip; a word whose middle was cut is dropped."""
    out: list[Word] = []
    offset = 0.0
    for keep in keeps:
        for word in words:
            mid = (word.start + word.end) / 2
            if mid < keep.start or mid >= keep.end:
                continue
            out.append(
                Word(
                    text=word.text,
                    start=round(offset + max(word.start, keep.start) - keep.start, 3),
                    end=round(offset + min(word.end, keep.end) - keep.start, 3),
                    speaker=word.speaker,
                )
            )
        offset += keep.duration
    return out


def cut_times(keeps: Sequence[Keep]) -> list[float]:
    """Output times of the jump cuts (where each kept piece after the first starts)."""
    out: list[float] = []
    t = 0.0
    for keep in keeps:
        if t > 0:
            out.append(round(t, 3))
        t += keep.duration
    return out


# --------------------------------------------------------------------------
# Pace
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PaceRule:
    #: Pauses longer than this are shortened.
    gap: float
    #: Pause left on each side of a cut.
    keep: float
    #: Quiet allowed at the very end of the clip.
    tail: float


PACE_RULES: dict[str, PaceRule] = {
    "natural": PaceRule(gap=0.6, keep=0.18, tail=0.7),
    "fast": PaceRule(gap=0.35, keep=0.10, tail=0.4),
}

FILLER = re.compile(r"^(u+m+|u+h+m*|e+r+m*|e+r+|a+h+|h+m+|mm+)[.,!?…]*$", re.IGNORECASE)
#: Shorter cuts aren't worth a jump.
MIN_CUT_S = 0.12
#: Never leave slivers of picture.
MIN_KEEP_S = 0.10
#: More than this share cut usually means the "silence" was quiet speech.
MAX_REMOVED = 0.5


def silences_in(
    silences: Sequence[object], start_s: float, end_s: float
) -> list[tuple[float, float]]:
    """Absolute silences (objects with ``start``/``end``) relative to one clip."""
    out: list[tuple[float, float]] = []
    for item in silences:
        a = float(item.start)  # type: ignore[attr-defined]
        b = float(item.end)  # type: ignore[attr-defined]
        if b > start_s and a < end_s:
            out.append((a - start_s, b - start_s))
    return out


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 0.02:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def plan_pace(
    duration: float,
    silences: Sequence[tuple[float, float]],
    words: Sequence[Word],
    pace: str,
) -> list[Keep]:
    """Which parts of a clip to keep, in play order. ``[Keep(0, duration)]`` = uncut.

    ``silences`` and ``words`` are clip-relative.
    """
    whole = [Keep(0.0, duration)]
    rule = PACE_RULES.get(pace)
    if rule is None or duration < 4.0:
        return whole

    sil = sorted(
        (max(0.0, a), min(duration, b))
        for a, b in silences
        if min(duration, b) - max(0.0, a) > 0.05
    )

    cuts: list[tuple[float, float]] = []
    for a, b in sil:
        if a <= 0.05:
            cuts.append((0.0, b - rule.keep))  # dead air before the first word
        elif b >= duration - 0.05:
            cuts.append((a + rule.tail, duration))  # long quiet tail
        elif b - a > rule.gap:
            cuts.append((a + rule.keep, b - rule.keep))

    # A cut must never swallow the middle of a transcribed word: silencedetect
    # can mistake quiet speech for silence. Fillers are fair game.
    spoken = [(w.start + w.end) / 2 for w in words if not FILLER.match(w.text.strip())]
    cuts = [(a, b) for a, b in cuts if not any(a < mid < b for mid in spoken)]

    # A filler alone between two pauses: "...and then. Um. We tried..." goes
    # as one cut (pause + um + pause).
    for first, second in zip(sil, sil[1:], strict=False):
        blip = second[0] - first[1]
        if blip < 0.12 or blip > 0.8:
            continue
        inside = [w for w in words if first[1] - 0.35 < (w.start + w.end) / 2 < second[0] + 0.35]
        if inside and all(FILLER.match(w.text.strip()) for w in inside):
            cuts.append((first[0] + rule.keep, second[1] - rule.keep))

    merged = _merge([(a, b) for a, b in cuts if b - a >= MIN_CUT_S])
    keeps: list[Keep] = []
    t = 0.0
    for a, b in merged:
        if a - t >= MIN_KEEP_S:
            keeps.append(Keep(t, a))
        t = max(t, b)
    if duration - t >= MIN_KEEP_S:
        keeps.append(Keep(t, duration))
    kept = output_duration(keeps)
    if not keeps or kept < duration * (1 - MAX_REMOVED):
        return whole
    return [Keep(round(k.start, 3), round(k.end, 3)) for k in keeps]


def snap_keeps(keeps: Sequence[Keep], duration: float, fps: int = FPS) -> list[Keep]:
    """Put every cut exactly on a video frame.

    Picture can only be cut between frames while sound can be cut anywhere; if
    the two are cut at slightly different spots each jump moves the sound a
    few ms against the picture, and after a dozen cuts lips and captions are
    visibly out of sync. Snapped, both are cut identically.
    """
    end_limit = int(duration * fps + 1e-6) / fps
    out: list[Keep] = []
    for keep in keeps:
        a = max(0.0, round(keep.start * fps) / fps)
        b = min(end_limit, round(keep.end * fps) / fps)
        if b - a < MIN_KEEP_S:
            continue
        if out and a <= out[-1].end + 1e-6:
            out[-1] = Keep(out[-1].start, max(out[-1].end, round(b, 4)))
        else:
            out.append(Keep(round(a, 4), round(b, 4)))
    return out or [Keep(0.0, round(end_limit, 4))]


# --------------------------------------------------------------------------
# Motion (punch-in zooms)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Zoom:
    #: Output seconds from which this zoom level holds.
    t: float
    #: 1 = no zoom.
    z: float


#: Zoom step at a jump cut when Motion is off: just enough to hide the jump.
CUT_ZOOM = 1.06
_LEVELS = {"subtle": (1.0, 1.08), "dynamic": (1.0, 1.12, 1.0, 1.2)}
_SPACING = {"subtle": 4.5, "dynamic": 2.2}
_CLAUSE_END = re.compile(r"[.!?,;:]$")


#: Variant zoom patterns: spacing and depth multipliers by ``seed % 3``.
_SEED_SPACING = (1.0, 0.8, 1.25)
_SEED_DEPTH = (1.0, 1.25, 0.75)
#: Steady punch-in of a variant when Motion is off, by ``seed % 3``.
_SEED_STATIC = (1.03, 1.05, 1.07)


def _seeded(levels: tuple[float, ...], spacing: float, seed: int) -> tuple[list[float], float]:
    """A variant's zoom pattern: odd seeds start punched in, others change pace and depth."""
    if seed <= 0:
        return list(levels), spacing
    k = seed % 3
    shift = seed % len(levels)
    rotated = list(levels[shift:] + levels[:shift])
    deeper = [round(1.0 + (z - 1.0) * _SEED_DEPTH[k], 3) for z in rotated]
    return deeper, spacing * _SEED_SPACING[k]


def plan_zooms(
    words: Sequence[Word],
    duration: float,
    motion: str,
    jump_cuts: Sequence[float] = (),
    seed: int = 0,
) -> list[Zoom]:
    """Step zooms like a short-form editor: normal and closer framing alternate.

    ``words`` and ``jump_cuts`` are on the output timeline. Subtle zooms every
    ~4.5 s at natural pauses, dynamic every ~2.2 s with two depths. Every jump
    cut always gets a zoom change (also with motion "off"), which hides the jump.

    ``seed`` > 0 gives a variant its own pattern (different spacing, depth and
    starting framing) so a re-post of the same moment is not frame-identical;
    with motion "off" a variant gets a steady slight punch-in instead.
    """
    if duration < 4.0:
        return []
    cuts = [t for t in jump_cuts if 0.3 < t < duration - 0.3]
    if motion not in _LEVELS:
        base = _SEED_STATIC[seed % 3] if seed > 0 else 1.0
        if not cuts:
            return [Zoom(0.0, base)] if seed > 0 else []
        phase = seed % 2
        out = [Zoom(0.0, base)]
        for i, t in enumerate(sorted(cuts)):
            stepped = (i + phase) % 2 == 0
            out.append(Zoom(round(t, 3), round(base * CUT_ZOOM, 3) if stepped else base))
        return out

    levels, spacing = _seeded(_LEVELS[motion], _SPACING[motion], seed)
    forced = set(cuts)
    events = list(cuts)
    last = float("-inf")
    for prev, word in zip(words, words[1:], strict=False):
        pause = word.start - prev.end >= 0.25 or bool(_CLAUSE_END.search(prev.text.strip()))
        t = max(0.0, word.start - 0.04)
        if (
            not pause
            or t - last < spacing
            or word.start > duration - 1.2
            or any(abs(c - t) < 1.0 for c in cuts)
        ):
            continue
        events.append(t)
        last = t
    events.sort()
    out = [Zoom(0.0, levels[0])]
    index = 0
    previous = 0.0
    for t in events:
        if t - previous < 0.5 and t not in forced:
            continue
        index = (index + 1) % len(levels)
        out.append(Zoom(round(t, 3), levels[index]))
        previous = t
    return out


def zoom_expression(zooms: Sequence[Zoom], intro: bool = False) -> str:
    """A zoompan ``z`` expression: piecewise-constant levels over input time ``it``.

    ``intro`` adds a quick ease-out from 1.18x over the first half second.
    """
    expr = f"{zooms[-1].z:.3f}" if zooms else "1"
    for i in range(len(zooms) - 2, -1, -1):
        expr = f"if(lt(it,{zooms[i + 1].t:.3f}),{zooms[i].z:.3f},{expr})"
    if intro:
        expr = f"if(lt(it,0.5),1.18-0.18*(it/0.5)*(2-it/0.5),{expr})"
    return expr


def zoom_filter(zooms: Sequence[Zoom], intro: bool, out_w: int, out_h: int, fps: int) -> str:
    """Zoom about the upper-middle of the frame, where faces sit in a crop."""
    return (
        f"zoompan=z='{zoom_expression(zooms, intro)}'"
        f":x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*0.4':d=1:s={out_w}x{out_h}:fps={fps}"
    )


# --------------------------------------------------------------------------
# Key words
# --------------------------------------------------------------------------

STOPWORDS = frozenset(
    """
    a about above after again against all almost also although always am among an and
    another any anybody anyone anything anyway are aren around as at away back be
    became because become been before being below between both but by can cannot
    could couldn did didn do does doesn doing don done down during each either else
    enough even ever every everybody everyone everything few for from further get gets
    getting give given go goes going gone gonna got gotta had hadn has hasn have haven
    having he her here hers herself him himself his how however i if in into is isn it
    its itself just kind kinda know let like literally lot made make makes making many
    may maybe me might mine more most much must my myself need never no nobody none nor
    not nothing now of off often oh ok okay on once one only onto or other others our
    ours ourselves out over own pretty quite rather really right said same say says see
    seem seems she should shouldn since so some somebody someone something sometimes
    sort still such sure take than that thats the their theirs them themselves then
    there these they thing things think this those though through thus to too toward
    under until up upon us very want wanna was wasn way we well went were weren what
    whatever when where whether which while who whole whom whose why will with within
    without won would wouldn yeah yes yet you your yours yourself yourselves youre
    actually basically going stuff guys guy thank thanks hello hey
    dont cant wont didnt doesnt isnt wasnt arent werent havent hasnt hadnt couldnt
    wouldnt shouldnt aint im ive id ill youve youll youd theyre theyve theyll hes
    shes whats theres heres lets weve were well thatll whos shouldve couldve wouldve
    mustve mightve
    """.split()  # noqa: SIM905 - a word list reads better as prose
)

_NORM = re.compile(r"[^\w$%'-]", re.UNICODE)


def normalise_word(text: str) -> str:
    return _NORM.sub("", text.lower()).strip("'-")


def _key(text: str) -> str:
    """Comparable form of a spoken word: lower case, no punctuation or possessive."""
    token = normalise_word(text)
    if token.endswith("'s"):
        token = token[:-2]
    return token.replace("'", "")


def pick_keywords(words: Sequence[Word], *, minimum: int = 3, maximum: int = 8) -> list[str]:
    """Pick 3-8 important words deterministically (no model call).

    Numbers and money first, then the longest non-stopword words; ties go to
    the earlier word. One candidate per word stem, roughly one key word per
    ten spoken words.
    """
    candidates: dict[str, tuple[float, int]] = {}
    for index, word in enumerate(words):
        bare = _key(word.text)
        if not bare or bare in STOPWORDS or FILLER.match(bare):
            continue
        has_digit = any(ch.isdigit() for ch in bare)
        if not has_digit and (len(bare) < 4 or not any(ch.isalpha() for ch in bare)):
            continue
        if any(_same_stem(bare, other) for other in candidates):
            continue
        score = len(bare) + (8.0 if has_digit or "$" in bare or "%" in bare else 0.0)
        candidates[bare] = (score, index)
    target = max(minimum, min(maximum, len(words) // 10))
    ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], item[1][1]))
    return [key for key, _ in ranked[:target]]


def _same_stem(a: str, b: str) -> bool:
    """Crude stemming: fail/failed/failing/failure share a stem."""
    if a == b:
        return True
    shorter = min(len(a), len(b))
    if shorter < 4:
        return False
    common = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        common += 1
    return common >= max(4, shorter - 3)


def keyword_of(text: str, keywords: Sequence[str]) -> str | None:
    """The key word this spoken word is (exact, or a short ending like fail/failed)."""
    token = _key(text)
    if len(token) < 2:
        return None
    for keyword in keywords:
        if token == keyword or (len(keyword) >= 4 and _same_stem(token, keyword)):
            return keyword
    return None


# --------------------------------------------------------------------------
# Music
# --------------------------------------------------------------------------

MUSIC_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac")
MUSIC_FADE_IN_S = 0.8
MUSIC_FADE_OUT_MAX_S = 1.8
AUDIO_FORMAT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"


def resolve_music(path_setting: str, seed: str = "") -> Path | None:
    """The track to use: the file itself, or one picked from a folder by ``seed``.

    The pick is deterministic per clip (a re-export keeps its track) and
    spreads a job's clips across the folder.
    """
    raw = path_setting.strip().strip('"')
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        tracks = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in MUSIC_EXTENSIONS
        )
        if not tracks:
            log.warning("No music tracks in %s; rendering without music.", path)
            return None
        digest = hashlib.sha1(seed.encode("utf-8")).digest()
        return tracks[int.from_bytes(digest[:4], "big") % len(tracks)]
    log.warning("Music %s not found; rendering without music.", path)
    return None


def music_chain(
    music_index: int,
    duration: float,
    volume: float,
    *,
    duck: bool,
    speech: str | None,
    out_label: str,
) -> str:
    """Music (input ``music_index``, looped by ``-stream_loop``) laid under ``speech``.

    The track is cut to the clip, faded in and out, and when ``duck`` is set
    compressed by the speech itself (sidechain), so the voice always stays clear.
    """
    d = max(1.0, duration)
    fade_out = min(MUSIC_FADE_OUT_MAX_S, d / 4)
    music = (
        f"[{music_index}:a]{AUDIO_FORMAT},atrim=0:{d:.3f},asetpts=PTS-STARTPTS,"
        f"volume={volume:.3f},afade=t=in:st=0:d={MUSIC_FADE_IN_S},"
        f"afade=t=out:st={d - fade_out:.3f}:d={fade_out:.3f}[mus]"
    )
    mix = "amix=inputs=2:duration=first:dropout_transition=0:normalize=0"
    if speech is None:
        return f"{music};[mus]anull{out_label}"
    if duck:
        return (
            f"{music};{speech}asplit=2[sp1][sp2];"
            "[mus][sp2]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[mduck];"
            f"[sp1][mduck]{mix}{out_label}"
        )
    return f"{music};{speech}[mus]{mix}{out_label}"


# --------------------------------------------------------------------------
# Watermark
# --------------------------------------------------------------------------

CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
#: Logo box (max width, max height) as a share of the frame width / height.
WATERMARK_BOX = (0.19, 0.07)


@dataclass(frozen=True)
class Watermark:
    path: Path
    corner: str = "top_right"
    opacity: float = 0.85


def watermark_margins(out_w: int, out_h: int) -> tuple[int, int, int]:
    """(side, top, bottom) gaps in px. 9:16 keeps clear of the app UI bands."""
    side = round(out_w * 0.04)
    if out_h > out_w:
        return side, round(out_h * 0.075), round(out_h * 0.16)
    return side, round(out_h * 0.04), round(out_h * 0.04)


def watermark_box(out_w: int, out_h: int) -> tuple[int, int]:
    return round(out_w * WATERMARK_BOX[0]), round(out_h * WATERMARK_BOX[1])


def watermark_filter(
    watermark: Watermark, input_index: int, in_label: str, out_label: str, out_w: int, out_h: int
) -> str:
    box_w, box_h = watermark_box(out_w, out_h)
    side, top, bottom = watermark_margins(out_w, out_h)
    corner = watermark.corner if watermark.corner in CORNERS else "top_right"
    x = str(side) if corner.endswith("left") else f"W-w-{side}"
    y = str(top) if corner.startswith("top") else f"H-h-{bottom}"
    return (
        f"[{input_index}:v]scale={box_w}:{box_h}:force_original_aspect_ratio=decrease,"
        f"format=rgba,colorchannelmixer=aa={watermark.opacity:.3f}[wm];"
        f"{in_label}[wm]overlay={x}:{y}:format=auto{out_label}"
    )


def resolve_watermark(settings: ExportSettings, cache_dir: Path | None = None) -> Watermark | None:
    """The configured watermark, or None (off, missing file, empty text)."""
    mode = settings.watermark
    if mode == "off":
        return None
    corner = settings.watermark_corner
    opacity = settings.watermark_opacity
    if mode == "image":
        raw = settings.watermark_image.strip().strip('"')
        path = Path(raw).expanduser() if raw else None
        if path is None or not path.is_file():
            log.warning("Watermark image %s not found; rendering without it.", raw or "(unset)")
            return None
        return Watermark(path=path, corner=corner, opacity=opacity)
    text = " ".join(settings.watermark_text.split())
    if not text:
        return None
    from ..gta import logo

    if cache_dir is None:
        from .. import paths

        cache_dir = paths.work_dir() / "logo"
    try:
        path = logo.text_logo(text, cache_dir=cache_dir)
        return Watermark(path=path, corner=corner, opacity=opacity)
    except logo.LogoError as exc:
        log.warning("Could not draw the text logo (%s); rendering without it.", exc)
        return None


# --------------------------------------------------------------------------
# Hook headline placement
# --------------------------------------------------------------------------

HOOK_SECONDS = 3.0
HOOK_MAX_CHARS = 80


# --------------------------------------------------------------------------
# Flash-forward (cold open)
# --------------------------------------------------------------------------

#: A tease line lasts this long (seconds of source).
TEASER_MIN_S = 0.8
TEASER_MAX_S = 3.6
#: It must start at least this far into the clip: the opening line itself
#: would just play twice in a row.
TEASER_LEAD_S = 2.5
#: Punch-in over the tease, so it reads as a flash-forward, not a stutter.
TEASER_ZOOM = 1.16
#: Clips shorter than this get no flash-forward.
TEASER_MIN_CLIP_S = 6.0
_CLAUSE = re.compile(r"[,;:\u2014\u2013]$")


def fit_teaser(
    span: tuple[float, float], duration: float, silences: Sequence[tuple[float, float]] = ()
) -> Keep | None:
    """Check and tidy a tease line (clip-relative) against the clip.

    It must sit inside the clip but not at its very start, last 0.8-3.6 s and,
    when a measured pause follows it, end in that pause so no word is clipped.
    """
    start, end = span
    s = max(0.0, start - 0.1)
    if s < TEASER_LEAD_S or s > duration - TEASER_MIN_S or end <= start:
        return None
    pause = next(
        (a for a, _ in sorted(silences) if s + TEASER_MIN_S <= a <= s + 3.5 and a >= end - 0.6),
        None,
    )
    e = min(duration, pause + 0.1) if pause is not None else min(end + 0.15, duration)
    if not (TEASER_MIN_S <= e - s <= TEASER_MAX_S):
        return None
    return Keep(round(s, 3), round(e, 3))


def _phrases(words: Sequence[Word]) -> list[tuple[int, int, bool]]:
    """(first, last, ends_sentence) runs, split at punctuation and pauses."""
    out: list[tuple[int, int, bool]] = []
    first = 0
    for index, word in enumerate(words):
        nxt = words[index + 1] if index + 1 < len(words) else None
        sentence = word.ends_sentence
        if (
            nxt is None
            or sentence
            or _CLAUSE.search(word.text.strip())
            or nxt.start - word.end >= 0.3
        ):
            out.append((first, index, sentence or nxt is None))
            first = index + 1
    return out


def _word_weight(word: Word, keywords: Sequence[str]) -> float:
    bare = _key(word.text)
    if not bare or bare in STOPWORDS or FILLER.match(bare):
        return 0.0
    if any(ch.isdigit() for ch in bare) or "$" in bare or "%" in bare:
        return 6.0
    weight = 4.0 if keyword_of(word.text, keywords) else 0.0
    if len(bare) >= 5:
        weight += 2.0
    elif len(bare) >= 4:
        weight += 1.0
    return weight


def pick_teaser(
    words: Sequence[Word],
    duration: float,
    silences: Sequence[tuple[float, float]] = (),
    keywords: Sequence[str] = (),
) -> Keep | None:
    """The best 1-3 s line for a flash-forward when the model flagged none.

    ``words`` are clip-relative. Candidates are runs of whole phrases inside
    one sentence; each is scored on what it says (numbers, key words, long
    content words, an exclamation or question) per square-root word, with a
    bonus for the second half, where punchlines live. The best one that
    :func:`fit_teaser` accepts wins.
    """
    if duration < TEASER_MIN_CLIP_S or not words:
        return None
    keywords = list(keywords) or pick_keywords(words)
    longest = TEASER_MAX_S - 0.2
    spans: set[tuple[int, int]] = set()
    phrases = _phrases(words)
    for i, (first, _, _) in enumerate(phrases):
        for j in range(i, len(phrases)):
            last = phrases[j][1]
            if words[last].end - words[first].start > longest:
                break
            spans.add((first, last))
            if phrases[j][2]:
                break  # never run on into the next sentence
    # A sentence too long to flash whole still ends on its payoff: its last
    # 1-3 seconds are a candidate too.
    first = 0
    for index, word in enumerate(words):
        if word.ends_sentence or index == len(words) - 1:
            head = first
            while head < index and word.end - words[head].start > longest:
                head += 1
            spans.add((head, index))
            first = index + 1

    candidates: list[tuple[float, float, float]] = []
    for first, last in spans:
        start, end = words[first].start, words[last].end
        if end - start < TEASER_MIN_S or start < TEASER_LEAD_S:
            continue
        run = words[first : last + 1]
        score = sum(_word_weight(w, keywords) for w in run) / max(1, len(run)) ** 0.5
        if run[-1].text.strip().endswith(("!", "?")):
            score += 2.0
        if start >= duration / 2:
            score += 1.5
        if score > 0:
            candidates.append((score, start, end))
    for _, start, end in sorted(candidates, key=lambda c: (-c[0], c[1])):
        fitted = fit_teaser((start, end), duration, silences)
        if fitted is not None:
            return fitted
    return None


def hook_top_px(out_w: int, out_h: int, *, split: bool, watermark: Watermark | None) -> int:
    """Top of the hook headline: below the app's top UI, and below a top-corner logo."""
    top = round(out_h * ((0.12 if split else 0.156) if out_h > out_w else 0.06))
    if watermark is not None and watermark.corner.startswith("top"):
        _, margin, _ = watermark_margins(out_w, out_h)
        top = max(top, margin + watermark_box(out_w, out_h)[1] + round(out_h * 0.015))
    return top


def hook_text(title: str) -> str:
    text = " ".join(title.split())
    if len(text) > HOOK_MAX_CHARS:
        text = text[: HOOK_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return text


# --------------------------------------------------------------------------
# Loudness (two-pass loudnorm)
# --------------------------------------------------------------------------

LOUDNORM_LRA = 11.0
_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)
_MEASURED = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def parse_loudnorm(stderr: str) -> dict[str, str] | None:
    """The measured values from a ``loudnorm=print_format=json`` run, if usable."""
    match = _LOUDNORM_JSON.search(stderr or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not all(key in data for key in _MEASURED):
        return None
    try:
        if float(data["input_i"]) < -70:  # (near) silent: linear gain would explode
            return None
    except (TypeError, ValueError):
        return None
    return {key: str(data[key]) for key in _MEASURED}


def measure_loudnorm(
    input_args: Sequence[str],
    graph: str,
    *,
    label: str = "[aout]",
) -> dict[str, str] | None:
    """First loudnorm pass: run ``graph`` (ending in a print_format=json loudnorm) to null."""
    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        *input_args,
        "-filter_complex",
        graph,
        "-map",
        label,
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, check=False, encoding="utf-8", errors="replace"
    )
    measured = parse_loudnorm(result.stderr or "") if result.returncode == 0 else None
    if measured is None:
        log.warning("Loudness measurement failed; falling back to one-pass loudnorm.")
    return measured


#: Most gain a quiet clip gets before limiting (beyond this it's mostly noise).
MAX_GAIN_DB = 24.0


def _gain_and_limit(measured: dict[str, str], *, target: float, true_peak: float) -> str:
    """Even gain + look-ahead limiter, when loudnorm can't stay linear.

    loudnorm only keeps ``linear=true`` if the gain fits under the true-peak
    ceiling; a quiet stream with loud shouts doesn't, and loudnorm then drops
    to its dynamic mode, whose first seconds come out loud and crunchy. Here
    the gain is applied evenly and only the peaks are limited (delay
    compensated, so audio stays in sync). '' = loudnorm can stay linear.
    """
    try:
        gain = target - float(measured["input_i"])
        peak = float(measured["input_tp"])
    except (KeyError, ValueError):
        return ""
    if peak + gain <= true_peak:
        return ""
    gain = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, gain))
    limit = 10 ** ((true_peak - 0.5) / 20)  # sample-peak limiter, 0.5 dB under the TP ceiling
    return (
        f"volume={gain:.2f}dB,"
        f"alimiter=limit={limit:.4f}:attack=5:release=80:level=false:latency=true"
    )


def loudnorm_filter(
    measured: dict[str, str] | None, *, target: float, true_peak: float, lra: float = LOUDNORM_LRA
) -> str:
    base = f"loudnorm=I={target}:TP={true_peak}:LRA={lra:g}"
    if not measured:
        return base
    limited = _gain_and_limit(measured, target=target, true_peak=true_peak)
    if limited:
        return limited
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def measure_filter(*, target: float, true_peak: float, lra: float = LOUDNORM_LRA) -> str:
    return f"loudnorm=I={target}:TP={true_peak}:LRA={lra:g}:print_format=json"


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass
class EditPlan:
    """Everything the export applies on top of the crop, for one clip."""

    #: Clip-relative spans to keep, in play order. None = the whole clip.
    keeps: list[Keep] | None = None
    zooms: list[Zoom] = field(default_factory=list)
    #: Quick zoom-out over the first half second (dynamic motion).
    intro_zoom: bool = False
    hook_text: str = ""
    hook_s: float = HOOK_SECONDS
    keywords: list[str] = field(default_factory=list)
    #: The flash-forward line (clip-relative source span) played first, or
    #: None. When set it is also ``keeps[0]``.
    teaser: Keep | None = None
    #: Output time the hook headline appears (after the flash-forward).
    hook_start_s: float = 0.0
    watermark: Watermark | None = None
    music: Path | None = None
    #: Linear gain applied to the music before mixing.
    music_volume: float = 0.3
    music_duck: bool = True
    fps: int = FPS
    two_pass_loudnorm: bool = True

    def kept(self, duration: float) -> list[Keep]:
        return self.keeps or [Keep(0.0, duration)]

    def output_duration(self, duration: float) -> float:
        return output_duration(self.kept(duration))

    @property
    def has_motion(self) -> bool:
        return self.intro_zoom or any(z.z != 1.0 for z in self.zooms)

    def summary(self) -> list[str]:
        """Edits actually applied (for logs)."""
        out: list[str] = []
        if self.teaser is not None:
            out.append("flash-forward")
        cuts = len(self.keeps or []) - 1 - (1 if self.teaser is not None else 0)
        if cuts > 0:
            out.append(f"{cuts} pause cuts")
        if self.has_motion:
            out.append("zooms")
        if self.hook_text:
            out.append("hook")
        if self.keywords:
            out.append("key words")
        if self.music is not None:
            out.append("music")
        if self.watermark is not None:
            out.append("watermark")
        return out


def plan_for_clip(
    settings: ExportSettings,
    *,
    start_s: float,
    end_s: float,
    words: Sequence[Word],
    silences: Sequence[object] = (),
    title: str = "",
    seed: str = "",
    allow_motion: bool = True,
    watermark: Watermark | None = None,
    display_words: Sequence[Word] | None = None,
    flash_forward: bool | None = None,
    teaser: tuple[float, float] | None = None,
    motion_seed: int = 0,
) -> EditPlan:
    """Build the edit plan for one clip from the export settings.

    ``words`` and ``silences`` are in absolute (source) time. ``allow_motion``
    is False for layouts zooms would break (the split facecam stack, audio-only).
    ``title`` is the hook headline (the clip's chosen hook, else its title).

    ``display_words`` are the caption words when they differ from the spoken
    ones (a translated caption language): key words are picked from them,
    while pause cuts keep using the measured ``words``.

    ``flash_forward`` (None = ``settings.flash_forward``) opens on a tease
    line: ``teaser`` (absolute, as flagged by the model) when it fits, else the
    best line :func:`pick_teaser` finds. The tease becomes the first keep, so
    the captions, the audio and the zooms (shifted by its length, with a
    punch-in over it) all follow it. ``motion_seed`` picks a variant's zoom
    pattern.
    """
    duration = max(0.0, end_s - start_s)

    def relative(items: Sequence[Word]) -> list[Word]:
        return [
            Word(text=w.text, start=w.start - start_s, end=w.end - start_s, speaker=w.speaker)
            for w in items
        ]

    rel_words = relative(words)
    rel_display = relative(display_words) if display_words is not None else rel_words
    fps = settings.fps
    clip_silences = silences_in(silences, start_s, end_s)
    keeps: list[Keep] | None = None
    if settings.pace != "off":
        planned = plan_pace(duration, clip_silences, rel_words, settings.pace)
        if len(planned) > 1 or (planned and planned[0].duration < duration - 1e-3):
            keeps = snap_keeps(planned, duration, fps)

    main = keeps or [Keep(0.0, duration)]
    out_words = retime_words(rel_display, main)
    out_duration = output_duration(main)
    keywords = pick_keywords(out_words) if settings.keyword_highlight else []

    teaser_keep: Keep | None = None
    wants_teaser = settings.flash_forward if flash_forward is None else flash_forward
    if wants_teaser and duration >= TEASER_MIN_CLIP_S:
        if teaser is not None:
            teaser_keep = fit_teaser(
                (teaser[0] - start_s, teaser[1] - start_s), duration, clip_silences
            )
        if teaser_keep is None:
            teaser_keep = pick_teaser(rel_words, duration, clip_silences, keywords)
        if teaser_keep is not None:
            # On the frame grid, so the tease and the story join without the
            # sound slipping against the picture; widened, never narrowed, so
            # the line's last word is not clipped.
            a = math.floor(teaser_keep.start * fps + 1e-6) / fps
            b = min(math.ceil(teaser_keep.end * fps - 1e-6), int(duration * fps + 1e-6)) / fps
            teaser_keep = Keep(round(a, 4), round(b, 4)) if b - a >= MIN_KEEP_S else None

    zooms: list[Zoom] = []
    intro = False
    if allow_motion:
        zooms = plan_zooms(out_words, out_duration, settings.motion, cut_times(main), motion_seed)
        intro = settings.motion == "dynamic" and out_duration >= 4.0 and teaser_keep is None

    hook_start = 0.0
    if teaser_keep is not None:
        lead = teaser_keep.duration
        hook_start = round(lead, 4)
        if keeps is None:
            # An uncut clip still needs a frame-exact second piece to join to.
            main = snap_keeps(main, duration, fps)
        keeps = [teaser_keep, *main]
        if allow_motion:
            shifted = [Zoom(round(z.t + lead, 3), z.z) for z in zooms]
            if not shifted or shifted[0].t > lead + 1e-6:
                shifted.insert(0, Zoom(round(lead, 3), 1.0))
            zooms = [Zoom(0.0, TEASER_ZOOM), *shifted]

    music = resolve_music(settings.music_path, seed) if settings.music_enabled else None
    return EditPlan(
        keeps=keeps,
        zooms=zooms,
        intro_zoom=intro,
        hook_text=hook_text(title) if settings.hook_title else "",
        keywords=keywords,
        teaser=teaser_keep,
        hook_start_s=hook_start,
        watermark=watermark if watermark is not None else resolve_watermark(settings),
        music=music,
        music_volume=settings.music_volume / 100.0,
        music_duck=settings.music_duck,
        fps=fps,
        two_pass_loudnorm=settings.two_pass_loudnorm,
    )
