"""Caption generation — word timings to burned-in ASS subtitles.

Word grouping decides readability; ASS styling decides whether it looks like a
real short or like a subtitle track. Four presets cover the range from
attention-grabbing to professional.

Per-word animation (Bold Pop, Boxed) is done by emitting one Dialogue event per
word, each showing the whole caption group with the active word overridden.
That's how libass does word highlighting — there's no per-glyph timeline — and
it's why event counts look high relative to word counts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pysubs2

from .transcript import Word

log = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

#: Words are grouped into caption events; more than this is hard to read at a
#: glance on a phone.
DEFAULT_MAX_WORDS = 4
#: A pause longer than this ends a caption group regardless of word count.
DEFAULT_MAX_GAP_S = 0.4

#: Reference canvas. Styles are authored against this and scaled to the real
#: output size, so a preset looks identical at 9:16, 1:1, and 16:9.
REFERENCE_HEIGHT = 1920


def hex_to_ass(colour: str, alpha: int = 0) -> pysubs2.Color:
    """Convert ``#RRGGBB`` to a pysubs2 colour.

    ASS stores colours as ``&HAABBGGRR`` — alpha first, then reversed RGB.
    pysubs2 handles the byte order; we only translate from hex.
    """
    r, g, b = _rgb(colour)
    return pysubs2.Color(r, g, b, alpha)


def _rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB, got {colour!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def ass_colour_override(colour: str) -> str:
    r"""Format ``#RRGGBB`` as an inline ASS colour value, e.g. ``&H00E5FF&``.

    Inline ``\c`` overrides take the raw ``&HBBGGRR&`` form — reversed byte
    order and no alpha component, unlike the style-level colour fields.
    """
    r, g, b = _rgb(colour)
    return f"&H{b:02X}{g:02X}{r:02X}&"


@dataclass
class CaptionStyle:
    """A caption preset."""

    key: str
    label: str
    description: str

    font: str = "Anton"
    font_file: str = "Anton-Regular.ttf"
    #: Font size as a fraction of frame height, so presets scale across ratios.
    size_ratio: float = 0.058
    primary: str = "#FFFFFF"
    #: Colour of the currently-spoken word. None disables per-word highlighting.
    accent: str | None = "#FFE500"
    outline: str = "#000000"
    outline_width: float = 4.0
    shadow: float = 0.0
    bold: bool = True
    all_caps: bool = True
    #: Distance from the bottom of the frame, as a fraction of height.
    margin_v_ratio: float = 0.24
    max_words: int = DEFAULT_MAX_WORDS
    #: "none" | "scale" | "karaoke"
    animation: str = "scale"
    #: Percentage the active word grows to. Only used when animation is "scale".
    scale_percent: int = 118
    #: Opaque background box behind the text (ASS BorderStyle 3).
    boxed: bool = False
    box_colour: str = "#000000"
    box_alpha: int = 40
    #: Colour and size of a key word (see :func:`build_ass` ``keywords``).
    emphasis: str = "#7CFF55"
    emphasis_scale: int = 122
    #: Outline colour of the spoken word (neon glow). None keeps the outline.
    active_outline: str | None = None
    #: Edge blur applied to the whole line (a glow with a coloured outline).
    blur: float = 0.0

    @property
    def font_path(self) -> Path:
        return FONT_DIR / self.font_file


PRESETS: dict[str, CaptionStyle] = {
    "bold_pop": CaptionStyle(
        key="bold_pop",
        label="Bold Pop",
        description="Chunky white with a heavy outline; the spoken word grows and turns yellow.",
        font="Anton",
        font_file="Anton-Regular.ttf",
        size_ratio=0.062,
        primary="#FFFFFF",
        accent="#FFE500",
        outline_width=5.0,
        all_caps=True,
        animation="scale",
        scale_percent=118,
        max_words=4,
        emphasis="#7CFF55",
    ),
    "karaoke_fill": CaptionStyle(
        key="karaoke_fill",
        label="Karaoke Fill",
        description="Words fill with colour exactly as they're spoken.",
        font="Anton",
        font_file="Anton-Regular.ttf",
        size_ratio=0.058,
        primary="#FFFFFF",
        accent="#31E981",
        outline_width=4.0,
        all_caps=True,
        animation="karaoke",
        max_words=5,
        emphasis="#FFE500",
    ),
    "clean_lower": CaptionStyle(
        key="clean_lower",
        label="Clean Lower",
        description="Minimal lower third, no animation. For talks and professional cuts.",
        font="Inter",
        font_file="Inter-Variable.ttf",
        size_ratio=0.034,
        primary="#FFFFFF",
        accent=None,
        outline_width=2.0,
        shadow=1.0,
        bold=False,
        all_caps=False,
        margin_v_ratio=0.10,
        animation="none",
        max_words=8,
        emphasis="#FFD54A",
        emphasis_scale=112,
    ),
    "boxed": CaptionStyle(
        key="boxed",
        label="Boxed",
        description="High-contrast text on a solid block. Readable on any footage.",
        font="Anton",
        font_file="Anton-Regular.ttf",
        size_ratio=0.052,
        primary="#FFFFFF",
        accent="#FF4D4D",
        outline_width=0.0,
        all_caps=True,
        animation="scale",
        scale_percent=112,
        max_words=4,
        boxed=True,
        box_colour="#000000",
        box_alpha=30,
        emphasis="#FFE500",
    ),
    # CrestClip-style presets.
    "pop": CaptionStyle(
        key="pop",
        label="Pop",
        description="Big three-word bursts; the spoken word turns gold, key words green.",
        font="Anton",
        font_file="Anton-Regular.ttf",
        size_ratio=0.061,
        primary="#FFFFFF",
        accent="#FFDD00",
        outline_width=7.0,
        shadow=3.0,
        all_caps=True,
        animation="scale",
        scale_percent=112,
        max_words=3,
        emphasis="#7CFF55",
        emphasis_scale=124,
    ),
    "neon": CaptionStyle(
        key="neon",
        label="Neon",
        description="Glowing cyan outline; the spoken word lights up mint, key words magenta.",
        font="Poppins ExtraBold",
        font_file="Poppins-ExtraBold.ttf",
        size_ratio=0.050,
        primary="#FFFFFF",
        accent="#7CFFB2",
        outline="#00E5FF",
        outline_width=3.0,
        bold=False,
        all_caps=True,
        animation="scale",
        scale_percent=100,
        max_words=3,
        emphasis="#FF4DFF",
        emphasis_scale=118,
        active_outline="#7CFFB2",
        blur=4.0,
    ),
    "comic": CaptionStyle(
        key="comic",
        label="Comic",
        description="Yellow comic-book lettering, two words at a time; key words in red.",
        font="Bangers",
        font_file="Bangers-Regular.ttf",
        size_ratio=0.0625,
        primary="#FFE600",
        accent="#FFFFFF",
        outline_width=9.0,
        shadow=5.0,
        bold=False,
        all_caps=True,
        animation="scale",
        scale_percent=118,
        max_words=2,
        emphasis="#FF3838",
        emphasis_scale=126,
    ),
    "minimal": CaptionStyle(
        key="minimal",
        label="Minimal",
        description="Clean sentence-case lines with a soft shadow; only key words take colour.",
        font="Poppins",
        font_file="Poppins-Bold.ttf",
        size_ratio=0.0375,
        primary="#FFFFFF",
        accent=None,
        outline_width=0.0,
        shadow=3.0,
        bold=True,
        all_caps=False,
        margin_v_ratio=0.20,
        animation="none",
        max_words=5,
        box_alpha=0x99,
        emphasis="#FFD700",
        emphasis_scale=100,
    ),
}

DEFAULT_STYLE = "bold_pop"


def get_style(key: str) -> CaptionStyle:
    style = PRESETS.get(key)
    if style is None:
        raise ValueError(f"Unknown caption style {key!r}. Available: {', '.join(PRESETS)}")
    return style


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


@dataclass
class CaptionGroup:
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text.strip() for w in self.words)


def group_words(
    words: list[Word],
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> list[CaptionGroup]:
    """Split words into caption-sized groups.

    Breaks on three signals, in priority order: sentence-ending punctuation, a
    pause longer than ``max_gap_s``, and the word-count ceiling. Punctuation
    first means a caption rarely spans two sentences, which reads badly even
    when the timing is correct.
    """
    groups: list[CaptionGroup] = []
    current: list[Word] = []

    for index, word in enumerate(words):
        if current:
            gap = word.start - current[-1].end
            if gap > max_gap_s or len(current) >= max_words:
                groups.append(CaptionGroup(words=current))
                current = []

        current.append(word)

        is_last = index == len(words) - 1
        if word.ends_sentence and not is_last:
            groups.append(CaptionGroup(words=current))
            current = []

    if current:
        groups.append(CaptionGroup(words=current))

    return [g for g in groups if g.words]


# --------------------------------------------------------------------------
# ASS generation
# --------------------------------------------------------------------------

STYLE_NAME = "AutoClip"


def build_ass(
    words: list[Word],
    style: CaptionStyle,
    *,
    width: int,
    height: int,
    time_offset_s: float = 0.0,
    margin_v_ratio: float | None = None,
    keywords: Sequence[str] = (),
    hook_text: str = "",
    hook_s: float = 3.0,
    hook_top_px: int | None = None,
    duration_s: float | None = None,
) -> pysubs2.SSAFile:
    """Build an ASS subtitle file for a clip.

    ``time_offset_s`` is subtracted from every timestamp, so absolute transcript
    times become clip-relative ones.

    ``keywords`` (normalised, see :func:`edit.pick_keywords`) are drawn in the
    style's emphasis colour and size: at most one per caption line, and each
    key word at most twice per clip. ``hook_text`` adds a headline at the top
    for the first ``hook_s`` seconds (clip-relative, never past ``duration_s``).

    Preconditions:
        words are in chronological order and carry start/end timings.
    """
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(width)
    subs.info["PlayResY"] = str(height)
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["WrapStyle"] = "0"

    scale = height / REFERENCE_HEIGHT
    subs.styles[STYLE_NAME] = _build_ass_style(style, height=height, scale=scale)
    if margin_v_ratio is not None:
        # Layouts with a panel the captions must stay clear of (the split
        # layout's facecam) override the preset's distance from the bottom.
        subs.styles[STYLE_NAME].marginv = round(height * margin_v_ratio)

    groups = group_words(words, max_words=style.max_words)
    used: dict[str, int] = {}

    for group in groups:
        emph = _emphasis_index(group, keywords, used)
        if style.animation == "karaoke":
            subs.events.append(_karaoke_event(group, style, time_offset_s, emph))
        elif style.animation == "scale" and style.accent:
            subs.events.extend(_per_word_events(group, style, time_offset_s, emph))
        else:
            subs.events.append(_static_event(group, style, time_offset_s, emph))

    if hook_text.strip():
        _add_hook(
            subs,
            hook_text,
            width=width,
            height=height,
            hook_s=hook_s,
            top_px=hook_top_px,
            duration_s=duration_s,
        )

    return subs


HOOK_STYLE_NAME = "AutoClipHook"
#: Headline look (CrestClip's): dark text on a white box, top centre.
HOOK_FONT = "Poppins ExtraBold"
HOOK_SIZE_RATIO = 0.0365
HOOK_TEXT = "#1E1B2B"
HOOK_BOX = "#FFFFFF"


def _add_hook(
    subs: pysubs2.SSAFile,
    text: str,
    *,
    width: int,
    height: int,
    hook_s: float,
    top_px: int | None,
    duration_s: float | None,
) -> None:
    hook = pysubs2.SSAStyle()
    hook.fontname = HOOK_FONT
    hook.fontsize = round(height * HOOK_SIZE_RATIO)
    hook.primarycolor = hex_to_ass(HOOK_TEXT)
    hook.secondarycolor = hex_to_ass(HOOK_TEXT)
    hook.outlinecolor = hex_to_ass(HOOK_BOX)
    hook.backcolor = hex_to_ass(HOOK_BOX)
    hook.bold = False
    # BorderStyle 3: an opaque box; the outline width is the box padding.
    hook.borderstyle = 3
    hook.outline = round(height * 0.0115)
    hook.shadow = 0
    hook.alignment = pysubs2.Alignment.TOP_CENTER
    hook.marginl = hook.marginr = round(width * 0.1)
    subs.styles[HOOK_STYLE_NAME] = hook

    end = hook_s + 0.2
    if duration_s is not None:
        end = min(end, max(1.5, duration_s - 0.2))
    top = top_px if top_px is not None else round(height * 0.156)
    body = escape_ass_text(text.strip())
    subs.events.append(
        pysubs2.SSAEvent(
            start=0,
            end=pysubs2.make_time(s=end),
            style=HOOK_STYLE_NAME,
            layer=1,
            text=f"{{\\an8\\pos({width // 2},{top})\\fad(120,220)}}{body}",
        )
    )


def _emphasis_index(group: CaptionGroup, keywords: Sequence[str], used: dict[str, int]) -> int:
    """Which word of ``group`` to draw as a key word (-1 for none)."""
    if not keywords:
        return -1
    from .edit import keyword_of

    for index, word in enumerate(group.words):
        keyword = keyword_of(word.text, keywords)
        if keyword is None or used.get(keyword, 0) >= 2:
            continue
        used[keyword] = used.get(keyword, 0) + 1
        return index
    return -1


def _emphasis_tags(style: CaptionStyle) -> str:
    scale = style.emphasis_scale
    tags = f"\\c{ass_colour_override(style.emphasis)}"
    if scale != 100:
        tags += f"\\fscx{scale}\\fscy{scale}"
    return tags


def _line_prefix(style: CaptionStyle) -> str:
    return f"{{\\blur{style.blur:g}}}" if style.blur else ""


def _reset(style: CaptionStyle) -> str:
    return f"{{\\r\\blur{style.blur:g}}}" if style.blur else "{\\r}"


def _build_ass_style(style: CaptionStyle, *, height: int, scale: float) -> pysubs2.SSAStyle:
    ass_style = pysubs2.SSAStyle()
    ass_style.fontname = style.font
    ass_style.fontsize = round(height * style.size_ratio)
    ass_style.primarycolor = hex_to_ass(style.primary)
    ass_style.secondarycolor = hex_to_ass(style.accent or style.primary)
    ass_style.outlinecolor = hex_to_ass(style.outline)
    ass_style.backcolor = hex_to_ass(style.box_colour, alpha=style.box_alpha)
    ass_style.bold = style.bold
    ass_style.outline = style.outline_width * scale
    ass_style.shadow = style.shadow * scale
    # BorderStyle 3 draws an opaque box using backcolor instead of an outline.
    ass_style.borderstyle = 3 if style.boxed else 1
    ass_style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    ass_style.marginv = round(height * style.margin_v_ratio)
    ass_style.marginl = ass_style.marginr = round(height * 0.03)
    return ass_style


#: U+2060 WORD JOINER: default-ignorable, so libass (shaping through HarfBuzz)
#: draws nothing for it, whatever the font. See :func:`escape_ass_text`.
_INERT = "⁠"


def escape_ass_text(text: str) -> str:
    r"""Make transcript text render literally inside an ASS Dialogue line.

    Event text is ASS markup, not plain text: ``{`` opens an override block,
    and ``\N`` / ``\n`` / ``\h`` are a line break, a soft break and a hard
    space. A word such as ``{laughs}`` vanished from the burned caption, a
    backslash before n/N/h broke the line, and a word ending in a backslash
    paired with the ``{`` of the highlight tag after it (``\{`` is an escape),
    so the tag itself was drawn on screen as text.

    ASS has no general escape, so two mechanisms are combined, both checked
    against real libass renders: an invisible word joiner after every
    backslash, which stops it pairing with whatever follows; then libass's own
    ``\{`` and ``\}`` escapes, which draw identically to bare braces. Real
    newlines and tabs become spaces — pysubs2 writes event text raw, and a
    newline would end the Dialogue line mid-caption.
    """
    text = " ".join(text.split())
    text = text.replace("\\", "\\" + _INERT)
    return text.replace("{", "\\{").replace("}", "\\}")


def _text_of(word: Word, style: CaptionStyle) -> str:
    text = word.text.strip()
    # Escape after upper-casing: upper() would turn an escaped "\n" into "\N".
    return escape_ass_text(text.upper() if style.all_caps else text)


def _event(start_s: float, end_s: float, text: str, offset: float) -> pysubs2.SSAEvent:
    return pysubs2.SSAEvent(
        start=pysubs2.make_time(s=max(0.0, start_s - offset)),
        end=pysubs2.make_time(s=max(0.0, end_s - offset)),
        text=text,
        style=STYLE_NAME,
    )


def _static_event(
    group: CaptionGroup, style: CaptionStyle, offset: float, emph: int = -1
) -> pysubs2.SSAEvent:
    parts: list[str] = []
    for index, word in enumerate(group.words):
        text = _text_of(word, style)
        if index == emph:
            text = f"{{{_emphasis_tags(style)}}}{text}{_reset(style)}"
        parts.append(text)
    return _event(group.start, group.end, _line_prefix(style) + " ".join(parts), offset)


def _karaoke_event(
    group: CaptionGroup, style: CaptionStyle, offset: float, emph: int = -1
) -> pysubs2.SSAEvent:
    r"""Build a ``\kf`` karaoke line.

    ``\kf`` sweeps the fill across each word over its own duration; durations
    are in centiseconds, which is the unit ASS karaoke tags use.
    """
    parts: list[str] = []
    for index, word in enumerate(group.words):
        duration_cs = max(1, round(word.duration * 100))
        # Roll the inter-word gap into the preceding word so the sweep stays
        # locked to the audio instead of drifting late across the line.
        if index + 1 < len(group.words):
            gap = group.words[index + 1].start - word.end
            if 0 < gap < 0.5:
                duration_cs += round(gap * 100)
        if index == emph:
            parts.append(
                f"{{\\kf{duration_cs}{_emphasis_tags(style)}}}{_text_of(word, style)}"
                f"{_reset(style)}"
            )
        else:
            parts.append(f"{{\\kf{duration_cs}}}{_text_of(word, style)}")

    return _event(group.start, group.end, _line_prefix(style) + " ".join(parts), offset)


def _per_word_events(
    group: CaptionGroup, style: CaptionStyle, offset: float, emph: int = -1
) -> list[pysubs2.SSAEvent]:
    """Emit one event per word, highlighting the active one.

    Every event renders the full group so the line stays stable on screen; only
    the emphasised word changes between them.
    """
    accent_tag = f"\\c{ass_colour_override(style.accent or style.primary)}"
    if style.active_outline:
        accent_tag += f"\\3c{ass_colour_override(style.active_outline)}"
    scale = style.scale_percent
    if scale != 100:
        accent_tag += f"\\fscx{scale}\\fscy{scale}"
    emphasis_tags = _emphasis_tags(style)
    reset = _reset(style)
    events: list[pysubs2.SSAEvent] = []

    for active, word in enumerate(group.words):
        rendered: list[str] = []
        for index, other in enumerate(group.words):
            text = _text_of(other, style)
            # A key word keeps its emphasis while spoken (its tags come last).
            tags = (accent_tag if index == active else "") + (
                emphasis_tags if index == emph else ""
            )
            rendered.append(f"{{{tags}}}{text}{reset}" if tags else text)

        # Hold the last word until the group ends so the caption doesn't blink
        # out early when a word's measured end precedes the next word's start.
        end = word.end if active + 1 < len(group.words) else group.end
        next_start = group.words[active + 1].start if active + 1 < len(group.words) else end
        events.append(
            _event(
                word.start,
                max(end, next_start),
                _line_prefix(style) + " ".join(rendered),
                offset,
            )
        )

    return events


def write_ass(
    path: Path,
    words: list[Word],
    style: CaptionStyle,
    *,
    width: int,
    height: int,
    time_offset_s: float = 0.0,
    margin_v_ratio: float | None = None,
    keywords: Sequence[str] = (),
    hook_text: str = "",
    hook_s: float = 3.0,
    hook_top_px: int | None = None,
    duration_s: float | None = None,
) -> Path:
    """Render captions to an .ass file and return its path.

    ``margin_v_ratio`` overrides the style's bottom margin (fraction of height);
    the key-word and hook options are as for :func:`build_ass`.
    """
    subs = build_ass(
        words,
        style,
        width=width,
        height=height,
        time_offset_s=time_offset_s,
        margin_v_ratio=margin_v_ratio,
        keywords=keywords,
        hook_text=hook_text,
        hook_s=hook_s,
        hook_top_px=hook_top_px,
        duration_s=duration_s,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(path), encoding="utf-8")
    return path


def write_srt(path: Path, words: list[Word], *, time_offset_s: float = 0.0) -> Path:
    """Write a plain .srt sidecar — for platforms that want an upload-time file."""
    subs = pysubs2.SSAFile()
    for group in group_words(words, max_words=8):
        subs.events.append(
            pysubs2.SSAEvent(
                start=pysubs2.make_time(s=max(0.0, group.start - time_offset_s)),
                end=pysubs2.make_time(s=max(0.0, group.end - time_offset_s)),
                # pysubs2 treats event text as ASS markup when writing SRT: it
                # strips {...} as override tags and turns \N, \n and \h into
                # breaks and spaces. The joiner after each backslash keeps
                # those literal; keep_ssa_tags below keeps the braces.
                text=" ".join(group.text.split()).replace("\\", "\\" + _INERT),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(path), encoding="utf-8", format_="srt", keep_ssa_tags=True)
    return path
