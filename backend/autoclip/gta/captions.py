"""GTA captions: the bottom-band word captions, in a choice of looks.

Builds on :mod:`autoclip.pipeline.captions` (its presets, fonts and per-word
event builders) with the two GTA rules the normal builder does not know:

* a caption is never more than two lines: a group whose text would wrap to a
  third line in the bottom band is split into shorter groups, measured with
  the style's real font;
* the captions sit just above the bottom safe zone at a fixed pixel margin,
  whatever the preset's own distance from the bottom.

Key words (see :func:`autoclip.pipeline.edit.pick_keywords`) are drawn bigger
in the style's emphasis colour, at most one per caption group (so at most one
per line) and each at most twice per clip.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

import pysubs2

from ..pipeline import captions
from ..pipeline.captions import (
    STYLE_NAME,
    CaptionGroup,
    CaptionStyle,
    _build_ass_style,
    _emphasis_index,
    _karaoke_event,
    _per_word_events,
    _static_event,
)
from ..pipeline.edit import pick_keywords
from ..pipeline.transcript import Word
from .settings import CAPTION_ALIASES, CAPTION_STYLES

log = logging.getLogger(__name__)

#: Most lines one caption may take in the bottom band.
MAX_LINES = 2
#: Safety factor on measured widths: libass synthesises bold for fonts that
#: have none (Anton), and the spoken word grows while it is highlighted.
WIDTH_SLACK = 1.06


def resolve_style(key: str) -> CaptionStyle:
    """The caption preset for a GTA look (``gta``, ``pop``...) or preset key."""
    look = CAPTION_ALIASES.get(key.strip().lower(), key.strip().lower())
    preset = CAPTION_STYLES.get(look, look)
    return captions.get_style(preset)


def font_size_px(style: CaptionStyle, height: int) -> int:
    return round(height * style.size_ratio)


def side_margin_px(height: int) -> int:
    """Left/right margin the caption style uses (see ``_build_ass_style``)."""
    return round(height * 0.03)


@lru_cache(maxsize=16)
def _font(path: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


def _display(word: Word, style: CaptionStyle) -> str:
    text = word.text.strip()
    return text.upper() if style.all_caps else text


def _grow(style: CaptionStyle, keywords: Sequence[str]) -> float:
    """How much bigger the widest word may be drawn than its plain size."""
    grow = 1.0
    if style.animation == "scale" and style.accent:
        grow = max(grow, style.scale_percent / 100.0)
    if keywords:
        grow = max(grow, style.emphasis_scale / 100.0)
    return grow


def line_count(
    group_words: Sequence[Word],
    style: CaptionStyle,
    *,
    width: int,
    height: int,
    keywords: Sequence[str] = (),
) -> int:
    """Lines ``group_words`` wrap to in the caption box (greedy, conservative).

    Measured at the ASS font size in em units with PIL, which is at least as
    wide as libass draws it; returns a large number when the font can't be
    read, so the caller splits rather than overflows.
    """
    size = font_size_px(style, height)
    try:
        font = _font(str(style.font_path), size)
    except OSError:
        log.warning("Caption font %s unreadable; splitting captions short.", style.font_path)
        return MAX_LINES + len(group_words)
    budget = width - 2 * side_margin_px(height)
    outline = 2 * style.outline_width * (height / captions.REFERENCE_HEIGHT)
    space = font.getlength(" ")
    grow = _grow(style, keywords)
    lines = 1
    used = 0.0
    for word in group_words:
        w = font.getlength(_display(word, style)) * WIDTH_SLACK * grow + outline
        if used == 0.0:
            used = w
        elif used + space + w <= budget:
            used += space + w
        else:
            lines += 1
            used = w
    return lines


def fit_groups(
    groups: Sequence[CaptionGroup],
    style: CaptionStyle,
    *,
    width: int,
    height: int,
    keywords: Sequence[str] = (),
) -> list[CaptionGroup]:
    """Split every group that would need more than :data:`MAX_LINES` lines."""
    out: list[CaptionGroup] = []
    pending = list(groups)
    while pending:
        group = pending.pop(0)
        too_long = (
            len(group.words) > 1
            and line_count(group.words, style, width=width, height=height, keywords=keywords)
            > MAX_LINES
        )
        if not too_long:
            out.append(group)
            continue
        half = (len(group.words) + 1) // 2
        pending[:0] = [
            CaptionGroup(words=group.words[:half]),
            CaptionGroup(words=group.words[half:]),
        ]
    return out


def build_ass(
    words: list[Word],
    style: CaptionStyle,
    *,
    width: int,
    height: int,
    margin_v_px: int,
    keywords: Sequence[str] = (),
) -> pysubs2.SSAFile:
    """Captions for one GTA clip (clip-relative word times)."""
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(width)
    subs.info["PlayResY"] = str(height)
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["WrapStyle"] = "0"
    ass_style = _build_ass_style(style, height=height, scale=height / captions.REFERENCE_HEIGHT)
    ass_style.marginv = margin_v_px
    subs.styles[STYLE_NAME] = ass_style

    groups = fit_groups(
        captions.group_words(words, max_words=style.max_words),
        style,
        width=width,
        height=height,
        keywords=keywords,
    )
    used: dict[str, int] = {}
    for group in groups:
        emph = _emphasis_index(group, keywords, used)
        if style.animation == "karaoke":
            subs.events.append(_karaoke_event(group, style, 0.0, emph))
        elif style.animation == "scale" and style.accent:
            subs.events.extend(_per_word_events(group, style, 0.0, emph))
        else:
            subs.events.append(_static_event(group, style, 0.0, emph))
    return subs


def clip_keywords(words: Sequence[Word], enabled: bool) -> list[str]:
    """The 3-8 key words of a clip (none when highlighting is off)."""
    return pick_keywords(words) if enabled and words else []
