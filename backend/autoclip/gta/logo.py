"""Text logo for the GTA Mode watermark ("GTA CLIPGOD" in Lemon Milk by default).

When no logo PNG is set, the watermark text is drawn once into a transparent
PNG and cached under ``<AutoClip home>/gta/logo``; the renderer then overlays
that PNG exactly like a user-supplied logo. The first word is drawn in the
reaction-text yellow and the rest in white, both with a black outline so the
logo reads on any footage.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"
#: Lemon Milk Bold by Marsnev (donationware; see assets/LEMONMILK-READ_ME.txt).
BUNDLED_FONT = ASSETS / "LEMONMILK-Bold.otf"

#: Drawn large and scaled down by the renderer, so edges stay crisp.
FONT_PX = 160
STROKE_PX = 12
PAD_PX = 16
ACCENT = (255, 229, 0, 255)  # same yellow as the reaction text (#FFE500)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)
#: Bump when the drawing changes so old cached PNGs aren't reused.
STYLE_VERSION = "1"


class LogoError(RuntimeError):
    """The text logo could not be drawn (missing font, empty text...)."""


def font_path(custom: str = "") -> Path:
    """The font to draw with: the user's file if given, else bundled Lemon Milk."""
    raw = custom.strip().strip('"')
    path = Path(raw).expanduser() if raw else BUNDLED_FONT
    if not path.is_file():
        raise LogoError(f"Logo font not found: {path}")
    return path


def _cache_name(text: str, font: Path) -> str:
    stat = font.stat()
    key = f"{STYLE_VERSION}|{text}|{font}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".png"


def draw(text: str, font: Path) -> Any:
    """Draw ``text`` (uppercased) on a tight transparent canvas (a PIL image)."""
    from PIL import Image, ImageDraw, ImageFont

    words = text.upper().split()
    if not words:
        raise LogoError("Logo text is empty.")
    face = ImageFont.truetype(str(font), FONT_PX)
    parts = [(words[0], ACCENT)]
    if len(words) > 1:
        parts.append((" " + " ".join(words[1:]), WHITE))

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    full = "".join(p for p, _ in parts)
    left, top, right, bottom = probe.textbbox((0, 0), full, font=face, stroke_width=STROKE_PX)
    image = Image.new("RGBA", (right - left + 2 * PAD_PX, bottom - top + 2 * PAD_PX), (0, 0, 0, 0))
    pen = ImageDraw.Draw(image)
    x = float(PAD_PX - left)
    y = PAD_PX - top
    for part, colour in parts:
        pen.text((x, y), part, font=face, fill=colour, stroke_width=STROKE_PX, stroke_fill=BLACK)
        # Advance by the plain advance width so the two colours join exactly
        # where one continuous string would.
        x += pen.textlength(part, font=face)
    return image


def text_logo(text: str, font: str = "", cache_dir: Path | None = None) -> Path:
    """Path to a cached PNG of ``text`` in ``font`` (drawn on first use)."""
    text = " ".join(text.split())
    if not text:
        raise LogoError("Logo text is empty.")
    font_file = font_path(font)
    if cache_dir is None:
        from .settings import work_root

        cache_dir = work_root() / "logo"
    target = Path(cache_dir) / _cache_name(text, font_file)
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = draw(text, font_file)
    except OSError as exc:  # unreadable / not a font file
        raise LogoError(f"Could not read logo font {font_file}: {exc}") from exc
    partial = target.with_name(f"{target.stem}.{os.getpid()}.partial.png")
    image.save(partial, format="PNG")
    os.replace(partial, target)
    log.info("Drew text logo %r with %s -> %s", text, font_file.name, target)
    return target
