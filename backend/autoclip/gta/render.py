"""GTA Mode rendering: dead air cut, uncropped frame, hook title, captions.

Layout (9:16, 1080x1920): the WHOLE stream frame — gameplay and the
streamer's own face-cam overlay exactly as broadcast, nothing cropped — sits
in the middle, scaled to the full width, over a blurred, zoomed copy of
itself. The hook title lives in the top band for the first seconds, the
word-by-word captions in the bottom band, so neither ever covers the stream.

Loudness is normalised in two passes (measure, then apply linearly) so every
clip lands on the Shorts/TikTok target instead of near it.

Flash-forward (optional): the clip opens on its 1-3 s peak reaction, punched
in, then plays from its start; the hook title follows the tease.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ExportSettings
from ..pipeline import captions, ffmpeg
from ..pipeline import edit as shared_edit
from ..pipeline.edit import Keep, loudnorm_filter, output_duration, remap_words
from ..pipeline.export import encoder_args
from ..pipeline.transcript import Word
from . import captions as gta_captions

__all__ = ["Keep", "loudnorm_filter", "output_duration", "remap_words"]

log = logging.getLogger(__name__)

OUT_W = 1080
OUT_H = 1920
FPS = 30

#: Safe zones: the app UI covers about this much of the top and bottom, so
#: no text (and no logo) is ever placed there.
SAFE_TOP_PX = 150
SAFE_BOTTOM_PX = 300
#: Captions sit in the bottom band, their baseline just above the bottom
#: safe zone (1-2 lines grow upwards from there).
CAPTION_MARGIN_PX = SAFE_BOTTOM_PX + 30
CAPTION_MARGIN_RATIO = CAPTION_MARGIN_PX / OUT_H
#: Hook title and reaction text: centred in the top band, below the logo row.
TITLE_FONT_RATIO = 0.052
TITLE_STYLE = "GtaHook"
REACTION_STYLE = "GtaReaction"
TEXT_CENTRE_Y = 430
#: Gap between the bottom caption box and the subtitles above it.
BOX_SUBTITLE_GAP_PX = 24
#: Reaction text centre when a static caption box takes the top band.
STATIC_REACTION_Y = 300
#: Logo row: just under the top safe zone.
LOGO_TOP_PX = SAFE_TOP_PX + 15
LOGO_SIDE_PX = 40


#: How far before the first kept frame the input is seeked. After ``-ss`` on
#: a live MPEG-TS window, decoding restarts at the NEXT keyframe (streams
#: send one every 2-4 s), so the first frames after the seek point are
#: missing. If the cut started inside that gap, ``setpts=PTS-STARTPTS`` pinned
#: the first frame that did arrive to t=0: the picture ran up to 2 s ahead of
#: the voice and ended early. Seeking a whole keyframe interval (plus margin)
#: early guarantees real frames exist at every cut start.
SEEK_PAD_S = 8.0
#: Find an audio track that starts late in a live window (100 s, in µs).
DEEP_PROBE = ("-analyzeduration", "100M", "-probesize", "100M")


class RenderError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Dead air
# --------------------------------------------------------------------------


def section_levels(path: Path, start_s: float, end_s: float, *, hop_s: float = 0.05) -> np.ndarray:
    """RMS level (dBFS) of every ``hop_s`` of the section's audio in a range."""
    from ..pipeline.prepare import audio_lead_s

    # Audio that starts after ``start_s`` (a live window right after a
    # reconnect) is padded, so level N is really second start_s + N.
    late = audio_lead_s(path) - max(0.0, start_s)
    pad = ["-af", f"adelay={round(late * 1000)}:all=1"] if late > 0.02 else []
    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        *DEEP_PROBE,
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-t",
        f"{max(0.05, end_s - start_s):.3f}",
        "-i",
        str(path),
        "-vn",
        *pad,
        "-t",
        f"{max(0.05, end_s - start_s):.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
    if result.returncode != 0 and b"does not contain any stream" in result.stderr:
        # No audio in this stretch (the first seconds after a reconnect).
        return np.zeros(0, dtype=np.float32)
    if result.returncode != 0:
        raise RenderError(
            f"Could not read the clip audio: {result.stderr.decode(errors='replace')[-400:]}"
        )
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(16000 * hop_s))
    count = len(samples) // hop
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    blocks = samples[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    with np.errstate(divide="ignore"):
        return np.maximum(20 * np.log10(rms), -90.0).astype(np.float32)


def plan_keeps(
    words: list[Word],
    start_s: float,
    end_s: float,
    *,
    levels: np.ndarray | None,
    hop_s: float = 0.05,
    min_gap_s: float = 0.6,
    keep_s: float = 0.12,
) -> list[Keep]:
    """Spans to keep between ``start_s`` and ``end_s`` (dead air removed).

    A gap is dead air only when NOBODY speaks in it (no transcribed word)
    AND the audio is actually quiet. Screams and big game moments are often
    not transcribed at all, so a word gap alone must never be cut.
    """
    spans = sorted(
        (max(start_s, w.start), min(end_s, w.end))
        for w in words
        if w.end > start_s and w.start < end_s
    )
    gaps: list[tuple[float, float]] = []
    cursor = start_s
    for a, b in spans:
        if a - cursor >= min_gap_s:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if end_s - cursor >= min_gap_s:
        gaps.append((cursor, end_s))

    speech_db = _speech_level(spans, start_s, levels, hop_s)
    threshold = None if speech_db is None else max(-50.0, speech_db - 16.0)

    cuts: list[tuple[float, float]] = []
    for a, b in gaps:
        if threshold is None or levels is None:
            break
        quiet = _quiet_runs(levels, a - start_s, b - start_s, hop_s, threshold, min_gap_s)
        for qa, qb in quiet:
            cut_a = start_s + qa + keep_s
            cut_b = start_s + qb - keep_s
            # Never cut the very first/last moments into nothing: the clip's
            # own edges keep their small breath of silence too.
            if cut_b - cut_a >= min_gap_s - 2 * keep_s and cut_b > cut_a:
                cuts.append((cut_a, cut_b))

    keeps: list[Keep] = []
    cursor = start_s
    for a, b in sorted(cuts):
        if a > cursor:
            keeps.append(Keep(cursor, a))
        cursor = max(cursor, b)
    if end_s > cursor:
        keeps.append(Keep(cursor, end_s))
    # Drop slivers a concat would only stutter on.
    keeps = [k for k in keeps if k.duration >= 0.08]
    return keeps or [Keep(start_s, end_s)]


def _speech_level(spans, start_s: float, levels: np.ndarray | None, hop_s: float) -> float | None:
    if levels is None or len(levels) == 0 or not spans:
        return None
    values: list[np.ndarray] = []
    for a, b in spans:
        i = max(0, int((a - start_s) / hop_s))
        j = min(len(levels), int(np.ceil((b - start_s) / hop_s)))
        if j > i:
            values.append(levels[i:j])
    if not values:
        return None
    return float(np.median(np.concatenate(values)))


def _quiet_runs(
    levels: np.ndarray, a: float, b: float, hop_s: float, threshold: float, min_len_s: float
) -> list[tuple[float, float]]:
    i = max(0, int(np.ceil(a / hop_s)))
    j = min(len(levels), int(b / hop_s))
    runs: list[tuple[float, float]] = []
    run_start: int | None = None
    for k in range(i, j):
        if levels[k] <= threshold:
            if run_start is None:
                run_start = k
        else:
            if run_start is not None and (k - run_start) * hop_s >= min_len_s:
                runs.append((run_start * hop_s, k * hop_s))
            run_start = None
    if run_start is not None and (j - run_start) * hop_s >= min_len_s:
        runs.append((run_start * hop_s, j * hop_s))
    return runs


# --------------------------------------------------------------------------
# Subtitles (captions + hook title)
# --------------------------------------------------------------------------


def write_subtitles(
    path: Path,
    words: list[Word],
    *,
    style_key: str,
    hook_title: str,
    title_s: float,
    duration_s: float,
    reaction_text: str = "",
    reaction_window: tuple[float, float] | None = None,
    keywords: Sequence[str] = (),
    title_start_s: float = 0.0,
    text_centre_y: int = TEXT_CENTRE_Y,
    caption_margin_px: int = CAPTION_MARGIN_PX,
) -> Path:
    """Captions (bottom band, 1-2 lines), hook title and reaction (top band).

    ``style_key`` is a GTA look (``gta``, ``pop``, ``neon``, ``comic``,
    ``minimal``) or a caption preset key; ``keywords`` are drawn bigger in
    the look's emphasis colour. ``title_start_s`` delays the hook title (it
    follows a flash-forward instead of covering it).
    """
    import pysubs2

    style = gta_captions.resolve_style(style_key)
    subs = gta_captions.build_ass(
        words,
        style,
        width=OUT_W,
        height=OUT_H,
        margin_v_px=caption_margin_px,
        keywords=keywords,
    )
    title = hook_title.strip()
    text_y = text_centre_y
    title_start = max(0.0, min(title_start_s, duration_s))
    title_end = min(title_start + title_s, duration_s)
    if title and title_end - title_start < 0.3:
        title = ""
    if title:
        hook = pysubs2.SSAStyle()
        hook.fontname = "Anton"
        hook.fontsize = round(OUT_H * TITLE_FONT_RATIO)
        hook.primarycolor = captions.hex_to_ass("#FFFFFF")
        hook.outlinecolor = captions.hex_to_ass("#000000")
        hook.backcolor = captions.hex_to_ass("#000000", alpha=110)
        hook.bold = False
        hook.outline = 6.0
        hook.shadow = 2.0
        hook.borderstyle = 1
        hook.alignment = pysubs2.Alignment.MIDDLE_CENTER
        hook.marginl = hook.marginr = 70
        subs.styles[TITLE_STYLE] = hook
        text = captions.escape_ass_text(title.upper())
        event = pysubs2.SSAEvent(
            start=round(title_start * 1000),
            end=round(title_end * 1000),
            style=TITLE_STYLE,
            text=f"{{\\an5\\pos({OUT_W // 2},{text_y})\\fad(80,220)}}{text}",
        )
        subs.events.append(event)
    reaction = reaction_text.strip()
    if reaction and reaction_window is not None:
        a, b = reaction_window
        # Never on top of the hook title: start once the title is gone.
        a = max(a, title_end if title else 0.0)
        b = min(max(b, a + 1.2), duration_s)
        if b - a >= 0.6:
            style = pysubs2.SSAStyle()
            style.fontname = "Anton"
            style.fontsize = round(OUT_H * TITLE_FONT_RATIO * 0.95)
            style.primarycolor = captions.hex_to_ass("#FFE500")
            style.outlinecolor = captions.hex_to_ass("#000000")
            style.outline = 6.0
            style.shadow = 2.0
            style.borderstyle = 1
            style.alignment = pysubs2.Alignment.MIDDLE_CENTER
            style.marginl = style.marginr = 70
            subs.styles[REACTION_STYLE] = style
            body = captions.escape_ass_text(reaction.upper())
            pop = "\\t(0,140,\\fscx112\\fscy112)\\t(140,280,\\fscx100\\fscy100)"
            subs.events.append(
                pysubs2.SSAEvent(
                    start=round(a * 1000),
                    end=round(b * 1000),
                    style=REACTION_STYLE,
                    text=f"{{\\an5\\pos({OUT_W // 2},{text_y})\\fad(60,160){pop}}}{body}",
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(path), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Filtergraph
# --------------------------------------------------------------------------


def build_filtergraph(
    keeps: list[Keep],
    *,
    seek_s: float,
    subtitle_name: str | None,
    fonts_name: str,
    audio_filter: str,
    punch_in: tuple[float, float, float] | Sequence[tuple[float, float, float]] | None = None,
    watermark: Watermark | None = None,
    motion: FrameMotion | None = None,
    split: tuple[int, int, int, int, int, int] | None = None,
) -> str:
    """Cut + concat, uncropped fit over a blurred fill, subtitles, audio chain.

    ``split`` = (cam x, y, w, h, source w, source h): the webcam is cropped
    across the top and the gameplay, centre-cropped, fills the rest; punch-ins
    and motion don't apply to it.

    Keep times are section-local; the input is seeked to ``seek_s`` so the
    trims are relative to that. ``punch_in`` = (start, end, zoom) in OUTPUT
    seconds: the stream frame cuts to a centred zoom for that span; a list of
    them (same zoom) punches in on several spans (flash-forward + key beat).
    ``watermark`` reads the logo from input 1. ``motion`` zooms the stream
    frame layer only (zoompan in place of its fit-scale): the blurred
    background, the punch-in copy and the text are never moved by it.
    """
    parts: list[str] = []
    labels: list[str] = []
    for i, keep in enumerate(keeps):
        a = max(0.0, keep.start - seek_s)
        b = max(a + 0.01, keep.end - seek_s)
        parts.append(f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(
            f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS{_edge_fades(b - a)}[a{i}]"
        )
        labels.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(keeps)}:v=1:a=1[vcat][acat]")
    if split is not None:
        cam_h, cam_crop, game_crop = split_geometry(*split)
        parts.append(f"[vcat]fps={FPS},split=2[camsrc][gamesrc]")
        parts.append(f"[camsrc]crop={cam_crop},scale={OUT_W}:{cam_h}:flags=lanczos[cam]")
        parts.append(f"[gamesrc]crop={game_crop},scale={OUT_W}:{OUT_H - cam_h}:flags=lanczos[game]")
        current = (
            f"[cam][game]vstack,drawbox=x=0:y={cam_h - 3}:w={OUT_W}:h=6:color=black@0.9:t=fill"
        )
        return _finish_graph(parts, current, watermark, subtitle_name, fonts_name, audio_filter)
    spans = _punch_spans(punch_in)
    if spans:
        parts.append(f"[vcat]fps={FPS},split=3[bgsrc][fgsrc][zsrc]")
    else:
        parts.append(f"[vcat]fps={FPS},split=2[bgsrc][fgsrc]")
    parts.append(
        f"[bgsrc]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},boxblur=28:2,eq=brightness=-0.06:saturation=1.15[bg]"
    )
    if motion is not None:
        parts.append(f"[fgsrc]{motion.filter()}[fg]")
    else:
        parts.append(
            f"[fgsrc]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fg]"
        )
    current = "[bg][fg]overlay=(W-w)/2:(H-h)/2"
    if spans:
        zoom = spans[0][2]
        enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b, _ in spans)
        # Same aspect as the frame, centred: the zoomed copy fits to the same
        # size and replaces the frame for the punch-in span.
        parts.append(
            f"[zsrc]crop=trunc(iw/{zoom:.4f}/2)*2:trunc(ih/{zoom:.4f}/2)*2,"
            f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fz]"
        )
        parts.append(f"{current}[base]")
        current = f"[base][fz]overlay=(W-w)/2:(H-h)/2:enable='{enable}'"
    return _finish_graph(parts, current, watermark, subtitle_name, fonts_name, audio_filter)


#: Webcam band height limits in the 1080x1920 frame (the rest is gameplay).
SPLIT_CAM_MIN_H = 520
SPLIT_CAM_MAX_H = 760


def split_geometry(cx: int, cy: int, cw: int, ch: int, sw: int, sh: int) -> tuple[int, str, str]:
    """(cam band height, cam crop, gameplay crop) for the split layout.

    The cam crop is trimmed to the band's aspect around its centre; the
    gameplay crop is the full source height, centred, as wide as the band
    below needs (never wider than the source).
    """
    band = round(OUT_W * ch / max(1, cw))
    band = max(SPLIT_CAM_MIN_H, min(SPLIT_CAM_MAX_H, band)) // 2 * 2
    target = OUT_W / band
    if cw / max(1, ch) > target:  # cam too wide: trim its sides
        new_w = ch * target
        cx, cw = cx + (cw - new_w) / 2, new_w
    else:  # cam too tall: trim top/bottom
        new_h = cw / target
        cy, ch = cy + (ch - new_h) / 2, new_h
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    cx = max(0, min(sw - cw, int(cx)))
    cy = max(0, min(sh - ch, int(cy)))
    game_h = OUT_H - band
    gw = min(sw, round(sh * OUT_W / game_h)) // 2 * 2
    gx = (sw - gw) // 2
    return band, f"{cw}:{ch}:{cx}:{cy}", f"{gw}:{sh}:{gx}:0"


def _finish_graph(
    parts: list[str],
    current: str,
    watermark: Watermark | None,
    subtitle_name: str | None,
    fonts_name: str,
    audio_filter: str,
) -> str:
    """Logo, captions and the audio chain on top of the composed frame."""
    if watermark is not None:
        parts.append(f"{current}[pre]")
        parts.append(
            f"[1:v]scale={watermark.width_px}:-2,format=rgba,"
            f"colorchannelmixer=aa={watermark.opacity:.3f}[logo]"
        )
        x = str(LOGO_SIDE_PX) if watermark.corner == "top_left" else f"W-w-{LOGO_SIDE_PX}"
        current = f"[pre][logo]overlay={x}:{LOGO_TOP_PX}:format=auto"
    tail = f"ass=filename={subtitle_name}:fontsdir={fonts_name}" if subtitle_name else "null"
    parts.append(f"{current},setsar=1,format=yuv420p,{tail}[vout]")
    parts.append(f"[acat]{audio_filter}[aout]")
    return ";".join(parts)


def _split_cam(
    section: Path, start_s: float, end_s: float
) -> tuple[int, int, int, int, int, int] | None:
    """The streamer's webcam rect in ``section`` (source px), or None to fit."""
    try:
        from ..pipeline.reframe.facecam import detect_facecam

        info = ffmpeg.probe(section)
        if not info.width or not info.height:
            return None
        cam = detect_facecam(
            section,
            start_s=start_s,
            end_s=end_s,
            source_width=info.width,
            source_height=info.height,
        )
    except Exception:  # detection is best effort: the fit layout always works
        log.warning("Webcam detection failed; using the fit layout.", exc_info=True)
        return None
    if cam is None:
        log.info("No webcam found in %s; using the fit layout.", section.name)
        return None
    return (cam.x, cam.y, cam.width, cam.height, info.width, info.height)


def _punch_spans(
    punch_in: tuple[float, float, float] | Sequence[tuple[float, float, float]] | None,
) -> list[tuple[float, float, float]]:
    if not punch_in:
        return []
    if isinstance(punch_in[0], int | float):
        return [punch_in]  # type: ignore[list-item]
    return [span for span in punch_in if span]  # type: ignore[misc]


#: Fade at both ends of every audio piece: a cut mid-waveform clicks, and
#: dead-air removal makes many cuts. 15 ms is inaudible as a fade.
EDGE_FADE_S = 0.015


def _edge_fades(duration: float) -> str:
    """',afade=in…,afade=out…' for a piece of ``duration`` seconds ('' if too short)."""
    if duration <= EDGE_FADE_S * 4:
        return ""
    out_at = duration - EDGE_FADE_S
    fade = f"d={EDGE_FADE_S}:curve=tri"
    return f",afade=t=in:{fade},afade=t=out:st={out_at:.4f}:{fade}"


def _audio_only_graph(keeps: list[Keep], *, seek_s: float, tail: str) -> str:
    parts: list[str] = []
    labels: list[str] = []
    for i, keep in enumerate(keeps):
        a = max(0.0, keep.start - seek_s)
        b = max(a + 0.01, keep.end - seek_s)
        parts.append(
            f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS{_edge_fades(b - a)}[a{i}]"
        )
        labels.append(f"[a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(keeps)}:v=0:a=1[acat]")
    parts.append(f"[acat]{tail}[aout]")
    return ";".join(parts)


def measure_loudnorm(
    source: Path,
    keeps: list[Keep],
    *,
    seek_s: float,
    span_s: float,
    target: float,
    true_peak: float,
) -> dict[str, str] | None:
    """First loudnorm pass over the exact audio that will be rendered."""
    graph = _audio_only_graph(
        keeps,
        seek_s=seek_s,
        tail=shared_edit.measure_filter(target=target, true_peak=true_peak),
    )
    return shared_edit.measure_loudnorm(
        [*DEEP_PROBE, "-ss", f"{seek_s:.4f}", "-t", f"{span_s:.4f}", "-i", str(source)], graph
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Watermark:
    path: Path
    corner: str = "top_right"
    width_px: int = 140
    opacity: float = 0.85


@dataclass(frozen=True)
class Extras:
    """Own-value editing applied on top of the clip."""

    punch_in: bool = True
    punch_in_zoom: float = 1.3
    punch_in_s: float = 1.5
    reaction_text: str = ""
    watermark: Watermark | None = None
    #: Key words drawn bigger in the caption look's accent colour.
    keywords: bool = False
    #: Step zooms on the stream frame: "off" | "subtle" | "dynamic".
    motion: str = "off"
    #: Open on the peak reaction (punched in), then play from the start.
    flash_forward: bool = False
    flash_forward_s: float = 2.0
    #: A variant's own zoom pattern (0 = the original's).
    motion_seed: int = 0
    #: "fit" or "split" (webcam on top, gameplay below; needs a found webcam).
    layout: str = "fit"
    #: A static caption box is added after the render: the hook title is left
    #: out (the box carries it) and the reaction text moves above the box.
    static_caption: bool = False


@dataclass(frozen=True)
class FrameMotion:
    """Zooms applied to the stream frame layer (output seconds)."""

    zooms: tuple[shared_edit.Zoom, ...]
    intro: bool
    #: Size of the fitted, uncropped frame in the 1080x1920 canvas.
    width: int
    height: int

    def filter(self) -> str:
        """zoompan about the frame's centre, output at the fitted size.

        zoompan crops the source frame and scales it straight to the fitted
        size, so the zoom costs no extra upscale of an already-scaled frame.
        """
        expr = shared_edit.zoom_expression(list(self.zooms), self.intro)
        return (
            f"zoompan=z='{expr}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'"
            f":d=1:s={self.width}x{self.height}:fps={FPS}"
        )


def fitted_size(width: int | None, height: int | None) -> tuple[int, int] | None:
    """The whole frame scaled to fit 1080x1920 (even sizes), or None."""
    if not width or not height or width <= 0 or height <= 0:
        return None
    scale = min(OUT_W / width, OUT_H / height)
    w = max(2, min(OUT_W, 2 * round(width * scale / 2)))
    h = max(2, min(OUT_H, 2 * round(height * scale / 2)))
    return w, h


def plan_motion(
    words: list[Word],
    duration: float,
    keeps: list[Keep],
    level: str,
    section: Path,
    seed: int = 0,
    *,
    intro_allowed: bool = True,
) -> FrameMotion | None:
    """Step zooms through the clip (edit.plan_zooms), or None for no motion.

    Jump cuts from the dead-air cut always get a zoom change, which hides
    the jump. ``seed`` > 0 is a variant: its own spacing/depth, and a steady
    slight punch-in even with motion off, so a re-post is not frame-identical.
    Never fatal: an unreadable section just renders without motion.
    """
    if (level not in {"subtle", "dynamic"} and seed <= 0) or duration < 4.0:
        return None
    zooms = shared_edit.plan_zooms(words, duration, level, shared_edit.cut_times(keeps), seed)
    intro = level == "dynamic" and intro_allowed
    if not intro and not any(z.z != 1.0 for z in zooms):
        return None
    try:
        info = ffmpeg.probe(section)
    except Exception:
        log.warning("Could not read %s for motion; rendering without it.", section.name)
        return None
    size = fitted_size(info.width, info.height)
    if size is None:
        return None
    return FrameMotion(zooms=tuple(zooms), intro=intro, width=size[0], height=size[1])


def key_moment(
    levels: np.ndarray | None,
    keeps: list[Keep],
    local_start: float,
    *,
    hop_s: float = 0.05,
    window_s: float = 1.0,
) -> float | None:
    """Output time of the clip's loudest ``window_s`` (its key beat).

    ``levels`` are dBFS per ``hop_s`` from ``local_start``; only audio that
    survives the dead-air cut counts.
    """
    if levels is None or len(levels) == 0:
        return None
    width = max(1, int(round(window_s / hop_s)))
    if len(levels) < width:
        return None
    power = np.power(10.0, levels.astype(np.float64) / 10.0)
    sums = np.convolve(power, np.ones(width), mode="valid")
    best_time: float | None = None
    best = -1.0
    offset = 0.0
    for keep in keeps:
        a = int(np.ceil(max(0.0, keep.start - local_start) / hop_s))
        b = int((keep.end - local_start - window_s) / hop_s)
        for i in range(max(0, a), min(b, len(sums) - 1) + 1):
            if sums[i] > best:
                best = float(sums[i])
                best_time = offset + (local_start + i * hop_s - keep.start)
        offset += keep.duration
    return best_time


def punch_in_window(moment: float | None, duration: float, length: float) -> tuple[float, float]:
    """Punch-in span around the key moment, kept inside the clip."""
    if moment is None:
        moment = duration * 0.4
    a = max(0.0, moment - 0.2)
    b = min(duration - 0.1, a + length)
    if b - a < 0.5:
        a = max(0.0, b - 0.5)
    return a, b


#: Flash-forward limits (shared with the normal pipeline's cold open).
TEASER_MIN_S = shared_edit.TEASER_MIN_S
TEASER_MAX_S = shared_edit.TEASER_MAX_S
TEASER_LEAD_S = shared_edit.TEASER_LEAD_S
TEASER_MIN_CLIP_S = shared_edit.TEASER_MIN_CLIP_S
#: A Hype Curve peak is refined to the loudest beat this close to it (s).
PEAK_SEARCH_S = 3.0


def loudest_time(
    levels: np.ndarray | None,
    levels_start: float,
    a: float,
    b: float,
    *,
    hop_s: float = 0.05,
    window_s: float = 1.0,
) -> float | None:
    """Middle of the loudest ``window_s`` between ``a`` and ``b`` (levels' clock)."""
    if levels is None or len(levels) == 0:
        return None
    width = max(1, int(round(window_s / hop_s)))
    i = max(0, int(math.ceil((a - levels_start) / hop_s)))
    j = min(len(levels) - width, int((b - levels_start - window_s) / hop_s))
    if j < i:
        return None
    power = np.power(10.0, levels.astype(np.float64) / 10.0)
    sums = np.convolve(power, np.ones(width), mode="valid")
    best = i + int(np.argmax(sums[i : j + 1]))
    return levels_start + best * hop_s + window_s / 2


def plan_teaser(
    words: Sequence[Word],
    clip_start: float,
    clip_end: float,
    *,
    levels: np.ndarray | None,
    length_s: float,
    span: tuple[float, float] | None = None,
    peak_s: float | None = None,
    fps: int = FPS,
) -> Keep | None:
    """The flash-forward span (same clock as ``words``/``clip_start``), or None.

    Priority: the judge's quoted peak line (``span``), else ``peak_s`` (the
    Hype Curve's peak) refined to the loudest beat near it, else the loudest
    beat of the clip. The tease is ``length_s`` long (a quoted line keeps its
    own length within 0.8-3.6 s), never cuts a word in half where it can
    avoid it, starts at least 2.5 s into the clip (else the opening would
    play twice in a row) and stays inside the clip.
    """
    if clip_end - clip_start < TEASER_MIN_CLIP_S:
        return None
    earliest = clip_start + TEASER_LEAD_S
    if span is not None and span[1] > span[0]:
        a, b = span
        if b - a > TEASER_MAX_S:
            a = b - TEASER_MAX_S  # a long line still ends on its payoff
        if b - a < TEASER_MIN_S:
            b = a + TEASER_MIN_S
    else:
        centre = None
        if peak_s is not None and clip_start <= peak_s <= clip_end:
            centre = loudest_time(
                levels,
                clip_start,
                max(earliest, peak_s - PEAK_SEARCH_S),
                min(clip_end, peak_s + PEAK_SEARCH_S),
            )
            centre = centre if centre is not None else peak_s
        if centre is None:
            centre = loudest_time(levels, clip_start, earliest, clip_end)
        if centre is None:
            return None
        a = centre - 0.4 * length_s
        b = a + length_s
        # Whole words: start on the word in progress, end once it is said.
        for w in words:
            if w.start < a < w.end and b - w.start <= TEASER_MAX_S:
                a = w.start
            if w.start < b < w.end:
                b = w.end if w.end - a <= TEASER_MAX_S else w.start
    if a < earliest:
        b += earliest - a
        a = earliest
    b = min(b, clip_end)
    if not TEASER_MIN_S <= b - a <= TEASER_MAX_S + 1e-6:
        return None
    # On the frame grid, widened (never narrowed) so no word is clipped.
    a = math.floor(a * fps + 1e-6) / fps
    b = min(math.ceil(b * fps - 1e-6) / fps, clip_end)
    return Keep(round(a, 4), round(b, 4)) if b - a >= TEASER_MIN_S - 1e-6 else None


def _shift(words: list[Word], by: float) -> list[Word]:
    if not by:
        return words
    return [Word(w.text, w.start + by, w.end + by, w.speaker) for w in words]


def _local(words: Sequence[Word], offset: float) -> list[Word]:
    return [Word(w.text, w.start - offset, w.end - offset, w.speaker) for w in words]


@dataclass
class RenderJob:
    section: Path
    #: Stream time at which the section file starts.
    section_offset_s: float
    #: Clip range in STREAM time.
    start_s: float
    end_s: float
    #: Transcript words of the clip, STREAM time.
    words: list[Word]
    hook_title: str
    destination: Path
    work_dir: Path
    extras: Extras = field(default_factory=Extras)
    #: Caption words when they differ from the spoken ones (a caption
    #: language), STREAM time. Dead air is always planned on ``words``.
    display_words: list[Word] | None = None
    #: The judge's quoted peak line (stream time), for the flash-forward.
    teaser: tuple[float, float] | None = None
    #: The Hype Curve's peak inside the clip (stream time), if known.
    peak_s: float | None = None


@dataclass
class RenderResult:
    path: Path
    duration_s: float
    keeps: list[Keep]
    cut_s: float
    #: Own-value edits actually applied (QA needs at least one).
    edits: list[str] = field(default_factory=list)
    watermarked: bool = False
    #: Output length of the flash-forward (0 = none).
    teaser_s: float = 0.0
    #: Where the static caption box goes: ("bottom", y of its bottom edge),
    #: just above the platform's bottom bar (the subtitles sit above it).
    caption_anchor: tuple[str, int] = ("bottom", OUT_H - CAPTION_MARGIN_PX)


def render(
    job: RenderJob,
    *,
    style_key: str,
    title_s: float,
    dead_air_min_s: float,
    dead_air_keep_s: float,
    max_duration_s: float,
    loudness_lufs: float,
    true_peak_db: float,
    export_settings: ExportSettings,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> RenderResult:
    """Render one GTA clip to ``job.destination`` (atomic replace)."""
    local_start = job.start_s - job.section_offset_s + trim_start_s
    local_end = job.end_s - job.section_offset_s - trim_end_s
    if local_end - local_start < 1.0:
        raise RenderError("The clip is too short to render.")
    local_words = _local(job.words, job.section_offset_s)
    shown_words = (
        _local(job.display_words, job.section_offset_s)
        if job.display_words is not None
        else local_words
    )

    try:
        levels = section_levels(job.section, local_start, local_end)
    except RenderError:
        log.warning("Could not read levels for dead-air detection; keeping every second.")
        levels = None
    keeps = plan_keeps(
        [Word(w.text, w.start - local_start, w.end - local_start) for w in local_words],
        0.0,
        local_end - local_start,
        levels=levels,
        min_gap_s=dead_air_min_s,
        keep_s=dead_air_keep_s,
    )
    keeps = [Keep(k.start + local_start, k.end + local_start) for k in keeps]
    extras = job.extras

    teaser: Keep | None = None
    if extras.flash_forward:
        teaser = plan_teaser(
            local_words,
            local_start,
            local_end,
            levels=levels,
            length_s=extras.flash_forward_s,
            span=_local_span(job.teaser, job.section_offset_s),
            peak_s=None if job.peak_s is None else job.peak_s - job.section_offset_s,
        )
        if teaser is None:
            log.info("No flash-forward for %s: no peak past the opening.", job.destination.name)
    lead = teaser.duration if teaser is not None else 0.0
    # The tease counts towards the length cap: the whole clip stays under it.
    keeps = _cap_duration(keeps, max_duration_s - lead)
    main_duration = output_duration(keeps)
    played = [teaser, *keeps] if teaser is not None else keeps
    duration = output_duration(played)

    seek = max(0.0, min(k.start for k in played) - SEEK_PAD_S)
    span = max(k.end for k in played) - seek + 0.5

    workspace = job.work_dir
    workspace.mkdir(parents=True, exist_ok=True)
    main_words = remap_words(local_words, keeps)
    out_words = _shift(remap_words(shown_words, keeps), lead)
    if teaser is not None:
        out_words = remap_words(shown_words, [teaser]) + out_words
    moment = key_moment(levels, keeps, local_start)
    window = punch_in_window(moment, main_duration, extras.punch_in_s)
    window = (window[0] + lead, window[1] + lead)
    edits: list[str] = []
    punches: list[tuple[float, float, float]] = []
    if teaser is not None:
        # The tease is punched in, so it reads as a flash-forward, not a stutter.
        punches.append((0.0, round(lead, 3), extras.punch_in_zoom))
        edits.append("flash-forward")
    if extras.punch_in and main_duration > 3.0:
        punches.append((window[0], window[1], extras.punch_in_zoom))
        edits.append("punch-in")
    motion = plan_motion(
        main_words,
        main_duration,
        keeps,
        extras.motion,
        job.section,
        extras.motion_seed,
        intro_allowed=teaser is None,
    )
    if motion is not None and teaser is not None:
        shifted = [shared_edit.Zoom(round(z.t + lead, 3), z.z) for z in motion.zooms]
        if not shifted or shifted[0].t > lead + 1e-6:
            shifted.insert(0, shared_edit.Zoom(round(lead, 3), 1.0))
        motion = FrameMotion(
            zooms=(shared_edit.Zoom(0.0, 1.0), *shifted),
            intro=False,
            width=motion.width,
            height=motion.height,
        )
    if motion is not None:
        edits.append("motion")
    keywords = gta_captions.clip_keywords(out_words, extras.keywords)
    if keywords:
        edits.append("key words")
    split = _split_cam(job.section, local_start, local_end) if extras.layout == "split" else None
    # The static caption box sits at the bottom, just above the platform's
    # bottom bar; the subtitles move up to sit right above it.
    anchor = ("bottom", OUT_H - CAPTION_MARGIN_PX)
    subtitle_margin = CAPTION_MARGIN_PX
    if extras.static_caption and job.hook_title.strip():
        from .narrate import caption_png

        box_h = caption_png(job.hook_title, workspace / "caption_box.png")[1]
        subtitle_margin = CAPTION_MARGIN_PX + box_h + BOX_SUBTITLE_GAP_PX
    ass = write_subtitles(
        workspace / "captions.ass",
        out_words,
        style_key=style_key,
        hook_title="" if extras.static_caption else job.hook_title,
        title_s=title_s,
        duration_s=duration,
        reaction_text=extras.reaction_text,
        reaction_window=window,
        keywords=keywords,
        title_start_s=lead,
        text_centre_y=STATIC_REACTION_Y if extras.static_caption else TEXT_CENTRE_Y,
        caption_margin_px=subtitle_margin,
    )
    if extras.reaction_text.strip() and f"{REACTION_STYLE}," in ass.read_text(encoding="utf-8"):
        edits.append("reaction text")
    watermark = extras.watermark
    if watermark is not None and not Path(watermark.path).is_file():
        log.warning("Watermark %s not found; rendering without it.", watermark.path)
        watermark = None
    render_cwd, subtitle_name, fonts_name = ffmpeg.relative_filter_workspace(ass, captions.FONT_DIR)

    measured = measure_loudnorm(
        job.section, played, seek_s=seek, span_s=span, target=loudness_lufs, true_peak=true_peak_db
    )
    audio_filter = (
        loudnorm_filter(measured, target=loudness_lufs, true_peak=true_peak_db) + ",aresample=48000"
    )
    if split is not None:
        punches, motion = [], None
        edits = [e for e in edits if e not in {"punch-in", "motion"}] + ["split layout"]
    graph = build_filtergraph(
        played,
        seek_s=seek,
        subtitle_name=subtitle_name,
        fonts_name=fonts_name,
        audio_filter=audio_filter,
        punch_in=punches or None,
        watermark=watermark,
        motion=motion,
        split=split,
    )
    logo_input = ["-i", str(watermark.path)] if watermark is not None else []

    job.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = job.destination.with_name(job.destination.stem + ".partial.mp4")

    def args(video_args: list[str]) -> list[str]:
        return [
            *DEEP_PROBE,
            "-ss",
            f"{seek:.4f}",
            "-t",
            f"{span:.4f}",
            "-i",
            str(job.section),
            *logo_input,
            "-filter_complex",
            graph,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            *video_args,
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(partial),
        ]

    try:
        video_args = encoder_args(export_settings)
        try:
            ffmpeg.run(
                args(video_args),
                total_duration_s=duration,
                on_progress=on_progress,
                cancelled=cancelled,
                cwd=render_cwd,
            )
        except ffmpeg.FFmpegError:
            if "h264_nvenc" not in video_args or (cancelled is not None and cancelled()):
                raise
            log.warning("GPU encoder failed for %s; rendering on the CPU.", job.destination.name)
            cpu = export_settings.model_copy(update={"prefer_hardware_encoder": False})
            ffmpeg.run(
                args(encoder_args(cpu)),
                total_duration_s=duration,
                on_progress=on_progress,
                cancelled=cancelled,
                cwd=render_cwd,
            )
        if not partial.exists() or partial.stat().st_size == 0:
            raise RenderError("ffmpeg finished but the clip file is empty.")
        os.replace(partial, job.destination)
    except ffmpeg.FFmpegError as exc:
        partial.unlink(missing_ok=True)
        raise RenderError(f"Could not render {job.destination.name}: {exc}") from exc
    except BaseException:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove %s.", partial)
        raise

    cut = (local_end - local_start) - main_duration
    return RenderResult(
        path=job.destination,
        duration_s=duration,
        keeps=played,
        cut_s=cut,
        edits=edits,
        watermarked=watermark is not None,
        teaser_s=round(lead, 3),
        caption_anchor=anchor,
    )


def _local_span(span: tuple[float, float] | None, offset: float) -> tuple[float, float] | None:
    if span is None:
        return None
    return span[0] - offset, span[1] - offset


def _cap_duration(keeps: list[Keep], max_s: float) -> list[Keep]:
    """Trim the END so the output never exceeds ``max_s`` (Shorts limit)."""
    out: list[Keep] = []
    total = 0.0
    for keep in keeps:
        if total + keep.duration <= max_s:
            out.append(keep)
            total += keep.duration
            continue
        room = max_s - total
        if room >= 0.08:
            out.append(Keep(keep.start, keep.start + room))
        break
    return out or [Keep(keeps[0].start, keeps[0].start + max_s)]
