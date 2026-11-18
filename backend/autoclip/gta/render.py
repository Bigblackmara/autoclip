"""GTA Mode rendering: dead air cut, uncropped frame, hook title, captions.

Layout (9:16, 1080x1920): the WHOLE stream frame — gameplay and the
streamer's own face-cam overlay exactly as broadcast, nothing cropped — sits
in the middle, scaled to the full width, over a blurred, zoomed copy of
itself. The hook title lives in the top band for the first seconds, the
word-by-word captions in the bottom band, so neither ever covers the stream.

Loudness is normalised in two passes (measure, then apply linearly) so every
clip lands on the Shorts/TikTok target instead of near it.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import ExportSettings
from ..pipeline import captions, ffmpeg
from ..pipeline import edit as shared_edit
from ..pipeline.edit import Keep, loudnorm_filter, output_duration, remap_words
from ..pipeline.export import encoder_args
from ..pipeline.transcript import Word

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
CAPTION_MARGIN_RATIO = (SAFE_BOTTOM_PX + 30) / OUT_H
#: Hook title and reaction text: centred in the top band, below the logo row.
TITLE_FONT_RATIO = 0.052
TITLE_STYLE = "GtaHook"
REACTION_STYLE = "GtaReaction"
TEXT_CENTRE_Y = 430
#: Logo row: just under the top safe zone.
LOGO_TOP_PX = SAFE_TOP_PX + 15
LOGO_SIDE_PX = 40


class RenderError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Dead air
# --------------------------------------------------------------------------


def section_levels(path: Path, start_s: float, end_s: float, *, hop_s: float = 0.05) -> np.ndarray:
    """RMS level (dBFS) of every ``hop_s`` of the section's audio in a range."""
    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-t",
        f"{max(0.05, end_s - start_s):.3f}",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)  # noqa: S603
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
) -> Path:
    import pysubs2

    style = captions.get_style(style_key)
    subs = captions.build_ass(
        words,
        style,
        width=OUT_W,
        height=OUT_H,
        time_offset_s=0.0,
        margin_v_ratio=CAPTION_MARGIN_RATIO,
    )
    title = hook_title.strip()
    text_y = TEXT_CENTRE_Y
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
            start=0,
            end=round(min(title_s, duration_s) * 1000),
            style=TITLE_STYLE,
            text=f"{{\\an5\\pos({OUT_W // 2},{text_y})\\fad(80,220)}}{text}",
        )
        subs.events.append(event)
    reaction = reaction_text.strip()
    if reaction and reaction_window is not None:
        a, b = reaction_window
        # Never on top of the hook title: start once the title is gone.
        a = max(a, min(title_s, duration_s) if title else 0.0)
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
    punch_in: tuple[float, float, float] | None = None,
    watermark: Watermark | None = None,
) -> str:
    """Cut + concat, uncropped fit over a blurred fill, subtitles, audio chain.

    Keep times are section-local; the input is seeked to ``seek_s`` so the
    trims are relative to that. ``punch_in`` = (start, end, zoom) in OUTPUT
    seconds: the stream frame cuts to a centred zoom for that span.
    ``watermark`` reads the logo from input 1.
    """
    parts: list[str] = []
    labels: list[str] = []
    for i, keep in enumerate(keeps):
        a = max(0.0, keep.start - seek_s)
        b = max(a + 0.01, keep.end - seek_s)
        parts.append(f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(labels)}concat=n={len(keeps)}:v=1:a=1[vcat][acat]")
    if punch_in:
        parts.append(f"[vcat]fps={FPS},split=3[bgsrc][fgsrc][zsrc]")
    else:
        parts.append(f"[vcat]fps={FPS},split=2[bgsrc][fgsrc]")
    parts.append(
        f"[bgsrc]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},boxblur=28:2,eq=brightness=-0.06:saturation=1.15[bg]"
    )
    parts.append(
        f"[fgsrc]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fg]"
    )
    current = "[bg][fg]overlay=(W-w)/2:(H-h)/2"
    if punch_in:
        a, b, zoom = punch_in
        # Same aspect as the frame, centred: the zoomed copy fits to the same
        # size and replaces the frame for the punch-in span.
        parts.append(
            f"[zsrc]crop=trunc(iw/{zoom:.4f}/2)*2:trunc(ih/{zoom:.4f}/2)*2,"
            f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fz]"
        )
        parts.append(f"{current}[base]")
        current = f"[base][fz]overlay=(W-w)/2:(H-h)/2:enable='between(t,{a:.3f},{b:.3f})'"
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


def _audio_only_graph(keeps: list[Keep], *, seek_s: float, tail: str) -> str:
    parts: list[str] = []
    labels: list[str] = []
    for i, keep in enumerate(keeps):
        a = max(0.0, keep.start - seek_s)
        b = max(a + 0.01, keep.end - seek_s)
        parts.append(f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[a{i}]")
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
        ["-ss", f"{seek_s:.4f}", "-t", f"{span_s:.4f}", "-i", str(source)], graph
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


@dataclass
class RenderResult:
    path: Path
    duration_s: float
    keeps: list[Keep]
    cut_s: float
    #: Own-value edits actually applied (QA needs at least one).
    edits: list[str] = field(default_factory=list)
    watermarked: bool = False


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
    local_words = [
        Word(
            text=w.text,
            start=w.start - job.section_offset_s,
            end=w.end - job.section_offset_s,
            speaker=w.speaker,
        )
        for w in job.words
    ]

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
    keeps = _cap_duration(keeps, max_duration_s)
    duration = output_duration(keeps)

    seek = max(0.0, keeps[0].start - 1.0)
    span = keeps[-1].end - seek + 0.5

    workspace = job.work_dir
    workspace.mkdir(parents=True, exist_ok=True)
    out_words = remap_words(local_words, keeps)
    extras = job.extras
    moment = key_moment(levels, keeps, local_start)
    window = punch_in_window(moment, duration, extras.punch_in_s)
    edits: list[str] = []
    punch = None
    if extras.punch_in and duration > 3.0:
        punch = (window[0], window[1], extras.punch_in_zoom)
        edits.append("punch-in")
    ass = write_subtitles(
        workspace / "captions.ass",
        out_words,
        style_key=style_key,
        hook_title=job.hook_title,
        title_s=title_s,
        duration_s=duration,
        reaction_text=extras.reaction_text,
        reaction_window=window,
    )
    if extras.reaction_text.strip() and f"{REACTION_STYLE}," in ass.read_text(encoding="utf-8"):
        edits.append("reaction text")
    watermark = extras.watermark
    if watermark is not None and not Path(watermark.path).is_file():
        log.warning("Watermark %s not found; rendering without it.", watermark.path)
        watermark = None
    render_cwd, subtitle_name, fonts_name = ffmpeg.relative_filter_workspace(ass, captions.FONT_DIR)

    measured = measure_loudnorm(
        job.section, keeps, seek_s=seek, span_s=span, target=loudness_lufs, true_peak=true_peak_db
    )
    audio_filter = (
        loudnorm_filter(measured, target=loudness_lufs, true_peak=true_peak_db) + ",aresample=48000"
    )
    graph = build_filtergraph(
        keeps,
        seek_s=seek,
        subtitle_name=subtitle_name,
        fonts_name=fonts_name,
        audio_filter=audio_filter,
        punch_in=punch,
        watermark=watermark,
    )
    logo_input = ["-i", str(watermark.path)] if watermark is not None else []

    job.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = job.destination.with_name(job.destination.stem + ".partial.mp4")

    def args(video_args: list[str]) -> list[str]:
        return [
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

    cut = (local_end - local_start) - duration
    return RenderResult(
        path=job.destination,
        duration_s=duration,
        keeps=keeps,
        cut_s=cut,
        edits=edits,
        watermarked=watermark is not None,
    )


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
