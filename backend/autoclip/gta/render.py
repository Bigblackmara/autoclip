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

import json
import logging
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ExportSettings
from ..pipeline import captions, ffmpeg
from ..pipeline.export import encoder_args
from ..pipeline.transcript import Word

log = logging.getLogger(__name__)

OUT_W = 1080
OUT_H = 1920
FPS = 30

#: Captions sit in the bottom band; their baseline is this far from the bottom.
CAPTION_MARGIN_RATIO = 0.17
#: Hook title: centre of the top band.
TITLE_FONT_RATIO = 0.052
TITLE_STYLE = "GtaHook"


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Keep:
    """A kept span of the section file, in section-local seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


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


def remap_words(words: list[Word], keeps: list[Keep]) -> list[Word]:
    """Word timings on the output timeline (cut spans removed)."""
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


def output_duration(keeps: list[Keep]) -> float:
    return sum(k.duration for k in keeps)


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
        # Centre of the top band (above the stream frame).
        frame_h = OUT_W * 9 / 16
        band_centre = round((OUT_H - frame_h) / 4)
        text = captions.escape_ass_text(title.upper())
        event = pysubs2.SSAEvent(
            start=0,
            end=round(min(title_s, duration_s) * 1000),
            style=TITLE_STYLE,
            text=f"{{\\an5\\pos({OUT_W // 2},{band_centre + 40})\\fad(80,220)}}{text}",
        )
        subs.events.append(event)
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
) -> str:
    """Cut + concat, uncropped fit over a blurred fill, subtitles, audio chain.

    Keep times are section-local; the input is seeked to ``seek_s`` so the
    trims are relative to that.
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
    parts.append(f"[vcat]fps={FPS},split=2[bgsrc][fgsrc]")
    parts.append(
        f"[bgsrc]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},boxblur=28:2,eq=brightness=-0.06:saturation=1.15[bg]"
    )
    parts.append(
        f"[fgsrc]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fg]"
    )
    tail = f"ass=filename={subtitle_name}:fontsdir={fonts_name}" if subtitle_name else "null"
    parts.append(f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p,{tail}[vout]")
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


_LOUDNORM_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


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
        tail=f"loudnorm=I={target}:TP={true_peak}:LRA=11:print_format=json",
    )
    command = [
        ffmpeg.ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-ss",
        f"{seek_s:.4f}",
        "-t",
        f"{span_s:.4f}",
        "-i",
        str(source),
        "-filter_complex",
        graph,
        "-map",
        "[aout]",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    match = _LOUDNORM_JSON.search(result.stderr or "")
    if result.returncode != 0 or not match:
        log.warning("Loudness measurement failed; falling back to one-pass loudnorm.")
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    needed = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not all(key in data for key in needed):
        return None
    try:
        if float(data["input_i"]) < -70:  # (near) silent: linear gain would explode
            return None
    except ValueError:
        return None
    return {key: str(data[key]) for key in needed}


def loudnorm_filter(measured: dict[str, str] | None, *, target: float, true_peak: float) -> str:
    base = f"loudnorm=I={target}:TP={true_peak}:LRA=11"
    if not measured:
        return base
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


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


@dataclass
class RenderResult:
    path: Path
    duration_s: float
    keeps: list[Keep]
    cut_s: float


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
    ass = write_subtitles(
        workspace / "captions.ass",
        out_words,
        style_key=style_key,
        hook_title=job.hook_title,
        title_s=title_s,
        duration_s=duration,
    )
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
    )

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
    return RenderResult(path=job.destination, duration_s=duration, keeps=keeps, cut_s=cut)


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
