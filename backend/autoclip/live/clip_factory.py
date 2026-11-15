"""Clip production: buffer range → finished vertical clip.

Takes a judged clip window (the LiveClipRow's ``clip_start_s``/``clip_end_s``,
which the manager has already expanded with pre/post roll) and turns the
matching range of the stream's rolling buffer into a caption-burned MP4 — the
Live Watch equivalent of pipeline/runner.py, but running against a RingBuffer
extraction instead of a stored source file.

Every heavy stage is reused from the batch pipeline rather than reimplemented:
``prepare.extract_audio`` → ``transcribe.transcribe`` →
``reframe.build_crop_path`` → ``export.export_clip``. Two live-specific steps
wrap them:

* an MPEG-TS remux (``-c copy``, container only) turns the buffer's
  concatenated segment extract into a seekable faststart MP4. The reused stages
  all seek, and a raw mid-GOP ``.ts`` concat seeks poorly if at all.
* stage artifacts persist in ``live/clips/<clip_id>/`` so a failed produce can
  be re-run without redoing earlier stages. That is not just an optimisation:
  the rolling buffer prunes itself after ``retain_minutes``, so a retry an hour
  later *cannot* re-extract — unless the extracted ``raw.ts`` was kept. A
  window manifest guards the reuse against a row whose window was revised
  between attempts, which would otherwise silently render the wrong moment.

The remux is where the MPEG-TS quirks of concatenated ``-reset_timestamps``
segments surface (packet-corrupt warnings, DTS jumps), which is exactly why it
gets the one real-ffmpeg test in this module's suite rather than a mock.

Failure contract: any stage failure raises ClipProductionError carrying a
``stage`` tag (``validate``, ``extract`` … ``export``); never a bare pipeline
exception. The manager marks the row failed and surfaces ``str(exc)`` verbatim.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from ..config import Settings
from ..pipeline import captions, export, ffmpeg, prepare, transcribe
from ..pipeline.reframe import ReframeConfig, build_crop_path
from ..pipeline.reframe.croppath import CropPath
from ..pipeline.transcript import Transcript
from . import gpu
from .buffer import SEGMENT_S, RingBuffer
from .models import LiveClipRow
from .paths import clip_dir

log = logging.getLogger(__name__)

#: Progress callback shape: ``(stage_name, overall_fraction_0_to_1)``. The
#: fraction is across the whole produce() run, so the manager can render one
#: bar per clip; the stage name says where the time went.
ProduceProgress = Callable[[str, float], None]

#: Rough wall-time proportions per stage, mirroring runner.STAGE_WEIGHTS.
STAGE_WEIGHTS: dict[str, float] = {
    "extract": 0.05,
    "remux": 0.05,
    "audio": 0.10,
    "transcribe": 0.35,
    "reframe": 0.20,
    "export": 0.25,
}
STAGE_ORDER: tuple[str, ...] = tuple(STAGE_WEIGHTS)

#: Fixed artifact names inside ``live/clips/<clip_id>/``. The rendered clip
#: itself is named by ``export.output_filename`` (title + ratio slug).
RAW_NAME = "raw.ts"
NORMALIZED_NAME = "normalized.mp4"
AUDIO_NAME = "audio.wav"
TRANSCRIPT_NAME = "transcript.json"
CROP_NAME = "crop_path.json"
WINDOW_NAME = "window.json"

#: Target aspect per row ratio, for the ReframeConfig. Keys mirror
#: export.RATIOS; validated there so the pipeline's list stays the single truth.
_ASPECTS: dict[str, tuple[int, int]] = {"9:16": (9, 16), "1:1": (1, 1), "16:9": (16, 9)}


class ClipProductionError(RuntimeError):
    """Producing a live clip failed at a named stage.

    ``stage`` is one of STAGE_ORDER (plus ``validate``); ``clip_id`` identifies
    the row. The message is written to be stored on the LiveClipRow as the
    user-facing error, so it carries the clip id and stage itself.
    """

    def __init__(self, message: str, *, stage: str, clip_id: str = "") -> None:
        super().__init__(message)
        self.stage = stage
        self.clip_id = clip_id


@dataclass
class ProduceResult:
    """Paths the manager persists on the LiveClipRow after a successful render."""

    rendered_path: Path
    srt_path: Path
    transcript_path: Path
    title: str


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _fail(
    stage: str,
    clip_id: str,
    message: str,
    exc: Exception | None = None,
) -> NoReturn:
    """Raise ClipProductionError with stage and clip context baked into the text."""
    text = f"live clip {clip_id or '?'} [{stage}]: {message}"
    if exc is not None:
        text = f"{text} ({exc})"
    log.error(text, exc_info=exc)
    raise ClipProductionError(text, stage=stage, clip_id=clip_id) from exc


def _valid_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


class _StageTracker:
    """Maps per-stage progress into the single overall 0..1 produce fraction."""

    def __init__(self, on_progress: ProduceProgress | None) -> None:
        self.on_progress = on_progress
        self.completed = 0.0

    def begin(self, stage: str) -> None:
        self._emit(stage, 0.0)

    def fraction(self, stage: str) -> Callable[[float], None]:
        def report(value: float) -> None:
            self._emit(stage, max(0.0, min(1.0, value)))

        return report

    def finish(self, stage: str) -> None:
        self.completed += STAGE_WEIGHTS[stage]
        self._emit(stage, 1.0, overall=self.completed)

    def _emit(self, stage: str, value: float, *, overall: float | None = None) -> None:
        if self.on_progress is None:
            return
        if overall is None:
            overall = self.completed + STAGE_WEIGHTS[stage] * value
        self.on_progress(stage, max(0.0, min(1.0, overall)))


def _validate_row(clip_row: LiveClipRow) -> captions.CaptionStyle:
    """Fail before any disk work when the row can't produce a sane clip.

    A typo'd caption style or ratio discovered after the reframe stage has
    already burned minutes of CPU is the exact failure runner._stage_captions
    exists to prevent; live clips get the same fail-fast treatment, up front.
    """
    if not clip_row.id:
        _fail("validate", "", "clip row has an empty id")
    if clip_row.clip_start_s < 0:
        _fail("validate", clip_row.id, f"clip window starts at {clip_row.clip_start_s:.1f}s")
    if clip_row.clip_end_s <= clip_row.clip_start_s:
        _fail(
            "validate",
            clip_row.id,
            f"clip window is empty or reversed "
            f"({clip_row.clip_start_s:.1f}s..{clip_row.clip_end_s:.1f}s)",
        )
    try:
        export.ratio_dimensions(clip_row.ratio)
    except export.ExportError as exc:
        _fail("validate", clip_row.id, "bad clip ratio", exc)
    try:
        return captions.get_style(clip_row.caption_style)
    except ValueError as exc:
        _fail("validate", clip_row.id, "bad caption style", exc)


# -- window manifest --------------------------------------------------------


def _window_matches(work: Path, clip_row: LiveClipRow) -> bool:
    """True when artifacts in ``work`` were produced for this exact window."""
    manifest = work / WINDOW_NAME
    if not manifest.is_file():
        return False
    try:
        decoded: object = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(decoded, dict):
        return False
    try:
        start = float(decoded.get("clip_start_s", -1.0))  # type: ignore[arg-type]
        end = float(decoded.get("clip_end_s", -1.0))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return abs(start - clip_row.clip_start_s) < 1e-6 and abs(end - clip_row.clip_end_s) < 1e-6


def _write_window_manifest(work: Path, clip_row: LiveClipRow) -> None:
    write_window_manifest(work, clip_row.clip_start_s, clip_row.clip_end_s)


def write_window_manifest(work: Path, clip_start_s: float, clip_end_s: float) -> None:
    """Record which window ``raw.ts`` in ``work`` was cut for."""
    (work / WINDOW_NAME).write_text(
        json.dumps({"clip_start_s": clip_start_s, "clip_end_s": clip_end_s}),
        encoding="utf-8",
    )


def _buffer_extract(
    buffer: Any, start_s: float, end_s: float, target: Path, anchor_s: float | None
) -> None:
    """``buffer.extract`` with the trigger anchor when the buffer supports it.

    The anchor lets a window that straddles a reconnect clamp to the
    trigger's epoch instead of failing; minimal test buffers without the
    keyword are called the old way.
    """
    if anchor_s is not None and _accepts_anchor(buffer):
        buffer.extract(start_s, end_s, target, anchor_s=anchor_s)
    else:
        buffer.extract(start_s, end_s, target)


def _accepts_anchor(buffer: Any) -> bool:
    try:
        return "anchor_s" in inspect.signature(buffer.extract).parameters
    except (TypeError, ValueError, AttributeError):
        return False


def window_staged(clip_row: LiveClipRow) -> bool:
    """True when ``raw.ts`` for exactly this row's window is on disk.

    A staged window makes the clip renderable without the rolling buffer —
    across a pause, a restart, or a retry an hour later.
    """
    work = clip_dir(clip_row.id)
    return _valid_file(work / RAW_NAME) and _window_matches(work, clip_row)


def stage_window(
    buffer: Any,
    clip_id: str,
    clip_start_s: float,
    clip_end_s: float,
    *,
    anchor_s: float | None = None,
) -> Path:
    """Cut the render window into ``live/clips/<id>/raw.ts`` right now.

    Called at accept time, while the window is guaranteed to be in the
    rolling buffer: a queued clip then no longer depends on the buffer
    holding its pre-roll until a render worker gets to it (the buffer may
    prune it, or be cleared by a pause/restart). ``produce()`` reuses the
    staged file via the window manifest.
    """
    work = clip_dir(clip_id)
    work.mkdir(parents=True, exist_ok=True)
    raw = work / RAW_NAME
    _buffer_extract(buffer, clip_start_s, clip_end_s, raw, anchor_s)
    if not _valid_file(raw):
        raise ClipProductionError(
            f"live clip {clip_id} [extract]: the staged window is empty",
            stage="extract",
            clip_id=clip_id,
        )
    write_window_manifest(work, clip_start_s, clip_end_s)
    return raw


#: Render intermediates — big, and useless once the clip is ready.
INTERMEDIATE_NAMES: tuple[str, ...] = (RAW_NAME, NORMALIZED_NAME, AUDIO_NAME)


def cleanup_intermediates(clip_id: str) -> int:
    """Delete a ready clip's render intermediates; returns bytes freed.

    raw.ts, normalized.mp4 and audio.wav are only needed to resume a failed
    render. The rendered .mp4/.srt, the transcript and the crop path stay.
    """
    work = clip_dir(clip_id)
    freed = 0
    for name in INTERMEDIATE_NAMES:
        path = work / name
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            log.debug("could not delete %s", path, exc_info=True)
            continue
        freed += size
    return freed


def _wipe_derived(work: Path) -> None:
    """Remove every stage artifact so a revised window can't reuse stale bytes."""
    for name in (RAW_NAME, NORMALIZED_NAME, AUDIO_NAME, TRANSCRIPT_NAME, CROP_NAME):
        (work / name).unlink(missing_ok=True)
    # The rendered clip's name depends on the title; sweep rather than enumerate.
    for pattern in ("*.mp4", "*.srt"):
        for stale in work.glob(pattern):
            stale.unlink(missing_ok=True)


# -- stages ------------------------------------------------------------------


def _extract_window(
    buffer: RingBuffer, clip_row: LiveClipRow, tracker: _StageTracker, target: Path
) -> None:
    """Pin the needed segments, extract [clip_start_s, clip_end_s), unpin.

    The pin is dropped as soon as the extraction is safely on disk: the heavy
    stages that follow read ``raw.ts``, not the buffer, and holding pins for
    minutes would make the buffer's disk cap unenforceable while a render
    queues behind the GPU.
    """
    tracker.begin("extract")
    if buffer is None:
        _fail(
            "extract",
            clip_row.id,
            "the clip window was never staged and the stream's rolling buffer is gone"
            " (stream paused or removed)",
        )
    pin_index = int(clip_row.clip_start_s // SEGMENT_S)
    buffer.pin_oldest(pin_index)
    try:
        try:
            anchor = clip_row.trigger_end_s if clip_row.trigger_end_s > 0 else None
            _buffer_extract(buffer, clip_row.clip_start_s, clip_row.clip_end_s, target, anchor)
        except Exception as exc:
            _fail(
                "extract",
                clip_row.id,
                f"could not extract {clip_row.clip_start_s:.1f}s.."
                f"{clip_row.clip_end_s:.1f}s from the rolling buffer",
                exc,
            )
        if not _valid_file(target):
            _fail(
                "extract",
                clip_row.id,
                f"buffer extraction for {clip_row.clip_start_s:.1f}s.."
                f"{clip_row.clip_end_s:.1f}s wrote an empty file",
            )
    finally:
        try:
            buffer.unpin(pin_index)
        except Exception:  # never mask the real failure with an unpin glitch
            log.warning("Unpinning segment %d failed; the pin will expire on close.", pin_index)
    _write_window_manifest(target.parent, clip_row)
    tracker.finish("extract")


def _remux(raw: Path, destination: Path, clip_id: str) -> None:
    """Copy the .ts range into a seekable faststart MP4 — container only.

    Live extracts are MPEG-TS: whole 6-second segments concatenated, so the
    stream carries a timestamp discontinuity at every segment boundary and
    may open mid-GOP (the extraction starts at an arbitrary second).
    ``+genpts`` fills gaps in packet timestamps and ``make_zero`` keeps the
    first frame at t≈0 so every later seek lands where the stages expect.
    """
    ffmpeg.run(
        [
            "-fflags",
            "+genpts",
            "-i",
            str(raw),
            "-map",
            "0:v:0",
            # Required, not optional: no audio track means no transcript and
            # no audible clip, which for Live Watch is a hard error, not a
            # warning — fail here with a clear message instead of at Whisper.
            "-map",
            "0:a:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(destination),
        ]
    )


def _probe_media(path: Path, clip_id: str) -> ffmpeg.MediaInfo:
    try:
        info = ffmpeg.probe(path)
    except Exception as exc:
        _fail("remux", clip_id, f"could not probe {path.name}", exc)
    if not info.has_video:
        _fail("remux", clip_id, f"{path.name} has no video stream to reframe")
    if not info.has_audio:
        _fail("remux", clip_id, f"{path.name} has no audio track to transcribe")
    return info


def _probe_reusable(path: Path) -> ffmpeg.MediaInfo | None:
    """Probe without failing: a bad artifact must trigger a redo, not an error."""
    if not _valid_file(path):
        return None
    try:
        info = ffmpeg.probe(path)
    except Exception:
        return None
    if not (info.has_video and info.has_audio):
        return None
    return info


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def produce(
    buffer: RingBuffer,
    clip_row: LiveClipRow,
    settings: Settings,
    on_progress: ProduceProgress | None = None,
) -> ProduceResult:
    """Render ``clip_row``'s window out of ``buffer`` into ``live/clips/<id>/``.

    ``settings`` is the app-level Settings (``config.load()``), not LiveSettings:
    loudness/encoder behaviour comes from ``settings.export`` and the Whisper
    model from ``settings.whisper``; the clip-specific choices (window, ratio,
    caption style, title) all live on the row.

    Idempotent by artifact: an interrupted run leaves resumable files, and a
    re-run with the same window skips every stage whose output already exists
    (a half-written file is detected by probe/parse and rebuilt).
    """
    style = _validate_row(clip_row)

    work = clip_dir(clip_row.id)
    work.mkdir(parents=True, exist_ok=True)
    tracker = _StageTracker(on_progress)

    raw = work / RAW_NAME
    normalized = work / NORMALIZED_NAME
    audio = work / AUDIO_NAME
    transcript_path = work / TRANSCRIPT_NAME
    crop_cache = work / CROP_NAME

    log.info(
        "Producing live clip %s (%.1fs..%.1fs, %s, %s)",
        clip_row.id,
        clip_row.clip_start_s,
        clip_row.clip_end_s,
        clip_row.ratio,
        clip_row.caption_style,
    )

    # 1 — extract. The manifest check comes first: a re-judged window must not
    # reuse any artifact cut from the old one.
    if not (_valid_file(raw) and _window_matches(work, clip_row)):
        if not _window_matches(work, clip_row):
            _wipe_derived(work)
        _extract_window(buffer, clip_row, tracker, raw)
    else:
        tracker.begin("extract")
        tracker.finish("extract")

    # 2 — remux to a seekable MP4, unless a previously remuxed pass probes clean.
    info = _probe_reusable(normalized)
    if info is None:
        tracker.begin("remux")
        try:
            _remux(raw, normalized, clip_row.id)
        except Exception as exc:
            _fail("remux", clip_row.id, "could not remux the extracted window to MP4", exc)
        if not _valid_file(normalized):
            _fail("remux", clip_row.id, "the remux wrote an empty file")
        info = _probe_media(normalized, clip_row.id)
        tracker.finish("remux")
    else:
        tracker.begin("remux")
        tracker.finish("remux")

    duration = info.duration_s or (clip_row.clip_end_s - clip_row.clip_start_s)

    # 3 — Whisper input. Word times come back relative to the extraction, so no
    # offset arithmetic is ever needed downstream.
    if not _valid_file(audio):
        tracker.begin("audio")
        try:
            prepare.extract_audio(
                normalized,
                audio,
                duration_s=duration,
                on_progress=tracker.fraction("audio"),
            )
        except Exception as exc:
            _fail("audio", clip_row.id, "could not extract audio for transcription", exc)
        tracker.finish("audio")
    else:
        tracker.begin("audio")
        tracker.finish("audio")

    transcript: Transcript | None = None
    if transcript_path.is_file():
        try:
            transcript = Transcript.load(transcript_path)
        except Exception:
            transcript = None
    if transcript is None:
        tracker.begin("transcribe")
        try:
            # Shared GPU slot + resident model: a render and a judge-phase
            # transcription never load two Whisper models side by side.
            transcript = gpu.transcribe(
                audio,
                settings.whisper,
                transcribe_fn=transcribe.transcribe,
                duration_s=duration,
                on_progress=tracker.fraction("transcribe"),
            )
        except Exception as exc:
            _fail("transcribe", clip_row.id, "transcription failed", exc)
        transcript.save(transcript_path)
        tracker.finish("transcribe")
    else:
        tracker.begin("transcribe")
        tracker.finish("transcribe")

    # 4 — vertical crop path. build_crop_path already degrades to a centre
    # crop when face detection finds nothing, so failures here are real ones.
    aspect_w, aspect_h = _ASPECTS[clip_row.ratio]
    crop_path: CropPath | None = None
    if crop_cache.is_file():
        try:
            crop_path = CropPath.load(crop_cache)
        except Exception:
            crop_path = None
    if crop_path is None:
        tracker.begin("reframe")
        try:
            crop_path = build_crop_path(
                normalized,
                start_s=0.0,
                end_s=duration,
                transcript=transcript,
                config=ReframeConfig(aspect_w=aspect_w, aspect_h=aspect_h),
            )
        except Exception as exc:
            _fail("reframe", clip_row.id, "could not compute the crop path", exc)
        crop_path.save(crop_cache)
        tracker.finish("reframe")
    else:
        tracker.begin("reframe")
        tracker.finish("reframe")

    # 5 — render. The .srt sidecar is contractual for live clips (uploads and
    # the gallery read it), so write_srt is forced on regardless of the
    # user's batch-export default.
    destination = work / export.output_filename(
        clip_row.title or f"clip-{clip_row.id}", clip_row.ratio
    )
    request = export.ExportRequest(
        source=normalized,
        destination=destination,
        start_s=0.0,
        end_s=duration,
        crop_path=crop_path,
        words=transcript.words,
        style=style,
        ratio=clip_row.ratio,
    )
    tracker.begin("export")
    try:
        with gpu.gpu_slot():  # the hardware encoder is the other GPU stage
            export.export_clip(
                request,
                work_dir=work,
                settings=settings.export.model_copy(update={"write_srt": True}),
                on_progress=tracker.fraction("export"),
            )
    except Exception as exc:
        _fail("export", clip_row.id, f"could not render {destination.name}", exc)

    srt_path = destination.with_suffix(".srt")
    if not _valid_file(destination):
        _fail("export", clip_row.id, f"{destination.name} is missing or empty after export")
    if not srt_path.is_file():
        _fail("export", clip_row.id, f"{srt_path.name} sidecar was not written")
    tracker.finish("export")

    log.info("Produced live clip %s -> %s", clip_row.id, destination.name)
    return ProduceResult(
        rendered_path=destination,
        srt_path=srt_path,
        transcript_path=transcript_path,
        title=clip_row.title,
    )
