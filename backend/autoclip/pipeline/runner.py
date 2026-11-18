"""Pipeline orchestration.

Runs a job stage by stage, writing each stage's artifacts into
``work/{job_id}/``. Because every stage's output is a file on disk, a retry
skips everything already done and resumes at the stage that failed — which
matters when stage two took eleven minutes and stage four hit a rate limit.

Progress is reported through a callback rather than written directly, so the
same runner serves the CLI (a progress bar) and the web API (an SSE stream).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..config import Settings
from ..config import load as load_settings
from ..db import store
from ..db.models import Clip, Export, Job, Source, new_id, utcnow
from ..db.models import Transcript as TranscriptRow
from ..providers import build_provider, detection_config
from . import Stage, captions, export, ffmpeg, highlights, prepare, transcribe
from . import judge as clip_judge
from . import signals as audience_signals
from . import youtube_captions as yt_captions
from .prepare import Loudness, Silence
from .reframe import ReframeConfig, build_crop_path
from .reframe.croppath import CropPath
from .reframe.facecam import (
    CamPrior,
    Facecam,
    build_cam_prior,
    detect_facecam,
    load_detection,
    load_prior,
    save_detection,
    save_prior,
)
from .reframe.faces import FaceDetectionUnavailable
from .transcript import Transcript

log = logging.getLogger(__name__)

#: Weight of each stage in the overall progress bar. Rough proportions of wall
#: time on a mid-range machine — transcription and export dominate.
STAGE_WEIGHTS: dict[Stage, float] = {
    Stage.PREPARE: 0.05,
    Stage.TRANSCRIBE: 0.35,
    Stage.HIGHLIGHTS: 0.15,
    Stage.REFRAME: 0.20,
    Stage.CAPTIONS: 0.02,
    Stage.EXPORT: 0.23,
}


class JobCancelled(RuntimeError):
    """The job was cancelled by the user."""


class PipelineError(RuntimeError):
    """A stage failed in a way the user should see."""

    def __init__(self, message: str, *, stage: Stage) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass
class ProgressEvent:
    stage: Stage
    #: Progress within the stage, 0..1.
    stage_progress: float
    #: Progress across the whole job, 0..1.
    overall: float
    message: str = ""


ProgressHandler = Callable[[ProgressEvent], None]


class JobWorkspace:
    """Paths for one job's intermediate artifacts."""

    def __init__(self, job_id: str) -> None:
        self.root = paths.job_work_dir(job_id)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def audio(self) -> Path:
        return self.root / "audio.wav"

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.json"

    @property
    def silences(self) -> Path:
        return self.root / "silences.json"

    @property
    def loudness(self) -> Path:
        return self.root / "loudness.json"

    @property
    def thumbnails(self) -> Path:
        return self.root / "thumbnails"

    @property
    def audience(self) -> Path:
        """The per-second audience curve (see :mod:`autoclip.pipeline.signals`)."""
        return self.root / "audience.json"

    @property
    def judge_report(self) -> Path:
        return self.root / "judge.json"

    def crop_path(self, clip_id: str) -> Path:
        return self.root / "crops" / f"{clip_id}.json"

    def facecam(self, clip_id: str) -> Path:
        """Cached facecam detection for one clip (``null`` when there is none)."""
        return self.root / "facecam" / f"{clip_id}.json"

    @property
    def facecam_prior(self) -> Path:
        """Cached home-cam prior for the job's source (``null`` when there is none)."""
        return self.root / "facecam" / "source-prior.json"

    @property
    def captions_dir(self) -> Path:
        return self.root / "captions"


#: Settings a job can override at creation time (see ``api.jobs``), as
#: ``(section, field)`` pairs. Everything else — provider models, base URLs,
#: encoder choice — comes from the current settings, so a retry after fixing
#: a bad endpoint in settings picks the fix up.
JOB_SCOPED_SETTINGS: tuple[tuple[str, str], ...] = (
    ("whisper", "model"),
    ("whisper", "language"),
    ("whisper", "diarization"),
    ("clips", "min_duration_s"),
    ("clips", "max_duration_s"),
    ("clips", "max_clips"),
    ("export", "caption_style"),
    ("export", "ratio"),
    ("export", "layout"),
)


def wants_facecam(layout: str, ratio: str, has_video: bool) -> bool:
    """Whether the split layout could apply, i.e. facecam detection is worth running."""
    return has_video and layout != "crop" and ratio in export.SPLIT_RATIOS


def facecam_for_clip(
    workspace: JobWorkspace,
    clip: Clip,
    source: Source,
) -> Facecam | None:
    """Detect (or load the cached) facecam overlay for one clip.

    Detection runs per clip rather than once per source: a long stream
    changes layout — lobby, match, a full-screen "just chatting" stretch — and
    the cam moves with it. Which of several cams is the streamer's comes from
    the source-wide home-cam prior (:func:`cam_prior_for_source`). The result,
    including "no facecam", is cached in the job workspace like the crop paths.
    """
    cache = workspace.facecam(clip.id)
    cached, facecam = load_detection(cache)
    if cached:
        return facecam
    if not source.width or not source.height:
        return None

    try:
        facecam = detect_facecam(
            Path(source.path),
            start_s=clip.start_s,
            end_s=clip.end_s,
            source_width=source.width,
            source_height=source.height,
            # Only built when the clip has a cam at all: talking heads and
            # cam-less videos never pay for the source-level pass.
            prior=lambda: cam_prior_for_source(workspace, source),
        )
    except FaceDetectionUnavailable as exc:
        # Not cached: the next run may have the model.
        log.warning("Facecam detection unavailable (%s); keeping the crop layout.", exc)
        return None
    except Exception:
        # The layout is an enhancement: a detection failure (a model that
        # can't be fetched offline, an undecodable frame) must not fail a job
        # whose ordinary crop would render fine. Not cached, so a retry tries again.
        log.warning(
            "Facecam detection failed for clip %s; keeping the crop layout.", clip.id, exc_info=True
        )
        return None
    save_detection(cache, facecam)
    return facecam


def cam_prior_for_source(workspace: JobWorkspace, source: Source) -> CamPrior | None:
    """Build (or load the cached) home-cam prior for the job's source.

    One cheap pass over frames spread across the whole source finds the cam
    that recurs throughout it — the streamer's, not a teammate's who joined
    for a few lobbies. Cached in the job workspace, including "none found";
    a failure is not cached, and detection then goes on without a prior.
    """
    cache = workspace.facecam_prior
    cached, prior = load_prior(cache)
    if cached:
        return prior
    if not source.width or not source.height or not source.duration_s:
        return None
    try:
        prior = build_cam_prior(
            Path(source.path),
            duration_s=source.duration_s,
            source_width=source.width,
            source_height=source.height,
        )
    except FaceDetectionUnavailable as exc:
        log.warning("Home-cam prior unavailable (%s).", exc)
        return None
    except Exception:
        log.warning("Building the home-cam prior failed; going without it.", exc_info=True)
        return None
    save_prior(cache, prior)
    return prior


def settings_for_job(job: Job) -> Settings:
    """Current settings with the job's own choices layered on top.

    The web API snapshots per-job overrides (clip lengths, clip count, ratio,
    caption style, Whisper model) into ``job.settings``, but the queue built
    its runner from the global settings alone — so every override chosen in
    the UI was silently ignored.
    """
    settings = load_settings()
    snapshot = job.settings or {}
    for section, name in JOB_SCOPED_SETTINGS:
        value = (snapshot.get(section) or {}).get(name)
        if value is not None:
            setattr(getattr(settings, section), name, value)
    return settings


class PipelineRunner:
    """Executes one job."""

    def __init__(
        self,
        job: Job,
        source: Source,
        *,
        settings: Settings | None = None,
        on_progress: ProgressHandler | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.job = job
        self.source = source
        self.settings = settings or settings_for_job(job)
        self.on_progress = on_progress
        self._is_cancelled = is_cancelled or (lambda: False)
        self.workspace = JobWorkspace(job.id)
        self._completed_weight = 0.0

    # -- progress ----------------------------------------------------------

    def _check_cancelled(self) -> None:
        if self._is_cancelled():
            raise JobCancelled("Job cancelled.")

    def _emit(
        self,
        stage: Stage,
        stage_progress: float,
        message: str = "",
        *,
        overall: float | None = None,
    ) -> None:
        if overall is None:
            overall = self._completed_weight + STAGE_WEIGHTS[stage] * stage_progress
        overall = min(1.0, overall)
        store.update_job(self.job.id, current_stage=stage.value, progress=round(overall, 4))
        if self.on_progress:
            self.on_progress(
                ProgressEvent(
                    stage=stage,
                    stage_progress=stage_progress,
                    overall=overall,
                    message=message or stage.label,
                )
            )

    def _finish_stage(self, stage: Stage) -> None:
        """Mark a stage complete.

        ``overall`` is passed explicitly rather than derived. Deriving it after
        incrementing the accumulated weight counts the stage twice, so the bar
        jumped past the true total and then snapped backwards when the next
        stage reported 0 — visibly, at every stage boundary.
        """
        self._completed_weight += STAGE_WEIGHTS[stage]
        self._emit(stage, 1.0, overall=self._completed_weight)

    def _stage_progress(self, stage: Stage) -> Callable[[float], None]:
        def report(fraction: float) -> None:
            self._emit(stage, max(0.0, min(1.0, fraction)))

        return report

    # -- entry point -------------------------------------------------------

    async def run(self) -> list[Clip]:
        """Run the full pipeline and return the exported clips."""
        store.update_job(self.job.id, status="running", started_at=utcnow(), error=None)

        try:
            audio = self._stage_prepare()
            transcript = self._stage_transcribe(audio)
            silences = self._load_or_detect_silences(audio)
            clips = await self._stage_highlights(transcript, silences, audio)
            crop_paths = self._stage_reframe(clips, transcript)
            self._stage_captions(clips, transcript)
            self._stage_export(clips, transcript, crop_paths)
        except JobCancelled:
            store.update_job(self.job.id, status="cancelled", finished_at=utcnow(), progress=0.0)
            raise
        except Exception as exc:
            if self._is_cancelled():
                # Stages abort their own long-running work when cancelled —
                # ffmpeg raises ``ffmpeg.Cancelled``, Whisper a
                # TranscriptionError — and those are not JobCancelled. Without
                # this, cancelling mid-transcription or mid-render left the job
                # marked "failed" with a "cancelled" error message.
                store.update_job(
                    self.job.id, status="cancelled", finished_at=utcnow(), progress=0.0
                )
                raise JobCancelled("Job cancelled.") from exc
            log.exception("Job %s failed.", self.job.id)
            store.update_job(self.job.id, status="failed", error=str(exc), finished_at=utcnow())
            raise

        store.update_job(self.job.id, status="done", progress=1.0, finished_at=utcnow())
        return clips

    # -- stages ------------------------------------------------------------

    def _stage_prepare(self) -> Path:
        stage = Stage.PREPARE
        self._check_cancelled()
        source_path = Path(self.source.path)

        if self.workspace.audio.exists() and self.workspace.audio.stat().st_size > 0:
            log.info("Reusing existing audio for job %s.", self.job.id)
        else:
            self._emit(stage, 0.0, "Extracting audio")
            prepare.extract_audio(
                source_path,
                self.workspace.audio,
                duration_s=self.source.duration_s,
                on_progress=self._stage_progress(stage),
                cancelled=self._is_cancelled,
            )

        if self.source.has_video and not self.workspace.thumbnails.exists():
            prepare.generate_thumbnails(source_path, self.workspace.thumbnails)

        self._finish_stage(stage)
        return self.workspace.audio

    def _stage_transcribe(self, audio: Path) -> Transcript:
        stage = Stage.TRANSCRIBE
        self._check_cancelled()

        if self.workspace.transcript.exists():
            log.info("Reusing existing transcript for job %s.", self.job.id)
            transcript = Transcript.load(self.workspace.transcript)
            self._finish_stage(stage)
            return transcript

        transcript = self._youtube_caption_transcript()
        if transcript is not None:
            self._emit(stage, 0.5, "Using YouTube's own captions")
        else:
            self._emit(stage, 0.0, "Transcribing")
            transcript = transcribe.transcribe(
                audio,
                self.settings.whisper,
                duration_s=self.source.duration_s,
                on_progress=self._stage_progress(stage),
                cancelled=self._is_cancelled,
            )

        if self.settings.whisper.diarization:
            from ..config import HF_TOKEN_KEY, get_secret

            self._emit(stage, 0.95, "Identifying speakers")
            transcribe.diarize(audio, transcript, hf_token=get_secret(HF_TOKEN_KEY, self.settings))

        transcript.save(self.workspace.transcript)
        store.upsert_transcript(
            TranscriptRow(
                job_id=self.job.id,
                json_path=str(self.workspace.transcript),
                language=transcript.language,
                model=transcript.model,
                has_diarization=transcript.has_diarization,
                word_count=len(transcript.words),
                source=transcript.source,
            )
        )

        self._finish_stage(stage)
        return transcript

    def _youtube_caption_transcript(self) -> Transcript | None:
        """YouTube's own captions saved at ingest, when wanted and trustworthy.

        Any doubt (setting off, not a YouTube source, no track, a track that
        fails the checks, an error) returns None, and Whisper runs instead.
        """
        if not self.settings.ingest.prefer_youtube_captions or self.source.type != "youtube":
            return None
        try:
            transcript = yt_captions.load_for_source(
                Path(self.source.path).parent,
                duration_s=self.source.duration_s,
                language=self.settings.whisper.language,
            )
        except Exception:
            log.warning("Could not read YouTube's captions; using Whisper.", exc_info=True)
            return None
        if transcript is not None:
            log.info("Using %s as the transcript for job %s.", transcript.model, self.job.id)
        return transcript

    def _load_or_build_audience(
        self, loudness: Loudness | None
    ) -> audience_signals.AudienceCurve | None:
        """The job's audience curve, built once and cached. Never raises."""
        detection = self.settings.detection
        if not detection.audience_signals:
            return None
        cache = self.workspace.audience
        if cache.exists():
            try:
                return audience_signals.AudienceCurve.load(cache)
            except (OSError, ValueError, TypeError, KeyError):
                log.warning("Ignoring an unreadable audience curve; building it again.")
        ingest_settings = self.settings.ingest

        def fetch(url: str) -> list[audience_signals.Comment]:
            return audience_signals.fetch_comments(
                url,
                max_comments=detection.max_comments,
                cookies_from_browser=ingest_settings.cookies_from_browser,
            )

        try:
            curve = audience_signals.build_for_source(
                source_dir=Path(self.source.path).parent,
                url=self.source.url,
                is_youtube=self.source.type == "youtube",
                duration_s=self.source.duration_s,
                loudness_values=loudness.values if loudness else None,
                loudness_step_s=loudness.step_s if loudness else 0.1,
                use_comments=detection.fetch_comments and detection.max_comments > 0,
                use_energy=detection.audio_energy,
                fetch=fetch,
            )
            curve.save(cache)
        except Exception:
            log.warning("Building the audience curve failed; going without it.", exc_info=True)
            return None
        return curve

    def _build_judge(self, provider: object, max_clips: int) -> clip_judge.ClipJudge | None:
        detection = self.settings.detection
        complete = getattr(provider, "_complete", None)
        if not detection.judge or complete is None:
            return None
        return clip_judge.ClipJudge(
            complete,
            max_candidates=max(detection.judge_max_candidates, 2 * max_clips),
            min_keep=detection.judge_min_keep,
            title=self.source.title or "",
        )

    def _load_or_detect_silences(self, audio: Path) -> list[Silence]:
        if self.workspace.silences.exists():
            raw = json.loads(self.workspace.silences.read_text(encoding="utf-8"))
            return [Silence(**item) for item in raw]

        silences = prepare.detect_silences(audio)
        self.workspace.silences.write_text(
            json.dumps([{"start": s.start, "end": s.end} for s in silences]),
            encoding="utf-8",
        )
        return silences

    def _load_or_measure_loudness(self, audio: Path) -> Loudness:
        if self.workspace.loudness.exists():
            try:
                raw = json.loads(self.workspace.loudness.read_text(encoding="utf-8"))
                return Loudness.from_dict(raw)
            except (ValueError, TypeError, AttributeError):
                log.warning("Ignoring an unreadable loudness profile; measuring again.")

        loudness = prepare.measure_loudness(audio)
        # An empty profile means ffmpeg failed; don't cache that, retry next run.
        if loudness:
            self.workspace.loudness.write_text(json.dumps(loudness.to_dict()), encoding="utf-8")
        return loudness

    async def _stage_highlights(
        self, transcript: Transcript, silences: list[Silence], audio: Path | None = None
    ) -> list[Clip]:
        stage = Stage.HIGHLIGHTS
        self._check_cancelled()

        existing = store.list_clips(self.job.id)
        if existing:
            log.info("Reusing %d existing clips for job %s.", len(existing), self.job.id)
            self._finish_stage(stage)
            return existing

        provider_name = self.job.provider or self.settings.active_provider
        provider = build_provider(provider_name, self.settings)
        config = detection_config(self.settings)

        self._emit(stage, 0.0, f"Finding highlights with {provider.name}")
        loudness = self._load_or_measure_loudness(audio) if audio is not None else None

        detection = self.settings.detection
        audience = None
        curve = self._load_or_build_audience(loudness)
        if curve is not None and curve.has_data:
            audience = highlights.AudienceOptions(
                curve=curve,
                audience_weight=detection.audience_weight,
                energy_weight=detection.energy_weight,
                hints_per_window=detection.hints_per_window if detection.audience_hints else 0,
            )
            if detection.audience_hints:
                config.prompt_version = "highlight_v2"
        judge = self._build_judge(provider, config.max_clips)

        clips = await highlights.detect(
            transcript,
            provider,
            config,
            job_id=self.job.id,
            silences=silences,
            loudness=loudness,
            on_progress=self._stage_progress(stage),
            # DetectionCancelled is turned into JobCancelled by run().
            cancelled=self._is_cancelled,
            audience=audience,
            judge=judge,
        )
        if judge is not None:
            try:
                self.workspace.judge_report.write_text(
                    json.dumps(judge.report(), indent=1), encoding="utf-8"
                )
            except OSError:
                log.debug("Could not write the judge report.", exc_info=True)

        store.replace_clips(self.job.id, clips)
        self._finish_stage(stage)
        return clips

    def _stage_reframe(self, clips: list[Clip], transcript: Transcript) -> dict[str, CropPath]:
        stage = Stage.REFRAME
        self._check_cancelled()
        source_path = Path(self.source.path)

        crop_paths: dict[str, CropPath] = {}

        if not self.source.has_video:
            # Audio-only sources render as captions on a solid background, so
            # there is nothing to reframe.
            self._finish_stage(stage)
            return crop_paths

        config = ReframeConfig(
            aspect_w=9 if self.settings.export.ratio == "9:16" else 1,
            aspect_h=16 if self.settings.export.ratio == "9:16" else 1,
        )
        if self.settings.export.ratio == "16:9":
            config = ReframeConfig(aspect_w=16, aspect_h=9)

        detect_cams = wants_facecam(
            self.settings.export.layout, self.settings.export.ratio, self.source.has_video
        )

        for index, clip in enumerate(clips):
            self._check_cancelled()
            if detect_cams:
                facecam_for_clip(self.workspace, clip, self.source)
            cached = self.workspace.crop_path(clip.id)
            if cached.exists():
                crop_paths[clip.id] = CropPath.load(cached)
            else:
                crop_paths[clip.id] = build_crop_path(
                    source_path,
                    start_s=clip.start_s,
                    end_s=clip.end_s,
                    transcript=transcript,
                    config=config,
                )
                crop_paths[clip.id].save(cached)

            self._emit(stage, (index + 1) / len(clips), f"Reframing clip {index + 1}")

        self._finish_stage(stage)
        return crop_paths

    def _stage_captions(self, clips: list[Clip], transcript: Transcript) -> None:
        stage = Stage.CAPTIONS
        self._check_cancelled()
        # Caption files are written during export, where the output dimensions
        # are known. This stage validates the style so a typo fails fast rather
        # than after the reframe work is already done.
        captions.get_style(self.settings.export.caption_style)
        self._finish_stage(stage)

    def _stage_export(
        self,
        clips: list[Clip],
        transcript: Transcript,
        crop_paths: dict[str, CropPath],
    ) -> None:
        stage = Stage.EXPORT
        self._check_cancelled()

        style = captions.get_style(self.settings.export.caption_style)
        ratio = self.settings.export.ratio
        source_path = Path(self.source.path)
        destination_dir = paths.exports_dir() / self.job.id
        destination_dir.mkdir(parents=True, exist_ok=True)

        for index, clip in enumerate(clips):
            self._check_cancelled()

            if self._already_exported(clip, ratio):
                # A retry after a failure at clip 7 must not re-render clips
                # 1-6 (minutes each) or stack duplicate export records on them.
                self._emit(stage, (index + 1) / len(clips), f"Exporting clip {index + 1}")
                continue

            crop_path = crop_paths.get(clip.id) or self._fallback_crop_path(clip, ratio)
            words = transcript.slice(clip.start_word, clip.end_word)
            facecam = (
                facecam_for_clip(self.workspace, clip, self.source)
                if wants_facecam(self.settings.export.layout, ratio, self.source.has_video)
                else None
            )
            destination = destination_dir / export.output_filename(
                clip.title or f"clip-{clip.rank}", ratio, rank=clip.rank
            )

            request = export.ExportRequest(
                source=source_path,
                destination=destination,
                start_s=clip.start_s,
                end_s=clip.end_s,
                crop_path=crop_path,
                words=words,
                style=style,
                ratio=ratio,
                has_video=self.source.has_video,
                layout=export.resolve_layout(self.settings.export.layout, facecam, ratio),
                facecam=facecam,
            )

            def clip_progress(fraction: float, i: int = index) -> None:
                self._emit(stage, (i + fraction) / len(clips), f"Exporting clip {i + 1}")

            export.export_clip(
                request,
                work_dir=self.workspace.captions_dir,
                settings=self.settings.export,
                on_progress=clip_progress,
                cancelled=self._is_cancelled,
            )

            store.create_export(
                Export(
                    id=new_id(),
                    clip_id=clip.id,
                    path=str(destination),
                    ratio=ratio,
                    style=style.key,
                    size_bytes=destination.stat().st_size,
                )
            )
            store.update_clip(clip.id, status="exported")

        self._finish_stage(stage)

    @staticmethod
    def _already_exported(clip: Clip, ratio: str) -> bool:
        """True when an earlier run of this job already rendered ``clip``."""
        if clip.status != "exported":
            return False
        return any(
            record.ratio == ratio and Path(record.path).exists()
            for record in store.list_exports(clip.id)
        )

    def _fallback_crop_path(self, clip: Clip, ratio: str) -> CropPath:
        """Centre crop for sources with no reframe data (audio-only, or a failure)."""
        from .reframe.croppath import centre_crop

        width, height = export.ratio_dimensions(ratio)
        info = ffmpeg.probe(Path(self.source.path))
        return centre_crop(
            info.width or width,
            info.height or height,
            clip.end_s - clip.start_s,
            aspect_w=9 if ratio == "9:16" else (1 if ratio == "1:1" else 16),
            aspect_h=16 if ratio == "9:16" else (1 if ratio == "1:1" else 9),
        )


async def run_job(
    job_id: str,
    *,
    settings: Settings | None = None,
    on_progress: ProgressHandler | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Clip]:
    """Load a job and run it to completion."""
    job = store.get_job(job_id)
    if job is None:
        raise PipelineError(f"Job {job_id} not found.", stage=Stage.PREPARE)

    source = store.get_source(job.source_id)
    if source is None:
        raise PipelineError(f"Source {job.source_id} not found.", stage=Stage.PREPARE)

    runner = PipelineRunner(
        job, source, settings=settings, on_progress=on_progress, is_cancelled=is_cancelled
    )
    # The pipeline is mostly blocking work (ffmpeg, Whisper, MediaPipe) with one
    # async stage. Running it in a worker thread keeps the web server's event
    # loop responsive while a job is going.
    return await asyncio.get_running_loop().run_in_executor(None, _run_sync, runner)


def _run_sync(runner: PipelineRunner) -> list[Clip]:
    return asyncio.run(runner.run())
