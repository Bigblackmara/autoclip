"""Clip review, editing, preview media, and export."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import paths
from ..config import load as load_settings
from ..db import store
from ..db.models import Export, new_id
from ..pipeline import captions as captions_module
from ..pipeline import edit as edit_module
from ..pipeline import export as export_module
from ..pipeline import segments as segments_module
from ..pipeline.prepare import Silence
from ..pipeline.reframe.croppath import CropPath, centre_crop
from ..pipeline.runner import (
    JobWorkspace,
    clip_spans,
    facecam_for_clip,
    settings_for_job,
    wants_facecam,
)
from ..pipeline.transcript import Transcript, Word
from .schemas import (
    CaptionPatchIn,
    CaptionStyleOut,
    ClipOut,
    ClipPatchIn,
    ExportOut,
    ExportRequestIn,
    WordOut,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["clips"])


def _job_export_defaults(job_id: str) -> tuple[str, str]:
    """The caption style and ratio a job actually rendered with.

    A clip with no edits yet was exported with its job's settings (per-job
    overrides layered on the saved ones), not with the hard-coded defaults —
    reporting "bold_pop"/"9:16" there made the review page pre-select the
    wrong ratio and style, so "Export this clip" re-rendered differently.
    """
    job = store.get_job(job_id)
    export = settings_for_job(job).export if job else load_settings().export
    return export.caption_style, export.ratio


def _job_layout(job_id: str) -> str:
    """The layout setting ("auto" / "split" / "crop") a job was created with."""
    job = store.get_job(job_id)
    return (settings_for_job(job).export if job else load_settings().export).layout


def _clip_out(clip, defaults: tuple[str, str] | None = None, layout: str | None = None) -> ClipOut:
    edit = store.get_clip_edit(clip.id)
    exports = store.list_exports(clip.id)
    style, ratio = defaults or _job_export_defaults(clip.job_id)
    return ClipOut.of(
        clip,
        edit=edit,
        exports=exports,
        default_style=style,
        default_ratio=ratio,
        default_layout=layout or _job_layout(clip.job_id),
    )


async def list_clips_for_job(job_id: str) -> list[ClipOut]:
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    clips = await asyncio.to_thread(store.list_clips, job_id)
    defaults = await asyncio.to_thread(_job_export_defaults, job_id)
    layout = await asyncio.to_thread(_job_layout, job_id)
    return [await asyncio.to_thread(_clip_out, clip, defaults, layout) for clip in clips]


@router.get("/clips/{clip_id}", response_model=ClipOut)
async def get_clip(clip_id: str) -> ClipOut:
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return await asyncio.to_thread(_clip_out, clip)


@router.get("/clips/{clip_id}/words", response_model=list[WordOut])
async def clip_words(clip_id: str) -> list[WordOut]:
    """Words for a clip — the caption editor's source of truth.

    Returns the user's edited words when they exist, falling back to the
    transcript's originals.
    """
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")

    edit = await asyncio.to_thread(store.get_clip_edit, clip_id)
    if edit is not None and edit.edited_words:
        return [WordOut(**word) for word in edit.edited_words]

    transcript = await asyncio.to_thread(_load_transcript, clip.job_id)
    words = transcript.slice(clip.start_word, clip.end_word)
    spans = await asyncio.to_thread(clip_spans, JobWorkspace(clip.job_id), clip)
    if spans:
        # A joined clip: only the words its parts actually play.
        words = segments_module.words_in(words, spans)
    return [WordOut(text=w.text, start=w.start, end=w.end, speaker=w.speaker) for w in words]


@router.get("/clips/{clip_id}/crop-path")
async def clip_crop_path(clip_id: str) -> dict:
    """The computed crop path for a clip, so the preview can match the export.

    Without this the review player can only centre-crop, which on a tracked shot
    shows different framing from the file that gets rendered — sometimes with
    the speaker half out of frame. Reviewing framing against the wrong framing
    is worse than not previewing it at all.
    """
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")

    cached = JobWorkspace(clip.job_id).crop_path(clip.id)
    if not cached.exists():
        # Audio-only sources and pre-reframe jobs legitimately have none; the
        # client falls back to a centre crop.
        raise HTTPException(status_code=404, detail="No crop path for this clip yet.")

    return (await asyncio.to_thread(CropPath.load, cached)).to_dict()


@router.patch("/clips/{clip_id}", response_model=ClipOut)
async def patch_clip(clip_id: str, payload: ClipPatchIn) -> ClipOut:
    """Trim, retitle, or keep/discard a clip.

    A trim re-derives the word range from the new times, so captions stay in
    step with the boundaries without the client having to compute indices.
    """
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")

    start_s = payload.start_s if payload.start_s is not None else clip.start_s
    end_s = payload.end_s if payload.end_s is not None else clip.end_s
    if end_s <= start_s:
        raise HTTPException(status_code=400, detail="End must come after start.")

    trimmed = payload.start_s is not None or payload.end_s is not None
    start_word = end_word = None

    if trimmed:
        transcript = await asyncio.to_thread(_load_transcript, clip.job_id)
        first, last = transcript.indices_in_range(start_s, end_s)
        start_word, end_word = first, max(first, last - 1)

    await asyncio.to_thread(
        store.update_clip,
        clip_id,
        start_s=start_s if trimmed else None,
        end_s=end_s if trimmed else None,
        start_word=start_word,
        end_word=end_word,
        title=payload.title,
        status=payload.status,
        user_trimmed=True if trimmed else None,
    )

    updated = await asyncio.to_thread(store.get_clip, clip_id)
    return await asyncio.to_thread(_clip_out, updated)


@router.patch("/clips/{clip_id}/captions", response_model=ClipOut)
async def patch_captions(clip_id: str, payload: CaptionPatchIn) -> ClipOut:
    """Store word-level caption edits and style choices for a clip."""
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")

    if payload.caption_style is not None:
        try:
            captions_module.get_style(payload.caption_style)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing = await asyncio.to_thread(store.get_clip_edit, clip_id)
    from ..db.models import ClipEdit

    # The first edit of a clip must keep whatever the user didn't change at
    # the job's values; falling back to "bold_pop"/"9:16" meant picking a
    # ratio silently reset a job's karaoke captions (and vice versa).
    default_style, default_ratio = await asyncio.to_thread(_job_export_defaults, clip.job_id)
    edit = ClipEdit(
        clip_id=clip_id,
        edited_words=(
            [w.model_dump() for w in payload.words]
            if payload.words is not None
            else (existing.edited_words if existing else None)
        ),
        caption_style=payload.caption_style
        or (existing.caption_style if existing else default_style),
        ratio=payload.ratio or (existing.ratio if existing else default_ratio),
    )
    await asyncio.to_thread(store.upsert_clip_edit, edit)

    return await asyncio.to_thread(_clip_out, clip)


@router.post("/clips/{clip_id}/export", response_model=ExportOut, status_code=201)
async def export_clip(clip_id: str, payload: ExportRequestIn) -> ExportOut:
    """Render one clip and return its download link."""
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")

    job = await asyncio.to_thread(store.get_job, clip.job_id)
    source = await asyncio.to_thread(store.get_source, job.source_id) if job else None
    if job is None or source is None:
        raise HTTPException(status_code=404, detail="The clip's source is missing.")

    try:
        style = captions_module.get_style(payload.style)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    words = [
        Word(text=w.text, start=w.start, end=w.end, speaker=w.speaker)
        for w in await clip_words(clip_id)
    ]

    settings = load_settings()
    settings.export.write_srt = payload.write_srt

    workspace = JobWorkspace(clip.job_id)
    # A joined clip renders on its own timeline (see pipeline.segments).
    spans = await asyncio.to_thread(clip_spans, workspace, clip)
    start_s, end_s = clip.start_s, clip.end_s
    silences = await asyncio.to_thread(_cached_silences, workspace)
    if spans:
        words = segments_module.words_on_timeline(words, spans)
        start_s, end_s = 0.0, segments_module.total(spans)
        silences = segments_module.silences_on_timeline(silences, spans)
    crop_path = await asyncio.to_thread(
        _crop_path_for, workspace, clip, source, payload.ratio, duration_s=end_s - start_s
    )

    layout_setting = payload.layout or await asyncio.to_thread(_job_layout, clip.job_id)
    facecam = None
    if wants_facecam(layout_setting, payload.ratio, source.has_video):
        # Cached from the job's reframe stage when it ran with a split-capable
        # layout; detected now (and cached) otherwise.
        facecam = await asyncio.to_thread(facecam_for_clip, workspace, clip, source)

    destination = (
        paths.exports_dir()
        / clip.job_id
        / export_module.output_filename(
            clip.title or f"clip-{clip.rank}", payload.ratio, rank=clip.rank
        )
    )

    layout = export_module.resolve_layout(layout_setting, facecam, payload.ratio)
    # The same short-form edits as the batch export (pause cuts, zooms, hook,
    # key words, music, watermark), from the current export settings.
    plan = await asyncio.to_thread(
        edit_module.plan_for_clip,
        settings.export,
        start_s=start_s,
        end_s=end_s,
        words=words,
        silences=silences,
        title=clip.title or clip.hook,
        seed=clip.id,
        allow_motion=source.has_video and layout != "split" and not crop_path.has_split,
    )

    request = export_module.ExportRequest(
        source=Path(source.path),
        destination=destination,
        start_s=start_s,
        end_s=end_s,
        spans=segments_module.as_pairs(spans) if spans else None,
        crop_path=crop_path,
        words=words,
        style=style,
        ratio=payload.ratio,
        has_video=source.has_video,
        layout=layout,
        facecam=facecam,
        has_audio=source.has_audio,
        edit=plan,
    )

    try:
        await asyncio.to_thread(
            export_module.export_clip,
            request,
            work_dir=workspace.captions_dir,
            settings=settings.export,
        )
    except export_module.ExportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record = Export(
        id=new_id(),
        clip_id=clip_id,
        path=str(destination),
        ratio=payload.ratio,
        style=style.key,
        size_bytes=destination.stat().st_size,
    )
    await asyncio.to_thread(store.create_export, record)
    await asyncio.to_thread(store.update_clip, clip_id, status="exported")

    return ExportOut.of(record)


@router.get("/exports/{export_id}/download")
async def download_export(export_id: str) -> FileResponse:
    record = await asyncio.to_thread(store.get_export, export_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Export not found.")

    path = Path(record.path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="The exported file has been moved or deleted.")

    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/jobs/{job_id}/media")
async def job_media(job_id: str) -> FileResponse:
    """Serve the source video, so the review player can scrub the original.

    Range requests are handled by FileResponse, which is what makes seeking in
    the preview player usable on a long source.
    """
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    source = await asyncio.to_thread(store.get_source, job.source_id)
    if source is None or not Path(source.path).exists():
        raise HTTPException(status_code=404, detail="Source media not found.")

    return FileResponse(Path(source.path))


@router.get("/caption-styles", response_model=list[CaptionStyleOut])
async def caption_styles() -> list[CaptionStyleOut]:
    """Style presets, with enough detail for the UI to approximate them in CSS."""
    return [
        CaptionStyleOut(
            key=style.key,
            label=style.label,
            description=style.description,
            preview={
                "font": style.font,
                "primary": style.primary,
                "accent": style.accent,
                "outline": style.outline,
                "outlineWidth": style.outline_width,
                "allCaps": style.all_caps,
                "sizeRatio": style.size_ratio,
                "marginRatio": style.margin_v_ratio,
                "boxed": style.boxed,
                "animation": style.animation,
                "maxWords": style.max_words,
            },
        )
        for style in captions_module.PRESETS.values()
    ]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _cached_silences(workspace: JobWorkspace) -> list[Silence]:
    """The job's measured silences (for pause cuts); empty when not measured."""
    if not workspace.silences.exists():
        return []
    try:
        raw = json.loads(workspace.silences.read_text(encoding="utf-8"))
        return [Silence(start=float(item["start"]), end=float(item["end"])) for item in raw]
    except (OSError, ValueError, TypeError, KeyError):
        log.warning("Ignoring an unreadable silences file in %s.", workspace.root)
        return []


def _load_transcript(job_id: str) -> Transcript:
    workspace = JobWorkspace(job_id)
    if not workspace.transcript.exists():
        raise HTTPException(status_code=404, detail="This job has no transcript yet.")
    return Transcript.load(workspace.transcript)


#: Slack for the cached crop path's span versus the clip duration, covering
#: float round-tripping through JSON — far below any real trim.
_CROP_DURATION_TOLERANCE_S = 0.05

#: Slack for a crop window's aspect ratio: windows are forced to even pixel
#: sizes, which moves 9:16 on a 1080p source by well under 1 %.
_CROP_ASPECT_TOLERANCE = 0.02


def _crop_path_fits(path: CropPath, duration_s: float, aspect: tuple[int, int]) -> bool:
    """Whether a cached crop path was built for this clip span and ratio."""
    if not path.segments or abs(path.duration_s - duration_s) > _CROP_DURATION_TOLERANCE_S:
        return False
    target = aspect[0] / aspect[1]
    for segment in path.segments:
        # Fit segments letterbox the whole frame and suit any output ratio.
        if segment.fit or not segment.height:
            continue
        if abs(segment.width / segment.height - target) > target * _CROP_ASPECT_TOLERANCE:
            return False
    return True


def _crop_path_for(
    workspace: JobWorkspace, clip, source, ratio: str, *, duration_s: float | None = None
) -> CropPath:
    """Reuse the job's cached crop path, falling back to a centre crop.

    A re-export at a different ratio can't reuse a path computed for the
    original one, so the geometry is recomputed rather than stretched.

    The cache is checked against what it was built for, not assumed: it is
    computed at the *job's* ratio (not always 9:16) and for the clip's
    boundaries at reframe time. Reusing a 1:1 path for a 9:16 export
    stretched every frame, and reusing a path after the user trimmed the
    clip shifted the framing in time and cut the video short of the audio
    (or trimmed into an empty segment) once the duration changed.
    """
    aspect = {"9:16": (9, 16), "1:1": (1, 1), "16:9": (16, 9)}[ratio]
    # A joined clip's path spans its parts, not its envelope.
    duration = duration_s if duration_s is not None else clip.end_s - clip.start_s
    cached = workspace.crop_path(clip.id)
    if cached.exists():
        path = CropPath.load(cached)
        if _crop_path_fits(path, duration, aspect):
            return path

    return centre_crop(
        source.width or 1920,
        source.height or 1080,
        duration,
        aspect_w=aspect[0],
        aspect_h=aspect[1],
    )
