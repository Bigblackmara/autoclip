"""Clip review, editing, preview media, variants and export."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import paths
from ..config import load as load_settings
from ..db import store
from ..db.models import Clip, ClipEdit, Export, new_id
from ..pipeline import captions as captions_module
from ..pipeline import edit as edit_module
from ..pipeline import export as export_module
from ..pipeline import language as language_module
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
from ..providers.base import normalise_hashtags
from .schemas import (
    CaptionPatchIn,
    CaptionStyleOut,
    ClipOut,
    ClipPatchIn,
    ExportOut,
    ExportRequestIn,
    VariantIn,
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


def _flash_forward_default() -> bool:
    """The export setting a clip without its own flash-forward toggle follows."""
    return load_settings().export.flash_forward


def _clip_out(
    clip,
    defaults: tuple[str, str] | None = None,
    layout: str | None = None,
    flash_default: bool | None = None,
) -> ClipOut:
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
        flash_forward_default=(
            flash_default if flash_default is not None else _flash_forward_default()
        ),
    )


async def list_clips_for_job(job_id: str) -> list[ClipOut]:
    job = await asyncio.to_thread(store.get_job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    clips = await asyncio.to_thread(store.list_clips, job_id)
    defaults = await asyncio.to_thread(_job_export_defaults, job_id)
    layout = await asyncio.to_thread(_job_layout, job_id)
    flash = await asyncio.to_thread(_flash_forward_default)
    return [await asyncio.to_thread(_clip_out, clip, defaults, layout, flash) for clip in clips]


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
    """Trim, retitle, keep/discard, pick a hook, edit post text, toggle flash-forward.

    A trim re-derives the word range from the new times, so captions stay in
    step with the boundaries without the client having to compute indices.
    The chosen hook may be one of the clip's options or free text.
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
        hook_text=" ".join(payload.hook_text.split()) if payload.hook_text is not None else None,
        description=payload.description.strip() if payload.description is not None else None,
        hashtags=normalise_hashtags(payload.hashtags) if payload.hashtags is not None else None,
        flash_forward=(
            payload.flash_forward if "flash_forward" in payload.model_fields_set else store.UNSET
        ),
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
    record = await _render(
        clip,
        ratio=payload.ratio,
        style_key=payload.style,
        layout_setting=payload.layout,
        write_srt=payload.write_srt,
    )
    return ExportOut.of(record)


@router.post("/clips/{clip_id}/variant", response_model=ClipOut, status_code=201)
async def make_variant(clip_id: str, payload: VariantIn | None = None) -> ClipOut:
    """A new clip for the same moment that can be posted again without being a duplicate.

    The variant takes the next caption style (cycling through the presets),
    the next zoom pattern (``motion_seed``) and the next hook option, keeps
    the original's trim, caption edits, post text and flash-forward choice,
    and is rendered straight away unless ``export`` is false.
    """
    payload = payload or VariantIn()
    parent = await asyncio.to_thread(store.get_clip, clip_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    variant = await asyncio.to_thread(_create_variant, parent)
    if payload.export:
        edit = await asyncio.to_thread(store.get_clip_edit, variant.id)
        default_style, default_ratio = await asyncio.to_thread(_job_export_defaults, variant.job_id)
        await _render(
            variant,
            ratio=payload.ratio or (edit.ratio if edit else default_ratio),
            style_key=edit.caption_style if edit else default_style,
            layout_setting=payload.layout,
            write_srt=False,
        )
    updated = await asyncio.to_thread(store.get_clip, variant.id)
    return await asyncio.to_thread(_clip_out, updated)


def next_style(current: str) -> str:
    """The preset after ``current`` in the preset order (wrapping round)."""
    keys = list(captions_module.PRESETS)
    if current not in keys:
        return keys[0]
    return keys[(keys.index(current) + 1) % len(keys)]


def next_hook(clip: Clip) -> str:
    """The hook option after the one ``clip`` shows; its own headline when there are none."""
    options = [o for o in clip.hook_options if o.strip()]
    if not options:
        return clip.hook_text
    current = clip.headline
    if current in options:
        return options[(options.index(current) + 1) % len(options)]
    return options[0]


def _create_variant(parent: Clip) -> Clip:
    """Insert the variant row (plus its caption edits and cached framing)."""
    root_id = parent.variant_of or parent.id
    root = store.get_clip(root_id) or parent
    family = [
        c for c in store.list_clips(parent.job_id) if c.id == root_id or c.variant_of == root_id
    ]
    number = sum(1 for c in family if c.variant_of == root_id) + 1
    seed = max((c.motion_seed for c in family), default=0) + 1

    edit = store.get_clip_edit(parent.id)
    default_style, default_ratio = _job_export_defaults(parent.job_id)
    style = next_style(edit.caption_style if edit else default_style)

    variant = Clip(
        id=new_id(),
        job_id=parent.job_id,
        rank=parent.rank,
        start_s=parent.start_s,
        end_s=parent.end_s,
        start_word=parent.start_word,
        end_word=parent.end_word,
        title=f"{root.title or f'Clip {root.rank}'} (variant {number})",
        hook=parent.hook,
        score=parent.score,
        reason=parent.reason,
        status="kept",
        user_trimmed=parent.user_trimmed,
        hook_options=list(parent.hook_options),
        hook_text=next_hook(parent),
        description=parent.description,
        hashtags=list(parent.hashtags),
        teaser_start_s=parent.teaser_start_s,
        teaser_end_s=parent.teaser_end_s,
        flash_forward=parent.flash_forward,
        variant_of=root_id,
        motion_seed=seed,
    )
    store.create_clip(variant)
    store.upsert_clip_edit(
        ClipEdit(
            clip_id=variant.id,
            edited_words=edit.edited_words if edit else None,
            caption_style=style,
            ratio=edit.ratio if edit else default_ratio,
        )
    )
    # The same span, so the same framing: without these the variant would be
    # re-framed from scratch (or fall back to a centre crop). A joined clip's
    # parts come along too, or its variant would render the whole envelope.
    workspace = JobWorkspace(parent.job_id)
    for cached in (workspace.crop_path, workspace.facecam, workspace.segments):
        source_file, target_file = cached(parent.id), cached(variant.id)
        if source_file.exists():
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target_file)
            except OSError:
                log.warning("Could not copy %s for variant %s.", source_file.name, variant.id)
    return variant


async def _render(
    clip: Clip,
    *,
    ratio: str,
    style_key: str,
    layout_setting: str | None,
    write_srt: bool,
) -> Export:
    """Render one clip with the current export settings and record the file."""
    clip_id = clip.id
    job = await asyncio.to_thread(store.get_job, clip.job_id)
    source = await asyncio.to_thread(store.get_source, job.source_id) if job else None
    if job is None or source is None:
        raise HTTPException(status_code=404, detail="The clip's source is missing.")

    try:
        style = captions_module.get_style(style_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    words = [
        Word(text=w.text, start=w.start, end=w.end, speaker=w.speaker)
        for w in await clip_words(clip_id)
    ]

    settings = load_settings()
    settings.export.write_srt = write_srt

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
        _crop_path_for, workspace, clip, source, ratio, duration_s=end_s - start_s
    )

    layout_setting = layout_setting or await asyncio.to_thread(_job_layout, clip.job_id)
    facecam = None
    if wants_facecam(layout_setting, ratio, source.has_video):
        # Cached from the job's reframe stage when it ran with a split-capable
        # layout; detected now (and cached) otherwise.
        facecam = await asyncio.to_thread(facecam_for_clip, workspace, clip, source)

    destination = (
        paths.exports_dir()
        / clip.job_id
        / export_module.output_filename(clip.title or f"clip-{clip.rank}", ratio, rank=clip.rank)
    )

    layout = export_module.resolve_layout(layout_setting, facecam, ratio)
    # Caption language (Hinglish / romanised / translated) for the captions,
    # the hook and the post text, as the batch export does.
    transcript_row = await asyncio.to_thread(store.get_transcript, clip.job_id)
    localised = await asyncio.to_thread(
        language_module.for_export,
        words,
        settings.export.caption_language,
        language=transcript_row.language if transcript_row else "",
        hook=clip.headline,
        description=clip.description,
        provider_name=job.provider or settings.active_provider,
        settings=settings,
        cache_dir=workspace.root / "caption_language",
    )
    if localised.description != clip.description:
        await asyncio.to_thread(store.update_clip, clip_id, description=localised.description)
    # The same short-form edits as the batch export (pause cuts, zooms, hook,
    # key words, flash-forward, music, watermark), from the current settings.
    plan = await asyncio.to_thread(
        edit_module.plan_for_clip,
        settings.export,
        start_s=start_s,
        end_s=end_s,
        words=words,
        silences=silences,
        title=localised.hook,
        seed=clip.id,
        allow_motion=source.has_video and layout != "split" and not crop_path.has_split,
        display_words=localised.words if localised.applied else None,
        flash_forward=clip.flash_forward,
        # Source-time tease line; on a joined clip's own timeline when it has one.
        teaser=segments_module.range_to_joined(clip.teaser, spans) if spans else clip.teaser,
        motion_seed=clip.motion_seed,
    )

    request = export_module.ExportRequest(
        source=Path(source.path),
        destination=destination,
        start_s=start_s,
        end_s=end_s,
        spans=segments_module.as_pairs(spans) if spans else None,
        crop_path=crop_path,
        words=localised.words,
        style=style,
        ratio=ratio,
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
        ratio=ratio,
        style=style.key,
        size_bytes=destination.stat().st_size,
    )
    await asyncio.to_thread(store.create_export, record)
    await asyncio.to_thread(store.update_clip, clip_id, status="exported")
    return record


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
