"""Export stage — crop path plus captions plus audio normalisation to an MP4.

The whole render is a single ffmpeg pass. Per-shot crop segments are trimmed out
of the source, cropped independently, scaled to a common output size, and
concatenated in the filtergraph — which is what allows a wide two-shot and a
tight single to sit in one clip without a mid-stream frame-size change.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import ExportSettings
from ..system import report
from . import captions as captions_module
from . import edit as edit_module
from . import ffmpeg
from .captions import CaptionStyle
from .edit import EditPlan
from .reframe.croppath import CropPath, CropRect, CropSegment, segment_crop_filter, split_at
from .reframe.facecam import Facecam
from .transcript import Word

log = logging.getLogger(__name__)

#: Output dimensions per aspect ratio. 1080-wide is the platform sweet spot:
#: high enough to avoid re-encode mush, low enough to upload quickly.
RATIOS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

#: Loudness targets. -14 LUFS integrated with -1.5 dBTP is what every major
#: platform normalises toward, so hitting it ourselves avoids their processing.
LOUDNESS_TRUE_PEAK = -1.5
LOUDNESS_RANGE = 11.0

#: Height of the facecam panel in the split layout, as a fraction of the
#: output height. The panel takes the cam's own aspect within these bounds;
#: outside them the cam is cropped (never stretched) to fit.
SPLIT_TOP_MIN = 0.30
SPLIT_TOP_MAX = 0.42

#: Ratios the split layout supports. Square and landscape outputs have no room
#: for a stacked panel and keep the ordinary crop.
SPLIT_RATIOS = ("9:16",)

AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 48_000


class ExportError(RuntimeError):
    """Rendering a clip failed."""


@dataclass
class ExportRequest:
    source: Path
    destination: Path
    start_s: float
    end_s: float
    crop_path: CropPath
    words: list[Word]
    style: CaptionStyle
    ratio: str = "9:16"
    burn_captions: bool = True
    #: False for audio-only sources (podcasts, voice memos). The clip is then
    #: rendered as captions over a generated solid background.
    has_video: bool = True
    #: Resolved layout: "split" stacks ``facecam`` above the gameplay, "crop"
    #: (or anything else) renders ``crop_path``. Settings resolve "auto" before
    #: building the request — see :func:`resolve_layout`.
    layout: str = "crop"
    facecam: Facecam | None = None
    #: False when the source has no audio track (screen recordings, GIFs).
    has_audio: bool = True
    #: Short-form editing on top of the crop (pace cuts, zooms, hook, key
    #: words, music, watermark, forced fps, two-pass loudness). None renders
    #: the plain crop + captions export, as Live Watch does.
    edit: EditPlan | None = None
    #: A multi-part clip: source spans (absolute seconds, play order) joined
    #: into one video. When set, ``start_s``/``end_s``, ``words``,
    #: ``crop_path`` and ``edit`` are all on the *joined* timeline (0 = start
    #: of the first part; see :mod:`autoclip.pipeline.segments`), and each span
    #: is read as its own seeked input. None for an ordinary one-span clip.
    spans: list[tuple[float, float]] | None = None

    @property
    def input_count(self) -> int:
        """Inputs carrying the source: one per span, or the single seeked one."""
        return len(self.spans) if self.spans else 1

    @property
    def is_split(self) -> bool:
        return (
            self.layout == "split"
            and self.facecam is not None
            and self.has_video
            and self.ratio in SPLIT_RATIOS
        )

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def output_duration_s(self) -> float:
        """Length of the rendered clip (shorter than the span when pauses are cut)."""
        if self.edit is None:
            return self.duration_s
        return self.edit.output_duration(self.duration_s)

    @property
    def is_cut(self) -> bool:
        keeps = self.edit.keeps if self.edit is not None else None
        return bool(keeps) and (
            len(keeps) > 1 or keeps[0].start > 1e-3 or keeps[0].end < self.duration_s - 1e-3
        )


@dataclass(frozen=True)
class InputLayout:
    """ffmpeg input indices of the optional extra inputs (None when absent)."""

    background: int | None = None
    watermark: int | None = None
    music: int | None = None


def input_layout(request: ExportRequest) -> InputLayout:
    """Inputs are: the source (one per span for a joined clip), then
    background, watermark and music when used."""
    index = request.input_count
    background = watermark = music = None
    if not request.has_video:
        background, index = index, index + 1
    edit = request.edit
    if edit is not None and edit.watermark is not None:
        watermark, index = index, index + 1
    if edit is not None and edit.music is not None:
        music = index
    return InputLayout(background=background, watermark=watermark, music=music)


def resolve_layout(setting: str, facecam: Facecam | None, ratio: str) -> str:
    """Turn the ``layout`` setting into what actually renders: "split" or "crop".

    "auto" only splits for what looks like a real overlay (see
    :attr:`Facecam.is_overlay`), so talking-head videos, podcasts and anything
    ambiguous keep the crop.
    """
    if ratio not in SPLIT_RATIOS or facecam is None or setting == "crop":
        return "crop"
    if setting == "split":
        return "split"
    return "split" if facecam.is_overlay else "crop"


@dataclass(frozen=True)
class SplitGeometry:
    """Where each panel of the split layout comes from, in source pixels."""

    top_h: int
    cam: tuple[int, int, int, int]
    game: tuple[int, int, int, int]


def split_geometry(
    facecam: Facecam, crop_path: CropPath | None, out_w: int, out_h: int
) -> SplitGeometry:
    """Lay out the facecam panel (top) and the gameplay panel (bottom).

    The panel height follows the cam's own aspect, clamped to
    :data:`SPLIT_TOP_MIN`..:data:`SPLIT_TOP_MAX` of the output; the cam rect
    is then cropped to the panel's aspect around the face, so nothing is ever
    stretched. Gameplay fills the rest at full source height, centred where the
    crop path framed the clip (or the frame centre), and moved clear of the cam.
    """
    sw, sh = facecam.source_width, facecam.source_height
    cam_x, cam_y, cam_w, cam_h = facecam.x, facecam.y, facecam.width, facecam.height

    ideal = out_w * cam_h / cam_w if cam_w else out_h * SPLIT_TOP_MIN
    top_h = _even(min(max(ideal, out_h * SPLIT_TOP_MIN), out_h * SPLIT_TOP_MAX))
    panel_aspect = out_w / top_h

    face_cx = facecam.face_left + facecam.face_width / 2
    face_cy = facecam.face_top + facecam.face_height / 2
    if cam_w / cam_h > panel_aspect:
        # Cam is wider than the panel: trim the sides, keeping the face centred.
        crop_w = min(cam_w - cam_w % 2, _even(cam_h * panel_aspect))
        crop_h = cam_h - cam_h % 2
        crop_x = _clamp_int(face_cx - crop_w / 2, cam_x, cam_x + cam_w - crop_w)
        crop_y = cam_y
    else:
        # Cam is taller: trim top and bottom, face a little above centre.
        crop_w = cam_w - cam_w % 2
        crop_h = min(cam_h - cam_h % 2, _even(cam_w / panel_aspect))
        crop_x = cam_x
        crop_y = _clamp_int(face_cy - crop_h * 0.42, cam_y, cam_y + cam_h - crop_h)

    bottom_h = out_h - top_h
    game_h = sh - sh % 2
    game_w = min(sw - sw % 2, _even(game_h * out_w / bottom_h))
    centre = _gameplay_centre(crop_path, facecam) or sw / 2
    game_x = _clamp_int(centre - game_w / 2, 0, sw - game_w)

    # Keep the cam out of the gameplay panel when there's room to.
    cam_right = cam_x + cam_w
    if game_x < cam_right and game_x + game_w > cam_x:
        if cam_x + cam_w / 2 < sw / 2 and cam_right + game_w <= sw:
            game_x = cam_right
        elif cam_x - game_w >= 0:
            game_x = cam_x - game_w
    game_x = _clamp_int(game_x, 0, sw - game_w)

    return SplitGeometry(
        top_h=top_h,
        cam=(crop_x & ~1, crop_y & ~1, crop_w, crop_h),
        game=(game_x & ~1, 0, game_w, game_h),
    )


def _gameplay_centre(crop_path: CropPath | None, facecam: Facecam) -> float | None:
    """Time-weighted horizontal centre of the crop path, ignoring the cam.

    Segments that framed the facecam itself (the speaker tracker happily
    follows the webcam face) or fitted the whole frame say nothing about
    where the action is.
    """
    if crop_path is None:
        return None
    total = 0.0
    weighted = 0.0
    for segment in crop_path.segments:
        if segment.fit or not segment.keyframes:
            continue
        xs = [k.x + segment.width / 2 for k in segment.keyframes]
        centre = sum(xs) / len(xs)
        if facecam.x <= centre <= facecam.x + facecam.width:
            continue
        weight = max(segment.duration_s, 0.0)
        total += weight
        weighted += centre * weight
    return weighted / total if total > 0 else None


def split_caption_margin(style: CaptionStyle, out_h: int, top_h: int) -> float:
    """Caption bottom margin (fraction of height) that keeps text off the cam.

    Captions sit in the gameplay panel. The style's own margin is kept unless
    its text block (up to three lines) would reach up into the facecam panel.
    """
    block = 3 * style.size_ratio * 1.25
    ceiling = 1.0 - top_h / out_h - block - 0.02
    return max(0.04, min(style.margin_v_ratio, ceiling))


def speakers_caption_margin(style: CaptionStyle) -> float:
    """Caption bottom margin that centres a two-line block on the split's seam.

    In the two-speaker split the bottom of the frame is the second person's
    face; the seam between the halves is the one place text covers nobody.
    """
    return max(0.04, 0.5 - style.size_ratio * 1.25)


def source_inputs(request: ExportRequest) -> list[str]:
    """ffmpeg input arguments for the source: one seeked input per span.

    Seeking before ``-i`` decodes from the nearest keyframe and is far faster
    than an output-side seek; modern ffmpeg keeps it frame-accurate. Reading
    each part of a joined clip as its own input means the gaps between parts
    are never decoded at all.
    """
    if request.spans:
        args: list[str] = []
        for start, end in request.spans:
            args += ["-ss", f"{start:.4f}", "-t", f"{end - start:.4f}", "-i", str(request.source)]
        return args
    return [
        "-ss",
        f"{request.start_s:.4f}",
        "-t",
        f"{request.duration_s:.4f}",
        "-i",
        str(request.source),
    ]


def span_offsets(request: ExportRequest) -> list[float]:
    """Where each span starts on the joined timeline ([0.0] for a plain clip)."""
    if not request.spans:
        return [0.0]
    offsets: list[float] = []
    total = 0.0
    for start, end in request.spans:
        offsets.append(total)
        total += end - start
    return offsets


def _joined_video(request: ExportRequest, label: str) -> str:
    """Concatenate the span inputs' pictures into ``label``."""
    assert request.spans
    parts = [
        f"[{k}:v]trim=duration={end - start:.4f},setpts=PTS-STARTPTS[jv{k}]"
        for k, (start, end) in enumerate(request.spans)
    ]
    joined = "".join(f"[jv{k}]" for k in range(len(request.spans)))
    parts.append(f"{joined}concat=n={len(request.spans)}:v=1:a=0{label}")
    return ";".join(parts)


def joined_audio(request: ExportRequest) -> tuple[list[str], str]:
    """Filter parts producing the clip's speech, and the label carrying it.

    A plain clip reads ``[0:a]`` directly. A joined clip concatenates its span
    inputs, with 12 ms fades either side of every join so the cut never clicks.
    """
    if not request.spans:
        return [], "[0:a]"
    parts: list[str] = []
    for k, (start, end) in enumerate(request.spans):
        duration = end - start
        fade = min(0.012, duration / 4)
        parts.append(
            f"[{k}:a]atrim=duration={duration:.4f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade:.3f},"
            f"afade=t=out:st={duration - fade:.3f}:d={fade:.3f}[ja{k}]"
        )
    joined = "".join(f"[ja{k}]" for k in range(len(request.spans)))
    parts.append(f"{joined}concat=n={len(request.spans)}:v=0:a=1[ajoin]")
    return parts, "[ajoin]"


def _even(value: float) -> int:
    rounded = int(round(value))
    return max(2, rounded - rounded % 2)


def _clamp_int(value: float, low: float, high: float) -> int:
    return int(round(max(low, min(value, high)))) if high > low else int(low)


def ratio_dimensions(ratio: str) -> tuple[int, int]:
    if ratio not in RATIOS:
        raise ExportError(f"Unknown ratio {ratio!r}. Available: {', '.join(RATIOS)}")
    return RATIOS[ratio]


def slugify_title(title: str, *, max_length: int = 50) -> str:
    """Filename-safe slug for a clip title."""
    slug = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length] or "clip"


def output_filename(title: str, ratio: str, *, rank: int | None = None) -> str:
    """Output name for a clip; ``rank`` keeps one job's clips apart.

    All of a job's clips render into one directory, and titles collide more
    than you'd think: the model reuses a title, two titles slug the same
    ("Wait... what?!" / "wait what"), and emoji-only titles all become
    ``clip``. Without the rank the later clip silently overwrote the earlier
    one's file, leaving two export records pointing at one video.
    """
    stem = slugify_title(title)
    if rank is not None:
        stem = f"{rank:02d}-{stem}"
    return f"{stem}_{ratio.replace(':', 'x')}.mp4"


# --------------------------------------------------------------------------
# Filtergraph construction
# --------------------------------------------------------------------------


def build_video_filtergraph(
    request: ExportRequest,
    *,
    subtitle_name: str | None,
    fonts_name: str = "fonts",
) -> str:
    """Build the ``-filter_complex`` video chain for one clip."""
    out_w, out_h = ratio_dimensions(request.ratio)
    if not request.has_video:
        background = input_layout(request).background
        return _captions_chain(
            request,
            f"[{background}:v]setsar=1,format=yuv420p[vbg]",
            "[vbg]",
            subtitle_name=subtitle_name,
            fonts_name=fonts_name,
        )

    if request.is_split:
        if request.spans:
            graph = ";".join(
                [
                    _joined_video(request, "[vjoin]"),
                    _split_chain(request, out_w, out_h, source="[vjoin]"),
                ]
            )
        else:
            graph = _split_chain(request, out_w, out_h)
        return _captions_chain(
            request,
            graph,
            "[vstack]",
            subtitle_name=subtitle_name,
            fonts_name=fonts_name,
        )

    crop_path = request.crop_path
    offsets = span_offsets(request)
    if request.spans:
        # Every crop segment must come from exactly one part.
        crop_path = split_at(crop_path, offsets[1:])
    segments = crop_path.segments
    if not segments:
        raise ExportError("The crop path has no segments.")

    parts: list[str] = []
    labels: list[str] = []

    for index, segment in enumerate(segments):
        label = f"v{index}"
        labels.append(f"[{label}]")

        if request.spans:
            # Each part is its own input whose time starts at 0: trim the
            # segment out of the part it falls in, in that part's time.
            part = _span_index(offsets, (segment.start_s + segment.end_s) / 2)
            source_label = f"[s{index}]"
            parts.append(
                f"[{part}:v]trim=start={segment.start_s - offsets[part]:.4f}:"
                f"end={segment.end_s - offsets[part]:.4f},setpts=PTS-STARTPTS{source_label}"
            )
        elif len(segments) == 1:
            # No trim needed; the input is already seeked to the clip.
            source_label = "[0:v]"
        else:
            source_label = f"[s{index}]"
            parts.append(
                f"[0:v]trim=start={segment.start_s:.4f}:end={segment.end_s:.4f},"
                f"setpts=PTS-STARTPTS{source_label}"
            )

        if segment.fit:
            parts.extend(_fit_chain(source_label, index, out_w, out_h))
            continue

        if segment.is_split and request.ratio in SPLIT_RATIOS:
            parts.extend(_speakers_chain(source_label, segment, index, out_w, out_h))
            continue

        chain = [segment_crop_filter(segment)]
        if segment.zoom > 0:
            chain.append(_zoom_filter(segment.zoom, out_w, out_h))
        chain.append(f"scale={out_w}:{out_h}:flags=lanczos")
        chain.append("setsar=1,format=yuv420p")

        parts.append(f"{source_label}{','.join(chain)}[{label}]")

    if len(segments) == 1:
        current = "[v0]"
    else:
        parts.append(f"{''.join(labels)}concat=n={len(segments)}:v=1:a=0[vcat]")
        current = "[vcat]"

    return _captions_chain(
        request, ";".join(parts), current, subtitle_name=subtitle_name, fonts_name=fonts_name
    )


def _span_index(offsets: list[float], t: float) -> int:
    """Which part of a joined clip timeline position ``t`` falls in."""
    index = 0
    for k, offset in enumerate(offsets):
        if t >= offset:
            index = k
    return index


#: Colour and thickness of the line between the two halves of a speaker split.
SPEAKERS_DIVIDER = "0xFFF7EC@1"
SPEAKERS_DIVIDER_PX = 6


def _speakers_chain(
    source_label: str, segment: CropSegment, index: int, out_w: int, out_h: int
) -> list[str]:
    """Two-speaker split: each pane cropped steady, stacked top/bottom into ``[v{index}]``."""
    top, bottom = segment.panes
    top_h = _even(out_h / 2)
    bottom_h = out_h - top_h
    divider_y = top_h - SPEAKERS_DIVIDER_PX // 2

    def crop(rect: CropRect) -> str:
        return f"crop={rect.width}:{rect.height}:{int(round(rect.x))}:{int(round(rect.y))}"

    return [
        f"{source_label}split=2[pa{index}][pb{index}]",
        f"[pa{index}]{crop(top)},scale={out_w}:{top_h}:flags=lanczos,setsar=1[pt{index}]",
        f"[pb{index}]{crop(bottom)},scale={out_w}:{bottom_h}:flags=lanczos,setsar=1[pu{index}]",
        (
            f"[pt{index}][pu{index}]vstack=inputs=2,"
            f"drawbox=x=0:y={divider_y}:w={out_w}:h={SPEAKERS_DIVIDER_PX}:"
            f"color={SPEAKERS_DIVIDER}:t=fill,setsar=1,format=yuv420p[v{index}]"
        ),
    ]


def _split_chain(request: ExportRequest, out_w: int, out_h: int, *, source: str = "[0:v]") -> str:
    """Facecam panel stacked over a gameplay panel, producing ``[vstack]``."""
    assert request.facecam is not None
    geometry = split_geometry(request.facecam, request.crop_path, out_w, out_h)
    cx, cy, cw, ch = geometry.cam
    gx, gy, gw, gh = geometry.game
    bottom_h = out_h - geometry.top_h
    return ";".join(
        [
            f"{source}split=2[camsrc][gamesrc]",
            (
                f"[camsrc]crop={cw}:{ch}:{cx}:{cy},"
                f"scale={out_w}:{geometry.top_h}:flags=lanczos,setsar=1[cam]"
            ),
            (
                f"[gamesrc]crop={gw}:{gh}:{gx}:{gy},"
                f"scale={out_w}:{bottom_h}:flags=lanczos,setsar=1[game]"
            ),
            "[cam][game]vstack=inputs=2,format=yuv420p[vstack]",
        ]
    )


def _captions_chain(
    request: ExportRequest,
    graph: str,
    current: str,
    *,
    subtitle_name: str | None,
    fonts_name: str,
) -> str:
    """Append the edit chain and the caption burn (or a pass-through) producing ``[vout]``."""
    if request.edit is not None:
        edit_parts, current = _edit_video_chain(request, current)
        graph = ";".join([graph, *edit_parts])
    if subtitle_name is not None and (request.burn_captions or request.edit is not None):
        # Bare relative names — ffmpeg runs with its cwd set to the render
        # workspace, so there is nothing here that needs escaping.
        tail = f"{current}ass=filename={subtitle_name}:fontsdir={fonts_name}[vout]"
    else:
        tail = f"{current}null[vout]"
    return f"{graph};{tail}"


def _edit_video_chain(request: ExportRequest, current: str) -> tuple[list[str], str]:
    """Forced fps, pause cuts, punch-in zooms and the watermark, in that order.

    Cuts are applied after the crop so the crop path keeps working in clip
    time; zooms after the cuts so each jump cut can change the framing.
    """
    edit = request.edit
    assert edit is not None
    out_w, out_h = ratio_dimensions(request.ratio)
    parts: list[str] = []
    fps = edit.fps
    if request.is_cut:
        keeps = edit.kept(request.duration_s)
        if len(keeps) == 1:
            keep = keeps[0]
            parts.append(
                f"{current}fps={fps},trim=start={keep.start:.4f}:end={keep.end:.4f},"
                "setpts=PTS-STARTPTS[ecut]"
            )
        else:
            labels = "".join(f"[ek{i}]" for i in range(len(keeps)))
            parts.append(f"{current}fps={fps},split={len(keeps)}{labels}")
            for i, keep in enumerate(keeps):
                parts.append(
                    f"[ek{i}]trim=start={keep.start:.4f}:end={keep.end:.4f},"
                    f"setpts=PTS-STARTPTS[ec{i}]"
                )
            joined = "".join(f"[ec{i}]" for i in range(len(keeps)))
            parts.append(f"{joined}concat=n={len(keeps)}:v=1:a=0[ecut]")
        current = "[ecut]"
    else:
        parts.append(f"{current}fps={fps}[efps]")
        current = "[efps]"

    # Zooms would cut into the facecam panel of the split stack, and there is
    # nothing to zoom on an audio-only background.
    # ... nor into the halves of a two-speaker split.
    if (
        edit.has_motion
        and request.has_video
        and not request.is_split
        and not request.crop_path.has_split
    ):
        zoom = edit_module.zoom_filter(edit.zooms, edit.intro_zoom, out_w, out_h, fps)
        parts.append(f"{current}{zoom},setsar=1[ezoom]")
        current = "[ezoom]"

    inputs = input_layout(request)
    if edit.watermark is not None and inputs.watermark is not None:
        parts.append(
            edit_module.watermark_filter(
                edit.watermark, inputs.watermark, current, "[ewm]", out_w, out_h
            )
        )
        current = "[ewm]"
    return parts, current


def build_edit_audio_graph(
    request: ExportRequest, *, tail: str, music_index: int | None
) -> str | None:
    """The edited audio chain ending in ``[aout]``, or None when there is no audio.

    Speech is cut to the same keeps as the picture (12 ms fades so the joins
    don't click), music is looped/faded/ducked under it, and ``tail`` (the
    loudnorm, or its measuring variant) runs last over the whole mix.
    """
    edit = request.edit
    assert edit is not None
    parts: list[str] = []
    speech: str | None = None
    if request.has_audio:
        join_parts, audio_in = joined_audio(request)
        parts.extend(join_parts)
        if request.is_cut:
            keeps = edit.kept(request.duration_s)
            if len(keeps) > 1:
                labels = "".join(f"[ea{i}]" for i in range(len(keeps)))
                parts.append(f"{audio_in}asplit={len(keeps)}{labels}")
                sources = [f"[ea{i}]" for i in range(len(keeps))]
            else:
                sources = [audio_in]
            for i, (keep, source) in enumerate(zip(keeps, sources, strict=True)):
                fade = min(0.012, keep.duration / 4)
                parts.append(
                    f"{source}atrim=start={keep.start:.4f}:end={keep.end:.4f},"
                    f"asetpts=PTS-STARTPTS,afade=t=in:st=0:d={fade:.3f},"
                    f"afade=t=out:st={keep.duration - fade:.3f}:d={fade:.3f}[eac{i}]"
                )
            joined = "".join(f"[eac{i}]" for i in range(len(keeps)))
            parts.append(f"{joined}concat=n={len(keeps)}:v=0:a=1,{edit_module.AUDIO_FORMAT}[esp]")
        else:
            parts.append(f"{audio_in}{edit_module.AUDIO_FORMAT}[esp]")
        speech = "[esp]"

    if edit.music is not None and music_index is not None:
        parts.append(
            edit_module.music_chain(
                music_index,
                request.output_duration_s,
                edit.music_volume,
                duck=edit.music_duck,
                speech=speech,
                out_label="[emix]",
            )
        )
        current = "[emix]"
    elif speech is not None:
        current = speech
    else:
        return None
    parts.append(f"{current}{tail}[aout]")
    return ";".join(parts)


def _fit_chain(source_label: str, index: int, out_w: int, out_h: int) -> list[str]:
    """Fit the whole frame into the output over a blurred copy of itself.

    Used when the subjects are spread wider than any crop can hold. Plain black
    bars would be honest but look cheap on a phone; a blurred fill is what
    viewers are used to seeing and keeps the frame full-bleed.
    """
    return [
        f"{source_label}split=2[fitbg{index}][fitfg{index}]",
        (
            f"[fitbg{index}]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=24:2[bg{index}]"
        ),
        (
            f"[fitfg{index}]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:"
            f"flags=lanczos[fg{index}]"
        ),
        (f"[bg{index}][fg{index}]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v{index}]"),
    ]


def _zoom_filter(zoom: float, out_w: int, out_h: int) -> str:
    """A slow push-in over the segment.

    Uses ``zoompan``, which is the only filter that can change apparent scale
    while holding a fixed output size. Off by default — at very small
    increments it can step visibly, and a locked frame beats a stuttering one.
    """
    end_zoom = 1.0 + zoom
    return f"zoompan=z='min(zoom+{zoom / 240:.6f},{end_zoom:.4f})':d=1:s={out_w}x{out_h}:fps=30"


def build_audio_filtergraph(settings: ExportSettings) -> str:
    return f"loudnorm=I={settings.loudness_lufs}:TP={LOUDNESS_TRUE_PEAK}:LRA={LOUDNESS_RANGE}"


def encoder_args(settings: ExportSettings) -> list[str]:
    """Pick the video encoder and its quality settings.

    NVENC is materially faster where available. Its quality knob is ``-cq``
    rather than ``-crf``, and the two scales are close enough that reusing the
    configured value keeps output consistent between machines.
    """
    if settings.prefer_hardware_encoder and report().ffmpeg.nvenc_works:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            str(settings.crf),
            "-b:v",
            "0",
            "-profile:v",
            "high",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(settings.crf),
        "-profile:v",
        "high",
    ]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def export_clip(
    request: ExportRequest,
    *,
    work_dir: Path,
    settings: ExportSettings | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Render one clip to an MP4 and return the output path.

    Preconditions:
        request.source exists; request.crop_path segments span the clip duration
        in clip-relative seconds.
    """
    settings = settings or ExportSettings()
    out_w, out_h = ratio_dimensions(request.ratio)

    # Each clip renders in its own workspace so concurrent exports can't collide
    # on the shared `captions.ass` name that the relative-path scheme requires.
    workspace = work_dir / request.destination.stem
    workspace.mkdir(parents=True, exist_ok=True)

    subtitle_name: str | None = None
    fonts_name = "fonts"
    render_cwd: Path | None = None
    edit = request.edit
    out_duration = request.output_duration_s

    # Caption words on the output timeline: absolute transcript times, or —
    # when the edit cuts pauses — re-timed onto the cut clip.
    caption_words = request.words
    caption_offset = request.start_s
    if edit is not None:
        caption_words = edit_module.retime_words(
            [
                Word(w.text, w.start - request.start_s, w.end - request.start_s, w.speaker)
                for w in request.words
            ],
            edit.kept(request.duration_s),
        )
        caption_offset = 0.0
    hook = edit.hook_text if edit is not None else ""

    if (request.burn_captions and caption_words) or hook:
        margin_v_ratio = None
        if request.is_split and request.facecam is not None:
            top_h = split_geometry(request.facecam, request.crop_path, out_w, out_h).top_h
            margin_v_ratio = split_caption_margin(request.style, out_h, top_h)
        elif (
            request.has_video
            and request.ratio in SPLIT_RATIOS
            and request.crop_path.split_share >= 0.5
        ):
            margin_v_ratio = speakers_caption_margin(request.style)
        ass_path = captions_module.write_ass(
            workspace / "captions.ass",
            caption_words if request.burn_captions else [],
            request.style,
            width=out_w,
            height=out_h,
            time_offset_s=caption_offset,
            margin_v_ratio=margin_v_ratio,
            keywords=edit.keywords if edit is not None else (),
            hook_text=hook,
            hook_s=edit.hook_s if edit is not None else 3.0,
            hook_start_s=edit.hook_start_s if edit is not None else 0.0,
            hook_top_px=(
                edit_module.hook_top_px(
                    out_w, out_h, split=request.is_split, watermark=edit.watermark
                )
                if edit is not None
                else None
            ),
            duration_s=out_duration,
        )
        render_cwd, subtitle_name, fonts_name = ffmpeg.relative_filter_workspace(
            ass_path, captions_module.FONT_DIR
        )

    filtergraph = build_video_filtergraph(
        request, subtitle_name=subtitle_name, fonts_name=fonts_name
    )
    inputs = input_layout(request)

    # The edited export mixes its audio inside the filtergraph (cuts, music)
    # and, with two-pass loudness, measures that exact mix first.
    audio_map = ["-map", "0:a?", "-af", build_audio_filtergraph(settings)]
    if edit is None and request.spans:
        # A joined clip's speech only exists inside the filtergraph.
        audio_map = []
        if request.has_audio:
            join_parts, audio_in = joined_audio(request)
            filtergraph = ";".join(
                [filtergraph, *join_parts, f"{audio_in}{build_audio_filtergraph(settings)}[aout]"]
            )
            audio_map = ["-map", "[aout]"]
    if edit is not None:
        audio_map = []
        target = settings.loudness_lufs
        measured = None
        if edit.two_pass_loudnorm:
            measure_graph = build_edit_audio_graph(
                request,
                tail=edit_module.measure_filter(target=target, true_peak=LOUDNESS_TRUE_PEAK),
                music_index=request.input_count if edit.music is not None else None,
            )
            if measure_graph is not None:
                measure_inputs = [
                    *source_inputs(request),
                    *(_music_input(edit.music) if edit.music is not None else []),
                ]
                measured = edit_module.measure_loudnorm(measure_inputs, measure_graph)
        audio_graph = build_edit_audio_graph(
            request,
            tail=edit_module.loudnorm_filter(measured, target=target, true_peak=LOUDNESS_TRUE_PEAK)
            + f",aresample={AUDIO_SAMPLE_RATE}",
            music_index=inputs.music,
        )
        if audio_graph is not None:
            filtergraph = f"{filtergraph};{audio_graph}"
            audio_map = ["-map", "[aout]"]

    extra_inputs: list[str] = []
    output_args: list[str] = []
    if edit is not None:
        if edit.watermark is not None:
            extra_inputs += ["-i", str(edit.watermark.path)]
        if edit.music is not None:
            extra_inputs += _music_input(edit.music)
        output_args = ["-r", str(edit.fps), "-t", f"{out_duration:.4f}"]

    request.destination.parent.mkdir(parents=True, exist_ok=True)

    # Audio-only sources have no [0:v] for the filtergraph to read, and the
    # render failed with "Stream specifier ':v' matches no streams" for every
    # uploaded podcast or voice memo. A generated solid frame stands in as
    # input 1, cut to the clip length so the output ends with the audio.
    background_input: list[str] = []
    if not request.has_video:
        background_input = [
            "-f",
            "lavfi",
            "-t",
            f"{request.duration_s:.4f}",
            "-i",
            f"color=c=black:s={out_w}x{out_h}:r=30",
        ]

    # ffmpeg writes a sibling ``.partial.mp4`` (same directory, so the final
    # rename is atomic; still .mp4 so the muxer is inferred) that replaces the
    # destination only once the render is complete. Rendering in place left a
    # truncated, unplayable MP4 in the exports folder after every failed or
    # cancelled render — and a failed re-export had already destroyed the good
    # file it was replacing, because -y truncates the output at start-up.
    partial = request.destination.with_name(request.destination.stem + ".partial.mp4")

    # Input and output paths are ordinary argv arguments, so they stay absolute
    # and need no escaping regardless of the working directory.
    def render_args(video_args: list[str]) -> list[str]:
        return [
            *source_inputs(request),
            *background_input,
            *extra_inputs,
            "-filter_complex",
            filtergraph,
            "-map",
            "[vout]",
            *audio_map,
            *video_args,
            "-pix_fmt",
            "yuv420p",
            *output_args,
            "-c:a",
            "aac",
            "-b:a",
            AUDIO_BITRATE,
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-movflags",
            "+faststart",
            str(partial),
        ]

    def run(video_args: list[str]) -> None:
        ffmpeg.run(
            render_args(video_args),
            total_duration_s=out_duration,
            on_progress=on_progress,
            cancelled=cancelled,
            cwd=render_cwd,
        )

    try:
        video_args = encoder_args(settings)
        try:
            run(video_args)
        except ffmpeg.FFmpegError as exc:
            uses_gpu = "h264_nvenc" in video_args
            if not uses_gpu or (cancelled is not None and cancelled()):
                raise ExportError(f"Could not render {request.destination.name}.\n{exc}") from exc
            # NVENC passed the start-up probe but can still fail at run time
            # (encoder sessions or GPU memory exhausted by other work). The CPU
            # encoder gives the same quality, only slower.
            log.warning(
                "The GPU encoder failed for %s (%s); rendering on the CPU instead.",
                request.destination.name,
                str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
            )
            cpu_settings = settings.model_copy(update={"prefer_hardware_encoder": False})
            try:
                run(encoder_args(cpu_settings))
            except ffmpeg.FFmpegError as cpu_exc:
                raise ExportError(
                    f"Could not render {request.destination.name}.\n{cpu_exc}"
                ) from cpu_exc

        if not partial.exists() or partial.stat().st_size == 0:
            raise ExportError(f"ffmpeg reported success but {request.destination.name} is empty.")
        try:
            os.replace(partial, request.destination)
        except OSError as exc:
            # Windows refuses to replace a file another program has open, e.g.
            # the previous render playing in a media player.
            raise ExportError(
                f"Could not save {request.destination.name} ({exc}). "
                "Close any program that has it open and try again."
            ) from exc
    except BaseException:
        # Cancellation (ffmpeg.Cancelled, KeyboardInterrupt) included. Cleanup
        # must not mask the real error, so an undeletable leftover is logged.
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove the partial render %s.", partial, exc_info=True)
        raise

    if settings.write_srt and caption_words:
        captions_module.write_srt(
            request.destination.with_suffix(".srt"),
            caption_words,
            time_offset_s=caption_offset,
        )

    log.info(
        "Exported %s (%.1fs, %s, %s%s%s)",
        request.destination.name,
        out_duration,
        request.ratio,
        "split" if request.is_split else "crop",
        f", {len(request.spans)} parts joined" if request.spans else "",
        f"; {', '.join(edit.summary())}" if edit is not None and edit.summary() else "",
    )
    return request.destination


def _music_input(track: Path) -> list[str]:
    """A music input looped forever; the filtergraph cuts it to the clip."""
    return ["-stream_loop", "-1", "-i", str(track)]
