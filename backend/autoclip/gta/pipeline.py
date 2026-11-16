"""The GTA Mode stages for one stream (and its clips).

Every stage is idempotent and resumable: its result is a file in the
stream's work directory, so a crash, a restart or a retry continues where it
stopped instead of re-downloading or re-transcribing hours of stream. All
times are STREAM time (seconds from the start of the VOD), including when
only a range of the stream is analysed.

Status meaning: a stream's ``status`` names the stage that runs NEXT (or is
running). Each stage function moves it on when it finishes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..config import load as load_settings
from ..db.models import new_id
from ..pipeline import prepare, transcribe
from ..pipeline.transcribe import WhisperModelCache
from ..pipeline.transcript import Transcript, Word
from . import align, fetch, hype, judging, qa, render, signals, store
from .gate import gate
from .settings import GtaSettings, output_root, work_root
from .store import GtaClip, GtaStream

log = logging.getLogger(__name__)

#: Next status after each stage.
NEXT = {
    "queued": "downloading_audio",
    "probing": "downloading_audio",
    "downloading_audio": "fetching_signals",
    "fetching_signals": "transcribing",
    "transcribing": "building_hype",
    "building_hype": "judging",
    "judging": "analyzed",
}

#: Which worker pool runs each stage.
HEAVY_STAGES = frozenset({"transcribing"})

_model_cache = WhisperModelCache(max_models=1)


class StreamCancelled(RuntimeError):
    pass


class StageError(RuntimeError):
    """A stage failed with a message the user should see."""


# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------


class Workspace:
    def __init__(self, stream_id: str) -> None:
        self.root = work_root() / stream_id
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def probe(self) -> Path:
        return self.root / "probe.json"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def parts_dir(self) -> Path:
        return self.root / "parts"

    @property
    def chat_dir(self) -> Path:
        return self.root / "chat"

    @property
    def messages(self) -> Path:
        return self.root / "chat" / "messages.jsonl"

    @property
    def viewer_clips(self) -> Path:
        return self.root / "viewer_clips.json"

    @property
    def transcripts_dir(self) -> Path:
        return self.root / "transcript"

    @property
    def transcript(self) -> Path:
        return self.root / "transcript.json"

    @property
    def silences(self) -> Path:
        return self.root / "silences.json"

    @property
    def hype(self) -> Path:
        return self.root / "hype.json"

    @property
    def chunks_dir(self) -> Path:
        return self.root / "chunks"

    @property
    def sections_dir(self) -> Path:
        return self.root / "sections"

    def render_dir(self, clip_id: str) -> Path:
        return self.root / "render" / clip_id


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# --------------------------------------------------------------------------
# Progress + cancellation
# --------------------------------------------------------------------------


class Reporter:
    """Throttled progress writes and a cheap cancel check for one stream."""

    def __init__(self, stream_id: str, is_cancelled: Callable[[], bool]) -> None:
        self.stream_id = stream_id
        self._is_cancelled = is_cancelled
        self._last = 0.0
        self._lock = threading.Lock()

    def cancelled(self) -> bool:
        return self._is_cancelled()

    def check(self) -> None:
        if self._is_cancelled():
            raise StreamCancelled("Cancelled.")

    def __call__(self, fraction: float, detail: str = "", *, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last < 1.0:
                return
            self._last = now
        fields = {"progress": round(max(0.0, min(1.0, fraction)), 4)}
        if detail:
            fields["detail"] = detail
        store.update_stream(self.stream_id, **fields)

    def scaled(self, detail: str) -> Callable[[float], None]:
        def report(fraction: float) -> None:
            if self._is_cancelled():
                raise StreamCancelled("Cancelled.")
            self(fraction, detail)

        return report


# --------------------------------------------------------------------------
# Stream stages
# --------------------------------------------------------------------------


def stage_probe(stream: GtaStream, report: Reporter) -> None:
    report(0.0, "Reading the stream link", force=True)
    ws = Workspace(stream.id)
    result = fetch.probe(stream.url)
    if result.is_live:
        raise StageError(
            'This stream is LIVE right now. Use GTA Live ("Clip this live stream") for it, '
            "or queue the VOD once the stream has ended."
        )
    if result.duration_s <= 0:
        raise StageError("Could not read the stream's length; is this a VOD link?")
    _save_json(
        ws.probe,
        {
            **asdict(result),
            "heatmap": [list(item) for item in result.heatmap],
        },
    )
    start, end = _range(stream, result.duration_s)
    store.update_stream(
        stream.id,
        platform=result.platform,
        title=result.title,
        channel=result.channel,
        duration_s=result.duration_s,
        # The user's range, clamped to the real length (None = whole stream).
        range_start_s=start if start > 0 else None,
        range_end_s=end if end < result.duration_s else None,
        signals={
            **stream.signals,
            "replayed": bool(result.heatmap),
            "chat_replay": result.has_live_chat_replay,
        },
    )


def _range(stream: GtaStream, duration_s: float) -> tuple[float, float]:
    start = max(0.0, stream.range_start_s or 0.0)
    end = stream.range_end_s if stream.range_end_s else duration_s
    end = min(end, duration_s) if duration_s else end
    if end <= start:
        raise StageError("The chosen range is empty.")
    return start, end


def _probe(ws: Workspace) -> dict:
    data = _load_json(ws.probe)
    if not isinstance(data, dict):
        raise StageError("The stream has not been read yet (probe missing).")
    return data


def stage_audio(stream: GtaStream, report: Reporter, settings: GtaSettings) -> None:
    ws = Workspace(stream.id)
    start, end = _range(stream, stream.duration_s)
    whole = start <= 0 and end >= stream.duration_s - 1
    report(0.0, "Downloading the stream's audio for analysis", force=True)
    audio = fetch.download_analysis_audio(
        stream.url,
        ws.audio_dir,
        fmt=settings.analysis_format,
        section=None if whole else (start, end),
        on_progress=lambda f: report.scaled("Downloading audio")(0.8 * f),
        cancelled=report.cancelled,
    )
    report.check()
    report(0.8, "Preparing audio", force=True)
    fetch.split_audio(
        audio,
        ws.parts_dir,
        duration_s=end - start,
        cancelled=report.cancelled,
        on_progress=lambda f: report.scaled("Preparing audio")(0.8 + 0.2 * f),
    )


def stage_signals(stream: GtaStream, report: Reporter) -> None:
    """Chat replay + viewer clips. Missing signals are recorded, never fatal."""
    ws = Workspace(stream.id)
    probe = _probe(ws)
    report(0.0, "Collecting chat and viewer signals", force=True)
    found = dict(stream.signals)
    notes: list[str] = []

    if not ws.messages.exists():
        messages: list[signals.ChatMessage] = []
        if stream.platform == "youtube" and probe.get("has_live_chat_replay"):
            path = fetch.download_youtube_chat(stream.url, ws.chat_dir, cancelled=report.cancelled)
            if path is not None:
                messages = signals.read_youtube_chat(path)
            else:
                notes.append("YouTube chat replay could not be downloaded")
        elif stream.platform == "youtube":
            notes.append("this YouTube stream has no chat replay")
        elif stream.platform == "twitch":
            notes.append("Twitch has no official API for old VOD chat")
        elif stream.platform == "kick":
            notes.append("Kick VOD chat is not available")
        signals.save_messages(ws.messages, messages)
    messages = signals.load_messages(ws.messages)
    found["chat_messages"] = len(messages)
    report.check()

    if not ws.viewer_clips.exists():
        clips: list[dict] = []
        vod_id = fetch.twitch_vod_id(stream.url) if stream.platform == "twitch" else None
        credentials = signals.twitch_credentials() if vod_id else None
        if vod_id and credentials and probe.get("channel_login"):
            try:
                clips = [
                    asdict(c)
                    for c in signals.twitch_viewer_clips(
                        vod_id, str(probe["channel_login"]), credentials=credentials
                    )
                ]
            except Exception as exc:
                log.warning("Twitch viewer clips failed: %s", exc)
                notes.append("Twitch viewer clips could not be read")
        elif vod_id:
            notes.append("add Twitch API credentials in GTA settings to use viewer clips")
        _save_json(ws.viewer_clips, clips)
    found["viewer_clips"] = len(_load_json(ws.viewer_clips, []) or [])
    if notes:
        found["notes"] = notes
    store.update_stream(stream.id, signals=found)


def stage_transcribe(stream: GtaStream, report: Reporter) -> None:
    ws = Workspace(stream.id)
    if ws.transcript.exists():
        return
    start, _ = _range(stream, stream.duration_s)
    parts = sorted(ws.parts_dir.glob("part_*.wav"))
    if not parts:
        raise StageError("The analysis audio is missing; retry the stream.")
    whisper = load_settings().whisper
    ws.transcripts_dir.mkdir(parents=True, exist_ok=True)
    words: list[Word] = []
    language = ""
    model = ""
    for index, part in enumerate(parts):
        report.check()
        cache = ws.transcripts_dir / f"{part.stem}.json"
        if cache.exists():
            piece = Transcript.load(cache)
        else:
            label = f"Transcribing part {index + 1} of {len(parts)}"
            report(index / len(parts), label, force=True)
            with gate.hold():
                piece = transcribe.transcribe(
                    part,
                    whisper,
                    on_progress=lambda f, i=index, lbl=label: report.scaled(lbl)(
                        (i + f) / len(parts)
                    ),
                    cancelled=report.cancelled,
                    model_cache=_model_cache,
                )
            piece.save(cache)
        offset = start + index * fetch.PART_S
        language = language or piece.language
        model = model or piece.model
        words.extend(
            Word(text=w.text, start=w.start + offset, end=w.end + offset, speaker=w.speaker)
            for w in piece.words
        )
    words.sort(key=lambda w: w.start)
    Transcript(words=words, language=language, model=model).save(ws.transcript)


def stage_hype(stream: GtaStream, report: Reporter) -> None:
    ws = Workspace(stream.id)
    report(0.0, "Building the Hype Curve", force=True)
    probe = _probe(ws)
    start, end = _range(stream, stream.duration_s)
    seconds = int(np.ceil(end))
    parts = sorted(ws.parts_dir.glob("part_*.wav"))

    loudness = np.full(seconds, signals.SILENCE_DB, dtype=np.float32)
    if parts:
        relative = signals.loudness_for_parts(parts, fetch.PART_S)
        a = int(start)
        n = min(len(relative), seconds - a)
        if n > 0:
            loudness[a : a + n] = relative[:n]

    messages = signals.load_messages(ws.messages)
    chat = signals.chat_series(messages, seconds) if messages else None
    heat = signals.heatmap_series([tuple(item) for item in probe.get("heatmap") or []], seconds)
    viewer = [signals.ViewerClip(**c) for c in _load_json(ws.viewer_clips, []) or []]
    clips_curve = signals.clips_series(viewer, seconds) if viewer else None

    curve = hype.build_curve(
        seconds, loudness_db=loudness, chat=chat, heat=heat, viewer_clips=clips_curve
    )
    # Outside the analysed range there is no audio; markers there are noise.
    curve.markers = [m for m in curve.markers if start <= m.t < end]
    curve.save(ws.hype)

    transcript = Transcript.load(ws.transcript)
    if not ws.silences.exists():
        silences = _silences_from_words(transcript)
        _save_json(ws.silences, [{"start": s.start, "end": s.end} for s in silences])
    found = dict(stream.signals)
    found["hype_signals"] = curve.available
    found["hype_markers"] = len(curve.markers)
    store.update_stream(stream.id, signals=found)


def _silences_from_words(transcript: Transcript, min_gap_s: float = 0.2) -> list[prepare.Silence]:
    """Word gaps stand in for silencedetect over a multi-hour stream."""
    out: list[prepare.Silence] = []
    words = transcript.words
    for a, b in zip(words, words[1:], strict=False):
        if b.start - a.end >= min_gap_s:
            out.append(prepare.Silence(start=a.end, end=b.start))
    return out


def _provider_complete():
    from ..providers import build_provider

    settings = load_settings()
    provider = build_provider(settings.active_provider, settings)
    return provider._complete  # noqa: SLF001 - the shared completion entry point


def stage_judge(
    stream: GtaStream,
    report: Reporter,
    settings: GtaSettings,
    *,
    complete=None,
) -> None:
    ws = Workspace(stream.id)
    transcript = Transcript.load(ws.transcript)
    if not transcript.words:
        raise StageError("Nothing was said in this stream (empty transcript).")
    curve = hype.HypeCurve.load(ws.hype) if ws.hype.exists() else None
    silences = [prepare.Silence(**s) for s in _load_json(ws.silences, []) or []]
    start, end = _range(stream, stream.duration_s)
    chunks = judging.build_chunks(
        transcript,
        chunk_s=settings.chunk_minutes * 60,
        tail_s=settings.chunk_tail_s,
        start_s=start,
        end_s=end,
    )
    complete = complete or _provider_complete()
    ws.chunks_dir.mkdir(parents=True, exist_ok=True)

    done: dict[int, list[judging.Candidate]] = {}
    for path in ws.chunks_dir.glob("chunk_*.json"):
        data = _load_json(path)
        if isinstance(data, dict) and "index" in data:
            done[int(data["index"])] = [judging.Candidate(**c) for c in data.get("candidates", [])]

    def save_chunk(index: int, found: list[judging.Candidate]) -> None:
        _save_json(
            ws.chunks_dir / f"chunk_{index:04d}.json",
            {"index": index, "candidates": [asdict(c) for c in found]},
        )

    def on_chunk(done_count: int, total: int) -> None:
        report.check()
        report(0.85 * done_count / max(1, total), f"Reading chunk {done_count} of {total}")

    streamer = stream.channel or "the streamer"
    report(0.0, f"Reading {len(chunks)} chunks of 12 minutes", force=True)

    async def both_rounds() -> list[judging.Candidate]:
        # One event loop for both rounds: the provider's HTTP client is bound
        # to the loop it was first used on.
        found = await judging.find_candidates(
            transcript,
            chunks,
            complete=complete,
            settings=settings,
            curve=curve,
            silences=silences,
            streamer=streamer,
            title=stream.title,
            on_chunk=on_chunk,
            cancelled=report.cancelled,
            done_chunks=done,
            on_chunk_result=save_chunk,
        )
        report.check()
        survivors = sum(1 for c in found if c.status == "candidate")
        report(0.88, f"Final judging round: {survivors} moments go head-to-head", force=True)
        return await judging.final_round(
            transcript,
            found,
            complete=complete,
            settings=settings,
            streamer=streamer,
            title=stream.title,
            curve=curve,
            cancelled=report.cancelled,
        )

    candidates = asyncio.run(both_rounds())
    _store_candidates(stream, candidates)


def _store_candidates(stream: GtaStream, candidates: list[judging.Candidate]) -> None:
    """Replace the stream's clip rows with this judging result."""
    store.delete_clips(stream.id)
    approved = sorted(
        (c for c in candidates if c.status == "approved"), key=lambda c: -c.rank_score
    )
    ranks = {c.id: i + 1 for i, c in enumerate(approved)}
    for c in candidates:
        store.create_clip(
            GtaClip(
                id=new_id(),
                stream_id=stream.id,
                start_s=c.start_s,
                end_s=c.end_s,
                status="approved" if c.status == "approved" else "rejected",
                chunk_index=c.chunk_index,
                hook_title=c.hook_title,
                hook_quote=c.hook_quote,
                moment_type=c.moment_type,
                summary=c.why,
                hype_score=round(c.hype_score, 1),
                score=c.score,
                rank=ranks.get(c.id),
                verdict=c.verdict,
                reject_reason=c.reject_reason,
                caption=c.caption,
                hashtags=c.hashtags,
            )
        )


# --------------------------------------------------------------------------
# Clip stages
# --------------------------------------------------------------------------


def section_bounds(clip: GtaClip, settings: GtaSettings) -> tuple[float, float]:
    start = max(0.0, clip.start_s - settings.section_pad_s)
    return start, clip.end_s + settings.section_pad_s


def stage_section(
    clip: GtaClip, stream: GtaStream, report: Reporter, settings: GtaSettings
) -> None:
    ws = Workspace(stream.id)
    start, end = section_bounds(clip, settings)
    fetch.download_section(
        stream.url,
        ws.sections_dir,
        clip.id,
        start,
        end,
        fmt=settings.clip_format,
        cancelled=report.cancelled,
    )


def aligned_section_start(
    clip: GtaClip, stream: GtaStream, section: Path, settings: GtaSettings
) -> float:
    """Stream time at which the downloaded section really starts (cached)."""
    ws = Workspace(stream.id)
    cache = ws.sections_dir / f"{clip.id}.offset.json"
    cached = _load_json(cache)
    if isinstance(cached, dict) and cached.get("section") == section.name:
        return float(cached["start_s"])
    expected, _ = section_bounds(clip, settings)
    range_start, _ = _range(stream, stream.duration_s)
    start, confidence = align.section_offset(
        section,
        expected_start_s=expected,
        parts=sorted(ws.parts_dir.glob("part_*.wav")),
        part_s=fetch.PART_S,
        range_start_s=range_start,
    )
    _save_json(cache, {"section": section.name, "start_s": start, "confidence": confidence})
    return start


def _slug(text: str, limit: int = 48) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:limit].strip("-") or "clip"


def clip_destination(clip: GtaClip, stream: GtaStream) -> Path:
    folder = output_root() / _slug(stream.channel or "stream", 40)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    name = f"{stamp}-{_slug(clip.hook_title)}-{clip.id[:6]}.mp4"
    return folder / name


def post_text(clip: GtaClip) -> str:
    """Caption + hashtags exactly as they should be pasted."""
    tags = " ".join(clip.hashtags)
    return f"{clip.caption}\n\n{tags}".strip()


def render_and_qa(
    *,
    section: Path,
    section_start_s: float,
    start_s: float,
    end_s: float,
    words: list[Word],
    hook_title: str,
    destination: Path,
    work_dir: Path,
    settings: GtaSettings,
    cancelled: Callable[[], bool],
    live: bool = False,
) -> dict:
    """Render one clip and run QA, fixing what can be fixed automatically.

    Loudness off target -> gain-corrected remux; black frames at the very
    start/end -> re-render trimmed (up to twice). Runs under the heavy gate.
    Returns the QA dict (``passed`` plus every check).
    """
    export_settings = load_settings().export
    trim_start = trim_end = 0.0
    report_dict: dict = {}
    with gate.hold(live=live):
        for attempt in range(3):
            if cancelled():
                raise StreamCancelled("Cancelled.")
            result = render.render(
                render.RenderJob(
                    section=section,
                    section_offset_s=section_start_s,
                    start_s=start_s,
                    end_s=end_s,
                    words=words,
                    hook_title=hook_title,
                    destination=destination,
                    work_dir=work_dir,
                ),
                style_key=settings.caption_style,
                title_s=settings.hook_title_s,
                dead_air_min_s=settings.dead_air_min_s,
                dead_air_keep_s=settings.dead_air_keep_s,
                max_duration_s=settings.max_clip_s,
                loudness_lufs=settings.loudness_lufs,
                true_peak_db=settings.true_peak_db,
                export_settings=export_settings,
                trim_start_s=trim_start,
                trim_end_s=trim_end,
                cancelled=cancelled,
            )
            local_start = start_s - section_start_s + trim_start
            local_end = end_s - section_start_s - trim_end

            def check(face_fn=None, path=result.path, a=local_start, b=local_end):
                return qa.run_qa(
                    path,
                    section=section,
                    section_start_s=a,
                    section_end_s=b,
                    target_lufs=settings.loudness_lufs,
                    tolerance_lu=settings.loudness_tolerance_lu,
                    true_peak_db=settings.true_peak_db,
                    face_min_ratio=settings.face_min_ratio,
                    face_ratio_fn=face_fn,
                )

            qa_report = check()
            if not qa_report.loudness.ok and qa_report.loudness_lufs is not None:
                qa.fix_loudness(
                    result.path,
                    qa_report.loudness_lufs,
                    settings.loudness_lufs,
                    settings.true_peak_db,
                )
                face_value = qa_report.facecam.value
                if face_value is not None:
                    qa_report = check(lambda *_, v=float(face_value): v)
                else:
                    qa_report = check()
            report_dict = {
                **qa_report.to_dict(),
                "cut_dead_air_s": round(result.cut_s, 2),
                "duration_s": round(result.duration_s, 2),
                "attempts": attempt + 1,
            }
            if qa_report.black_frames.ok or not qa_report.black_spans:
                break
            trims = qa.edge_trims(qa_report.black_spans, result.duration_s)
            if trims is None:
                break
            trim_start += trims[0]
            trim_end += trims[1]
            if (end_s - start_s) - trim_start - trim_end < settings.min_clip_s:
                break
    return report_dict


def stage_render(clip: GtaClip, stream: GtaStream, report: Reporter, settings: GtaSettings) -> dict:
    """Render + QA one approved clip of a VOD. Returns {"path", "qa"}."""
    ws = Workspace(stream.id)
    section = fetch._find_media(ws.sections_dir, clip.id)  # noqa: SLF001
    if section is None:
        raise StageError("The clip's video section is missing; it will be downloaded again.")
    section_start = aligned_section_start(clip, stream, section, settings)
    transcript = Transcript.load(ws.transcript)
    first, last = transcript.indices_in_range(clip.start_s, clip.end_s)
    words = transcript.slice(first, last) if last >= first else []
    destination = Path(clip.path) if clip.path else clip_destination(clip, stream)
    qa_dict = render_and_qa(
        section=section,
        section_start_s=section_start,
        start_s=clip.start_s,
        end_s=clip.end_s,
        words=words,
        hook_title=clip.hook_title,
        destination=destination,
        work_dir=ws.render_dir(clip.id),
        settings=settings,
        cancelled=report.cancelled,
    )
    # The caption file next to the clip: one copy-paste away from posting.
    destination.with_suffix(".txt").write_text(post_text(clip) + "\n", encoding="utf-8")
    return {"path": str(destination), "qa": qa_dict}


def cleanup_stream_files(stream_id: str) -> None:
    root = work_root() / stream_id
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
