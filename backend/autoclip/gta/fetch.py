"""Everything GTA Mode pulls from the internet.

Only what is needed is downloaded: the stream's AUDIO for analysis (cheap
even for a 12-hour VOD), its public signals (YouTube "Most Replayed", chat
replay), and later just the short video SECTIONS of the clips that survived
judging, in full quality. The full-quality stream itself never touches disk.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..pipeline import ffmpeg, youtube_captions
from ..pipeline.ingest import IngestError, _translate_ytdlp_error, js_runtime_options

log = logging.getLogger(__name__)

#: Analysis audio is split into parts this long before transcription, so a
#: 12-hour VOD never has to sit in memory at once.
PART_S = 30 * 60


class FetchCancelled(RuntimeError):
    """The stream was cancelled while a download was running."""


def detect_platform(url: str) -> str:
    """``youtube``, ``twitch``, ``kick`` or ``other`` from a link."""
    host = (urlparse(url.strip()).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host in {"youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}:
        return "youtube"
    if host.endswith("twitch.tv"):
        return "twitch"
    if host.endswith("kick.com"):
        return "kick"
    return "other"


def is_supported_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and detect_platform(url) != "other"


@dataclass
class ProbeResult:
    platform: str
    title: str = ""
    channel: str = ""
    channel_login: str = ""
    video_id: str = ""
    duration_s: float = 0.0
    is_live: bool = False
    #: YouTube "Most Replayed": (start_s, end_s, value 0..1).
    heatmap: list[tuple[float, float, float]] = field(default_factory=list)
    has_live_chat_replay: bool = False
    webpage_url: str = ""
    #: YouTube's own caption track in the spoken language ({lang, kind, url}),
    #: chosen like the normal pipeline's captions-first path; {} = none.
    caption_track: dict[str, str] = field(default_factory=dict)


def _translate(exc: Exception) -> IngestError:
    """yt-dlp failure -> actionable message (with the user's cookie setting)."""
    from ..config import load as load_settings

    return _translate_ytdlp_error(exc, load_settings().ingest)


def _ydl_base_options() -> dict[str, Any]:
    from ..config import load as load_settings

    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 10,
        "noplaylist": True,
        **js_runtime_options(),
    }
    cookies = load_settings().ingest.cookies_from_browser
    if cookies:
        options["cookiesfrombrowser"] = (cookies,)
    return options


def parse_heatmap(raw: Any) -> list[tuple[float, float, float]]:
    """yt-dlp's ``heatmap`` field into clean, ordered triples."""
    out: list[tuple[float, float, float]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start_time"])
            end = float(item["end_time"])
            value = float(item["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start >= 0:
            out.append((start, end, max(0.0, min(1.0, value))))
    out.sort()
    return out


def probe_from_info(info: dict[str, Any], url: str) -> ProbeResult:
    """Build a :class:`ProbeResult` from a yt-dlp info dict (no network)."""
    platform = detect_platform(info.get("webpage_url") or url)
    live_status = str(info.get("live_status") or "")
    subtitles = info.get("subtitles") or {}
    track = youtube_captions.pick_track(info) if platform == "youtube" else None
    return ProbeResult(
        platform=platform,
        title=str(info.get("title") or info.get("fulltitle") or ""),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        channel_login=str(info.get("uploader_id") or info.get("channel_id") or ""),
        video_id=str(info.get("id") or ""),
        duration_s=float(info.get("duration") or 0.0),
        is_live=bool(info.get("is_live")) or live_status == "is_live",
        heatmap=parse_heatmap(info.get("heatmap")),
        has_live_chat_replay=isinstance(subtitles, dict) and "live_chat" in subtitles,
        webpage_url=str(info.get("webpage_url") or url),
        caption_track=({"lang": track.lang, "kind": track.kind, "url": track.url} if track else {}),
    )


def download_caption_track(track: dict[str, str], directory: Path) -> Path | None:
    """Save a probed caption track (json3) into ``directory``. Never raises.

    Caption links expire within hours, so this runs right after the probe.
    The file is named like the normal pipeline's (``captions.<kind>.<lang>
    .json3``) so :func:`youtube_captions.load_for_source` reads it.
    """
    import json

    url = str(track.get("url") or "")
    if not url:
        return None
    caption = youtube_captions.CaptionTrack(
        lang=str(track.get("lang") or ""), kind=str(track.get("kind") or "auto"), url=url
    )
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL(_ydl_base_options()) as ydl, ydl.urlopen(url) as response:
            raw = response.read()
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        youtube_captions.parse_json3(data)  # validates the shape before saving
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / caption.filename
        path.write_text(json.dumps(data), encoding="utf-8")
        return path
    except Exception as exc:
        log.warning("Could not fetch YouTube's captions (%s); Whisper will transcribe.", exc)
        return None


def probe(url: str) -> ProbeResult:
    """Read a link's metadata without downloading any media."""
    import yt_dlp

    options = {**_ydl_base_options(), "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise _translate(exc) from exc
    if not isinstance(info, dict):
        raise IngestError(f"Could not read {url}.")
    if info.get("_type") == "playlist":
        raise IngestError(
            "That link is a playlist or channel, not one stream.",
            hint="Paste the link of a single stream or VOD.",
        )
    return probe_from_info(info, url)


def _progress_hook(
    on_progress: Callable[[float], None] | None, cancelled: Callable[[], bool] | None
) -> Callable[[dict], None]:
    def hook(status: dict) -> None:
        if cancelled is not None and cancelled():
            raise FetchCancelled("Cancelled.")
        if on_progress is None or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes")
        if total and done:
            on_progress(min(1.0, done / total))
        elif status.get("fragment_count") and status.get("fragment_index") is not None:
            on_progress(min(1.0, status["fragment_index"] / status["fragment_count"]))

    return hook


def _find_media(directory: Path, stem: str) -> Path | None:
    candidates = [
        path
        for path in directory.glob(f"{stem}.*")
        if path.is_file()
        and path.stat().st_size > 0
        and not path.name.endswith((".part", ".ytdl", ".json", ".vtt", ".tmp"))
        and ".part" not in path.suffixes
    ]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def _download(
    url: str,
    directory: Path,
    stem: str,
    *,
    fmt: str,
    section: tuple[float, float] | None,
    on_progress: Callable[[float], None] | None,
    cancelled: Callable[[], bool] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    import yt_dlp
    from yt_dlp.utils import download_range_func

    directory.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        **_ydl_base_options(),
        "format": fmt,
        "outtmpl": str(directory / f"{stem}.%(ext)s"),
        "progress_hooks": [_progress_hook(on_progress, cancelled)],
        **(extra or {}),
    }
    if section is not None:
        start, end = section
        options["download_ranges"] = download_range_func(None, [(max(0.0, start), end)])
        # Keyframe-exact cuts would re-encode the whole section here; the
        # render cuts frame-accurately later, so a padded copy is enough.
        options["force_keyframes_at_cuts"] = False
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
    except FetchCancelled:
        raise
    except yt_dlp.utils.DownloadError as exc:
        if cancelled is not None and cancelled():
            raise FetchCancelled("Cancelled.") from exc
        raise _translate(exc) from exc
    media = _find_media(directory, stem)
    if media is None:
        raise IngestError("The download finished but produced no media file.")
    return media


def download_analysis_audio(
    url: str,
    directory: Path,
    *,
    fmt: str,
    section: tuple[float, float] | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Download the (audio of the) stream used for analysis; reused if present."""
    existing = _find_media(directory, "analysis")
    if existing is not None:
        return existing
    return _download(
        url,
        directory,
        "analysis",
        fmt=fmt,
        section=section,
        on_progress=on_progress,
        cancelled=cancelled,
    )


def download_section(
    url: str,
    directory: Path,
    stem: str,
    start_s: float,
    end_s: float,
    *,
    fmt: str,
    cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Download one clip's video section in full quality; reused if present."""
    existing = _find_media(directory, stem)
    if existing is not None:
        return existing
    return _download(
        url,
        directory,
        stem,
        fmt=fmt,
        section=(start_s, end_s),
        on_progress=on_progress,
        cancelled=cancelled,
        extra={"merge_output_format": "mp4"},
    )


def split_audio(
    source: Path,
    directory: Path,
    *,
    part_s: float = PART_S,
    duration_s: float | None = None,
    cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> list[Path]:
    """Convert the analysis audio to 16 kHz mono WAV parts of ``part_s`` each.

    Reuses existing parts when a previous run finished this step (marked by
    ``parts.done``); a half-finished run is redone from scratch.
    """
    marker = directory / "parts.done"
    if marker.exists():
        parts = sorted(directory.glob("part_*.wav"))
        if parts:
            return parts
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "segment",
            "-segment_time",
            f"{part_s:.0f}",
            "-reset_timestamps",
            "1",
            str(directory / "part_%04d.wav"),
        ],
        total_duration_s=duration_s,
        on_progress=on_progress,
        cancelled=cancelled,
    )
    parts = sorted(directory.glob("part_*.wav"))
    if not parts:
        raise IngestError("The stream's audio could not be decoded.")
    marker.write_text(str(len(parts)), encoding="utf-8")
    return parts


def download_youtube_chat(
    url: str, directory: Path, *, cancelled: Callable[[], bool] | None = None
) -> Path | None:
    """Download a YouTube stream's chat replay (JSON lines). None when absent."""
    import yt_dlp

    existing = list(directory.glob("chat*.live_chat.json"))
    if existing and (directory / "chat.done").exists():
        return existing[0]
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        **_ydl_base_options(),
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": str(directory / "chat.%(ext)s"),
        "progress_hooks": [_progress_hook(None, cancelled)],
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
    except FetchCancelled:
        raise
    except Exception:
        log.warning("Chat replay download failed for %s.", url, exc_info=True)
        return None
    found = list(directory.glob("chat*.live_chat.json"))
    if not found:
        return None
    (directory / "chat.done").write_text("1", encoding="utf-8")
    return found[0]


_TWITCH_VOD = re.compile(r"twitch\.tv/(?:videos/|[^/]+/v(?:ideo)?/)(\d+)")


def twitch_vod_id(url: str) -> str | None:
    match = _TWITCH_VOD.search(url)
    return match.group(1) if match else None
