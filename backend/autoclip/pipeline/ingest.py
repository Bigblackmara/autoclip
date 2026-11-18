"""Ingestion — YouTube URLs, file links and local uploads become source records.

File links (Google Drive, Dropbox, OneDrive, direct video URLs; see
:mod:`autoclip.pipeline.filelink`) are downloaded over plain HTTP and then
take exactly the upload path, :func:`ingest_file`. All paths converge on a
validated :class:`~autoclip.db.models.Source` with a file on disk and probed
metadata.

A note on YouTube: as of 2026 most anonymous downloads hit a bot check, and
proof-of-origin tokens no longer clear it reliably — browser cookies do. So
:class:`IngestError` distinguishes that specific failure and tells the user how
to fix it, rather than surfacing a raw yt-dlp traceback.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import paths
from ..config import IngestSettings
from ..db.models import Source, new_id
from . import ffmpeg

log = logging.getLogger(__name__)

#: Container and audio formats we accept for upload.
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
ACCEPTED_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES

_YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com")

#: yt-dlp error fragments that mean "YouTube wants proof you're a human".
_BOT_CHECK_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you are not a bot",
    "this content isn't available",
    "player response",
)


#: yt-dlp error fragments that mean YouTube refused this network address
#: outright (rate limiting or a blocked VPN / datacenter range).
_BLOCKED_MARKERS = (
    "http error 403",
    "http error 429",
    "too many requests",
    "only images are available",
    "no video formats found",
)

#: JavaScript runtimes yt-dlp can use to solve YouTube's player challenge, in
#: yt-dlp's own order of preference. Without one, YouTube serves storyboard
#: images only and every real download fails.
_JS_RUNTIMES = ("deno", "node", "bun")


def js_runtime_options() -> dict:
    """yt-dlp ``js_runtimes`` for whichever runtimes are installed.

    yt-dlp only enables Deno by default, but most Windows machines have Node
    instead, which works just as well once it is named here.
    """
    found = {name: {} for name in _JS_RUNTIMES if shutil.which(name)}
    return {"js_runtimes": found} if found else {}


class IngestError(RuntimeError):
    """Ingestion failed in a way worth explaining to the user."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n\n{self.hint}" if self.hint else base


def is_youtube_url(url: str) -> bool:
    from urllib.parse import urlparse

    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host in _YOUTUBE_HOSTS


def slugify(text: str, *, max_length: int = 60) -> str:
    """Turn a title into a filesystem- and URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_length] or "clip"


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------


def ingest_youtube(
    url: str,
    settings: IngestSettings | None = None,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Source:
    """Download a YouTube video and return a validated source record.

    When the download fails with an error that usually means yt-dlp is
    outdated (and ``settings.ytdlp_auto_update`` is on), yt-dlp is upgraded
    once and the download retried once, in a fresh interpreter so the new
    version is the one that runs (see :mod:`autoclip.ytdlp_update`).

    Preconditions:
        url points at content the user owns or has the rights to process.
    """
    from .. import ytdlp_update

    settings = settings or IngestSettings()
    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # After an upgrade this process still holds the old yt_dlp in memory;
    # a subprocess imports the new one from disk.
    in_subprocess = ytdlp_update.stale_in_process()
    try:
        try:
            metadata = _download(url, settings, target_dir, on_progress, in_subprocess)
        except YtdlpFailure as exc:
            failed_version = (
                ytdlp_update.installed_version()
                if in_subprocess
                else ytdlp_update.loaded_version() or ytdlp_update.installed_version()
            )
            if not (
                settings.ytdlp_auto_update
                and ytdlp_update.updater.update_for_failure(str(exc), failed_version)
            ):
                raise
            log.warning(
                "Retrying the download once with yt-dlp %s (was %s).",
                ytdlp_update.installed_version(),
                failed_version,
            )
            shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            metadata = _download(url, settings, target_dir, on_progress, True)
    except YtdlpFailure as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        if exc.download_error:
            raise _translate_ytdlp_error(exc, settings) from exc
        raise IngestError(f"Could not download {url}: {exc}") from exc
    except BaseException:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise

    downloaded = _find_downloaded_file(target_dir)
    if downloaded is None:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError("yt-dlp reported success but produced no media file.")

    info = _probe_and_validate(downloaded)

    return Source(
        id=source_id,
        type="youtube",
        url=url,
        path=str(downloaded),
        filename=downloaded.name,
        title=metadata.get("title") or downloaded.stem,
        channel=metadata.get("uploader") or metadata.get("channel"),
        duration_s=info.duration_s or float(metadata.get("duration") or 0.0),
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


class YtdlpFailure(Exception):
    """A yt-dlp download failed (in this process or in the worker).

    ``download_error`` is True for yt-dlp's own ``DownloadError`` (which
    :func:`_translate_ytdlp_error` explains), False for anything else.
    """

    def __init__(self, message: str, *, download_error: bool) -> None:
        super().__init__(message)
        self.download_error = download_error


def youtube_options(
    settings: IngestSettings, target_dir: Path, hook: Callable[[dict], None]
) -> dict:
    options: dict = {
        "format": settings.ytdlp_format,
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "retries": 3,
        "fragment_retries": 3,
        **js_runtime_options(),
    }
    if settings.cookies_from_browser:
        # yt-dlp expects a tuple; only the browser name is required.
        options["cookiesfrombrowser"] = (settings.cookies_from_browser,)
    return options


def _metadata_summary(metadata: Any) -> dict[str, Any]:
    """The few info fields the Source record needs (JSON-safe, for the worker)."""
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata.get(key)
        for key in ("title", "uploader", "channel", "duration")
        if isinstance(metadata.get(key), str | int | float)
    }


def download_in_process(
    url: str,
    settings: IngestSettings,
    target_dir: Path,
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Run yt-dlp here. Raises :class:`YtdlpFailure`. Returns metadata."""
    import yt_dlp

    def hook(status: dict) -> None:
        if not on_progress or status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate")
        done = status.get("downloaded_bytes")
        if total and done:
            on_progress(min(1.0, done / total))

    try:
        with yt_dlp.YoutubeDL(youtube_options(settings, target_dir, hook)) as ydl:
            metadata = ydl.extract_info(url, download=True)
            # Sidecars for the audience signals and the captions-first
            # transcript. Best effort: they never fail an ingest.
            save_sidecars(ydl, metadata, target_dir, captions=settings.prefer_youtube_captions)
    except yt_dlp.utils.DownloadError as exc:
        raise YtdlpFailure(str(exc), download_error=True) from exc
    except Exception as exc:
        raise YtdlpFailure(str(exc), download_error=False) from exc
    return _metadata_summary(metadata)


#: Lines the worker prints; everything else on its output is yt-dlp chatter.
WORKER_PROGRESS = "AUTOCLIP-PROGRESS "
WORKER_RESULT = "AUTOCLIP-RESULT "

#: Seam for tests: how the worker process is started.
_popen = subprocess.Popen


def _worker_env() -> dict[str, str]:
    """The environment for the worker, with this autoclip package importable."""
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{package_root}{os.pathsep}{existing}" if existing else package_root
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def download_in_subprocess(
    url: str,
    settings: IngestSettings,
    target_dir: Path,
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Run yt-dlp in a fresh interpreter (:mod:`autoclip.pipeline.ytdlp_worker`).

    Same code path as :func:`download_in_process`, but the worker imports
    yt_dlp from disk, so an upgrade made after this process started is the
    version that runs. Raises :class:`YtdlpFailure`.
    """
    payload = json.dumps(
        {"url": url, "target_dir": str(target_dir), "settings": settings.model_dump(mode="json")}
    )
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = _popen(
            [sys.executable, "-m", "autoclip.pipeline.ytdlp_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_worker_env(),
            **kwargs,
        )
    except OSError as exc:
        raise YtdlpFailure(
            f"Could not start the download worker: {exc}", download_error=False
        ) from exc

    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(payload)
    process.stdin.close()
    result: dict[str, Any] | None = None
    chatter: list[str] = []
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        if line.startswith(WORKER_PROGRESS):
            if on_progress:
                with contextlib.suppress(ValueError):
                    on_progress(float(line[len(WORKER_PROGRESS) :]))
        elif line.startswith(WORKER_RESULT):
            with contextlib.suppress(ValueError):
                result = json.loads(line[len(WORKER_RESULT) :])
        elif line.strip():
            chatter = [*chatter[-19:], line]
    process.wait()

    if not isinstance(result, dict):
        detail = "\n".join(chatter[-8:]) or f"exit code {process.returncode}"
        raise YtdlpFailure(f"The download worker crashed: {detail}", download_error=False)
    if not result.get("ok"):
        raise YtdlpFailure(
            str(result.get("message") or "unknown error"),
            download_error=bool(result.get("download_error")),
        )
    metadata = result.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _download(
    url: str,
    settings: IngestSettings,
    target_dir: Path,
    on_progress: Callable[[float], None] | None,
    in_subprocess: bool,
) -> dict[str, Any]:
    runner = download_in_subprocess if in_subprocess else download_in_process
    return runner(url, settings, target_dir, on_progress)


#: Info-json keys dropped before saving: large and useless after the download.
_INFO_DROP = ("formats", "requested_formats", "requested_downloads", "thumbnails")


def save_sidecars(ydl: Any, metadata: Any, target_dir: Path, *, captions: bool) -> None:
    """Write ``info.json`` and (when ``captions``) the chosen caption track.

    The info json carries YouTube's "Most replayed" heatmap and chapters for
    :mod:`autoclip.pipeline.signals`. The caption track is fetched now, from
    the link in the fresh info, because those links expire within hours.
    Nothing here may fail the ingest: every error is logged and skipped.
    """
    from . import signals, youtube_captions

    if not isinstance(metadata, dict):
        return
    try:
        sanitized = ydl.sanitize_info(metadata) if hasattr(ydl, "sanitize_info") else metadata
        info = {k: v for k, v in dict(sanitized).items() if k not in _INFO_DROP}
        (target_dir / signals.INFO_FILE).write_text(
            json.dumps(info, ensure_ascii=False, default=str), encoding="utf-8"
        )
    except Exception:
        log.warning("Could not save the video's info json.", exc_info=True)
        info = metadata

    if not captions:
        return
    try:
        track = youtube_captions.pick_track(info)
        if track is None:
            log.info("No caption track in the video's own language; Whisper will transcribe.")
            return
        with ydl.urlopen(track.url) as response:
            raw = response.read()
        data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        youtube_captions.parse_json3(data)  # validates the shape before saving
        (target_dir / track.filename).write_text(json.dumps(data), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not fetch YouTube's captions (%s); Whisper will transcribe.", exc)


def _translate_ytdlp_error(exc: Exception, settings: IngestSettings) -> IngestError:
    """Turn a yt-dlp failure into something the user can act on."""
    message = str(exc).lower()

    if any(marker in message for marker in _BOT_CHECK_MARKERS):
        if settings.cookies_from_browser:
            hint = (
                f"Cookies are already being read from {settings.cookies_from_browser}, but "
                "YouTube still refused. Make sure you are signed in to YouTube in that "
                "browser and that the browser is fully closed — it locks its cookie "
                "database while running."
            )
        else:
            hint = (
                "YouTube is asking for proof you're not a bot. Set a browser to pull "
                "cookies from — in Settings, or via config.json's "
                '`ingest.cookies_from_browser` (e.g. "chrome", "firefox", "edge"). '
                "You must be signed in to YouTube in that browser, and it must be closed "
                "while AutoClip downloads."
            )
        return IngestError("YouTube blocked this download with a bot check.", hint=hint)

    if any(marker in message for marker in _BLOCKED_MARKERS):
        return IngestError(
            "YouTube refused to serve this video to your network.",
            hint=(
                "YouTube rate-limits and blocks many VPN and datacenter addresses "
                "(Cloudflare WARP, Proton VPN and similar). Turn the VPN off, or "
                "exclude youtube.com from it, and try again — or set "
                "`ingest.cookies_from_browser` so the download is signed in. If it "
                "still fails, run `autoclip update-ytdlp`.\n\n"
                f"Original error: {exc}"
            ),
        )

    if "private video" in message or "members-only" in message:
        return IngestError(
            "This video is private or members-only.",
            hint="AutoClip only downloads content you can access and have the rights to use.",
        )
    if "unavailable" in message or "removed" in message:
        return IngestError("This video is unavailable or has been removed.")
    if "drm" in message:
        return IngestError(
            "This content is DRM-protected.",
            hint="AutoClip does not and will not circumvent DRM.",
        )

    return IngestError(
        "yt-dlp could not download this video.",
        hint=(
            "YouTube changes frequently and yt-dlp is updated often. Try "
            "`autoclip update-ytdlp` to pull the latest version.\n\n"
            f"Original error: {exc}"
        ),
    )


def _find_downloaded_file(directory: Path) -> Path | None:
    """Return the largest media file in a directory, ignoring sidecars."""
    candidates = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ACCEPTED_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


def ingest_file(path: Path, *, move: bool = False, title: str | None = None) -> Source:
    """Register a local file as a source, copying it into AutoClip's media store.

    Copying rather than referencing in place means a job stays reproducible even
    if the user moves or deletes the original.

    Preconditions:
        path exists and is a readable media file.
    """
    path = Path(path)
    if not path.exists():
        raise IngestError(f"{path} does not exist.")
    if not path.is_file():
        raise IngestError(f"{path} is not a file.")

    suffix = path.suffix.lower()
    if suffix not in ACCEPTED_SUFFIXES:
        accepted = ", ".join(sorted(ACCEPTED_SUFFIXES))
        raise IngestError(
            f"{suffix or 'This file'} is not a supported format.",
            hint=f"Accepted formats: {accepted}",
        )

    source_id = new_id()
    target_dir = paths.source_media_dir(source_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"source{suffix}"

    try:
        if move:
            shutil.move(str(path), target)
        else:
            shutil.copy2(path, target)
    except OSError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise IngestError(f"Could not store {path.name}: {exc}") from exc

    try:
        info = _probe_and_validate(target)
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise

    return Source(
        id=source_id,
        type="upload",
        path=str(target),
        filename=path.name,
        title=title or info.title or path.stem,
        duration_s=info.duration_s,
        width=info.width,
        height=info.height,
        fps=info.fps,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )


# --------------------------------------------------------------------------
# File links (Google Drive, Dropbox, OneDrive, direct URLs)
# --------------------------------------------------------------------------


def is_file_link(url: str) -> bool:
    """True for a Drive/Dropbox/OneDrive share link or a direct video URL."""
    from . import drivefolder, filelink

    return drivefolder.folder_id(url) is not None or filelink.is_file_link(url)


def ingest_link(
    url: str,
    settings: IngestSettings | None = None,
    *,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Source:
    """Download a file link and register it exactly like an upload.

    ``on_progress(done_bytes, total_bytes_or_None)`` is called while the file
    streams in. The download lands in a staging folder inside the media store
    (same disk, so handing it to :func:`ingest_file` is a rename) that is
    removed whatever happens.
    """
    from . import filelink

    settings = settings or IngestSettings()
    link = filelink.classify(url)
    if link is None:
        raise IngestError(
            "That link is not a supported video file link.",
            hint="Paste a YouTube link, a Google Drive / Dropbox / OneDrive share link, "
            "or a direct link ending in .mp4, .mov, .mkv or .webm.",
        )
    staging = paths.media_dir() / ".incoming" / new_id()
    try:
        downloaded, name = filelink.download(
            link,
            staging,
            max_bytes=settings.link_max_mb * 1024 * 1024,
            on_progress=on_progress,
        )
        source = ingest_file(downloaded, move=True, title=Path(name).stem or link.label)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    source.url = link.url
    source.filename = name
    return source


def _probe_and_validate(path: Path) -> ffmpeg.MediaInfo:
    """Probe a media file and reject anything the pipeline can't process."""
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegError as exc:
        raise IngestError(f"{path.name} could not be read as media.", hint=str(exc)) from exc

    if not info.has_audio:
        raise IngestError(
            f"{path.name} has no audio track.",
            hint=(
                "AutoClip finds clips by transcribing speech, so a file with no audio "
                "has nothing to work from."
            ),
        )
    if info.duration_s <= 0:
        raise IngestError(
            f"{path.name} reports zero duration.",
            hint="The file may be corrupt or still being written.",
        )

    return info
