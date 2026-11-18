"""Health checks: everything AutoClip needs, each ok / warn / fail with a fix.

Used by ``autoclip doctor`` (CLI), ``GET /api/system/doctor`` and the Health
panel. Every probe is injectable (:class:`Probes`) so tests run with fakes.

Nothing here costs money or posts anything: the LLM check only reads the
configuration (key stored? model set?) unless ``live=True``, which asks the
provider's own health check (a model listing, not a completion).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

Status = Literal["ok", "warn", "fail"]

GB = 1024**3
DISK_FAIL_BYTES = 2 * GB
DISK_WARN_BYTES = 10 * GB
#: A yt-dlp older than this many days usually can't read YouTube any more.
YTDLP_STALE_DAYS = 60
JS_RUNTIMES = ("deno", "node", "bun")


@dataclass
class Check:
    id: str
    group: str
    label: str
    status: Status
    detail: str
    fix: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _db_version(path: Path) -> int | None:
    """``PRAGMA user_version`` read-only; None when there is no database yet."""
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return -1
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error:
        return -1
    finally:
        conn.close()


def _youtube_state() -> dict[str, Any]:
    from .live.uploads import tokens as tk
    from .live.uploads.youtube import can_edit

    out: dict[str, Any] = {"rights": False, "client": False, "linked": False, "can_edit": False}
    try:
        out["rights"] = tk.has_rights_attestation()
        out["client"] = tk.load_client_credentials(tk.PLATFORM_YOUTUBE) is not None
        bundle = tk.load_tokens(tk.PLATFORM_YOUTUBE)
        out["linked"] = bundle is not None
        out["can_edit"] = can_edit(bundle)
        out["account"] = bundle.account if bundle else None
    except Exception as exc:  # keyring trouble, corrupt blob
        out["error"] = str(exc)[:200]
    return out


def _zernio_key() -> str:
    from .gta import zernio

    try:
        return zernio.load_api_key() or ""
    except Exception:
        return ""


def _zernio_live(key: str) -> tuple[bool, str]:
    from .gta import zernio

    ok, message, _username = zernio.test_connection(key)
    return ok, message


def _provider_live(name: str) -> tuple[bool, str]:
    from .providers import build_provider

    async def check() -> tuple[bool, str]:
        provider = build_provider(name)
        status = await provider.health_check()
        return status.available, status.detail

    return asyncio.run(check())


def _system_report():
    from . import system

    return system.report()


def _ytdlp_installed() -> str | None:
    from .ytdlp_update import installed_version

    return installed_version()


def _ytdlp_loaded() -> str | None:
    from .ytdlp_update import loaded_version

    return loaded_version()


def _settings():
    from .config import load

    return load()


def _get_secret(name: str, settings: Any) -> str | None:
    from .config import get_secret

    return get_secret(name, settings)


def _home() -> Path:
    from .paths import root

    return root()


def _db_path() -> Path:
    from .paths import db_path

    return db_path()


def _schema_version() -> int:
    from .db.schema import SCHEMA_VERSION

    return SCHEMA_VERSION


def _today() -> date:
    return datetime.now(UTC).date()


@dataclass
class Probes:
    """Everything the checks read; tests replace any of these."""

    report: Callable[[], Any] = _system_report
    which: Callable[[str], str | None] = shutil.which
    module_available: Callable[[str], bool] = _module_available
    ytdlp_version: Callable[[], str | None] = _ytdlp_installed
    ytdlp_loaded: Callable[[], str | None] = _ytdlp_loaded
    settings: Callable[[], Any] = _settings
    get_secret: Callable[[str, Any], str | None] = _get_secret
    youtube_state: Callable[[], dict[str, Any]] = _youtube_state
    zernio_key: Callable[[], str] = _zernio_key
    zernio_live: Callable[[str], tuple[bool, str]] = _zernio_live
    provider_live: Callable[[str], tuple[bool, str]] = _provider_live
    disk_usage: Callable[[str], Any] = shutil.disk_usage
    home: Callable[[], Path] = _home
    db_path: Callable[[], Path] = _db_path
    db_version: Callable[[Path], int | None] = _db_version
    schema_version: Callable[[], int] = _schema_version
    today: Callable[[], date] = _today


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _ffmpeg_hint() -> str:
    import sys

    if sys.platform == "win32":
        return "Install the full build: winget install Gyan.FFmpeg (not the Essentials build)."
    if sys.platform == "darwin":
        return "Install the full build: brew install ffmpeg-full."
    return "Install ffmpeg from your package manager (e.g. sudo apt install ffmpeg)."


def _video_checks(report: Any) -> list[Check]:
    ff = report.ffmpeg
    hint = _ffmpeg_hint()
    out = [
        Check(
            "ffmpeg",
            "Video tools",
            "ffmpeg",
            "ok" if ff.found else "fail",
            f"{ff.version or 'found'} at {ff.path}" if ff.found else "not found on PATH",
            "" if ff.found else hint,
        ),
        Check(
            "ffprobe",
            "Video tools",
            "ffprobe",
            "ok" if ff.ffprobe_found else "fail",
            "found" if ff.ffprobe_found else "not found on PATH",
            "" if ff.ffprobe_found else hint,
        ),
    ]
    if not ff.found:
        return out
    out.append(
        Check(
            "libass",
            "Video tools",
            "Captions (libass)",
            "ok" if ff.has_libass else "fail",
            "subtitle burn-in available" if ff.has_libass else "this ffmpeg can't burn captions",
            "" if ff.has_libass else hint,
        )
    )
    has_drawtext = "drawtext" in ff.filters
    out.append(
        Check(
            "drawtext",
            "Video tools",
            "Text overlays (drawtext)",
            "ok" if has_drawtext else "warn",
            "available" if has_drawtext else "missing: hook text and watermarks can't be drawn",
            "" if has_drawtext else f"Use a full ffmpeg build with freetype. {hint}",
        )
    )
    missing = list(ff.missing_filters)
    if missing:
        out.append(
            Check(
                "filters",
                "Video tools",
                "Required filters",
                "fail",
                f"missing: {', '.join(missing)}",
                hint,
            )
        )
    if ff.nvenc_works:
        out.append(Check("nvenc", "Video tools", "GPU encoding (NVENC)", "ok", "works"))
    elif ff.has_nvenc:
        out.append(
            Check(
                "nvenc",
                "Video tools",
                "GPU encoding (NVENC)",
                "warn",
                ff.nvenc_error or "listed but not usable",
                "Update the NVIDIA driver to encode on the GPU (CPU encoding works, slower).",
            )
        )
    else:
        out.append(
            Check(
                "nvenc",
                "Video tools",
                "GPU encoding (NVENC)",
                "ok",
                "not available - exports use the CPU encoder (same quality, slower)",
            )
        )
    return out


def _ytdlp_checks(p: Probes) -> list[Check]:
    out: list[Check] = []
    version = p.ytdlp_version()
    if not version:
        out.append(
            Check(
                "ytdlp",
                "Downloads",
                "yt-dlp",
                "fail",
                "not installed",
                "Reinstall AutoClip's dependencies (uv pip install -e .).",
            )
        )
    else:
        status: Status = "ok"
        detail = version
        fix = ""
        with contextlib.suppress(ValueError):
            y, m, d = (int(x) for x in version.split(".")[:3])
            age = (p.today() - date(y, m, d)).days
            if age > YTDLP_STALE_DAYS:
                status, fix = "warn", "Run autoclip update-ytdlp (YouTube changes often)."
                detail = f"{version} ({age} days old)"
        loaded = p.ytdlp_loaded()
        if loaded and loaded != version:
            status = "warn"
            detail += f"; this process still runs {loaded}"
            fix = fix or "Restart AutoClip to use the updated yt-dlp."
        out.append(Check("ytdlp", "Downloads", "yt-dlp", status, detail, fix))
    runtimes = [name for name in JS_RUNTIMES if p.which(name)]
    out.append(
        Check(
            "js_runtime",
            "Downloads",
            "JavaScript runtime",
            "ok" if runtimes else "warn",
            f"found: {', '.join(runtimes)}" if runtimes else "none found (node, deno or bun)",
            ""
            if runtimes
            else "Install Node.js LTS (winget install OpenJS.NodeJS.LTS): YouTube downloads "
            "need it to read the video formats.",
        )
    )
    ejs = p.module_available("yt_dlp_ejs")
    out.append(
        Check(
            "ytdlp_ejs",
            "Downloads",
            "yt-dlp-ejs (YouTube challenge solver)",
            "ok" if ejs else "warn",
            "installed" if ejs else "not installed",
            "" if ejs else "Install it into AutoClip's environment: uv pip install yt-dlp-ejs",
        )
    )
    return out


def _provider_checks(p: Probes, settings: Any, live: bool) -> list[Check]:
    from .config import KEYED_PROVIDERS

    name = settings.active_provider
    model = settings.provider(name).model
    label = f"AI provider ({name})"
    if name in KEYED_PROVIDERS:
        has_key = p.get_secret(name, settings) is not None
        if not has_key:
            return [
                Check(
                    "llm",
                    "AI",
                    label,
                    "fail",
                    "no API key stored",
                    f"Add the key in Settings, or run: autoclip config set-secret {name}",
                )
            ]
        if not model:
            return [Check("llm", "AI", label, "fail", "no model set", "Pick a model in Settings.")]
    else:
        base = settings.provider(name).base_url or "http://localhost:11434"
        if not model:
            return [
                Check(
                    "llm",
                    "AI",
                    label,
                    "fail",
                    "no model set",
                    "Pull a model (ollama pull llama3.1) and pick it in Settings.",
                )
            ]
        model = f"{model} at {base}"
    if not live:
        return [Check("llm", "AI", label, "ok", f"configured: {model} (not contacted)")]
    try:
        ok, detail = p.provider_live(name)
    except Exception as exc:
        ok, detail = False, str(exc)[:200]
    return [
        Check(
            "llm",
            "AI",
            label,
            "ok" if ok else "fail",
            f"{model}: {detail or ('reachable' if ok else 'unreachable')}",
            "" if ok else "Check the API key / model in Settings and your internet connection.",
        )
    ]


def _secret_checks(report: Any, settings: Any) -> list[Check]:
    ok = bool(report.deps.keyring_backend)
    out = [
        Check(
            "keyring",
            "Secrets",
            "OS keyring",
            "ok" if ok else "warn",
            "available" if ok else "no keyring backend: secrets can't be stored safely",
            ""
            if ok
            else "On headless Linux install a keyring backend (keyrings.alt or Secret Service).",
        )
    ]
    if settings.insecure_secret_storage:
        out.append(
            Check(
                "plaintext_secrets",
                "Secrets",
                "Stored secrets",
                "warn",
                "one or more secrets are in config.json in plaintext",
                "Fix the keyring, then save the keys again in Settings.",
            )
        )
    return out


def _connection_checks(p: Probes, live: bool) -> list[Check]:
    yt = p.youtube_state()
    if yt.get("error"):
        youtube = Check(
            "youtube",
            "Posting",
            "YouTube",
            "warn",
            f"connection can't be read: {yt['error']}",
            "Reconnect YouTube in GTA Mode's Posting tab.",
        )
    elif not yt.get("rights"):
        youtube = Check(
            "youtube",
            "Posting",
            "YouTube",
            "warn",
            "posting locked: the rights notice is not accepted",
            "Accept the rights notice in GTA Mode's Posting tab (only if you will post).",
        )
    elif not yt.get("client"):
        youtube = Check(
            "youtube",
            "Posting",
            "YouTube",
            "warn",
            "no Google OAuth client saved",
            "Add your Google OAuth client ID and secret in GTA Mode's Posting tab.",
        )
    elif not yt.get("linked"):
        youtube = Check(
            "youtube",
            "Posting",
            "YouTube",
            "warn",
            "channel not connected",
            "Press Connect YouTube in GTA Mode's Posting tab.",
        )
    elif not yt.get("can_edit"):
        youtube = Check(
            "youtube",
            "Posting",
            "YouTube",
            "warn",
            f"connected{(' as ' + yt['account']) if yt.get('account') else ''}, upload-only",
            "Reconnect and allow managing the channel so scheduled videos can be cancelled.",
        )
    else:
        youtube = Check(
            "youtube",
            "Posting",
            "YouTube",
            "ok",
            f"connected{(' as ' + yt['account']) if yt.get('account') else ''}",
        )
    key = p.zernio_key()
    if not key:
        zernio = Check(
            "zernio",
            "Posting",
            "Instagram (Zernio)",
            "warn",
            "no Zernio API key saved",
            "Add your Zernio API key in GTA Mode's Posting tab (only if you post Reels).",
        )
    elif live:
        try:
            ok, message = p.zernio_live(key)
        except Exception as exc:
            ok, message = False, str(exc)[:200]
        zernio = Check(
            "zernio",
            "Posting",
            "Instagram (Zernio)",
            "ok" if ok else "fail",
            message,
            "" if ok else "Check the key and the Instagram connection at zernio.com.",
        )
    else:
        zernio = Check(
            "zernio", "Posting", "Instagram (Zernio)", "ok", "API key saved (not contacted)"
        )
    return [youtube, zernio]


def _storage_checks(p: Probes) -> list[Check]:
    home = p.home()
    out: list[Check] = []
    probe = home
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = p.disk_usage(str(probe))
        free = int(usage.free)
    except OSError as exc:
        out.append(Check("disk", "Storage", "Disk space", "warn", f"can't read: {exc}"))
    else:
        free_gb = free / GB
        if free < DISK_FAIL_BYTES:
            status: Status = "fail"
        elif free < DISK_WARN_BYTES:
            status = "warn"
        else:
            status = "ok"
        out.append(
            Check(
                "disk",
                "Storage",
                "Disk space",
                status,
                f"{free_gb:.1f} GB free for {home}",
                ""
                if status == "ok"
                else "Free some space, lower the retention days in Settings, or move "
                "AUTOCLIP_HOME to a bigger drive.",
            )
        )
    path = p.db_path()
    version = p.db_version(path)
    latest = p.schema_version()
    if version is None:
        db = Check(
            "db",
            "Storage",
            "Database",
            "warn",
            "not created yet",
            "It is created on the first start (autoclip serve or autoclip init).",
        )
    elif version < 0:
        db = Check(
            "db",
            "Storage",
            "Database",
            "fail",
            f"can't be opened: {path}",
            "Close other programs using it; restore a backup if it is damaged.",
        )
    elif version > latest:
        db = Check(
            "db",
            "Storage",
            "Database",
            "fail",
            f"version {version} is newer than this build ({latest})",
            "Update AutoClip to the latest version.",
        )
    elif version < latest:
        db = Check(
            "db",
            "Storage",
            "Database",
            "warn",
            f"version {version}, will migrate to {latest} on the next start",
            "Restart AutoClip.",
        )
    else:
        db = Check("db", "Storage", "Database", "ok", f"version {version} (current)")
    out.append(db)
    return out


def _gpu_checks(report: Any) -> list[Check]:
    gpu = report.gpu
    if gpu.name and gpu.ctranslate2_cuda:
        return [Check("cuda", "GPU", "GPU / CUDA", "ok", f"{gpu.name}: Whisper runs on the GPU")]
    if gpu.name:
        missing = " and ".join(gpu.cuda_libraries_missing) or "the CUDA runtime"
        return [
            Check(
                "cuda",
                "GPU",
                "GPU / CUDA",
                "warn",
                f"{gpu.name} found, but {missing} can't be loaded: Whisper uses the CPU",
                "Install the CUDA libraries: uv pip install 'autoclip[gpu]'",
            )
        ]
    return [Check("cuda", "GPU", "GPU / CUDA", "ok", "no NVIDIA GPU: transcription on the CPU")]


def run_checks(*, live: bool = False, probes: Probes | None = None) -> list[Check]:
    """Every check, in display order."""
    p = probes or Probes()
    report = p.report()
    settings = p.settings()
    checks: list[Check] = [
        Check(
            "python",
            "Runtime",
            "Python",
            "ok" if report.python_ok else "fail",
            report.python_version,
            "" if report.python_ok else "Use Python 3.11 or 3.12 (uv venv --python 3.11).",
        )
    ]
    checks += _video_checks(report)
    checks += _ytdlp_checks(p)
    checks += _provider_checks(p, settings, live)
    checks += _secret_checks(report, settings)
    checks += _connection_checks(p, live)
    checks += _storage_checks(p)
    checks += _gpu_checks(report)
    return checks


def summary(checks: list[Check]) -> dict[str, Any]:
    counts = {s: sum(1 for c in checks if c.status == s) for s in ("ok", "warn", "fail")}
    overall: Status = "fail" if counts["fail"] else "warn" if counts["warn"] else "ok"
    return {"status": overall, **counts}


def doctor_report(*, live: bool = False, probes: Probes | None = None) -> dict[str, Any]:
    checks = run_checks(live=live, probes=probes)
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "live": live,
        "summary": summary(checks),
        "checks": [c.as_dict() for c in checks],
    }
