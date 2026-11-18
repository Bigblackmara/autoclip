"""Keep yt-dlp current.

YouTube changes often enough that an install a few weeks old starts failing
with errors such as "nsig extraction failed" or HTTP 403 on every format. So,
while ``ingest.ytdlp_auto_update`` is on:

* at startup and every 24 hours, the installed version is compared to PyPI's
  latest (https://pypi.org/pypi/yt-dlp/json) and, when PyPI is newer,
  ``<python> -m pip install -U "yt-dlp[default]"`` runs (``uv pip`` where the
  environment has no pip; ``[default]`` keeps yt-dlp-ejs, YouTube's player
  challenge solver, in step);
* a YouTube download that fails with an "outdated yt-dlp" error triggers one
  such check (at most once an hour), and the download is retried once.

The running process imported yt_dlp at its first download and that module
stays in memory, so an upgrade on disk does not change it. Reloading it in
place (``importlib.reload``) is not safe: yt_dlp is a large package whose
submodules would mix old and new code, possibly under a download running in
another thread. Instead, once the version on disk differs from the one in
memory, normal-mode YouTube downloads run in a fresh Python subprocess
(:mod:`autoclip.pipeline.ytdlp_worker`), which imports the new version. Other
yt-dlp users (GTA Mode, Live Watch, comment fetching) keep the in-memory
version until AutoClip restarts; the status says when a restart is pending.

Every check and its outcome is logged and kept in ``<home>/ytdlp_update.json``
for ``GET /api/system/ytdlp``.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from . import paths

log = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/yt-dlp/json"
PACKAGE = "yt-dlp[default]"
STATE_FILE = "ytdlp_update.json"

CHECK_EVERY_S = 24 * 3600.0
#: Startup checks wait a little so they don't compete with the app booting.
STARTUP_DELAY_S = 20.0
#: A failing download triggers at most one check per this long.
FAILURE_CHECK_COOLDOWN_S = 3600.0
PIP_TIMEOUT_S = 600.0

#: yt-dlp error fragments that typically mean "this yt-dlp is outdated".
OUTDATED_MARKERS = (
    "sign in to confirm",
    "nsig extraction failed",
    "signature extraction failed",
    "unable to extract",
    "http error 403",
    "failed to extract any player response",
    "please report this issue",
)


@dataclass
class UpdateState:
    #: ISO timestamps (UTC).
    last_check_at: str | None = None
    last_update_at: str | None = None
    last_failure_check_at: str | None = None
    installed_version: str | None = None
    latest_version: str | None = None
    #: "" (never checked), up_to_date, updated, failed.
    last_result: str = ""
    #: What started the last check: startup, scheduled, download_failure, manual.
    last_reason: str = ""
    last_message: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> UpdateState:
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**known)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------


def parse_version(value: str) -> tuple[int, ...]:
    """yt-dlp's date versions ("2026.10.26", "2026.10.26.1") as a tuple."""
    head = re.split(r"[^\d.]", value.strip(), maxsplit=1)[0]
    return tuple(int(part) for part in head.split(".") if part.isdigit())


def is_newer(candidate: str | None, current: str | None) -> bool:
    if not candidate:
        return False
    if not current:
        return True
    return parse_version(candidate) > parse_version(current)


def installed_version() -> str | None:
    """The version on disk (what a fresh interpreter would import)."""
    try:
        return importlib.metadata.version("yt-dlp")
    except importlib.metadata.PackageNotFoundError:
        return None


def loaded_version() -> str | None:
    """The version imported into this process, or None if not imported yet."""
    module = sys.modules.get("yt_dlp.version")
    version = getattr(module, "__version__", None) if module else None
    return str(version) if version else None


def stale_in_process() -> bool:
    """True when this process holds an older yt_dlp than the one on disk."""
    loaded, installed = loaded_version(), installed_version()
    return bool(loaded and installed and loaded != installed)


def looks_outdated(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in OUTDATED_MARKERS)


def fetch_latest_version(client: httpx.Client | None = None) -> str:
    """PyPI's latest yt-dlp version. Raises on any failure."""
    own = client is None
    client = client or httpx.Client(timeout=15.0, headers={"User-Agent": "autoclip"})
    try:
        response = client.get(PYPI_URL)
        response.raise_for_status()
        version = response.json()["info"]["version"]
    finally:
        if own:
            client.close()
    if not isinstance(version, str) or not parse_version(version):
        raise ValueError(f"PyPI answered with an unexpected version {version!r}")
    return version


# --------------------------------------------------------------------------
# Upgrading
# --------------------------------------------------------------------------


def upgrade_command(runner: Callable[..., Any] = subprocess.run) -> list[str] | None:
    """The command that upgrades yt-dlp in this environment, or None.

    pip when this interpreter has it; uv's pip otherwise (uv-made venvs
    ship without pip).
    """
    has_pip = (
        runner(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if has_pip:
        return [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE]
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, "--upgrade", PACKAGE]
    return None


def run_upgrade(runner: Callable[..., Any] = subprocess.run) -> tuple[bool, str]:
    """Run the upgrade. Returns (succeeded, combined output)."""
    command = upgrade_command(runner)
    if command is None:
        return False, (
            "This environment has neither pip nor uv. Install uv, or run: "
            f"{sys.executable} -m ensurepip"
        )
    log.info("Upgrading yt-dlp: %s", " ".join(command))
    try:
        result = runner(command, capture_output=True, text=True, check=False, timeout=PIP_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
    return result.returncode == 0, output


# --------------------------------------------------------------------------
# State + decisions
# --------------------------------------------------------------------------


def state_path():
    return paths.root() / STATE_FILE


def load_state() -> UpdateState:
    try:
        raw = json.loads(state_path().read_text(encoding="utf-8"))
        return UpdateState.from_dict(raw) if isinstance(raw, dict) else UpdateState()
    except (OSError, ValueError, TypeError):
        return UpdateState()


def save_state(state: UpdateState) -> None:
    path = state_path()
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        os.replace(tmp, path)


def _tail(text: str, lines: int = 6) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])[:1500]


class Updater:
    """Decides when to check and upgrade; every dependency is injectable."""

    def __init__(
        self,
        *,
        fetch_latest: Callable[[], str] = fetch_latest_version,
        upgrade: Callable[[], tuple[bool, str]] = run_upgrade,
        installed: Callable[[], str | None] = installed_version,
        now: Callable[[], datetime] = _now,
    ) -> None:
        self.fetch_latest = fetch_latest
        self.upgrade = upgrade
        self.installed = installed
        self.now = now
        self._lock = threading.Lock()
        self.running = False

    def check(self, *, reason: str) -> UpdateState:
        """Compare with PyPI and upgrade when PyPI is newer. Never raises."""
        with self._lock:
            self.running = True
            try:
                return self._check_locked(reason)
            finally:
                self.running = False

    def _check_locked(self, reason: str) -> UpdateState:
        state = load_state()
        state.last_check_at = _iso(self.now())
        state.last_reason = reason
        current = self.installed()
        state.installed_version = current
        try:
            latest = self.fetch_latest()
        except Exception as exc:
            state.last_result = "failed"
            state.last_message = f"Could not ask PyPI for the latest yt-dlp: {exc}"
            log.warning("yt-dlp update check (%s) failed: %s", reason, exc)
            save_state(state)
            return state

        state.latest_version = latest
        if not is_newer(latest, current):
            state.last_result = "up_to_date"
            state.last_message = f"yt-dlp {current} is the latest."
            log.info("yt-dlp update check (%s): %s is the latest.", reason, current)
            save_state(state)
            return state

        log.info("yt-dlp update check (%s): upgrading %s -> %s.", reason, current, latest)
        ok, output = self.upgrade()
        after = self.installed()
        state.installed_version = after
        if ok and after and after != current:
            state.last_result = "updated"
            state.last_update_at = _iso(self.now())
            state.last_message = f"Updated yt-dlp {current} -> {after}."
            log.info("Updated yt-dlp %s -> %s.", current, after)
        else:
            state.last_result = "failed"
            state.last_message = (
                f"Upgrading yt-dlp to {latest} failed: {_tail(output)}"
                if not ok
                else f"pip finished but yt-dlp is still {after}."
            )
            log.warning("%s", state.last_message)
        save_state(state)
        return state

    def update_for_failure(self, message: str, failed_version: str | None) -> bool:
        """After a failed YouTube download: maybe upgrade, and say whether to retry.

        Returns True when a fresh interpreter would now run a different
        yt-dlp than ``failed_version`` (the one that just failed), which is
        the only case where retrying can help.
        """
        if not looks_outdated(message):
            return False
        state = load_state()
        last = _parse_iso(state.last_failure_check_at)
        now = self.now()
        if last is None or now - last >= timedelta(seconds=FAILURE_CHECK_COOLDOWN_S):
            state.last_failure_check_at = _iso(now)
            save_state(state)
            log.warning(
                "YouTube download failed in a way that usually means yt-dlp is outdated; "
                "checking for an update."
            )
            self.check(reason="download_failure")
        else:
            log.info("Skipping the yt-dlp update check: one ran less than an hour ago.")
        current = self.installed()
        return bool(current and failed_version and current != failed_version)

    def status(self, *, auto_update: bool) -> dict[str, Any]:
        state = load_state()
        installed = self.installed()
        loaded = loaded_version()
        return {
            "auto_update": auto_update,
            "installed_version": installed,
            "loaded_version": loaded,
            "latest_version": state.latest_version,
            "last_check_at": state.last_check_at,
            "last_update_at": state.last_update_at,
            "last_result": state.last_result,
            "last_reason": state.last_reason,
            "last_message": state.last_message,
            "restart_pending": bool(loaded and installed and loaded != installed),
            "checking": self.running,
        }


updater = Updater()


def _auto_update_enabled() -> bool:
    from .config import load

    try:
        return load().ingest.ytdlp_auto_update
    except Exception:  # a broken config must not stop the app
        return True


class UpdaterService:
    """Background schedule: one check at startup, then every 24 hours."""

    def __init__(self, target: Updater | None = None) -> None:
        self.target = target or updater
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="autoclip-ytdlp-update")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _loop(self) -> None:
        await asyncio.sleep(STARTUP_DELAY_S)
        reason = "startup"
        while True:
            if await asyncio.to_thread(_auto_update_enabled):
                try:
                    await asyncio.to_thread(self.target.check, reason=reason)
                except Exception:  # pragma: no cover - check() never raises
                    log.exception("yt-dlp update check crashed")
            else:
                log.debug("yt-dlp auto-update is off; skipping the %s check.", reason)
            reason = "scheduled"
            await asyncio.sleep(CHECK_EVERY_S)


service = UpdaterService()
