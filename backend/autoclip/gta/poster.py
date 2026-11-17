"""GTA Mode posting schedule: the best finished clip goes to YouTube per slot.

Every daily slot ("12:00", "17:00", ...) gets one Short. ``lead_minutes``
before a slot, the best-scored clip that passed QA and isn't posted yet is
uploaded as PRIVATE with ``publishAt`` = the slot, so YouTube itself makes it
public exactly on time, whether or not AutoClip is still running then.

Safety rules, because a duplicate or a wrong post is worse than a missed one:

* One upload at a time, one row per clip (unique index), one clip per slot.
* A slot whose time has (nearly) passed is never back-filled: no burst of
  posts after the PC was off.
* An upload cut short by AutoClip stopping is marked ``interrupted`` and is
  never retried on its own (YouTube may have it already); the user decides.
* Failed uploads retry after ``RETRY_AFTER_S``, at most ``MAX_ATTEMPTS``
  times; quota/limit errors pause posting for an hour without using up a
  clip's attempts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..live.uploads import tokens as tk
from ..live.uploads.tokens import UploadMeta, UploadResult
from . import store
from .settings import GtaSettings
from .settings import load as load_settings
from .store import GtaClip, GtaUpload

log = logging.getLogger(__name__)

TICK_S = 30.0
#: Never start an upload for a slot closer than this (processing needs time).
MIN_LEAD_S = 120.0
RETRY_AFTER_S = 300.0
MAX_ATTEMPTS = 3
QUOTA_PAUSE_S = 3600.0
#: How many days ahead slots are listed.
HORIZON_DAYS = 2
TAGS_MAX_CHARS = 450  # YouTube's limit is 500 incl. separators; keep a margin
INTERRUPTED_NOTE = (
    "AutoClip stopped during this upload. Check YouTube Studio: "
    "if the video is there, leave it; if not, press Retry."
)

Uploader = Callable[[Path, UploadMeta], Awaitable[UploadResult]]


def _default_uploader(path: Path, meta: UploadMeta) -> Awaitable[UploadResult]:
    from ..live.uploads.youtube import upload_youtube

    return upload_youtube(path, meta)


# --------------------------------------------------------------------------
# Slots
# --------------------------------------------------------------------------


def slot_key(moment: datetime) -> str:
    """How a slot is stored: UTC, second precision, ``Z`` suffix."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_slot_key(key: str) -> datetime:
    return datetime.strptime(key, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def slot_datetimes(
    slots: list[str], now: datetime, *, days: int = HORIZON_DAYS, tz: Any = None
) -> list[datetime]:
    """Every slot after ``now`` for today and the next ``days`` days.

    Slots are wall-clock times in ``tz`` (default: this PC's local zone).
    """
    local_now = now.astimezone(tz)
    out: set[datetime] = set()
    for offset in range(days + 1):
        day = (local_now + timedelta(days=offset)).date()
        for slot in slots:
            hour, minute = (int(p) for p in slot.split(":"))
            naive = datetime(day.year, day.month, day.day, hour, minute)
            moment = naive.replace(tzinfo=tz) if tz is not None else naive.astimezone()
            if moment > now:
                out.add(moment)
    return sorted(out)


def due_slot(
    slots: list[str], lead_minutes: int, now: datetime, taken: set[str], *, tz: Any = None
) -> datetime | None:
    """The next free slot whose upload window is open, if any."""
    for moment in slot_datetimes(slots, now, tz=tz):
        if slot_key(moment) in taken:
            continue
        ahead = (moment - now).total_seconds()
        if ahead < MIN_LEAD_S:
            continue  # too close: never rush or back-fill a slot
        return moment if ahead <= lead_minutes * 60 else None
    return None


# --------------------------------------------------------------------------
# Which clip, and what it's posted as
# --------------------------------------------------------------------------


def _age_s(stamp: str, now: datetime) -> float:
    try:
        then = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds()


def is_postable(clip: GtaClip) -> bool:
    """Finished, QA passed, and the file is still on disk."""
    return (
        clip.status == "ready"
        and clip.qa.get("passed") is True
        and bool(clip.path)
        and Path(clip.path).is_file()
    )


def eligible_clips(now: datetime) -> list[tuple[GtaClip, GtaUpload | None]]:
    """Postable clips in posting order (best first) with their failed row, if any."""
    uploads = {u.clip_id: u for u in store.list_uploads(limit=100_000)}
    out: list[tuple[GtaClip, GtaUpload | None]] = []
    for clip in store.list_clips(statuses=frozenset({"ready"}), limit=100_000):
        if not is_postable(clip):
            continue
        row = uploads.get(clip.id)
        if row is not None and not (
            row.status == "failed"
            and row.attempts < MAX_ATTEMPTS
            and _age_s(row.updated_at, now) >= RETRY_AFTER_S
        ):
            continue
        out.append((clip, row))
    out.sort(key=lambda pair: (-pair[0].score, -pair[0].hype_score, pair[0].created_at))
    return out


def _clean_text(text: str) -> str:
    # YouTube rejects titles/descriptions containing angle brackets.
    return text.replace("<", "").replace(">", "").strip()


def post_title(clip: GtaClip) -> str:
    title = _clean_text(clip.hook_title)
    if not title and clip.caption:
        title = _clean_text(clip.caption.splitlines()[0])
    return (title or "Stream clip")[:100]


def build_meta(clip: GtaClip, settings: GtaSettings, publish_at: datetime) -> UploadMeta:
    from .pipeline import post_text

    posting = settings.posting
    description = _clean_text(post_text(clip))
    if posting.shorts_tag and "#shorts" not in description.lower():
        description = f"{description} #Shorts".strip()
    tags: list[str] = []
    seen: set[str] = set()
    used = 0
    for tag in clip.hashtags:
        word = _clean_text(tag.lstrip("#"))
        if not word or word.lower() in seen:
            continue
        if used + len(word) + 1 > TAGS_MAX_CHARS:
            break
        tags.append(word)
        seen.add(word.lower())
        used += len(word) + 1
    return UploadMeta(
        title=post_title(clip),
        description=description,
        tags=tags,
        privacy="private",
        category_id=posting.category_id,
        made_for_kids=False,
        publish_at=publish_at,
    )


def readiness() -> str:
    """Why uploads can't run right now ('' = ready). Reads the OS keyring."""
    try:
        if not tk.has_rights_attestation():
            return "Accept the rights notice in the YouTube tab to unlock uploads."
        if tk.load_client_credentials(tk.PLATFORM_YOUTUBE) is None:
            return "Add your Google OAuth client ID and secret in the YouTube tab."
        if tk.load_tokens(tk.PLATFORM_YOUTUBE) is None:
            return "Connect your YouTube channel in the YouTube tab."
    except tk.UploadError as exc:
        return f"YouTube connection problem: {exc}"
    return ""


def _is_quota_error(message: str) -> bool:
    lower = message.lower()
    return any(k in lower for k in ("quotaexceeded", "uploadlimitexceeded", "ratelimitexceeded"))


# --------------------------------------------------------------------------
# The poster
# --------------------------------------------------------------------------


@dataclass
class PosterState:
    message: str = "Posting schedule is off."
    paused_until: datetime | None = None
    uploading_clip_id: str | None = None
    last_run: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "paused_until": slot_key(self.paused_until) if self.paused_until else None,
            "uploading_clip_id": self.uploading_clip_id,
            "last_run": slot_key(self.last_run) if self.last_run else None,
        }


class GtaPoster:
    def __init__(
        self,
        *,
        uploader: Uploader | None = None,
        clock: Callable[[], datetime] | None = None,
        ready_check: Callable[[], str] = readiness,
        tz: Any = None,
    ) -> None:
        self._uploader = uploader or _default_uploader
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ready_check = ready_check
        self._tz = tz
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self.state = PosterState()

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        await asyncio.to_thread(self.recover)
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="gta-poster")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._wakeup = None
        self._loop = None
        self._lock = None

    @property
    def running(self) -> bool:
        return self._task is not None

    def notify(self) -> None:
        if self._loop is None or self._wakeup is None:
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._wakeup.set)

    @staticmethod
    def recover() -> int:
        """Rows left "uploading" by a crash: unknown outcome -> interrupted."""
        rows = store.list_uploads(statuses=frozenset({"uploading"}))
        for row in rows:
            store.update_upload(row.id, status="interrupted", error=INTERRUPTED_NOTE)
        return len(rows)

    async def _run(self) -> None:
        assert self._wakeup is not None
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("GTA posting pass failed.")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=TICK_S)
            self._wakeup.clear()

    # -- read side -------------------------------------------------------

    def plan(self, settings: GtaSettings, now: datetime | None = None) -> list[dict[str, Any]]:
        """Upcoming slots: what holds each, or which clip would go next."""
        now = now or self._clock()
        holders = {
            u.slot_at: u
            for u in store.list_uploads(statuses=store.SLOT_HOLDING_STATUSES, limit=100_000)
            if u.slot_at
        }
        queue = [clip for clip, _ in eligible_clips(now)]
        out: list[dict[str, Any]] = []
        for moment in slot_datetimes(settings.posting.slots, now, tz=self._tz):
            key = slot_key(moment)
            row = holders.get(key)
            entry: dict[str, Any] = {"slot_at": key, "status": "open", "clip_id": None}
            if row is not None:
                entry.update(status=row.status, clip_id=row.clip_id, upload_id=row.id)
            elif (moment - now).total_seconds() < MIN_LEAD_S:
                entry["status"] = "missed"
            elif queue:
                entry.update(status="planned", clip_id=queue.pop(0).id)
            out.append(entry)
        return out

    # -- one pass --------------------------------------------------------

    async def run_once(self) -> str:
        """One scheduling pass; returns what happened (for tests and the UI)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await self._run_once()

    async def _run_once(self) -> str:
        now = self._clock()
        self.state.last_run = now
        settings = await asyncio.to_thread(load_settings)
        posting = settings.posting
        if not posting.enabled:
            self.state.message = "Posting schedule is off."
            return "off"
        if self.state.paused_until is not None and now < self.state.paused_until:
            return "paused"
        self.state.paused_until = None
        problem = await asyncio.to_thread(self._ready_check)
        if problem:
            self.state.message = problem
            return "not_ready"

        holding = await asyncio.to_thread(
            store.list_uploads, statuses=store.SLOT_HOLDING_STATUSES, limit=100_000
        )
        taken = {u.slot_at for u in holding if u.slot_at}
        slot = due_slot(posting.slots, posting.lead_minutes, now, taken, tz=self._tz)
        if slot is None:
            upcoming = [
                m
                for m in slot_datetimes(posting.slots, now, tz=self._tz)
                if slot_key(m) not in taken and (m - now).total_seconds() >= MIN_LEAD_S
            ]
            self.state.message = (
                f"Waiting for the next free slot ({slot_key(upcoming[0])})."
                if upcoming
                else "Every listed slot already has a clip."
            )
            return "waiting"

        candidates = await asyncio.to_thread(eligible_clips, now)
        if not candidates:
            self.state.message = (
                f"No finished clip that passed QA is waiting for the {slot_key(slot)} slot."
            )
            return "no_clip"
        clip, row = candidates[0]
        return await self._upload(clip, row, slot, settings)

    async def _upload(
        self, clip: GtaClip, row: GtaUpload | None, slot: datetime, settings: GtaSettings
    ) -> str:
        key = slot_key(slot)
        title = post_title(clip)
        if row is None:
            row = await asyncio.to_thread(
                store.create_upload, clip.id, status="uploading", slot_at=key, title=title
            )
        else:
            await asyncio.to_thread(
                store.update_upload,
                row.id,
                status="uploading",
                slot_at=key,
                title=title,
                error=None,
            )
        self.state.uploading_clip_id = clip.id
        self.state.message = f"Uploading “{title}” for the {key} slot…"
        try:
            meta = build_meta(clip, settings, slot)
            result = await self._uploader(Path(clip.path), meta)
        except asyncio.CancelledError:
            await asyncio.to_thread(
                store.update_upload, row.id, status="interrupted", error=INTERRUPTED_NOTE
            )
            raise
        except (
            tk.NotLinkedError,
            tk.RightsNotAccepted,
            tk.TokenStateError,
            tk.TokenRefreshError,
        ) as exc:
            # The account side is broken, not the clip: keep its attempts.
            await asyncio.to_thread(
                store.update_upload, row.id, status="failed", error=f"YouTube connection: {exc}"
            )
            self.state.message = f"YouTube connection problem: {exc}"
            return "failed"
        except Exception as exc:  # UploadApiError, OSError, ...
            message = str(exc) or type(exc).__name__
            if _is_quota_error(message):
                await asyncio.to_thread(
                    store.update_upload, row.id, status="failed", error=f"Quota: {message[:400]}"
                )
                self.state.paused_until = self._clock() + timedelta(seconds=QUOTA_PAUSE_S)
                self.state.message = "YouTube's upload limit was hit; posting paused for an hour."
                return "paused"
            await asyncio.to_thread(
                store.update_upload,
                row.id,
                status="failed",
                error=message[:500],
                attempts=row.attempts + 1,
            )
            log.warning("GTA upload of %s failed: %s", clip.id, message)
            self.state.message = f"Upload of “{title}” failed: {message[:200]}"
            return "failed"
        finally:
            self.state.uploading_clip_id = None

        await asyncio.to_thread(
            store.update_upload,
            row.id,
            status="scheduled",
            video_id=result.media_id,
            url=result.url,
            error=None,
            attempts=row.attempts + 1,
        )
        self.state.message = f"Scheduled “{title}” to go public at {key}."
        log.info("GTA clip %s uploaded as %s, public at %s", clip.id, result.media_id, key)
        return "scheduled"


#: The app-wide poster (started by the app lifespan when the worker runs).
poster = GtaPoster()
