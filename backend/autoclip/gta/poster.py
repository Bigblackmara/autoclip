"""GTA Mode posting schedule: the best finished clip goes out per slot.

Platforms: YouTube (scheduled ``publishAt`` upload) and Instagram Reels
through Zernio (zernio.com, a post scheduled at Zernio for the slot). Each
enabled platform runs its own pass: per slot it takes the best clip not yet
posted THERE, so normally the same clip goes to both at the same time.

Every daily slot ("12:00", "17:00", ...) gets one post per platform.
``lead_minutes`` before a slot, the best-scored clip that passed QA and isn't
posted yet is uploaded: to YouTube as PRIVATE with ``publishAt`` = the slot,
to Zernio as a post scheduled for the slot, so the platform itself makes it
public exactly on time, whether or not AutoClip is still running then.

Safety rules, because a duplicate or a wrong post is worse than a missed one:

* One upload at a time, one row per clip per platform (unique index), one
  clip per slot per platform.
* A slot whose time has (nearly) passed is never back-filled: no burst of
  posts after the PC was off.
* A YouTube upload cut short by AutoClip stopping is marked ``interrupted``
  and is never retried on its own (YouTube may have it already).
* Failed uploads retry after ``RETRY_AFTER_S``, at most ``MAX_ATTEMPTS``
  times; quota/limit errors pause posting without using up a clip's attempts.
* Instagram: Zernio has no idempotency key, so a create whose answer was
  unclear is ``checking``: AutoClip keeps asking Zernio (by post id or the
  uploaded media URL) and never sends it again on its own.
* Variants (a re-cut of the same moment) are separate clips, so one can go
  out on another day or platform, but never two of one family (an original
  and its variants) on the same platform on the same local day.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..live.uploads import tokens as tk
from ..live.uploads.tokens import UploadMeta, UploadResult
from . import store, zernio
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

#: Instagram follow-ups: re-ask Zernio about an unclear post this often, and
#: confirm a scheduled reel went live from 5 min after its slot, every 10 min,
#: for a day.
IG_CHECK_EVERY_S = 300.0
IG_CONFIRM_AFTER_S = 300.0
IG_CONFIRM_EVERY_S = 600.0
IG_CONFIRM_FOR_S = 86_400.0
#: Account trouble / rate limit / outage at Zernio: pause Instagram this long.
IG_PAUSE_S = 900.0
IG_CAPTION_MAX = 2200
IG_HASHTAGS_MAX = 30
IG_INTERRUPTED_NOTE = (
    "AutoClip stopped while sending this reel to Zernio. It is being checked "
    "with Zernio and will not be sent twice."
)
IG_NOT_SENT_NOTE = "AutoClip stopped before this reel was sent; it will be tried again."
IG_NOT_LISTED_NOTE = "Zernio doesn't list this reel. If it isn't on Instagram either, press Retry."

PLATFORM_LABEL = {"youtube": "YouTube", "instagram": "Instagram"}

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


def slot_day(key: str, tz: Any = None) -> str:
    """The local calendar day (``YYYY-MM-DD``) of a stored slot key."""
    return parse_slot_key(key).astimezone(tz).date().isoformat()


def families_by_day(platform: str, tz: Any = None) -> dict[str, set[str]]:
    """Local day -> families (original clip ids) already holding a slot there.

    Covers every slot-holding row of ``platform`` (uploading, scheduled,
    posted, interrupted, checking), so a variant never lands on the same
    day as its original or a sibling, whichever went first.
    """
    rows = store.list_uploads(
        statuses=store.SLOT_HOLDING_STATUSES, platform=platform, limit=100_000
    )
    family: dict[str, str] = {}
    out: dict[str, set[str]] = {}
    for row in rows:
        if not row.slot_at:
            continue
        if row.clip_id not in family:
            clip = store.get_clip(row.clip_id)
            family[row.clip_id] = clip.family_id if clip is not None else row.clip_id
        out.setdefault(slot_day(row.slot_at, tz), set()).add(family[row.clip_id])
    return out


def eligible_clips(
    now: datetime,
    platform: str = "youtube",
    *,
    slot: datetime | None = None,
    tz: Any = None,
) -> list[tuple[GtaClip, GtaUpload | None]]:
    """Postable clips for ``platform`` in posting order (best first), each
    with its failed row there, if any. A clip skipped anywhere is skipped.

    With ``slot``, a clip whose family (original + variants) already has a
    post on ``platform`` that local day is left out.
    """
    rows = store.list_uploads(limit=100_000)
    skipped = {u.clip_id for u in rows if u.status == "skipped"}
    uploads = {u.clip_id: u for u in rows if u.platform == platform}
    busy: set[str] = set()
    if slot is not None:
        busy = families_by_day(platform, tz).get(slot.astimezone(tz).date().isoformat(), set())
    out: list[tuple[GtaClip, GtaUpload | None]] = []
    for clip in store.list_clips(statuses=frozenset({"ready"}), limit=100_000):
        if clip.id in skipped or not is_postable(clip) or clip.family_id in busy:
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
    from .pipeline import shown_hook

    title = _clean_text(shown_hook(clip))
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


def instagram_caption(clip: GtaClip) -> str:
    """Caption + hashtags for a Reel, within Instagram's 2200 chars / 30 tags.

    The caption is the one written in the caption language at render time.
    """
    tags: list[str] = []
    seen: set[str] = set()
    for tag in clip.hashtags:
        word = tag.strip().lstrip("#")
        if not word or " " in word or word.lower() in seen:
            continue
        tags.append(f"#{word}")
        seen.add(word.lower())
        if len(tags) == IG_HASHTAGS_MAX:
            break
    tail = " ".join(tags)
    room = IG_CAPTION_MAX - (len(tail) + 2 if tail else 0)
    text = (str(clip.localised.get("caption") or "") or clip.caption).strip()
    if len(text) > room:
        text = text[: max(0, room - 1)].rstrip() + "…"
    return f"{text}\n\n{tail}".strip()[:IG_CAPTION_MAX]


def readiness() -> str:
    """Why YouTube uploads can't run right now ('' = ready). Reads the OS keyring."""
    try:
        if not tk.has_rights_attestation():
            return "Accept the rights notice in the Posting tab to unlock uploads."
        if tk.load_client_credentials(tk.PLATFORM_YOUTUBE) is None:
            return "Add your Google OAuth client ID and secret in the Posting tab."
        if tk.load_tokens(tk.PLATFORM_YOUTUBE) is None:
            return "Connect your YouTube channel in the Posting tab."
    except tk.UploadError as exc:
        return f"YouTube connection problem: {exc}"
    return ""


def instagram_readiness() -> str:
    """Why Instagram posts can't run right now ('' = ready)."""
    try:
        if not tk.has_rights_attestation():
            return "Accept the rights notice in the Posting tab to unlock uploads."
        if not zernio.load_api_key():
            return "Add your Zernio API key in the Posting tab."
    except Exception as exc:  # keyring trouble
        return f"Instagram connection problem: {exc}"
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
        instagram_ready_check: Callable[[], str] = instagram_readiness,
        instagram: Any = None,
        tz: Any = None,
    ) -> None:
        self._uploader = uploader or _default_uploader
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ready_checks = {"youtube": ready_check, "instagram": instagram_ready_check}
        #: Zernio client (``schedule_reel`` / ``check_post``); tests pass a fake.
        self._ig = instagram or zernio
        self._tz = tz
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self.states = {platform: PosterState() for platform in store.PLATFORMS}

    @property
    def state(self) -> PosterState:
        """YouTube's state (kept for older callers)."""
        return self.states["youtube"]

    def uploading_clip_ids(self) -> set[str]:
        return {s.uploading_clip_id for s in self.states.values() if s.uploading_clip_id}

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
        """Rows a crash left unfinished.

        YouTube "uploading" -> interrupted (the user checks Studio). Instagram
        "uploading" -> checking once its media URL was recorded (the create may
        have gone out; it's looked up, never re-sent), else failed (nothing
        was sent). Instagram "checking" with nothing to look up by -> failed.
        """
        count = 0
        for row in store.list_uploads(statuses=frozenset({"uploading", "checking"})):
            if row.platform == "youtube":
                if row.status == "uploading":
                    store.update_upload(row.id, status="interrupted", error=INTERRUPTED_NOTE)
                    count += 1
                continue
            if row.media_url or row.video_id:
                if row.status == "uploading":
                    store.update_upload(row.id, status="checking", error=IG_INTERRUPTED_NOTE)
                    count += 1
            else:
                store.update_upload(row.id, status="failed", error=IG_NOT_SENT_NOTE)
                count += 1
        return count

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

    @staticmethod
    def platforms_shown(settings: GtaSettings) -> list[str]:
        posting = settings.posting
        chosen = [p for p in store.PLATFORMS if getattr(posting, p)]
        return chosen or ["youtube"]

    def plan(self, settings: GtaSettings, now: datetime | None = None) -> list[dict[str, Any]]:
        """Upcoming slots per platform: what holds each, or which clip goes next."""
        now = now or self._clock()
        moments = slot_datetimes(settings.posting.slots, now, tz=self._tz)
        out: list[dict[str, Any]] = []
        for platform in self.platforms_shown(settings):
            holders = {
                u.slot_at: u
                for u in store.list_uploads(
                    statuses=store.SLOT_HOLDING_STATUSES, platform=platform, limit=100_000
                )
                if u.slot_at
            }
            queue = [clip for clip, _ in eligible_clips(now, platform)]
            # Families already out that day, plus the ones this plan places.
            days = families_by_day(platform, self._tz)
            for moment in moments:
                key = slot_key(moment)
                row = holders.get(key)
                entry: dict[str, Any] = {
                    "slot_at": key,
                    "platform": platform,
                    "status": "open",
                    "clip_id": None,
                }
                day = days.setdefault(moment.astimezone(self._tz).date().isoformat(), set())
                if row is not None:
                    entry.update(status=row.status, clip_id=row.clip_id, upload_id=row.id)
                elif (moment - now).total_seconds() < MIN_LEAD_S:
                    entry["status"] = "missed"
                else:
                    pick = next((c for c in queue if c.family_id not in day), None)
                    if pick is not None:
                        queue.remove(pick)
                        day.add(pick.family_id)
                        entry.update(status="planned", clip_id=pick.id)
                out.append(entry)
        order = {p: i for i, p in enumerate(store.PLATFORMS)}
        out.sort(key=lambda e: (e["slot_at"], order[e["platform"]]))
        return out

    # -- one pass --------------------------------------------------------

    async def run_once(self) -> dict[str, str]:
        """One scheduling pass per platform; what happened on each."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            settings = await asyncio.to_thread(load_settings)
            results: dict[str, str] = {}
            for platform in store.PLATFORMS:
                results[platform] = await self._run_platform(platform, settings)
            return results

    async def _run_platform(self, platform: str, settings: GtaSettings) -> str:
        state = self.states[platform]
        label = PLATFORM_LABEL[platform]
        now = self._clock()
        state.last_run = now
        posting = settings.posting
        if not posting.enabled:
            state.message = "Posting schedule is off."
            return "off"
        if not getattr(posting, platform):
            state.message = f"Posting to {label} is off."
            return "off"
        if state.paused_until is not None and now < state.paused_until:
            return "paused"
        state.paused_until = None
        problem = await asyncio.to_thread(self._ready_checks[platform])
        if problem:
            state.message = problem
            return "not_ready"
        if platform == "instagram":
            stopped = await self._instagram_followups(now)
            if stopped:
                return stopped

        holding = await asyncio.to_thread(
            store.list_uploads,
            statuses=store.SLOT_HOLDING_STATUSES,
            platform=platform,
            limit=100_000,
        )
        taken = {u.slot_at for u in holding if u.slot_at}
        slot = due_slot(posting.slots, posting.lead_minutes, now, taken, tz=self._tz)
        if slot is None:
            upcoming = [
                m
                for m in slot_datetimes(posting.slots, now, tz=self._tz)
                if slot_key(m) not in taken and (m - now).total_seconds() >= MIN_LEAD_S
            ]
            state.message = (
                f"Waiting for the next free slot ({slot_key(upcoming[0])})."
                if upcoming
                else "Every listed slot already has a clip."
            )
            return "waiting"

        candidates = await asyncio.to_thread(
            lambda: eligible_clips(now, platform, slot=slot, tz=self._tz)
        )
        if not candidates:
            state.message = (
                f"No finished clip that passed QA is waiting for the {slot_key(slot)} slot."
            )
            return "no_clip"
        clip, row = candidates[0]
        if platform == "instagram":
            return await self._upload_instagram(clip, row, slot, settings)
        return await self._upload(clip, row, slot, settings)

    async def _claim(
        self, platform: str, clip: GtaClip, row: GtaUpload | None, key: str, title: str
    ) -> GtaUpload:
        """Mark the clip's row for ``platform`` as uploading for slot ``key``."""
        if row is None:
            return await asyncio.to_thread(
                store.create_upload,
                clip.id,
                status="uploading",
                platform=platform,
                slot_at=key,
                title=title,
            )
        await asyncio.to_thread(
            store.update_upload,
            row.id,
            status="uploading",
            slot_at=key,
            title=title,
            error=None,
            video_id="",
            url="",
            media_url="",
        )
        return row

    async def post_now(
        self, clip_id: str, title: str = "", publish_at: datetime | None = None
    ) -> str:
        """Upload one ready clip to YouTube as public right now (outside the slots).

        ``title`` replaces the clip's hook title when given. Returns the
        outcome ("posted", "failed", "paused"); raises ValueError when the
        clip can't be posted (not ready, or already on YouTube).
        """
        clip = await asyncio.to_thread(store.get_clip, clip_id)
        if clip is None or not is_postable(clip):
            raise ValueError("That clip isn't ready to post (finished and QA passed).")
        row = await asyncio.to_thread(store.upload_for_clip, clip_id, "youtube")
        if row is not None and row.status in {"uploading", "scheduled", "posted", "checking"}:
            raise ValueError("That clip is already on YouTube or being uploaded.")
        settings = await asyncio.to_thread(load_settings)
        if publish_at is not None:
            if publish_at.tzinfo is None:
                raise ValueError("The schedule time needs a timezone.")
            if publish_at <= self._clock() + timedelta(minutes=2):
                raise ValueError("Pick a time at least a few minutes from now.")
            # Uploaded private now; YouTube makes it public at publish_at.
            return await self._upload(clip, row, publish_at, settings, title_override=title)
        return await self._upload(
            clip, row, self._clock(), settings, title_override=title, public_now=True
        )

    async def _upload(
        self,
        clip: GtaClip,
        row: GtaUpload | None,
        slot: datetime,
        settings: GtaSettings,
        *,
        title_override: str = "",
        public_now: bool = False,
    ) -> str:
        state = self.states["youtube"]
        key = slot_key(slot)
        title = (_clean_text(title_override) or post_title(clip))[:100]
        row = await self._claim("youtube", clip, row, key, title)
        state.uploading_clip_id = clip.id
        state.message = (
            f"Posting “{title}” now…" if public_now else f"Uploading “{title}” for the {key} slot…"
        )
        try:
            meta = build_meta(clip, settings, slot)
            meta = replace(
                meta,
                title=title,
                **({"privacy": "public", "publish_at": None} if public_now else {}),
            )
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
            state.message = f"YouTube connection problem: {exc}"
            return "failed"
        except Exception as exc:  # UploadApiError, OSError, ...
            message = str(exc) or type(exc).__name__
            if _is_quota_error(message):
                await asyncio.to_thread(
                    store.update_upload, row.id, status="failed", error=f"Quota: {message[:400]}"
                )
                state.paused_until = self._clock() + timedelta(seconds=QUOTA_PAUSE_S)
                state.message = "YouTube's upload limit was hit; posting paused for an hour."
                return "paused"
            await asyncio.to_thread(
                store.update_upload,
                row.id,
                status="failed",
                error=message[:500],
                attempts=row.attempts + 1,
            )
            log.warning("GTA upload of %s failed: %s", clip.id, message)
            state.message = f"Upload of “{title}” failed: {message[:200]}"
            return "failed"
        finally:
            state.uploading_clip_id = None

        await asyncio.to_thread(
            store.update_upload,
            row.id,
            status="posted" if public_now else "scheduled",
            video_id=result.media_id,
            url=result.url,
            error=None,
            attempts=row.attempts + 1,
        )
        if public_now:
            state.message = f"“{title}” is live on YouTube."
            log.info("GTA clip %s posted now as %s", clip.id, result.media_id)
            return "posted"
        state.message = f"Scheduled “{title}” to go public at {key}."
        log.info("GTA clip %s uploaded as %s, public at %s", clip.id, result.media_id, key)
        return "scheduled"

    # -- Instagram (Zernio) ----------------------------------------------

    def _pause_instagram(self, seconds: float, message: str) -> str:
        state = self.states["instagram"]
        state.paused_until = self._clock() + timedelta(seconds=max(seconds, 60.0))
        state.message = message
        return "paused"

    async def _upload_instagram(
        self, clip: GtaClip, row: GtaUpload | None, slot: datetime, settings: GtaSettings
    ) -> str:
        state = self.states["instagram"]
        key = slot_key(slot)
        title = post_title(clip)
        api_key = await asyncio.to_thread(zernio.load_api_key) or ""
        row = await self._claim("instagram", clip, row, key, title)
        row_id, attempts = row.id, row.attempts

        def remember_media(url: str) -> None:
            store.update_upload(row_id, media_url=url)

        def remember_post(post_id: str) -> None:
            store.update_upload(row_id, video_id=post_id)

        state.uploading_clip_id = clip.id
        state.message = f"Sending “{title}” to Zernio for the {key} slot…"
        try:
            result = await asyncio.to_thread(
                self._ig.schedule_reel,
                api_key,
                settings.posting.instagram_account,
                clip.path,
                instagram_caption(clip),
                slot,
                on_media_url=remember_media,
                on_post_id=remember_post,
            )
        except asyncio.CancelledError:
            # The worker thread may still create the post: look it up later.
            await asyncio.to_thread(
                store.update_upload, row_id, status="checking", error=IG_INTERRUPTED_NOTE
            )
            raise
        except Exception as exc:  # a bug or local I/O error; nothing confirmed sent
            message = str(exc) or type(exc).__name__
            fresh = await asyncio.to_thread(store.get_upload, row_id)
            sent = fresh is not None and bool(fresh.media_url or fresh.video_id)
            await asyncio.to_thread(
                store.update_upload,
                row_id,
                status="checking" if sent else "failed",
                error=message[:500],
                attempts=attempts + 1,
            )
            log.warning("GTA Instagram post of %s failed: %s", clip.id, message)
            state.message = f"Instagram post of “{title}” failed: {message[:200]}"
            return "checking" if sent else "failed"
        finally:
            state.uploading_clip_id = None

        fields: dict[str, Any] = {"error": (result.message or "")[:500] or None}
        if result.post_id:
            fields["video_id"] = result.post_id
        if result.media_url:
            fields["media_url"] = result.media_url
        kind = result.kind
        if kind in {"scheduled", "posted"}:
            fields["error"] = None
            await asyncio.to_thread(
                store.update_upload,
                row_id,
                status=kind,
                url=result.post_url,
                attempts=attempts + 1,
                **fields,
            )
            state.message = (
                f"Scheduled “{title}” on Instagram for {key}."
                if kind == "scheduled"
                else f"“{title}” is live on Instagram."
            )
            log.info("GTA clip %s sent to Zernio (%s) for %s", clip.id, result.post_id, key)
            return kind
        if kind == "unconfirmed":
            await asyncio.to_thread(
                store.update_upload, row_id, status="checking", attempts=attempts + 1, **fields
            )
            state.message = f"Zernio's answer for “{title}” was unclear; checking with Zernio."
            return "checking"
        if kind == "blocked":
            await asyncio.to_thread(store.update_upload, row_id, status="failed", **fields)
            return self._pause_instagram(
                result.retry_after or IG_PAUSE_S, f"Instagram paused: {result.message[:200]}"
            )
        if kind == "invalid":
            await asyncio.to_thread(
                store.update_upload, row_id, status="failed", attempts=MAX_ATTEMPTS, **fields
            )
            state.message = f"Instagram can't take “{title}”: {result.message[:200]}"
            return "failed"
        # failed: nothing was published, safe to retry later
        await asyncio.to_thread(
            store.update_upload,
            row_id,
            status="failed",
            attempts=attempts + (0 if result.transient else 1),
            **fields,
        )
        if result.transient:
            return self._pause_instagram(
                IG_PAUSE_S, f"Zernio is unreachable; Instagram paused. ({result.message[:150]})"
            )
        state.message = f"Instagram post of “{title}” failed: {result.message[:200]}"
        return "failed"

    async def _instagram_followups(self, now: datetime) -> str:
        """Ask Zernio about unclear posts and confirm scheduled reels went live.

        Returns "paused" if Zernio blocked us, else "".
        """
        rows = await asyncio.to_thread(
            store.list_uploads,
            statuses=frozenset({"checking", "scheduled"}),
            platform="instagram",
            limit=500,
        )
        due: list[GtaUpload] = []
        for row in rows:
            if row.status == "checking":
                if _age_s(row.updated_at, now) >= IG_CHECK_EVERY_S:
                    due.append(row)
                continue
            if not row.slot_at or not (row.video_id or row.media_url):
                continue
            since = (now - parse_slot_key(row.slot_at)).total_seconds()
            if IG_CONFIRM_AFTER_S <= since <= IG_CONFIRM_FOR_S and (
                _age_s(row.updated_at, now) >= IG_CONFIRM_EVERY_S
            ):
                due.append(row)
        if not due:
            return ""
        api_key = await asyncio.to_thread(zernio.load_api_key) or ""
        for row in due:
            if not (row.video_id or row.media_url):
                await asyncio.to_thread(
                    store.update_upload, row.id, status="failed", error=IG_NOT_SENT_NOTE
                )
                continue
            # A lower bound for when it was sent (uploads start <= 1 day ahead).
            sent_after = parse_slot_key(row.slot_at) - timedelta(days=1) if row.slot_at else None
            try:
                result = await asyncio.to_thread(
                    self._ig.check_post, api_key, row.video_id, row.media_url, sent_after
                )
            except Exception as exc:
                log.warning("Zernio check of %s failed: %s", row.id, exc)
                await asyncio.to_thread(store.update_upload, row.id, error=row.error)
                continue
            fields: dict[str, Any] = {}
            if result.post_id:
                fields["video_id"] = result.post_id
            if result.kind == "posted":
                await asyncio.to_thread(
                    store.update_upload,
                    row.id,
                    status="posted",
                    url=result.post_url,
                    error=None,
                    **fields,
                )
            elif result.kind == "scheduled":
                await asyncio.to_thread(
                    store.update_upload, row.id, status="scheduled", error=None, **fields
                )
            elif result.kind == "failed":
                await asyncio.to_thread(
                    store.update_upload,
                    row.id,
                    status="failed",
                    error=(result.message or "Instagram did not accept the reel.")[:500],
                    **fields,
                )
            elif result.kind == "blocked":
                return self._pause_instagram(
                    result.retry_after or IG_PAUSE_S, f"Instagram paused: {result.message[:200]}"
                )
            else:  # unconfirmed: ask again later (the write touches updated_at)
                if row.status == "checking":
                    note = (
                        IG_NOT_LISTED_NOTE
                        if result.not_listed
                        else result.message or "Still waiting for Zernio's answer."
                    )
                else:
                    note = row.error
                await asyncio.to_thread(store.update_upload, row.id, error=note, **fields)
        return ""


#: The app-wide poster (started by the app lifespan when the worker runs).
poster = GtaPoster()
