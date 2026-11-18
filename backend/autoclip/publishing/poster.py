"""The normal-mode poster: sends queued posts of exported clips.

A post is created by the user ("Post" on a clip) or booked by an Autopilot.
This loop (every ``TICK_S``) sends each one when it's due:

* **now** posts at once: YouTube as public (or the chosen privacy), Instagram
  as a reel Zernio publishes ``IG_NOW_DELAY_S`` later (Zernio's API only
  schedules);
* **scheduled** posts ``lead_minutes`` before their time: YouTube as PRIVATE
  with ``publishAt`` (YouTube makes it public on time even if AutoClip is
  closed), Instagram as a Zernio post scheduled for that time. Until it's
  sent, a scheduled post can be cancelled with no trace anywhere.

The safety rules are GTA Mode's (see :mod:`autoclip.gta.poster`): one upload
at a time, one active post per clip per platform (``posts.ACTIVE_STATUSES``),
a time that has (nearly) passed is never back-filled, a YouTube upload cut
short is ``interrupted`` and never retried on its own, failures retry after
``RETRY_AFTER_S`` at most ``MAX_ATTEMPTS`` times, quota errors pause the
platform without using up attempts, and an unclear Zernio answer is
``checking``: asked about, never re-sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..gta import zernio
from ..gta.poster import PosterState
from ..live.uploads import tokens as tk
from ..live.uploads.tokens import UploadMeta, UploadResult
from . import clipsource, remote
from . import posts as store
from .core import (
    IG_CHECK_EVERY_S,
    IG_CONFIRM_AFTER_S,
    IG_CONFIRM_EVERY_S,
    IG_CONFIRM_FOR_S,
    IG_INTERRUPTED_NOTE,
    IG_NOT_LISTED_NOTE,
    IG_NOT_SENT_NOTE,
    IG_PAUSE_S,
    INTERRUPTED_NOTE,
    MAX_ATTEMPTS,
    MIN_LEAD_S,
    PLATFORM_LABEL,
    PLATFORMS,
    QUOTA_PAUSE_S,
    RETRY_AFTER_S,
    age_s,
    instagram_caption,
    instagram_readiness,
    is_quota_error,
    iso,
    parse_iso,
    readiness,
    youtube_description,
    youtube_tags,
)
from .posts import Post, StateChanged
from .settings import PublishSettings, instagram_account
from .settings import load as load_settings

log = logging.getLogger(__name__)

TICK_S = 30.0
#: Zernio only schedules: a "post now" reel is scheduled this far ahead.
IG_NOW_DELAY_S = 120.0
#: At most this many posts are sent per platform per pass.
MAX_PER_PASS = 3
MISSED_NOTE = (
    "AutoClip wasn't running when this post had to be sent, so it was not posted late. "
    "Press Retry to post it now."
)
MISSING_FILE_NOTE = "The clip's exported file is missing. Export the clip again, then press Retry."

Uploader = Callable[[Path, UploadMeta], Awaitable[UploadResult]]


def _default_uploader(path: Path, meta: UploadMeta) -> Awaitable[UploadResult]:
    from ..live.uploads.youtube import upload_youtube

    return upload_youtube(path, meta)


def _set(post_id: str, expect: frozenset[str], **fields: Any) -> None:
    """Update a post only while it is still in ``expect`` (else leave it)."""
    with contextlib.suppress(StateChanged, KeyError):
        store.update_post(post_id, expect=expect, **fields)


class CancelError(RuntimeError):
    """A post can't be withdrawn (``studio`` = only YouTube Studio can do it)."""

    def __init__(self, message: str, *, studio: bool = False) -> None:
        super().__init__(message)
        self.studio = studio


class Poster:
    def __init__(
        self,
        *,
        uploader: Uploader | None = None,
        instagram: Any = None,
        clock: Callable[[], datetime] | None = None,
        ready_check: Callable[[], str] = readiness,
        instagram_ready_check: Callable[[], str] = instagram_readiness,
        file_for: Callable[[str], Path | None] = clipsource.file_for,
        settings_loader: Callable[[], PublishSettings] = load_settings,
        api_key: Callable[[], str | None] = zernio.load_api_key,
    ) -> None:
        self._uploader = uploader or _default_uploader
        #: Zernio client (``schedule_reel`` / ``check_post``); tests pass a fake.
        self._ig = instagram or zernio
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ready_checks = {"youtube": ready_check, "instagram": instagram_ready_check}
        self._file_for = file_for
        self._settings = settings_loader
        self._api_key = api_key
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self.states = {p: PosterState(message="Nothing to post yet.") for p in PLATFORMS}

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        await asyncio.to_thread(self.recover)
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="normal-poster")

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
        """Records a crash left unfinished (same rules as GTA Mode)."""
        count = 0
        for post in store.list_posts(statuses=frozenset({"uploading", "checking"})):
            if post.platform == "youtube":
                if post.status == "uploading":
                    store.update_post(post.id, status="interrupted", error=INTERRUPTED_NOTE)
                    count += 1
                continue
            if post.media_url or post.remote_id:
                if post.status == "uploading":
                    store.update_post(post.id, status="checking", error=IG_INTERRUPTED_NOTE)
                    count += 1
            else:
                store.update_post(post.id, status="failed", error=IG_NOT_SENT_NOTE)
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
                log.exception("Posting pass failed.")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=TICK_S)
            self._wakeup.clear()

    # -- which posts are due ---------------------------------------------

    def due_posts(self, platform: str, now: datetime, settings: PublishSettings) -> list[Post]:
        """Queued (and auto-retryable failed) posts whose sending time has come.

        A queued post whose time is closer than ``MIN_LEAD_S`` (or past) is
        marked failed with :data:`MISSED_NOTE`: never posted late on its own.
        """
        lead_s = settings.lead_minutes * 60
        out: list[tuple[datetime, Post]] = []
        for post in store.list_posts(platform=platform, statuses=frozenset({"queued", "failed"})):
            if post.status == "failed" and (
                post.attempts >= MAX_ATTEMPTS or age_s(post.updated_at, now) < RETRY_AFTER_S
            ):
                continue
            when = parse_iso(post.scheduled_at)
            if when is None:
                out.append((parse_iso(post.created_at) or now, post))
                continue
            ahead = (when - now).total_seconds()
            if ahead < MIN_LEAD_S:
                if post.status == "queued":
                    with contextlib.suppress(StateChanged, KeyError):
                        store.update_post(
                            post.id,
                            expect=frozenset({"queued"}),
                            status="failed",
                            error=MISSED_NOTE,
                            attempts=MAX_ATTEMPTS,
                        )
                continue
            if ahead <= lead_s:
                out.append((when, post))
        out.sort(key=lambda pair: pair[0])
        return [post for _, post in out]

    # -- one pass --------------------------------------------------------

    async def run_once(self) -> dict[str, str]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            settings = await asyncio.to_thread(self._settings)
            results: dict[str, str] = {}
            for platform in PLATFORMS:
                results[platform] = await self._run_platform(platform, settings)
            return results

    async def _run_platform(self, platform: str, settings: PublishSettings) -> str:
        state = self.states[platform]
        label = PLATFORM_LABEL[platform]
        now = self._clock()
        state.last_run = now
        if platform == "youtube":
            await asyncio.to_thread(self._youtube_went_public, now)
        if state.paused_until is not None and now < state.paused_until:
            return "paused"
        state.paused_until = None
        due = await asyncio.to_thread(self.due_posts, platform, now, settings)
        waiting = await asyncio.to_thread(
            store.list_posts,
            platform=platform,
            statuses=frozenset({"checking", "scheduled", "queued"}),
        )
        if not due and not waiting:
            state.message = f"Nothing waiting for {label}."
            return "idle"
        problem = await asyncio.to_thread(self._ready_checks[platform])
        if problem:
            state.message = problem
            return "not_ready"
        if platform == "instagram":
            stopped = await self._instagram_followups(now)
            if stopped:
                return stopped
        if not due:
            upcoming = sorted(
                p.scheduled_at for p in waiting if p.status == "queued" and p.scheduled_at
            )
            state.message = (
                f"Next {label} post is sent before {upcoming[0]}."
                if upcoming
                else f"Nothing due for {label}."
            )
            return "waiting"
        outcome = "idle"
        for post in due[:MAX_PER_PASS]:
            if platform == "youtube":
                outcome = await self._upload_youtube(post, settings)
            else:
                outcome = await self._upload_instagram(post, settings)
            if outcome == "paused":
                break
        return outcome

    def _youtube_went_public(self, now: datetime) -> None:
        """A scheduled YouTube video becomes public at its time by itself."""
        for post in store.list_posts(platform="youtube", statuses=frozenset({"scheduled"})):
            when = parse_iso(post.scheduled_at)
            if when is not None and when <= now:
                with contextlib.suppress(StateChanged, KeyError):
                    store.update_post(post.id, expect=frozenset({"scheduled"}), status="posted")

    def _claim(self, post: Post) -> Post | None:
        """Move a due post to ``uploading``; None if it changed meanwhile."""
        try:
            return store.update_post(
                post.id,
                expect=frozenset({"queued", "failed"}),
                status="uploading",
                error=None,
                remote_id="",
                url="",
                media_url="",
            )
        except (StateChanged, KeyError):
            return None

    # -- YouTube ---------------------------------------------------------

    async def _upload_youtube(self, post: Post, settings: PublishSettings) -> str:
        state = self.states["youtube"]
        path = await asyncio.to_thread(self._file_for, post.clip_id)
        if path is None:
            await asyncio.to_thread(
                _set,
                post.id,
                frozenset({"queued", "failed"}),
                status="failed",
                error=MISSING_FILE_NOTE,
                attempts=MAX_ATTEMPTS,
            )
            return "failed"
        claimed = await asyncio.to_thread(self._claim, post)
        if claimed is None:
            return "skipped"
        when = parse_iso(post.scheduled_at)
        title = post.title or "New clip"
        state.uploading_clip_id = post.clip_id
        state.message = f"Uploading “{title}” to YouTube…"
        try:
            meta = UploadMeta(
                title=title,
                description=youtube_description(
                    post.caption, post.hashtags, shorts_tag=settings.shorts_tag
                ),
                tags=youtube_tags(post.hashtags),
                privacy="private" if when else post.privacy,
                category_id=settings.category_id,
                made_for_kids=False,
                publish_at=when,
            )
            result = await self._uploader(path, meta)
        except asyncio.CancelledError:
            await asyncio.to_thread(
                store.update_post, post.id, status="interrupted", error=INTERRUPTED_NOTE
            )
            raise
        except (
            tk.NotLinkedError,
            tk.RightsNotAccepted,
            tk.TokenStateError,
            tk.TokenRefreshError,
        ) as exc:
            # The account is broken, not the clip: its attempts are kept.
            await asyncio.to_thread(
                store.update_post, post.id, status="failed", error=f"YouTube connection: {exc}"
            )
            state.message = f"YouTube connection problem: {exc}"
            return "failed"
        except Exception as exc:  # UploadApiError, OSError, ...
            message = str(exc) or type(exc).__name__
            if is_quota_error(message):
                await asyncio.to_thread(
                    store.update_post, post.id, status="failed", error=f"Quota: {message[:400]}"
                )
                state.paused_until = self._clock() + timedelta(seconds=QUOTA_PAUSE_S)
                state.message = "YouTube's upload limit was hit; posting paused for an hour."
                return "paused"
            await asyncio.to_thread(
                store.update_post,
                post.id,
                status="failed",
                error=message[:500],
                attempts=post.attempts + 1,
            )
            log.warning("YouTube post %s failed: %s", post.id, message)
            state.message = f"Upload of “{title}” failed: {message[:200]}"
            return "failed"
        finally:
            state.uploading_clip_id = None

        status = "scheduled" if when else "posted"
        await asyncio.to_thread(
            store.update_post,
            post.id,
            status=status,
            remote_id=result.media_id,
            url=result.url,
            error=None,
            attempts=post.attempts + 1,
            sent_at=iso(self._clock()),
        )
        state.message = (
            f"Scheduled “{title}” to go public at {post.scheduled_at}."
            if when
            else f"“{title}” is on YouTube."
        )
        return status

    # -- Instagram (Zernio) ----------------------------------------------

    def _pause_instagram(self, seconds: float, message: str) -> str:
        state = self.states["instagram"]
        state.paused_until = self._clock() + timedelta(seconds=max(seconds, 60.0))
        state.message = message
        return "paused"

    async def _upload_instagram(self, post: Post, settings: PublishSettings) -> str:
        state = self.states["instagram"]
        path = await asyncio.to_thread(self._file_for, post.clip_id)
        if path is None:
            await asyncio.to_thread(
                _set,
                post.id,
                frozenset({"queued", "failed"}),
                status="failed",
                error=MISSING_FILE_NOTE,
                attempts=MAX_ATTEMPTS,
            )
            return "failed"
        api_key = await asyncio.to_thread(self._api_key) or ""
        account = await asyncio.to_thread(instagram_account, settings)
        claimed = await asyncio.to_thread(self._claim, post)
        if claimed is None:
            return "skipped"
        post_id, attempts, title = post.id, post.attempts, post.title or "New clip"
        now = self._clock()
        publish_at = parse_iso(post.scheduled_at) or now + timedelta(seconds=IG_NOW_DELAY_S)

        def remember_media(url: str) -> None:
            store.update_post(post_id, media_url=url, sent_at=iso(self._clock()))

        def remember_post(remote_id: str) -> None:
            store.update_post(post_id, remote_id=remote_id)

        state.uploading_clip_id = post.clip_id
        state.message = f"Sending “{title}” to Zernio…"
        try:
            result = await asyncio.to_thread(
                self._ig.schedule_reel,
                api_key,
                account,
                path,
                instagram_caption(post.caption, post.hashtags),
                publish_at,
                on_media_url=remember_media,
                on_post_id=remember_post,
            )
        except asyncio.CancelledError:
            await asyncio.to_thread(
                store.update_post, post_id, status="checking", error=IG_INTERRUPTED_NOTE
            )
            raise
        except Exception as exc:  # a bug or local I/O error; nothing confirmed sent
            message = str(exc) or type(exc).__name__
            fresh = await asyncio.to_thread(store.get_post, post_id)
            sent = fresh is not None and bool(fresh.media_url or fresh.remote_id)
            await asyncio.to_thread(
                store.update_post,
                post_id,
                status="checking" if sent else "failed",
                error=message[:500],
                attempts=attempts + 1,
            )
            state.message = f"Instagram post of “{title}” failed: {message[:200]}"
            return "checking" if sent else "failed"
        finally:
            state.uploading_clip_id = None

        fields: dict[str, Any] = {"error": (result.message or "")[:500] or None}
        if result.post_id:
            fields["remote_id"] = result.post_id
        if result.media_url:
            fields["media_url"] = result.media_url
        kind = result.kind
        if kind in {"scheduled", "posted"}:
            fields["error"] = None
            await asyncio.to_thread(
                store.update_post,
                post_id,
                status=kind,
                url=result.post_url,
                attempts=attempts + 1,
                **fields,
            )
            state.message = (
                f"“{title}” is queued at Zernio for Instagram."
                if kind == "scheduled"
                else f"“{title}” is live on Instagram."
            )
            return kind
        if kind == "unconfirmed":
            await asyncio.to_thread(
                store.update_post, post_id, status="checking", attempts=attempts + 1, **fields
            )
            state.message = f"Zernio's answer for “{title}” was unclear; checking with Zernio."
            return "checking"
        if kind == "blocked":
            await asyncio.to_thread(store.update_post, post_id, status="failed", **fields)
            return self._pause_instagram(
                result.retry_after or IG_PAUSE_S, f"Instagram paused: {result.message[:200]}"
            )
        if kind == "invalid":
            await asyncio.to_thread(
                store.update_post, post_id, status="failed", attempts=MAX_ATTEMPTS, **fields
            )
            state.message = f"Instagram can't take “{title}”: {result.message[:200]}"
            return "failed"
        await asyncio.to_thread(
            store.update_post,
            post_id,
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
        """Ask Zernio about unclear posts and confirm scheduled reels went live."""
        rows = await asyncio.to_thread(
            store.list_posts,
            platform="instagram",
            statuses=frozenset({"checking", "scheduled"}),
        )
        due: list[Post] = []
        for row in rows:
            if row.status == "checking":
                if age_s(row.updated_at, now) >= IG_CHECK_EVERY_S:
                    due.append(row)
                continue
            if not (row.remote_id or row.media_url):
                continue
            goes_live = parse_iso(row.scheduled_at) or (
                (parse_iso(row.sent_at) or now) + timedelta(seconds=IG_NOW_DELAY_S)
            )
            since = (now - goes_live).total_seconds()
            if IG_CONFIRM_AFTER_S <= since <= IG_CONFIRM_FOR_S and (
                age_s(row.updated_at, now) >= IG_CONFIRM_EVERY_S
            ):
                due.append(row)
        if not due:
            return ""
        api_key = await asyncio.to_thread(self._api_key) or ""
        for row in due:
            if not (row.remote_id or row.media_url):
                await asyncio.to_thread(
                    _set, row.id, frozenset({row.status}), status="failed", error=IG_NOT_SENT_NOTE
                )
                continue
            try:
                result = await asyncio.to_thread(
                    self._ig.check_post,
                    api_key,
                    row.remote_id,
                    row.media_url,
                    parse_iso(row.sent_at),
                )
            except Exception as exc:
                log.warning("Zernio check of %s failed: %s", row.id, exc)
                await asyncio.to_thread(_set, row.id, frozenset({row.status}), error=row.error)
                continue
            fields: dict[str, Any] = {}
            if result.post_id:
                fields["remote_id"] = result.post_id
            # ``expect``: a post cancelled meanwhile is left cancelled.
            was = frozenset({row.status})
            if result.kind == "posted":
                await asyncio.to_thread(
                    _set, row.id, was, status="posted", url=result.post_url, error=None, **fields
                )
            elif result.kind == "scheduled":
                await asyncio.to_thread(_set, row.id, was, status="scheduled", error=None, **fields)
            elif result.kind == "failed":
                await asyncio.to_thread(
                    _set,
                    row.id,
                    was,
                    status="failed",
                    error=(result.message or "Instagram did not accept the reel.")[:500],
                    **fields,
                )
            elif result.kind == "blocked":
                return self._pause_instagram(
                    result.retry_after or IG_PAUSE_S, f"Instagram paused: {result.message[:200]}"
                )
            else:
                if row.status == "checking":
                    note = (
                        IG_NOT_LISTED_NOTE
                        if result.not_listed
                        else result.message or "Still waiting for Zernio's answer."
                    )
                else:
                    note = row.error
                await asyncio.to_thread(_set, row.id, was, error=note, **fields)
        return ""

    # -- user actions ----------------------------------------------------

    async def cancel(self, post_id: str, action: str = "private") -> Post:
        """Withdraw an upcoming post.

        Not sent yet -> cancelled here. Scheduled at Zernio -> deleted there.
        Scheduled on YouTube -> unscheduled (``private``) or deleted
        (``delete``), only with the manage scope; else :class:`CancelError`
        with ``studio=True``.
        """
        # No poster lock: every write below names the state it expects, and a
        # post being sent is ``uploading``, which is refused.
        post = await asyncio.to_thread(store.get_post, post_id)
        if post is None:
            raise KeyError(post_id)
        if post.status in {"queued", "failed"}:
            return await asyncio.to_thread(
                store.update_post,
                post.id,
                expect=frozenset({"queued", "failed"}),
                status="canceled",
                error=None,
            )
        if post.status == "scheduled" and post.platform == "instagram":
            api_key = await asyncio.to_thread(self._api_key) or ""
            deleter = getattr(self._ig, "delete_post", None) or remote.delete_zernio_post
            outcome = await asyncio.to_thread(deleter, api_key, post.remote_id)
            if not outcome.ok:
                raise CancelError(outcome.message or "Zernio did not remove the post.")
            return await asyncio.to_thread(
                store.update_post,
                post.id,
                expect=frozenset({"scheduled"}),
                status="canceled",
                error=outcome.message,
            )
        if post.status == "scheduled" and post.platform == "youtube":
            if not await asyncio.to_thread(remote.youtube_can_edit):
                raise CancelError(remote.STUDIO_NOTE, studio=True)
            try:
                if action == "delete":
                    await remote.youtube_delete(post.remote_id)
                    note = "Deleted from YouTube."
                else:
                    await remote.youtube_make_private(post.remote_id)
                    note = "Unscheduled: the video stays private on YouTube."
            except tk.UploadError as exc:
                raise CancelError(str(exc)) from exc
            return await asyncio.to_thread(
                store.update_post,
                post.id,
                expect=frozenset({"scheduled", "posted"}),
                status="canceled",
                error=note,
            )
        if post.status == "posted":
            raise CancelError("This post is already live.", studio=post.platform == "youtube")
        if post.status == "canceled":
            return post
        raise CancelError(
            {
                "uploading": "This post is being sent right now; try again in a moment.",
                "checking": "AutoClip is still checking this reel with Zernio.",
                "interrupted": "This upload was interrupted. Check YouTube Studio first.",
            }.get(post.status, f"A {post.status} post can't be cancelled.")
        )

    def retry(self, post_id: str) -> Post:
        """Put a failed / interrupted / checking / cancelled post back in line.

        For ``interrupted`` (YouTube) and ``checking`` (Instagram) the UI asks
        the user to confirm it is really not on the platform first. A time
        that has (nearly) passed becomes "post now". Refused with
        DoublePostError when the clip got another active post meanwhile.
        """
        post = store.get_post(post_id)
        if post is None:
            raise KeyError(post_id)
        retryable = frozenset({"failed", "interrupted", "checking", "canceled"})
        if post.status not in retryable:
            raise CancelError(f"A {post.status} post can't be retried.")
        when = parse_iso(post.scheduled_at)
        now = self._clock()
        changes: dict[str, Any] = {}
        if when is not None and (when - now).total_seconds() < MIN_LEAD_S + 60:
            changes["scheduled_at"] = None
        updated = store.update_post(
            post.id,
            expect=retryable,
            status="queued",
            attempts=0,
            error=None,
            remote_id="",
            url="",
            media_url="",
            sent_at=None,
            **changes,
        )
        self.notify()
        return updated

    def overview(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "states": {p: s.as_dict() for p, s in self.states.items()},
        }


#: The app-wide poster (started by the app lifespan when the worker runs).
poster = Poster()
