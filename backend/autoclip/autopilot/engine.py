"""The Autopilot engine: find new videos, clip the best, book the clips.

Every ``TICK_S`` (and right after a change in the UI), for each autopilot:

1. **harvest** - a clipping job finished: take its best exported clips
   (``clips_per_video``) into "review" (approval on) or "ready".
2. **tidy** - booked clips follow their posts: all cancelled -> rejected (the
   slot is free again), posted -> posted, all failed -> failed.
3. **scan** (hourly) - list each source's newest videos through Explore.
   Every video is remembered, so none is looked at twice. Obvious misses
   (live, a Short, too short/long, too old, already clipped by another
   autopilot posting to the same platform) are skipped at once; the rest
   wait.
4. **check** - waiting videos are peeked (at most hourly each): too few views
   -> keep waiting (skipped once ``max_age_days`` passes); YouTube's "Most
   replayed" graph present -> ready; no graph yet -> wait, and after
   ``heatmap_wait_hours`` clip it anyway (the AI picks from audio and
   comments).
5. **clip** - when fewer than two days of clips are in stock and nothing is
   clipping, the best ready video (rewatch graph first, then views) is
   ingested and a normal job is queued.
6. **book** - ready clips, best first, go into the next free daily slots
   (``PLAN_DAYS`` ahead): one normal post per platform at the slot time. The
   poster sends them ``lead_minutes`` early; until then they can be
   cancelled with no trace anywhere.

A paused autopilot only harvests and tidies: it doesn't scan, clip or book,
and posts it already booked still go out (cancel them on the Schedule page).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from ..publishing import posts as post_store
from ..publishing import service as publishing
from ..publishing.core import MAX_ATTEMPTS, free_slots, iso, parse_iso, slot_key
from ..publishing.posts import DoublePostError
from . import store
from .adapters import Candidate, Details, Discovery, ExploreDiscovery, JobRunner, QueueJobRunner
from .models import Autopilot, AutopilotSource

log = logging.getLogger(__name__)

TICK_S = 600.0
SCAN_EVERY_S = 3600.0
#: A waiting video is peeked again at most this often.
RECHECK_S = 3600.0
#: Videos peeked per autopilot per run (each is a full yt-dlp lookup).
MAX_CHECKS_PER_RUN = 8
#: Videos listed per source per scan.
LIST_LIMIT = 15
#: Days of slots clips are booked into.
PLAN_DAYS = 3
#: A slot is only booked this far ahead (the post must be creatable).
BOOK_MIN_AHEAD_S = publishing.MIN_SCHEDULE_AHEAD_S + 60
SHORT_MAX_S = 60

_PENDING_POST = frozenset({"queued", "uploading", "scheduled", "checking", "interrupted"})


def _date_moment(date: str | None) -> datetime | None:
    if not date:
        return None
    try:
        return datetime.strptime(date[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def _minutes(seconds: float) -> str:
    return f"{round(seconds / 60)} min"


# --------------------------------------------------------------------------
# Decisions (pure; unit-tested directly)
# --------------------------------------------------------------------------


def judge_listing(c: Candidate, rule: Autopilot, now: datetime) -> tuple[str, str]:
    """(status, reason) for a newly listed video: ``skipped`` or ``waiting``."""
    f = rule.filters
    if c.is_live:
        return "skipped", "A live stream"
    if c.is_short or (c.duration_s is not None and c.duration_s <= SHORT_MAX_S):
        return "skipped", "A Short"
    if c.duration_s is not None and c.duration_s < f.min_minutes * 60:
        return "skipped", f"{_minutes(c.duration_s)} long - shorter than {f.min_minutes:g} min"
    if c.duration_s is not None and c.duration_s > f.max_minutes * 60:
        return "skipped", f"{_minutes(c.duration_s)} long - longer than {f.max_minutes:g} min"
    uploaded = _date_moment(c.upload_date)
    if uploaded is not None and (now - uploaded).days > f.max_age_days:
        return "skipped", f"Older than {f.max_age_days} days"
    if c.views is not None and c.views < f.min_views and uploaded is not None:
        # still young: may get there, so it waits (re-checked hourly)
        return "waiting", f"{c.views:,} views so far - waiting for {f.min_views:,}"
    return "waiting", "New - checking views and the rewatch graph"


def judge_details(
    video: dict[str, Any], d: Details, rule: Autopilot, now: datetime
) -> tuple[str, str]:
    """(status, reason) after peeking a waiting video: skipped, waiting or ready."""
    f = rule.filters
    if d.is_live:
        return "skipped", "A live stream"
    if d.duration_s and d.duration_s <= SHORT_MAX_S:
        return "skipped", "A Short"
    if d.duration_s and d.duration_s < f.min_minutes * 60:
        return "skipped", f"{_minutes(d.duration_s)} long - shorter than {f.min_minutes:g} min"
    if d.duration_s and d.duration_s > f.max_minutes * 60:
        return "skipped", f"{_minutes(d.duration_s)} long - longer than {f.max_minutes:g} min"
    uploaded = _date_moment(d.upload_date or video.get("upload_date"))
    age_days = (now - uploaded).days if uploaded is not None else None
    if age_days is not None and age_days > f.max_age_days:
        return "skipped", f"Older than {f.max_age_days} days"
    wanted = str(video.get("language") or "")
    if wanted and d.language and not d.language.startswith(wanted):
        return "skipped", f"Spoken language is {d.language}, not {wanted}"
    views = d.views if d.views is not None else video.get("views")
    if (views or 0) < f.min_views:
        return "waiting", f"{(views or 0):,} views so far - waiting for {f.min_views:,}"
    if d.has_heatmap:
        return "ready", "Most replayed graph found"
    first_seen = parse_iso(video.get("first_seen")) or now
    waited_h = (now - first_seen).total_seconds() / 3600
    if waited_h >= rule.heatmap_wait_hours:
        return (
            "ready",
            f"No Most replayed graph after {rule.heatmap_wait_hours:g} h - "
            "the AI picks from audio and comments",
        )
    return (
        "waiting",
        f"Waiting for YouTube's Most replayed graph ({waited_h:.0f} of "
        f"{rule.heatmap_wait_hours:g} h)",
    )


def stock(state: dict[str, Any]) -> int:
    """Clips in line or booked (not yet out)."""
    return sum(
        1 for c in state.get("clips", []) if c.get("status") in {"review", "ready", "booked"}
    )


def taken_slots(state: dict[str, Any]) -> set[str]:
    return {
        c["slot_key"]
        for c in state.get("clips", [])
        if c.get("slot_key") and c.get("status") in {"booked", "posted"}
    }


def best_ready_video(state: dict[str, Any]) -> dict[str, Any] | None:
    ready = [v for v in state.get("videos", {}).values() if v.get("status") == "ready"]
    ready.sort(key=lambda v: (not v.get("has_heatmap"), -(v.get("views") or 0)))
    return ready[0] if ready else None


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class AutopilotEngine:
    def __init__(
        self,
        *,
        discovery: Discovery | None = None,
        runner: JobRunner | None = None,
        clock: Callable[[], datetime] | None = None,
        on_booked: Callable[[], None] | None = None,
    ) -> None:
        self.discovery: Discovery = discovery or ExploreDiscovery()
        self.runner: JobRunner = runner or QueueJobRunner()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._on_booked = on_booked or _wake_poster
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self._force_scan: set[str] = set()

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="autopilot")

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

    def notify(self, scan: str | None = None) -> None:
        """Run a pass soon (``scan`` = also scan that autopilot now)."""
        if scan:
            self._force_scan.add(scan)
        if self._loop is None or self._wakeup is None:
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._wakeup.set)

    async def _run(self) -> None:
        assert self._wakeup is not None
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Autopilot pass failed.")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wakeup.wait(), timeout=TICK_S)
            self._wakeup.clear()

    # -- a pass ----------------------------------------------------------

    async def run_once(self, *, force_scan: set[str] | None = None) -> dict[str, dict[str, Any]]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            forced = set(force_scan or ()) | self._force_scan
            self._force_scan.clear()
            results: dict[str, dict[str, Any]] = {}
            for rule in await asyncio.to_thread(store.list_autopilots):
                try:
                    results[rule.id] = await self.run_rule(rule, force_scan=rule.id in forced)
                except Exception as exc:
                    log.exception("Autopilot %s failed.", rule.name)
                    with store.edit_state(rule.id) as st:
                        st["last_error"] = str(exc)[:500]
                    results[rule.id] = {"error": str(exc)}
            return results

    async def run_rule(self, rule: Autopilot, *, force_scan: bool = False) -> dict[str, Any]:
        now = self._clock()
        run: dict[str, Any] = {
            "at": iso(now),
            "scanned": False,
            "found": 0,
            "checked": 0,
            "started": None,
            "harvested": 0,
            "booked": 0,
            "errors": [],
        }
        await self._harvest(rule, run)
        await asyncio.to_thread(self._tidy, rule)
        if rule.enabled:
            state = await asyncio.to_thread(store.get_state, rule.id)
            last_scan = parse_iso(state.get("last_scan_at"))
            if force_scan or last_scan is None or (now - last_scan).total_seconds() >= SCAN_EVERY_S:
                await self._scan(rule, now, run)
                await self._check_waiting(rule, now, run)
            await self._clip(rule, run)
            await self._book(rule, now, run)
        message = _summary(rule, run)
        with store.edit_state(rule.id) as st:
            st["last_run_at"] = iso(now)
            st["message"] = message
            st["last_error"] = " · ".join(run["errors"])[:500] or None
            if any(run[k] for k in ("scanned", "started", "harvested", "booked")) or run["errors"]:
                st["runs"].append({**run, "message": message})
        return run

    # -- 1. harvest ------------------------------------------------------

    async def _harvest(self, rule: Autopilot, run: dict[str, Any]) -> None:
        state = await asyncio.to_thread(store.get_state, rule.id)
        clipping = [v for v in state["videos"].values() if v.get("status") == "clipping"]
        if not clipping:
            return
        outcomes = {}
        for video in clipping:
            outcomes[video["id"]] = await asyncio.to_thread(
                self.runner.outcome, video.get("job_id", "")
            )
        with store.edit_state(rule.id) as st:
            known = {c["clip_id"] for c in st["clips"]}
            for video_id, outcome in outcomes.items():
                video = st["videos"].get(video_id)
                if video is None or video.get("status") != "clipping":
                    continue
                if outcome.state in {"queued", "running"}:
                    continue
                if outcome.state != "done":
                    video.update(
                        status="failed",
                        reason=(outcome.error or f"Clipping {outcome.state}")[:300],
                    )
                    continue
                best = sorted(
                    (c for c in outcome.clips if c.ready and c.clip_id not in known),
                    key=lambda c: -c.score,
                )
                best = best[: rule.clips_per_video]
                for clip in best:
                    st["clips"].append(
                        {
                            "clip_id": clip.clip_id,
                            "job_id": video.get("job_id", ""),
                            "video_id": video_id,
                            "title": clip.title,
                            "score": clip.score,
                            "status": "review" if rule.approve_before_posting else "ready",
                            "slot_key": "",
                            "post_ids": [],
                            "reason": "",
                            "created_at": iso(self._clock()),
                        }
                    )
                run["harvested"] += len(best)
                video.update(
                    status="done",
                    reason=(
                        f"{len(best)} clip{'s' if len(best) != 1 else ''} made"
                        if best
                        else "No clip-worthy moments were found"
                    ),
                )

    # -- 2. tidy ---------------------------------------------------------

    def _tidy(self, rule: Autopilot) -> None:
        state = store.get_state(rule.id)
        booked = [c for c in state["clips"] if c.get("status") == "booked"]
        if not booked:
            return
        verdicts: dict[str, tuple[str, str]] = {}
        for clip in booked:
            statuses = []
            for pid in clip.get("post_ids", []):
                post = post_store.get_post(pid)
                if post is None:
                    statuses.append("canceled")
                elif post.status == "failed" and post.attempts < MAX_ATTEMPTS:
                    statuses.append("queued")  # the poster retries it by itself
                else:
                    statuses.append(post.status)
            if not statuses or all(s == "canceled" for s in statuses):
                verdicts[clip["clip_id"]] = ("rejected", "Cancelled on the Schedule page")
            elif any(s in _PENDING_POST for s in statuses):
                continue
            elif any(s == "posted" for s in statuses):
                verdicts[clip["clip_id"]] = ("posted", "")
            else:
                verdicts[clip["clip_id"]] = ("failed", "Posting failed (see the Schedule page)")
        if not verdicts:
            return
        with store.edit_state(rule.id) as st:
            for clip in st["clips"]:
                verdict = verdicts.get(clip["clip_id"])
                if verdict and clip.get("status") == "booked":
                    clip["status"], clip["reason"] = verdict

    # -- 3. scan ---------------------------------------------------------

    async def _scan(self, rule: Autopilot, now: datetime, run: dict[str, Any]) -> None:
        listings: list[tuple[AutopilotSource, list[Candidate]]] = []
        for source in rule.sources:
            try:
                found = await asyncio.to_thread(
                    self.discovery.list_source,
                    source,
                    max_age_days=rule.filters.max_age_days,
                    limit=LIST_LIMIT,
                )
            except Exception as exc:
                run["errors"].append(f"{source.label}: {str(exc)[:160]}")
                continue
            listings.append((source, found))
        used_elsewhere = await asyncio.to_thread(_clipped_by_others, rule)
        run["scanned"] = True
        with store.edit_state(rule.id) as st:
            st["last_scan_at"] = iso(now)
            videos = st["videos"]
            for source, found in listings:
                for c in found:
                    if not c.video_id or c.video_id in videos:
                        continue  # every video is looked at once
                    status, reason = judge_listing(c, rule, now)
                    other = used_elsewhere.get(c.video_id)
                    if status == "waiting" and other:
                        status, reason = "skipped", f"Already clipped by your autopilot “{other}”"
                    videos[c.video_id] = {
                        "id": c.video_id,
                        "url": c.url,
                        "title": c.title,
                        "channel": c.channel,
                        "views": c.views,
                        "duration_s": c.duration_s,
                        "upload_date": c.upload_date,
                        "language": source.wanted_language,
                        "source": source.label,
                        "status": status,
                        "reason": reason,
                        "has_heatmap": False,
                        "first_seen": iso(now),
                        "last_checked": None,
                        "job_id": "",
                    }
                    run["found"] += 1

    # -- 4. check waiting videos ------------------------------------------

    async def _check_waiting(self, rule: Autopilot, now: datetime, run: dict[str, Any]) -> None:
        state = await asyncio.to_thread(store.get_state, rule.id)
        due = [
            v
            for v in state["videos"].values()
            if v.get("status") == "waiting"
            and (
                v.get("last_checked") is None
                or (now - (parse_iso(v.get("last_checked")) or now)).total_seconds() >= RECHECK_S
            )
        ]
        due.sort(key=lambda v: -(v.get("views") or 0))
        results: dict[str, tuple[str, str, Details | None]] = {}
        for video in due[:MAX_CHECKS_PER_RUN]:
            try:
                details = await asyncio.to_thread(self.discovery.details, video["url"])
            except Exception as exc:
                results[video["id"]] = (
                    "waiting",
                    f"Couldn't open the video: {str(exc)[:160]}",
                    None,
                )
                continue
            status, reason = judge_details(video, details, rule, now)
            results[video["id"]] = (status, reason, details)
        run["checked"] += len(results)
        with store.edit_state(rule.id) as st:
            for video_id, (status, reason, details) in results.items():
                video = st["videos"].get(video_id)
                if video is None or video.get("status") != "waiting":
                    continue
                video.update(status=status, reason=reason, last_checked=iso(now))
                if details is not None:
                    video.update(
                        title=details.title or video.get("title"),
                        views=details.views if details.views is not None else video.get("views"),
                        duration_s=details.duration_s or video.get("duration_s"),
                        upload_date=details.upload_date or video.get("upload_date"),
                        has_heatmap=details.has_heatmap,
                    )

    # -- 5. clip ---------------------------------------------------------

    async def _clip(self, rule: Autopilot, run: dict[str, Any]) -> None:
        state = await asyncio.to_thread(store.get_state, rule.id)
        if any(v.get("status") == "clipping" for v in state["videos"].values()):
            return  # one video at a time per autopilot
        if stock(state) >= max(2, len(rule.slots) * 2):
            return  # about two days of clips in stock already
        video = best_ready_video(state)
        if video is None:
            return
        try:
            job_id = await asyncio.to_thread(
                self.runner.start,
                video["url"],
                max_clips=min(10, rule.clips_per_video + 2),
                title=video.get("title", ""),
            )
        except Exception as exc:
            message = str(exc)[:300] or type(exc).__name__
            with store.edit_state(rule.id) as st:
                if video["id"] in st["videos"]:
                    st["videos"][video["id"]].update(
                        status="failed", reason=f"Couldn't download: {message}"
                    )
            run["errors"].append(f"{video.get('title') or video['url']}: {message[:160]}")
            return
        with store.edit_state(rule.id) as st:
            if video["id"] in st["videos"]:
                st["videos"][video["id"]].update(
                    status="clipping",
                    job_id=job_id,
                    reason=(
                        "Clipping (Most replayed graph found)"
                        if video.get("has_heatmap")
                        else "Clipping (no Most replayed graph - the AI picks)"
                    ),
                )
        run["started"] = video.get("title") or video["url"]

    # -- 6. book ---------------------------------------------------------

    async def _book(self, rule: Autopilot, now: datetime, run: dict[str, Any]) -> None:
        state = await asyncio.to_thread(store.get_state, rule.id)
        ready = [c for c in state["clips"] if c.get("status") == "ready"]
        if not ready:
            return
        ready.sort(key=lambda c: (-(c.get("score") or 0), c.get("created_at") or ""))
        slots = free_slots(
            rule.slots,
            now,
            taken_slots(state),
            days=PLAN_DAYS,
            tz=rule.tzinfo(),
            min_lead_s=BOOK_MIN_AHEAD_S,
        )
        changes: dict[str, dict[str, Any]] = {}
        for clip, slot in zip(ready, slots, strict=False):
            key = slot_key(slot)
            platforms = [
                p
                for p in rule.platforms
                if await asyncio.to_thread(post_store.active_post, clip["clip_id"], p) is None
            ]
            if not platforms:
                changes[clip["clip_id"]] = {
                    "status": "posted",
                    "reason": "Already posted or booked by hand",
                }
                continue
            made = []
            problem = ""
            for platform in platforms:
                try:
                    made += await asyncio.to_thread(
                        publishing.create_posts,
                        clip["clip_id"],
                        [platform],
                        scheduled_at=slot,
                        autopilot_id=rule.id,
                        slot_key=key,
                        now=now,
                    )
                except DoublePostError:
                    continue  # posted by hand meanwhile: not twice
                except (publishing.PostRequestError, LookupError) as exc:
                    problem = str(exc) or "The clip is gone"
            if made:
                changes[clip["clip_id"]] = {
                    "status": "booked",
                    "slot_key": key,
                    "post_ids": [p.id for p in made],
                    "reason": "",
                }
                run["booked"] += 1
            else:
                changes[clip["clip_id"]] = {
                    "status": "failed",
                    "reason": problem or "Already posted by hand",
                }
        if not changes:
            return
        with store.edit_state(rule.id) as st:
            for clip in st["clips"]:
                change = changes.get(clip["clip_id"])
                if change and clip.get("status") == "ready":
                    clip.update(change)
        if run["booked"]:
            self._on_booked()

    # -- user actions ----------------------------------------------------

    def set_clip(self, rule: Autopilot, clip_id: str, action: str) -> dict[str, Any]:
        """``approve`` a clip in review, or ``reject`` one (unbooking it).

        Rejecting a booked clip cancels its posts that haven't been sent;
        once one is at the platform it must be cancelled on the Schedule page.
        """
        state = store.get_state(rule.id)
        clip = next((c for c in state["clips"] if c["clip_id"] == clip_id), None)
        if clip is None:
            raise KeyError(clip_id)
        if action == "approve":
            if clip["status"] != "review":
                raise ValueError("Only clips waiting for review can be approved.")
            new = {"status": "ready", "reason": ""}
        elif action == "reject":
            if clip["status"] not in {"review", "ready", "booked"}:
                raise ValueError("This clip is no longer in line.")
            if clip["status"] == "booked":
                for pid in clip.get("post_ids", []):
                    post = post_store.get_post(pid)
                    if post is None or post.status in {"canceled", "failed"}:
                        continue
                    if post.status != "queued":
                        raise ValueError(
                            "This clip was already sent to the platform; cancel it on the "
                            "Schedule page."
                        )
                for pid in clip.get("post_ids", []):
                    with contextlib.suppress(KeyError, post_store.StateChanged):
                        post_store.update_post(
                            pid,
                            expect=frozenset({"queued", "failed"}),
                            status="canceled",
                            error="Removed from the Autopilot",
                        )
            new = {"status": "rejected", "reason": "Removed by you"}
        else:
            raise ValueError(f"Unknown action {action!r}")
        with store.edit_state(rule.id) as st:
            for c in st["clips"]:
                if c["clip_id"] == clip_id:
                    c.update(new)
                    clip = c
        self.notify()
        return clip

    def cancel_unsent(self, autopilot_id: str) -> int:
        """Cancel every post of this autopilot that hasn't been sent yet."""
        count = 0
        for post in post_store.list_posts(
            autopilot_id=autopilot_id, statuses=frozenset({"queued"})
        ):
            with contextlib.suppress(KeyError, post_store.StateChanged):
                post_store.update_post(
                    post.id,
                    expect=frozenset({"queued"}),
                    status="canceled",
                    error="The autopilot was deleted",
                )
                count += 1
        return count


def _clipped_by_others(rule: Autopilot) -> dict[str, str]:
    """Video ids other autopilots posting to a shared platform already clip(ped)."""
    out: dict[str, str] = {}
    mine = set(rule.platforms)
    for other in store.list_autopilots():
        if other.id == rule.id or not mine & set(other.platforms):
            continue
        for video_id, video in store.get_state(other.id)["videos"].items():
            if video.get("status") in {"clipping", "done", "ready"}:
                out.setdefault(video_id, other.name)
    return out


def _summary(rule: Autopilot, run: dict[str, Any]) -> str:
    parts: list[str] = []
    if not rule.enabled:
        parts.append("Paused")
    if run["scanned"]:
        parts.append(f"found {run['found']} new video{'s' if run['found'] != 1 else ''}")
    if run["checked"]:
        parts.append(f"checked {run['checked']}")
    if run["started"]:
        parts.append(f"clipping “{run['started']}”")
    if run["harvested"]:
        parts.append(f"{run['harvested']} new clip{'s' if run['harvested'] != 1 else ''}")
    if run["booked"]:
        parts.append(f"booked {run['booked']}")
    if run["errors"]:
        parts.append(f"{len(run['errors'])} problem{'s' if len(run['errors']) != 1 else ''}")
    text = ", ".join(parts) or "Nothing new"
    return text[0].upper() + text[1:]


def _wake_poster() -> None:
    from ..publishing.poster import poster

    poster.notify()


def upcoming_slots(rule: Autopilot, now: datetime, days: int = PLAN_DAYS) -> list[dict[str, Any]]:
    """The next days of slots with the clip booked into each (for the detail page)."""
    state = store.get_state(rule.id)
    by_slot = {
        c["slot_key"]: c
        for c in state["clips"]
        if c.get("slot_key") and c.get("status") in {"booked", "posted"}
    }
    out: list[dict[str, Any]] = []
    from ..publishing.core import slot_datetimes

    for moment in slot_datetimes(rule.slots, now - timedelta(hours=1), days=days, tz=rule.tzinfo()):
        key = slot_key(moment)
        clip = by_slot.get(key)
        out.append(
            {
                "slot_at": key,
                "clip_id": clip["clip_id"] if clip else None,
                "title": clip.get("title") if clip else None,
                "status": clip["status"] if clip else "open",
            }
        )
    return out


#: The app-wide engine (started by the app lifespan when the worker runs).
engine = AutopilotEngine()
