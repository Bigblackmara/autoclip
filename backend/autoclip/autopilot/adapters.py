"""What the Autopilot engine talks to: finding videos, and clipping them.

Both are small protocols so tests pass fakes (no network, no jobs). The real
implementations use the Explore service (yt-dlp, cached) and the normal job
queue (ingest -> job -> the worker clips and exports it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from .models import AutopilotSource

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One video a source lists (flat: some fields may be unknown)."""

    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    views: int | None = None
    duration_s: float | None = None
    #: YYYY-MM-DD when known.
    upload_date: str | None = None
    is_live: bool = False
    is_short: bool = False


@dataclass
class Details:
    """A full look at one video (Explore's Peek)."""

    title: str = ""
    channel: str = ""
    views: int | None = None
    duration_s: float = 0.0
    upload_date: str | None = None
    is_live: bool = False
    #: YouTube's "Most replayed" graph is available.
    has_heatmap: bool = False
    #: The spoken language yt-dlp reports, if any.
    language: str = ""


class Discovery(Protocol):
    def list_source(
        self, source: AutopilotSource, *, max_age_days: int, limit: int
    ) -> list[Candidate]: ...

    def details(self, url: str) -> Details: ...


@dataclass
class ClipInfo:
    clip_id: str
    score: int = 0
    title: str = ""
    #: An exported file exists (the clip can be posted).
    ready: bool = False


@dataclass
class JobOutcome:
    #: queued | running | done | failed | cancelled | missing
    state: str
    error: str = ""
    clips: list[ClipInfo] = field(default_factory=list)


class JobRunner(Protocol):
    def start(self, url: str, *, max_clips: int, title: str = "") -> str:
        """Ingest ``url`` and queue a normal job; returns the job id."""
        ...

    def outcome(self, job_id: str) -> JobOutcome: ...


# --------------------------------------------------------------------------
# Real implementations
# --------------------------------------------------------------------------


class ExploreDiscovery:
    """Lists channels / topic searches and peeks videos via Explore."""

    def list_source(
        self, source: AutopilotSource, *, max_age_days: int, limit: int = 15
    ) -> list[Candidate]:
        from ..explore import service

        if source.kind == "channel":
            items = service.channel(source.value, tab="videos", limit=limit).items
        else:
            uploaded = (
                "today"
                if max_age_days <= 1
                else "week"
                if max_age_days <= 7
                else "month"
                if max_age_days <= 31
                else "year"
            )
            # Most-viewed within the age window: the videos most worth clipping.
            items = service.search(
                source.value, type="videos", sort="views", uploaded=uploaded, limit=limit
            ).items
        out: list[Candidate] = []
        for item in items:
            if item.kind not in ("video", "short", "live"):
                continue
            out.append(
                Candidate(
                    video_id=item.id,
                    url=item.url,
                    title=item.title,
                    channel=item.channel or "",
                    views=item.views,
                    duration_s=item.duration_s,
                    upload_date=item.upload_date,
                    is_live=item.is_live or item.kind == "live",
                    is_short=item.kind == "short",
                )
            )
        return out

    def details(self, url: str) -> Details:
        from ..explore import service, ytdlp

        info = ytdlp.extract(url, flat=False)
        peek = service.peek_from_info(info, url)
        language = info.get("language")
        return Details(
            title=peek.title,
            channel=peek.channel or "",
            views=peek.views,
            duration_s=peek.duration_s,
            upload_date=peek.upload_date,
            is_live=peek.is_live,
            has_heatmap=bool(peek.heatmap),
            language=language.strip().lower() if isinstance(language, str) else "",
        )


class QueueJobRunner:
    """Downloads the video as a source and queues a normal clipping job."""

    def start(self, url: str, *, max_clips: int, title: str = "") -> str:
        from ..config import load as load_settings
        from ..db import store
        from ..db.models import Job, new_id
        from ..jobs.queue import queue
        from ..pipeline import ingest

        settings = load_settings()
        source = ingest.ingest_youtube(url, settings.ingest)
        store.create_source(source)
        merged = settings.model_copy(deep=True)
        merged.clips.max_clips = max(1, min(50, max_clips))
        job = Job(
            id=new_id(),
            source_id=source.id,
            provider=merged.active_provider,
            settings=merged.model_dump(mode="json"),
        )
        store.create_job(job)
        queue.notify()
        log.info("Autopilot queued job %s for %s", job.id, url)
        return job.id

    def outcome(self, job_id: str) -> JobOutcome:
        from ..db import store
        from ..publishing import clipsource

        job = store.get_job(job_id)
        if job is None:
            return JobOutcome("missing", "The clipping job was deleted.")
        if job.status != "done":
            return JobOutcome(job.status, job.error or "")
        clips: list[ClipInfo] = []
        for clip in store.list_clips(job_id):
            if clip.status == "discarded":
                continue
            item = clipsource.postable(clip)
            clips.append(
                ClipInfo(clip_id=clip.id, score=clip.score, title=item.title, ready=item.ready)
            )
        clips.sort(key=lambda c: -c.score)
        return JobOutcome("done", "", clips)
