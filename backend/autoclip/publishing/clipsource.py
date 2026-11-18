"""Normal-mode clips as something postable: file, title, caption, tags."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..db import store
from ..db.models import Clip, Export


@dataclass
class Postable:
    clip_id: str
    job_id: str
    path: Path | None
    title: str
    text: str
    hashtags: list[str] = field(default_factory=list)
    export_id: str = ""
    source_title: str = ""

    @property
    def ready(self) -> bool:
        return self.path is not None and self.path.is_file()


def best_export(exports: list[Export]) -> Export | None:
    """The newest vertical (9:16) export whose file exists, else any existing one."""
    existing = [e for e in exports if e.path and Path(e.path).is_file()]
    for export in existing:
        if export.ratio == "9:16":
            return export
    return existing[0] if existing else None


def postable(clip: Clip) -> Postable:
    export = best_export(store.list_exports(clip.id))
    job = store.get_job(clip.job_id)
    source = store.get_source(job.source_id) if job else None
    return Postable(
        clip_id=clip.id,
        job_id=clip.job_id,
        path=Path(export.path) if export else None,
        title=clip.headline or clip.title,
        text=clip.description or clip.hook or "",
        hashtags=list(clip.hashtags),
        export_id=export.id if export else "",
        source_title=source.title if source else "",
    )


def find(clip_id: str) -> Postable | None:
    clip = store.get_clip(clip_id)
    return postable(clip) if clip is not None else None


def file_for(clip_id: str) -> Path | None:
    """The file a post of this clip uploads (None when nothing exported exists)."""
    item = find(clip_id)
    return item.path if item is not None and item.ready else None


def recent(limit: int = 40, jobs: int = 25) -> list[Postable]:
    """Exported clips of the most recent finished jobs, newest job first, best clip first."""
    out: list[Postable] = []
    for job in store.list_jobs(jobs):
        if job.status != "done":
            continue
        clips = [c for c in store.list_clips(job.id) if c.status != "discarded"]
        clips.sort(key=lambda c: (-c.score, c.rank))
        for clip in clips:
            item = postable(clip)
            if item.ready:
                out.append(item)
            if len(out) >= limit:
                return out
    return out
