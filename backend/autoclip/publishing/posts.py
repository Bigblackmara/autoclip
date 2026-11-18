"""Post records for normal-mode clips (JSON, not SQLite).

One record per clip per platform per attempt at posting it. The
never-double-post rule lives here: a clip can have at most ONE *active*
record per platform (queued, being sent, scheduled, posted, or in an
unclear state). A second "post this clip to YouTube" is refused until the
first one is cancelled or has failed.

Statuses::

    queued      waiting for its time (or for the poster's next pass)
    uploading   being sent right now
    scheduled   the platform holds it and publishes it at ``scheduled_at``
                (YouTube: private + publishAt; Instagram: queued at Zernio)
    posted      live
    failed      not posted; retried by hand (or automatically while
                ``attempts < MAX_ATTEMPTS`` and its time is still ahead)
    interrupted AutoClip stopped mid-upload to YouTube: outcome unknown,
                never retried on its own (check YouTube Studio)
    checking    sent to Zernio but the answer was unclear: AutoClip keeps
                asking Zernio and never re-sends it on its own
    canceled    withdrawn by the user (or by an Autopilot clip rejection)
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from typing import Any

from .core import PLATFORMS, iso
from .jsonstore import JsonFile

STATUSES = frozenset(
    {"queued", "uploading", "scheduled", "posted", "failed", "interrupted", "checking", "canceled"}
)
#: A record in one of these blocks another post of the same clip on the same platform.
ACTIVE_STATUSES = frozenset(
    {"queued", "uploading", "scheduled", "posted", "interrupted", "checking"}
)
#: Still to go out (shown under "Upcoming").
UPCOMING_STATUSES = frozenset({"queued", "uploading", "scheduled", "checking"})
#: How many records are kept (oldest finished ones are dropped first).
MAX_RECORDS = 5000

_FILE = JsonFile("publishing", "posts.json", lambda: {"posts": []})


class DoublePostError(ValueError):
    """The clip already has an active post on that platform."""

    def __init__(self, existing: Post) -> None:
        self.existing = existing
        label = "YouTube" if existing.platform == "youtube" else "Instagram"
        state = {
            "posted": "already posted",
            "scheduled": "already scheduled",
            "queued": "already waiting to be posted",
            "uploading": "being uploaded right now",
            "checking": "being checked with Zernio",
            "interrupted": "interrupted mid-upload (check YouTube Studio)",
        }.get(existing.status, existing.status)
        super().__init__(f"This clip is {state} on {label}; it is never posted twice.")


def _now() -> str:
    return iso(datetime.now(UTC))


@dataclass
class Post:
    id: str
    clip_id: str
    platform: str
    status: str = "queued"
    job_id: str = ""
    #: When it should go public (UTC ``...Z``); None = as soon as possible.
    scheduled_at: str | None = None
    #: YouTube privacy for an immediate post (a scheduled one is always
    #: private until ``scheduled_at``, then public).
    privacy: str = "public"
    title: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    #: YouTube video id / Zernio post id.
    remote_id: str = ""
    url: str = ""
    #: Instagram: the uploaded media's public URL at Zernio (looked up by it).
    media_url: str = ""
    error: str | None = None
    attempts: int = 0
    #: Set when an Autopilot booked this post.
    autopilot_id: str = ""
    slot_key: str = ""
    sent_at: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Post:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_MUTABLE = frozenset(
    {
        "status",
        "scheduled_at",
        "privacy",
        "title",
        "caption",
        "hashtags",
        "remote_id",
        "url",
        "media_url",
        "error",
        "attempts",
        "slot_key",
        "sent_at",
    }
)


def _check(status: str | None = None, platform: str | None = None) -> None:
    if status is not None and status not in STATUSES:
        raise ValueError(f"Unknown post status {status!r}")
    if platform is not None and platform not in PLATFORMS:
        raise ValueError(f"Unknown platform {platform!r}")


def _rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("posts")
    if not isinstance(rows, list):
        rows = data["posts"] = []
    return rows


def list_posts(
    *,
    statuses: frozenset[str] | None = None,
    platform: str | None = None,
    clip_id: str | None = None,
    autopilot_id: str | None = None,
) -> list[Post]:
    """Matching records, newest first."""
    out: list[Post] = []
    for row in _rows(_FILE.read()):
        if not isinstance(row, dict):
            continue
        post = Post.from_dict(row)
        if statuses is not None and post.status not in statuses:
            continue
        if platform is not None and post.platform != platform:
            continue
        if clip_id is not None and post.clip_id != clip_id:
            continue
        if autopilot_id is not None and post.autopilot_id != autopilot_id:
            continue
        out.append(post)
    out.sort(key=lambda p: (p.scheduled_at or p.created_at, p.created_at), reverse=True)
    return out


def get_post(post_id: str) -> Post | None:
    for row in _rows(_FILE.read()):
        if isinstance(row, dict) and row.get("id") == post_id:
            return Post.from_dict(row)
    return None


def active_post(clip_id: str, platform: str) -> Post | None:
    for post in list_posts(clip_id=clip_id, platform=platform, statuses=ACTIVE_STATUSES):
        return post
    return None


def create_post(
    clip_id: str,
    platform: str,
    *,
    job_id: str = "",
    scheduled_at: str | None = None,
    privacy: str = "public",
    title: str = "",
    caption: str = "",
    hashtags: list[str] | None = None,
    autopilot_id: str = "",
    slot_key: str = "",
) -> Post:
    """Add a queued record; :class:`DoublePostError` if the clip is already active there.

    The check and the insert happen under one lock, so two simultaneous
    requests can't both get through.
    """
    _check(platform=platform)
    if privacy not in ("public", "unlisted", "private"):
        raise ValueError("privacy must be public, unlisted or private")
    post = Post(
        id=uuid.uuid4().hex[:16],
        clip_id=clip_id,
        platform=platform,
        job_id=job_id,
        scheduled_at=scheduled_at,
        privacy=privacy,
        title=title,
        caption=caption,
        hashtags=list(hashtags or []),
        autopilot_id=autopilot_id,
        slot_key=slot_key,
    )
    with _FILE.edit() as data:
        rows = _rows(data)
        for row in rows:
            if (
                isinstance(row, dict)
                and row.get("clip_id") == clip_id
                and row.get("platform") == platform
                and row.get("status") in ACTIVE_STATUSES
            ):
                raise DoublePostError(Post.from_dict(row))
        rows.append(post.as_dict())
        _trim(rows)
    return post


def _trim(rows: list[Any]) -> None:
    if len(rows) <= MAX_RECORDS:
        return
    finished = [
        i
        for i, r in enumerate(rows)
        if isinstance(r, dict) and r.get("status") in {"posted", "failed", "canceled"}
    ]
    drop = set(finished[: len(rows) - MAX_RECORDS])
    rows[:] = [r for i, r in enumerate(rows) if i not in drop]


def update_post(post_id: str, *, expect: frozenset[str] | None = None, **changes: Any) -> Post:
    """Change fields of a record; ``expect`` = the statuses it must be in now.

    Raises KeyError when missing, ``StateChanged`` when ``expect`` doesn't
    hold (someone else moved it first), and DoublePostError when the change
    would make a second active record for the clip on that platform.
    """
    unknown = set(changes) - _MUTABLE
    if unknown:
        raise ValueError(f"Cannot change {', '.join(sorted(unknown))}")
    _check(status=changes.get("status"))
    with _FILE.edit() as data:
        rows = _rows(data)
        for row in rows:
            if not isinstance(row, dict) or row.get("id") != post_id:
                continue
            if expect is not None and row.get("status") not in expect:
                raise StateChanged(Post.from_dict(row))
            new_status = changes.get("status")
            if (
                new_status in ACTIVE_STATUSES
                and row.get("status") not in ACTIVE_STATUSES
                and (
                    clash := next(
                        (
                            r
                            for r in rows
                            if isinstance(r, dict)
                            and r.get("id") != post_id
                            and r.get("clip_id") == row.get("clip_id")
                            and r.get("platform") == row.get("platform")
                            and r.get("status") in ACTIVE_STATUSES
                        ),
                        None,
                    )
                )
            ):
                raise DoublePostError(Post.from_dict(clash))
            row.update(changes)
            row["updated_at"] = _now()
            return Post.from_dict(row)
    raise KeyError(post_id)


class StateChanged(RuntimeError):
    """The record is no longer in the state the caller expected."""

    def __init__(self, post: Post) -> None:
        self.post = post
        super().__init__(f"Post {post.id} is {post.status} now.")


def delete_all() -> None:
    """Testing helper: forget every record."""
    with _FILE.edit() as data:
        data["posts"] = []
