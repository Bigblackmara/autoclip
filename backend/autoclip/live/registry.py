"""Persisted registry of Live Watch streams — the user's watch list.

The watch list mutates while the app runs: the manager adds, removes, and
toggles streams as the user clicks, and every change must survive a restart.
It deliberately does NOT live in ``config.json``: routing runtime watch-list
writes through :func:`autoclip.config.save` would round-trip the entire
:class:`~autoclip.config.Settings` object (provider config and secret-fallback
state included) on every toggle, churning unrelated settings and racing the
settings UI writing the same file. A dedicated file keeps the hot path small
and crash-safe.

Storage (this location and shape are the contract the manager and API consume
through this module — they must not read the file directly)::

    <root>/streams.json
    {"version": 1,
     "streams": [{"id": "...", "url": "...", "label": "", "enabled": true,
                  "created_at": "2026-11-13T09:30:00+00:00"}, ...]}

Durability mirrors :func:`autoclip.config.save`: write a sibling
``streams.json.tmp`` then :func:`os.replace`, so an interrupted write or full
disk can never leave a truncated watch list behind — the previous list stays
byte-identical and the sidecar is removed. A module-level lock serialises the
read-modify-write cycle of every call in-process (the manager invokes these
sync functions from ``asyncio.to_thread`` workers). Multi-process concurrency
is unsupported by design: one server process owns the file, and the CLI is a
single interactive user.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .. import paths
from .models import new_id, utcnow
from .settings import LiveStreamConfig

log = logging.getLogger(__name__)

STREAMS_FILE = "streams.json"

#: Bumped when the on-disk shape changes. Readers tolerate unknown keys, so
#: a file written by a newer AutoClip still loads (with new fields dropped
#: harmlessly on the next rewrite).
REGISTRY_VERSION = 1

#: Serialises every registry call in-process. Reads take the lock too: each
#: mutating call is a read-modify-write of the whole file, and locking both
#: sides keeps those cycles linearisable against each other.
_LOCK = threading.Lock()


def streams_path() -> Path:
    """Return the registry file location: ``<AUTOCLIP_HOME>/streams.json``."""
    return paths.root() / STREAMS_FILE


def list_streams() -> list[LiveStreamConfig]:
    """Return every registered stream, in insertion order."""
    with _LOCK:
        return _read()


def get(stream_id: str) -> LiveStreamConfig | None:
    """Return one stream by id, or None when it is not registered."""
    with _LOCK:
        for stream in _read():
            if stream.id == stream_id:
                return stream
    return None


def upsert(stream: LiveStreamConfig) -> LiveStreamConfig:
    """Insert or replace ``stream``, keyed by ``id``, and return what was stored.

    An empty ``id`` is assigned a fresh one (so the manager can register a
    stream without minting ids itself). An empty ``created_at`` keeps the
    existing record's stamp — re-upserting a stream must not reset its age —
    or stamps now for a new record.

    Raises ValueError for an empty url: a stream that can never be watched
    must not be persisted. Write failures (OSError) propagate to the caller
    while leaving the previous registry byte-identical.
    """
    if not stream.url.strip():
        raise ValueError("a watched stream needs a non-empty url")
    with _LOCK:
        streams = _read()
        stored = _prepare(stream, streams)
        for index, existing in enumerate(streams):
            if existing.id == stored.id:
                streams[index] = stored
                break
        else:
            streams.append(stored)
        _write(streams)
        return stored


def delete(stream_id: str) -> bool:
    """Remove a stream. Returns False — and writes nothing — for unknown ids."""
    with _LOCK:
        streams = _read()
        kept = [stream for stream in streams if stream.id != stream_id]
        if len(kept) == len(streams):
            return False
        _write(kept)
        return True


def set_enabled(stream_id: str, enabled: bool) -> bool:
    """Toggle a stream's enabled flag (persistence-level pause/resume).

    Returns True when the stream is registered — including when the flag
    already matched, in which case nothing is written. Returns False for
    unknown ids, again writing nothing.
    """
    with _LOCK:
        streams = _read()
        found = False
        changed = False
        for index, stream in enumerate(streams):
            if stream.id != stream_id:
                continue
            found = True
            if stream.enabled != enabled:
                streams[index] = stream.model_copy(update={"enabled": enabled})
                changed = True
        if not found:
            return False
        if changed:
            _write(streams)
        return True


def _prepare(stream: LiveStreamConfig, existing: Sequence[LiveStreamConfig]) -> LiveStreamConfig:
    """Fill in id/created_at without mutating the caller's object."""
    updates: dict[str, str] = {}
    if not stream.id:
        updates["id"] = new_id()
    if not stream.created_at:
        stream_id = updates.get("id", stream.id)
        prior = next((s for s in existing if s.id == stream_id), None)
        updates["created_at"] = prior.created_at if prior else utcnow()
    return stream.model_copy(update=updates)


def _read() -> list[LiveStreamConfig]:
    """Load the registry; safe on a missing, corrupt, or odd-shaped file."""
    path = streams_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        # Same contract as config.load: a bad file must not take startup down.
        log.warning("Could not read stream registry %s (%s); treating as empty.", path, exc)
        return []

    entries = raw.get("streams") if isinstance(raw, dict) else raw  # bare-list tolerance
    if not isinstance(entries, list):
        log.warning("Stream registry %s has no streams array; treating as empty.", path)
        return []

    streams: list[LiveStreamConfig] = []
    for index, entry in enumerate(entries):
        try:
            streams.append(LiveStreamConfig.model_validate(entry))
        except ValidationError as exc:
            # One bad hand edit must not cost the user their whole watch list.
            log.warning("Skipping bad stream entry #%d in %s: %s", index, path, exc)
    return streams


def _write(streams: list[LiveStreamConfig]) -> None:
    """Persist the registry atomically: tmp file, then :func:`os.replace`."""
    path = streams_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": REGISTRY_VERSION,
        "streams": [stream.model_dump(mode="json") for stream in streams],
    }
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # Keep the atomic-write contract observable: previous file intact, no
        # sidecar left to confuse the next run. The error itself propagates.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
