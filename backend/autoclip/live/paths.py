"""Filesystem layout additions for Live Watch.

Live Watch artifacts live beside the existing layout under the same root::

    <root>/
      live/
        buffers/<stream_id>/       rolling .ts segments (pruned aggressively)
        clips/<clip_id>/           per-clip working directory (judge I/O, renders)
        tokens/                    upload OAuth token blobs (opaque, keyring-

Freeze note: the ``streams.json`` persisted stream list goes in config, not
here; A3 owns that. This module exposes read-only path helpers only.
"""

from __future__ import annotations

from pathlib import Path

from .. import paths


def live_dir() -> Path:
    return paths.root() / "live"


def buffers_dir() -> Path:
    return live_dir() / "buffers"


def stream_buffer_dir(stream_id: str) -> Path:
    return buffers_dir() / stream_id


def clips_dir() -> Path:
    return live_dir() / "clips"


def clip_dir(clip_id: str) -> Path:
    return clips_dir() / clip_id


def ensure_live_layout() -> Path:
    base = live_dir()
    for path in (base, buffers_dir(), clips_dir()):
        path.mkdir(parents=True, exist_ok=True)
    return base
