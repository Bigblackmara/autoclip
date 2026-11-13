"""Rolling buffer for a live stream's captured segments.

The watcher's capture layer writes short .ts segments continuously; the buffer
tracks what's on disk, prunes anything older than the retention window and the
disk cap, and can extract a contiguous range on demand for clipping.

Segment naming: ``seg_{index:06d}.ts`` where ``index`` counts from the start of
the watch (not from zero each reconnect). Small integers map to wall-clock
monotonically; index → seconds is ``index * SEGMENT_S``.

This file is a frozen interface skeleton: A1 (capture/buffer implementation)
fills in the bodies. The signatures below are the contract other workers code
against — do not change them without updating docs/LIVE_WATCH_WORK_ORDERS.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Seconds of stream time per segment file.
SEGMENT_S = 6.0
#: Extension for segments. MPEG-TS is resilient to being cut mid-GOP, which is
#: exactly what a rolling live buffer needs.
SEGMENT_EXT = ".ts"

#: Hard ceiling on in-process pending clip bytes before watchers stall capture
#: to disk (safety valve; enforced by the manager, not the buffer itself).
PENDING_CAP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


@dataclass
class SegmentInfo:
    """Metadata for one segment on disk."""

    index: int
    path: Path
    size_bytes: int
    #: Segments written after a reconnect get a new ``epoch`` so clipping can
    #: refuse ranges that would jump across a discontinuity.
    epoch: int = 0


class RingBuffer:
    """Segments on disk for one stream, newest at the end."""

    def __init__(
        self,
        directory: Path,
        *,
        retain_minutes: float = 5.0,
        max_disk_mb: int = 512,
    ) -> None:
        self.directory = directory
        self.retain_s = retain_minutes * 60.0
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.segments: list[SegmentInfo] = []
        self._epoch = 0

    # -- capture side ------------------------------------------------------

    def next_segment_path(self, index: int) -> Path:
        """Return the path the capture layer should write segment ``index`` to.

        A frozen file must never exist at the returned path: scissors paths
        assume every listed segment is fully written.
        """
        raise NotImplementedError("A1: implement")

    def commit_segment(self, index: int, *, epoch: int | None = None) -> SegmentInfo:
        """Register a fully written segment after ffmpeg closes it.

        Returns the committed SegmentInfo. Commits are strictly ordered per
        epoch; a commit for an older index than one already registered raises.
        """
        raise NotImplementedError("A1: implement")

    def begin_epoch(self) -> int:
        """Start a new epoch (after a reconnect). Returns the new epoch id."""
        raise NotImplementedError("A1: implement")

    # -- clipping side -----------------------------------------------------

    def span_seconds(self) -> float:
        """Total stream seconds currently held on disk."""
        raise NotImplementedError("A1: implement")

    def clipped_seconds(self) -> float:
        """Seconds freed/lost (pruned or beyond cap) — for stats only."""
        raise NotImplementedError("A1: implement")

    def range_paths(self, start_index: int, end_index: int) -> list[Path]:
        """Paths for a contiguous, same-epoch index range (inclusive)."""
        raise NotImplementedError("A1: implement")

    def indices_back(self, seconds: float) -> int:
        """Index ``seconds`` behind the newest committed segment.

        Clamped to the oldest retained segment. Volatile: valid only
        immediately after the call — new segments may commit at any moment.
        """
        raise NotImplementedError("A1: implement")

    def extract(self, start_s: float, end_s: float, target: Path) -> Path:
        """Concatenate the range [start_s, end_s) of stream time into ``target``.

        Caller must hold a per-stream clipping lock so a prune race can't cut
        the segment set underneath us. The buffer prunes nothing while a
        extraction pin is held— see ``pin_oldest``.
        """
        raise NotImplementedError("A1: implement")

    def pin_oldest(self, index: int) -> None:
        """Prevent pruning of segments at or above ``index`` until un-pinned.

        Used while a clip is being produced so an aggressive prune can't delete
        the very segments the clip needs. Reference-counted per index pin.
        """
        raise NotImplementedError("A1: implement")

    def unpin(self, index: int | None = None) -> None:
        """Release pin(s). ``None`` clears all pins for this buffer."""
        raise NotImplementedError("A1: implement")

    # -- maintenance -------------------------------------------------------

    def prune(self, *, now_index: int | None = None) -> int:
        """Delete and forget segments outside retention / disk cap.

        Returns bytes freed. Respects pins. Idempotent.
        """
        raise NotImplementedError("A1: implement")

    def close(self) -> None:
        """Drop all pins and forget the directory (stream removed)."""
        raise NotImplementedError("A1: implement")
