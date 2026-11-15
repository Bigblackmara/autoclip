"""Rolling buffer for a live stream's captured segments.

The watcher's capture layer writes short .ts segments continuously; the buffer
tracks what's on disk, prunes anything older than the retention window and the
disk cap, and can extract a contiguous range on demand for clipping.

Segment naming: ``seg_{index:06d}.ts`` where ``index`` counts from the start of
the watch (not from zero each reconnect). Small integers map to wall-clock
monotonically; index → seconds is ``index * SEGMENT_S``.

Implementation notes (why it looks like this):

* **One lock guards the metadata; pins guard the data.** The capture tailer
  thread commits segments while the watcher prunes and clip workers extract
  from other threads — the segment list is shared storage, so mutations take
  one lock. Extraction copies bytes *without* holding the lock: it pins its
  range up front instead, so a long copy never stalls the live capture. The
  pin is taken inside the same lock instant that validates the range;
  validating, releasing the lock, then pinning would let a prune delete the
  very segments the extract just chose.

* **Pins are thresholds, not points.** ``pin_oldest(p)`` shields every segment
  at or above ``p`` — a clipper needs its pre-roll start *and* everything the
  pre-roll precedes. Pruning stops dead at a shielded segment rather than
  stepping over it to free newer ones. Accepted cost: a leaked pin can hold
  the buffer past its disk cap, because pins win over pruning on purpose —
  deleting the bytes an in-flight clip needs is the one unrecoverable mistake
  here, while a temporarily fat buffer only costs disk.

* **Un-committed files do not exist.** ``commit_segment`` is the only way a
  file becomes real to this bookkeeping, so a partial write from a crashed
  process can never leak into an extraction.
"""

from __future__ import annotations

import logging
import math
import threading
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

#: Copy chunk for extraction. Big enough to be I/O-bound rather than
#: syscall-bound, small enough that peak RAM never scales with clip length.
_CHUNK = 1024 * 1024


def segment_name(index: int) -> str:
    """The one place the on-disk segment naming contract lives."""
    return f"seg_{index:06d}{SEGMENT_EXT}"


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
        self.directory = Path(directory)
        self.retain_s = retain_minutes * 60.0
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.segments: list[SegmentInfo] = []
        self._epoch = 0

        # -- A1 internals (implementation details, not the frozen contract) ---
        # RLock: extract() works under one nesting level, then the helpers it
        # calls (range_paths/pin_oldest) re-enter the lock on the same thread.
        self._lock = threading.RLock()
        #: index -> refcount. A pin at index p shields every segment >= p.
        self._pins: dict[int, int] = {}
        #: Highest committed index per epoch — enforces per-epoch ordering.
        self._last_committed: dict[int, int] = {}
        #: Cumulative stream seconds deleted by prune() — stats only.
        self._clipped_s = 0.0

        # Eager mkdir: the capture layer (and tests) write segment files
        # before any commit, and a buffer pointing at a missing directory is
        # never useful to anyone.
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- capture side ------------------------------------------------------

    def next_segment_path(self, index: int) -> Path:
        """Return the path the capture layer should write segment ``index`` to.

        A frozen file must never exist at the returned path: scissors paths
        assume every listed segment is fully written. Two situations can put a
        file there — an uncommitted partial from a crashed process (safe to
        clear), or an already-committed segment (would rewrite history).
        """
        with self._lock:
            if any(seg.index == index for seg in self.segments):
                raise ValueError(
                    f"segment index {index} is already committed; rewriting"
                    " history would corrupt every index→seconds calculation"
                )
            path = self.directory / segment_name(index)
            if path.exists():
                # Un-committed partial: by the commit invariant nothing real
                # can depend on it, so discarding is safe.
                path.unlink(missing_ok=True)
            return path

    def commit_segment(self, index: int, *, epoch: int | None = None) -> SegmentInfo:
        """Register a fully written segment after ffmpeg closes it.

        Returns the committed SegmentInfo. Commits are strictly ordered per
        epoch; a commit for an older index than one already registered raises.
        """
        if index < 0:
            raise ValueError(f"segment index must be >= 0, got {index}")
        with self._lock:
            seg_epoch = self._epoch if epoch is None else epoch
            if seg_epoch > self._epoch:
                raise ValueError(
                    f"epoch {seg_epoch} is ahead of the buffer's current epoch"
                    f" {self._epoch}; commits from the future are a tailer bug"
                )
            if any(seg.index == index for seg in self.segments):
                # Indices are global to the watch and never reset, so a
                # duplicate would double-count one file's stream time.
                raise ValueError(f"segment index {index} is already registered")
            last = self._last_committed.get(seg_epoch, -1)
            if index <= last:
                raise ValueError(
                    f"segment {index} is not newer than epoch {seg_epoch}'s last"
                    f" committed index {last}; commits must be strictly ordered"
                )
            path = self.directory / segment_name(index)
            try:
                size = path.stat().st_size
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"segment file {path.name} is not on disk; commit only after"
                    " ffmpeg has fully written and closed the file"
                ) from exc
            info = SegmentInfo(index=index, path=path, size_bytes=size, epoch=seg_epoch)
            self.segments.append(info)
            if len(self.segments) > 1 and self.segments[-2].index > info.index:
                # Only reachable when a previous epoch's tailer commits late
                # (reconnect race); every consumer walks the list oldest →
                # newest, so keep it index-sorted.
                self.segments.sort(key=lambda seg: seg.index)
            self._last_committed[seg_epoch] = index
            return info

    def begin_epoch(self) -> int:
        """Start a new epoch (after a reconnect). Returns the new epoch id."""
        with self._lock:
            self._epoch += 1
            return self._epoch

    # -- clipping side -----------------------------------------------------

    def span_seconds(self) -> float:
        """Total stream seconds currently held on disk."""
        with self._lock:
            if not self.segments:
                return 0.0
            return (self.segments[-1].index - self.segments[0].index + 1) * SEGMENT_S

    def clipped_seconds(self) -> float:
        """Seconds freed/lost (pruned or beyond cap) — for stats only."""
        with self._lock:
            return self._clipped_s

    def range_paths(self, start_index: int, end_index: int) -> list[Path]:
        """Paths for a contiguous, same-epoch index range (inclusive)."""
        if start_index > end_index:
            raise ValueError(f"range start {start_index} is after end {end_index}")
        with self._lock:
            in_range = [seg for seg in self.segments if start_index <= seg.index <= end_index]
            if len(in_range) != end_index - start_index + 1:
                raise ValueError(
                    f"segments {start_index}..{end_index} are not all held"
                    " (pruned, never captured, or an epoch boundary sits inside"
                    " the range)"
                )
            epochs = {seg.epoch for seg in in_range}
            if len(epochs) > 1:
                raise ValueError(
                    f"range {start_index}..{end_index} spans epochs"
                    f" {sorted(epochs)}; a reconnect discontinuity sits inside"
                    " it and the concatenated timeline would lie"
                )
            return [seg.path for seg in in_range]

    def indices_back(self, seconds: float) -> int:
        """Index ``seconds`` behind the newest committed segment.

        Clamped to the oldest retained segment. Volatile: valid only
        immediately after the call — new segments may commit at any moment.
        """
        with self._lock:
            if not self.segments:
                raise ValueError("buffer holds no segments; nothing to compute a pre-roll from")
            newest = self.segments[-1].index
            oldest = self.segments[0].index
            if seconds < 0:
                seconds = 0.0
            # floor, never ceil: subscription math must over-cover, never
            # under-cover — a pre-roll one segment short silently cuts the
            # moment's setup line, and no metric ever catches that.
            target = newest - math.floor(seconds / SEGMENT_S)
            return max(oldest, min(newest, target))

    def covered_until_s(self) -> float | None:
        """Stream time up to which committed segments exist (None if empty).

        ``(newest index + 1) * SEGMENT_S`` — the end of the newest committed
        segment. A clip window ending after this has not been captured yet.
        """
        with self._lock:
            if not self.segments:
                return None
            return (self.segments[-1].index + 1) * SEGMENT_S

    def extract(
        self,
        start_s: float,
        end_s: float,
        target: Path,
        *,
        anchor_s: float | None = None,
    ) -> Path:
        """Concatenate the range [start_s, end_s) of stream time into ``target``.

        Caller must hold a per-stream clipping lock so a prune race can't cut
        the segment set underneath us. The buffer prunes nothing while a
        extraction pin is held— see ``pin_oldest``.

        ``anchor_s`` (the trigger moment) makes a window that straddles a
        reconnect epoch usable: the range is clamped to the contiguous run of
        the anchor's own epoch instead of raising. Without an anchor an
        epoch-straddling range still raises (no way to tell which side
        matters).
        """
        if end_s <= start_s:
            raise ValueError(f"end_s ({end_s}) must exceed start_s ({start_s})")
        target = Path(target)
        with self._lock:
            if not self.segments:
                raise ValueError("buffer holds no segments; nothing to extract")
            oldest = self.segments[0].index
            newest = self.segments[-1].index
            # seconds → index: first segment touching start_s, last segment
            # touching end_s (end_s itself excluded — the range is half-open),
            # then both clamped to the segments that actually exist.
            raw_start = math.floor(max(start_s, 0.0) / SEGMENT_S)
            raw_end = math.ceil(end_s / SEGMENT_S) - 1
            if raw_start > newest:
                raise ValueError(
                    f"range starts at index {raw_start}, ahead of the newest"
                    f" captured index {newest}; nothing to extract there yet"
                )
            if raw_end < oldest:
                raise ValueError(
                    f"range ends below the oldest retained segment ({oldest});"
                    " it has already been pruned"
                )
            start_index = max(raw_start, oldest)
            end_index = min(raw_end, newest)
            if anchor_s is not None:
                start_index, end_index = self._clamp_to_anchor_epoch(
                    start_index, end_index, anchor_s
                )
            # Validate contiguity + same-epoch and pin in ONE lock instant:
            # releasing the lock between those steps would let a prune delete
            # the range we just chose.
            paths = self.range_paths(start_index, end_index)
            self.pin_oldest(start_index)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _concat(paths, target)
        finally:
            self.unpin(start_index)
        return target

    def pin_oldest(self, index: int) -> None:
        """Prevent pruning of segments at or above ``index`` until un-pinned.

        Used while a clip is being produced so an aggressive prune can't delete
        the very segments the clip needs. Reference-counted per index pin.
        """
        with self._lock:
            self._pins[index] = self._pins.get(index, 0) + 1

    def unpin(self, index: int | None = None) -> None:
        """Release pin(s). ``None`` clears all pins for this buffer."""
        with self._lock:
            if index is None:
                self._pins.clear()
                return
            count = self._pins.get(index, 0)
            if count <= 0:
                # Lenient on purpose: unpin runs in finally blocks, and killing
                # a watcher over bookkeeping would turn a warning-level bug
                # into a stream outage. The warning is how the leak gets seen.
                log.warning("unpin(%s) without a matching pin_oldest(%s)", index, index)
                return
            if count == 1:
                del self._pins[index]
            else:
                self._pins[index] = count - 1

    # -- maintenance -------------------------------------------------------

    def prune(self, *, now_index: int | None = None) -> int:
        """Delete and forget segments outside retention / disk cap.

        Returns bytes freed. Respects pins. Idempotent.
        """
        with self._lock:
            if not self.segments:
                return 0
            now = now_index if now_index is not None else self.segments[-1].index
            freed = 0

            # Pass 1 — retention. Age is monotonic along the sorted list, so
            # the first unexpired segment ends the pass. A pin stops it dead:
            # pruning must never step over a shielded segment to free newer
            # bytes.
            while self.segments:
                seg = self.segments[0]
                if (now - seg.index) * SEGMENT_S < self.retain_s:
                    break
                if self._shielded(seg.index):
                    break
                got = self._delete(seg)
                if got == 0:
                    break  # file held open; retry next prune cycle
                freed += got

            # Pass 2 — disk cap, oldest-deletable-first so the newest (most
            # clip-worthy) bytes survive pressure. Pinned segments are skipped
            # and the buffer may legitimately stay over cap.
            total = sum(seg.size_bytes for seg in self.segments)
            while total > self.max_disk_bytes:
                victim = next((s for s in self.segments if not self._shielded(s.index)), None)
                if victim is None:
                    log.debug("disk cap %d held hostage by pins", self.max_disk_bytes)
                    break
                got = self._delete(victim)
                if got == 0:
                    break  # file held open; retry next prune cycle
                total -= got
                freed += got

            return freed

    def close(self) -> None:
        """Drop all pins and forget the directory (stream removed)."""
        with self._lock:
            self.segments.clear()
            self._pins.clear()
            self._last_committed.clear()
            self._epoch = 0
        # Files stay on disk: removing the directory is the caller's job —
        # the manager owns the stream subtree and any rmtree must also catch
        # non-segment artifacts the buffer doesn't know about.

    # -- internals ----------------------------------------------------------

    def _clamp_to_anchor_epoch(
        self, start_index: int, end_index: int, anchor_s: float
    ) -> tuple[int, int]:
        """Shrink ``[start_index, end_index]`` to the anchor's epoch run.

        A reconnect inside a clip window leaves two timelines in it; the one
        holding the trigger moment is the clip. Called under the lock.
        """
        in_range = [s for s in self.segments if start_index <= s.index <= end_index]
        if len({s.epoch for s in in_range}) <= 1:
            return start_index, end_index
        anchor_index = max(start_index, min(end_index, math.floor(max(anchor_s, 0.0) / SEGMENT_S)))
        anchor = min(in_range, key=lambda s: (abs(s.index - anchor_index), -s.index))
        position = in_range.index(anchor)
        first = last = position
        while first > 0 and in_range[first - 1].epoch == anchor.epoch:
            first -= 1
        while last < len(in_range) - 1 and in_range[last + 1].epoch == anchor.epoch:
            last += 1
        log.info(
            "window %d..%d straddles a reconnect; clamped to epoch %d (%d..%d)",
            start_index,
            end_index,
            anchor.epoch,
            in_range[first].index,
            in_range[last].index,
        )
        return in_range[first].index, in_range[last].index

    def _shielded(self, index: int) -> bool:
        """Whether any pin shields the segment at ``index`` (pin at or below it)."""
        return any(pin <= index for pin in self._pins)

    def _delete(self, seg: SegmentInfo) -> int:
        """Unlink and forget ``seg``; bytes freed, or 0 if the delete must retry."""
        try:
            seg.path.unlink(missing_ok=True)
        except PermissionError:
            # Windows: something holds the file open. On POSIX an unlink races
            # harmlessly; here it raises. Keep the segment listed so the next
            # prune retries, and never kill a watcher over a file lock.
            log.debug("segment %s is in use; deferring delete", seg.path.name)
            return 0
        self.segments.remove(seg)
        self._clipped_s += SEGMENT_S
        return seg.size_bytes


def discard_segment_files(directory: Path) -> int:
    """Delete every ``seg_*.ts`` file in ``directory``; returns files removed.

    A fresh RingBuffer knows nothing about files a previous lifecycle left
    behind, while a fresh ffmpeg numbers from 0 — so stale high-index files
    would be committed as if new and every real segment refused as out of
    order. A new watch lifecycle therefore starts from an empty folder.
    Files still held open (Windows) are skipped, never fatal.
    """
    directory = Path(directory)
    removed = 0
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return 0
    for entry in entries:
        name = entry.name
        if not (name.startswith("seg_") and name.endswith(SEGMENT_EXT)):
            continue
        try:
            entry.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except OSError:
            log.debug("could not delete stale segment %s", entry, exc_info=True)
    return removed


def _concat(sources: list[Path], target: Path) -> None:
    """Stream-copy ``sources`` into ``target`` in order.

    MPEG-TS segments concatenate cleanly at the byte level (self-synchronizing
    188-byte packets), so no container-aware remux is needed — but the copy
    still streams in fixed chunks rather than reading whole files, so peak RAM
    never scales with clip length.
    """
    with open(target, "wb") as out:
        for source in sources:
            with open(source, "rb") as src:
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)


__all__ = [
    "PENDING_CAP_BYTES",
    "SEGMENT_EXT",
    "SEGMENT_S",
    "RingBuffer",
    "SegmentInfo",
    "discard_segment_files",
    "segment_name",
]
