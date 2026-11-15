"""Rolling buffer for a live stream's captured segments.

The watcher's capture layer writes short .ts segments continuously; the buffer
tracks what's on disk, prunes anything older than the retention window and the
disk cap, and can extract a contiguous range on demand for clipping.

Segment naming: ``seg_{index:06d}.ts`` where ``index`` counts from the start of
the watch (not from zero each reconnect).

Stream time is NOT ``index * SEGMENT_S``. The capture stream-copies, so
ffmpeg can only cut at keyframes: with ``-segment_time 6`` and a 4 s GOP the
segments come out 8 s, 4 s, 8 s ... (variable). Every segment therefore
carries its real ``start_s``/``duration_s``: the duration comes from ffmpeg's
segment list (``capture.py``), and the start is the running sum of the
durations before it, so the clock stays monotonic and gap-free within a
watch. Every time ↔ segment mapping below (extraction, pins, pruning, the
watcher's stream clock) reads that index, never the nominal length. When no
real duration is known (old callers, test doubles, a killed ffmpeg's last
segment) a segment falls back to the nominal ``SEGMENT_S`` — which is exactly
the old ``index * SEGMENT_S`` timeline.

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

#: NOMINAL seconds per segment file: the ``-segment_time`` ffmpeg is asked
#: for, and the fallback duration of a segment whose real length is unknown.
#: Real segments run longer (and vary) because stream copy cuts at keyframes
#: only — never convert index ↔ seconds with this; use the per-segment index.
SEGMENT_S = 6.0
#: Extension for segments. MPEG-TS is resilient to being cut mid-GOP, which is
#: exactly what a rolling live buffer needs.
SEGMENT_EXT = ".ts"
#: ffmpeg's per-spawn segment list (``seglist_<start index>.csv``) lives in
#: the buffer directory next to the segments it describes.
SEGMENT_LIST_PREFIX = "seglist_"

#: Hard ceiling on in-process pending clip bytes before watchers stall capture
#: to disk (safety valve; enforced by the manager, not the buffer itself).
PENDING_CAP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

#: How many of the newest committed segments the adaptive timers look at.
#: A handful is enough to learn the stream's keyframe interval, few enough
#: that a one-off long segment (a source hiccup) ages out quickly.
RECENT_SEGMENTS = 8

#: Copy chunk for extraction. Big enough to be I/O-bound rather than
#: syscall-bound, small enough that peak RAM never scales with clip length.
_CHUNK = 1024 * 1024


def segment_name(index: int) -> str:
    """The one place the on-disk segment naming contract lives."""
    return f"seg_{index:06d}{SEGMENT_EXT}"


def segment_list_name(start_index: int) -> str:
    """Name of the segment list an ffmpeg numbering from ``start_index`` writes."""
    return f"{SEGMENT_LIST_PREFIX}{start_index:06d}.csv"


@dataclass
class SegmentInfo:
    """Metadata for one segment on disk."""

    index: int
    path: Path
    size_bytes: int
    #: Segments written after a reconnect get a new ``epoch`` so clipping can
    #: refuse ranges that would jump across a discontinuity.
    epoch: int = 0
    #: Stream time this segment's media starts at. ``None`` (hand-built
    #: infos, old test doubles) means the nominal ``index * SEGMENT_S``.
    start_s: float | None = None
    #: Real media length (from ffmpeg's segment list); nominal when unknown.
    duration_s: float = SEGMENT_S

    def __post_init__(self) -> None:
        if self.start_s is None:
            self.start_s = self.index * SEGMENT_S

    @property
    def end_s(self) -> float:
        """Stream time this segment's media ends at (exclusive)."""
        return float(self.start_s) + self.duration_s  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExtractedWindow:
    """What :meth:`RingBuffer.extract_window` actually wrote.

    Extraction copies WHOLE segments, so the file usually starts before the
    requested start and ends after the requested end. ``start_s`` is the
    stream time of the file's first frame: media time ``t`` in the file is
    stream time ``start_s + t`` (transcript word times, trims, render cuts).
    """

    path: Path
    start_s: float
    end_s: float
    start_index: int
    end_index: int


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
        #: (index, end_s) of the newest segment ever committed. Survives
        #: pruning, so the stream clock keeps running even if the disk cap
        #: deleted every held segment.
        self._clock: tuple[int, float] | None = None

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

    def commit_segment(
        self,
        index: int,
        *,
        epoch: int | None = None,
        duration_s: float | None = None,
    ) -> SegmentInfo:
        """Register a fully written segment after ffmpeg closes it.

        ``duration_s`` is the segment's real media length (ffmpeg's segment
        list); ``None`` or nonsense falls back to the nominal ``SEGMENT_S``.
        The segment's ``start_s`` is derived from the segments before it.

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
            if duration_s is None or not math.isfinite(duration_s) or duration_s <= 0:
                duration_s = SEGMENT_S
            info = SegmentInfo(
                index=index,
                path=path,
                size_bytes=size,
                epoch=seg_epoch,
                start_s=self._start_for(index),
                duration_s=float(duration_s),
            )
            self.segments.append(info)
            if len(self.segments) > 1 and self.segments[-2].index > info.index:
                # Only reachable when a previous epoch's tailer commits late
                # (reconnect race); every consumer walks the list oldest →
                # newest, so keep it index-sorted.
                self.segments.sort(key=lambda seg: seg.index)
            self._last_committed[seg_epoch] = index
            if self._clock is None or index > self._clock[0]:
                self._clock = (index, info.end_s)
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
            return self.segments[-1].end_s - float(self.segments[0].start_s)  # type: ignore[arg-type]

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
            if seconds < 0:
                seconds = 0.0
            # Step back over WHOLE older segments while their real durations
            # still fit in ``seconds`` — the time-based form of the old
            # ``newest - floor(seconds / SEGMENT_S)`` (floor, never ceil).
            target = self.segments[-1].index
            budget = seconds
            for seg in reversed(self.segments[:-1]):
                if seg.duration_s > budget + 1e-9:
                    break
                budget -= seg.duration_s
                target = seg.index
            return target

    def covered_until_s(self) -> float | None:
        """Stream time up to which committed segments exist (None if empty).

        The real end of the newest committed segment. A clip window ending
        after this has not been captured yet.
        """
        with self._lock:
            if not self.segments:
                return None
            return self.segments[-1].end_s

    def recent_segment_max_s(self, count: int = RECENT_SEGMENTS) -> float | None:
        """Longest real duration among the newest ``count`` committed segments.

        The capture's stall watchdog and the watcher's coverage wait scale
        with this: segments are keyframe-cut, so a 20 s GOP means a new
        segment only every ~20 s, however short ``-segment_time`` is. None
        when nothing is committed yet (callers keep their base timeout).
        """
        with self._lock:
            recent = self.segments[-max(1, int(count)) :]
            if not recent:
                return None
            return max(seg.duration_s for seg in recent)

    def index_at(self, t_s: float) -> int:
        """Index of the segment holding stream time ``t_s`` (for pins).

        Before the oldest held segment → the oldest; past the newest → the
        nominal extrapolation beyond it; an empty buffer → nominal. A pin at
        the returned index therefore shields everything from ``t_s`` on.
        """
        with self._lock:
            return self._index_at_locked(t_s)

    def extract(
        self,
        start_s: float,
        end_s: float,
        target: Path,
        *,
        anchor_s: float | None = None,
    ) -> Path:
        """Concatenate the range [start_s, end_s) of stream time into ``target``.

        Thin wrapper over :meth:`extract_window` for callers that only need
        the path. Caller must hold a per-stream clipping lock so a prune race can't cut
        the segment set underneath us. The buffer prunes nothing while a
        extraction pin is held— see ``pin_oldest``.

        ``anchor_s`` (the trigger moment) makes a window that straddles a
        reconnect epoch usable: the range is clamped to the contiguous run of
        the anchor's own epoch instead of raising. Without an anchor an
        epoch-straddling range still raises (no way to tell which side
        matters).
        """
        return self.extract_window(start_s, end_s, target, anchor_s=anchor_s).path

    def extract_window(
        self,
        start_s: float,
        end_s: float,
        target: Path,
        *,
        anchor_s: float | None = None,
    ) -> ExtractedWindow:
        """:meth:`extract`, also reporting the media span actually written.

        Whole segments are copied, so the file covers ``[result.start_s,
        result.end_s)`` ⊇ the requested range (clamped to what is held).
        Consumers that place a moment inside the file — the judge's word
        offsets, the render's cut points — subtract ``result.start_s``.
        """
        if end_s <= start_s:
            raise ValueError(f"end_s ({end_s}) must exceed start_s ({start_s})")
        target = Path(target)
        with self._lock:
            if not self.segments:
                raise ValueError("buffer holds no segments; nothing to extract")
            oldest = self.segments[0]
            newest = self.segments[-1]
            # seconds → index through the REAL per-segment times: first
            # segment whose media reaches past start_s, last segment starting
            # before end_s (end_s itself excluded — the range is half-open),
            # both clamped to the segments that actually exist.
            start_s = max(start_s, 0.0)
            if start_s >= newest.end_s - 1e-9:
                raise ValueError(
                    f"range starts at {start_s:.1f}s, ahead of the newest captured"
                    f" segment {newest.index} (ends {newest.end_s:.1f}s); nothing to"
                    " extract there yet"
                )
            if end_s <= float(oldest.start_s) + 1e-9:  # type: ignore[arg-type]
                raise ValueError(
                    f"range ends below the oldest retained segment ({oldest.index});"
                    " it has already been pruned"
                )
            start_index = next(s.index for s in self.segments if s.end_s > start_s + 1e-9)
            end_index = max(s.index for s in self.segments if s.start_s < end_s - 1e-9)
            if anchor_s is not None:
                start_index, end_index = self._clamp_to_anchor_epoch(
                    start_index, end_index, anchor_s
                )
            # Validate contiguity + same-epoch and pin in ONE lock instant:
            # releasing the lock between those steps would let a prune delete
            # the range we just chose.
            paths = self.range_paths(start_index, end_index)
            held = {s.index: s for s in self.segments}
            window = ExtractedWindow(
                path=target,
                start_s=float(held[start_index].start_s),  # type: ignore[arg-type]
                end_s=held[end_index].end_s,
                start_index=start_index,
                end_index=end_index,
            )
            self.pin_oldest(start_index)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _concat(paths, target)
        finally:
            self.unpin(start_index)
        return window

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
            # Ages are measured in REAL stream time from the newest segment's
            # start (or ``now_index``'s), so long keyframe-cut segments do
            # not stretch the retention window.
            now_s = self._start_for_now(now_index)
            freed = 0

            # Pass 1 — retention. Age is monotonic along the sorted list, so
            # the first unexpired segment ends the pass. A pin stops it dead:
            # pruning must never step over a shielded segment to free newer
            # bytes.
            while self.segments:
                seg = self.segments[0]
                if now_s - float(seg.start_s) < self.retain_s:  # type: ignore[arg-type]
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
        anchor_index = max(start_index, min(end_index, self._index_at_locked(anchor_s)))
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

    def _start_for(self, index: int) -> float:
        """Stream start of a NEW segment ``index``: the end of the one before.

        The predecessor is the nearest lower index still held, or the clock
        record of the newest ever committed (which survives pruning). Index
        gaps (a skipped zero-byte segment) count nominally, so an all-nominal
        buffer reproduces the old ``index * SEGMENT_S`` timeline exactly.
        Called under the lock.
        """
        pred: tuple[int, float] | None = None
        for seg in self.segments:
            if seg.index >= index:
                break
            pred = (seg.index, seg.end_s)
        clock = self._clock
        if clock is not None and clock[0] < index and (pred is None or clock[0] > pred[0]):
            pred = clock
        if pred is None:
            return index * SEGMENT_S
        return pred[1] + (index - pred[0] - 1) * SEGMENT_S

    def _start_for_now(self, now_index: int | None) -> float:
        """Stream time prune() measures ages from. Called under the lock."""
        newest = self.segments[-1]
        newest_start = float(newest.start_s)  # type: ignore[arg-type]
        if now_index is None:
            return newest_start
        for seg in self.segments:
            if seg.index == now_index:
                return float(seg.start_s)  # type: ignore[arg-type]
        return newest_start + (now_index - newest.index) * SEGMENT_S

    def _index_at_locked(self, t_s: float) -> int:
        """See :meth:`index_at`. Called under the lock."""
        t_s = max(0.0, float(t_s))
        if not self.segments:
            return int(t_s // SEGMENT_S)
        newest = self.segments[-1]
        if t_s >= newest.end_s:
            return newest.index + 1 + int((t_s - newest.end_s) // SEGMENT_S)
        # The holder of t_s — or, in a hole / before the oldest, the next
        # held segment (a pin there still covers t_s onward).
        return next(seg.index for seg in self.segments if t_s < seg.end_s)

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
        self._clipped_s += seg.duration_s
        return seg.size_bytes


def segment_index_at(buffer: object, t_s: float) -> int:
    """Segment index holding stream time ``t_s`` on any buffer-like (pins).

    Uses :meth:`RingBuffer.index_at` (the real per-segment index); minimal
    test doubles without it get the nominal ``t_s // SEGMENT_S``.
    """
    index_at = getattr(buffer, "index_at", None)
    if callable(index_at):
        try:
            return int(index_at(t_s))
        except Exception:
            log.debug("index_at(%s) failed; using the nominal index", t_s, exc_info=True)
    return int(max(0.0, t_s) // SEGMENT_S)


def discard_segment_files(directory: Path) -> int:
    """Delete every ``seg_*.ts`` (and stale ``seglist_*.csv``) in ``directory``.

    Returns the number of files removed.
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
        is_segment = name.startswith("seg_") and name.endswith(SEGMENT_EXT)
        is_list = name.startswith(SEGMENT_LIST_PREFIX) and name.endswith(".csv")
        if not (is_segment or is_list):
            continue
        try:
            entry.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except OSError:
            log.debug("could not delete stale segment %s", entry, exc_info=True)
    return removed


def adaptive_timeout_s(
    base_s: float,
    longest_segment_s: float | None,
    *,
    segments: float,
    ceiling_s: float,
) -> float:
    """A timeout that stretches with the stream's real segment length.

    ``max(base, min(ceiling, segments × longest))``: never shorter than the
    base (the nominal-segment behaviour), stretched for long-GOP streams so a
    healthy 20 s-keyframe capture is not mistaken for a stall, and capped so
    a dead capture is still caught. The ceiling never pulls below the base.
    """
    base = float(base_s)
    if longest_segment_s is None or not math.isfinite(longest_segment_s) or longest_segment_s <= 0:
        return base
    return max(base, min(float(ceiling_s), float(segments) * float(longest_segment_s)))


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
    "RECENT_SEGMENTS",
    "SEGMENT_EXT",
    "SEGMENT_LIST_PREFIX",
    "SEGMENT_S",
    "ExtractedWindow",
    "RingBuffer",
    "SegmentInfo",
    "adaptive_timeout_s",
    "discard_segment_files",
    "segment_index_at",
    "segment_list_name",
    "segment_name",
]
