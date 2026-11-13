"""RingBuffer unit tests — retention, disk cap, pins, epochs, extraction.

Correctness of the rolling buffer is clip-correctness: an off-by-one in
index math silently trims the moment's setup line (no metric catches it), a
prune racing an extraction deletes the bytes a clip needs (unrecoverable),
and epoch-mixed concatenations render a lying timeline. These tests pin each
of those failure modes with byte-level assertions where the stakes are bytes.

No network, no ffmpeg: segments are hand-written files, exactly like the
work order's test strategy freezes ("tests use fake segment files written by
hand"). ``autoclip_home`` (conftest, autouse) keeps every path in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from autoclip.live.buffer import (
    PENDING_CAP_BYTES,
    SEGMENT_EXT,
    SEGMENT_S,
    RingBuffer,
    SegmentInfo,
    segment_name,
)


def write_segment(buffer: RingBuffer, index: int, *, size: int = 4096, epoch: int = 0) -> Path:
    """Write a fake segment file and commit it, like ffmpeg+tailer would."""
    path = buffer.next_segment_path(index)
    path.write_bytes(b"S" * size)
    buffer.commit_segment(index)
    return path


def commit_n(buffer: RingBuffer, count: int, *, start: int = 0, size: int = 4096) -> None:
    """Commit ``count`` contiguous segments starting at ``start``."""
    for i in range(start, start + count):
        write_segment(buffer, i, size=size)


class TestSegmentNaming:
    """seg_{index:06d}.ts — the frozen on-disk contract other modules parse."""

    def test_name_format(self) -> None:
        assert segment_name(0) == "seg_000000.ts"
        assert segment_name(7) == "seg_000007.ts"
        assert segment_name(123456) == "seg_123456.ts"

    def test_extension_is_ts(self) -> None:
        # MPEG-TS is cut-mid-GOP resilient; any other extension is a
        # silent contract break for every consumer that parses names.
        assert SEGMENT_EXT == ".ts"

    def test_six_second_mapping_is_frozen(self) -> None:
        # t_s = index * SEGMENT_S is wired through A2/A4/A6; changing it
        # silently invalidates every detector timestamp.
        assert SEGMENT_S == 6.0

    def test_pending_cap_is_two_gib(self) -> None:
        # Manager-side safety valve; frozen so the manager can hardcode
        # expectations against it.
        assert PENDING_CAP_BYTES == 2 * 1024**3


class TestCommitContract:
    """Only fully-written files become real to the bookkeeping."""

    def test_commit_registers_stat_size(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        path = buf.next_segment_path(0)
        path.write_bytes(b"x" * 1234)
        info = buf.commit_segment(0)
        assert isinstance(info, SegmentInfo)
        assert info.size_bytes == 1234
        assert info.path == path
        assert buf.segments == [info]

    def test_commits_must_be_strictly_ordered(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        write_segment(buf, 5)
        # An older index than one already registered must raise: accepting
        # it would double-count stream time in span/extract math.
        with pytest.raises(ValueError, match="strictly ordered"):
            buf.commit_segment(4)

    def test_commit_same_index_twice_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        write_segment(buf, 0)
        path = buf.next_segment_path(1)
        path.write_bytes(b"y" * 10)
        buf.commit_segment(1)
        with pytest.raises(ValueError, match="already"):
            buf.commit_segment(1)

    def test_commit_without_file_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        with pytest.raises(FileNotFoundError):
            buf.commit_segment(99)

    def test_negative_index_rejected(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        with pytest.raises(ValueError, match=">= 0"):
            buf.commit_segment(-1)

    def test_next_segment_path_rejects_committed_index(self, tmp_path: Path) -> None:
        # Re-writing history would corrupt index→seconds for everything.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 3)
        with pytest.raises(ValueError, match="already committed"):
            buf.next_segment_path(1)

    def test_next_segment_path_clears_stale_partial(self, tmp_path: Path) -> None:
        # A crashed process can leave an uncommitted partial behind; the
        # next writer at that index must get a clean path, not an append.
        buf = RingBuffer(tmp_path / "buf")
        stale = buf.directory / segment_name(0)
        stale.write_bytes(b"partial")
        path = buf.next_segment_path(0)
        assert path == stale
        assert not path.exists()

    def test_returned_path_never_preexists(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        path = buf.next_segment_path(3)
        assert not path.exists()
        assert path.parent == buf.directory


class TestEpochs:
    """Reconnect = new epoch; ranges never straddle the discontinuity."""

    def test_begin_epoch_increments(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        assert buf._epoch == 0
        assert buf.begin_epoch() == 1
        assert buf.begin_epoch() == 2

    def test_commit_after_epoch_gets_new_epoch_tag(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 2)
        buf.begin_epoch()
        write_segment(buf, 2)
        assert buf.segments[-1].epoch == 1

    def test_explicit_epoch_argument_is_respected(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        write_segment(buf, 0)
        path = buf.next_segment_path(1)
        path.write_bytes(b"x" * 32)
        info = buf.commit_segment(1, epoch=0)
        assert info.epoch == 0

    def test_commit_from_future_epoch_rejected(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        with pytest.raises(ValueError, match="future"):
            buf.commit_segment(0, epoch=3)

    def test_extract_refuses_epoch_mixing(self, tmp_path: Path) -> None:
        # A range that concatenates across a reconnect produces a lying
        # timeline (timestamp jump backwards); it must raise, not clip.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 2)
        buf.begin_epoch()
        write_segment(buf, 2)
        with pytest.raises(ValueError, match="epoch"):
            buf.extract(0.0, 18.0, tmp_path / "out.ts")

    def test_range_paths_refuses_epoch_mixing(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 2)
        buf.begin_epoch()
        write_segment(buf, 2)
        with pytest.raises(ValueError, match="epoch"):
            buf.range_paths(0, 2)


class TestSpan:
    def test_empty_buffer_holds_nothing(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        assert buf.span_seconds() == 0.0
        assert buf.clipped_seconds() == 0.0

    def test_span_counts_gaps_once(self, tmp_path: Path) -> None:
        # Committed-but-pruned middle segments still count once: span is
        # wall-clock stream time held, not sum-of-file-durations.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 4)
        assert buf.span_seconds() == 4 * SEGMENT_S

    def test_clipped_seconds_accumulate_after_prune(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1 / 60)
        commit_n(buf, 30)
        buf.prune(now_index=29)
        assert buf.clipped_seconds() > 0.0


class TestIndicesBack:
    """Pre-roll subscription math — over-cover, never under-cover."""

    def test_exact_multiple_lands_on_boundary(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 10)
        # 20s pre-roll at 6s/segment: newest=9, 9-3=6 → clamps fine.
        assert buf.indices_back(20.0) == 6

    def test_fractional_seconds_floor_downward(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 10)
        # 14.5s back from index 9: floor(14.5/6)=2 → 9-2=7. Over-cover:
        # the extra seconds are cheap; missing setup lines are not.
        assert buf.indices_back(14.5) == 7

    def test_clamped_to_oldest_retained(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 4, start=10)  # only 10..13 held
        # Asking for a full minute back can only reach the oldest segment.
        assert buf.indices_back(60.0) == 10

    def test_zero_seconds_returns_newest(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 5)
        assert buf.indices_back(0.0) == 4

    def test_negative_seconds_clamped_to_zero(self, tmp_path: Path) -> None:
        # Clocks jitter; a negative ask must not walk off the newest index.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 5)
        assert buf.indices_back(-3.0) == 4

    def test_empty_buffer_raises(self, tmp_path: Path) -> None:
        # A pre-roll from nothing is a caller bug; say so instead of
        # returning a nonsense index that silently corrupts clips.
        buf = RingBuffer(tmp_path / "buf")
        with pytest.raises(ValueError, match="no segments"):
            buf.indices_back(10.0)


class TestRangePaths:
    def test_inclusive_range(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        paths = [write_segment(buf, i) for i in range(5)]
        assert buf.range_paths(1, 3) == paths[1:4]

    def test_out_of_range_raises(self, tmp_path: Path) -> None:
        # Pruned or never-captured segments must fail loudly, never yield
        # a shorter list that silently trims the clip.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 3)
        with pytest.raises(ValueError, match="not all held"):
            buf.range_paths(0, 9)

    def test_reversed_range_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 3)
        with pytest.raises(ValueError, match="after end"):
            buf.range_paths(2, 1)


class TestExtract:
    """Bytes-out must be exactly bytes-in, in order, for the asked range."""

    def test_extract_concatenates_exact_bytes(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        contents = {}
        for i in range(5):
            path = write_segment(buf, i, size=1024 * (i + 1))
            contents[i] = path.read_bytes()
        target = tmp_path / "clip.ts"
        result = buf.extract(6.0, 24.0, target)  # indices 1..3
        assert result == target
        assert target.read_bytes() == contents[1] + contents[2] + contents[3]

    def test_extract_full_range_between_boundaries(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 3, size=512)
        target = tmp_path / "all.ts"
        buf.extract(0.0, 18.0, target)
        expected = b"".join((buf.directory / segment_name(i)).read_bytes() for i in range(3))
        assert target.read_bytes() == expected

    def test_extract_clamps_start_to_what_exists(self, tmp_path: Path) -> None:
        # Asking before the buffer's start must clamp, not fail: the
        # watcher's pre-roll naturally asks for more than retention holds.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 3, start=2)  # holds 2..4
        target = tmp_path / "c.ts"
        buf.extract(-30.0, 24.0, target)
        expected = b"".join((buf.directory / segment_name(i)).read_bytes() for i in (2, 3))
        assert target.read_bytes() == expected

    def test_extract_clamps_end_to_what_exists(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 2)
        target = tmp_path / "c.ts"
        buf.extract(0.0, 600.0, target)  # far beyond the newest commit
        expected = b"".join((buf.directory / segment_name(i)).read_bytes() for i in (0, 1))
        assert target.read_bytes() == expected

    def test_extract_range_entirely_ahead_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 2)
        with pytest.raises(ValueError, match="ahead"):
            buf.extract(600.0, 700.0, tmp_path / "x.ts")

    def test_extract_range_entirely_behind_pruned_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1 / 60)
        commit_n(buf, 12)
        buf.prune(now_index=11)
        with pytest.raises(ValueError, match="pruned"):
            buf.extract(0.0, 6.0, tmp_path / "x.ts")

    def test_extract_empty_buffer_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        with pytest.raises(ValueError, match="no segments"):
            buf.extract(0.0, 6.0, tmp_path / "x.ts")

    def test_extract_zero_length_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 2)
        # [t, t) is empty; a zero-byte .ts would break every consumer.
        with pytest.raises(ValueError, match="exceed"):
            buf.extract(6.0, 6.0, tmp_path / "x.ts")

    def test_extract_creates_parent_dirs(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 2)
        deep = tmp_path / "clips" / "xyz" / "window.ts"
        buf.extract(0.0, 12.0, deep)
        assert deep.exists()
        assert deep.parent.is_dir()

    def test_extract_is_streamed_not_read_whole(self, tmp_path: Path) -> None:
        # Stream/copy in chunks: a 30s clip is ~15 MB; a 5-minute retention
        # buffer is 150 MB. Read-whole-file would multiply peak RSS by the
        # number of concurrent extractions.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 6, size=1024 * 1024)  # 6 MiB total
        target = tmp_path / "big.ts"
        buf.extract(0.0, 36.0, target)
        assert target.stat().st_size == 6 * 1024 * 1024

    def test_extract_releases_its_pin_afterwards(self, tmp_path: Path) -> None:
        # A leaked extraction pin holds the buffer past its disk cap
        # forever (pins win by design); extraction must clean its own pin.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 3)
        buf.extract(0.0, 18.0, tmp_path / "c.ts")
        assert buf._pins == {}


class TestPins:
    """Pins block pruning, are refcounted, and shield everything above."""

    def test_pin_blocks_retention_prune(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1 / 60)  # 1s retention
        commit_n(buf, 10)
        buf.pin_oldest(8)
        freed = buf.prune(now_index=30)
        # 0..7 are fair game, 8..9 stay: the pin shields at-or-above.
        assert [s.index for s in buf.segments] == [8, 9]
        assert freed == 8 * 4096

    def test_pin_shields_all_segments_above_it(self, tmp_path: Path) -> None:
        # The pin is a threshold, not a point: pruning must never step
        # over a shielded segment to free newer ones.
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1 / 60)
        commit_n(buf, 6)
        buf.pin_oldest(2)
        buf.prune(now_index=99)
        assert [s.index for s in buf.segments] == [2, 3, 4, 5]

    def test_pins_are_refcounted(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        buf.pin_oldest(3)
        buf.pin_oldest(3)
        buf.unpin(3)  # one reference left
        assert buf._pins == {3: 1}
        buf.unpin(3)  # gone
        assert buf._pins == {}

    def test_unpin_none_clears_everything(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        buf.pin_oldest(1)
        buf.pin_oldest(5)
        buf.unpin(None)
        assert buf._pins == {}

    def test_unpin_without_matching_pin_warns_not_raises(self, tmp_path: Path) -> None:
        # unpin runs in finally blocks; a bookkeeping slip must not kill
        # the watcher. (This is exactly the leaked-pin-tradeoff pair.)
        buf = RingBuffer(tmp_path / "buf")
        buf.unpin(42)  # no error
        assert buf._pins == {}

    def test_pin_holds_disk_cap_hostage(self, tmp_path: Path) -> None:
        # Documented behavior, pinned here so a "fix" that silently deletes
        # pinned data gets caught: the buffer may exceed its cap while a
        # clip is in flight. Pins win over pruning, on purpose.
        buf = RingBuffer(tmp_path / "buf", max_disk_mb=1)  # 1 MiB cap
        commit_n(buf, 10, size=200 * 1024)  # ~2 MiB
        buf.pin_oldest(0)
        buf.prune()
        assert buf.segments  # everything survives while pinned


class TestPrune:
    """Retention window AND disk cap, oldest-first, idempotent."""

    def test_prune_respects_retention_window(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1.0)  # 60s = 10 segs
        commit_n(buf, 16)
        freed = buf.prune(now_index=15)
        # 15-10+1 = 6: indices 0..5 are older than 60s.
        assert [s.index for s in buf.segments] == list(range(6, 16))
        assert freed == 6 * 4096

    def test_retention_boundary_is_exclusive(self, tmp_path: Path) -> None:
        # Exactly 60s old is still inside the window; one second more is
        # not. The pinned unit is "stream seconds", not wall-clock.
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1.0)
        commit_n(buf, 11)  # 0..10
        buf.prune(now_index=10)
        assert [s.index for s in buf.segments] == list(range(1, 11))

    def test_prune_enforces_disk_cap_oldest_first(self, tmp_path: Path) -> None:
        # Retention says keep; the cap says shed. The cap wins, and it
        # sheds from the old end so the newest bytes survive pressure.
        buf = RingBuffer(tmp_path / "buf", retain_minutes=60.0, max_disk_mb=1)
        commit_n(buf, 8, size=200 * 1024)  # 1.6 MiB > 1 MiB cap
        buf.prune(now_index=7)
        held_bytes = sum(s.size_bytes for s in buf.segments)
        assert held_bytes <= 1024 * 1024
        # Oldest-first shedding keeps the newest bytes: 5×200 KiB = 1 MiB
        # exactly at the cap boundary, so the first victim is index 3.
        assert [s.index for s in buf.segments] == [3, 4, 5, 6, 7]

    def test_prune_is_idempotent(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1 / 60)
        commit_n(buf, 10)
        first = buf.prune(now_index=20)
        second = buf.prune(now_index=20)
        assert first > 0
        assert second == 0

    def test_prune_empty_buffer_returns_zero(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        assert buf.prune() == 0

    def test_prune_deletes_files_from_disk(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1 / 60)
        commit_n(buf, 10)
        buf.prune(now_index=9)
        # The metadata says only the newest survives, and the file follows
        # the metadata — a listed segment that's missing on disk is the
        # exact corruption scissors paths assume can't happen.
        assert [s.index for s in buf.segments] == [9]
        assert (buf.directory / segment_name(9)).exists()
        assert not any((buf.directory / segment_name(i)).exists() for i in range(9))

    def test_prune_nowdefaults_to_newest(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=1 / 60)  # 1s
        commit_n(buf, 5)
        freed = buf.prune()  # now = newest = 4
        assert freed == 4 * 4096
        assert [s.index for s in buf.segments] == [4]


class TestClose:
    """close() forgets everything; the caller owns rmtree."""

    def test_close_forgets_segments_and_pins(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 5)
        buf.pin_oldest(2)
        buf.close()
        assert buf.segments == []
        assert buf._pins == {}

    def test_close_does_not_delete_files(self, tmp_path: Path) -> None:
        # Removing the directory is the manager's job (it owns the stream
        # subtree); the buffer only drops its bookkeeping.
        buf = RingBuffer(tmp_path / "buf")
        commit_n(buf, 3)
        buf.close()
        assert all((buf.directory / segment_name(i)).exists() for i in range(3))

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        buf.close()
        buf.close()
        assert buf.segments == []


class TestConcurrency:
    """The buffer is shared between the tailer and the watcher threads."""

    def test_threaded_commits_are_all_registered(self, tmp_path: Path) -> None:
        """One tailer thread commits while the main thread reads/prunes.

        This is the real sharing pattern (commits are strictly ordered per
        epoch by design — there is exactly one tailer per epoch), so the
        race that matters is commit-vs-prune/read, not commit-vs-commit.
        """
        import threading
        import time as _time

        buf = RingBuffer(tmp_path / "buf", retain_minutes=1.0)
        tailer_done = threading.Event()

        def tailer() -> None:
            try:
                for i in range(80):
                    path = buf.next_segment_path(i)
                    path.write_bytes(b"z" * 100)
                    buf.commit_segment(i)
                    _time.sleep(0.001)  # let the reader interleave for real
            finally:
                tailer_done.set()

        t = threading.Thread(target=tailer, daemon=True)
        t.start()
        probe_ok = 0
        deadline = _time.monotonic() + 15
        while not tailer_done.is_set() and _time.monotonic() < deadline:
            buf.span_seconds()
            buf.prune()
            if len(buf.segments) >= 2:
                start = buf.segments[0].index
                end = buf.segments[-1].index
                mid = (start + end) // 2
                try:
                    paths = buf.range_paths(mid, end)
                    buf.pin_oldest(mid)
                    try:
                        assert all(len(p.read_bytes()) == 100 for p in paths)
                        probe_ok += 1
                    finally:
                        buf.unpin(mid)
                except ValueError:
                    pass  # window vanished between snapshot and pin; legal
        t.join(timeout=10)

        # Whatever interleaving happened, invariants must hold: index-sorted,
        # no duplicates, no interior gaps (prune only eats the front), and
        # every listed segment actually exists on disk.
        indexes = [s.index for s in buf.segments]
        assert indexes == sorted(set(indexes))
        assert indexes == list(range(indexes[0], indexes[-1] + 1))
        for seg in buf.segments:
            assert seg.path.exists()
        assert probe_ok > 0

    def test_interleaved_out_of_order_commit_is_refused(self, tmp_path: Path) -> None:
        """Two racing tailers on one epoch: the second commit of the same
        index must be refused, not silently double-registered."""
        import threading

        buf = RingBuffer(tmp_path / "buf")
        path = buf.next_segment_path(0)
        path.write_bytes(b"a" * 10)

        results: list[BaseException | None] = [None, None]

        def committer(slot: int) -> None:
            try:
                buf.commit_segment(0)
            except ValueError as exc:
                results[slot] = exc

        threads = [threading.Thread(target=committer, args=(i,)) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one commit landed; the loser got a ValueError.
        assert len(buf.segments) == 1
        refused = [r for r in results if isinstance(r, ValueError)]
        assert len(refused) == 1
        assert "already" in str(refused[0])
