"""Real per-segment timing: keyframe-cut segments are NOT ``SEGMENT_S`` long.

Live capture stream-copies with ``-segment_time 6``; ffmpeg can only cut at
keyframes, so a 4 s GOP gives 8 s segments (and 4 s ones, and others). Every
consumer used to map index ↔ seconds as ``index * 6``, so clips started and
ended at the wrong moment and the judge's word-offset trims drifted. These
tests feed variable-length fake segments through the REAL buffer, capture
tailer, watcher candidate path, clip factory and audio worker, and assert that
clip boundaries and trims land on the right media time.

No network, no GPU, no real ffmpeg: segment files are marker bytes, durations
come from a hand-written ffmpeg-style CSV segment list.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from autoclip.config import Settings
from autoclip.live import audio_signals, clip_factory, store
from autoclip.live.buffer import (
    SEGMENT_S,
    RingBuffer,
    discard_segment_files,
    segment_index_at,
    segment_list_name,
    segment_name,
)
from autoclip.live.capture import StreamCapture, parse_segment_list_row
from autoclip.live.judge import LiveJudgement
from autoclip.live.models import LiveClipRow, LiveStreamRow
from autoclip.live.paths import clip_dir
from autoclip.live.settings import LiveSettings
from autoclip.live.signals import DetectorSignal, SignalKind, SignalWindow
from autoclip.live.states import StreamState
from autoclip.live.watcher import JUDGE_VERDICT_NAME, ClipWorkItem, StreamWatcher, WatcherDeps
from autoclip.pipeline.reframe.croppath import CropPath
from autoclip.pipeline.transcript import Transcript
from autoclip.pipeline.transcript import Word as TWord

#: What a 4 s-GOP stream really produces under ``-segment_time 6``: every
#: cut lands on the first keyframe at/after the target, i.e. 8 s segments.
GOP8 = 8.0


def seg_bytes(index: int) -> bytes:
    """Unique, recognisable content per segment, so extracts can be checked."""
    return f"<seg{index:03d}>".encode()


def fill(buffer: RingBuffer, durations: list[float], *, start_index: int = 0) -> None:
    """Write + commit one fake segment per duration, in order."""
    for offset, duration in enumerate(durations):
        index = start_index + offset
        buffer.next_segment_path(index).write_bytes(seg_bytes(index))
        buffer.commit_segment(index, duration_s=duration)


def concat(indices: range) -> bytes:
    return b"".join(seg_bytes(i) for i in indices)


# --------------------------------------------------------------------------
# Buffer: the per-segment (start_s, duration_s) index
# --------------------------------------------------------------------------


class TestBufferIndex:
    def test_starts_chain_real_durations(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [8.0, 4.0, 8.0, 4.0, 8.0])
        assert [s.start_s for s in buf.segments] == [0.0, 8.0, 12.0, 20.0, 24.0]
        assert [s.end_s for s in buf.segments] == [8.0, 12.0, 20.0, 24.0, 32.0]
        assert buf.covered_until_s() == pytest.approx(32.0)  # nominal said 30
        assert buf.span_seconds() == pytest.approx(32.0)

    def test_missing_duration_falls_back_to_nominal(self, tmp_path: Path) -> None:
        # Old callers / test doubles commit without a duration: the timeline
        # is then exactly the old index * SEGMENT_S.
        buf = RingBuffer(tmp_path / "buf")
        for i in range(3):
            buf.next_segment_path(i).write_bytes(b"x")
            buf.commit_segment(i)
        assert [s.start_s for s in buf.segments] == [0.0, SEGMENT_S, 2 * SEGMENT_S]
        assert all(s.duration_s == SEGMENT_S for s in buf.segments)

    def test_index_gap_counts_nominally(self, tmp_path: Path) -> None:
        # A skipped (zero-byte) segment leaves an index hole; the clock
        # bridges it with the nominal length and stays monotonic.
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [GOP8])
        fill(buf, [GOP8], start_index=2)
        assert buf.segments[1].start_s == pytest.approx(GOP8 + SEGMENT_S)

    def test_clock_survives_a_full_prune(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=5.0, max_disk_mb=0)
        fill(buf, [GOP8, GOP8])
        buf.prune()  # disk cap 0: everything goes
        assert buf.segments == []
        fill(buf, [GOP8], start_index=2)
        assert buf.segments[0].start_s == pytest.approx(16.0)

    def test_index_at_uses_real_times(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [GOP8] * 5)  # 0..40
        assert buf.index_at(0.0) == 0
        assert buf.index_at(17.0) == 2  # nominal: 17 // 6 == 2 (coincidence)
        assert buf.index_at(31.9) == 3  # nominal: 5 — a segment never written
        assert buf.index_at(40.0) == 5  # past the end: next index
        assert segment_index_at(buf, 31.9) == 3
        # Doubles without index_at keep the nominal mapping.
        assert segment_index_at(object(), 31.9) == 5

    def test_extract_window_selects_by_real_time_and_reports_media_start(
        self, tmp_path: Path
    ) -> None:
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [GOP8] * 5)  # 0-8, 8-16, 16-24, 24-32, 32-40
        window = buf.extract_window(26.0, 30.0, tmp_path / "o.ts")
        # Nominal mapping picked index 4 (= real 32..40s: the wrong moment).
        assert (window.start_index, window.end_index) == (3, 3)
        assert window.start_s == pytest.approx(24.0)
        assert window.end_s == pytest.approx(32.0)
        assert window.path.read_bytes() == seg_bytes(3)

    def test_extract_window_spanning_segments(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [8.0, 4.0, 8.0, 4.0, 8.0])  # 0-8, 8-12, 12-20, 20-24, 24-32
        window = buf.extract_window(10.0, 21.0, tmp_path / "o.ts")
        assert (window.start_index, window.end_index) == (1, 3)
        assert (window.start_s, window.end_s) == (8.0, 24.0)
        assert window.path.read_bytes() == concat(range(1, 4))
        # A window ending exactly on a boundary excludes the next segment.
        window = buf.extract_window(12.0, 20.0, tmp_path / "p.ts")
        assert (window.start_index, window.end_index) == (2, 2)

    def test_extract_beyond_real_end_raises(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [4.0, 4.0])  # covers 0..8; nominal believed 0..12
        with pytest.raises(ValueError, match="ahead"):
            buf.extract(9.0, 12.0, tmp_path / "o.ts")

    def test_anchor_clamp_uses_real_times(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [GOP8] * 3)  # epoch 0: 0..24
        buf.begin_epoch()
        fill(buf, [GOP8] * 3, start_index=3)  # epoch 1: 24..48
        # Anchor 20 s is in epoch 0 by real time (nominal: index 3, epoch 1).
        window = buf.extract_window(0.0, 48.0, tmp_path / "o.ts", anchor_s=20.0)
        assert (window.start_index, window.end_index) == (0, 2)

    def test_prune_ages_by_real_time(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf", retain_minutes=0.5)  # 30 s
        fill(buf, [GOP8] * 10)  # starts 0, 8, ..., 72
        buf.prune()
        # Keep segments that started < 30 s before the newest start (72):
        # 48, 56, 64, 72 → indices 6..9. Nominal ages kept 5..9 (stale 40 s).
        assert [s.index for s in buf.segments] == [6, 7, 8, 9]
        assert buf.clipped_seconds() == pytest.approx(6 * GOP8)

    def test_indices_back_steps_over_real_durations(self, tmp_path: Path) -> None:
        buf = RingBuffer(tmp_path / "buf")
        fill(buf, [GOP8] * 5)
        assert buf.indices_back(0.0) == 4
        assert buf.indices_back(7.9) == 4  # one 8 s segment does not fit
        assert buf.indices_back(16.0) == 2

    def test_discard_also_clears_stale_segment_lists(self, tmp_path: Path) -> None:
        directory = tmp_path / "buf"
        directory.mkdir()
        (directory / segment_name(0)).write_bytes(b"x")
        (directory / segment_list_name(0)).write_text("seg_000000.ts,0,8\n", encoding="utf-8")
        (directory / "notes.txt").write_text("keep", encoding="utf-8")
        assert discard_segment_files(directory) == 2
        assert [p.name for p in directory.iterdir()] == ["notes.txt"]


# --------------------------------------------------------------------------
# Capture: ffmpeg's CSV segment list → real durations on commit
# --------------------------------------------------------------------------


class _AliveProc:
    """Minimal Popen stand-in that stays alive until killed."""

    pid = 4243
    stderr = None

    def __init__(self) -> None:
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = time.monotonic() + (timeout or 30.0)
        while self.returncode is None:
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired("fake", timeout)
            time.sleep(0.01)
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


@pytest.fixture
def _no_tree_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autoclip.live.capture.kill_process_tree", lambda pid: None)


class TestSegmentList:
    def test_row_parsing(self) -> None:
        assert parse_segment_list_row("seg_000012.ts,48.066667,56.066667") == (
            12,
            pytest.approx(8.0),
        )
        # Times are ffmpeg's input timeline (not zero-based): only the
        # difference matters.
        assert parse_segment_list_row("seg_000001.ts,10000.5,10004.5\r") == (1, 4.0)
        assert parse_segment_list_row("C:/buf/seg_000002.ts,0,8") == (2, 8.0)
        for bad in ("", "seg_000001.ts,1", "notes.txt,0,8", "seg_000001.ts,8,8", "x,y,z"):
            assert parse_segment_list_row(bad) is None

    def test_commits_carry_list_durations(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        cap = StreamCapture("https://twitch.tv/x", buffer, resolver=lambda u: "http://x")
        cap.segment_list_path = buffer.directory / segment_list_name(0)
        for i in range(4):
            (buffer.directory / segment_name(i)).write_bytes(seg_bytes(i))
        # Rows 0..2 complete; row 3 is half-flushed (no newline yet): it must
        # not be trusted ("...,2" would read as a bogus end time).
        cap.segment_list_path.write_bytes(
            b"seg_000000.ts,1.4,9.4\nseg_000001.ts,9.4,13.4\nseg_000002.ts,13.4,21.4\nseg_00000"
        )
        assert cap._poll_once() is True
        assert [(s.index, s.start_s, s.duration_s) for s in buffer.segments] == [
            (0, 0.0, pytest.approx(8.0)),
            (1, pytest.approx(8.0), pytest.approx(4.0)),
            (2, pytest.approx(12.0), pytest.approx(8.0)),
        ]
        # The rest of row 3 lands; ffmpeg exits → the last segment commits
        # with its real duration too.
        with open(cap.segment_list_path, "ab") as handle:
            handle.write(b"3.ts,21.4,25.4\n")
        assert cap._poll_once(final=True) is True
        assert buffer.segments[-1].start_s == pytest.approx(20.0)
        assert buffer.segments[-1].duration_s == pytest.approx(4.0)

    def test_missing_row_falls_back_to_probe_then_nominal(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        probed: list[str] = []

        def probe(path: Path) -> float | None:
            probed.append(path.name)
            return 7.5 if path.name == segment_name(0) else None

        cap = StreamCapture(
            "https://twitch.tv/x", buffer, resolver=lambda u: "http://x", duration_probe=probe
        )
        cap.segment_list_path = buffer.directory / segment_list_name(0)  # never written
        for i in range(2):
            (buffer.directory / segment_name(i)).write_bytes(seg_bytes(i))
        cap._poll_once(final=True)
        assert probed == [segment_name(0), segment_name(1)]
        assert [s.duration_s for s in buffer.segments] == [7.5, SEGMENT_S]
        assert buffer.segments[1].start_s == pytest.approx(7.5)

    def test_without_a_list_no_probe_runs(self, tmp_path: Path) -> None:
        buffer = RingBuffer(tmp_path / "buf")

        def probe(path: Path) -> float:
            raise AssertionError("no list → no ffprobe")

        cap = StreamCapture("u", buffer, resolver=lambda u: "x", duration_probe=probe)
        (buffer.directory / segment_name(0)).write_bytes(b"x")
        cap._poll_once(final=True)
        assert buffer.segments[0].duration_s == SEGMENT_S

    def test_spawned_capture_reads_the_list_it_was_given(
        self, tmp_path: Path, _no_tree_kill: None
    ) -> None:
        # The list path is set BEFORE spawning, so an injected spawn (here:
        # a fake ffmpeg) can write one; the tailer picks the rows up live.
        buffer = RingBuffer(tmp_path / "buf")
        holder: dict[str, StreamCapture] = {}
        durations = [8.0, 4.0, 8.0, 4.0]

        def spawn(url: str) -> _AliveProc:
            cap = holder["cap"]
            assert cap.segment_list_path == buffer.directory / segment_list_name(0)

            def write() -> None:
                t = 100.0
                rows = []
                for i, d in enumerate(durations):
                    if i:  # the muxer lists N as it closes N, BEFORE opening N+1
                        cap.segment_list_path.write_text("".join(rows), encoding="utf-8")
                    (buffer.directory / segment_name(i)).write_bytes(seg_bytes(i))
                    rows.append(f"{segment_name(i)},{t:.3f},{t + d:.3f}\n")
                    t += d
                    time.sleep(0.05)

            threading.Thread(target=write, daemon=True).start()
            return _AliveProc()

        cap = StreamCapture(
            "https://twitch.tv/x",
            buffer,
            resolver=lambda u: "http://x",
            spawn=spawn,
            poll_s=0.02,
            drain_grace_s=0.1,
        )
        holder["cap"] = cap
        try:
            cap.start()
            deadline = time.monotonic() + 5.0
            while len(buffer.segments) < 3 and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            cap.stop()
        assert [s.duration_s for s in buffer.segments[:3]] == [8.0, 4.0, 8.0]
        assert [s.start_s for s in buffer.segments[:3]] == [0.0, 8.0, 12.0]
        # stop() removes the list: its durations now live on the segments.
        assert not (buffer.directory / segment_list_name(0)).exists()

    def test_argv_asks_ffmpeg_for_a_live_csv_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _no_tree_kill: None
    ) -> None:
        argv: list[str] = []
        monkeypatch.setattr("autoclip.live.capture._ffmpeg_binary", lambda: "ffmpeg")
        monkeypatch.setattr("autoclip.live.capture._new_process_group_flags", lambda: 0)
        monkeypatch.setattr(
            subprocess, "Popen", lambda command, **kwargs: argv.extend(command) or _AliveProc()
        )
        buffer = RingBuffer(tmp_path / "buf")
        fill(buffer, [GOP8] * 3)
        cap = StreamCapture("u", buffer, resolver=lambda u: "http://media/x.m3u8")
        cap.start()
        cap.stop()
        # A respawn continues at index 3 and gets its own list.
        listed = argv[argv.index("-segment_list") + 1]
        assert Path(listed) == buffer.directory / segment_list_name(3)
        assert argv[argv.index("-segment_list_type") + 1] == "csv"
        assert argv[argv.index("-segment_list_flags") + 1] == "+live"
        assert argv.index("-segment_list") < len(argv) - 1  # before the output


# --------------------------------------------------------------------------
# Watcher: judge window media start → trims on the right stream time
# --------------------------------------------------------------------------


@dataclass
class _Word:
    text: str
    start: float
    end: float


@dataclass
class _Transcript:
    words: list = field(default_factory=list)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path


class _Capture:
    def __init__(self, buffer: Any, on_event: Any) -> None:
        self.buffer = buffer
        self.on_event = on_event

    def start(self) -> None:
        self.on_event("started", "fake capture started")

    def stop(self) -> None:
        pass


class _Aggregator:
    def __init__(self) -> None:
        self.signals: list[DetectorSignal] = []

    def add(self, signal: DetectorSignal) -> None:
        self.signals.append(signal)

    def evaluate(self, window_s: float = 30.0) -> SignalWindow:
        end = max((s.t_s for s in self.signals), default=0.0)
        return SignalWindow(start_s=max(0.0, end - window_s), end_s=end, signals=list(self.signals))


@dataclass
class _Fire:
    fire: bool = True


class _Yes:
    def decide(self, window: Any) -> _Fire:
        return _Fire()

    def can_trigger(self, *a: Any, **k: Any) -> bool:
        return True

    def mark_clip(self, *a: Any, **k: Any) -> None:
        pass

    def try_acquire(self, n: float = 1.0) -> bool:
        return True


async def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"condition not met within {timeout}s")


async def test_judge_trim_and_staged_window_land_on_real_media_time(
    initialised_db: int, tmp_path: Path
) -> None:
    store.upsert_stream(LiveStreamRow(id="s-timing", url="url", label=""))
    buffer = RingBuffer(tmp_path / "buf")
    fill(buffer, [GOP8] * 20)  # 0..160 s in real 8 s segments
    enqueued: list[ClipWorkItem] = []
    transcribed: list[Path] = []

    def transcribe(path: Path) -> _Transcript:
        transcribed.append(path)
        # FILE-relative word times, as Whisper returns them.
        return _Transcript(words=[_Word("setup", 14.0, 15.0), _Word("payoff", 37.0, 38.0)])

    async def judge(label: str, words: list, signals: dict) -> LiveJudgement:
        return LiveJudgement(score=95, title="T", start_word_offset=0, end_word_offset=1)

    yes = _Yes()
    deps = WatcherDeps(
        capture_factory=_Capture,
        buffer_factory=lambda: buffer,
        aggregator_factory=_Aggregator,
        trigger=yes,
        cooldown=yes,
        judge_minute=yes,
        judge_hour=yes,
        judge_day=yes,
        transcribe=transcribe,
        audio_extract=lambda source, dest: dest.write_bytes(source.read_bytes()),
        audio_start=lambda *a, **k: None,
        judge=judge,
        enqueue_clip=lambda item: enqueued.append(item) or True,
        may_capture=lambda: True,
        publish=lambda *a, **k: None,
        chat_runner=lambda *a, **k: False,
        coverage_timeout_s=0.0,
    )
    live = LiveSettings()
    live.rolling_buffer.judge_context_s = 30.0
    watcher = StreamWatcher("s-timing", "url", "", live, deps=deps)
    task = asyncio.create_task(watcher.run())
    try:
        await _wait_for(lambda: watcher.state is StreamState.WATCHING)
        watcher.submit_signal(DetectorSignal(kind=SignalKind.AUDIO_ENERGY, t_s=120.0, strength=1.0))
        await _wait_for(lambda: len(enqueued) == 1)
    finally:
        watcher.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    item = enqueued[0]
    # Judge window asked 90..130 s; whole segments copied → the file starts
    # at segment 11's real start, 88 s (nominal believed 90). Words at file
    # 14 s / 38 s are stream 102 s / 126 s — NOT 104 / 128.
    assert (item.clip_start_s, item.clip_end_s) == (pytest.approx(102.0), pytest.approx(126.0))
    work = clip_dir(item.clip_id)
    verdict = json.loads((work / JUDGE_VERDICT_NAME).read_text(encoding="utf-8"))
    assert verdict["judge_media_start_s"] == pytest.approx(88.0)
    # The staged render window is segments 12..15 (96..128 s) — the ones
    # really holding 102..126 s (nominal picked 17..21, i.e. 136..168 s) —
    # and the manifest records where that file starts.
    assert item.source_path.read_bytes() == concat(range(12, 16))
    manifest = json.loads((work / clip_factory.WINDOW_NAME).read_text(encoding="utf-8"))
    assert manifest["media_start_s"] == pytest.approx(96.0)
    row = store.get_clip(item.clip_id)
    assert (row.clip_start_s, row.clip_end_s) == (pytest.approx(102.0), pytest.approx(126.0))


# --------------------------------------------------------------------------
# Clip factory: the render cuts the clip window out of the extract
# --------------------------------------------------------------------------


@dataclass
class _Info:
    duration_s: float
    has_video: bool = True
    has_audio: bool = True


class _RecordingBuffer(RingBuffer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pins: list[int] = []

    def pin_oldest(self, index: int) -> None:
        self.pins.append(index)
        super().pin_oldest(index)


def _stub_render(monkeypatch: pytest.MonkeyPatch, duration_s: float) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def extract_audio(source, destination, *, duration_s=None, on_progress=None):
        destination.write_bytes(b"RIFF")
        return destination

    def transcribe(audio, settings, *, duration_s=None, on_progress=None, **kwargs):
        return Transcript(words=[TWord(text="hi", start=7.0, end=7.5)])

    def build_crop_path(video, *, start_s, end_s, transcript=None, config=None):
        seen["crop"] = (start_s, end_s)
        return CropPath(source_width=1280, source_height=720, segments=[])

    def export_clip(request, *, work_dir, settings=None, on_progress=None, **kwargs):
        seen["export"] = (request.start_s, request.end_s)
        request.destination.write_bytes(b"MP4")
        request.destination.with_suffix(".srt").write_text("1\n", encoding="utf-8")
        return request.destination

    def remux(raw: Path, destination: Path, clip_id: str) -> None:
        destination.write_bytes(raw.read_bytes())

    monkeypatch.setattr(clip_factory.prepare, "extract_audio", extract_audio)
    monkeypatch.setattr(clip_factory.transcribe, "transcribe", transcribe)
    monkeypatch.setattr(clip_factory, "build_crop_path", build_crop_path)
    monkeypatch.setattr(clip_factory.export, "export_clip", export_clip)
    monkeypatch.setattr(clip_factory, "_remux", remux)
    monkeypatch.setattr(clip_factory, "_probe_reusable", lambda path: None)
    monkeypatch.setattr(clip_factory.ffmpeg, "probe", lambda path: _Info(duration_s))
    return seen


def _row(clip_id: str, start: float, end: float) -> LiveClipRow:
    return LiveClipRow(
        id=clip_id,
        stream_id="s1",
        trigger_start_s=110.0,
        trigger_end_s=120.0,
        clip_start_s=start,
        clip_end_s=end,
        title="Moment",
        caption_style="bold_pop",
        ratio="9:16",
    )


class TestRenderCut:
    def test_produce_cuts_the_window_at_its_real_offset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        buffer = _RecordingBuffer(tmp_path / "buf")
        fill(buffer, [GOP8] * 20)
        seen = _stub_render(monkeypatch, duration_s=32.0)  # segments 12..15
        clip_factory.produce(buffer, _row("c-cut", 102.0, 126.0), Settings())
        assert (clip_dir("c-cut") / clip_factory.RAW_NAME).read_bytes() == concat(range(12, 16))
        # 102..126 s inside a file starting at 96 s → file 6..30 s.
        assert seen["export"] == (pytest.approx(6.0), pytest.approx(30.0))
        assert seen["crop"] == (pytest.approx(6.0), pytest.approx(30.0))
        assert buffer.pins[0] == 12  # nominal 102 // 6 == 17 under-pinned

    def test_staged_window_renders_without_the_buffer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        buffer = RingBuffer(tmp_path / "buf")
        fill(buffer, [8.0, 4.0] * 10)  # alternating keyframe cuts
        clip_factory.stage_window(buffer, "c-staged", 25.0, 49.0, anchor_s=40.0)
        # 25 s is in segment 4 (24..32); 49 s in segment 8 (48..56).
        raw = clip_dir("c-staged") / clip_factory.RAW_NAME
        assert raw.read_bytes() == concat(range(4, 9))
        seen = _stub_render(monkeypatch, duration_s=32.0)
        clip_factory.produce(None, _row("c-staged", 25.0, 49.0), Settings())  # type: ignore[arg-type]
        assert seen["export"] == (pytest.approx(1.0), pytest.approx(25.0))

    def test_manifest_without_media_start_renders_the_whole_extract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Staged before the index existed (or by a buffer that can't say).
        work = clip_dir("c-old")
        work.mkdir(parents=True, exist_ok=True)
        (work / clip_factory.RAW_NAME).write_bytes(b"old-raw")
        clip_factory.write_window_manifest(work, 102.0, 126.0)
        seen = _stub_render(monkeypatch, duration_s=30.0)
        clip_factory.produce(None, _row("c-old", 102.0, 126.0), Settings())  # type: ignore[arg-type]
        assert seen["export"] == (0.0, 30.0)


# --------------------------------------------------------------------------
# Audio: per-second signal times follow the real segment starts
# --------------------------------------------------------------------------


def test_audio_seconds_are_stamped_from_real_segment_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    buffer = RingBuffer(tmp_path / "buf")
    fill(buffer, [8.0, 4.0, 8.0])
    pcm_seconds = {0: 8, 1: 4, 2: 8}

    def decode(path: Path, *, stop_event: Any = None) -> np.ndarray:
        index = int(path.name[4:10])
        return np.zeros(pcm_seconds[index] * audio_signals.SAMPLE_RATE, dtype=np.float32)

    monkeypatch.setattr(audio_signals, "decode_segment_pcm", decode)
    analyzer = audio_signals.AudioSignalAnalyzer(LiveSettings(), lambda signal: None)
    stamps: list[float] = []
    original = analyzer.process_second

    def record(t_s: float, rms: float) -> Any:
        stamps.append(t_s)
        return original(t_s, rms)

    monkeypatch.setattr(analyzer, "process_second", record)
    audio_signals.process_pass(buffer, analyzer, None, None)
    # Contiguous 0..19 s; nominal stamping restarted at 6 s and 12 s,
    # overlapping/gapping the timeline the detector windows read.
    assert stamps == [float(t) for t in range(20)]
