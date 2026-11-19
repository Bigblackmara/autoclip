"""Live stream capture: resolve a URL, then ffmpeg writes rolling .ts segments.

Pipeline shape (and why it is two subprocesses at most, never a pipe chain):

1. **Resolve, raising.** ``start()`` resolves the page URL (twitch.tv/...,
   youtube.com/...) to a playable stream URL *synchronously* and raises
   :class:`CaptureError` if nothing is playable. The watcher (A8) runs
   ``start()`` in a worker thread and classifies the exception: a not-live
   channel must read as "No playable streams" (or carry ``.offline``) so it
   parks in ``waiting_for_live`` instead of burning the reconnect budget.
   streamlink's Python API is preferred (already a hard dep of the [live]
   extra, and its plugin set is what makes odd stream URLs work at all);
   yt-dlp is the fallback. The resolver is a *seam* — injectable and
   monkeypatchable — so unit tests never touch the network.

2. **Copy.** One long-lived ffmpeg reads the resolved URL and writes
   ``seg_{index:06d}.ts`` files into the buffer directory with the ``segment``
   muxer (``-f segment``, NOT the HLS muxer — the work order pins this) at
   ``-segment_time 6`` (one continuous clock), codecs copied. A pipe chain
   (streamlink | ffmpeg) would tie ffmpeg's lifetime to a third party's
   socket; two decoupled processes mean a reconnect restarts only resolution.

   ``-segment_time`` is only a TARGET: with ``-c copy`` the muxer can cut at
   keyframes only, so a 4 s GOP yields 8 s, 4 s, 8 s ... segments. ffmpeg
   therefore also writes a CSV segment list (``-segment_list seglist_<n>.csv
   -segment_list_type csv -segment_list_flags +live``: one
   ``name,start,end`` row appended as each segment closes — verified on the
   ffmpeg 9 build), and the tailer hands each segment's real duration to the
   buffer, whose per-segment time index every consumer reads. A missing row
   (a killed ffmpeg's last segment) falls back to an ffprobe of the file,
   then to the nominal length.

3. **Tail.** A daemon thread commits segments strictly in sequence, starting
   at the number this ffmpeg was told to start at: segment N is committed
   only once segment N+1 exists (the segment muxer closes N before it opens
   N+1), or once ffmpeg has exited (then every file is closed). A
   size-stable-across-two-polls rule is NOT enough: live HLS arrives in
   bursts, so a half-written segment holds still between bursts and was
   committed at ~25 %, starving audio analysis. The size is re-read at
   commit time. Files outside the expected sequence (stale leftovers, a
   previous ffmpeg's late write) are never committed.

   The tailer is also the stall watchdog: when ffmpeg is alive but no
   segment has been committed for ≈3 segments, the process is killed and an
   ``"error"`` event fires, so the watcher reconnects instead of sitting on
   a frozen capture forever. "3 segments" means 3 × the longest RECENT real
   segment (never less than ``stall_timeout_s``, at most
   ``STALL_CEILING_S``): a stream with a 20 s keyframe interval only closes
   a segment every ~20 s, and a fixed 18 s timeout would restart it in a
   loop.

Events go through ``on_event(kind, detail)`` with a *string* detail (the
watcher marshals them onto its event loop and logs the text): ``"started"``
once ffmpeg is up (informational), ``"ended"`` when ffmpeg exits cleanly on
its own (stream over — distinct from a user-initiated stop), and ``"error"``
if ffmpeg dies mid-run. Callback failures are swallowed and logged: a
crashing consumer must never take a live capture down with it.

Cancellation must survive Windows: killing a child does not kill its
children here, so ``stop()`` kills the whole tree via ``taskkill /PID x /T
/F`` (POSIX hosts signal the dedicated process group instead). ``stop()`` is
idempotent and safe from any thread; an orphaned ffmpeg outliving a removed
stream is the specific failure this design exists to prevent.

Verified against the real ffmpeg 9 build: the argv below turns an HLS source
into numbered mpegts segments, byte-identical with or without
``-movflags +faststart`` (a no-op on mpegts, kept because the frozen work
order pins it). Nothing here is speculative.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .buffer import (
    SEGMENT_EXT,
    SEGMENT_S,
    RingBuffer,
    adaptive_timeout_s,
    segment_list_name,
    segment_name,
)

log = logging.getLogger(__name__)

#: ``-segment_time`` asked of ffmpeg — a nominal target (real segments are
#: keyframe-cut, see the module docstring). Re-exported from the buffer so the
#: fallback length a segment gets without a list row matches what was asked.
SEGMENT_TIME_S = SEGMENT_S

#: Sentinel: "pick the duration probe for me" (real ffprobe for a real
#: ffmpeg spawn, none for an injected spawn — test doubles write no media).
_AUTO = object()

#: Tailer poll interval. One interval of commit latency is harmless
#: (downstream detectors see the segment a moment later); a premature commit
#: is not harmless — so the loop errs slow, not fast.
TAILER_POLL_S = 0.5

#: ffmpeg ``-rw_timeout`` for the input, seconds: a read that blocks longer
#: than this fails the input instead of hanging the process forever.
RW_TIMEOUT_S = 15.0

#: Stall watchdog: ffmpeg alive but no segment committed for ≈3 segments
#: means a frozen capture — kill it so the watcher reconnects. This is the
#: BASE (3 nominal segments); the live limit is STALL_SEGMENTS × the longest
#: recent real segment when that is longer, capped at STALL_CEILING_S.
STALL_TIMEOUT_S = 3 * SEGMENT_S
#: Segments of silence that count as a stall (scales with real lengths).
STALL_SEGMENTS = 3.0
#: Upper bound of the adaptive stall limit: even a long-GOP stream (or one
#: freak segment after a source hiccup) cannot hide a dead capture longer.
STALL_CEILING_S = 90.0

#: The first segment gets a longer grace: resolution, HLS startup and the
#: first full segment all have to happen before anything can commit.
FIRST_SEGMENT_TIMEOUT_S = 45.0

#: How much of ffmpeg's stderr is kept for error events (last lines).
STDERR_TAIL_LINES = 20

#: After ffmpeg exits, the tailer keeps polling this long for stragglers: the
#: final segment can appear in a directory listing a beat after the process
#: is gone, and dropping it would silently lose the end of the stream.
DRAIN_GRACE_S = 3.0

#: Upper bound for one resolver attempt before giving up. Resolution is a
#: couple of HTTP round-trips; anything past this is a hung socket, and a
#: hung watcher slot is worse than a clean "could not resolve" error.
RESOLVE_TIMEOUT_S = 45.0

#: streamlink qualities tried, in order, before falling back to manifests.
_BEST_QUALITIES = ("best", "live", "worst")


class CaptureError(RuntimeError):
    """Capture failed in a way worth explaining to the user.

    Mirrors IngestError/ProviderError: a ``hint`` rides along and the UI and
    CLI render it, so resolution failures tell the user what to do (check the
    channel is live, update plugins) instead of dumping stack traces.

    ``offline`` marks the "channel exists but is not live" case. The watcher
    also sniffs ``str(exc)`` for "No playable streams" (streamlink's exact
    words) — both signals route a not-live channel to waiting_for_live
    instead of counting it as a connect failure.
    """

    def __init__(self, message: str, *, hint: str = "", offline: bool = False) -> None:
        super().__init__(message)
        self.hint = hint
        self.offline = offline

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n\n{self.hint}" if self.hint else base


class StreamCapture:
    """One live watch: resolver + ffmpeg + tailer writing into a RingBuffer.

    Contract with the watcher (A8, frozen in watcher.py's docstring):

    * ``start()`` blocks on resolution and raises :class:`CaptureError` when
      the URL is not playable — the watcher runs it in a thread and routes
      the exception into its offline/backoff classification. That is why
      resolution is *not* pushed onto a background thread here.
    * ``on_event(kind, detail)`` fires with ``kind`` in
      ``{"started", "ended", "error"}`` and the string detail the watcher
      logs. "ended" means the stream finished on its own; user-initiated
      ``stop()`` does not emit it (the watcher already knows it asked).
    * ``stop()`` kills the ffmpeg tree and joins the threads; calling it
      twice, or from inside an event callback, is safe.
    * Every committed segment is valid input for buffer.extract() — the
      tailer only commits files it has seen closed.
    """

    def __init__(
        self,
        url: str,
        buffer: RingBuffer,
        settings=None,
        *,
        on_event: Callable[[str, str], None] | None = None,
        resolver: Callable[[str], str] | None = None,
        spawn: Callable[[str], subprocess.Popen] | None = None,
        poll_s: float = TAILER_POLL_S,
        drain_grace_s: float = DRAIN_GRACE_S,
        stall_timeout_s: float | None = STALL_TIMEOUT_S,
        stall_segments: float = STALL_SEGMENTS,
        stall_ceiling_s: float = STALL_CEILING_S,
        first_segment_timeout_s: float | None = FIRST_SEGMENT_TIMEOUT_S,
        duration_probe: Callable[[Path], float | None] | None | object = _AUTO,
    ) -> None:
        self.url = url
        self.buffer = buffer
        self.settings = settings  # kept for future tuning; argv is fixed
        self.on_event = on_event
        self.resolver = resolver
        self._custom_spawn = spawn
        self.poll_s = poll_s
        self.drain_grace_s = drain_grace_s
        self.stall_timeout_s = stall_timeout_s
        self.stall_segments = stall_segments
        self.stall_ceiling_s = stall_ceiling_s
        self.first_segment_timeout_s = first_segment_timeout_s
        #: Fallback for a segment with no list row: file → seconds (or None).
        self._duration_probe: Callable[[Path], float | None] | None = (
            (probe_segment_duration if spawn is None else None)
            if duration_probe is _AUTO
            else duration_probe  # type: ignore[assignment]
        )
        #: ffmpeg's CSV segment list for the current spawn (set just before
        #: spawning, so an injected spawn can write one too). None = no list.
        self.segment_list_path: Path | None = None
        #: index -> real duration parsed from the list, until committed.
        self._list_durations: dict[int, float] = {}
        #: Bytes of the list already parsed, and an unterminated last line.
        self._list_offset = 0
        self._list_partial = b""

        self._proc: subprocess.Popen | None = None
        self._tailer: threading.Thread | None = None
        self._waiter: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._started = False
        self._stop_done = threading.Event()
        #: Serializes spawn against stop(): a stop() landing while Popen runs
        #: must still see (and kill) the process it creates.
        self._lock = threading.Lock()
        #: Next segment index the tailer expects (strict sequence).
        self._next_index = 0
        #: Segments this capture committed (the watcher's health signal).
        self.committed = 0
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        #: Set when the stall watchdog killed ffmpeg; the exit event says so.
        self._stall_detail = ""

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Resolve, spawn ffmpeg, start the tailer. Raises on dead URLs.

        Resolution + spawn run inline (blocking) because the watcher wraps
        this call in a worker thread and wants the failure *as an exception*
        to classify; only tailing and the ffmpeg-exit watch are threads here.
        Every thread spawned is a daemon, so a crashed host process can
        never leak one.
        """
        if self._started:
            raise RuntimeError("capture already started")
        stream_url = self._resolve()
        with self._lock:
            # Re-check the stop flag after resolution: a stop() racing a slow
            # resolver must not spawn a process nobody will ever manage.
            if self._stop_flag.is_set():
                return
            self._next_index = self._next_start_number()
            # Per-spawn name: a respawn never reads (or truncates under the
            # reader) the previous ffmpeg's list.
            self.segment_list_path = self.buffer.directory / segment_list_name(self._next_index)
            self._spawn_ffmpeg(stream_url)
            if self._stop_flag.is_set():
                # stop() landed while Popen ran: it is waiting on this lock
                # and saw no process yet, so the kill is ours to do.
                self._kill_tree()
                return
            self._started = True
            self._start_stderr_reader()
            self._tailer = threading.Thread(
                target=self._tail_loop, name="capture-tailer", daemon=True
            )
            self._tailer.start()
            self._waiter = threading.Thread(
                target=self._wait_ffmpeg_exit, name="capture-exit", daemon=True
            )
            self._waiter.start()
        self._emit("started", f"capture started: {stream_url}")

    def stop(self) -> None:
        """Kill ffmpeg (whole tree) and join the threads. Idempotent."""
        if self._stop_done.is_set():
            return
        self._stop_flag.set()
        with self._lock:
            self._kill_tree()
        current = threading.current_thread()
        # Bounded join: the tailer polls at most one more interval and the
        # waiter exits with the process it watches; anything past this is a
        # leak the caller should see, not wait on silently.
        join_timeout = self.drain_grace_s + max(self.poll_s, 1.0)
        for t in (self._tailer, self._waiter, self._stderr_reader):
            if t is not None and t is not current:
                t.join(timeout=join_timeout)
        self._drop_segment_list()
        self._stop_done.set()

    @property
    def stderr_tail(self) -> str:
        """The last lines ffmpeg wrote to stderr (for error reports)."""
        return " | ".join(line for line in self._stderr_tail if line)

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until ffmpeg exits; its returncode, or None on timeout."""
        proc = self._proc
        if proc is None:
            return None
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    @property
    def running(self) -> bool:
        """Whether ffmpeg is alive and copying right now."""
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def stopped(self) -> bool:
        """Whether stop() fully completed (threads joined, tree killed)."""
        return self._stop_done.is_set()

    # -- internals: resolve + spawn ----------------------------------------

    def _resolve(self) -> str:
        """Resolve self.url to a playable URL via the seam or real resolvers."""
        if self.resolver is not None:
            url = self.resolver(self.url)
            if not url:
                # The watcher sniffs this text for "no playable streams" —
                # keep the phrase so offline classification still works when
                # a test or custom resolver fakes the dead-channel case.
                raise CaptureError(
                    f"No playable streams found for {self.url}",
                    hint=_offline_hint(self.url),
                    offline=True,
                )
            return url
        return resolve_stream_url(self.url)

    def _spawn_ffmpeg(self, stream_url: str) -> None:
        """Start the ffmpeg copy process writing numbered .ts segments."""
        if self._custom_spawn is not None:
            self._proc = self._custom_spawn(stream_url)
            return
        start_number = self._next_index
        command = [
            _ffmpeg_binary(),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            # A read that blocks this long fails the input instead of
            # freezing ffmpeg (and with it the whole stream) forever.
            *_input_args(stream_url),
            "-c",
            "copy",
            *_map_args(stream_url),
            "-f",
            "segment",
            "-segment_time",
            str(int(SEGMENT_TIME_S)),
            # No -reset_timestamps: segments keep one continuous clock, so a
            # byte-concatenated window plays back exactly. Resetting every
            # segment made each join a timestamp jump that ffmpeg "repaired"
            # per stream, and the audio drifted ~10 ms later per join (measured
            # +119 ms after 55 s) — the voice delay in live clips.
            "-segment_start_number",
            str(start_number),
            # Real per-segment start/end times (stream copy cuts at
            # keyframes, so segments are NOT segment_time long). CSV rows
            # are appended as each segment closes; +live keeps it streaming.
            "-segment_list",
            str(self.segment_list_path or self.buffer.directory / segment_list_name(start_number)),
            "-segment_list_type",
            "csv",
            "-segment_list_flags",
            "+live",
            # Harmless on mpegts (verified byte-identical output) and pinned
            # by the frozen work order; left in for argv parity with the doc.
            "-movflags",
            "+faststart",
            str(self.buffer.directory / f"seg_%06d{SEGMENT_EXT}"),
        ]
        log.debug("spawning ffmpeg: %s", " ".join(command[1:]))
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=_new_process_group_flags(),
        )
        # stop() kills the tree on every orderly exit; the job covers the
        # disorderly ones (crash, console closed, killed from Task Manager).
        bind_to_host_lifetime(self._proc)

    def _next_start_number(self) -> int:
        """Segment number ffmpeg starts numbering at.

        Indices are global to the watch (index → seconds must stay monotonic
        across reconnects), so a respawn continues past the last committed
        index. An empty buffer starts at 0.
        """
        with self.buffer._lock:
            if not self.buffer.segments:
                return 0
            return self.buffer.segments[-1].index + 1

    def _wait_ffmpeg_exit(self) -> None:
        """Watch ffmpeg; emit ended/error from its exit status."""
        proc = self._proc
        assert proc is not None
        rc = proc.wait()
        if self._stop_flag.is_set():
            # User-initiated stop: the watcher already knows; an "ended" here
            # would make a stream removal look like the broadcaster signing
            # off (and the watcher would park it in waiting_for_live).
            return
        if self._stall_detail:
            self._emit("error", self._stall_detail)
        elif rc == 0:
            self._emit("ended", "stream ended")
        else:
            reader = self._stderr_reader
            if reader is not None:
                # Let the reader collect ffmpeg's last words (bounded).
                reader.join(timeout=1.0)
            tail = self.stderr_tail
            detail = f"ffmpeg exited with code {rc}"
            self._emit("error", f"{detail}: {tail[-300:]}" if tail else detail)

    def _start_stderr_reader(self) -> None:
        """Drain ffmpeg's stderr PIPE (keep a tail for error events).

        An unread PIPE fills its 4-8 KB OS buffer and then BLOCKS ffmpeg on
        its next log line — a silent freeze. The reader owns the pipe and
        closes it at EOF.
        """
        stream = getattr(self._proc, "stderr", None)
        if stream is None:
            return

        def drain() -> None:
            try:
                for raw in iter(stream.readline, b""):
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", "replace")
                    line = str(raw).strip()
                    if line:
                        self._stderr_tail.append(line[:500])
            except (OSError, ValueError):
                pass  # pipe closed under us at teardown
            finally:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()

        self._stderr_reader = threading.Thread(target=drain, name="capture-stderr", daemon=True)
        self._stderr_reader.start()

    # -- internals: tail + commit -------------------------------------------

    def _tail_loop(self) -> None:
        """Commit closed segments in sequence; watch for stalls; drain + exit.

        Exits when stopped, or when ffmpeg has been dead for the drain grace
        period (final files flushed). Polling *while* ffmpeg is still alive
        but has written nothing yet is the normal steady state — until it
        lasts longer than the stall timeout.
        """
        ffmpeg_dead_since: float | None = None
        last_progress = time.monotonic()
        while not self._stop_flag.is_set():
            proc = self._proc
            dead = proc is not None and proc.poll() is not None
            now = time.monotonic()
            if self._poll_once(final=dead):
                last_progress = now
            if dead:
                if ffmpeg_dead_since is None:
                    ffmpeg_dead_since = now
                elif now - ffmpeg_dead_since >= self.drain_grace_s:
                    self._poll_once(final=True)
                    # Every segment is committed; the list has served its
                    # purpose (its durations now live on the SegmentInfos).
                    self._drop_segment_list()
                    break
            elif self._stalled(now - last_progress):
                self._kill_stalled(now - last_progress)
            time.sleep(self.poll_s)

    def stall_limit_s(self) -> float | None:
        """Seconds without a new segment that count as a stall (None = off).

        Before this capture's first commit: the first-segment grace. After:
        ``stall_timeout_s`` stretched to ``stall_segments`` × the longest
        recent segment in the buffer (shared across reconnects, so a respawn
        keeps what the stream taught us), capped at ``stall_ceiling_s``.
        """
        if not self.committed:
            return self.first_segment_timeout_s or None
        if not self.stall_timeout_s:
            return None
        return adaptive_timeout_s(
            self.stall_timeout_s,
            self.buffer.recent_segment_max_s(),
            segments=self.stall_segments,
            ceiling_s=self.stall_ceiling_s,
        )

    def _stalled(self, idle_s: float) -> bool:
        limit = self.stall_limit_s()
        return limit is not None and idle_s >= float(limit)

    def _kill_stalled(self, idle_s: float) -> None:
        """No new segment for too long: kill ffmpeg; the waiter reports it."""
        if self._stall_detail:
            return
        self._stall_detail = (
            f"capture stalled: no new segment for {idle_s:.0f}s (ffmpeg killed, reconnecting)"
        )
        log.warning("tailer: %s", self._stall_detail)
        with self._lock:
            self._kill_tree()

    def _poll_once(self, *, final: bool = False) -> bool:
        """Commit every segment that is provably closed; True if any was.

        Segment N is closed once N+1 exists (the muxer closes a segment
        before it opens the next) or, with ``final``, once ffmpeg is gone.
        Only the expected index is ever looked at, so leftovers and late
        writes outside this ffmpeg's sequence are never committed.
        """
        directory = self.buffer.directory
        progressed = False
        # Rows for N are written before N+1 is opened, so reading the list
        # first means every segment found closed below already has its row.
        self._read_segment_list()
        while not self._stop_flag.is_set():
            index = self._next_index
            path = directory / segment_name(index)
            if not path.exists():
                break
            if not final and not (directory / segment_name(index + 1)).exists():
                break  # still being written
            try:
                size = path.stat().st_size
            except OSError:
                break  # vanished mid-poll; retry next cycle
            self._next_index = index + 1
            if size <= 0:
                # A zero-byte .ts is a muxer error artifact, not a segment.
                log.debug("tailer: skipping empty segment %s", path.name)
                continue
            if self._commit_index(index):
                progressed = True
        return progressed

    def _read_segment_list(self) -> None:
        """Parse rows ffmpeg appended to the segment list since the last read.

        Incremental (a day-long watch appends ~15k rows) and line-exact: an
        unterminated last line is kept for the next read, because a half-
        flushed ``seg_000003.ts,18.0,2`` would parse as a bogus end time.
        A file shorter than what was read means it was rewritten: start over.
        """
        path = self.segment_list_path
        if path is None:
            return
        try:
            size = path.stat().st_size
            if size < self._list_offset:
                self._list_offset, self._list_partial = 0, b""
            if size == self._list_offset:
                return
            with open(path, "rb") as handle:
                handle.seek(self._list_offset)
                chunk = handle.read(size - self._list_offset)
        except OSError:
            return  # not created yet (no segment closed), or briefly locked
        self._list_offset += len(chunk)
        data = self._list_partial + chunk
        lines = data.split(b"\n")
        self._list_partial = lines.pop()  # b"" when data ended with a newline
        for raw in lines:
            row = parse_segment_list_row(raw.decode("utf-8", "replace"))
            if row is not None and row[0] >= self._next_index:
                self._list_durations[row[0]] = row[1]

    def _segment_duration(self, index: int, path: Path) -> float | None:
        """Real duration of a closed segment: list row → probe → None."""
        duration = self._list_durations.pop(index, None)
        if duration is not None:
            return duration
        if self._duration_probe is None or self.segment_list_path is None:
            # No list at all (test double, old caller): the buffer's nominal
            # fallback keeps the old timeline without spawning ffprobe.
            return None
        try:
            return self._duration_probe(path)
        except Exception:
            log.debug("tailer: duration probe failed for %s", path.name, exc_info=True)
            return None

    def _drop_segment_list(self) -> None:
        path = self.segment_list_path
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Windows: ffmpeg still dying with the file open. A new watch
            # lifecycle's discard_segment_files sweeps stale lists.
            log.debug("could not delete segment list %s", path, exc_info=True)

    def _commit_index(self, index: int) -> bool:
        """Hand one closed segment to the buffer (it re-stats the size)."""
        try:
            duration = self._segment_duration(index, self.buffer.directory / segment_name(index))
            if duration is None:
                self.buffer.commit_segment(index)
            else:
                self.buffer.commit_segment(index, duration_s=duration)
        except ValueError as exc:
            # The buffer refused a state change (duplicate or out-of-order
            # commit). Log and keep tailing: killing a whole live stream over
            # one bad file is the worse failure.
            log.warning("tailer: buffer refused commit of segment %d: %s", index, exc)
            return False
        except FileNotFoundError:
            log.warning("tailer: segment %d vanished before commit", index)
            return False
        self.committed += 1
        return True

    # -- small helpers ------------------------------------------------------

    def _emit(self, event: str, detail: str) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event, detail)
        except Exception:
            # A callback crash must never take the capture (and its stream)
            # down; the watcher owns deciding what a callback failure means.
            log.exception("capture on_event(%s) raised; swallowed", event)

    def _kill_tree(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            kill_process_tree(proc.pid)
        except Exception:
            log.exception("tree kill failed for pid %s; falling back to kill()", proc.pid)
        # Also signal the direct child: idempotent after taskkill /T (the
        # process is either gone or dying), and it is what unblocks a wait()
        # promptly on POSIX where killpg covers the group without marking
        # this Popen handle exited.
        with contextlib.suppress(OSError, AttributeError):
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # taskkill /T /F is asynchronous teardown; wait it out rather
            # than leak a zombie holding the directory.
            with contextlib.suppress(OSError):
                proc.kill()
        if self._stderr_reader is None and getattr(proc, "stderr", None) is not None:
            # No reader owns the pipe (never started): close it here. With a
            # reader, it closes the pipe itself at EOF.
            with contextlib.suppress(OSError):
                proc.stderr.close()


def is_segment_name(name: str) -> bool:
    """True for ``seg_<digits>.ts`` — the exact on-disk naming contract."""
    return name.startswith("seg_") and name.endswith(SEGMENT_EXT) and name[4:-3].isdigit()


def index_from_name(name: str) -> int:
    """Parse the index out of a segment file name; ValueError otherwise."""
    if not is_segment_name(name):
        raise ValueError(f"not a segment name: {name!r}")
    return int(name[4:-3])


def parse_segment_list_row(line: str) -> tuple[int, float] | None:
    """``seg_000012.ts,48.07,56.07`` → ``(12, 8.0)``; None for anything else.

    The segment muxer's CSV times are in ffmpeg's INPUT timeline (they ignore
    ``-reset_timestamps`` and restart with every spawn), so only the
    difference is used — the buffer chains durations into its own clock.
    """
    parts = line.strip().rsplit(",", 2)
    if len(parts) != 3:
        return None
    name = parts[0].strip().strip('"').replace("\\", "/").rsplit("/", 1)[-1]
    if not is_segment_name(name):
        return None
    try:
        start, end = float(parts[1]), float(parts[2])
    except ValueError:
        return None
    duration = end - start
    if not (0.0 < duration < 3600.0):
        return None
    return index_from_name(name), duration


def probe_segment_duration(path: Path) -> float | None:
    """ffprobe fallback for a segment the list never described."""
    from ..pipeline.ffmpeg import probe

    duration = probe(path).duration_s
    return float(duration) if duration and duration > 0 else None


# ---------------------------------------------------------------------------
# Stream URL resolution (two short-lived attempts, both bounded)
# ---------------------------------------------------------------------------


def resolve_stream_url(url: str) -> str:
    """Resolve a watch URL to a playable stream URL (streamlink → yt-dlp).

    Raises CaptureError with a user-facing hint when neither can. When both
    refusals look like "nothing live to read" the message carries streamlink's
    exact words ("No playable streams") so the watcher's text sniffing — and
    the human reading the log — both land on offline, not broken.
    """
    errors: list[str] = []
    offline = True
    for resolver in (_resolve_with_streamlink, _resolve_with_ytdlp):
        try:
            stream_url = resolver(url)
        except CaptureError as exc:
            errors.append(f"{resolver.__name__}: {exc}")
            offline = offline and exc.offline
            log.debug("resolver %s failed for %s: %s", resolver.__name__, url, exc)
            continue
        if stream_url:
            return stream_url
        errors.append(f"{resolver.__name__}: returned no stream URL")
    message = f"No playable streams found for {url} (" + "; ".join(errors) + ")"
    raise CaptureError(message, hint=_offline_hint(url), offline=offline)


def _resolve_with_streamlink(url: str) -> str:
    """streamlink Python API (preferred: the plugin set is the whole point)."""
    try:
        from streamlink.exceptions import NoStreamsError, PluginError
        from streamlink.session import Streamlink
    except ImportError as exc:
        raise CaptureError(
            "streamlink is not installed",
            hint="Install the live extra: pip install autoclip[live].",
        ) from exc

    session = Streamlink()
    try:
        streams = session.streams(url)
    except NoStreamsError:
        raise CaptureError(f"No playable streams found: {url}", offline=True) from None
    except PluginError as exc:
        # Plugin errors mean "the page said no" (login wall, geo block, or a
        # genuinely dead channel) — offline-shaped, not capture-shaped.
        raise CaptureError(f"streamlink could not read {url}: {exc}", offline=True) from exc
    except Exception as exc:  # noqa: BLE001 - third-party API, surface as hint
        raise CaptureError(f"streamlink failed on {url}: {exc}") from exc

    if not streams:
        raise CaptureError(f"No playable streams found: {url}", offline=True)
    for quality in _BEST_QUALITIES:
        stream = streams.get(quality)
        if stream is None:
            continue
        stream_url = getattr(stream, "url", None)
        if stream_url:
            return str(stream_url)
    # Some plugins expose only the master manifest URL (ffmpeg ingests HLS
    # masters natively), so try to_manifest_url before giving up.
    for quality in _BEST_QUALITIES:
        stream = streams.get(quality)
        to_manifest = getattr(stream, "to_manifest_url", None)
        if callable(to_manifest):
            try:
                manifest_url = to_manifest()
            except Exception:  # noqa: BLE001 - best-effort alternate handle
                manifest_url = None
            if manifest_url:
                return str(manifest_url)
    raise CaptureError(f"No playable streams found: {url}", offline=True)


#: Tried in order after a "confirm you're not a bot" page: plain IPv4, then
#: IPv4 with the mobile-web player client (it still passes when the
#: default web client is challenged on both routes).
_BOT_CHECK_FALLBACKS: tuple[dict, ...] = (
    {"source_address": "0.0.0.0"},
    {"source_address": "0.0.0.0", "extractor_args": {"youtube": {"player_client": ["mweb"]}}},
)


def _is_bot_check(message: str) -> bool:
    lowered = message.lower()
    return "not a bot" in lowered or "sign in to confirm" in lowered


def _extract(yt_dlp, options: dict, url: str):  # noqa: ANN001 - yt_dlp module
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)


def _resolve_with_ytdlp(url: str) -> str:
    """yt-dlp fallback (already a hard dep): extract_info, take a format URL."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise CaptureError(
            "yt-dlp is not installed",
            hint="pip install -U yt-dlp.",
        ) from exc

    class _Quiet:
        """yt-dlp logs through this; silence keeps resolution errors clean."""

        def debug(self, msg: str) -> None: ...

        def info(self, msg: str) -> None: ...

        def warning(self, msg: str) -> None: ...

        def error(self, msg: str) -> None: ...

    options = {
        # A muxed stream first; else video + audio as two inputs (YouTube live).
        "format": "best[vcodec!=none][acodec!=none]/bestvideo[height<=1080]+bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "logger": _Quiet(),
        "socket_timeout": RESOLVE_TIMEOUT_S,
    }
    from ..pipeline.ingest import js_runtime_options

    options.update(js_runtime_options())
    try:
        info = _extract(yt_dlp, options, url)
    except Exception as exc:  # noqa: BLE001 - third-party API, surface as hint
        if not _is_bot_check(str(exc)):
            # Raw yt-dlp text mentions "not exist"/"offline" for dead channels;
            # keep it unfiltered so the watcher's marker sniffing can work.
            raise CaptureError(f"yt-dlp could not resolve {url}: {exc}") from exc
        # YouTube flags VPN (e.g. Cloudflare WARP) IPv6 addresses with a
        # "confirm you're not a bot" page; the same request over IPv4 passes.
        log.info("YouTube bot check over the default route; retrying %s over IPv4.", url)
        info = None
        last: Exception = exc
        for extra in _BOT_CHECK_FALLBACKS:
            try:
                info = _extract(yt_dlp, {**options, **extra}, url)
                break
            except Exception as again:  # noqa: BLE001
                last = again
                if not _is_bot_check(str(again)):
                    break
        if info is None:
            raise CaptureError(f"yt-dlp could not resolve {url}: {last}") from last
    if not info:
        raise CaptureError(f"yt-dlp returned no info for {url}")
    if isinstance(info, list):  # noplaylist violated: take the first entry
        if not info:
            raise CaptureError(f"yt-dlp playlist resolution empty for {url}")
        info = info[0]
    # DASH answers arrive as requested_formats (video+audio split); simpler
    # sources carry a single direct URL. Both shapes are handled because
    # "works on Twitch" must not mean "breaks on YouTube Live".
    requested = info.get("requested_formats")
    if requested:
        urls = [str(f["url"]) for f in requested if isinstance(f, dict) and f.get("url")]
        if len(urls) >= 2:
            # Video first, audio second: both are captured and muxed.
            return SPLIT_SEPARATOR.join(urls[:2])
        if urls:
            return urls[0]
    direct = info.get("url")
    if direct:
        return str(direct)
    raise CaptureError(f"yt-dlp found no stream URL for {url}")


#: Joins a separate video URL and audio URL into one resolved "stream URL"
#: (YouTube live now serves its HLS video and audio as separate streams).
SPLIT_SEPARATOR = " || "


def _input_args(stream_url: str) -> list[str]:
    """``-rw_timeout … -i url`` for each input (one, or video + audio).

    Two inputs keep their real timestamps (``-copyts``): YouTube's video and
    audio playlists rarely open on the same segment, and ffmpeg's default
    per-input reset to zero would shift the audio by that gap (5 s seen).
    """
    args: list[str] = ["-copyts"] if SPLIT_SEPARATOR in stream_url else []
    for part in stream_url.split(SPLIT_SEPARATOR):
        # A read that blocks this long fails the input instead of
        # freezing ffmpeg (and with it the whole stream) forever.
        args += ["-rw_timeout", str(int(RW_TIMEOUT_S * 1_000_000)), "-i", part]
    return args


def _map_args(stream_url: str) -> list[str]:
    if SPLIT_SEPARATOR in stream_url:
        return ["-map", "0:v:0", "-map", "1:a:0"]
    return ["-map", "0"]


def _offline_hint(url: str) -> str:
    """The shared, actionable hint for resolution failures."""
    return (
        "The stream may be offline, or the URL may not point at a live stream "
        f"({url}). If the channel is live, update plugins: "
        "pip install -U streamlink yt-dlp."
    )


# ---------------------------------------------------------------------------
# Process management (Windows is the hostile case)
# ---------------------------------------------------------------------------


def _ffmpeg_binary() -> str:
    """ffmpeg path via the pipeline module (same discovery as renders)."""
    from ..pipeline.ffmpeg import ffmpeg_path

    return ffmpeg_path()


def _new_process_group_flags() -> int:
    """creationflags putting ffmpeg in its own process group.

    Windows: detaches ffmpeg from the console group so a stray CTRL_C_EVENT
    aimed at the watcher cannot cascade into it (and vice versa), while
    taskkill /T handles tree teardown. POSIX: 0 (no creationflags concept);
    stop() signals the group instead.
    """
    import sys

    if sys.platform == "win32":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


#: Windows job object every capture ffmpeg is assigned to; created lazily,
#: its handle held for the life of this process (see bind_to_host_lifetime).
_HOST_JOB: int | None = None
_HOST_JOB_LOCK = threading.Lock()


def _create_kill_on_close_job() -> int:
    """A new Windows job object whose processes die when its last handle closes."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job_object_extended_limit_information = 9
    job_object_limit_kill_on_job_close = 0x2000
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = job_object_limit_kill_on_job_close
    if not kernel32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(wintypes.HANDLE(job))
        raise OSError(error, "SetInformationJobObject failed")
    return int(job)


def _assign_to_job(job: int, proc: subprocess.Popen) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    handle = getattr(proc, "_handle", None)
    if handle is None:
        raise OSError("process has no Windows handle")
    if not kernel32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(handle))):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")


def bind_to_host_lifetime(proc: subprocess.Popen) -> bool:
    """Windows: make ``proc`` die with this process, however this process dies.

    ``CREATE_NEW_PROCESS_GROUP`` detaches ffmpeg from the console, so when
    the backend crashed or was killed, the capture kept running for as long
    as the broadcast did — writing segments nobody prunes into the stream's
    buffer folder and holding its files open. A kill-on-close job object
    owned by this process ends every capture the moment the process is
    gone. Best effort: returns False (logged) where unsupported.
    """
    import sys

    if sys.platform != "win32":
        return False
    global _HOST_JOB
    try:
        with _HOST_JOB_LOCK:
            if _HOST_JOB is None:
                _HOST_JOB = _create_kill_on_close_job()
            job = _HOST_JOB
        _assign_to_job(job, proc)
    except Exception:  # noqa: BLE001 - a missing job must never block a capture
        log.warning("could not tie ffmpeg (pid %s) to AutoClip's lifetime", proc.pid, exc_info=True)
        return False
    return True


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants.

    Windows has no POSIX process groups; ``taskkill /T`` walks the tree
    instead. On POSIX, one SIGTERM+SIGKILL to the dedicated group (created
    by _new_process_group_flags) covers every descendant. Fire-and-forget on
    purpose: stop() must work even mid-teardown.
    """
    import sys

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        import os
        import signal

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.2)


__all__ = [
    "CaptureError",
    "DRAIN_GRACE_S",
    "FIRST_SEGMENT_TIMEOUT_S",
    "RESOLVE_TIMEOUT_S",
    "RW_TIMEOUT_S",
    "SEGMENT_TIME_S",
    "STALL_CEILING_S",
    "STALL_SEGMENTS",
    "STALL_TIMEOUT_S",
    "TAILER_POLL_S",
    "StreamCapture",
    "bind_to_host_lifetime",
    "index_from_name",
    "is_segment_name",
    "kill_process_tree",
    "parse_segment_list_row",
    "probe_segment_duration",
    "resolve_stream_url",
]
