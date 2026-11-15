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
   ``-segment_time 6 -reset_timestamps 1``, codecs copied. A pipe chain
   (streamlink | ffmpeg) would tie ffmpeg's lifetime to a third party's
   socket; two decoupled processes mean a reconnect restarts only resolution.

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
   segment has been committed for ``stall_timeout_s`` (≈3 segments), the
   process is killed and an ``"error"`` event fires, so the watcher
   reconnects instead of sitting on a frozen capture forever.

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

from .buffer import SEGMENT_EXT, SEGMENT_S, RingBuffer, segment_name

log = logging.getLogger(__name__)

#: Seconds of stream time per ffmpeg segment. Re-exported from the buffer so
#: capture and buffer can never drift apart on the one constant that maps
#: index → seconds for every downstream consumer.
SEGMENT_TIME_S = SEGMENT_S

#: Tailer poll interval. One interval of commit latency is harmless
#: (downstream detectors see the segment a moment later); a premature commit
#: is not harmless — so the loop errs slow, not fast.
TAILER_POLL_S = 0.5

#: ffmpeg ``-rw_timeout`` for the input, seconds: a read that blocks longer
#: than this fails the input instead of hanging the process forever.
RW_TIMEOUT_S = 15.0

#: Stall watchdog: ffmpeg alive but no segment committed for this long
#: (≈3 segments) means a frozen capture — kill it so the watcher reconnects.
STALL_TIMEOUT_S = 3 * SEGMENT_S

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
        first_segment_timeout_s: float | None = FIRST_SEGMENT_TIMEOUT_S,
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
        self.first_segment_timeout_s = first_segment_timeout_s

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
            "-rw_timeout",
            str(int(RW_TIMEOUT_S * 1_000_000)),
            "-i",
            stream_url,
            "-c",
            "copy",
            "-map",
            "0",
            "-f",
            "segment",
            "-segment_time",
            str(int(SEGMENT_TIME_S)),
            "-reset_timestamps",
            "1",
            "-segment_start_number",
            str(start_number),
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
                    break
            elif self._stalled(now - last_progress):
                self._kill_stalled(now - last_progress)
            time.sleep(self.poll_s)

    def _stalled(self, idle_s: float) -> bool:
        limit = self.stall_timeout_s if self.committed else self.first_segment_timeout_s
        return bool(limit) and idle_s >= float(limit)  # type: ignore[arg-type]

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

    def _commit_index(self, index: int) -> bool:
        """Hand one closed segment to the buffer (it re-stats the size)."""
        try:
            self.buffer.commit_segment(index)
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
        "format": "best/bestvideo+bestaudio",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "logger": _Quiet(),
        "socket_timeout": RESOLVE_TIMEOUT_S,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - third-party API, surface as hint
        # Raw yt-dlp text mentions "not exist"/"offline" for dead channels;
        # keep it unfiltered so the watcher's marker sniffing can work.
        raise CaptureError(f"yt-dlp could not resolve {url}: {exc}") from exc
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
        for fmt in requested:
            if isinstance(fmt, dict) and fmt.get("url"):
                return str(fmt["url"])
    direct = info.get("url")
    if direct:
        return str(direct)
    raise CaptureError(f"yt-dlp found no stream URL for {url}")


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
    "STALL_TIMEOUT_S",
    "TAILER_POLL_S",
    "StreamCapture",
    "index_from_name",
    "is_segment_name",
    "kill_process_tree",
    "resolve_stream_url",
]
