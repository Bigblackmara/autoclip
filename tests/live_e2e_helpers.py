"""Helpers for the Live Watch end-to-end test (one ``live``-marked suite).

Everything here exists to make ``tests/test_live_e2e.py`` short and readable.

REAL components (the whole point of the suite):

* ffmpeg — encodes the fake broadcast, captures it into ``.ts`` segments,
  decodes segment audio for analysis, remuxes extracted windows, burns
  captions into the 9:16 render, and probes the result.
* ``StreamCapture`` → ``RingBuffer`` → the real tailer/commit machinery.
* ``autoclip.live.audio_signals`` — real per-second RMS z-scoring on the
  committed segments, driven by ffmpeg PCM decodes.
* the watcher / manager state machines and the real SQLite store (through
  the temporary ``AUTOCLIP_HOME`` from ``tests/conftest.py``).
* ``autoclip.live.clip_factory.produce`` — a REAL render: window extract,
  remux, audio extraction, crop path, caption burn + export, .srt sidecar.

STUBBED components, exactly two:

1. URL resolution — ``resolve_stream_url`` is monkeypatched (the sanctioned
   ``capture.py`` seam) to a local HTTP URL. No network beyond loopback.
2. Whisper and the LLM judge — replaced through the manager's injection
   parameters. The stub transcript returns real ``Transcript``/``Word``
   objects so captions and the .srt come out of the real code paths.

The trigger/aggregator/circuit below are faithful in-test bodies for the
repo's still-skeletal ``detector.py``/``circuit.py`` (work order A4) —
injection is the seam the watcher was designed around, the same pattern the
unit suites use — and the trigger still only fires on windows that the REAL
audio detector flagged.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from autoclip.pipeline import ffmpeg as ffmpeg_mod
from autoclip.pipeline.transcript import Transcript
from autoclip.pipeline.transcript import Word as TWord

# -- the fake broadcast -------------------------------------------------------


def ffmpeg_path() -> str:
    """The real pipeline ffmpeg discovery (raises clearly when missing)."""
    return ffmpeg_mod.ffmpeg_path()


def run_ffmpeg(args: list[str]) -> None:
    """Run the real ffmpeg; fail the test loudly on a non-zero exit."""
    subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args],
        check=True,
        capture_output=True,
    )


@dataclass
class FakeBroadcast:
    """A served "live" source with a known loud-burst moment."""

    path: Path
    duration_s: float
    #: Stream seconds at which the loud burst starts (with a 2s ramp) and ends.
    burst_start_s: float
    burst_end_s: float


#: Length of the linear ramp into (and out of) the loud burst, seconds.
BURST_RAMP_S = 2.0


def build_fake_broadcast(destination: Path, duration_s: float = 78.0) -> FakeBroadcast:
    """Encode a "stream": quiet narration tone, then a LOUD burst.

    The audio is a 500 Hz sine with a ``volume`` expression that ramps:

    * very quiet (0.008) through the first ~40 s — enough RMS movement for a
      baseline, far below the detector's 0.02 absolute floor;
    * a 2-second linear ramp up to full 1.0 amplitude at ``burst_start_s``;
    * full amplitude until ``burst_end_s``, then a 2-second ramp back down
      (symmetric ramps keep the encoder's loudness analysis stable).

    A 1.5 kHz "hype" sine is mixed in only during the burst, so the spectral
    character differs too. Video is a 24 fps 640x360 H.264 stream —
    ``testsrc2`` is time-varying by construction, so every segment's frames
    differ with no drawtext/font plumbing to break. 640x360 keeps the
    decode, remux and 1080x1920 caption burn cheap — the E2E's job is
    correctness of the chain, not resolution.
    """
    burst_start = 40.0
    burst_end = burst_start + 8.0

    tone_volume = (
        f"volume=volume='if(lt(t,{burst_start}),0.008,"
        f"if(lt(t,{burst_start + BURST_RAMP_S}),"
        f"0.008+(1.0-0.008)*(t-{burst_start})/{BURST_RAMP_S},"
        f"if(lt(t,{burst_end}),1.0,"
        f"if(lt(t,{burst_end + BURST_RAMP_S}),"
        f"1.0-(1.0-0.008)*(t-{burst_end})/{BURST_RAMP_S},0.008))))':eval=frame"
    )
    hype_volume = f"volume=volume='if(between(t,{burst_start},{burst_end}),0.55,0)':eval=frame"

    run_ffmpeg(
        [
            # Quiet narration tone, volume-shaped over time.
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=500:duration={duration_s}",
            # Hype tone, present only during the burst.
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=1500:duration={duration_s}",
            # Time-varying video.
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=24:duration={duration_s}",
            "-filter_complex",
            f"[0:a]{tone_volume}[tone];"
            f"[1:a]{hype_volume}[hype];"
            "[tone][hype]amix=inputs=2:duration=longest:dropout_transition=0,"
            "pan=stereo|c0=c0|c1=c0[aout]",
            "-map",
            "2:v",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            # 2s GOP: the 6s segment muxer can cut cleanly at its boundaries.
            "-g",
            "48",
            "-keyint_min",
            "48",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ar",
            "44100",
            "-ac",
            "2",
            # +faststart: the HTTP reader needs the moov atom up front;
            # the mp4 muxer writes it at the end without this flag.
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    return FakeBroadcast(
        path=destination,
        duration_s=duration_s,
        burst_start_s=burst_start,
        burst_end_s=burst_end,
    )


# -- the lightly paced local source --------------------------------------------


class _RateLimitedFileHandler(BaseHTTPRequestHandler):
    """Serves the fake broadcast at a fixed bytes-per-second budget.

    Capture runs ``-c copy`` with no ``-re``, so ffmpeg reads as fast as TCP
    allows. Serving the file raw would let capture finish ingesting before
    the audio worker analyzed anything, collapsing the test's concurrency
    into a sequential capture-then-analyze run. Mild throttling spreads
    ingest over ~8s of wall time so the tailer commits, the audio worker
    decodes, and the candidate extraction genuinely interleave — a
    few-times-faster-than-real-time stand-in for a CDN that doesn't pay 78s
    of wall clock for a real-paced one.
    """

    #: Bound onto the handler subclass by :func:`serve_throttled`.
    file_path: Path
    file_size: int
    rate_bps: int

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(self.file_size))
        self.send_header("Accept-Ranges", "none")
        self.end_headers()

        window_s = 0.25
        chunk_budget = max(1, int(self.rate_bps * window_s))
        sent = 0
        started = time.monotonic()
        with open(self.file_path, "rb") as source:
            while sent < self.file_size:
                chunk = source.read(chunk_budget)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # capture stopped mid-read: normal teardown
                sent += len(chunk)
                self.wfile.flush()
                delay = sent / self.rate_bps - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)

    def log_message(self, *args: Any) -> None:
        pass  # keep pytest output clean


def serve_throttled(video: Path, rate_bps: int) -> tuple[ThreadingHTTPServer, str]:
    """Serve ``video`` at ``rate_bps`` on 127.0.0.1; returns (server, url).

    The caller owns ``server.shutdown()``; the returned server runs on daemon
    threads so an unramped test teardown can never hang the suite.
    """
    handler = type(
        "BoundHandler",
        (_RateLimitedFileHandler,),
        {"file_path": video, "file_size": video.stat().st_size, "rate_bps": rate_bps},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/live.mp4"
    return server, url


# -- faithful bodies for the repo's A4 skeletons -------------------------------


class TokenBucket:
    """A working body for :class:`autoclip.live.circuit.TokenBucket`.

    Same constructor and ``try_acquire`` contract the watcher probes for; the
    repo's version still raises ``NotImplementedError`` (work order A4
    pending). Capacity N refilled at N/period per second, exactly as
    ``watcher._resolve_defaults`` constructs them.
    """

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = float(rate)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self._last = time.monotonic()

    def try_acquire(self, n: float = 1.0) -> bool:
        self._refill()
        if self.tokens < n:
            return False
        self.tokens -= n
        return True

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(float(self.capacity), self.tokens + (now - self._last) * self.rate)
        self._last = now


class PerStreamCooldown:
    """A working body for :class:`autoclip.live.circuit.PerStreamCooldown`.

    Marks and gates per stream on a monotonic clock. With the default 90 s
    cooldown, the burst's plateau (which keeps emitting peak signals for
    several seconds) cannot mint a second candidate out of one moment — an
    always-allow stand-in would produce two clips and double the render
    cost.
    """

    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = float(cooldown_s)
        self.marks: list[str] = []
        self._last_clip: dict[str, float] = {}

    def can_trigger(self, stream_id: str, *, now: float | None = None) -> bool:
        last = self._last_clip.get(stream_id)
        if last is None:
            return True
        current = now if now is not None else time.monotonic()
        return current - last >= self.cooldown_s

    def mark_clip(self, stream_id: str, *, now: float | None = None) -> None:
        self._last_clip[stream_id] = now if now is not None else time.monotonic()
        self.marks.append(stream_id)

    def remaining(self, stream_id: str, *, now: float | None = None) -> float:
        last = self._last_clip.get(stream_id)
        if last is None:
            return 0.0
        current = now if now is not None else time.monotonic()
        return max(0.0, self.cooldown_s - (current - last))


@dataclass
class _WindowDecision:
    fire: bool = False
    reason: str = ""


class RecordingTrigger:
    """Fires only when the window carries a real audio spike.

    Not "always fire": making the decision read the signal the REAL audio
    worker produced is what keeps this an end-to-end test of detection rather
    than of the queue plumbing. The contract shape (``decide(window)`` →
    ``.fire``) matches the frozen one the watcher probes for.
    """

    def __init__(self, min_strength: float = 0.5) -> None:
        self.min_strength = min_strength
        self.calls: list[Any] = []

    def decide(self, window: Any) -> _WindowDecision:
        self.calls.append(window)
        best = max((getattr(s, "strength", 0.0) for s in window.signals), default=0.0)
        has_spike = any(getattr(s, "kind", None) == "audio_energy" for s in window.signals)
        if has_spike and best >= self.min_strength:
            return _WindowDecision(fire=True, reason=f"audio spike strength={best:.2f}")
        return _WindowDecision(fire=False, reason="no spike yet")


class RecordingAggregator:
    """SignalAggregator body (A4 skeleton): collect + evaluate a window.

    Mirrors the frozen contract the watcher probes for: ``add(signal)`` and
    ``evaluate(window_s)`` → :class:`~autoclip.live.signals.SignalWindow`.
    """

    def __init__(self) -> None:
        self.signals: list[Any] = []

    def add(self, signal: Any) -> None:
        self.signals.append(signal)

    def evaluate(self, window_s: float = 30.0) -> Any:
        from autoclip.live.signals import SignalWindow

        if not self.signals:
            return SignalWindow(start_s=0.0, end_s=0.0, signals=[])
        newest = max(s.t_s for s in self.signals)
        oldest = newest - window_s
        return SignalWindow(
            start_s=max(0.0, oldest),
            end_s=newest,
            signals=[s for s in self.signals if s.t_s >= oldest],
        )


# -- stubbed heavy brains (LLM judge + Whisper) --------------------------------


@dataclass
class FakeJudgement:
    """LiveJudgement-shaped verdict; the watcher reads score/title/+offsets."""

    score: int
    title: str = ""
    hook: str = ""
    reason: str = ""
    start_word_offset: int | None = None
    end_word_offset: int | None = None


async def judge_accept_everything(
    label: str, words: list[Any], signals: dict[str, float]
) -> FakeJudgement:
    """Stub LLM judge: accept with a score far above the gate.

    The real judge calls a hosted LLM; this stub keeps the suite offline
    while exercising the exact call path (async, keyword shape, signal dict
    keyed by signal kind).
    """
    spike = signals.get("audio_energy", 0.0)
    return FakeJudgement(
        score=93,
        title="The Fake Stream Went Off",
        hook=f"audio spike {spike:.2f}",
        reason="e2e stub accepted the window",
    )


def stub_transcribe_factory(clip_words: list[TWord]):
    """Build a ``transcribe`` seam returning a fixed synthetic transcript.

    The stub returns the real path's data shape — genuine ``Word`` objects
    with start/end — so captions, the .srt and the judge all run the exact
    code paths on the exact data type they expect. Whisper's GPU/CPU cost is
    the one thing a fast end-to-end run cannot afford.
    """

    def transcribe(path: Path, *args: Any, **kwargs: Any) -> Transcript:
        return Transcript(
            words=list(clip_words),
            segments=[],
            language="en",
            model="stub-e2e",
            source="whisper",
        )

    return transcribe


def make_clip_words() -> list[TWord]:
    """Uniformly-spaced words spanning the rendered window's timeline.

    Word times are clip-relative (the factory renders with start_s=0),
    spaced 0.6 s apart so caption groups never overlap and the burn yields
    clean ASS events across the whole render.
    """
    texts = [
        "odds",
        "this",
        "fake",
        "stream",
        "exploded",
        "today",
        "we",
        "are",
        "never",
        "getting",
        "another",
        "moment",
        "like",
        "this",
        "hold",
        "on",
        "to",
        "something",
        "because",
        "it",
        "is",
        "happening",
        "right",
        "now",
        "live",
        "on",
        "the",
        "internet",
        "absolutely",
        "unreal",
        "scenes",
    ]
    words: list[TWord] = []
    t = 1.0
    for text in texts:
        words.append(TWord(text=text, start=t, end=t + 0.45))
        t += 0.6
    return words


# -- verification utilities ----------------------------------------------------


def probe_streams(path: Path) -> dict[str, Any]:
    """Stream facts via a real ffprobe, for the ready-clip assertions."""
    info = ffmpeg_mod.probe(path)
    return {
        "width": info.width,
        "height": info.height,
        "video_codec": info.video_codec,
        "audio_codec": info.audio_codec,
        "has_audio": info.has_audio,
        "duration_s": info.duration_s,
    }
