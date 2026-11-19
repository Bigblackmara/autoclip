"""GTA Live — "Clip this live stream".

Captures a live stream into a short rolling buffer (Live Watch's proven
capture + ring buffer), listens every ``listen_interval_s`` (20 s), and when
the chat explodes or the streamer loses it, judges the last ``context_s``
(60 s) with the same strict GTA rules as VODs. A "yes" is rendered straight
from the buffer (full stream quality) with the GTA layout and QA, and the
turnaround from the moment to the finished file is recorded against the
``turnaround_s`` target (3 min).

Chat signals come from Twitch's public IRC (read-only, anonymous — the same
source Live Watch uses). YouTube/Kick live chat isn't read, so on those the
trigger is the streamer's audio alone.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ..db.models import new_id
from ..pipeline import transcribe
from ..pipeline.transcript import Transcript, Word
from ..providers.base import ProviderError
from . import judging, render, signals, store
from .gate import gate
from .pipeline import (
    _model_cache,
    _provider_complete,
    caption_look,
    clip_destination,
    localise_clip,
    narration_context,
    post_text,
    render_and_qa,
    wants_flash_forward,
)
from .settings import GtaSettings, work_root
from .settings import load as load_settings
from .store import GtaClip

log = logging.getLogger(__name__)

#: Seconds of history the audio/chat baselines use.
BASELINE_S = 300.0
#: Trigger thresholds.
AUDIO_Z_TRIGGER = 3.0
CHAT_RATIO_TRIGGER = 2.5
CHAT_MIN_MSGS = 12
CLIP_WORDS_TRIGGER = 3
#: After a clip, the same moment can't trigger again for this long.
COOLDOWN_S = 45.0
#: Extra seconds of stream kept after the trigger point (the payoff): at
#: least PAYOFF_MIN_S, then PAYOFF_STEP_S more at a time while the streamer
#: or chat is still hyped, up to PAYOFF_MAX_S, so a situation that keeps
#: escalating is seen through to its end before the judge picks the cut.
PAYOFF_TAIL_S = 15.0
PAYOFF_MIN_S = PAYOFF_TAIL_S
PAYOFF_STEP_S = 5.0
PAYOFF_MAX_S = 45.0
#: Still "hot" while the loudest second is this many sigma over normal …
SETTLE_Z = 1.0
#: … or chat runs at this multiple of its usual rate.
SETTLE_CHAT_X = 1.5
#: No new video for this long counts as a stalled capture (shown on the card).
STALL_S = 90.0
#: Judge windows kept per live stream (older ones are deleted).
KEEP_WINDOWS = 15
#: A window that produced a clip is kept this long anyway (re-render source).
KEEP_SOURCE_S = 48 * 3600
KEEP_MARKER = "KEEP"
#: Shortest live clip the judge may cut (the user's floor: 15 s).
LIVE_MIN_CLIP_S = 15.0
#: A context check with fewer words than this skips the AI judge.
CONTEXT_MIN_WORDS = 30
#: Prefix of the "why" for the every-minute judgement of what was said.
CONTEXT_WHY = "context check"
#: Groq's hosted Whisper: a minute of audio in under a second, no GPU wait.
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_STT_MODEL = "whisper-large-v3-turbo"
_LANGUAGE_CODES = {
    "english": "en", "hindi": "hi", "spanish": "es", "portuguese": "pt", "french": "fr",
    "german": "de", "italian": "it", "japanese": "ja", "korean": "ko", "russian": "ru",
    "arabic": "ar", "turkish": "tr", "chinese": "zh", "urdu": "ur", "bengali": "bn",
}  # fmt: skip
#: Seconds to wait before the one retry of a failed AI judge call.
JUDGE_RETRY_WAIT_S = 5.0


def transcribe_hosted(audio: Path) -> tuple[list[Word], str] | None:
    """Words + language from Groq's hosted Whisper; None to fall back to local."""
    import httpx

    from ..providers import usage
    from ..providers.openai_provider import _next_groq_order, groq_fallback_keys

    # Keys resting after a rate limit are skipped; none left = local Whisper.
    keys = [k for k in groq_fallback_keys() if usage.resting(GROQ_STT_URL, k) <= 0]
    if not keys:
        return None
    data = audio.read_bytes()
    form = {
        "model": GROQ_STT_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    for key in _next_groq_order(keys):
        try:
            response = httpx.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (audio.name, data, "audio/wav")},
                data=form,
                timeout=30,
            )
            if response.status_code != 200:
                log.info("Hosted Whisper %s: %s", response.status_code, response.text[:200])
                usage.record(
                    GROQ_STT_URL,
                    key,
                    kind="audio",
                    ok=False,
                    status=response.status_code,
                    headers=response.headers,
                    error=response.text[:300],
                )
                continue
            body = response.json()
            usage.record(
                GROQ_STT_URL,
                key,
                kind="audio",
                audio_s=float(body.get("duration") or 0.0),
                headers=response.headers,
            )
        except (httpx.HTTPError, ValueError) as exc:
            log.info("Hosted Whisper failed: %s", exc)
            usage.record(GROQ_STT_URL, key, kind="audio", ok=False, error=str(exc))
            continue
        words = [
            Word(str(w.get("word", "")).strip(), float(w["start"]), float(w["end"]))
            for w in body.get("words") or []
            if str(w.get("word", "")).strip() and "start" in w and "end" in w
        ]
        lang = str(body.get("language") or "").strip().lower()
        return words, (lang if len(lang) == 2 else _LANGUAGE_CODES.get(lang, ""))
    return None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def name_from_url(url: str) -> str:
    """A channel name read off the link itself ("" when the link has none).

    twitch.tv/<login>, kick.com/<login> and youtube.com/@handle carry it;
    a youtube watch?v= link doesn't, so that one waits for the probe.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return ""
    if host.endswith("twitch.tv") or host.endswith("kick.com"):
        first = parts[0]
        if first.lower() in {"videos", "directory", "search", "settings"}:
            return ""
        return first if re.fullmatch(r"[A-Za-z0-9_.-]{2,40}", first) else ""
    if "youtube.com" in host and parts[0].startswith("@"):
        return parts[0]
    return ""


@dataclass
class _Msg:
    t: float
    text: str


class _RecordingChat:
    """Twitch IRC reader that keeps the raw messages (stream-timed)."""

    def __init__(self, url: str, clock) -> None:
        from ..live.chat_signals import ChatDisabled, TwitchChatReader, channel_from_url
        from ..live.settings import LiveChatSettings

        self.messages: deque[_Msg] = deque(maxlen=50_000)
        self._lock = threading.Lock()
        try:
            channel = channel_from_url(url)
        except ChatDisabled:
            self.reader = None
            return
        outer = self

        class Reader(TwitchChatReader):
            def _observe(self, nick: str, text: str) -> None:  # noqa: D401
                t = clock()
                if t is None:
                    return
                with outer._lock:
                    outer.messages.append(_Msg(t, text))

        self.reader = Reader(channel, LiveChatSettings(), on_signal=lambda _s: None, clock=clock)

    def window(self, a: float, b: float) -> list[_Msg]:
        with self._lock:
            return [m for m in self.messages if a <= m.t < b]


class GtaLiveWatch:
    def __init__(self, stream_id: str, url: str) -> None:
        self.stream_id = stream_id
        self.url = url
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._buffer = None
        self._chat: _RecordingChat | None = None
        self._levels: deque[tuple[float, float]] = deque()  # (t, dB) per second
        self._last_clip_end = -1e9
        self._ended = False
        self._root = work_root() / stream_id / "live"
        self._signals_lock = threading.Lock()
        self._capture_down = ""
        self._on_capture_event = None
        self._reconnects = 0
        self._moments = 0
        self._stalled = False
        self._listens = 0
        self._pulse: dict = {}
        self._judge_now = threading.Event()
        #: Stream time of the last judgement (spike, manual or context check).
        self._last_judged_s = 0.0
        self._timing = ""

    def request_judge_now(self) -> None:
        """Judge the last minute on the next pass, whatever the signals say."""
        self._judge_now.set()

    # -- what the card shows ------------------------------------------------

    def _activity(self, text: str) -> None:
        """Append to the card's activity log (newest first, last 12 kept)."""
        with self._signals_lock:
            stream = store.get_stream(self.stream_id)
            if stream is None:
                return
            sig = dict(stream.signals or {})
            log_rows = [{"at": _now_iso(), "text": text[:300]}, *list(sig.get("activity") or [])]
            sig["activity"] = log_rows[:12]
            store.update_stream(self.stream_id, signals=sig)

    def _note(self, **values) -> None:
        """Merge ``values`` into the stream's signals (shown on its card)."""
        with self._signals_lock:
            stream = store.get_stream(self.stream_id)
            if stream is None:
                return
            merged = {**(stream.signals or {}), **values}
            store.update_stream(self.stream_id, signals=merged)

    def _problem(self, where: str, text: str) -> None:
        """Record a background problem so it shows on the card, not only in the log."""
        log.warning("GTA Live (%s) %s: %s", self.url, where, text)
        self._note(last_problem={"where": where, "text": text[:500], "at": _now_iso()})
        self._activity(f"⚠ {text}")

    def _fetch_meta(self) -> None:
        """Fill in the channel name, title and thumbnail (never raises)."""
        from .fetch import probe

        try:
            info = probe(self.url)
        except Exception as exc:
            log.info("GTA Live could not read %s's details: %s", self.url, exc)
            return
        fields = {k: v for k, v in (("channel", info.channel), ("title", info.title)) if v}
        if fields:
            store.update_stream(self.stream_id, **fields)
        if info.thumbnail:
            self._note(thumbnail=info.thumbnail)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"gta-live-{self.stream_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _status(self, status: str, detail: str = "", **fields) -> None:
        store.update_stream(self.stream_id, status=status, detail=detail, **fields)

    def _clock(self) -> float | None:
        buffer = self._buffer
        if buffer is None:
            return None
        return buffer.covered_until_s()

    def _run(self) -> None:
        from ..live.buffer import RingBuffer
        from ..live.capture import CaptureError, StreamCapture

        settings = load_settings()
        try:
            self._status("live_connecting", "Connecting to the live stream")
            buffer_dir = self._root / "buffer"
            # A previous run's segments would be mistaken for new video.
            shutil.rmtree(buffer_dir, ignore_errors=True)
            self._buffer = RingBuffer(buffer_dir, retain_minutes=6.0, max_disk_mb=2048)
            self._note(stalled=False, last_problem=None, pulse={}, moments=0, last_moment=None)

            def on_event(kind: str, detail: str) -> None:
                # "ended" isn't trusted on its own: YouTube's HLS also exits
                # cleanly when its manifest expires. Reconnect; a stream that
                # is really over refuses to resolve and ends the watch then.
                if kind in {"ended", "error"}:
                    self._capture_down = f"{kind}: {detail}"

            self._on_capture_event = on_event
            self._capture = StreamCapture(self.url, self._buffer, on_event=on_event)
            self._capture.start()
            threading.Thread(
                target=self._fetch_meta, daemon=True, name=f"gta-live-meta-{self.stream_id}"
            ).start()
            self._chat = _RecordingChat(self.url, self._clock)
            if self._chat.reader is not None:
                threading.Thread(
                    target=self._chat.reader.run,
                    args=(self._stop,),
                    daemon=True,
                    name="gta-live-chat",
                ).start()
            self._status("live_watching", "Listening")
            self._activity("Connected. Listening every few seconds")
            self._loop(settings)
        except CaptureError as exc:
            self._status("failed", "", error=str(exc)[:2000])
            return
        except Exception as exc:  # pragma: no cover - logged and surfaced
            log.exception("GTA Live failed for %s.", self.url)
            self._status("failed", "", error=str(exc)[:2000])
            return
        finally:
            if self._capture is not None:
                try:
                    self._capture.stop()
                except Exception:
                    log.warning("Stopping the live capture failed.", exc_info=True)
            self._stop.set()
        if self._ended:
            self._status("live_ended", "The stream ended")
        else:
            self._status("live_stopped", "Stopped")

    # -- the listen loop -------------------------------------------------

    def _loop(self, settings: GtaSettings) -> None:
        interval = max(5.0, settings.live.listen_interval_s)
        last_checked = None
        last_progress = time.monotonic()
        while not self._stop.wait(interval):
            if self._capture_down and not self._reconnect():
                return
            now = self._clock()
            if now is not None and self._judge_now.is_set():
                self._judge_now.clear()
                self._judge_moment(now, "manual check (Judge now)", settings)
                last_checked = now
                last_progress = time.monotonic()
                continue
            if now is None or (last_checked is not None and now <= last_checked):
                if time.monotonic() - last_progress > STALL_S:
                    # No new video: reconnect rather than wait on a dead capture.
                    self._capture_down = f"no new video for {int(STALL_S)} s"
                    last_progress = time.monotonic()
                continue
            last_progress = time.monotonic()
            if self._stalled:
                self._stalled = False
                self._note(stalled=False)
            since = last_checked if last_checked is not None else max(0.0, now - interval)
            last_checked = now
            try:
                fired, why = self._listen(since, now)
            except Exception as exc:
                log.debug("listen pass failed", exc_info=True)
                self._problem("listen", f"Couldn't measure the last {int(interval)} s: {exc}")
                continue
            self._listens += 1
            cooling = fired and now - self._last_clip_end < COOLDOWN_S
            self._note(
                pulse={
                    **self._pulse,
                    "at": _now_iso(),
                    "listens": self._listens,
                    "stream_s": round(now, 1),
                    "cooldown": cooling,
                }
            )
            if not fired:
                every = settings.live.judge_every_s
                if (
                    every > 0
                    and now - self._last_judged_s >= every
                    and now - self._last_clip_end >= COOLDOWN_S
                ):
                    self._judge_moment(now, f"{CONTEXT_WHY} (what was said)", settings)
                continue
            if cooling:
                self._activity(f"Spike ({why}) ignored: just clipped this moment")
                continue
            self._judge_moment(now, why, settings)

    def _judge_moment(self, now: float, why: str, settings: GtaSettings) -> None:
        """Judge (and render) one moment; every outcome lands on the card."""
        self._moments += 1
        self._note(
            moments=self._moments,
            last_moment={"why": why, "at": _now_iso(), "result": "judging"},
        )
        self._activity(f"Moment flagged: {why}. Sending the last minute to the AI judge")
        store.update_stream(self.stream_id, detail=f"Moment detected ({why}) — judging")
        self._last_judged_s = now
        self._timing = ""
        started = time.monotonic()
        try:
            result = self._evaluate(now, why, settings, triggered_at=time.time())
        except ProviderError as exc:
            result = "AI judge unavailable — moment skipped"
            self._problem("ai", f"AI judge failed: {exc}")
        except Exception as exc:
            # Full traceback in the server log: the card only shows str(exc).
            log.warning("GTA Live evaluation failed", exc_info=True)
            result = "error — moment skipped"
            self._problem("judge", f"Judging the moment failed: {exc}")
        self._note(last_moment={"why": why, "at": _now_iso(), "result": result or ""})
        self._prune_windows()
        took = f"took {time.monotonic() - started:.0f} s"
        self._activity(
            f"Judge result: {result or 'done'} ({took}: {self._timing})"
            if self._timing
            else f"Judge result: {result or 'done'} ({took})"
        )
        store.update_stream(self.stream_id, detail="Listening")

    def _prune_windows(self, keep: int = KEEP_WINDOWS) -> None:
        """Delete all but the newest ``keep`` judge windows (each is ~10-25 MB).

        Every judgement saves a minute of video; at one a minute per stream
        they filled the disk within hours. Finished clips live in exports.
        """
        try:
            dirs = [d for d in self._root.iterdir() if d.is_dir() and d.name != "buffer"]
            dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            return
        now = time.time()
        for old in dirs[keep:]:
            marker = old / KEEP_MARKER
            try:
                if marker.exists() and now - marker.stat().st_mtime < KEEP_SOURCE_S:
                    continue  # the source of a made clip: kept for re-renders
            except OSError:
                pass
            shutil.rmtree(old, ignore_errors=True)

    def _reconnect(self) -> bool:
        """Restart a capture whose ffmpeg exited; False when the stream is over."""
        from ..live.capture import CaptureError, StreamCapture

        why = self._capture_down
        self._problem("capture", f"Capture dropped ({why[:200]}); reconnecting")
        store.update_stream(self.stream_id, detail="Reconnecting to the stream")
        delay = 3.0
        deadline = time.monotonic() + 300
        while not self._stop.is_set():
            if self._capture is not None:
                try:
                    self._capture.stop()
                except Exception:
                    log.debug("old capture stop failed", exc_info=True)
            self._capture_down = ""
            try:
                assert self._buffer is not None
                # New capture = new clock: a window must never join segments
                # from both sides of the break (their timestamps clash and the
                # killed capture's last segment ends mid-packet), so the
                # buffer clamps every window to one epoch.
                self._buffer.begin_epoch()
                self._capture = StreamCapture(
                    self.url, self._buffer, on_event=self._on_capture_event
                )
                self._capture.start()
            except CaptureError as exc:
                if time.monotonic() > deadline:
                    self._ended = True
                    return False
                log.info("GTA Live reconnect failed (%s); retrying in %.0f s", exc, delay)
                if self._stop.wait(delay):
                    return False
                delay = min(30.0, delay * 2)
                continue
            self._reconnects += 1
            # Recovered: the drop stays in the activity log, not as a warning.
            self._note(reconnects=self._reconnects, stalled=False, last_problem=None)
            self._activity("Reconnected to the stream")
            store.update_stream(self.stream_id, detail="Listening")
            return True
        return False

    def _listen(self, a: float, b: float) -> tuple[bool, str]:
        """Measure the last interval; True when the chat explodes or the streamer loses it."""
        assert self._buffer is not None
        target = self._root / "listen.ts"
        window = self._buffer.extract_window(max(0.0, a), b, target)
        levels = render.section_levels(
            window.path, max(0.0, a - window.start_s), b - window.start_s, hop_s=1.0
        )
        for i, db in enumerate(levels):
            self._levels.append((a + i, float(db)))
        while self._levels and self._levels[0][0] < b - BASELINE_S:
            self._levels.popleft()
        reasons: list[str] = []

        history = np.array([db for t, db in self._levels if t < a and db > signals.SILENCE_DB + 1])
        recent = np.array([db for t, db in self._levels if t >= a])
        pulse: dict = {
            "need_z": AUDIO_Z_TRIGGER,
            "baseline_s": int(len(history)),
            "warming_up": len(history) < 30,
            "chat": self._chat is not None and self._chat.reader is not None,
        }
        if len(recent):
            pulse["loudest_db"] = round(float(np.max(recent)), 1)
        if len(history) >= 30 and len(recent):
            centre = float(np.median(history))
            spread = max(1.5, float(np.median(np.abs(history - centre))) * 1.4826)
            z = (float(np.max(recent)) - centre) / spread
            pulse.update(audio_z=round(z, 2), typical_db=round(centre, 1))
            if z >= AUDIO_Z_TRIGGER:
                reasons.append(f"streamer lost it (+{z:.1f} sigma)")
        self._pulse = pulse

        if self._chat is not None and self._chat.reader is not None:
            recent_msgs = self._chat.window(a, b)
            past = self._chat.window(b - BASELINE_S, a)
            span_past = max(1.0, min(BASELINE_S, a - max(0.0, b - BASELINE_S)))
            rate_now = len(recent_msgs) / max(1.0, b - a)
            rate_past = len(past) / span_past
            if (
                len(recent_msgs) >= CHAT_MIN_MSGS
                and rate_past > 0
                and rate_now / rate_past >= CHAT_RATIO_TRIGGER
            ):
                reasons.append(f"chat exploded ({rate_now / rate_past:.1f}x)")
            clip_words = sum(signals.keyword_hits(m.text)[0] for m in recent_msgs)
            if clip_words >= CLIP_WORDS_TRIGGER:
                reasons.append(f'chat spamming "clip" (x{clip_words})')
            self._pulse.update(
                chat_msgs=len(recent_msgs),
                chat_x=round(rate_now / rate_past, 1) if rate_past > 0 else None,
            )
        return bool(reasons), "; ".join(reasons)

    # -- judging + rendering ----------------------------------------------

    def _still_hot(self, moment: float, a: float, b: float) -> bool:
        """True while the streamer (audio) or chat is still well above its normal
        (before ``moment``) in [a, b]."""
        assert self._buffer is not None
        try:
            window = self._buffer.extract_window(max(0.0, a), b, self._root / "payoff.ts")
            levels = render.section_levels(
                window.path, max(0.0, a - window.start_s), b - window.start_s, hop_s=1.0
            )
        except Exception:
            log.debug("payoff level check failed", exc_info=True)
            levels = np.array([])
        history = np.array(
            [db for t, db in self._levels if t < moment - 5 and db > signals.SILENCE_DB + 1]
        )
        if len(levels) and len(history) >= 30:
            centre = float(np.median(history))
            spread = max(1.5, float(np.median(np.abs(history - centre))) * 1.4826)
            if (float(np.max(levels)) - centre) / spread >= SETTLE_Z:
                return True
        if self._chat is not None and self._chat.reader is not None:
            now_msgs = self._chat.window(a, b)
            past = self._chat.window(moment - BASELINE_S, moment - 5)
            span_past = max(1.0, BASELINE_S - 5)
            rate_past = len(past) / span_past
            if rate_past > 0 and len(now_msgs) / max(1.0, b - a) / rate_past >= SETTLE_CHAT_X:
                return True
        return False

    def _wait_for_payoff(self, now: float) -> float | None:
        """Seconds of stream after ``now`` to judge with (None = stopped).

        Waits PAYOFF_MIN_S, then keeps extending while it's still hot.
        """
        tail = PAYOFF_MIN_S
        while True:
            deadline = time.monotonic() + tail + 30
            while (self._clock() or 0) < now + tail and time.monotonic() < deadline:
                if self._stop.wait(1.0):
                    return None
            if (self._clock() or 0) < now + tail:  # capture stalled: judge what's there
                return max(0.0, (self._clock() or now) - now)
            hot = self._still_hot(now, now + tail - PAYOFF_STEP_S, now + tail)
            if tail >= PAYOFF_MAX_S or not hot:
                return tail
            tail = min(PAYOFF_MAX_S, tail + PAYOFF_STEP_S)
            store.update_stream(
                self.stream_id, detail=f"Moment still going — watching {tail:.0f} s past it"
            )

    def _evaluate(self, now: float, why: str, settings: GtaSettings, *, triggered_at: float) -> str:
        """Judge (and render) one flagged moment; returns a one-line outcome."""
        assert self._buffer is not None
        t0 = time.monotonic()
        context_check = why.startswith(CONTEXT_WHY)
        # Let the payoff land: wait for a few more seconds of capture (a
        # context check reads what's already there).
        if context_check:
            tail = 0.0
        else:
            waited = self._wait_for_payoff(now)
            if waited is None:
                return "stopped"
            tail = waited
        end = min(self._clock() or now, now + tail)
        start = max(0.0, now - settings.live.context_s)
        clip_id = new_id()
        work = self._root / clip_id
        window = self._buffer.extract_window(start, end, work / "window.ts", anchor_s=now)

        audio = work / "window.wav"
        from ..pipeline import prepare

        prepare.extract_audio(window.path, audio)
        t1 = time.monotonic()
        hosted = transcribe_hosted(audio)
        if hosted is not None:
            raw_words, language, how = hosted[0], hosted[1], "hosted"
        else:
            with gate.hold(live=True):
                piece = transcribe.transcribe(audio, load_whisper(), model_cache=_model_cache)
            raw_words, language, how = piece.words, piece.language, "local"
        words = [
            Word(w.text, w.start + window.start_s, w.end + window.start_s, w.speaker)
            for w in raw_words
        ]
        t2 = time.monotonic()
        self._timing = f"wait {t1 - t0:.0f} s · transcribe {t2 - t1:.1f} s ({how})"
        transcript = Transcript(words=words)
        if not words:
            return "nothing was said — skipped"
        if context_check and len(words) < CONTEXT_MIN_WORDS:
            # Not worth an AI call (each costs ~2k tokens of a daily quota).
            return f"only {len(words)} words said — skipped"
        chunk = judging.Chunk(
            index=0,
            start_s=start,
            end_s=end,
            first_word=0,
            last_word=len(words) - 1,
            tail_word=len(words) - 1,
        )
        stream = store.get_stream(self.stream_id)
        streamer = (stream.channel if stream else "") or "the streamer"
        title = stream.title if stream else ""
        complete = _provider_complete()
        live_settings = settings.model_copy(
            update={"min_clip_s": min(settings.min_clip_s, LIVE_MIN_CLIP_S)}
        )

        focus = (
            "no loudness spike; judge what was SAID. Clip it if the streamer says something "
            "important, newsworthy or revealing, a strong take, a funny line, or tells a story "
            "worth watching. Talk moments count, not only loud ones"
            if context_check
            else (
                f"moment flagged: {why} at {now - start:.0f} s into this window; the window "
                f"runs {end - now:.0f} s past it so the situation can play out. End the clip "
                "only once the situation has escalated AND resolved (the reaction settles, the "
                "chase, fight, argument or conversation is over, or the scene changes). Never "
                "cut while it is still building; a longer clip with the full payoff beats a "
                "short one that stops early"
            )
        )
        # Shared context: everyone scrolling knows GTA 6 and its launch, so a
        # countdown, an unlock or a launch reaction never "needs outside context".
        focus += (
            ". Every viewer already knows GTA 6 and that it is launching right now: never "
            "reject for needing outside context about GTA 6, the countdown, the launch or the "
            "viewer count; a countdown hitting zero, the game unlocking or a streamer losing it "
            "over GTA 6 IS the payoff. If a great line is short, widen the clip around it "
            "instead of rejecting it"
        )

        async def judge() -> list[judging.Candidate]:
            found = await judging.find_candidates(
                transcript,
                [chunk],
                complete=complete,
                settings=live_settings,
                curve=None,
                silences=[],
                streamer=streamer,
                title=f"{title} (LIVE — {focus})",
            )
            return await judging.final_round(
                transcript,
                found,
                complete=complete,
                settings=live_settings,
                streamer=streamer,
                title=f"{title} (LIVE — {focus})",
            )

        t3 = time.monotonic()
        try:
            candidates = asyncio.run(judge())
        except ProviderError as exc:
            # One retry: hosted APIs hiccup (503s) and a live moment can't be re-run later.
            self._problem("ai", f"AI judge failed, retrying once: {exc}")
            if self._stop.wait(JUDGE_RETRY_WAIT_S):
                return "stopped"
            candidates = asyncio.run(judge())
        self._note(ai_ok_at=_now_iso())
        self._timing += f" · AI {time.monotonic() - t3:.1f} s"
        approved = sum(1 for c in candidates if c.status == "approved")
        if approved:
            # This window is a clip's only source: keep it past the pruning.
            (work / KEEP_MARKER).touch()
        for c in candidates:
            clip = GtaClip(
                id=new_id(),
                stream_id=self.stream_id,
                start_s=c.start_s,
                end_s=c.end_s,
                status="rejected" if c.status != "approved" else "rendering",
                hook_title=c.hook_title,
                hook_quote=c.hook_quote,
                moment_type=c.moment_type,
                summary=c.why,
                score=c.score,
                # Where the clip came from, so it can be re-rendered later.
                verdict={
                    **c.verdict,
                    "window_path": str(window.path),
                    "window_start_s": window.start_s,
                },
                reject_reason=c.reject_reason,
                caption=c.caption,
                hashtags=c.hashtags,
                hook_options=c.hook_options,
                teaser_start_s=c.teaser_start_s,
                teaser_end_s=c.teaser_end_s,
            )
            store.create_clip(clip)
            if clip.status != "rendering":
                continue
            self._render(
                clip,
                window.path,
                window.start_s,
                words,
                settings,
                triggered_at,
                language=language,
            )
            self._last_clip_end = max(self._last_clip_end, c.end_s)
        if approved:
            return f"{approved} clip{'s' if approved != 1 else ''} made"
        if candidates:
            reasons = "; ".join(sorted({(c.reject_reason or "rejected")[:90] for c in candidates}))
            return f"judge said no ({reasons})"
        return "judge found nothing clip-worthy"

    def _render(
        self,
        clip: GtaClip,
        source: Path,
        source_start_s: float,
        words: list[Word],
        settings: GtaSettings,
        triggered_at: float,
        *,
        language: str = "",
    ) -> None:
        stream = store.get_stream(self.stream_id)
        if stream is None:
            return
        destination = clip_destination(clip, stream)
        clip_words = [w for w in words if w.end > clip.start_s and w.start < clip.end_s]
        work_dir = self._root / clip.id / "render"
        try:
            display_words, hook_title = localise_clip(
                clip, clip_words, settings, language=language, cache_dir=work_dir / "language"
            )
            flash = wants_flash_forward(clip, settings)
            teaser = (
                (clip.teaser_start_s, clip.teaser_end_s)
                if flash and clip.teaser_start_s is not None and clip.teaser_end_s is not None
                else None
            )
            qa_dict = render_and_qa(
                section=source,
                section_start_s=source_start_s,
                start_s=clip.start_s,
                end_s=clip.end_s,
                words=clip_words,
                hook_title=hook_title,
                destination=destination,
                work_dir=work_dir,
                settings=settings,
                cancelled=lambda: False,
                live=True,
                reaction_text=str(clip.verdict.get("reaction_text") or ""),
                display_words=display_words,
                teaser=teaser,
                flash_forward=flash,
                style_key=caption_look(clip, settings),
                narration_context=narration_context(clip, stream),
            )
        except Exception as exc:
            store.update_clip(clip.id, status="failed", error=str(exc)[:2000])
            self._problem("render", f"Rendering a clip failed: {exc}")
            return
        turnaround = time.time() - triggered_at
        qa_dict["turnaround_s"] = round(turnaround, 1)
        qa_dict["turnaround_ok"] = turnaround <= settings.live.turnaround_s
        destination.with_suffix(".txt").write_text(post_text(clip) + "\n", encoding="utf-8")
        passed = bool(qa_dict.get("passed"))
        store.update_clip(
            clip.id,
            status="ready" if passed else "qa_failed",
            path=str(destination),
            qa=qa_dict,
            error=None if passed else "Failed QA — see the checks.",
        )


_frame_lock = threading.Lock()
#: stream id -> (segment the frame came from, JPEG bytes)
_frame_cache: dict[str, tuple[str, bytes]] = {}


def latest_frame(stream_id: str) -> bytes | None:
    """A JPEG of the newest finished buffer segment (a near-live preview)."""
    from ..pipeline.ffmpeg import ffmpeg_path

    buffer_dir = work_root() / stream_id / "live" / "buffer"
    try:
        segments = sorted(buffer_dir.glob("seg_*.ts"), key=lambda f: f.stat().st_mtime)
    except OSError:  # a segment pruned mid-listing
        return _frame_cache.get(stream_id, ("", b""))[1] or None
    if not segments:
        return None
    # The newest segment is usually still being written; take the one before.
    segment = segments[-2] if len(segments) >= 2 else segments[-1]
    key = str(segment)
    with _frame_lock:
        cached = _frame_cache.get(stream_id)
        if cached and cached[0] == key:
            return cached[1]
        command = [
            ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-i", key,
            "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "5",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-",
        ]  # fmt: skip
        try:
            result = subprocess.run(command, capture_output=True, timeout=20, check=False)  # noqa: S603
        except (OSError, subprocess.TimeoutExpired):
            return cached[1] if cached else None
        if result.returncode != 0 or not result.stdout:
            return cached[1] if cached else None
        _frame_cache[stream_id] = (key, result.stdout)
        return result.stdout


def load_whisper():
    from ..config import load as load_app_settings

    return load_app_settings().whisper


class GtaLiveManager:
    def __init__(self) -> None:
        self._watches: dict[str, GtaLiveWatch] = {}
        self._lock = threading.Lock()

    def start(self, url: str) -> store.GtaStream:
        from .fetch import detect_platform

        stream = store.create_stream(url)
        name = name_from_url(url)
        store.update_stream(
            stream.id,
            platform=detect_platform(url),
            status="live_connecting",
            signals={"mode": "live"},
            **({"channel": name} if name else {}),
        )
        self._launch(stream.id, url)
        return store.get_stream(stream.id) or stream

    def restart(self, stream_id: str) -> None:
        stream = store.get_stream(stream_id)
        if stream is None:
            raise KeyError(stream_id)
        store.update_stream(stream_id, status="live_connecting", error=None)
        self._launch(stream_id, stream.url)

    def _launch(self, stream_id: str, url: str) -> None:
        with self._lock:
            old = self._watches.get(stream_id)
            if old is not None and old.alive:
                return
            watch = GtaLiveWatch(stream_id, url)
            self._watches[stream_id] = watch
        watch.start()

    def stop(self, stream_id: str) -> None:
        with self._lock:
            watch = self._watches.get(stream_id)
        if watch is not None:
            watch.stop()

    def judge_now(self, stream_id: str) -> bool:
        """Ask a running watch to judge its last minute now; False if not running."""
        with self._lock:
            watch = self._watches.get(stream_id)
        if watch is None or not watch.alive:
            return False
        watch.request_judge_now()
        return True

    def is_running(self, stream_id: str) -> bool:
        with self._lock:
            watch = self._watches.get(stream_id)
        return watch is not None and watch.alive

    def shutdown(self, timeout: float = 15.0) -> None:
        with self._lock:
            watches = list(self._watches.values())
        for watch in watches:
            watch.stop()
        for watch in watches:
            watch.join(timeout)

    def recover(self) -> None:
        """Rows left 'watching' by a previous run are marked stopped."""
        for stream in store.list_streams(statuses=frozenset({"live_connecting", "live_watching"})):
            store.update_stream(
                stream.id, status="live_stopped", detail="AutoClip restarted — start it again"
            )


live_manager = GtaLiveManager()
