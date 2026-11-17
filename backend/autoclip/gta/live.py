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
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..db.models import new_id
from ..pipeline import transcribe
from ..pipeline.transcript import Transcript, Word
from . import judging, render, signals, store
from .gate import gate
from .pipeline import (
    _model_cache,
    _provider_complete,
    clip_destination,
    post_text,
    render_and_qa,
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
#: Extra seconds of stream kept after the trigger point (the payoff).
PAYOFF_TAIL_S = 6.0


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
            self._buffer = RingBuffer(self._root / "buffer", retain_minutes=6.0, max_disk_mb=2048)

            def on_event(kind: str, detail: str) -> None:
                if kind == "ended":
                    self._ended = True
                    self._stop.set()

            self._capture = StreamCapture(self.url, self._buffer, on_event=on_event)
            self._capture.start()
            self._chat = _RecordingChat(self.url, self._clock)
            if self._chat.reader is not None:
                threading.Thread(
                    target=self._chat.reader.run,
                    args=(self._stop,),
                    daemon=True,
                    name="gta-live-chat",
                ).start()
            self._status("live_watching", "Listening")
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
        while not self._stop.wait(interval):
            now = self._clock()
            if now is None or (last_checked is not None and now <= last_checked):
                continue
            since = last_checked if last_checked is not None else max(0.0, now - interval)
            last_checked = now
            try:
                fired, why = self._listen(since, now)
            except Exception:
                log.warning("GTA Live listen pass failed.", exc_info=True)
                continue
            if not fired or now - self._last_clip_end < COOLDOWN_S:
                continue
            store.update_stream(self.stream_id, detail=f"Moment detected ({why}) — judging")
            try:
                self._evaluate(now, why, settings, triggered_at=time.time())
            except Exception:
                log.warning("GTA Live evaluation failed.", exc_info=True)
            store.update_stream(self.stream_id, detail="Listening")

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
        if len(history) >= 30 and len(recent):
            centre = float(np.median(history))
            spread = max(1.5, float(np.median(np.abs(history - centre))) * 1.4826)
            z = (float(np.max(recent)) - centre) / spread
            if z >= AUDIO_Z_TRIGGER:
                reasons.append(f"streamer lost it (+{z:.1f} sigma)")

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
        return bool(reasons), "; ".join(reasons)

    # -- judging + rendering ----------------------------------------------

    def _evaluate(
        self, now: float, why: str, settings: GtaSettings, *, triggered_at: float
    ) -> None:
        assert self._buffer is not None
        # Let the payoff land: wait for a few more seconds of capture.
        deadline = time.monotonic() + 30
        while (self._clock() or 0) < now + PAYOFF_TAIL_S and time.monotonic() < deadline:
            if self._stop.wait(1.0):
                return
        end = min(self._clock() or now, now + PAYOFF_TAIL_S)
        start = max(0.0, end - settings.live.context_s - PAYOFF_TAIL_S)
        clip_id = new_id()
        work = self._root / clip_id
        window = self._buffer.extract_window(start, end, work / "window.ts", anchor_s=now)

        audio = work / "window.wav"
        from ..pipeline import prepare

        prepare.extract_audio(window.path, audio)
        with gate.hold(live=True):
            piece = transcribe.transcribe(audio, load_whisper(), model_cache=_model_cache)
        words = [
            Word(w.text, w.start + window.start_s, w.end + window.start_s, w.speaker)
            for w in piece.words
        ]
        transcript = Transcript(words=words)
        if not words:
            return
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
        live_settings = settings.model_copy(update={"min_clip_s": min(settings.min_clip_s, 10.0)})

        async def judge() -> list[judging.Candidate]:
            found = await judging.find_candidates(
                transcript,
                [chunk],
                complete=complete,
                settings=live_settings,
                curve=None,
                silences=[],
                streamer=streamer,
                title=f"{title} (LIVE — moment flagged: {why})",
            )
            return await judging.final_round(
                transcript,
                found,
                complete=complete,
                settings=live_settings,
                streamer=streamer,
                title=title,
            )

        candidates = asyncio.run(judge())
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
                verdict=c.verdict,
                reject_reason=c.reject_reason,
                caption=c.caption,
                hashtags=c.hashtags,
            )
            store.create_clip(clip)
            if clip.status != "rendering":
                continue
            self._render(clip, window.path, window.start_s, words, settings, triggered_at)
            self._last_clip_end = max(self._last_clip_end, c.end_s)

    def _render(
        self,
        clip: GtaClip,
        source: Path,
        source_start_s: float,
        words: list[Word],
        settings: GtaSettings,
        triggered_at: float,
    ) -> None:
        stream = store.get_stream(self.stream_id)
        if stream is None:
            return
        destination = clip_destination(clip, stream)
        clip_words = [w for w in words if w.end > clip.start_s and w.start < clip.end_s]
        try:
            qa_dict = render_and_qa(
                section=source,
                section_start_s=source_start_s,
                start_s=clip.start_s,
                end_s=clip.end_s,
                words=clip_words,
                hook_title=clip.hook_title,
                destination=destination,
                work_dir=self._root / clip.id / "render",
                settings=settings,
                cancelled=lambda: False,
                live=True,
                reaction_text=str(clip.verdict.get("reaction_text") or ""),
            )
        except Exception as exc:
            store.update_clip(clip.id, status="failed", error=str(exc)[:2000])
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
        store.update_stream(
            stream.id,
            platform=detect_platform(url),
            status="live_connecting",
            signals={"mode": "live"},
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
