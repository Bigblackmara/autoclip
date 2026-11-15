"""Twitch chat signal reader — Live Watch's cheapest honest signal (A5).

When something actually happens on a stream, thousands of people type "KEKW"
inside the same ten seconds. This module turns that reflex into
DetectorSignals through Twitch's public, anonymous IRC-over-WebSocket
gateway: no account, no OAuth, no quota, one firehose we mostly ignore.

The architecture is a hard split, and the split is the point:

* PARSING + SCORING are pure functions/classes over strings and timestamps
  (``take_complete_lines``, ``parse_privmsg``, ``ChatRateScorer``). No socket,
  no clock of their own. Everything that decides *what is a signal* lives
  here and is unit-tested against synthetic lines.
* The SOCKET LOOP (``TwitchChatReader``) is deliberately thin: connect,
  handshake, JOIN, translate frames into scorer calls, answer PINGs, die
  quietly and reconnect with capped exponential backoff. It is not
  unit-tested because network layers are where tests go to lie; its
  contracts are structural — ``run()`` cannot raise (callbacks only), and it
  honours a ``threading.Event`` stop within one socket read.

Failure stories that shaped this file:

* Twitch batches several ``\\r\\n``-terminated IRC lines into one websocket
  text frame and splits others mid-line, so lines are re-assembled through a
  buffered ``take_complete_lines`` instead of a naive split.
* Moderated chats sit at an exactly constant message rate for minutes; the
  straight z-score divides by zero and turns one extra message into
  "infinite sigma". The scorer's spread floor (Poisson noise, half a message
  per window) keeps z finite and honest.
* The public gateway drops connections routinely (server-side restarts,
  route flaps). A chat signal is never allowed to become an operator
  incident: ``run()`` catches everything, reports through ``on_state``, and
  retries with capped exponential backoff.
* /me actions arrive as CTCP (``\\x01ACTION dances\\x01``) whose sentinel
  strips as punctuation; without that, every role-play emote counts as a
  caps word.

Why IRC and not Helix/EventSub: those need app credentials, tokens and
quotas for what is at heart one public read-only pipe. Anonymous
``justinfan`` logins are documented, free and stable — v1 scope.
"""

from __future__ import annotations

import contextlib
import logging
import math
import re
import statistics
import string
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from urllib.parse import urlparse

from .settings import LiveChatSettings
from .signals import DetectorSignal, SignalKind

try:
    # websocket-client ships as a transitive dep of streamlink (the [live]
    # extra); kept optional so the rest of the package import-clean without it.
    import websocket
except ImportError:  # pragma: no cover - bare installs only
    websocket = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

#: Twitch's documented anonymous IRC-over-WSS endpoint.
TWITCH_IRC_WS_URL = "wss://irc-ws.chat.twitch.tv:443"

#: Anonymous read-only nickname. Twitch server policy rejects PRIVMSG from
#: justinfan accounts, which suits a read-only signal reader perfectly.
ANON_NICK = "justinfan12345"

#: Evaluation bucket for message-rate scoring (seconds of stream time).
#: Settings' 60 s baseline_window_s is the *memory*; 10 s buckets are the
#: *resolution*, so the signal lands seconds after the moment, not a minute.
RATE_WINDOW_S = 10.0

#: Hypetrain velocity and CAPS-ratio horizon: hypetrain is configured in
#: msgs/min, so the message deque tracks a full minute of history.
VELOCITY_WINDOW_S = 60.0

#: Closed rate buckets needed before z-scoring is allowed. One prior bucket
#: is a number, not a baseline; a fresh reader must not spew signals.
MIN_BASELINE_WINDOWS = 2

#: Recent messages before the CAPS/emote ratio is trusted. "Per 100
#: messages" from a dozen samples is two people shouting, not a crowd.
MIN_HYPE_SAMPLE_MSGS = 20

#: One hypetrain signal per minute at most: A4's cooldown polices *after* a
#: signal; re-firing on every message would only inflate the window.
HYPE_TRAIN_REFRACTORY_S = 60.0

#: CAPS/emote spam re-arms after half a minute of continued shouting.
CAPS_REFRACTORY_S = 30.0

#: Backoff ceiling: 1, 2, 4, ... capped here on socket drops.
BACKOFF_CAP_S = 60.0

#: Socket read timeout; halves worst-case stop latency and paces PING
#: keepalives on an idle connection.
SOCKET_TIMEOUT_S = 15.0

#: Smallest rate spread (msgs per second) we will divide by. Below that a
#: deviation is one message — evidence, not statistics.
MIN_SPREAD_PER_WINDOW = 0.5

#: CAPS/emote strength scale: threshold-level spam lands just above the
#: aggregator's noise floor (0.4); ~3.3x the threshold reaches the 2.0 cap.
CAPS_STRENGTH_SCALE = 0.6

#: ~30 of the most-typed global emote words. Matching is case-insensitive.
#: Third-party-ecosystem names (KEKW, catJAM, peepo*) are in on purpose:
#: that is what people actually type during a hype train, regardless of
#: which extension renders it. This is a heuristic constant, not an API dump.
COMMON_EMOTES: tuple[str, ...] = (
    "5Head",
    "catJAM",
    "Clap",
    "COPIUM",
    "DansGame",
    "EleGiggle",
    "EZ",
    "FailFish",
    "FML",
    "Hypers",
    "Kappa",
    "KappaPride",
    "KEKW",
    "LUL",
    "LULW",
    "monkaS",
    "NotLikeThis",
    "OMEGALUL",
    "peepoGiggle",
    "peepoHappy",
    "peepoSad",
    "PJSalt",
    "Pog",
    "PogChamp",
    "PogChimp",
    "Poggers",
    "ResidentSleeper",
    "Sadge",
    "SeemsGood",
    "SwiftRage",
    "TriHard",
    "VoHiYo",
)

_EMOTE_WORDS: frozenset[str] = frozenset(e.lower() for e in COMMON_EMOTES)

#: Characters trimmed from message words before caps/emote matching: nick
#: punctuation, the CTCP sentinel, Twitch's zero-width space injections.
_WORD_STRIP = string.punctuation + "\x01\u200b\ufeff"

_TwitchChannel = re.compile(r"^[A-Za-z0-9_]{2,25}$")

#: First path segments that are Twitch site chrome, not channel pages.
_RESERVED_PATH_SEGMENTS = frozenset(
    {
        "activate",
        "directory",
        "downloads",
        "friends",
        "login",
        "p",
        "search",
        "settings",
        "signup",
        "team",
        "teams",
        "turbo",
        "videos",
    }
)

OnSignal = Callable[[DetectorSignal], None]
OnState = Callable[["ChatState", str], None]


class ChatState(StrEnum):
    """Lifecycle states the reader reports through ``on_state``."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    STOPPED = "stopped"


class ChatDisabled(Exception):
    """Chat cannot run for this source (not a Twitch channel page).

    This is a configuration fact, not a failure: the watcher keeps running,
    just without a chat signal. ``make_chat_reader``/``run_chat_reader``
    swallow it and report "no chat" (None / False) instead.
    """


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------


def channel_from_url(url: str) -> str:
    """Extract the Twitch channel name from a channel-page URL.

    Preconditions: ``url`` is the same string the watcher captures; it does
    not need to carry a scheme (``www.twitch.tv/x`` parses).

    Raises:
        ChatDisabled: for anything that is not a Twitch channel page — other
            platforms (youtube.com, kick.com), VODs (``/videos/<id>``),
            directory pages, bare domains, malformed URLs.
    """
    raw = (url or "").strip()
    if not raw:
        raise ChatDisabled("empty URL; no chat channel to read")
    if "://" not in raw:
        raw = "//" + raw
    try:
        parsed = urlparse(raw)
    except ValueError as exc:
        raise ChatDisabled(f"unreadable URL {url!r}") from exc
    host = parsed.hostname or ""
    if host != "twitch.tv" and not host.endswith(".twitch.tv"):
        found = host or "no host"
        raise ChatDisabled(f"chat is Twitch-only in v1; {found!r} has no chat reader")
    segments = [s for s in parsed.path.split("/") if s]
    if segments and segments[0] == "popout":
        segments = segments[1:]
    if not segments or segments[0].lower() in _RESERVED_PATH_SEGMENTS:
        raise ChatDisabled(f"twitch URL {url!r} is not a channel page; nothing to read")
    channel = segments[0]
    if not _TwitchChannel.match(channel):
        raise ChatDisabled(f"{channel!r} is not a valid Twitch channel name")
    return channel.lower()


def take_complete_lines(text: str) -> tuple[list[str], str]:
    """Split buffered socket text into complete IRC lines plus the remainder.

    Twitch terminates lines with ``\\r\\n`` but frames may also cut a line in
    half, and a trailing lone ``\\r`` must survive until the next chunk
    arrives (it may be half of a separator).
    """
    if "\n" not in text:
        return [], text
    parts = text.split("\n")
    tail = parts.pop()
    lines = [part[:-1] if part.endswith("\r") else part for part in parts]
    return lines, tail


def parse_privmsg(line: str) -> tuple[str, str] | None:
    """Parse one raw IRC line into ``(nick, text)``, or None if not PRIVMSG.

    Handles both tagged and untagged lines::

        @badge-info=;display-name=Nick :nick!nick@nick.tmi.twitch.tv PRIVMSG #chan :Hi
        :nick!nick@nick.tmi.twitch.tv PRIVMSG #chan :Hello: world

    Only the *leading* colon of the message body is a separator — colons
    inside the message survive, or every reaction line ("wtf?!?: https://…")
    would be mangled. IRCv3 tags are dropped: v1 scores what people typed,
    not how Twitch decorated it.
    """
    msg = line
    if not msg:
        return None
    if msg.startswith("@"):
        msg = msg.partition(" ")[2]
    if not msg:
        return None
    nick = ""
    if msg.startswith(":"):
        prefix, _, msg = msg.partition(" ")
        nick = prefix[1:].partition("!")[0]
    pieces = msg.split(" ", 2)
    if len(pieces) < 2 or pieces[0].upper() != "PRIVMSG":
        return None
    body = pieces[2] if len(pieces) > 2 else ""
    if body.startswith(":"):
        body = body[1:]
    return nick, body


def is_caps_word(word: str) -> bool:
    """True for shouty words: 3+ uppercase letters, all-caps (``KEKW``, ``GOOOO``).

    ``lol``, ``GG``, digits-only and mixed case are not hype — caps spam is
    about size, not volume.
    """
    cleaned = word.strip(_WORD_STRIP)
    uppers = sum(1 for ch in cleaned if ch.isupper())
    return uppers >= 3 and cleaned.isupper()


def is_emote_word(word: str) -> bool:
    """Case-insensitive membership in the embedded common-emote list."""
    return word.strip(_WORD_STRIP).lower() in _EMOTE_WORDS


def count_hype_words(text: str) -> int:
    """Hype words (caps or emote) in one message. Each word counts once.

    /me actions arrive wrapped in the CTCP sentinel (``\\x01ACTION ...\\x01``);
    the "ACTION" keyword itself is protocol, not shouting, and is skipped —
    or every role-play emote would count as a caps word.
    """
    count = 0
    for raw in text.split():
        if raw in ("\x01ACTION", "ACTION\x01", "\x01", "ACTION"):
            continue
        if is_caps_word(raw) or is_emote_word(raw):
            count += 1
    return count


def next_backoff(attempt: int, *, cap: float = BACKOFF_CAP_S, base: float = 1.0) -> float:
    """Exponential reconnect delay: base, 2·base, 4·base, ... capped at ``cap``.

    ``attempt`` is 1-based so the first drop waits ``base`` seconds, not zero.
    """
    return min(cap, base * (2 ** max(attempt - 1, 0)))


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------


class ChatRateScorer:
    """Deterministic PRIVMSG scoring: rate z-spikes, hypetrain, CAPS/emote spam.

    Pure: time arrives as ``t_s`` arguments (from the socket loop in
    production, synthetic stamps in tests), never from a wall clock.
    Message-rate scoring splits time into buckets (``window_s``) and
    z-scores each closed bucket's msgs/sec against the ``baseline_window_s``
    seconds of history before it — the same recipe as the audio detector
    (A2), with two chat-specific guards because chat counts are quantized:

    * flat baselines: spread falls back to Poisson noise
      (``sqrt(mean / window)``) and then to ``MIN_SPREAD_PER_WINDOW / window``
      so one extra message is never "infinite sigma";
    * quiet-chat floor: the *evaluated* bucket must carry at least
      ``floor_msgs_per_min`` msgs/min — three people in a silent chat are
      not a spike.

    Signals produced (strength normalised the way A4's Trigger expects:
    threshold == 1.0, capped at 2.0):

    * ``CHAT_RATE`` per closed bucket with z >= ``rate_z`` and rate over the
      floor; strength = min(2.0, z / rate_z).
    * ``CHAT_HYPE`` strength 2.0 the moment rolling velocity reaches
      ``hypetrain_msgs_per_min`` msgs/min (checked per message, refractory
      one minute — hypetrain is the one signal that outweighs everything).
    * ``CHAT_HYPE`` scaled CAPS/emote spam: hype words per 100 messages over
      the last minute crossing ``emotes_threshold``; strength scales
      0.6–2.0, so threshold-level shouting registers just above the
      aggregator's noise floor instead of equaling a hypetrain.

    Preconditions: ``t_s`` mostly non-decreasing (small skew is clamped into
    the open bucket); use from one thread, or hand the instance over.
    """

    def __init__(self, settings: LiveChatSettings, *, window_s: float = RATE_WINDOW_S) -> None:
        self.settings = settings
        self.window_s = float(window_s)
        # Closed ``(bucket_start, msgs_per_s)`` pairs inside the baseline horizon.
        self._rates: deque[tuple[float, float]] = deque()
        self._cur_start: float | None = None
        self._cur_count = 0
        # Message timestamps + per-message hype-word counts, last minute.
        self._msg_times: deque[float] = deque()
        self._hype_words: deque[int] = deque()
        self._last_hype_t: float | None = None
        self._last_caps_t: float | None = None

    def observe(self, t_s: float, nick: str, text: str) -> list[DetectorSignal]:
        """Account one PRIVMSG stamped at stream-time ``t_s``; may emit signals.

        ``nick`` is accepted for parity with ``parse_privmsg`` and richer
        evidence later; v1 scoring uses only the text.
        """
        signals: list[DetectorSignal] = []
        self._grid(t_s)
        self._advance(t_s, signals)
        t = max(t_s, self._cur_start)  # clock-skew guard
        self._cur_count += 1
        self._msg_times.append(t)
        self._hype_words.append(count_hype_words(text))
        self._prune_velocity(t)
        self._maybe_hypetrain(t, signals)
        return signals

    def tick(self, t_s: float) -> list[DetectorSignal]:
        """Advance time with no message: close stale buckets, age velocity.

        Without this, a burst followed by silence would never close its
        bucket, and the spike would be scored by nobody.
        """
        if self._cur_start is None:
            return []
        signals: list[DetectorSignal] = []
        self._advance(t_s, signals)
        self._prune_velocity(max(t_s, self._cur_start))
        return signals

    # -- internals ----------------------------------------------------------

    def _grid(self, t_s: float) -> None:
        """Anchor buckets on the fixed time grid (aligned across restarts)."""
        if self._cur_start is None:
            self._cur_start = math.floor(t_s / self.window_s) * self.window_s

    def _advance(self, t_s: float, signals: list[DetectorSignal]) -> None:
        assert self._cur_start is not None
        while t_s >= self._cur_start + self.window_s:
            self._close_bucket(self._cur_start, self._cur_count, signals)
            self._cur_start += self.window_s
            self._cur_count = 0

    def _prune_velocity(self, t: float) -> None:
        horizon = t - VELOCITY_WINDOW_S
        while self._msg_times and self._msg_times[0] < horizon:
            self._msg_times.popleft()
            self._hype_words.popleft()

    def _maybe_hypetrain(self, t: float, signals: list[DetectorSignal]) -> None:
        threshold = self.settings.hypetrain_msgs_per_min
        velocity = len(self._msg_times)  # msgs in the last minute == msgs/min
        if velocity < threshold:
            return
        if self._last_hype_t is None or t - self._last_hype_t >= HYPE_TRAIN_REFRACTORY_S:
            self._last_hype_t = t
            signals.append(
                DetectorSignal(
                    kind=SignalKind.CHAT_HYPE,
                    t_s=t,
                    strength=2.0,
                    peak_z=velocity / max(threshold, 1e-9),
                    detail=f"chat velocity {velocity} msgs/min >= {threshold:g} (hypetrain)",
                )
            )

    def _close_bucket(self, start: float, count: int, signals: list[DetectorSignal]) -> None:
        end = start + self.window_s
        rate = count / self.window_s
        # Baseline strictly before this bucket, limited to the horizon.
        while self._rates and self._rates[0][0] < start - self.settings.baseline_window_s:
            self._rates.popleft()
        prior = [pair[1] for pair in self._rates]
        if len(prior) >= MIN_BASELINE_WINDOWS:
            mean = statistics.fmean(prior)
            std = statistics.pstdev(prior)
            poisson = math.sqrt(mean / self.window_s) if mean > 0 else 0.0
            spread = max(std, poisson, MIN_SPREAD_PER_WINDOW / self.window_s)
            z = (rate - mean) / spread
            msgs_per_min = rate * 60.0
            if z >= self.settings.rate_z and msgs_per_min >= self.settings.floor_msgs_per_min:
                signals.append(
                    DetectorSignal(
                        kind=SignalKind.CHAT_RATE,
                        t_s=end,
                        strength=min(2.0, z / self.settings.rate_z),
                        peak_z=z,
                        detail=(
                            f"chat rate {rate:.2f} msgs/s vs baseline {mean:.2f} msgs/s "
                            f"(z={z:.1f} >= {self.settings.rate_z:g})"
                        ),
                    )
                )
            self._maybe_caps_spam(end, msgs_per_min, signals)
        self._rates.append((start, rate))

    def _maybe_caps_spam(
        self, end: float, msgs_per_min: float, signals: list[DetectorSignal]
    ) -> None:
        if self._last_caps_t is not None and end - self._last_caps_t < CAPS_REFRACTORY_S:
            return  # constant spam is one event, not one per bucket
        msgs = len(self._msg_times)
        if msgs < MIN_HYPE_SAMPLE_MSGS:
            return
        ratio = 100.0 * sum(self._hype_words) / msgs
        threshold = self.settings.emotes_threshold
        if ratio >= threshold and msgs_per_min >= self.settings.floor_msgs_per_min:
            self._last_caps_t = end
            signals.append(
                DetectorSignal(
                    kind=SignalKind.CHAT_HYPE,
                    t_s=end,
                    strength=min(2.0, CAPS_STRENGTH_SCALE * ratio / max(threshold, 1e-9)),
                    peak_z=ratio / max(threshold, 1e-9),
                    detail=f"CAPS/emote words {ratio:.1f} per 100 msgs (threshold {threshold:g})",
                )
            )


# ---------------------------------------------------------------------------
# Socket loop (thin; not unit-tested — no network in tests)
# ---------------------------------------------------------------------------


class TwitchChatReader:
    """Blocking chat reader for one Twitch channel.

    The manager (A8) runs it in a plain thread::

        reader = make_chat_reader(url, live_settings, on_signal=cb, clock=clock)
        if reader is not None:
            threading.Thread(target=reader.run, args=(stop_event,), daemon=True).start()

    Contracts:

    * ``run()`` never raises: every socket/protocol problem becomes an
      ``on_state(DISCONNECTED, detail)`` report followed by a capped
      exponential-backoff reconnect (1, 2, 4, ... max 60 s by default).
    * ``stop_event`` stops the loop within roughly one socket timeout;
      nothing is reported after the final ``STOPPED`` state.
    * ``t_s`` on every signal comes from the injected ``clock`` — pass one
      aligned with the stream's capture timeline (the default is
      ``time.monotonic``, i.e. process time, which will NOT match the
      buffer's epoch). A clock returning ``None`` pauses scoring (the
      stream is not being captured).

    Preconditions: ``channel`` is a non-empty Twitch channel name (use
    ``channel_from_url`` or ``make_chat_reader``, which validate it);
    ``on_signal``/``on_state`` must be thread-safe — they run on this
    reader's thread.
    """

    def __init__(
        self,
        channel: str,
        settings: LiveChatSettings | object,
        *,
        on_signal: OnSignal,
        on_state: OnState | None = None,
        ws_url: str = TWITCH_IRC_WS_URL,
        nick: str = ANON_NICK,
        clock: Callable[[], float | None] | None = None,
        backoff_base: float = 1.0,
        backoff_cap: float = BACKOFF_CAP_S,
        socket_timeout_s: float = SOCKET_TIMEOUT_S,
    ) -> None:
        if not channel or not channel.strip():
            raise ValueError("channel must be a non-empty Twitch channel name")
        self.channel = channel.strip().lstrip("#").lower()
        self._settings = _coerce_chat_settings(settings)
        self._on_signal = on_signal
        self._on_state = on_state
        self._ws_url = ws_url
        self._nick = nick
        self._clock = clock if clock is not None else time.monotonic
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._socket_timeout_s = socket_timeout_s
        # Set by _session on first message/JOIN echo; consulted by run() when
        # the session dies by exception, to decide reset-vs-escalate.
        self._session_read_traffic = False
        # The scorer survives reconnects: rate baselines are expensive to
        # rebuild and remain meaningful after a short drop.
        self._scorer = ChatRateScorer(self._settings)
        #: The live socket of the current session (closed by stop).
        self._ws = None
        self._ws_lock = threading.Lock()

    def _close_on_stop(self, stop_event: threading.Event, finished: threading.Event) -> None:
        """Shut the live socket down the moment ``stop_event`` is set.

        ``recv()`` only returns on traffic or after the socket timeout (15 s),
        so without this a stop waited up to that long — and the watcher's
        teardown waited with it. Closing the socket makes recv() fail now.
        """
        while not stop_event.wait(0.25):
            if finished.is_set():
                return
        with self._ws_lock:
            ws = self._ws
        if ws is None:
            return
        closer = getattr(ws, "shutdown", None) or getattr(ws, "close", None)
        if closer is not None:
            with contextlib.suppress(Exception):
                closer()

    def run(self, stop_event: threading.Event) -> None:
        """Serve chat until ``stop_event`` is set. Structurally cannot raise."""
        if websocket is None:  # bare install without the [live] extra
            self._report(
                ChatState.DISCONNECTED,
                "websocket-client (via streamlink) is not installed; "
                "install autoclip[live] to enable chat",
            )
            self._report(ChatState.STOPPED, "chat reader stopped: no websocket library")
            return
        finished = threading.Event()
        threading.Thread(
            target=self._close_on_stop,
            args=(stop_event, finished),
            name=f"chat-closer-{self.channel}",
            daemon=True,
        ).start()
        try:
            self._serve(stop_event)
        finally:
            finished.set()

    def _serve(self, stop_event: threading.Event) -> None:
        attempt = 0
        while not stop_event.is_set():
            saw_traffic = False
            detail = "socket closed"
            try:
                saw_traffic = self._session(stop_event)
            except Exception as exc:  # noqa: BLE001 — nothing may escape
                if stop_event.is_set():
                    break
                # A session that already read messages was healthy; Twitch
                # routinely drops long-lived anonymous connections, so its
                # death resets (not escalates) the backoff ladder.
                saw_traffic = self._session_read_traffic
                detail = f"{type(exc).__name__}: {exc}"[:300]
            if stop_event.is_set():
                break
            # Backoff ladder: failures in a row. A session that read traffic
            # was healthy — Twitch routinely drops long-lived anonymous
            # connections — so its death restarts the ladder at the base
            # delay (attempt 1) instead of escalating.
            attempt = 1 if saw_traffic else attempt + 1
            delay = next_backoff(attempt, cap=self._backoff_cap, base=self._backoff_base)
            self._report(
                ChatState.DISCONNECTED,
                f"{detail}; reconnecting in {delay:g}s (attempt {attempt})",
            )
            if stop_event.wait(delay):
                break
        self._report(ChatState.STOPPED, "chat reader stopped")

    # -- internals ----------------------------------------------------------

    def _session(self, stop_event: threading.Event) -> bool:
        """One connection, handshake to drop. True if it ever read traffic."""
        self._session_read_traffic = False
        self._report(ChatState.CONNECTING, f"connecting to {self._ws_url} for #{self.channel}")
        ws = websocket.create_connection(self._ws_url, timeout=self._socket_timeout_s)
        with self._ws_lock:
            self._ws = ws
        if stop_event.is_set():
            # Stopped while connecting: the closer may have looked already.
            with self._ws_lock:
                self._ws = None
            with contextlib.suppress(Exception):
                ws.close()
            return False
        saw_traffic = False
        tail = ""
        joined = False
        try:
            # Anonymous read-only registration: NICK only (Twitch needs no
            # PASS for justinfan), then JOIN. The server answers with 001 and
            # a JOIN echo; everything before PRIVMSG is noise we ignore.
            ws.send(f"NICK {self._nick}\r\n")
            ws.send(f"JOIN #{self.channel}\r\n")
            while not stop_event.is_set():
                try:
                    data = ws.recv()
                except websocket.WebSocketTimeoutException:
                    # Idle socket: keepalive so NATs and the gateway both
                    # remember we exist, then go back to waiting.
                    ws.send("PING :autoclip\r\n")
                    continue
                if not data:
                    continue
                if isinstance(data, bytes):
                    data = data.decode("utf-8", "replace")
                lines, tail = take_complete_lines(tail + data)
                for line in lines:
                    if not line:
                        continue
                    if line.startswith("PING"):
                        # Mirror the token back exactly; heartbeats are what
                        # keeps this connection beating idle timeouts.
                        ws.send("PONG" + line.removeprefix("PING"))
                        continue
                    if not joined and self._is_join_echo(line):
                        joined = True
                        saw_traffic = True
                        self._session_read_traffic = True
                        self._report(ChatState.CONNECTED, f"joined #{self.channel}")
                        continue
                    parsed = parse_privmsg(line)
                    if parsed is None:
                        continue
                    saw_traffic = True
                    self._session_read_traffic = True
                    nick, text = parsed
                    self._observe(nick, text)
            return saw_traffic or joined
        finally:
            with self._ws_lock:
                self._ws = None
            with contextlib.suppress(Exception):
                ws.close()

    def _observe(self, nick: str, text: str) -> None:
        """Score one chat message at the clock's stream time.

        A clock returning ``None`` means the stream is not being captured
        (offline, reconnecting): the message is not scored. Stream time is
        frozen then, so every offline message would pile into one bucket
        and fire a spurious spike — a paid judge call — on reconnect.
        """
        t_s = self._clock()
        if t_s is None:
            return
        for signal in self._scorer.observe(t_s, nick, text):
            self._emit(signal)

    def _is_join_echo(self, line: str) -> bool:
        msg = line
        if msg.startswith("@"):
            msg = msg.partition(" ")[2]
        if msg.startswith(":"):
            msg = msg.partition(" ")[2]
        head = msg.split(" ", 2)
        return (
            len(head) >= 2 and head[0].upper() == "JOIN" and head[1].lower() == f"#{self.channel}"
        )

    def _emit(self, signal: DetectorSignal) -> None:
        try:
            self._on_signal(signal)
        except Exception:  # noqa: BLE001 — a callback bug must not kill the reader
            log.exception("on_signal callback failed for %s", signal.kind)

    def _report(self, state: ChatState, detail: str = "") -> None:
        if self._on_state is None:
            return
        try:
            self._on_state(state, detail)
        except Exception:  # noqa: BLE001 — same contract as on_signal
            log.exception("on_state callback failed (%s)", state)


# ---------------------------------------------------------------------------
# Manager entry points (A8 wires these)
# ---------------------------------------------------------------------------


def _coerce_chat_settings(settings: LiveChatSettings | object) -> LiveChatSettings:
    """Accept a LiveChatSettings, or anything with ``.chat`` / ``.detection.chat``."""
    if isinstance(settings, LiveChatSettings):
        return settings
    chat = getattr(settings, "chat", None)
    if isinstance(chat, LiveChatSettings):
        return chat
    detection = getattr(settings, "detection", None)
    chat = getattr(detection, "chat", None)
    if isinstance(chat, LiveChatSettings):
        return chat
    raise TypeError(
        f"settings must be LiveChatSettings or carry .detection.chat; got {type(settings).__name__}"
    )


def make_chat_reader(
    url: str,
    settings: LiveChatSettings | object,
    *,
    on_signal: OnSignal,
    on_state: OnState | None = None,
    clock: Callable[[], float] | None = None,
) -> TwitchChatReader | None:
    """Build a reader for ``url``, or None when chat is off/unavailable.

    None — not an exception — means "no chat signal for this stream": wrong
    platform (YouTube/Kick chat is out of v1 scope on purpose) or chat
    disabled in settings. The stream keeps running without the signal.
    """
    chat = _coerce_chat_settings(settings)
    if not chat.enabled:
        return None
    try:
        channel = channel_from_url(url)
    except ChatDisabled as exc:
        log.debug("chat disabled for %r: %s", url, exc)
        return None
    return TwitchChatReader(channel, chat, on_signal=on_signal, on_state=on_state, clock=clock)


def run_chat_reader(
    url: str,
    settings: LiveChatSettings | object,
    *,
    on_signal: OnSignal,
    stop_event: threading.Event,
    on_state: OnState | None = None,
    clock: Callable[[], float] | None = None,
) -> bool:
    """Manager thread entry point: serve chat for ``url`` until stopped.

    Returns True if a reader actually ran, False when this URL has no chat
    (non-Twitch platform or chat disabled). Blocking; never raises; the only
    way out is ``stop_event``.
    """
    reader = make_chat_reader(url, settings, on_signal=on_signal, on_state=on_state, clock=clock)
    if reader is None:
        return False
    reader.run(stop_event)
    return True


__all__ = [
    "ANON_NICK",
    "BACKOFF_CAP_S",
    "COMMON_EMOTES",
    "ChatDisabled",
    "ChatRateScorer",
    "ChatState",
    "TwitchChatReader",
    "TWITCH_IRC_WS_URL",
    "channel_from_url",
    "count_hype_words",
    "is_caps_word",
    "is_emote_word",
    "make_chat_reader",
    "next_backoff",
    "parse_privmsg",
    "run_chat_reader",
    "take_complete_lines",
]
