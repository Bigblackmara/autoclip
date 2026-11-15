"""Twitch chat signal reader (A5) — parsing and scoring against synthetic lines.

Everything here feeds synthetic IRC text into the pure functions; nothing
touches the network. The socket loop's *contracts* (never raises, honours
stop, capped backoff, callback bugs are contained) are tested with a fake
``websocket`` module injected into ``chat_signals``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from autoclip.live import chat_signals as cs
from autoclip.live.chat_signals import (
    ANON_NICK,
    BACKOFF_CAP_S,
    COMMON_EMOTES,
    RATE_WINDOW_S,
    ChatDisabled,
    ChatRateScorer,
    ChatState,
    channel_from_url,
    count_hype_words,
    is_caps_word,
    is_emote_word,
    make_chat_reader,
    next_backoff,
    parse_privmsg,
    run_chat_reader,
    take_complete_lines,
)
from autoclip.live.settings import LiveChatSettings
from autoclip.live.signals import SignalKind


def privmsg(nick: str, text: str, channel: str = "chan") -> str:
    """One raw tagged PRIVMSG line, shaped like real Twitch frames."""
    return (
        f"@badge-info=;color=#FF0000;display-name={nick} :{nick}!{nick}@{nick}.tmi.twitch.tv "
        f"PRIVMSG #{channel} :{text}"
    )


def frame(*lines: str) -> str:
    """Join IRC lines into one websocket text frame, Twitch-style CRLF."""
    return "\r\n".join(lines) + "\r\n"


def default_chat() -> LiveChatSettings:
    return LiveChatSettings()


def quiet_chat() -> LiveChatSettings:
    """Hypetrain reachable in tests without sending hundreds of messages."""
    return LiveChatSettings(hypetrain_msgs_per_min=20.0, emotes_threshold=2.0)


def feed(scorer: ChatRateScorer, messages: list[tuple[float, str]]) -> list[Any]:
    """Observe ``(t, text)`` pairs; collect every emitted signal."""
    out: list[Any] = []
    for t, text in messages:
        out.extend(scorer.observe(t, "viewer", text))
    return out


class TestTakeCompleteLines:
    def test_multi_line_frame(self) -> None:
        lines, tail = take_complete_lines("PING :tmi\r\nPRIVMSG #c :hi\r\n")
        assert lines == ["PING :tmi", "PRIVMSG #c :hi"]
        assert tail == ""

    def test_partial_line_is_kept_as_tail(self) -> None:
        lines, tail = take_complete_lines("PRIVMSG #c :he")
        assert lines == []
        assert tail == "PRIVMSG #c :he"

    def test_split_across_frames_reassembles(self) -> None:
        first = take_complete_lines(":n!n@n.tmi.twitch.tv PRIVMSG #c :hello w")
        assert first[0] == []
        lines, tail = take_complete_lines(first[1] + "orld\r\n")
        assert lines == [":n!n@n.tmi.twitch.tv PRIVMSG #c :hello world"]
        assert tail == ""

    def test_lone_trailing_cr_survives(self) -> None:
        # A trailing \r may be half of a separator arriving in the next frame.
        lines, tail = take_complete_lines("PRIVMSG #c :hi\r")
        assert lines == []
        assert tail == "PRIVMSG #c :hi\r"
        lines, tail = take_complete_lines(tail + "\n")
        assert lines == ["PRIVMSG #c :hi"]

    def test_bare_lf_is_tolerated(self) -> None:
        lines, _ = take_complete_lines("a\nb\n")
        assert lines == ["a", "b"]


class TestParsePrivmsg:
    def test_tagged_line(self) -> None:
        nick, text = parse_privmsg(privmsg("viewer", "KEKW that was insane"))
        assert nick == "viewer"
        assert text == "KEKW that was insane"

    def test_untagged_line(self) -> None:
        nick, text = parse_privmsg(":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #chan :hello")
        assert nick == "viewer"
        assert text == "hello"

    def test_colons_inside_message_survive(self) -> None:
        _, text = parse_privmsg(privmsg("v", "wtf?!?: https://twitch.tv/x"))
        assert text == "wtf?!?: https://twitch.tv/x"

    def test_empty_message(self) -> None:
        assert parse_privmsg(privmsg("v", "")) == ("v", "")

    def test_non_privmsg_lines_are_none(self) -> None:
        assert parse_privmsg(":tmi.twitch.tv 001 justinfan12345 :Welcome, GLHF!") is None
        assert parse_privmsg("PING :tmi.twitch.tv") is None
        assert parse_privmsg("") is None
        assert parse_privmsg(":n!n@n JOIN #chan") is None

    def test_ctcp_action_is_not_hype(self) -> None:
        # /me actions arrive wrapped in \x01ACTION ... \x01.
        _, text = parse_privmsg(privmsg("v", "\x01ACTION dances\x01"))
        assert count_hype_words(text) == 0


class TestHypeWordHelpers:
    def test_caps_words(self) -> None:
        assert is_caps_word("KEKW")
        assert is_caps_word("GOOOO")
        assert is_caps_word("WOW!!!")  # punctuation stripped
        assert not is_caps_word("lol")
        assert not is_caps_word("GG")  # under 3 uppercase letters
        assert not is_caps_word("12345")  # digits are not shouting
        assert not is_caps_word("Kappa")

    def test_emote_words_case_insensitive(self) -> None:
        assert is_emote_word("Kappa")
        assert is_emote_word("kappa")
        assert is_emote_word("PogChamp!")
        assert not is_emote_word("notAnEmote")

    def test_emote_list_is_about_thirty_entries(self) -> None:
        assert 25 <= len(COMMON_EMOTES) <= 40
        assert "Kappa" in COMMON_EMOTES
        assert "PogChimp" in COMMON_EMOTES

    def test_count_hype_words(self) -> None:
        assert count_hype_words("KEKW PogChamp that was INSANE lol") == 3
        assert count_hype_words("nothing hype here") == 0


class TestChannelFromUrl:
    def test_twitch_channel_page(self) -> None:
        assert channel_from_url("https://www.twitch.tv/somechannel") == "somechannel"
        assert channel_from_url("https://twitch.tv/SomeChannel") == "somechannel"
        assert channel_from_url("https://m.twitch.tv/chan/about") == "chan"

    def test_no_scheme(self) -> None:
        assert channel_from_url("www.twitch.tv/chan") == "chan"
        assert channel_from_url("twitch.tv/chan") == "chan"

    def test_popout_url(self) -> None:
        assert channel_from_url("https://www.twitch.tv/popout/chan/chat") == "chan"

    def test_youtube_is_disabled_not_error(self) -> None:
        with pytest.raises(ChatDisabled):
            channel_from_url("https://www.youtube.com/watch?v=abc")

    def test_kick_is_disabled(self) -> None:
        with pytest.raises(ChatDisabled):
            channel_from_url("https://kick.com/somechannel")

    def test_non_channel_twitch_paths_are_disabled(self) -> None:
        for url in (
            "https://www.twitch.tv/videos/123456789",
            "https://www.twitch.tv/directory",
            "https://www.twitch.tv/",
            "https://www.twitch.tv",
            "https://www.twitch.tv/settings/profile",
        ):
            with pytest.raises(ChatDisabled):
                channel_from_url(url)

    def test_invalid_channel_name(self) -> None:
        with pytest.raises(ChatDisabled):
            channel_from_url("https://www.twitch.tv/a")  # too short
        with pytest.raises(ChatDisabled):
            channel_from_url("https://www.twitch.tv/bad*name")

    def test_empty_url(self) -> None:
        with pytest.raises(ChatDisabled):
            channel_from_url("")
        with pytest.raises(ChatDisabled):
            channel_from_url("   ")


class TestNextBackoff:
    def test_exponential_sequence(self) -> None:
        assert [next_backoff(n) for n in range(1, 7)] == [1, 2, 4, 8, 16, 32]

    def test_capped_at_60(self) -> None:
        assert next_backoff(7) == BACKOFF_CAP_S
        assert next_backoff(50) == BACKOFF_CAP_S

    def test_custom_base_and_cap(self) -> None:
        assert next_backoff(1, cap=10.0, base=3.0) == 3.0
        assert next_backoff(3, cap=10.0, base=3.0) == 10.0


# ---------------------------------------------------------------------------
# Pure scoring — ChatRateScorer
# ---------------------------------------------------------------------------


class TestRateSpike:
    def test_quiet_then_burst_fires_chat_rate(self) -> None:
        scorer = ChatRateScorer(default_chat())
        msgs: list[tuple[float, str]] = []
        # Two baseline buckets at 1 msg/s (60 msgs/min, above the 10/min floor).
        for bucket in (0, 1):
            for i in range(10):
                msgs.append((bucket * 10.0 + i + 0.1, "hello"))
        # Burst bucket at 8 msgs/s — the message stream must reach into the
        # *next* bucket so the burst bucket actually closes.
        for i in range(80):
            msgs.append((20.0 + i * 0.12, "WUT"))
        msgs.append((31.0, "hello"))  # closes the 20–30 bucket
        signals = feed(scorer, msgs)
        rate = [s for s in signals if s.kind is SignalKind.CHAT_RATE]
        assert rate, f"expected CHAT_RATE, got {[(s.kind, s.detail) for s in signals]}"
        assert rate[0].strength >= 1.0  # z >= rate_z normalises to >= 1.0
        assert rate[0].peak_z >= default_chat().rate_z
        assert rate[0].t_s >= 20.0  # stamped at/after the burst bucket

    def test_constant_rate_never_fires(self) -> None:
        # Flat chat is the zero-variance trap: the spread floor keeps z finite
        # — and constant is not a spike even when the floor lets z exist.
        scorer = ChatRateScorer(default_chat())
        msgs: list[tuple[float, str]] = []
        for bucket in range(5):
            for i in range(10):
                msgs.append((bucket * 10.0 + i, "hey"))  # constant 1 msg/s
        signals = feed(scorer, msgs)
        assert [s for s in signals if s.kind is SignalKind.CHAT_RATE] == []

    def test_below_floor_msgs_is_not_a_spike(self) -> None:
        # A "burst" of 5 msgs in a bucket after quiet buckets: the evaluated
        # bucket's 30 msgs/min clears the 10/min floor... so instead push the
        # floor check with a burst that is statistically huge but thin in
        # absolute terms: 30 msgs/s threshold needs rate*60 >= 10, so 1 msg
        # in a 10 s bucket (6 msgs/min) must stay silent despite z >= 3.
        scorer = ChatRateScorer(LiveChatSettings(rate_z=3.0))
        msgs = [(i, "hi") for i in range(2)]  # bucket 0: 0.2 msgs/s baseline
        msgs += [(10.0, "hi"), (11.0, "hi")]  # bucket 1: same rate, no burst
        msgs += [(20.0, "hi")]  # bucket 2: 0.1 msgs/s — below floor, z huge
        msgs += [(30.5, "hi")]  # close bucket 2
        signals = feed(scorer, msgs)
        assert [s for s in signals if s.kind is SignalKind.CHAT_RATE] == []

    def test_no_signal_before_baseline_exists(self) -> None:
        scorer = ChatRateScorer(default_chat())
        msgs = [(i * 0.05, "hi") for i in range(100)]  # one busy bucket only
        signals = feed(scorer, msgs)
        assert [s for s in signals if s.kind is SignalKind.CHAT_RATE] == []

    def test_tick_closes_bucket_after_silence(self) -> None:
        # Burst then silence: without tick() the spike bucket never closes.
        scorer = ChatRateScorer(default_chat())
        msgs: list[tuple[float, str]] = []
        for bucket in (0, 1):
            for i in range(10):
                msgs.append((bucket * 10.0 + i + 0.1, "hi"))
        for i in range(80):  # 8 msgs/s burst inside bucket 2
            msgs.append((20.0 + i * 0.12, "WUT"))
        signals = feed(scorer, msgs)
        assert [s for s in signals if s.kind is SignalKind.CHAT_RATE] == []
        signals += scorer.tick(45.0)  # silence closes bucket 2
        assert any(s.kind is SignalKind.CHAT_RATE for s in signals)

    def test_strength_capped_at_two(self) -> None:
        scorer = ChatRateScorer(default_chat())
        msgs: list[tuple[float, str]] = []
        for bucket in (0, 1):
            for i in range(10):
                msgs.append((bucket * 10.0 + i + 0.1, "hi"))
        for i in range(200):  # 20 msgs/s — z is enormous
            msgs.append((20.0 + i * 0.04, "WUT"))
        msgs.append((31.0, "hi"))
        signals = feed(scorer, msgs)
        rate = [s for s in signals if s.kind is SignalKind.CHAT_RATE]
        assert rate
        assert all(s.strength <= 2.0 for s in rate)

    def test_clock_skew_is_clamped(self) -> None:
        # A message stamped slightly before the current bucket start must not
        # corrupt the open bucket index.
        scorer = ChatRateScorer(default_chat())
        out = feed(scorer, [(5.0, "hi"), (4.5, "hi"), (15.5, "hi"), (25.5, "hi")])
        assert isinstance(out, list)  # no crash, no phantom signals
        assert [s for s in out if s.kind is SignalKind.CHAT_RATE] == []


class TestHypeTrain:
    def test_velocity_burst_fires_hype_at_2(self) -> None:
        scorer = ChatRateScorer(quiet_chat())
        signals = feed(scorer, [(i * 0.5, "hi") for i in range(20)])  # 20 msgs in 10 s
        hype = [s for s in signals if s.kind is SignalKind.CHAT_HYPE and "velocity" in s.detail]
        assert hype, f"expected hypetrain, got {[(s.kind, s.detail) for s in signals]}"
        assert hype[0].strength == 2.0

    def test_hypetrain_refractory_one_per_minute(self) -> None:
        scorer = ChatRateScorer(quiet_chat())
        msgs: list[tuple[float, str]] = []
        for wave in range(3):  # three bursts inside the same minute
            msgs += [(wave * 3.0 + i * 0.05, "GO") for i in range(20)]
        signals = feed(scorer, msgs)
        hype = [s for s in signals if s.kind is SignalKind.CHAT_HYPE and "velocity" in s.detail]
        assert len(hype) == 1

    def test_velocity_re_fires_after_a_minute(self) -> None:
        scorer = ChatRateScorer(quiet_chat())
        msgs: list[tuple[float, str]] = []
        msgs += [(i * 0.05, "GO") for i in range(20)]  # burst at t≈0
        msgs += [(70.0 + i * 0.05, "GO") for i in range(20)]  # burst at t≈70
        signals = feed(scorer, msgs)
        hype = [s for s in signals if s.kind is SignalKind.CHAT_HYPE and "velocity" in s.detail]
        assert len(hype) == 2

    def test_below_threshold_no_hype(self) -> None:
        scorer = ChatRateScorer(quiet_chat())
        signals = feed(scorer, [(i * 1.0, "hi") for i in range(10)])
        assert [s for s in signals if s.kind is SignalKind.CHAT_HYPE] == []


class TestCapsEmoteSpam:
    def test_caps_spam_fires_scaled_hype(self) -> None:
        # Default settings: hypetrain needs 600 msgs/min, so a 25-msg burst
        # can't masquerade as velocity hype; emotes_threshold is 2.0/100.
        scorer = ChatRateScorer(default_chat())
        msgs: list[tuple[float, str]] = []
        # Three buckets: two quiet ones to build a baseline (the scorer needs
        # MIN_BASELINE_WINDOWS=2 closed buckets first), then a bucket holding
        # 2 emote words among its 4 messages. Across the 60 s velocity window:
        # 44 msgs, 2 hype words -> 4.5 per 100, i.e. ~2.3x the 2.0 threshold,
        # so the scaled strength lands mid-range (0.6 x 2.3 ~= 1.4).
        for i in range(20):
            msgs.append((i * 0.45, "nice countdown keep it going"))
        for i in range(20):
            msgs.append((10.0 + i * 0.45, "gg wp everyone"))
        msgs.append((21.0, "hi"))
        msgs.append((23.0, "hi"))
        msgs.append((25.0, "Kappa"))
        msgs.append((27.0, "Poggers"))
        msgs.append((31.0, "hello"))  # closes the 20-30 bucket
        signals = feed(scorer, msgs)
        caps = [s for s in signals if s.kind is SignalKind.CHAT_HYPE and "CAPS" in s.detail]
        assert caps, f"expected CAPS signal, got {[(s.kind, s.detail) for s in signals]}"
        assert 0.5 < caps[0].strength < 2.0  # scaled, above the noise floor
        assert caps[0].peak_z >= default_chat().emotes_threshold

    def test_extreme_caps_ratio_caps_at_hypetrain_strength(self) -> None:
        # A screaming wall of caps is, for v1 purposes, as strong as chat gets:
        # the strength floor at 2.0 matches how A4 treats CHAT_HYPE signals.
        scorer = ChatRateScorer(quiet_chat())
        msgs: list[tuple[float, str]] = []
        for i in range(30):
            msgs.append((i * 0.9, "KEKW POG THAT WAS AMAZING"))
        msgs.append((31.0, "hello"))
        signals = feed(scorer, msgs)
        caps = [s for s in signals if s.kind is SignalKind.CHAT_HYPE and "CAPS" in s.detail]
        assert caps
        assert caps[0].strength == 2.0

    def test_normal_chat_no_caps_signal(self) -> None:
        scorer = ChatRateScorer(quiet_chat())
        msgs = [(i * 0.4, "gg that was a nice play wp") for i in range(40)]
        msgs.append((18.0, "hi"))
        signals = feed(scorer, msgs)
        assert [s for s in signals if s.kind is SignalKind.CHAT_HYPE and "CAPS" in s.detail] == []

    def test_small_sample_never_flags(self) -> None:
        # 3 shouts among 4 messages is 75/100 — but 4 messages is not a crowd.
        scorer = ChatRateScorer(quiet_chat())
        msgs = [(i * 0.3, "KEKW POG CAPS") for i in range(4)]
        msgs.append((10.5, "hi"))
        signals = feed(scorer, msgs)
        assert [s for s in signals if s.kind is SignalKind.CHAT_HYPE and "CAPS" in s.detail] == []


# ---------------------------------------------------------------------------
# Socket loop contracts — fake websocket module, zero network
# ---------------------------------------------------------------------------


class FakeSocketClosed(RuntimeError):
    """Connection-dead stand-in (websocket's connection-closed error)."""


class FakeTimeout(Exception):
    """Idle-socket stand-in for websocket's WebSocketTimeoutException."""


class FakeWebSocket:
    """Scripted websocket: queued frames, recorded sends, dies when empty."""

    def __init__(self, frames: list[str] | None = None) -> None:
        self.frames = list(frames or [])
        self.sent: list[str] = []
        self.closed = False

    def send(self, data: str) -> None:
        self.sent.append(data)

    def recv(self) -> str:
        if self.frames:
            return self.frames.pop(0)
        raise FakeSocketClosed("connection reset by peer")

    def close(self) -> None:
        self.closed = True


class FakeWSModule:
    """Duck-typed stand-in for the websocket-client module (refuses to dial).

    Timeout and closure are DISTINCT classes here, mirroring the real library:
    conflating them turns every dropped connection into an endless
    keepalive loop (a bug this test suite caught in an earlier fake).
    """

    WebSocketTimeoutException = FakeTimeout

    @staticmethod
    def create_connection(url: str, timeout: float) -> FakeWebSocket:
        raise AssertionError(f"unexpected real connection to {url}")


@pytest.fixture
def no_real_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point chat_signals at a fake websocket module that never dials out."""
    monkeypatch.setattr(cs, "websocket", FakeWSModule)


class TestMakeChatReader:
    def test_twitch_url_builds_reader(self, no_real_websocket: None) -> None:
        reader = make_chat_reader(
            "https://twitch.tv/chan", default_chat(), on_signal=lambda s: None
        )
        assert reader is not None
        assert reader.channel == "chan"

    def test_non_twitch_returns_none_not_error(self, no_real_websocket: None) -> None:
        assert (
            make_chat_reader("https://youtu.be/x", default_chat(), on_signal=lambda s: None) is None
        )
        assert (
            make_chat_reader("https://kick.com/chan", default_chat(), on_signal=lambda s: None)
            is None
        )

    def test_disabled_settings_returns_none(self, no_real_websocket: None) -> None:
        assert (
            make_chat_reader(
                "https://twitch.tv/chan",
                LiveChatSettings(enabled=False),
                on_signal=lambda s: None,
            )
            is None
        )

    def test_accepts_full_live_settings_tree(self, no_real_websocket: None) -> None:
        from autoclip.live.settings import LiveSettings

        reader = make_chat_reader(
            "https://twitch.tv/chan", LiveSettings(), on_signal=lambda s: None
        )
        assert reader is not None

    def test_bad_settings_type_raises(self, no_real_websocket: None) -> None:
        with pytest.raises(TypeError):
            make_chat_reader("https://twitch.tv/chan", object(), on_signal=lambda s: None)


class TestReaderConstruction:
    def test_channel_hash_prefix_is_normalised(self, no_real_websocket: None) -> None:
        reader = cs.TwitchChatReader("#Chan", default_chat(), on_signal=lambda s: None)
        assert reader.channel == "chan"

    def test_empty_channel_is_value_error(self, no_real_websocket: None) -> None:
        with pytest.raises(ValueError, match="channel"):
            cs.TwitchChatReader("", default_chat(), on_signal=lambda s: None)


class TestRunLoopContracts:
    def test_stop_before_start_exits_cleanly(self, no_real_websocket: None) -> None:
        signals: list[Any] = []
        states: list[tuple[ChatState, str]] = []

        def on_state(state: ChatState, detail: str = "") -> None:
            states.append((state, detail))

        reader = cs.TwitchChatReader(
            "chan", default_chat(), on_signal=signals.append, on_state=on_state
        )
        stop = threading.Event()
        stop.set()
        reader.run(stop)
        # Pre-stopped: the loop body never ran, but the final STOPPED state
        # is still reported exactly once (the manager counts on it).
        assert [state for state, _ in states] == [ChatState.STOPPED]
        assert signals == []

    def test_missing_websocket_library_reports_and_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cs, "websocket", None)
        states: list[tuple[ChatState, str]] = []

        def on_state(state: ChatState, detail: str = "") -> None:
            states.append((state, detail))

        reader = cs.TwitchChatReader(
            "chan", default_chat(), on_signal=lambda s: None, on_state=on_state
        )
        reader.run(threading.Event())
        assert states[0][0] is ChatState.DISCONNECTED
        assert "install" in states[0][1]
        assert states[-1][0] is ChatState.STOPPED

    def test_run_swallows_socket_failures_and_retries(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every dial fails; the loop must report DISCONNECTED and keep retrying
        # with capped backoff instead of raising out of run().
        created = []

        class AlwaysDown(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                created.append(url)
                raise FakeSocketClosed("dns broke")

        monkeypatch.setattr(cs, "websocket", AlwaysDown)
        disconnects = 0
        stop = threading.Event()

        def on_state(state: ChatState, detail: str = "") -> None:
            nonlocal disconnects
            if state is ChatState.DISCONNECTED:
                disconnects += 1
                if disconnects >= 3:
                    stop.set()

        reader = cs.TwitchChatReader(
            "chan",
            default_chat(),
            on_signal=lambda s: None,
            on_state=on_state,
            backoff_base=0.01,
            backoff_cap=0.02,
        )
        reader.run(stop)  # must not raise
        assert len(created) >= 3
        assert disconnects == 3

    def test_attempt_numbers_passed_to_backoff_grow(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        class Down(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                raise FakeSocketClosed("nope")

        def fake_next_backoff(attempt: int, **_kw: Any) -> float:
            attempts.append(attempt)
            return 0.0  # retry instantly; the loop logic is what's under test

        monkeypatch.setattr(cs, "websocket", Down)
        monkeypatch.setattr(cs, "next_backoff", fake_next_backoff)
        stop = threading.Event()

        def on_state(state: ChatState, detail: str = "") -> None:
            if state is ChatState.DISCONNECTED and len(attempts) >= 6:
                stop.set()

        reader = cs.TwitchChatReader(
            "chan", default_chat(), on_signal=lambda s: None, on_state=on_state
        )
        reader.run(stop)
        assert attempts == [1, 2, 3, 4, 5, 6]  # 1-based, no reset on pure failure

    def test_successful_session_resets_backoff_attempt(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A session that read traffic must reset the backoff so the NEXT drop
        # starts from 1 again — permanent flaps otherwise never recover fast.
        attempts: list[int] = []
        sessions: list[FakeWebSocket] = []
        good_frames = [frame(privmsg("v", "hi"))]

        class Flappy(FakeWSModule):
            state = {"fails": 2}

            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                if Flappy.state["fails"] > 0:
                    Flappy.state["fails"] -= 1
                    raise FakeSocketClosed("flap")
                ws = FakeWebSocket(good_frames)
                sessions.append(ws)
                return ws

        def fake_next_backoff(attempt: int, **_kw: Any) -> float:
            attempts.append(attempt)
            return 0.0

        monkeypatch.setattr(cs, "websocket", Flappy)
        monkeypatch.setattr(cs, "next_backoff", fake_next_backoff)
        stop = threading.Event()

        def on_state(state: ChatState, detail: str = "") -> None:
            if state is ChatState.STOPPED:
                return
            if len(sessions) >= 1 and "reconnecting" in (detail or "") and len(attempts) >= 3:
                stop.set()

        reader = cs.TwitchChatReader(
            "chan", default_chat(), on_signal=lambda s: None, on_state=on_state
        )
        reader.run(stop)
        assert attempts == [1, 2, 1]  # two failures, then a good session resets

    def test_honours_stop_event_quickly(self, no_real_websocket: None) -> None:
        stop = threading.Event()
        states: list[ChatState] = []
        reader = cs.TwitchChatReader(
            "chan",
            default_chat(),
            on_signal=lambda s: None,
            on_state=lambda s, d="": states.append(s),
            backoff_base=0.01,
            backoff_cap=0.01,
        )
        t = threading.Thread(target=reader.run, args=(stop,), daemon=True)
        t.start()
        time.sleep(0.05)  # mid-reconnect is where stop latency is worst
        stop.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert states[-1] is ChatState.STOPPED

    def test_ping_pong_and_join_handshake(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket(
            [
                frame(":tmi.twitch.tv 001 justinfan12345 :Welcome, GLHF!"),
                frame("PING :tmi.twitch.tv"),
                frame(f":{ANON_NICK}!{ANON_NICK}@{ANON_NICK}.tmi.twitch.tv JOIN #chan"),
            ]
        )
        connected = threading.Event()

        class OneSession(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                return ws

        monkeypatch.setattr(cs, "websocket", OneSession)

        def on_state(state: ChatState, detail: str = "") -> None:
            if state is ChatState.CONNECTED:
                connected.set()

        stop = threading.Event()
        reader = cs.TwitchChatReader(
            "chan", default_chat(), on_signal=lambda s: None, on_state=on_state
        )
        t = threading.Thread(target=reader.run, args=(stop,), daemon=True)
        t.start()
        assert connected.wait(timeout=2.0), "never saw the JOIN echo"
        stop.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
        sent = " ".join(ws.sent)
        assert f"NICK {ANON_NICK}" in sent
        assert "JOIN #chan" in sent
        assert "PONG :tmi.twitch.tv" in ws.sent
        assert ws.closed

    def test_messages_flow_to_on_signal(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A scripted burst session proves the full wiring:
        # frame -> line assembler -> parser -> scorer -> on_signal callback.
        ws = FakeWebSocket([frame(*[privmsg("viewer", "GO") for _ in range(20)])])
        got: list[Any] = []

        class OneSession(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                return ws

        monkeypatch.setattr(cs, "websocket", OneSession)

        def on_signal(sig: Any) -> None:
            got.append(sig)
            if sig.kind is SignalKind.CHAT_HYPE and "velocity" in sig.detail:
                stop.set()  # proven; the session dies with its frames anyway

        stop = threading.Event()
        stopped = threading.Event()

        def watchdog() -> None:
            # Safety: if wiring is broken, fail the test instead of hanging.
            if not stopped.wait(timeout=5.0):
                stop.set()

        reader = cs.TwitchChatReader(
            "chan", quiet_chat(), on_signal=on_signal, backoff_base=0.01, backoff_cap=0.01
        )
        threading.Thread(target=watchdog, daemon=True).start()
        reader.run(stop)  # the session dies with the frames; loop sees stop
        stopped.set()
        kinds = [(s.kind, s.detail) for s in got]
        assert any(s.kind is SignalKind.CHAT_HYPE for s in got), f"got {kinds}"

    def test_callback_exception_does_not_kill_reader(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Frame 1: JOIN echo (CONNECTED). Frame 2: 20-msg burst so the scorer
        # fires (quiet_chat's hypetrain threshold is 20 msgs/min) into a
        # callback that raises. Then the socket dies; surviving to the next
        # DISCONNECTED report proves the bug didn't kill the reader.
        ws = FakeWebSocket(
            [
                frame(f":{ANON_NICK}!{ANON_NICK}@{ANON_NICK}.tmi.twitch.tv JOIN #chan"),
                frame(*[privmsg("v", "GO") for _ in range(20)]),
            ]
        )
        calls: list[int] = []

        class OneSession(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                return ws

        monkeypatch.setattr(cs, "websocket", OneSession)

        def on_signal(sig: Any) -> None:
            calls.append(1)
            raise RuntimeError("manager callback bug")

        stop = threading.Event()

        def on_state(state: ChatState, detail: str = "") -> None:
            # CONNECTED comes first (JOIN echo), the burst next; the DISCONNECTED
            # state only proves survival if the crashing callback already ran.
            if state is ChatState.DISCONNECTED and calls:
                stop.set()

        reader = cs.TwitchChatReader(
            "chan",
            quiet_chat(),
            on_signal=on_signal,
            on_state=on_state,
            backoff_base=0.01,
            backoff_cap=0.01,
        )
        reader.run(stop)  # must not raise despite the callback bug
        assert calls, "the callback was never invoked"

    def test_reconnect_reports_detail_through_on_state(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Down(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                raise OSError("name resolution failed")

        monkeypatch.setattr(cs, "websocket", Down)
        details: list[str] = []
        stop = threading.Event()

        def on_state(state: ChatState, detail: str = "") -> None:
            if state is ChatState.DISCONNECTED:
                details.append(detail)
                stop.set()

        reader = cs.TwitchChatReader(
            "chan",
            default_chat(),
            on_signal=lambda s: None,
            on_state=on_state,
            backoff_base=0.01,
        )
        reader.run(stop)
        assert details and "name resolution failed" in details[0]
        assert "reconnecting" in details[0]


class TestRunChatReaderEntryPoint:
    def test_non_twitch_returns_false(self) -> None:
        assert (
            run_chat_reader(
                "https://youtube.com/watch?v=x",
                default_chat(),
                on_signal=lambda s: None,
                stop_event=threading.Event(),
            )
            is False
        )

    def test_disabled_returns_false(self) -> None:
        assert (
            run_chat_reader(
                "https://twitch.tv/chan",
                LiveChatSettings(enabled=False),
                on_signal=lambda s: None,
                stop_event=threading.Event(),
            )
            is False
        )

    def test_twitch_stops_cleanly(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([])
        stop = threading.Event()
        stop.set()  # exits before the first dial

        class OneSession(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                return ws

        monkeypatch.setattr(cs, "websocket", OneSession)
        assert (
            run_chat_reader(
                "https://twitch.tv/chan",
                default_chat(),
                on_signal=lambda s: None,
                stop_event=stop,
            )
            is True
        )


class TestConstantsContract:
    def test_backoff_cap_is_60(self) -> None:
        assert BACKOFF_CAP_S == 60.0

    def test_anon_nick_shape(self) -> None:
        assert ANON_NICK.startswith("justinfan")

    def test_rate_window_smaller_than_baseline(self) -> None:
        assert LiveChatSettings().baseline_window_s > RATE_WINDOW_S


class TestStopClosesTheSocket:
    """stop must not wait out a 15 s recv(): the socket is shut on stop."""

    def test_stop_interrupts_a_blocked_recv(
        self, no_real_websocket: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time

        class BlockingSocket(FakeWebSocket):
            def __init__(self) -> None:
                super().__init__()
                self.shut = threading.Event()

            def recv(self) -> str:
                # A quiet channel: recv blocks until the socket is shut down
                # (or a long "socket timeout" passes).
                if self.shut.wait(timeout=10.0):
                    raise FakeSocketClosed("socket shut down")
                raise FakeTimeout()

            def shutdown(self) -> None:
                self.shut.set()

        sockets: list[BlockingSocket] = []

        class Quiet(FakeWSModule):
            @staticmethod
            def create_connection(url: str, timeout: float) -> FakeWebSocket:
                sockets.append(BlockingSocket())
                return sockets[-1]

        monkeypatch.setattr(cs, "websocket", Quiet)
        reader = cs.TwitchChatReader("chan", default_chat(), on_signal=lambda s: None)
        stop = threading.Event()
        runner = threading.Thread(target=reader.run, args=(stop,), daemon=True)
        runner.start()
        deadline = _time.monotonic() + 5
        while not sockets and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert sockets, "the reader never connected"

        started = _time.monotonic()
        stop.set()
        runner.join(timeout=5)
        assert not runner.is_alive()
        assert _time.monotonic() - started < 2.0  # not the 10 s recv timeout
        assert sockets[0].shut.is_set()
