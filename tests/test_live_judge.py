"""Live judge gate — prompt rendering, repair loop, clamping.

The judge is the only place Live Watch spends real money, so its contract is
tested the way the provider layer's is: scripted fakes, malformed JSON shapes
models actually emit, and the assertion that a bad window never produces a
fake good verdict. No test touches a real provider or the network.
"""

from __future__ import annotations

import pytest
from autoclip.live.judge import (
    PROMPT_VERSION,
    LiveJudgement,
    LiveJudgeOut,
    judge_window,
    render_judge_prompt,
)
from autoclip.live.signals import DetectorSignal, SignalKind
from autoclip.pipeline.transcript import Word
from autoclip.providers import DetectionConfig, ProviderError, ProviderStatus
from autoclip.providers.base import LLMProvider

VALID = (
    '{"score":81,"title":"Rage quit ends the run","hook":"you have got to be kidding me",'
    '"reason":"Real payoff seconds in.","start_word_offset":3,"end_word_offset":17}'
)


class ScriptedJudgeProvider(LLMProvider):
    """A provider that returns queued responses, for testing the judge loop."""

    name = "scripted"
    requires_key = False

    def __init__(self, responses: list[str]) -> None:
        super().__init__("test-model")
        self.responses = list(responses)
        self.calls: list[tuple[str, str, DetectionConfig]] = []

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.calls.append((system, user, config))
        return self.responses.pop(0) if self.responses else "{}"

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)


def make_words(count: int, *, speaker: str | None = None) -> list:
    """A window of synthetic monotonically-timed words with sentence shape."""
    return [
        Word(
            text="Okay." if i % 7 == 6 else f"word{i}",
            start=float(i),
            end=float(i) + 0.5,
            speaker=speaker,
        )
        for i in range(count)
    ]


WORDS = make_words(20)


class TestPromptRendering:
    def test_words_are_indexed_from_zero(self) -> None:
        prompt = render_judge_prompt("shroud", WORDS[:3])

        assert "[0]word0" in prompt
        assert "[1]word1" in prompt
        assert "[2]word2" in prompt

    def test_sentence_shaped_words_render_verbatim(self) -> None:
        words = make_words(7)

        prompt = render_judge_prompt("shroud", words)

        assert "[6]Okay." in prompt

    def test_prompt_carries_the_stream_label_and_signal_block(self) -> None:
        prompt = render_judge_prompt(
            "  loso  raids  ",
            make_words(2),
            {"audio_energy": 1.4, "chat_hype": 2.0},
        )

        assert 'live stream "loso raids"' in prompt
        assert "audio_energy: 1.40" in prompt
        assert "chat_hype: 2.00" in prompt

    def test_signals_are_sorted_strongest_first(self) -> None:
        prompt = render_judge_prompt(
            "s",
            make_words(2),
            {"audio_energy": 1.4, "chat_hype": 2.0},
        )

        assert prompt.index("chat_hype") < prompt.index("audio_energy")

    def test_signal_values_may_be_detector_signal_objects(self) -> None:
        prompt = render_judge_prompt(
            "s",
            make_words(2),
            {
                SignalKind.AUDIO_ENERGY: DetectorSignal(
                    kind=SignalKind.AUDIO_ENERGY,
                    t_s=12.0,
                    strength=1.3,
                    detail="RMS spike while laughing",
                )
            },
        )

        assert "audio_energy: 1.30" in prompt
        assert "RMS spike while laughing" in prompt

    def test_known_signal_kinds_get_a_detail_explanation(self) -> None:
        prompt = render_judge_prompt("s", make_words(2), {"chat_rate": 1.1})

        assert "message velocity" in prompt

    def test_no_signals_block_is_still_intelligible(self) -> None:
        prompt = render_judge_prompt("s", make_words(2))

        assert "SIGNALS" in prompt
        assert "without cheap-signal evidence" in prompt

    def test_strength_scale_note_is_present(self) -> None:
        prompt = render_judge_prompt("s", make_words(2), {"chat_rate": 1.1})

        assert "0.0-2.0 scale" in prompt
        assert "below 0.4 is treated as noise" in prompt

    def test_speaker_turns_render_with_speaker_tags(self) -> None:
        words = [
            Word(text="hi", start=0.0, end=0.4, speaker="SPEAKER_00"),
            Word(text="yo", start=0.5, end=0.9, speaker="SPEAKER_01"),
        ]

        prompt = render_judge_prompt("s", words)

        assert "\n<SPEAKER_00> [0]hi" in prompt
        assert "\n<SPEAKER_01> [1]yo" in prompt


class TestJudgementLoop:
    async def test_valid_response_maps_to_judgement(self) -> None:
        provider = ScriptedJudgeProvider([VALID])

        judgement = await judge_window("s", WORDS, {"audio_energy": 1.2}, provider=provider)

        assert judgement.score == 81
        assert judgement.title == "Rage quit ends the run"
        assert judgement.hook == "you have got to be kidding me"
        assert judgement.reason == "Real payoff seconds in."
        assert judgement.start_word_offset == 3
        assert judgement.end_word_offset == 17
        assert judgement.signals == {"audio_energy": 1.2}

    async def test_one_call_with_valid_json(self) -> None:
        provider = ScriptedJudgeProvider([VALID])

        await judge_window("s", WORDS, {}, provider=provider)

        assert len(provider.calls) == 1

    async def test_malformed_json_triggers_one_repair_retry(self) -> None:
        provider = ScriptedJudgeProvider(["this is not json at all", VALID])

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert judgement.score == 81
        assert len(provider.calls) == 2

    async def test_repair_prompt_carries_the_validation_error(self) -> None:
        provider = ScriptedJudgeProvider(["not json even a little", VALID])

        await judge_window("s", WORDS, {}, provider=provider)

        first_user = provider.calls[0][1]
        second_user = provider.calls[1][1]
        assert second_user.startswith(first_user)
        assert "did not match the required schema" in second_user

    async def test_missing_score_is_a_schema_error_not_a_default(self) -> None:
        no_score = (
            '{"title":"x","hook":"y","reason":"z","start_word_offset":null,"end_word_offset":null}'
        )
        provider = ScriptedJudgeProvider([no_score, VALID])

        result = await judge_window("s", WORDS, {}, provider=provider)

        assert result.score == 81  # repaired on the retry
        assert len(provider.calls) == 2

    async def test_two_failures_raise_provider_error(self) -> None:
        provider = ScriptedJudgeProvider(["nope", "still nope"])

        with pytest.raises(ProviderError, match="malformed judgement data twice"):
            await judge_window("s", WORDS, {}, provider=provider)

    async def test_provider_error_names_the_provider(self) -> None:
        provider = ScriptedJudgeProvider(["nope", "still nope"])
        provider.name = "ollama"

        with pytest.raises(ProviderError) as excinfo:
            await judge_window("s", WORDS, {}, provider=provider)

        assert excinfo.value.provider == "ollama"

    async def test_empty_words_window_is_rejected_locally(self) -> None:
        provider = ScriptedJudgeProvider([VALID])

        judgement = await judge_window("s", [], {"audio_energy": 2.0}, provider=provider)

        assert judgement.score == 0
        assert judgement.start_word_offset is None
        assert judgement.end_word_offset is None
        assert judgement.signals == {"audio_energy": 2.0}
        assert provider.calls == []  # no token spent

    async def test_fenced_json_is_accepted(self) -> None:
        provider = ScriptedJudgeProvider([f"```json\n{VALID}\n```"])

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert judgement.score == 81
        assert len(provider.calls) == 1

    async def test_temperature_and_prompt_version_are_sent(self) -> None:
        provider = ScriptedJudgeProvider([VALID])

        await judge_window("s", WORDS, {}, provider=provider, temperature=0.2)

        system, _, config = provider.calls[0]
        assert config.temperature == 0.2
        assert config.prompt_version == PROMPT_VERSION
        assert "quality gate" in system

    async def test_default_specializations_of_judgement(self) -> None:
        provider = ScriptedJudgeProvider(['{"score":42,"title":null,"hook":null,"reason":null}'])

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert judgement.title == ""
        assert judgement.hook == ""
        assert judgement.reason == ""
        assert judgement.start_word_offset is None
        assert judgement.end_word_offset is None


class TestOffsetClamping:
    async def test_offsets_beyond_the_window_are_clamped(self) -> None:
        provider = ScriptedJudgeProvider(
            ['{"score":70,"start_word_offset":99,"end_word_offset":150}']
        )

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert judgement.start_word_offset == 19
        assert judgement.end_word_offset == 19

    async def test_offsets_clamp_to_zero_for_a_single_word_window(self) -> None:
        provider = ScriptedJudgeProvider(["""{"score":70}"""])

        judgement = await judge_window("s", make_words(1), {}, provider=provider)

        assert judgement.start_word_offset is None
        assert judgement.end_word_offset is None

    async def test_negative_offsets_are_clamped_to_zero(self) -> None:
        provider = ScriptedJudgeProvider(
            ['{"score":70,"start_word_offset":-5,"end_word_offset":-1}']
        )

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert judgement.start_word_offset == 0
        assert judgement.end_word_offset == 0

    async def test_float_and_string_offsets_are_coerced(self) -> None:
        provider = ScriptedJudgeProvider(
            ['{"score":70,"start_word_offset":"4","end_word_offset":16.7}']
        )

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert judgement.start_word_offset == 4
        assert judgement.end_word_offset == 17

    async def test_reversed_offsets_are_dropped_rather_than_swapped(self) -> None:
        provider = ScriptedJudgeProvider(
            ['{"score":70,"start_word_offset":10,"end_word_offset":2}']
        )

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert judgement.start_word_offset is None
        assert judgement.end_word_offset is None


class TestOutputCoercion:
    def test_fractional_score_is_scaled_to_percent(self) -> None:
        assert LiveJudgeOut.model_validate({"score": 0.87}).score == 87

    def test_float_score_is_rounded(self) -> None:
        assert LiveJudgeOut.model_validate({"score": 82.6}).score == 83

    def test_out_of_range_score_is_clamped(self) -> None:
        assert LiveJudgeOut.model_validate({"score": 250}).score == 100
        assert LiveJudgeOut.model_validate({"score": -3}).score == 0

    def test_missing_score_defaults_to_the_middle(self) -> None:
        # The model must SAY something; a silent response is repaired, not defaulted.
        assert LiveJudgeOut.model_validate({}).score == 50

    def test_null_fields_become_empty_strings(self) -> None:
        out = LiveJudgeOut.model_validate(
            {"score": 80, "title": None, "hook": None, "reason": None}
        )

        assert out.title == ""
        assert out.hook == ""
        assert out.reason == ""

    def test_none_offsets_survive_validation(self) -> None:
        out = LiveJudgeOut.model_validate(
            {"score": 80, "start_word_offset": None, "end_word_offset": None}
        )

        assert out.start_word_offset is None
        assert out.end_word_offset is None

    def test_junk_offsets_become_none(self) -> None:
        out = LiveJudgeOut.model_validate(
            {"score": 80, "start_word_offset": "big", "end_word_offset": "mid"}
        )

        assert out.start_word_offset is None
        assert out.end_word_offset is None

    def test_a_score_that_cannot_be_read_falls_to_the_middle(self) -> None:
        # Same policy as ClipCandidate: an unreadable score is a model slip,
        # tolerated rather than burning a repair retry on it.
        assert LiveJudgeOut.model_validate({"score": "x"}).score == 50
        assert LiveJudgeOut.model_validate({"score": [1, 2, 3]}).score == 50


class TestSignalCollection:
    async def test_signals_are_echoed_as_plain_string_keys(self) -> None:
        provider = ScriptedJudgeProvider([VALID])

        judgement = await judge_window(
            "s",
            WORDS,
            {SignalKind.AUDIO_ENERGY: 1.5, SignalKind.CHAT_HYPE: 2.0},
            provider=provider,
        )

        assert judgement.signals == {"audio_energy": 1.5, "chat_hype": 2.0}

    def test_unreadable_signal_strength_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid strength"):
            render_judge_prompt("s", make_words(2), {"audio_energy": "very loud"})

    def test_non_finite_signal_strength_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid strength"):
            render_judge_prompt("s", make_words(2), {"chat_rate": float("nan")})

    async def test_prompt_is_built_once_and_only_once(self) -> None:
        provider = ScriptedJudgeProvider([VALID])

        await judge_window("s", WORDS, {"chat_hype": 2.0, "audio_energy": 1.8}, provider=provider)

        user = provider.calls[0][1]
        assert user.count("live stream") == 1
        assert user.count("SIGNALS") == 1


class TestLiveJudgementDataclass:
    def test_signals_default_factory_gives_independent_dicts(self) -> None:
        # Pylint-style guard: two judgements must never share one signals dict.
        a = LiveJudgement(score=1, signals={"audio_energy": 1.0})
        b = LiveJudgement(score=2)

        assert a.signals == {"audio_energy": 1.0}
        assert b.signals == {}


class TestNoSecretsInLogs:
    async def test_neither_prompt_nor_judgement_contains_an_api_key(self) -> None:
        provider = ScriptedJudgeProvider([VALID])
        provider.api_key = "sk-super-secret-value"

        judgement = await judge_window("s", WORDS, {}, provider=provider)

        assert "sk-super-secret-value" not in provider.calls[0][1]
        assert "sk-super-secret-value" not in provider.calls[0][0]
        assert "sk-super-secret-value" not in repr(judgement)


class TestScoreIsNotInflated:
    """An integer 1 is a flat rejection, never a 100-point pass."""

    @pytest.mark.parametrize("raw", [1, "1", 1.0])
    def test_integer_one_stays_one(self, raw) -> None:
        assert LiveJudgeOut.model_validate({"score": raw}).score == 1

    @pytest.mark.parametrize("raw,expected", [(0.87, 87), ("0.5", 50), (0.05, 5)])
    def test_non_integer_fractions_still_scale(self, raw, expected) -> None:
        assert LiveJudgeOut.model_validate({"score": raw}).score == expected

    def test_booleans_and_non_finite_values_are_unreadable(self) -> None:
        assert LiveJudgeOut.model_validate({"score": True}).score == 50
        assert LiveJudgeOut.model_validate({"score": float("nan")}).score == 50


class TestBudgetedCalls:
    """Every paid request — the repair retry included — asks the budget first."""

    async def test_each_provider_call_is_announced(self) -> None:
        provider = ScriptedJudgeProvider(["not json", VALID])
        announced: list[int] = []

        def before_call() -> bool:
            announced.append(len(provider.calls))
            return True

        judgement = await judge_window("s", WORDS, {}, provider=provider, before_call=before_call)

        assert judgement.score == 81
        assert announced == [0, 1]  # before the first call, and before the repair

    async def test_refused_repair_is_never_sent(self) -> None:
        from autoclip.live.judge import JudgeBudgetExhausted

        provider = ScriptedJudgeProvider(["not json", VALID])
        answers = iter([True, False])

        with pytest.raises(JudgeBudgetExhausted, match="repair"):
            await judge_window("s", WORDS, {}, provider=provider, before_call=lambda: next(answers))
        assert len(provider.calls) == 1

    async def test_refused_first_call_sends_nothing(self) -> None:
        from autoclip.live.judge import JudgeBudgetExhausted

        provider = ScriptedJudgeProvider([VALID])
        with pytest.raises(JudgeBudgetExhausted):
            await judge_window("s", WORDS, {}, provider=provider, before_call=lambda: False)
        assert provider.calls == []

    async def test_empty_window_spends_nothing(self) -> None:
        provider = ScriptedJudgeProvider([VALID])
        asked: list[bool] = []
        judgement = await judge_window(
            "s", [], {}, provider=provider, before_call=lambda: asked.append(True) or True
        )
        assert judgement.score == 0
        assert asked == [] and provider.calls == []
