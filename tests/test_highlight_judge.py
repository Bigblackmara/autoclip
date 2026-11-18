"""Audience hints and score blend in detection, the judge pass, and the runner wiring.

The provider is a scripted fake: detection answers with fixed spans, the
judge with whatever the test scripts. No network, no real model.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pytest
from autoclip.config import Settings
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import highlights, signals
from autoclip.pipeline import runner as runner_module
from autoclip.pipeline.judge import ClipJudge
from autoclip.pipeline.runner import PipelineRunner
from autoclip.pipeline.signals import AudienceCurve
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import DetectionConfig, LLMProvider, ProviderStatus

#: Five non-overlapping 10-second candidates, best AI score first.
SPANS = [
    ("A", 0, 19, 90),
    ("B", 40, 59, 85),
    ("C", 80, 99, 80),
    ("D", 120, 139, 75),
    ("E", 160, 179, 70),
]


def make_transcript(count: int = 400) -> Transcript:
    """Half-second words; every tenth closes a sentence."""
    words = []
    for i in range(count):
        text = f"word{i}" + ("." if i % 10 == 9 else "")
        words.append(Word(text=text, start=i * 0.5, end=i * 0.5 + 0.4))
    return Transcript(words=words)


def detection_reply() -> str:
    return json.dumps(
        {
            "clips": [
                {
                    "start_word_index": first,
                    "end_word_index": last,
                    "title": title,
                    "hook": f"word{first}",
                    "score": score,
                    "reason": f"Candidate {title}.",
                }
                for title, first, last, score in SPANS
            ]
        }
    )


def verdicts(user: str, failing: set[str] = frozenset(), *, bare: bool = False) -> str:
    """Pass every candidate in the judge prompt except titles in ``failing``.

    The judge prompt carries no titles, so a candidate is recognised by its
    first word ("word40" is clip B).
    """
    by_first = {f"word{first}": title for title, first, _, _ in SPANS}
    out = []
    for key, opening in re.findall(r"### id: (\S+)\n.*?first 2 seconds: (\S+)", user, re.S):
        title = by_first.get(opening, "?")
        answer = "no" if title in failing else "yes"
        if bare:
            out.append({"id": key, "scroll_stop": answer, "context_free": "yes", "payoff": "yes"})
        else:
            out.append(
                {
                    "id": key,
                    "scroll_stop": {"answer": answer, "why": "opening"},
                    "context_free": {"answer": "yes", "why": ""},
                    "payoff": {"answer": "yes", "why": ""},
                }
            )
    return json.dumps({"verdicts": out})


class FakeProvider(LLMProvider):
    name = "fake"
    requires_key = False

    def __init__(self, judge=None) -> None:
        super().__init__("fake-model")
        self.judge = judge or (lambda user: verdicts(user))
        self.calls: list[tuple[str, str, str]] = []

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        self.calls.append((config.prompt_version, system, user))
        if config.prompt_version == "highlight_judge_v1":
            return self.judge(user)
        return detection_reply()

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(name=self.name, available=True)

    def prompts(self, version: str) -> list[tuple[str, str]]:
        return [(s, u) for v, s, u in self.calls if v == version]


CONFIG = {"min_duration_s": 5.0, "max_duration_s": 40.0, "max_clips": 3}


def run_detect(provider, *, judge=None, audience=None, config=None) -> list[Clip]:
    return asyncio.run(
        highlights.detect(
            make_transcript(),
            provider,
            config or DetectionConfig(**CONFIG),
            job_id="job",
            audience=audience,
            judge=judge,
        )
    )


def hot_curve(hot_start: int, hot_end: int, *, kind: str = "audience") -> AudienceCurve:
    heat = [
        (float(s), float(s + 2), 1.0 if hot_start <= s < hot_end else 0.1) for s in range(0, 200, 2)
    ]
    curve = signals.build_curve(200.0, heatmap=heat)
    curve.kind = kind
    return curve


# --------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------


class TestJudge:
    def test_without_a_judge_the_best_ai_scores_win(self) -> None:
        clips = run_detect(FakeProvider())

        assert [c.title for c in clips] == ["A", "B", "C"]
        assert [c.rank for c in clips] == [1, 2, 3]

    def test_a_failed_test_drops_the_clip_and_the_next_one_fills_in(self) -> None:
        provider = FakeProvider(judge=lambda user: verdicts(user, {"A", "C"}))
        judge = ClipJudge(provider._complete, max_candidates=10)

        clips = run_detect(provider, judge=judge)

        assert [c.title for c in clips] == ["B", "D", "E"]
        assert [c.rank for c in clips] == [1, 2, 3]
        outcomes = {e.title: e.outcome for e in judge.entries}
        assert outcomes["A"] == "rejected" and outcomes["B"] == "passed"
        rejected = next(e for e in judge.entries if e.title == "A")
        assert rejected.failed == ["weak hook in the first 2 s"]

    def test_the_judge_sees_the_opening_and_the_full_clip(self) -> None:
        provider = FakeProvider()
        run_detect(provider, judge=ClipJudge(provider._complete))

        (system, user), *_ = provider.prompts("highlight_judge_v1")
        assert "scroll_stop" in system
        assert "first 2 seconds: word0 word1 word2 word3" in user
        assert "FULL CLIP TRANSCRIPT: word0" in user

    def test_bare_yes_no_answers_are_accepted(self) -> None:
        provider = FakeProvider(judge=lambda user: verdicts(user, {"A"}, bare=True))
        clips = run_detect(provider, judge=ClipJudge(provider._complete))

        assert "A" not in [c.title for c in clips]

    def test_a_provider_error_keeps_every_candidate(self) -> None:
        def broken(user: str) -> str:
            raise RuntimeError("rate limited")

        provider = FakeProvider(judge=broken)
        judge = ClipJudge(provider._complete)

        clips = run_detect(provider, judge=judge)

        assert [c.title for c in clips] == ["A", "B", "C"]
        assert judge.errors and "rate limited" in judge.errors[0]
        assert all(e.note == "judge failed" for e in judge.entries)

    def test_malformed_json_twice_keeps_every_candidate(self) -> None:
        provider = FakeProvider(judge=lambda user: "I think they are all great!")
        judge = ClipJudge(provider._complete)

        clips = run_detect(provider, judge=judge)

        assert [c.title for c in clips] == ["A", "B", "C"]
        assert len(provider.prompts("highlight_judge_v1")) == 2  # one repair retry
        assert judge.errors

    def test_a_repaired_reply_is_used(self) -> None:
        replies = iter(["not json"])

        def flaky(user: str) -> str:
            return next(replies, None) or verdicts(user, {"A"})

        provider = FakeProvider(judge=flaky)
        clips = run_detect(provider, judge=ClipJudge(provider._complete))

        assert [c.title for c in clips] == ["B", "C", "D"]

    def test_a_missing_verdict_keeps_the_candidate(self) -> None:
        provider = FakeProvider(judge=lambda user: json.dumps({"verdicts": []}))
        judge = ClipJudge(provider._complete)

        clips = run_detect(provider, judge=judge)

        assert [c.title for c in clips] == ["A", "B", "C"]
        assert {e.note for e in judge.entries} == {"no verdict"}

    def test_min_keep_rescues_the_best_reject(self) -> None:
        provider = FakeProvider(judge=lambda user: verdicts(user, {"A", "B", "C", "D", "E"}))
        judge = ClipJudge(provider._complete, min_keep=1)

        clips = run_detect(provider, judge=judge)

        assert [c.title for c in clips] == ["A"]
        assert "[judge: kept to reach the minimum" in clips[0].reason
        assert [e.outcome for e in judge.entries if e.title == "A"] == ["rescued"]

    def test_min_keep_zero_can_drop_everything(self) -> None:
        provider = FakeProvider(judge=lambda user: verdicts(user, {"A", "B", "C", "D", "E"}))

        assert run_detect(provider, judge=ClipJudge(provider._complete, min_keep=0)) == []

    def test_only_the_pool_is_judged(self) -> None:
        provider = FakeProvider(judge=lambda user: verdicts(user, {"A"}))
        judge = ClipJudge(provider._complete, max_candidates=2, batch_size=1)

        clips = run_detect(provider, judge=judge)

        assert [c.title for c in clips] == ["B"]
        assert len(provider.prompts("highlight_judge_v1")) == 2
        assert sum(e.outcome == "unjudged" for e in judge.entries) == 3

    def test_report_is_json_serialisable(self) -> None:
        provider = FakeProvider(judge=lambda user: verdicts(user, {"A"}))
        judge = ClipJudge(provider._complete)
        run_detect(provider, judge=judge)

        report = json.loads(json.dumps(judge.report()))
        assert report["prompt"] == "highlight_judge_v1"
        assert len(report["entries"]) == 5


# --------------------------------------------------------------------------
# Audience hints and the score blend
# --------------------------------------------------------------------------


class TestAudienceInDetection:
    def test_hotspots_are_passed_to_the_model_as_hints(self) -> None:
        provider = FakeProvider()
        audience = highlights.AudienceOptions(curve=hot_curve(80, 90))

        run_detect(
            provider,
            audience=audience,
            config=DetectionConfig(**CONFIG, prompt_version="highlight_v2"),
        )

        (system, user), *_ = provider.prompts("highlight_v2")
        assert "## Audience signals" in system
        assert "Audience signals for this section" in user
        assert "1:20-" in user  # the hotspot's clock time
        assert "most replayed" in user
        # The hint cites word indices the model can copy (80 s = word 160).
        assert re.search(r"around words \[1[5-6]\d\]-\[1\d\d\]", user)

    def test_no_curve_data_means_no_hints(self) -> None:
        provider = FakeProvider()
        empty = highlights.AudienceOptions(curve=signals.build_curve(200.0))

        run_detect(provider, audience=empty)

        (_, user), *_ = provider.prompts("highlight_v1")
        assert "Audience signals" not in user

    def test_hints_can_be_switched_off(self) -> None:
        provider = FakeProvider()
        audience = highlights.AudienceOptions(curve=hot_curve(80, 90), hints_per_window=0)

        run_detect(provider, audience=audience)

        (_, user), *_ = provider.prompts("highlight_v1")
        assert "Audience signals" not in user

    def test_audience_data_reranks_clips(self) -> None:
        """E is the weakest AI pick but the most re-watched moment (80-90 s)."""
        audience = highlights.AudienceOptions(curve=hot_curve(80, 90), audience_weight=0.4)

        clips = run_detect(FakeProvider(), audience=audience)

        assert clips[0].title == "E"
        assert "[score" in clips[0].reason and "AI 70 x0.60" in clips[0].reason
        assert "heatmap" in clips[0].reason

    def test_energy_alone_only_nudges(self) -> None:
        audience = highlights.AudienceOptions(
            curve=hot_curve(80, 90, kind="energy"), audience_weight=0.4, energy_weight=0.12
        )

        clips = run_detect(FakeProvider(), audience=audience)

        # The hot stretch only nudges: the weakest AI pick stays out.
        assert "E" not in [c.title for c in clips]
        assert all("x0.12" in c.reason for c in clips)

    def test_blend_formula(self) -> None:
        curve = AudienceCurve(curve=np.linspace(0, 1, 100, dtype=np.float32), kind="audience")
        clip = Clip(id="c", job_id="j", start_s=90, end_s=100, score=50, reason="r")

        highlights.blend_score(clip, highlights.AudienceOptions(curve=curve, audience_weight=0.4))

        audience_value = curve.range_score(90, 100)
        assert clip.score == round(0.6 * 50 + 0.4 * audience_value)
        assert clip.reason.startswith("r [score")


# --------------------------------------------------------------------------
# Runner wiring
# --------------------------------------------------------------------------


@pytest.fixture
def youtube_source(initialised_db, tmp_path: Path) -> Source:
    media = tmp_path / "media"
    media.mkdir()
    info = {
        "duration": 200,
        "heatmap": [
            {"start_time": s, "end_time": s + 2, "value": 1.0 if 80 <= s < 90 else 0.1}
            for s in range(0, 200, 2)
        ],
        "chapters": [{"start_time": 60, "end_time": 200, "title": "The reveal"}],
    }
    (media / signals.INFO_FILE).write_text(json.dumps(info), "utf-8")
    return store.create_source(
        Source(
            id=new_id(),
            type="youtube",
            path=str(media / "source.mp4"),
            url="https://youtu.be/abc",
            duration_s=200.0,
        )
    )


def _run_highlights(source: Source, settings: Settings, provider, monkeypatch) -> tuple:
    monkeypatch.setattr(runner_module, "build_provider", lambda *a, **k: provider)
    job = store.create_job(Job(id=new_id(), source_id=source.id))
    runner = PipelineRunner(job, source, settings=settings)
    clips = asyncio.run(runner._stage_highlights(make_transcript(), [], None))
    return runner, clips


def _settings() -> Settings:
    settings = Settings()
    settings.clips.min_duration_s = 5.0
    settings.clips.max_duration_s = 40.0
    settings.clips.max_clips = 3
    return settings


class TestRunnerWiring:
    def test_defaults_use_the_curve_the_v2_prompt_and_the_judge(
        self, youtube_source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The runner's comment fetcher goes through signals.fetch_comments.
        fetched: list[str] = []

        def fake_comments(url, **kwargs):
            fetched.append(url)
            return []

        monkeypatch.setattr(signals, "fetch_comments", fake_comments)
        provider = FakeProvider(judge=lambda user: verdicts(user, {"B"}))

        runner, clips = _run_highlights(youtube_source, _settings(), provider, monkeypatch)

        assert fetched == ["https://youtu.be/abc"]
        assert provider.prompts("highlight_v2"), "audience data switches to the v2 prompt"
        assert "The reveal" in provider.prompts("highlight_v2")[0][1]
        assert provider.prompts("highlight_judge_v1")
        assert "B" not in [c.title for c in clips]
        assert clips[0].title == "E"  # the most replayed stretch
        assert runner.workspace.audience.exists()
        assert json.loads(runner.workspace.judge_report.read_text("utf-8"))["entries"]
        assert [c.title for c in store.list_clips(runner.job.id)] == [c.title for c in clips]

    def test_everything_can_be_switched_off(
        self, youtube_source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings()
        settings.detection.audience_signals = False
        settings.detection.judge = False
        provider = FakeProvider()

        runner, clips = _run_highlights(youtube_source, settings, provider, monkeypatch)

        assert [c.title for c in clips] == ["A", "B", "C"]
        assert not provider.prompts("highlight_v2")
        assert not provider.prompts("highlight_judge_v1")
        assert not runner.workspace.audience.exists()

    def test_a_broken_signal_stage_never_fails_the_job(
        self, youtube_source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(**kwargs):
            raise RuntimeError("numpy fell over")

        monkeypatch.setattr(signals, "build_for_source", explode)
        settings = _settings()
        settings.detection.judge = False

        _, clips = _run_highlights(youtube_source, settings, FakeProvider(), monkeypatch)

        assert [c.title for c in clips] == ["A", "B", "C"]

    def test_a_cached_curve_is_reused(
        self, youtube_source: Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _settings()
        settings.detection.judge = False
        settings.detection.fetch_comments = False
        runner, _ = _run_highlights(youtube_source, settings, FakeProvider(), monkeypatch)
        cached = AudienceCurve.load(runner.workspace.audience)

        monkeypatch.setattr(
            signals, "build_for_source", lambda **k: (_ for _ in ()).throw(AssertionError)
        )
        again = runner._load_or_build_audience(None)

        assert again is not None and again.kind == cached.kind == "audience"
