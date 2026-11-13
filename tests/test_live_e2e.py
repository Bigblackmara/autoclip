"""Live Watch REAL end-to-end: fake live source → capture → detect → judge → produce.

Marked ``live`` (excluded from the default gate) because it runs real ffmpeg
for tens of seconds of wall time. This is the Wave 3 proof the architecture
doc calls for — asserted on disk artifacts + DB rows, not on mocks.

The chain exercised, with every component's real/stubbed status spelled out
(helpers live in ``tests/live_e2e_helpers.py``):

===========================  ==============================================
component                   status
===========================  ==============================================
fake broadcast encode       REAL ffmpeg (sine + volume ramp, testsrc2)
"CDN"                       real local HTTP server, paced slower than ingest
URL resolution              STUB (resolve_stream_url seam → local URL)
capture ffmpeg copy         REAL StreamCapture argv, no ``-re``
ring buffer + tailer        REAL RingBuffer commits, pruning, pins
audio analysis              REAL audio_signals (z-scored RMS bursts)
trigger + aggregator        INJECTED faithful bodies (repo A4 skeleton)
circuit (cooldown/buckets)  INJECTED faithful bodies (repo A4 skeleton)
judge (LLM)                 STUB (score 93, above the gate — offline)
transcribe (Whisper)         STUB (real Word objects; both call sites)
manager/queue/store         REAL LiveManager, real SQLite store
clip_factory produce        REAL: extract → remux → audio → crop → burn,
                             export, .srt sidecar
===========================  ==============================================

The scenario: ~40 quiet seconds lull the baseline, an 8-second loud burst
fires the real detector around stream-second 42, the (stubbed-verdict) judge
accepts, and the manager's queue renders the 20s pre-roll + 10s post-roll
window into a caption-burned 9:16 H.264 file.

Two product behaviors this test pins that unit tests cannot:

* the served source ends slower than ffmpeg ingests it, so the watcher sees
  a live capture for the whole render — and is back to plain ``watching``
  after the candidate finishes, exactly like a broadcaster that keeps
  talking through a clip;
* the render-phase weighting end-to-end: the same window must survive
  extraction (pinned against pruning), a container remux, an audio pull,
  crop-path construction, and a caption burn with an .srt sidecar.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from autoclip.live import capture as live_capture
from autoclip.live import clip_factory, store
from autoclip.live.manager import LiveManager
from autoclip.live.settings import LiveSettings
from autoclip.live.states import StreamState
from live_e2e_helpers import (
    FakeBroadcast,
    PerStreamCooldown,
    RecordingAggregator,
    RecordingTrigger,
    TokenBucket,
    build_fake_broadcast,
    judge_accept_everything,
    make_clip_words,
    probe_streams,
    serve_throttled,
    stub_transcribe_factory,
)

pytestmark = [pytest.mark.live]

log = logging.getLogger(__name__)

#: Length of the fake broadcast, seconds of stream time.
BROADCAST_S = 78.0

#: Serve rate for the local "CDN", bytes per second — DELIBERATELY SLOWER
#: than capture's ingest so the source behaves like a live stream: ffmpeg's
#: copy capture reads whatever is available, and a wait for more bytes is
#: exactly what a live watch does. At ~105 KB/s the burst (at stream-second
#: 40) reaches the buffer ≈40 s into the test, with 38 s of stream still
#: unserved behind it — plenty of headroom for the render to finish while
#: the watcher still sees a live capture.
SERVE_RATE_BPS = 105_000

#: Wall-clock ceilings. Generorous on purpose — the observed run is faster;
#: these exist so a broken chain fails instead of hanging the suite.
WATCHING_TIMEOUT_S = 60.0
READY_TIMEOUT_S = 90.0


async def wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.1) -> None:
    """Await ``predicate()`` truthy or fail; polls on the running loop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    pytest.fail(f"condition not met within {timeout}s")


@pytest.fixture
def broadcast(tmp_path: Path) -> FakeBroadcast:
    """The encoded fake stream — one real ffmpeg encode, a few seconds."""
    return build_fake_broadcast(tmp_path / "broadcast.mp4", duration_s=BROADCAST_S)


@pytest.fixture
def served(broadcast: FakeBroadcast):
    """The paced local "CDN"; yields (server, url), tears down after."""
    server, url = serve_throttled(broadcast.path, SERVE_RATE_BPS)
    yield server, url
    server.shutdown()


# -- clip-status capture -------------------------------------------------------

#: Manager clip events → the live_clips row status each one accompanies,
#: 1:1 by construction: the watcher and the manager publish exactly these
#: events at the same instant they write the matching status to the row.
_EVENT_TO_STATUS = {
    "clip_detecting": "detecting",
    "clip_queued": "queued",
    "clip_producing": "rendering",
    "clip_ready": "ready",
}


def status_event_recorder() -> tuple[Callable, list[str]]:
    """Capture every clip-status change via the manager's publish choke-point.

    A poller can legitimately miss a row status that lives for one loop
    slice (``queued`` → ``rendering`` can be sub-100 ms), but the manager
    publishes an event for every status write it performs — hooking the
    publish seam records the true traversal losslessly, so the assertion
    sees the contractual sequence instead of racing the worker's schedule.
    """
    observed: list[str] = []

    def record(channel: str, event_type: str, data: dict | None = None) -> None:
        status = _EVENT_TO_STATUS.get(event_type)
        if status is not None:
            observed.append(status)

    return record, observed


# -- the test -------------------------------------------------------------------


async def test_full_chain_produces_a_ready_vertical_clip(
    initialised_db: int,
    monkeypatch: pytest.MonkeyPatch,
    broadcast: FakeBroadcast,
    served,
) -> None:
    """detect → judge → enqueue → render: the one real end-to-end pass."""
    _server, url = served

    # 1) Resolution seam: the watched URL resolves to the local "CDN". The
    #    label deliberately avoids twitch.tv — a twitch URL would start the
    #    real IRC chat worker (network) which this offline suite forbids.
    monkeypatch.setattr(live_capture, "resolve_stream_url", lambda _watch_url: url)

    # 2) Transcribe stub at BOTH call sites: the watcher's judge-phase call
    #    (injected via the manager) and clip_factory's render-phase call
    #    (patched at the module it imports from). Real Whisper needs CUDA
    #    libraries this machine lacks — the stage is out of the E2E's scope.
    stub_transcribe = stub_transcribe_factory(make_clip_words())
    monkeypatch.setattr(clip_factory.transcribe, "transcribe", stub_transcribe)

    trigger = RecordingTrigger()
    cooldown = PerStreamCooldown(cooldown_s=90.0)
    rate = LiveSettings().rate_limit
    publisher, statuses = status_event_recorder()

    manager = LiveManager(
        settings=LiveSettings(),
        transcribe=stub_transcribe,
        judge=judge_accept_everything,
        aggregator_factory=RecordingAggregator,
        trigger=trigger,
        cooldown=cooldown,
        judge_minute=TokenBucket(rate.judge_per_minute / 60.0, rate.judge_per_minute),
        judge_hour=TokenBucket(rate.judge_per_hour / 3600.0, rate.judge_per_hour),
        judge_day=TokenBucket(rate.judge_per_day / 86400.0, rate.judge_per_day),
        publish=publisher,
    )

    log.info(
        "E2E: broadcast %.0fs (%.1f MB), burst %.0f-%.0fs, served at %d B/s",
        BROADCAST_S,
        broadcast.path.stat().st_size / 1e6,
        broadcast.burst_start_s,
        broadcast.burst_end_s,
        SERVE_RATE_BPS,
    )

    await manager.start()
    stream_id = await manager.add_stream("https://e2e.invalid/fake", "fake-e2e")
    try:
        # 3) The watcher connects and starts capturing the paced source.
        await wait_until(
            lambda: (
                manager._watchers.get(stream_id) is not None
                and manager._watchers[stream_id].state is StreamState.WATCHING
            ),
            timeout=WATCHING_TIMEOUT_S,
        )
        watcher = manager._watchers[stream_id]

        # 4) The REAL audio worker flags the burst; the injected trigger
        #    consumes real signals and fires on them.
        await wait_until(
            lambda: trigger.calls and trigger.calls[-1].signals,
            timeout=WATCHING_TIMEOUT_S,
        )
        signals_seen = list(trigger.calls[-1].signals)
        audio = [s for s in signals_seen if s.kind.value == "audio_energy"]
        assert audio, f"no audio_energy signal reached the trigger: {signals_seen!r}"
        assert max(s.strength for s in audio) >= 0.5, signals_seen

        # 5) A clip row exists for the burst, with pre-roll behind it.
        await wait_until(lambda: "detecting" in statuses, timeout=WATCHING_TIMEOUT_S)
        clips = store.list_clips(stream_id=stream_id)
        assert clips, "no live_clips row was created for the burst"
        clip_id = clips[0].id
        row = store.get_clip(clip_id)
        assert row is not None
        assert row.trigger_end_s >= broadcast.burst_start_s - 2.0, row.trigger_end_s
        assert row.clip_start_s < broadcast.burst_start_s, "no pre-roll before burst"

        # 6) The render completes: ready event, row, and a real artifact.
        await wait_until(lambda: "ready" in statuses, timeout=READY_TIMEOUT_S)
        for required in ("detecting", "queued", "rendering", "ready"):
            assert required in statuses, f"{required!r} never observed: {statuses}"
        order = [statuses.index(state) for state in ("detecting", "queued", "rendering", "ready")]
        assert order == sorted(order), f"status traversal out of order: {statuses}"

        row = store.get_clip(clip_id)
        assert row is not None and row.status == "ready"
        assert row.rendered_path, "ready row carries no rendered_path"
        rendered = Path(row.rendered_path)
        assert rendered.is_file(), f"{rendered} missing on disk"
        assert rendered.stat().st_size > 0

        # 7) The rendered artifact: a real ffprobe — the horizontal
        #    640x360 source must come out VERTICAL 9:16 H.264 with AAC.
        media = probe_streams(rendered)
        assert media["video_codec"] == "h264", media
        assert media["has_audio"] and media["audio_codec"] == "aac", media
        assert media["width"] == 1080 and media["height"] == 1920, media

        # 8) The .srt sidecar — contractual for live clips (gallery/uploads).
        srt = rendered.with_suffix(".srt")
        assert srt.is_file(), f"missing .srt sidecar next to {rendered.name}"
        assert "-->" in srt.read_text(encoding="utf-8")

        # 9) The window that was rendered brackets the burst, and the
        #    cooldown marked the clip (the circuit engaged for real).
        row = store.get_clip(clip_id)
        assert row is not None
        assert row.clip_end_s > broadcast.burst_end_s, "no post-roll after burst"
        assert cooldown.marks == [stream_id], "cooldown never marked the clip"

        # 10) Waits 3-4 (real-time pacing) mean the source is still being
        #     served here, so the watcher never saw the capture end: it
        #     went clipping for the candidate and is back to plain
        #     watching, exactly like a stream that kept talking through a clip.
        await wait_until(lambda: watcher.state is StreamState.WATCHING, timeout=15.0)

        # 11) One clip only: the cooldown kept the burst's plateau from
        #     minting a second candidate out of the same moment.
        assert len(store.list_clips(stream_id=stream_id)) == 1
    finally:
        await manager.stop()
