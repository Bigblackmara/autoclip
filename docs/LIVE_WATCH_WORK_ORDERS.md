# Live Watch — worker orders (Wave 1)

One file per worker. You OWN exactly the files listed under WRITE. You may
READ anything in the repo. Never edit a file you don't own — if something in
a file you don't own must change, state it in your final report instead.

Shared context ALL workers read first:
- docs/LIVE_WATCH.md (plan + house style)
- docs/CONTRIBUTING.md, docs/ARCHITECTURE.md
- The frozen skeletons in backend/autoclip/live/ — signatures are contracts.

Repo: C:/d/autoclip (Windows). Python: .venv/Scripts/python.exe (3.11).
Run tests from repo root:
  .venv/Scripts/python.exe -m pytest tests/test_live_<yours>.py -q
Full gate (must stay green when you finish):
  .venv/Scripts/python.exe -m pytest -m "not slow and not golden and not e2e" -q
Lint gate: .venv/Scripts/python.exe -m ruff check --fix backend/autoclip/live
           .venv/Scripts/python.exe -m ruff format backend/autoclip/live
Both must pass. Existing tests must remain 276 passed.

---------------------------------------------------------------

## A1 — capture + rolling buffer
WRITE: backend/autoclip/live/capture.py, backend/autoclip/live/buffer.py
(fill the frozen skeletons), tests/test_live_buffer.py, tests/test_live_capture.py

Deliver:
1. RingBuffer: implement every method in the skeleton. Segments = .ts files
   named seg_{index:06d}.ts. Pruning: retention window AND disk cap, respecting
   pins. commit_segment registers stat()ed size. Epochs: reconnect starts a
   new epoch; range extraction refuses epoch mixing (raise ValueError).
   extract(start_s,end_s,target): pick pin index cover, concat via python
   file copy (streams, not read-whole-file), return target Path.
   CRITICAL correctness rules:
   - index math: seconds->index uses SEGMENT_S (6.0), clamped to what exists.
   - pinned segments never pruned even if retention/disk says delete.
   - close() unregisters everything (rmtree by caller, not buffer).
2. Capture: StreamCapture class — spawn streamlink (preferred) or yt-dlp
   (fallback) piping to ffmpeg writing segments. Implementation approach:
   - resolve stream URL via streamlink CLI: `streamlink --json <url> best`
     (or python API streamlink.streams) to get the stream URL; fall back to
   - subprocess: ffmpeg -i <stream_url> -c copy -f mpegts -map 0
     with segment muxer into the buffer dir: -segment_time 6
     -reset_timestamps 1 -segment_start_number <start> -movflags +faststart
     NOTE: use `-f segment` muxer (not HLS) with .ts files; copy codecs.
   - a reader thread/task tails the directory: any closed seg_NNNNNN.ts file
     (size stable across two polls) gets commit_segment()'d.
   - on ffmpeg exit: emit on_event("ended") callback.
   - streamlink/yt-dlp resolution failures raise CaptureError with hint text.
   - MUST be cancellable: stop() kills the subprocess tree (create_process_group
     or taskkill /T on Windows) and stops the tailer.
   - No network in unit tests: mock subprocess resolution. A `file://`-style
   -test mode is NOT required; tests use fake segment files written by hand.
3. Buffer test coverage: retention, disk cap, pins block pruning, epoch
   refusal, extract correctness (bytes identical to concatenated sources),
   indices_back clamping. Capture tests: subprocess mocked, commit ordering
   enforced, cancellation kills (mock), URL resolution error mapping.
Report: API quirks you hit, anything another worker must know.

## A2 — audio energy signals
WRITE: backend/autoclip/live/audio_signals.py, tests/test_live_audio.py

Deliver AudioSignalAnalyzer class + per-stream worker function:
- input: shared access to the RingBuffer (via A1's skeleton contract —
  use ONLY next committed segments), ffmpeg reads a segment, downmix mono
  16k s16le via subprocess pipe, numpy computes per-second RMS.
- maintain a rolling baseline (mean+std over settings.baseline_window_s);
  a window is a SPIKE when z >= spike_z AND rms >= floor_rms. Sustained
  excitement = consecutive spike seconds; if it exceeds max_sustained_s just
  keep flagging the edge (don't spam signals: one DetectorSignal per burst
  onset + peak update via callback signature on_signal(DetectorSignal)).
- stream time base: t_s = segment_index * SEGMENT_S (+ offset within segment).
- API for the manager: start(buffer, settings, on_signal, stop_event) — a
  plain function run in a thread; lightweight: at most one ffmpeg child per
  stream at a time, sequential segment analysis, skip-ahead allowed (if
  behind, analyze only every other segment and log).
- Unit tests: synthetic pcm (numpy→wav) segments on disk; spike at t=X
  detected with correct t_s; baseline mean/std math; floor_rms suppression;
  max_sustained burst handling; stop_event honored.
Note: fft/laughter detection is OUT of scope (rms only) — docstring why.

## A4 — detector + circuit
WRITE: backend/autoclip/live/detector.py, backend/autoclip/live/circuit.py,
tests/test_live_detector.py, tests/test_live_circuit.py

Deliver:
1. circuit.py real bodies: TokenBucket(rate, capacity) refill on try_acquire
   (lazy refill by elapsed monotonic time — no background thread).
   PerStreamCooldown: dict of last clip instants; can_trigger/remaining/mark.
2. detector.py: SignalAggregator + Trigger.
   - SignalAggregator keeps per-stream deque of recent signals (window
     configurable, default 30s). evaluate(window_s) -> SignalWindow view.
   - Trigger.decide(signal_window, *, now) -> TriggerDecision dataclass:
     {fire: bool, strength: float, reasons: list[str]} — rules:
     audio spike alone: fire if strength >= 1.0 (normalized so a clean
     single-stream spike is 1.0); chat rate spike alone: fire if >= 1.2;
     audio+chat combined: fire if combined >= 1.5 with neither < 0.4;
     never fire when all strengths < 0.4 (noise floor);
     never fire more often than every `min_interval_s` (default 20) per stream.
   Strengths: audio = z/spike_z capped at 2.0; chat = z/rate_z capped 2.0;
   chat_hype events (hypetrain) count as strength 2.0 directly.
   - Everything deterministic + unit-testable: no I/O in this module.
Report the exact strength formulas (A6 embeds them in the judge prompt).

## A5 — Twitch chat signals
WRITE: backend/autoclip/live/chat_signals.py, tests/test_live_chat.py

Deliver TwitchChatReader:
- Connect to wss://irc-ws.chat.twitch.tv:443 (websocket-client via
  streamlink's dep or stdlib — check what streamlink pulls in; prefer the
  transitive websocket-client lib). Anonymous login (justinfan12345),
  JOIN #<channel>, parse PRIVMSG lines. Extract channel from a Twitch URL
  (twitch.tv/<name>); non-Twitch URLs → DisabledChat (no error).
- Rate stats: msgs/sec in a 60s baseline window; z-score like audio; also
  CAPS-ratio (>=3 caps words per 100 msgs) and emote-spam via a small
  common-emote list you embed (Kappa, PogChimp... use global emote names,
  ~30 entries, embedded constant) — emotes_threshold per settings.
- hypetrain: msgs/min >= hypetrain_msgs_per_min → immediate DetectorSignal
  kind=CHAT_HYPE strength=2.0.
- Reconnect with exponential backoff (1,2,4...max 60s) on socket drop;
  it MUST surface nothing via exceptions to the manager: callbacks only
  (on_signal, on_state). Cancellable via stop_event.
- Unit tests: feed synthetic PRIVMSG lines through the parser (no network):
  rate z-spike, hype burst, caps spam, backoff timing, channel extraction
  from URLs (twitch/yt/kick), stop behavior.

## A6 — LLM judge
WRITE: backend/autoclip/live/judge.py, backend/autoclip/prompts/live_judge_v1.txt,
tests/test_live_judge.py

Deliver judge_window() real body:
- Build the user prompt: words rendered [i]word (SAME style as
  highlights.render_window_text — import and reuse it) + a SIGNALS block:
  each cheap signal kind with strength and detector detail text.
- system prompt: live_judge_v1.txt — you author it. It must:
  - explain the setting (live stream excerpt, cheap signals fired, decide
    if this is a viral moment worth clipping)
  - demand STRICT JSON: {score, title, hook, reason, start_word_offset,
    end_word_offset} with score 0-100 same philosophy as highlight_v1
  - scoring guide aligned to A4's strength formulas ( mention them so the
    model knows what the signals mean)
- Call providers.build_provider(name, settings) with the ACTIVE provider,
  temperature=0.2, ONE repair retry (mirror providers.base pattern: catch
  ValidationError, append the error, retry once; a second failure raises
  ProviderError with hint).
- Parse via a pydantic model LiveJudgeOut (define in judge.py): coerce like
  ClipCandidate (nulls→empty, float scores→0-100 int, clamp offsets>=0).
- Return LiveJudgement (frozen skeleton). word offsets are relative to the
  window's first word; clamp to window length; None means "no trim advice".
- Unit tests: fake provider classes (copy the fake in tests/test_providers.py
  style), prompt renders with signals, repair-retry path, malformed-JSON
  failure path, offset clamping. NO network.

## A8 — watcher + manager
WRITE: backend/autoclip/live/watcher.py, backend/autoclip/live/manager.py,
tests/test_live_watcher.py, tests/test_live_manager.py

You are the biggest piece. The manager is the ONLY class the API (A10) talks
to. Read the frozen skeletons' docstrings — manager.py's API needs:
- start()/stop() lifecycle (asyncio), add_stream(url,label)->id,
  remove_stream(id), pause_stream(id), resume_stream(id),
  status()->dict, snapshots for API.
- owns a bounded asyncio.Queue of clip work items (maxsize=clip_queue_max;
  on full: drop candidate + publish warning event, never block watchers),
  clip workers = clip_queue_workers (default 1): they call
  clip_factory.produce (A8b placeholder — you define an internal
  _placeholder_produce to be replaced; your tests stub it).
- the judge+pipeline flow per stream (this is the heart):
  detectors fire → Trigger.decide → if fire and circuit.can_trigger:
    build candidate window [trigger_end - pre_roll, ... + post_roll]
    transcript slice for the window via A1's buffer.extract to a temp clip
    dir + whisper transcribe (reuse pipeline transcribe on the extracted
    audio; wrap in asyncio.to_thread) → judge.judge_window (A6) →
    if score >= judge_min_score: enqueue clip work item; circuit.mark_clip;
    update DB live_clips row (create on candidate detection, update on
    verdict). If below threshold: mark row 'rejected'.
  - RATE LIMITS from settings live.rate_limit enforced THROUGH circuit.py's
    buckets: per-minute judge_per_minute (global bucket), per-hour
    judge_per_hour, per-day judge_per_day; plus per-stream cooldown_s.
- watcher task per stream: the state machine in states.py is your single
  truth; every transition goes through can_transition + DB update +
  events.publish. Reconnect: exponential backoff 2,4,8,...cap 120s,
  jitter ±20%. Stream-ended (streamlink says "No playable streams") →
  state waiting_for_live again after policy wait (30s default poll).
- crash isolation: watcher task wrapped so ANY exception → stream state
  error + manager-continues; never propagates to the manager loop.
- Watch queue coordination: capture subprocesses write segments; audio
  analyzer (A2) and possibly chat (A5) are attached per stream; their
  callbacks route into your aggregator. You do NOT own those files —
  import and wire. Use the frozen skeletons' APIs only.
- pause semantics: stop capture + signals, keep config, state = off;
  resume: fresh watcher start.
- STARTUP: on manager.start(), load persisted streams (registry config —
  A3 placeholder note) and start watchers. On manager.stop(): cancel all,
  kill subprocesses, flush.
- Disk safety: before enqueueing clip work, check queue depth AND total
  pending bytes (sum of live/ clips dir) vs PENDING_CAP_BYTES; if exceeded,
  drop candidate with a "disk_pressure" event.
Report anything you had to stub from A1/A2/A5/A6/A8b that wasn't ready.

## A9 — DB store
WRITE: migration in backend/autoclip/db/schema.py (append _V2 + list entry —
       see the notes in live/store.py docstring),
     backend/autoclip/live/models.py from_row methods,
     backend/autoclip/live/store.py real bodies,
     tests/test_live_store.py
You own ONLY the additive parts: _V2 migration function + MIGRATIONS append +
live/models.py + live/store.py. You do NOT edit db/store.py, db/models.py,
db/schema.py's v1 — append-only migration v2 with three tables
(live_streams, live_clips, live_uploads per live/models.py field lists).
- store functions: follow db/store.py conventions (connection per thread,
  WAL already on). get/list/upsert/update/delete for streams;
  create/get/list/update for clips; create/list/update for uploads.
  update_clip/update_upload take **fields with UNSET sentinel for None clearing.
- reset_interrupted_states() changes any non-terminal state to
  waiting_for_live (crash recovery). Wire it: db/init is NOT yours; call it
  from live/store.py at first connection… simplest: expose it and have
  manager.start() call it — document that A8 must call it (put a note in
  your report; do not edit manager.py yourself).
- Tests: migration from fresh v0/v1 db (user_version 1→2), idempotence,
  round-trips, reset_interrupted_states, UNSET semantics.

## A3 — persisted stream registry
WRITE: backend/autoclip/live/registry.py, tests/test_live_registry.py
(read live/settings.py LiveStreamConfig — frozen; extend ONLY if you must,
and keep every existing field)
Deliver StreamRegistry backed by config.json (via autoclip.config load/save —
you own NO config.py edits; store streams under settings key "live_streams":
[]  — NO, better: persist in a SEPARATE file live/streams.json under
paths.root() to avoid Settings round-trip churn; decide and document).
Functions: list_streams() -> list[LiveStreamConfig], upsert(stream), delete(id),
set_enabled(id,enabled). Atomic write (tmp+replace like config.save).
Tests: empty init, round-trip, delete missing id is no-op, atomic-write
leaves old file on failure (simulate), concurrent writers serialized via lock.
Report: chosen storage location + shape (A8 and A10 read it).

---------------------------------------------------------------
Reports: each worker returns (a) what landed, (b) tests added + pass status,
(c) stubs/assumptions, (d) anything the integrator must change. No prose
essays — bullet lists. Write files, run the gates, report.
