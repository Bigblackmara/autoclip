# Live Watch — integration ledger (Wave 1 + 2 results -> Wave 3 plan)

Everything below is REQUIRED wiring discovered from worker reports. Wave 3
agents (A10 REST, C1 CLI, integrator) implement against this file so the
cross-agent contracts don't live only in chat history.

## COMPLETED (verified by me, gates green)

- A3 registry, A9 store/migration-v2, A7 uploads (67 tests, zero network,
  ruff clean, keyring-only secrets + rights gate), A8b clip_factory (42
  tests, real-ffmpeg remux test, resume/idempotence), F1 StreamsPanel, F2
  ClipsPanel, F3 LiveSettingsSection (all: tsc -p tsconfig.json exit 0).
- A6 judge, A1 buffer (67 buffer tests green; capture has 18 failing tests
  mid-flight — A1 still running), A2 audio, A5 chat (49/64 tests green in
  isolation).

## REQUIRED WIRING (handles for Wave 3)

1. A8 manager must adapt A8b's frozen signature:
   produce(buffer, clip_row, settings, on_progress=None) -> ProduceResult
   — build LiveClipRow(stream_id, clip_start/end incl. pre/post roll,
   title from judgement, caption_style, ratio) and map ProduceResult →
   ClipWorkResult incl. transcript_path + .srt (rendered_path.with_suffix).
   A8's current produce(item) single-arg is WRONG vs the frozen contract.
2. Manager success branch: persist result.transcript_path into the
   LiveClipRow (store.update_clip), not just rendered_path.
3. Manager upload flow: RightsNotAccepted → live_uploads row 'skipped' (not
   'failed'); NotLinkedError → likewise skipped with reason; persist
   UploadResult.media_id/url into LiveUploadRow.platform_media_id/upload_url.
   Pass a shared httpx.AsyncClient via the client kwarg.
4. A10 REST endpoints still to write (api/live.py), incl. from worker
   reports:
   - POST /api/live/clips/{id}/upload → upload_youtube/upload_instagram;
     RightsNotAccepted→4xx 'accept the rights notice'; NotLinkedError→409;
     UploadApiError→502. Returns the LiveUpload row (F2 renders it).
   - POST /api/live/clips/{id}/discard → returns updated LiveClip row.
   - POST /api/live/uploads/{platform}/start → authorization_url(platform,
     redirect_uri, state) with persisted CSRF state.
   - OAuth callback endpoint → exchange_code(platform, code, redirect_uri).
   - POST /api/live/uploads/attestation → accept_rights() (keyring flag
     'accepted:<date>'); ADD GET attestation so F3 shows persisted state.
   - Client-credentials save endpoint (user's own app id/secret).
   - SSE channels MUST emit named events 'clip'/'clip_status'/'upload'/
     'state' with data = the row itself or a patch carrying id/clip_id.
   - PUT /api/live/settings: full-body LiveSettings replace (F1/F3 send the
     merged whole object).
5. Settings semantics decision (FINAL): judge_per_minute/hour/day are
   PER-STREAM caps enforced by the manager via circuit.TokenBucket per
   stream (matches settings.py docstrings and F3's hint copy). Cooldown is
   per-stream. No global bucket.
6. F1/F2/F3 integration: compose <StreamsPanel/> <ClipsPanel/>
   <LiveSettingsSection/> inside LiveWatch.tsx (integrator-owned file).
7. tsconfig note: repo has ONE frontend/tsconfig.json (no tsconfig.app.json)
   — all frontend gates run `npx tsc -p tsconfig.json --noEmit`.
8. A8b relies on A1 RingBuffer.extract() segment-granular semantics —
   extraction may exceed the requested window; downstream starts at 0
   relative to raw.ts. Do "fix" this; exact-second trim belongs in a later
   revision if ever needed.
9. A9 crash-recovery: manager.start() MUST call store.reset_interrupted_
   states() before launching watchers (A9 exposed it; A8's order didn't
   include it — integrator patches manager.py if A8 didn't).
10. Migration v2 (live_streams/live_clips/live_uploads) is append-only and
    idempotent; test_db.py needed no version bump (user_version derived
    from len(MIGRATIONS)); A9 verified.

## WAVE-2 worker file ownership (do not reassign)

- uploads/: tokens.py, youtube.py, instagram.py, __init__.py (A7)
- clip_factory.py (A8b), F1/F2/F3 three panel files (frontend/src/live/)
- tests/test_live_uploads.py, tests/test_live_clip_factory.py

## Known open items at time of writing

- A1 capture: 18 failing tests in tests/test_live_capture.py (in-flight;
  FakeProc.stderr, POSIX-signals-on-Windows, timing tests). A1 still runs;
  if its final gate is red, I fix capture.py/tests myself at integration.
- A2/A5 files visible as "M" in git — landed bodies (49/64 tests green in
  isolation) but their final combined-suite runs conflicted mid-flight on
  transient sibling states; re-verify at integration.
- A6 judge prompt: verify no local-model assumptions after its report lands
  (steer attempted; child had already finished — reconcile from report).
- docs/LIVE_WATCH.md legal notice says attestation stored in config — actual
  storage: keyring. Fix doc at integration.
