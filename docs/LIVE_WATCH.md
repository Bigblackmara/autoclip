# Live Watch

**Watch live streams around the clock; cut their best moments into clips while the stream is still going.**

Live Watch is the live counterpart of the batch pipeline. Instead of one finished
video in, it watches several streams at once, keeps a rolling buffer of each, and
runs cheap detection continuously. When something happens — the room explodes, chat
detonates — it cuts the window out, transcribes it, asks your configured LLM whether
it is actually a moment worth publishing, and renders it through the same
transcribe → reframe → captions → export stages a batch clip uses.

> **You are responsible for having the rights or permission to clip and repost
> each streamer's content, and for following each platform's terms of
> service.** This is not fine print. Uploads refuse to run at all until you
> accept that notice in the Live settings — and the acceptance is a dated flag
> in your OS keyring, not a checkbox that lives in a config file.

## What it does

```
 stream URL ──▶ capture (streamlink → ffmpeg) ──▶ rolling buffer (6 s .ts segments)
                                                       │
        audio energy spikes vs the stream's own ▶──────┤   cheap signals, always on
        rolling baseline                                │
        Twitch chat spikes (public anonymous   ▶────────┘
        IRC — no account, no bot)
                                                       ▼
                              LLM judge gate (0–100, your provider)
                                                       │ score ≥ 70
                                                       ▼
     extract the window (+20 s pre-roll, +10 s post-roll) →
     transcribe → reframe → captions → export → ready clip (.mp4 + .srt)
                                                       │  optional, per platform
                                                       ▼
              YouTube / Instagram upload (your own OAuth apps, official APIs)
```

Every rejected window is recorded too — with the score it got and why it was
dropped — so a stream that stays quiet never looks like a bug.

## Requirements

- **The `[live]` extra** — adds streamlink, the capture resolver:

  ```bash
  uv pip install -e ".[live]"
  ```

- **The same full ffmpeg build** as the batch pipeline (`autoclip doctor` checks
  this; nothing extra is needed here).
- **A configured LLM provider** for the judge (see below).
- Twitch chat following needs nothing but the URL — it reads the public
  anonymous chat gateway with no account, bot, or token.

## Quickstart

```bash
autoclip live add https://twitch.tv/someone --label "someone"
```

```bash
autoclip serve
```

`serve` starts the Live Watch manager alongside everything else (it does
nothing while `"live": {"enabled": false}` is set in config, or when the server
runs with `AUTOCLIP_NO_WORKER=1`). Each enabled stream gets a watcher:
offline channels park in `waiting_for_live` and are polled every 30 seconds;
live ones start capturing immediately.

List what your streams are doing, and what came out of them:

```bash
autoclip live              # streams table + caps
autoclip live clips        # latest clips, newest first, with judge scores
```

## How the loop works

**Capture.** Each stream is resolved once (streamlink preferred, yt-dlp
fallback) and copied into 6-second MPEG-TS segments on disk by one long-lived
ffmpeg. No re-encode, no pipe chain: two decoupled processes, so a reconnect
restarts only the resolution. If the channel is offline or drops, the watcher
reconnects with exponential backoff (2 s doubling to a 120 s cap, with jitter);
ten rapid failures park the stream in `error` instead of hot-looping.

**Rolling buffer.** The segments live in `~/.autoclip/live/buffers/<stream>/`,
kept to the last few minutes (5 min / 512 MB defaults) and pruned every few
seconds. The moment detection fires on is still on disk by construction.

**Signals.** Two cheap, always-on detectors, both measured against *each
stream's own rolling baseline* rather than absolute thresholds: audio RMS
z-score (a scream, a clutch play, a laughing fit) and Twitch chat — message
rate z-score, emote/CAPS density, and a hype-train velocity that fires
outright. Their only job is deciding when to spend an LLM call; they never
trigger a clip on their own.

**Judge.** The candidate window is transcribed and handed to your configured
provider (see the next section), which returns a 0–100 virality score, a title,
and optionally trim advice. Windows scoring below `judge_min_score` (default
70) are dropped with the score recorded. A window with no transcribed speech —
silence, music, dead air — is rejected locally without spending a call.

**Production.** The candidate waits until the buffer has captured its
post-roll, and an accepted window is cut into the clip folder right away
(`raw.ts`), so a queued clip no longer depends on the rolling buffer. It is
remuxed into a seekable MP4 and rendered through the standard pipeline:
transcribe → reframe → captions → export. The result is a vertical clip with
burned-in captions plus an `.srt` sidecar, in
`~/.autoclip/live/clips/<clip_id>/`. A failed render keeps its stage
artifacts (resume, or `POST /api/live/clips/<id>/retry`); a successful one
deletes its intermediates. The judge's window video and audio are deleted as
soon as the transcript exists; only small JSON evidence is kept.

**Upload.** Optional, per platform, off by default. Clips go out through the
official APIs (YouTube Data API v3 resumable uploads; Instagram Graph API
Reels) using *your own* registered OAuth app — your Google Cloud or Meta app's
client id/secret, stored only in your OS keyring. One attempt per platform per
clip; failures are recorded, not retried behind your back.

## The judge and your provider

The judge runs on **your configured API provider** — the same provider, model,
and key the batch highlight picker uses, picked in **Settings** like everywhere
else. Only transcript text and the signal numbers are sent; the judge never
sees or hears the stream, and with Ollama nothing leaves your machine at all.

Detection quality follows model quality, exactly as batch clip selection does,
so a hosted provider (Anthropic, OpenAI, Gemini, or any OpenAI-compatible
endpoint) is the recommendation here — your GPU is better spent on Whisper and
encoding. **Ollama remains fully supported**; the judge just runs one window at
a time instead of up to three in parallel.

Each judged window is one LLM call against your key. The budgets below are the
ceiling on that spend.

## Rights, attestation, and uploads

That bold notice at the top is enforced, not suggested:

- **You are responsible for having the rights or permission to clip and repost
  each streamer's content, and for following each platform's terms of
  service.** Uploads are **locked until you accept that notice** once in the
  Live settings. The acceptance is stored as `live_rights_attestation` =
  `accepted:<date>` in your OS keyring (on a machine with no keyring backend,
  the flag falls back to config storage like other non-secrets — but see the
  next point).
- OAuth tokens and your app's client id/secret are **keyring-only, strictly**:
  on a machine with no usable keyring backend the upload subsystem raises and
  refuses rather than writing an account-wide credential into a config file.
- Auto-upload is a per-platform opt-in flag, default off.

## CLI

The CLI manages the watch list and reads state; `autoclip serve` owns the
watching. A CLI-added stream starts the next time `serve` starts.

| command | does |
|---|---|
| `autoclip live` | every registered stream, its state, and the caps (same as `live status`) |
| `autoclip live add <url> [--label]` | register a stream on the watch list |
| `autoclip live remove <id>` | remove it — registry entry, state, and its clips |
| `autoclip live pause <id>` | stop watching, keep the configuration |
| `autoclip live resume <id>` | resume; a fresh watcher launches |
| `autoclip live clips [--stream <id>] [-n N]` | latest clips: state, score, length, title |
| `autoclip live settings` | print the `live` section of config.json |

Stream states you'll see: `off` (paused by you), `waiting_for_live`,
`connecting`, `watching`, `clipping` (a window is being judged; capture keeps
running), `reconnecting`, `ended`, `error`.

Clip rows carry: `detecting` → `queued` → `rendering` → **`ready`** — or
`rejected` (rate-limited, below min score, queue full, judge unavailable: the
reason is kept) and `failed` (render error, resumable).

## Settings

Live settings live in config.json under `"live"` (absent = all defaults), and
`autoclip live settings` prints the merged result. Hand-edit
`~/.autoclip/config.json`:

| key | default | does |
|---|---|---|
| `enabled` | `true` | the feature switch; write `false` and `serve` runs nothing live |
| `rolling_buffer.retain_minutes` | `5` | minutes of each stream kept on disk |
| `rolling_buffer.max_disk_mb` | `512` | hard cap for **one** stream's buffer |
| `rolling_buffer.pre_roll_s` | `20` | seconds captured before the moment |
| `rolling_buffer.post_roll_s` | `10` | seconds captured after |
| `rate_limit.judge_per_minute` | `3` | max judged windows **per stream** per minute |
| `rate_limit.judge_per_hour` | `60` | per stream per hour |
| `rate_limit.judge_per_day` | `500` | per stream per day — the cost ceiling |
| `rate_limit.judge_min_score` | `70` | windows below are dropped, score recorded |
| `rate_limit.cooldown_s` | `90` | after a clip, that stream can't fire again this soon |
| `detection.audio.baseline_window_s` | `30` | RMS baseline window |
| `detection.audio.spike_z` | `2.5` | z-score a window must beat to count as a spike |
| `detection.audio.floor_rms` | `0.02` | minimum absolute RMS, against loudness-war baselines |
| `detection.audio.max_sustained_s` | `30` | the longest run of spike windows still counted as one moment |
| `detection.chat.enabled` | `true` | Twitch chat reading (the only chat source in v1) |
| `detection.chat.rate_z` | `3.0` | message-rate z-score threshold |
| `detection.chat.floor_msgs_per_min` | `10` | below this, chat counts as quiet regardless of z |
| `detection.chat.emotes_threshold` | `2.0` | emote/CAPS words per 100 messages counted as a hype sign |
| `detection.chat.hypetrain_msgs_per_min` | `600` | fires above this outright |
| `workers.max_concurrent_streams` | `5` | streams being captured simultaneously |
| `workers.clip_queue_workers` | `1` | render workers; the GPU wants the whole machine, keep 1 |
| `workers.clip_queue_max` | `10` | candidate queue depth; overflow is recorded as rejected |
| `workers.disk_budget_mb` | `10240` | feature-wide budget for buffers + unfinished clips; oldest failed/rejected artifacts are freed first, still over → new candidates refused |
| `workers.min_free_disk_mb` | `2048` | new candidates are refused while the disk has less free space |
| `uploads.youtube.auto_upload` / `title_prefix` | `false` / `""` | per-platform upload opt-in |
| `uploads.instagram.auto_upload` / `caption_prefix` | `false` / `""` | per-platform upload opt-in |

Two declared knobs aren't wired yet this version and are documented as such:
`rolling_buffer.segment_s` (segments are a fixed 6 s) and
`workers.cpu_budget_percent` / `workers.ram_budget_gb` (advisory, not enforced).

## Safety nets

A 24/7 watcher must not wreck the machine or the wallet:

- **Cooldown** — 90 s per stream after any clipped moment (noise rarely waits).
- **Judge budgets** — 3/minute, 60/hour, 500/day per stream, counted from
  the `live_clips` table (a repair retry counts as a second call), so they
  survive restarts. When a budget cannot be checked, the candidate is
  **refused**, not guessed. A judged (or refused) spike is consumed: it never
  buys a second verdict, and provider errors back every stream off.
- **Queue bounds** — at most 10 candidates waiting, 1 render worker by default
  (the same "GPU wants the whole machine" philosophy as the batch queue).
  Overflow candidates are dropped *with a recorded reason*, never silently.
- **Disk caps** — 512 MB per stream buffer (pruned continuously), a 2 GiB
  ceiling on pending clip work, the feature-wide `disk_budget_mb`, and a
  free-space floor below which new candidates are refused. Removing a stream
  deletes its buffer and clip folders.
- **Crash isolation** — every watcher is its own task with its own
  subprocesses. One stream dying takes down only itself; the others and the
  app keep running.

## Where things live

| path | what |
|---|---|
| `~/.autoclip/streams.json` | the watch list (atomic writes, survives restarts) |
| `~/.autoclip/live/buffers/<stream>/` | rolling segments, pruned |
| `~/.autoclip/live/clips/<clip>/` | per-clip artifacts: window, transcript, render, `.srt` |
| `~/.autoclip/autoclip.db` | `live_streams` / `live_clips` / `live_uploads` tables |
| OS keyring | OAuth tokens, your app's client id/secret, the rights acceptance |

## What's verified, and what isn't

**Verified:** every subsystem has an offline test suite that runs in the
standard gate alongside the rest of the repo — capture and buffer (including
Windows process-tree kills), audio and chat signals, the judge, clip
production (including a real-ffmpeg remux of a capture-shaped MPEG-TS
extract into a normalized MP4), the uploaders (fake HTTP transports only —
zero network), the manager, the store, the registry, and the CLI. The
keyring-only rule for OAuth credentials is test-enforced: the code raises
rather than falling back to config storage.

**Not verified, and not pretending to be:**

- **No end-to-end run on a real live source yet.** Detection → judged clip has
  not been proven against a live HLS/RTMP push; the `live` pytest marker is
  reserved for exactly that test and it hasn't been run.
- **The trigger and its pacing circuit are the last piece being integrated.**
  Until they land, watchers run capture-only: signals are collected, but
  candidates are refused (fail-closed) rather than risked, so detection
  produces no clips yet and no uncontrolled LLM spend is possible.
- **The `/api/live` REST+SSE layer is still landing**, so the Live Watch tab in
  the web app renders its layout but has no backend to talk to yet. The CLI is
  the working surface today.
- **Uploads have never run against real Google or Meta accounts.** The flows
  are coded to the documented APIs and tested against fake transports; the
  first real attempt will be the first real proof.
- **Auto-upload flags exist in settings but their wiring lands with the REST
  layer** — today there is also no surface to accept the rights attestation
  yet, so uploads stay locked by design.
- **The same unverified items as the batch pipeline apply** — see the
  [README](../README.md#what-has-and-hasnt-been-verified): the reframe
  golden-set bar hasn't been measured on live-clipped footage, and whether
  the picks are *good* is a judgement about your streams and your model that
  no test settles.

Saving the core Settings page keeps the `live` section intact, and a Live
settings save (`PUT /api/live/settings`) is applied to the running server
immediately — switching Live Watch off stops every capture.

## Scope

- **Chat is Twitch-only in v1.** Kick and YouTube Live have no public chat
  API; their *video* is capturable, but chat signals don't exist for them.
- **No visual-only signals** (scene-cut detection on live H.264 is noisy and
  costly) and no reframe-quality work beyond what the batch reframe stage
  already does.
- **One upload attempt** per platform per clip — manual or auto, no retry
  loops. TikTok has no official upload API path and is not supported.
