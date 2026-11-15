# Changelog

Notable changes. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [semantic versioning](https://semver.org/) once 0.1.0 ships.

## Unreleased

Everything so far. The first tagged release waits on the reframe acceptance bar
being validated against a fixed golden video set — see
[the README](README.md#what-has-and-hasnt-been-verified).

### Added

- **Pipeline** — ingest (YouTube via yt-dlp, or file upload), audio preparation,
  transcription with word-level timings, LLM highlight detection, speaker-tracked
  reframing, ASS caption generation, and export at 9:16 / 1:1 / 16:9.
- **Four LLM providers** behind one interface: Anthropic, OpenAI-compatible (any
  `base_url`, covering OpenRouter, Groq, DeepSeek, LM Studio), Google Gemini, and
  Ollama. Malformed responses are retried once with the validation error attached.
- **Reframing** with shot detection, MediaPipe face tracking, active-speaker
  selection from mouth movement correlated against diarization, and a
  One Euro Filter with dead zone and velocity clamp. Shots are framed as TRACK,
  WIDE, or GENERAL; subjects too far apart to crop get a fitted frame over a
  blurred fill rather than someone being cut out.
- **Four caption styles** with bundled OFL fonts, so nothing is fetched at runtime.
- **Web UI** — ingest, live job progress over SSE, clip review with word-snapping
  trim handles and caption editing, and export.
- **Resumable stages.** Artifacts live on disk per job, so a retry restarts at the
  stage that failed.
- **`autoclip doctor`** — probes ffmpeg features, GPU acceleration, compute-type
  support, and provider reachability, and explains what to do about each.
- **End-to-end test** against real media, covering every stage with only the
  language model's answer scripted.

### Added

- **Live Watch** — watch several livestreams at once, around the clock, and clip
  their viral moments while the stream is still going:
  - Multiple concurrent stream watches (streamlink → ffmpeg capture, offline
    channels parked and re-polled, reconnect with exponential backoff).
  - A rolling on-disk buffer per stream (5 min / 512 MB defaults, 6 s segments,
    continuously pruned) so the moment is still there when detection fires.
  - Audio-energy and Twitch-chat detection, scored as z-spikes against each
    stream's own baseline (Twitch chat is the public anonymous IRC gateway — no
    account, no bot).
  - An LLM judge gate on the configured provider (Ollama still supported; the
    judge just runs one window at a time), returning a 0–100 score, title, and
    trim advice before any clip is rendered. Windows with no speech are
    rejected locally without spending a call.
  - Clip production reusing the batch pipeline stages (transcribe → reframe →
    captions → export) with a per-clip artifact directory, so a failed render
    resumes at the failed stage.
  - Optional YouTube/Instagram uploads through the official APIs using the
    user's own OAuth apps — tokens and client credentials are keyring-only, and
    uploads stay locked until the user accepts the content-rights notice
    (attestation stored in the keyring, not config). One attempt per platform
    per clip, recorded either way.
  - A `live` CLI command group (`autoclip live add/remove/pause/resume/clips/…`)
    that manages the watch list; `autoclip serve` owns the watching.
  - A Live Watch tab in the web UI — streams dashboard, clips gallery, and live
    settings — fed by REST + SSE under `/api/live`.
  - Safety nets throughout: per-stream cooldown, per-stream judge budgets
    (3/min, 60/hour, 500/day) that fail closed, a bounded render queue that
    drops overflow with the reason recorded, per-stream buffer caps plus a
    global pending-work disk ceiling, and full crash isolation per watcher.
  - [docs/LIVE_WATCH.md](docs/LIVE_WATCH.md) — the user guide, including what
    is verified so far and what isn't.

### Fixed

- **YouTube downloads**: yt-dlp is given every installed JavaScript runtime
  (Deno, Node, Bun) for YouTube's player challenge, and `yt-dlp[default]` is a
  dependency. HTTP 403/429 and "only images available" now explain that YouTube
  refused the network (usually a VPN) instead of showing a raw error.
  `autoclip update-ytdlp` falls back to uv when the environment has no pip.
- **Batch jobs honour their own options**: clip lengths and count, ratio,
  caption style and Whisper model chosen for a job were silently replaced by the
  saved settings.
- **Cancel and retry**: cancelling during transcription or a render is recorded
  as *cancelled* (not failed), a cancelled job can be retried, and a retry skips
  clips already exported.
- **Providers**: Anthropic requests no longer 400 on current models;
  OpenAI-compatible endpoints that reject `temperature` or `response_format`
  are retried without it; Gemini can no longer hang the queue; Ollama gets a
  context window large enough for the prompt. When every window fails, the job
  shows the provider's error rather than "No clips were found".
- **Exports**: files carry a rank prefix (`01-title_9x16.mp4`) so duplicate or
  emoji titles can't overwrite each other; renders are written to a temporary
  file and renamed, so a failed render never leaves a truncated MP4 or destroys
  the previous one; after a ratio change or trim, re-export no longer stretches
  framing computed for the old shape or length;
  audio-only sources render over a generated background; captions show braces
  and backslashes literally.
- **Web UI**: the review page pre-selects each clip's job ratio and caption
  style (and keeps the style when only the ratio changes); re-exporting to the
  same file replaces its download entry instead of listing a stale one;
  switching clips quickly can no longer load (and save) one clip's captions
  onto another; "Export kept" and Cancel can't be double-fired; long multi-line
  errors wrap and keep their line breaks; validation errors name the field;
  job options are checked before the download starts; the per-job speakers
  checkbox can turn diarization off; the recent-jobs list refreshes while a job
  runs; settings edits send only the changed field, so quick successive edits
  no longer revert each other, and clearing a number field no longer saves 0.

### Notable fixes during development

Each of these was found by running the thing, not by reading it:

- **Compute-type selection asked the wrong question.** Inferring from CUDA compute
  capability chose `int8_float16` on a GTX 1060, which advertises fp16 to CUDA and
  then can't use it. Selection now queries CTranslate2 for what it actually
  supports.
- **CUDA libraries were installed and unloadable.** pip places them where the OS
  loader doesn't look, so the model loaded and died at first inference. AutoClip
  registers the directories itself.
- **`h264_nvenc` being listed is not proof it works.** A build can require a newer
  NVENC API than the driver provides. Encoder selection now runs a one-frame probe.
- **Filtergraph paths need two levels of escaping**, and quoting interacts badly
  with apostrophes. Renders now use bare relative names with ffmpeg's working
  directory set, removing the problem instead of out-escaping it.
- **The wheel shipped with no UI.** hatchling honours `.gitignore`, and the built
  frontend is gitignored; declared via `artifacts`.
- **Retry could never clear a job's error message**, because the update helper
  skipped `None` values.
- **Progress ran backwards** at every stage boundary, double-counting each
  completed stage.
- **A 9:16 preview rendered nearly square.** A max-height clamp shortened the box
  without narrowing it, silently violating the declared aspect ratio.
- **The preview showed different framing from the export**, centre-cropping while
  the renderer tracked the speaker.
