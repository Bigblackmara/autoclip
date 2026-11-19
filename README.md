# AutoClip

**Open-source, local-first AI video clipper.** Long video in → ranked, caption-burned, speaker-tracked 9:16 clips out.

[![CI](https://github.com/artbyjazi/autoclip/actions/workflows/ci.yml/badge.svg)](https://github.com/artbyjazi/autoclip/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)

Paste a YouTube link or drop a file. AutoClip transcribes it, uses an LLM to find the moments worth clipping, reframes them to vertical while tracking whoever is speaking, burns in animated captions, and exports platform-ready MP4s.

No accounts. No uploads to anyone's servers. No watermarks. No subscription.

---

## Why this exists

Existing open-source clippers stop at "works on my machine" — no UI, janky reframing, ugly captions. AutoClip ships the whole loop: ingest → clips → review → export.

Two ways to run it:

- **Fully local** — Whisper + Ollama. Nothing leaves your machine, no API costs.
- **Bring your own key** — Anthropic, OpenAI, Gemini, or any OpenAI-compatible endpoint (OpenRouter, Groq, DeepSeek, LM Studio) for better clip selection.

Only transcript *text* is ever sent to a provider — never video or audio. With Ollama, nothing is sent at all.

## Status

Working end to end: ingest, transcription, highlight detection across four providers, speaker-tracked reframing, four caption styles, export at three aspect ratios, and the full review UI. Verified on real footage — see [what "verified" means](#what-has-and-hasnt-been-verified).

Pre-1.0. The reframe quality bar hasn't been validated against a fixed golden set yet, which is the gate for tagging v0.1.0.

## Requirements

| | |
|---|---|
| **Python** | 3.11 or 3.12 — **not 3.13.** MediaPipe publishes no 3.13 wheels and the reframe stage needs it. |
| **ffmpeg** | A *full* build with `libass` and `libx264`. Both `ffmpeg` and `ffprobe` on PATH. See below — the default package is the wrong one on macOS and Windows. |
| **Node** | 20+, to build the UI. Not needed at runtime. |
| **GPU** | Optional. NVIDIA or Apple Silicon speeds up transcription several-fold; CPU works, just slower. |

`autoclip doctor` checks all of this and tells you exactly what to fix. Run it before anything else.

### Installing ffmpeg correctly

Captions are burned in with **libass**, and on two platforms the obvious package doesn't include it. It installs cleanly, then fails to burn a single caption.

**macOS** — Homebrew split the formula; the plain one is a reduced build:

```bash
brew install ffmpeg-full
```

If you already installed the plain one: `brew unlink ffmpeg && brew link --force --overwrite ffmpeg-full`.

**Windows** — install the *full* build, not `Gyan.FFmpeg.Essentials`:

```bash
winget install Gyan.FFmpeg
```

**Linux** — the distro package is fine:

```bash
sudo apt install ffmpeg
```

## Quickstart

```bash
git clone https://github.com/artbyjazi/autoclip.git
```

```bash
cd autoclip && uv venv --python 3.11 && uv pip install -e ".[dev]"
```

```bash
cd frontend && npm install && npm run build && cd ..
```

```bash
autoclip doctor
```

```bash
autoclip serve
```

That opens `http://localhost:8000`. If you don't use [uv](https://docs.astral.sh/uv/), a plain `python -m venv .venv` and `pip install -e ".[dev]"` works the same way.

### Add a provider

Clip selection needs a language model. Either paste a key in **Settings → Keys**, or:

```bash
autoclip config set-secret anthropic
```

Keys go into your OS keyring — Credential Manager, Keychain, or Secret Service — never into a config file. If no keyring backend exists, AutoClip falls back to a file **and says so**, in the UI and in `doctor`.

For a fully local setup, install [Ollama](https://ollama.com) and pull a model instead:

```bash
ollama pull llama3.1:8b
```

Clip quality tracks model quality closely. A 7B model returns valid JSON full of mediocre picks; a frontier model is noticeably better at spotting a real hook. That's the honest trade for running offline.

### Optional extras

```bash
uv pip install -e ".[gpu]"
```

CUDA runtime libraries for NVIDIA GPUs. Install these if `doctor` reports that CTranslate2 can't see your GPU — pip puts them somewhere the OS loader doesn't search, and AutoClip registers the location itself. On macOS this deliberately installs nothing: there's no CUDA there, and Apple Silicon takes the CPU path.

```bash
uv pip install -e ".[diarization]"
```

Speaker diarization via WhisperX — what lets the reframe stage cut to whoever is talking in a multi-person video. Pulls PyTorch (large), and needs a HuggingFace token plus acceptance of the gated pyannote model licences.

### Docker

The guaranteed path — ffmpeg, Python version, and fonts all pinned:

```bash
docker compose -f docker/compose.yaml up --build
```

```bash
docker compose -f docker/compose.yaml --profile gpu up --build
```

## Using it

Everything the UI does is also on the CLI:

| command | does |
|---|---|
| `autoclip doctor` | check this machine and explain what's missing |
| `autoclip serve` | start the web app |
| `autoclip clip <url\|file>` | run the whole pipeline and export |
| `autoclip jobs` | list recent jobs |
| `autoclip providers` | check which providers are reachable |
| `autoclip styles` | list caption presets |
| `autoclip config show` | print settings |
| `autoclip update-ytdlp` | update yt-dlp after a YouTube change |
| `autoclip live` | manage the live watch list and its clips |

### Batch jobs

- **Options** on the New page (provider, Whisper model, clip count and lengths, speakers) apply to that job only, and the job really runs with them. Invalid values are caught before anything is downloaded.
- **Cancel** stops a job at its next checkpoint, including mid-transcription and mid-render; it is recorded as *cancelled*, not failed. **Retry** works on failed and cancelled jobs, resumes at the stage that stopped, and skips clips that were already exported.
- Exported files are named by rank, e.g. `01-the-title_9x16.mp4`, so clips with the same or emoji-only titles never overwrite each other. Re-exporting a clip at the same ratio replaces its file and its download entry.
- The review page opens each clip at the ratio and caption style its job rendered with. After a ratio change or a trim, re-export frames the clip for its new shape and length (a centre crop) instead of stretching the old framing.
- Audio-only sources (mp3, wav, m4a) render over a generated background.

### Caption styles

| style | look |
|---|---|
| `bold_pop` | chunky white, heavy outline, spoken word grows and turns yellow |
| `karaoke_fill` | words fill with colour exactly as they're spoken |
| `clean_lower` | minimal lower third, no animation |
| `boxed` | high-contrast text on a solid block |
| `pop` | big three-word bursts, spoken word gold, key words green |
| `neon` | glowing cyan outline, spoken word mint, key words magenta |
| `comic` | yellow comic lettering two words at a time, key words red |
| `minimal` | sentence-case lines with a soft shadow, only key words coloured |

Fonts are bundled under the SIL Open Font License, so nothing is fetched at runtime.

### Editing (normal-mode exports)

Every batch export and re-export gets short-form editing on top of the crop, in the same ffmpeg pass. All of it lives in the `export` settings (Settings → Editing):

| setting | default | does |
|---|---|---|
| `pace` | `natural` | cuts pauses longer than ~0.6 s (`fast`: nearly every pause; `off`: none). Cuts land only on silences measured from the audio, never through a transcribed word; a lone "um" between two pauses goes with them. Captions are re-timed and each jump cut gets a small zoom step to hide it. |
| `hook_title` | on | the clip's title as a headline for the first ~3 s |
| `motion` | `subtle` | punch-in zooms at natural pauses (`subtle` every ~4-5 s, `dynamic` more often plus a quick zoom-out on the hook, `off`). Never on the facecam split layout. |
| `keyword_highlight` | on | 3-8 important words per clip (numbers first, then the longest non-stopwords) drawn bigger in the style's accent colour, one per line |
| `music_enabled`, `music_path`, `music_volume`, `music_duck` | off, —, 30 %, on | a track file or a folder (one track picked per clip), looped, faded in/out and ducked under speech |
| `watermark`, `watermark_text`, `watermark_image`, `watermark_corner`, `watermark_opacity` | `off` | a text logo or a PNG in any corner |
| `fps`, `two_pass_loudnorm` | 30, on | forced output frame rate; loudness measured first, then normalised linearly to `loudness_lufs` (-14) |

## Streamer clips: facecam layout

Gameplay streams with a webcam overlay lose the streamer's face in a plain 9:16 crop. The `layout` export setting can instead stack the facecam on top and a full-height gameplay crop underneath.

| layout | does |
|---|---|
| `auto` (default) | splits only when the detected cam looks like an overlay; everything else keeps the crop |
| `split` | splits whenever a facecam is found at all; with none found, it crops |
| `crop` | always uses the speaker-following crop, and skips facecam detection |

**When `auto` splits:** the face is small (at most 8% of the frame width), the cam covers at most 36% of the frame, and it is docked against an edge, away from the centre. Talking heads, podcasts and full-screen "just chatting" stretches keep the crop. The cam panel takes 30–42% of the height, following the cam's own shape, and is never stretched; captions sit below the seam.

**Choosing it per job:**

- UI: **Layout (9:16)** under **Options** on the New page (this job only), as the default in **Settings → Export**, or per clip on the Review page before re-exporting.
- CLI:

```bash
autoclip clip <url|file> --layout split
```

**Known limits:**

- In squad lobbies with several cams, the streamer's cam is picked by a home-cam prior: about 60 frames are sampled across the whole source and the cam seen most often is taken as home. A clip where the home cam can't be matched keeps the crop rather than showing a teammate. If the streamer's cam is on screen less than a teammate's, the prior can pick the wrong one.
- 1:1 and 16:9 always crop.
- Detection runs on the CPU (MediaPipe) and adds about a minute per long source. Results are cached in `~/.autoclip/work/<job>/facecam/` (`source-prior.json` plus one file per clip), so a retry or re-export doesn't pay it again. A detection failure falls back to the crop and never fails the job.

## Live Watch

Watch live streams around the clock and cut their viral moments into caption-burned vertical clips while the stream is still going — several streams at once, detection running continuously against each stream's own rolling baseline (audio energy + Twitch chat), an LLM gate that scores the window before a single clip is rendered, and the same transcribe → reframe → captions → export stages a batch clip uses.

**You are responsible for having the rights or permission to clip and repost each streamer's content, and for following each platform's terms of service.** Uploads refuse to run until you accept that notice, and the acceptance is a dated flag in your OS keyring — next to the OAuth tokens, which never touch a config file.

```bash
uv pip install -e ".[live]"     # streamlink, the capture resolver
autoclip live add https://twitch.tv/someone --label "someone"
autoclip serve
```

Optional uploads go out through the official YouTube and Instagram APIs using your own OAuth app — one exception to the "no direct posting" stance below, and only ever with apps you own and accounts you control.

Detection runs on your configured LLM provider (Ollama still supported), per-stream judge budgets cap the spend, and everything the loop does is bounded: rolling buffers (5 min / 512 MB per stream), a clip queue that drops overflow with a recorded reason, and a global disk ceiling. See [docs/LIVE_WATCH.md](docs/LIVE_WATCH.md) for the full loop, settings, and — honestly — what is verified so far and what isn't yet.

### GTA Mode caption box and narrator

GTA clips get a white caption box near the bottom and an AI narrator (free
Microsoft neural voice via `edge-tts`, or Kokoro offline). Setup and settings:
[docs/GTA_NARRATOR.md](docs/GTA_NARRATOR.md).

## Troubleshooting

Everything here is a real failure hit during development, not hypothetical.

**`autoclip doctor` says Python is wrong.** You're on 3.13. MediaPipe has no wheels for it. `uv venv --python 3.11`.

**Transcription fails with "Library cublas64_12.dll is not found".** The CUDA runtime libraries aren't installed. `uv pip install -e ".[gpu]"`. AutoClip registers their location itself — pip installs them somewhere the OS loader doesn't search, which is why the error is so unhelpful.

**"Requested int8_float16 compute type, but the target device does not support..."** Clear `whisper.compute_type` in `~/.autoclip/config.json` and let AutoClip choose. It queries the backend for what's actually supported rather than guessing from your GPU model.

**Captions don't appear in exports, or ffmpeg says "No such filter: ass".** Your ffmpeg has no libass. `doctor` reports this and prints the right command for your platform. On macOS that means `brew install ffmpeg-full`, not `brew install ffmpeg`; on Windows the full Gyan build, not "essentials".

**YouTube downloads fail with a bot check.** As of 2026, YouTube blocks most anonymous downloads and proof-of-origin tokens no longer clear it. Set **Settings → Ingest → cookies from browser** to a browser you're signed into, and **close that browser** first — it locks its cookie database while running. Uploading a file always works and needs none of this.

**"YouTube refused to serve this video to your network."** YouTube answered with HTTP 403/429 or offered only storyboard images. This is almost always a VPN or datacenter address (Cloudflare WARP, Proton VPN and similar): turn the VPN off or exclude youtube.com from it, or set cookies from browser as above. If it persists, run `autoclip update-ytdlp`. The error shows yt-dlp's original message under the hint.

**YouTube gives "no video formats" on a machine with Node but not Deno.** YouTube now requires its player challenge to be solved by a JavaScript runtime. AutoClip hands yt-dlp every runtime it finds on `PATH` (Deno, Node or Bun), so installing any one of them is enough. `autoclip update-ytdlp` installs `yt-dlp[default]` (which includes the challenge solver) and also works in uv-created environments that have no pip.

**A job fails at Highlights with an API error.** When every transcript window fails, the job now shows the provider's own error (bad key, unknown model, wrong base URL, unreachable server) instead of "No clips were found". Fix it in **Settings** and press **Retry**: the transcript is reused, and provider model and base URL are read fresh on retry.

**Exports are slower than expected.** Check `doctor` for GPU encoding. A build can list `h264_nvenc` and still be unusable if your driver is older than the NVENC API it was compiled against; AutoClip probes this and falls back to CPU encoding, which is identical quality and just slower.

**No sound in the review player.** Click **Test audio** under the player. It measures the actual signal leaving the video element and tells you whether the problem is in AutoClip or between your browser and your speakers.

## How it works

```
ingest → prepare → transcribe → highlights → reframe → captions → export
```

Each stage writes artifacts to `~/.autoclip/work/{job_id}/`, so a retry resumes at the stage that failed rather than starting over. Two decisions carry most of the design:

**Highlight detection returns word indices, not timestamps.** Models are unreliable at arithmetic and completely reliable at copying a number they can see. Timing is looked up from measured word timings afterwards.

**Crop paths never interpolate across a cut.** Shots are detected first and framed independently. Panning through an edit is the most obvious sign of an auto-reframed video.

### How clip boundaries are chosen

The model proposes a word range and quotes its hook; the cut points are then fixed from measured timings:

- **Start on the hook.** When the quoted hook is found near the proposed start, the clip starts there, not on a preamble before it. Up to three leading connectives or hesitations ("and", "but", "so", "um") are dropped.
- **End on a whole thought.** An end with no punctuation backs off a dangling function word ("the", "to", "let"), and padding never runs into the neighbouring word.
- **Pauses stand in for missing punctuation.** Whisper often leaves fast speech unpunctuated. Where no sentence-ending punctuation is within reach, a long pause (at least `max(0.6 s, p90 of the transcript's own gaps)`, capped at 1.5 s) counts as a sentence break: the longest one in reach wins, pauses after a function word don't count, and punctuation always wins when both are in reach.
- **Keep the reaction.** A loud reaction that starts right after the last line (laughter, a shout) is kept, up to 4 s and within the max clip length. Loudness is measured once per job with ffmpeg's ebur128 filter and cached as `loudness.json` in the job folder.
- **No overlaps.** Partially overlapping clips are merged when the result still fits the max length; otherwise the weaker one is trimmed at a sentence edge, or dropped if what's left is too short.

The prompt asks for the same things: start on the hook, end after the reaction, never return overlapping clips.

[ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the rest, including why several odd-looking choices exist.

## What has and hasn't been verified

**Has been:** the full pipeline on real talking-head footage, asserted end to end — 1080×1920 h264/yuv420p output, AAC at −14 LUFS, crop segments tiling each clip without gaps, captions burned in, framing correctly following a cut to a second speaker. Run it yourself:

```bash
AUTOCLIP_E2E_MEDIA=/path/to/clip.mp4 pytest -m e2e
```

**Hasn't been:** the §6.4 reframe acceptance bar — no visible jitter, no cut-off faces, speaker on screen ≥95% of speaking time — against a fixed three-video golden set. The mechanics have unit coverage, but "looks right on footage I picked" isn't the bar. This is the gate for v0.1.0.

Also unverified: whether the clip *picks* are good. That's a judgement call about your material and your model, and no test settles it.

## Non-goals

Deliberate, not oversights:

- **No direct posting** to TikTok/Instagram/YouTube. Their APIs are approval-gated; AutoClip exports files. *(One exception: Live Watch's optional uploading, which runs through the official YouTube/Instagram APIs via the user's own OAuth apps when the user has the rights — see [Live Watch](#live-watch).)*
- **No cloud version, no accounts, no telemetry.**
- **No timeline editor** beyond trim handles and caption edits.
- **No DRM circumvention**, ever.

## Contributing

Prompts and caption styles are the highest-leverage places to start, and neither needs deep knowledge of the codebase — `backend/autoclip/prompts/highlight_v1.txt` is plain text and affects output quality more than most code changes. See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Legal

AutoClip bundles [yt-dlp](https://github.com/yt-dlp/yt-dlp). **Only download content you own or have the rights to process.** AutoClip contains no workarounds for DRM or paywalled content and never will.

## License

MIT — see [LICENSE](LICENSE). Bundled fonts (Anton, Inter, Bangers, Poppins) are under the SIL Open Font License.
