# GTA Mode: caption box and narrator

Every GTA clip can get two extras after it is rendered (both on by default):

- **Caption box** (`static_caption`) — the clip's hook in black text on a white
  rounded box, on screen for the whole clip, near the bottom just above the
  Shorts/Reels bottom bar. The subtitles move up to sit above it, and the
  separate hook title is no longer drawn.
- **Narrator** (`voiceover`) — an AI-written opening line (who the streamer is,
  what is about to happen) and a closing question for the comments, voiced and
  mixed over the clip with the stream audio ducked underneath. This is original
  commentary for YouTube's reused-content rule.

## Voices

`voiceover_voice` picks the voice:

| Value | Engine | Needs |
| --- | --- | --- |
| `en-US-ChristopherNeural` (default), `en-US-GuyNeural`, `en-US-AndrewNeural`, any `…Neural` | Microsoft neural voices via `edge-tts` | internet; installed with AutoClip |
| `am_michael`, `af_heart`, … | Kokoro, offline on the CPU | `pip install "autoclip[voice-offline]"` + the model files below |

When a Microsoft voice can't be reached (3 tries), the Kokoro voice
`am_michael` is used instead if Kokoro is installed; otherwise the clip is
rendered without a narrator (it is never failed for this).

### Kokoro model files (offline voice only)

Put these two files in `<AutoClip home>/models/kokoro/` (or point
`AUTOCLIP_KOKORO_DIR` at another folder):

- `kokoro-v1.0.onnx` — https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
- `voices-v1.0.bin` — https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

## The lines

The active AI provider writes the two lines (120 s limit). If it fails or is
too slow, the first sentence of the clip's post caption becomes the opening
line and a stock question closes it. Clips shorter than 14 s only get the
opening line.

## Settings (GTA Mode, `config.json` → `gta`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `static_caption` | `true` | caption box on/off |
| `voiceover` | `true` | narrator on/off |
| `voiceover_voice` | `en-US-ChristopherNeural` | see above |
| `voiceover_speed` | `1.05` | speaking rate |
| `voiceover_duck_db` | `-11` | stream audio level under the narrator |
| `voiceover_outro` | `true` | the closing question |

Change them with `PUT /api/gta/settings`, e.g.
`{"voiceover_voice": "en-US-GuyNeural"}`.
