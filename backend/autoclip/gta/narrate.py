"""Static caption box + AI narrator voiceover, applied to a rendered clip.

YouTube's reused-content rule (Partner Program) wants original commentary on
top of stream clips, not a straight re-upload. So after the render:

* a static caption box (white, rounded, black text: the hook) sits on screen
  for the whole clip, just above the stream frame;
* an AI narrator opens the clip with a line of context and closes it with a
  question for the comments. The lines are written by the AI (rule-based
  fallback) and voiced locally with Kokoro (free, offline), and the stream
  audio is ducked under the narrator.

Never fatal: anything that fails leaves the rendered clip as it was and says
why (the clip still goes through QA).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..pipeline import edit as shared_edit
from ..pipeline import ffmpeg
from ..pipeline.captions import FONT_DIR
from ..pipeline.export import encoder_args

log = logging.getLogger(__name__)

OUT_W, OUT_H = 1080, 1920
#: Caption box: widest it gets, inner padding, corner radius, gap above the frame.
BOX_MAX_W = 840
BOX_PAD_X, BOX_PAD_Y = 40, 26
BOX_RADIUS = 30
BOX_GAP_PX = 26
#: Never above / below these (the platform's top and bottom UI cover them).
BOX_MIN_TOP = 170
BOX_MIN_BOTTOM = 330
FONT_FILE = "Poppins-ExtraBold.ttf"
FONT_SIZES = (68, 62, 56, 50, 46)
MAX_LINES = 3

#: Narrator timing (output seconds).
INTRO_AT_S = 0.15
OUTRO_END_PAD_S = 0.3
OUTRO_MIN_CLIP_S = 14.0
#: Stream audio left clear between the intro and the outro.
MIN_CLEAR_S = 5.0
DUCK_ATTACK_S = 0.15
DUCK_RELEASE_S = 0.3
#: Narrator loudness before the mix (the stream sits at -14 LUFS).
VOICE_LUFS = -13.0
#: Microsoft neural voice (edge-tts): tries and timeout per line.
EDGE_TRIES = 3
EDGE_TIMEOUT_S = 30.0
#: Offline voice used when the Microsoft voice can't be reached.
KOKORO_FALLBACK = "am_michael"
#: The AI gets this long for the lines, then the rule-based ones are used.
LINES_TIMEOUT_S = 120.0
INTRO_MAX_WORDS = 18
OUTRO_MAX_WORDS = 12

OUTROS = (
    "Would you have done the same?",
    "Be honest, what would you have done?",
    "Rate this moment from one to ten.",
    "Who else would react like that?",
    "What would you have done here?",
    "Would you have stayed calm?",
)


def model_dir() -> Path:
    override = os.environ.get("AUTOCLIP_KOKORO_DIR")
    return Path(override) if override else paths.root() / "models" / "kokoro"


def kokoro_installed() -> bool:
    folder = model_dir()
    return (folder / "kokoro-v1.0.onnx").is_file() and (folder / "voices-v1.0.bin").is_file()


@dataclass(frozen=True)
class Narration:
    intro: str
    outro: str = ""


# --------------------------------------------------------------------------
# The caption box
# --------------------------------------------------------------------------


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _balanced(draw, text: str, font, max_w: int) -> list[str]:
    """Same line count as a greedy wrap, but lines as even as possible."""
    lines = _wrap(draw, text, font, max_w)
    if len(lines) < 2:
        return lines
    count, best = len(lines), lines
    lo, hi = 1, max_w
    while lo < hi:  # the narrowest width that still fits in ``count`` lines
        mid = (lo + hi) // 2
        trial = _wrap(draw, text, font, mid)
        if len(trial) <= count and all(draw.textlength(t, font=font) <= mid for t in trial):
            hi, best = mid, trial
        else:
            lo = mid + 1
    return best


def caption_png(text: str, out: Path) -> tuple[int, int]:
    """A white rounded box with ``text`` in black, centred. Returns (w, h)."""
    from PIL import Image, ImageDraw, ImageFont

    text = " ".join(text.split())
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    inner = BOX_MAX_W - 2 * BOX_PAD_X
    font = lines = None
    for size in FONT_SIZES:
        font = ImageFont.truetype(str(FONT_DIR / FONT_FILE), size)
        lines = _balanced(probe, text, font, inner)
        if len(lines) <= MAX_LINES - 1 or (size == FONT_SIZES[-1] and len(lines) <= MAX_LINES):
            break
    assert font is not None and lines is not None
    lines = lines[:MAX_LINES]
    ascent, descent = font.getmetrics()
    line_h = round((ascent + descent) * 1.08)
    text_w = max(int(probe.textlength(line, font=font)) for line in lines)
    w = min(BOX_MAX_W, text_w + 2 * BOX_PAD_X)
    h = line_h * len(lines) + 2 * BOX_PAD_Y
    shadow = 8
    img = Image.new("RGBA", (w + shadow, h + shadow), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (shadow, shadow, w + shadow - 1, h + shadow - 1), BOX_RADIUS, fill=(0, 0, 0, 70)
    )
    draw.rounded_rectangle((0, 0, w - 1, h - 1), BOX_RADIUS, fill=(255, 255, 255, 255))
    y = BOX_PAD_Y
    for line in lines:
        lw = probe.textlength(line, font=font)
        draw.text(((w - lw) / 2, y), line, font=font, fill=(0, 0, 0, 255))
        y += line_h
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return img.size


def box_top(anchor: tuple[str, int], box_h: int) -> int:
    """Top edge for the box: ("bottom", y) ends it at y; ("centre", y) centres
    it on y; ("above", y) puts it just above y."""
    kind, y = anchor
    if kind == "bottom":
        top = y - box_h
    elif kind == "centre":
        top = y - box_h // 2
    else:
        top = y - BOX_GAP_PX - box_h
    return max(BOX_MIN_TOP, min(OUT_H - box_h - BOX_MIN_BOTTOM, top))


# --------------------------------------------------------------------------
# The narrator's lines
# --------------------------------------------------------------------------

SYSTEM = (
    "You write the narrator voiceover for a YouTube Short built from a streamer's GTA 6 "
    "clip. The narrator adds context and commentary, like a friend telling the story. "
    'Return JSON only: {"intro": str, "outro": str}.\n'
    "intro: ONE spoken sentence, 8-16 words, present tense, says who the streamer is and "
    "sets up what's about to happen WITHOUT giving away the punchline. Natural, casual, "
    "no emojis, no hashtags, no 'in this video', no quotes from the clip.\n"
    "outro: ONE short question to the viewer, 5-10 words, that invites a comment about "
    "this exact moment.\n"
    "Write numbers as words. Keep swear words out."
)


def _words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit]).rstrip(",;:")


def _sentence(text: str) -> str:
    text = re.sub(r"[#@]\w+", "", text or "")
    text = re.sub(r"[^\w\s,.'!?$%-]", "", text)
    text = " ".join(text.split())
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]


def fallback_lines(streamer: str, caption: str, hook: str, seed: str = "") -> Narration:
    """Rule-based narration from the post caption (always available)."""
    intro = _sentence(caption)
    if not intro and streamer:
        intro = f"{streamer} is playing GTA 6, and then this happens."
    intro = _words(intro, INTRO_MAX_WORDS)
    if intro and intro[-1] not in ".!?":
        intro += "."
    outro = OUTROS[sum(map(ord, seed or hook)) % len(OUTROS)]
    return Narration(intro=intro, outro=outro)


def _clean(line: str, limit: int) -> str:
    line = " ".join(str(line or "").replace('"', "").split())
    line = re.sub(r"[#*_~`]|[\U0001F000-\U0001FFFF]", "", line).strip()
    return _words(line, limit)


def write_lines(
    *,
    streamer: str,
    hook: str,
    caption: str,
    summary: str,
    transcript: str,
    complete: Callable | None = None,
    seed: str = "",
) -> tuple[Narration, str]:
    """The narrator's intro + outro, and where they came from ("ai" / "rules")."""
    fallback = fallback_lines(streamer, caption, hook, seed)
    try:
        from ..providers.base import DetectionConfig, extract_json_object

        if complete is None:
            from .pipeline import _provider_complete

            complete = _provider_complete()
        user = (
            f"Streamer: {streamer}\nClip title: {hook}\nWhat happens: {summary or caption}\n"
            f"What is said (transcript): {transcript[:1200]}"
        )
        raw = asyncio.run(
            asyncio.wait_for(
                complete(SYSTEM, user, DetectionConfig(temperature=0.7)), LINES_TIMEOUT_S
            )
        )
        data = extract_json_object(raw)
        intro = _clean(data.get("intro", ""), INTRO_MAX_WORDS)
        outro = _clean(data.get("outro", ""), OUTRO_MAX_WORDS)
        if len(intro.split()) < 5:
            raise ValueError("intro too short")
        if intro[-1] not in ".!?":
            intro += "."
        if outro and outro[-1] not in ".!?":
            outro += "?"
        return Narration(intro=intro, outro=outro or fallback.outro), "ai"
    except Exception as exc:  # no provider / quota / bad answer
        log.info("Narration from rules (%s: %s).", type(exc).__name__, exc)
        return fallback, "rules"


# --------------------------------------------------------------------------
# The voice (Microsoft neural via edge-tts; Kokoro offline)
# --------------------------------------------------------------------------

_kokoro = None
_kokoro_lock = threading.Lock()


def _engine():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        folder = model_dir()
        _kokoro = Kokoro(str(folder / "kokoro-v1.0.onnx"), str(folder / "voices-v1.0.bin"))
    return _kokoro


def is_edge_voice(voice: str) -> bool:
    """Microsoft neural voices ("en-US-ChristopherNeural") go through edge-tts."""
    return voice.endswith("Neural")


def _edge_samples(text: str, voice: str, speed: float, scratch: Path):
    """Microsoft neural voice via edge-tts (free, online), 3 tries."""
    import edge_tts
    import soundfile as sf

    rate = f"{round((speed - 1.0) * 100):+d}%"
    last: Exception | None = None
    for attempt in range(EDGE_TRIES):
        try:
            asyncio.run(
                asyncio.wait_for(
                    edge_tts.Communicate(text, voice, rate=rate).save(str(scratch)),
                    EDGE_TIMEOUT_S,
                )
            )
            return sf.read(str(scratch), dtype="float32", always_2d=False)
        except Exception as exc:  # NoAudioReceived / network: try again
            last = exc
            log.info("edge-tts try %d failed: %s", attempt + 1, exc)
    raise RuntimeError(f"edge-tts gave no audio ({last})")


def _kokoro_samples(text: str, voice: str, speed: float):
    lang = "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"
    with _kokoro_lock:  # one onnx session, one caller at a time
        return _engine().create(text, voice=voice, speed=speed, lang=lang)


def speak(text: str, out: Path, *, voice: str, speed: float) -> float:
    """Voice ``text`` into ``out`` (wav). Returns its length in seconds.

    A Microsoft neural voice is used when named (falls back to Kokoro's
    ``KOKORO_FALLBACK`` voice offline, if installed); else a Kokoro voice.
    """
    import numpy as np
    import soundfile as sf

    if is_edge_voice(voice):
        try:
            samples, rate = _edge_samples(text, voice, speed, out.with_suffix(".mp3"))
        except Exception:
            if not kokoro_installed():
                raise
            log.warning("Microsoft voice unavailable; using the offline Kokoro voice.")
            samples, rate = _kokoro_samples(text, KOKORO_FALLBACK, 1.1)
    else:
        samples, rate = _kokoro_samples(text, voice, speed)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    loud = np.flatnonzero(np.abs(samples) > 0.01)
    if loud.size:
        a = max(0, loud[0] - int(0.03 * rate))
        b = min(len(samples), loud[-1] + int(0.08 * rate))
        samples = samples[a:b]
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), samples, rate)
    return len(samples) / rate


# --------------------------------------------------------------------------
# Putting it on the clip
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiceClip:
    path: Path
    start_s: float
    length_s: float


def plan_voice(
    duration: float, intro_s: float, outro_s: float, *, lead_s: float = 0.0
) -> tuple[float, float | None]:
    """(intro start, outro start or None) in output seconds."""
    intro_at = max(0.0, lead_s) + INTRO_AT_S
    if outro_s <= 0 or duration < OUTRO_MIN_CLIP_S:
        return intro_at, None
    outro_at = duration - OUTRO_END_PAD_S - outro_s
    if outro_at - (intro_at + intro_s) < MIN_CLEAR_S:
        return intro_at, None
    return intro_at, outro_at


def duck_expression(windows: list[tuple[float, float]], duck_db: float) -> str:
    """volume= expression: 1 outside the windows, ``duck_db`` inside, with ramps."""
    if not windows:
        return "1"
    depth = 1.0 - 10 ** (duck_db / 20)
    terms = []
    for a, b in windows:
        a0, b1 = a - DUCK_ATTACK_S, b + DUCK_RELEASE_S
        terms.append(
            f"clip((t-{a0:.3f})/{DUCK_ATTACK_S},0,1)*clip(({b1:.3f}-t)/{DUCK_RELEASE_S},0,1)"
        )
    return f"1-{depth:.4f}*min(1,{'+'.join(terms)})"


def _audio_graph(voices: list[VoiceClip], duck_db: float, tail: str) -> str:
    windows = [(v.start_s, v.start_s + v.length_s) for v in voices]
    expr = duck_expression(windows, duck_db).replace(",", "\\,")
    parts = [f"[0:a]volume='{expr}':eval=frame[bed]"]
    labels = ["[bed]"]
    for i, v in enumerate(voices, start=2):
        ms = round(v.start_s * 1000)
        parts.append(
            f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"loudnorm=I={VOICE_LUFS}:TP=-2:LRA=7,aresample=48000,"
            f"adelay={ms}:all=1[vo{i}]"
        )
        labels.append(f"[vo{i}]")
    parts.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0,{tail}[aout]"
    )
    return ";".join(parts)


def apply(
    clip: Path,
    *,
    caption_text: str,
    narration: Narration | None,
    anchor: tuple[str, int],
    work_dir: Path,
    voice: str,
    speed: float,
    duck_db: float,
    outro: bool,
    lead_s: float,
    loudness_lufs: float,
    true_peak_db: float,
    export_settings,
) -> dict:
    """Burn the caption box and mix the narrator into ``clip`` (atomic replace).

    Returns what was applied: {"caption": bool, "voiceover": [lines], "edits": [...]}.
    """
    info = ffmpeg.probe(clip)
    duration = float(info.duration_s or 0.0)
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = ["-i", str(clip)]
    edits: list[str] = []
    video = "[0:v]null[vout]"
    if caption_text.strip():
        png = work_dir / "caption_box.png"
        _, box_h = caption_png(caption_text, png)
        top = box_top(anchor, box_h)
        inputs += ["-loop", "1", "-framerate", "30", "-i", str(png)]
        video = f"[0:v][1:v]overlay=x=(W-w)/2:y={top}:shortest=1:format=auto,format=yuv420p[vout]"
        edits.append("caption box")
    else:  # keeps the voice inputs at index 2+
        inputs += ["-f", "lavfi", "-i", "color=c=black:s=16x16:r=30:d=0.1"]

    voices: list[VoiceClip] = []
    spoken: list[str] = []
    if narration is not None and narration.intro:
        intro_wav = work_dir / "vo_intro.wav"
        intro_s = speak(narration.intro, intro_wav, voice=voice, speed=speed)
        outro_s = 0.0
        outro_wav = work_dir / "vo_outro.wav"
        if outro and narration.outro:
            outro_s = speak(narration.outro, outro_wav, voice=voice, speed=speed)
        intro_at, outro_at = plan_voice(duration, intro_s, outro_s, lead_s=lead_s)
        if intro_at + intro_s < duration - 0.5:
            voices.append(VoiceClip(intro_wav, intro_at, intro_s))
            spoken.append(narration.intro)
            if outro_at is not None:
                voices.append(VoiceClip(outro_wav, outro_at, outro_s))
                spoken.append(narration.outro)
        for v in voices:
            inputs += ["-i", str(v.path)]
        if voices:
            edits.append("voiceover")

    if not edits:
        return {"caption": False, "voiceover": [], "edits": []}

    if voices:
        measured = shared_edit.measure_loudnorm(
            inputs,
            _audio_graph(
                voices,
                duck_db,
                shared_edit.measure_filter(target=loudness_lufs, true_peak=true_peak_db),
            ),
        )
        tail = (
            shared_edit.loudnorm_filter(measured, target=loudness_lufs, true_peak=true_peak_db)
            + ",aresample=48000"
        )
        audio_graph = _audio_graph(voices, duck_db, tail)
        audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    else:
        audio_graph = "[0:a]anull[aout]"
        audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]

    partial = clip.with_name(clip.stem + ".narrate.mp4")

    def args(video_args: list[str]) -> list[str]:
        return [
            *inputs,
            "-filter_complex",
            f"{video};{audio_graph}",
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-t",
            f"{duration:.3f}",
            *video_args,
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            *audio_args,
            "-movflags",
            "+faststart",
            str(partial),
        ]

    try:
        video_args = encoder_args(export_settings)
        try:
            ffmpeg.run(args(video_args), total_duration_s=duration)
        except ffmpeg.FFmpegError:
            if "h264_nvenc" not in video_args:
                raise
            cpu = export_settings.model_copy(update={"prefer_hardware_encoder": False})
            ffmpeg.run(args(encoder_args(cpu)), total_duration_s=duration)
        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError("ffmpeg finished but the narrated clip is empty.")
        os.replace(partial, clip)
    finally:
        partial.unlink(missing_ok=True)
    return {"caption": "caption box" in edits, "voiceover": spoken, "edits": edits}


def transcript_text(words) -> str:
    return " ".join(getattr(w, "text", "") for w in words or []).strip()


def lines_for(settings, *, hook: str, context: dict | None, words, complete=None):
    """(Narration, source) when the voiceover is on, else (None, ""). Never raises."""
    if not settings.voiceover or context is None:
        return None, ""
    if not is_edge_voice(settings.voiceover_voice) and not kokoro_installed():
        log.warning("No narrator: the Kokoro voice model isn't in %s.", model_dir())
        return None, ""
    try:
        return write_lines(
            streamer=str(context.get("streamer") or ""),
            hook=hook,
            caption=str(context.get("caption") or ""),
            summary=str(context.get("summary") or ""),
            transcript=transcript_text(words),
            complete=complete,
            seed=str(context.get("seed") or ""),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Narration lines failed: %s", exc)
        return None, ""


def finish(
    clip: Path,
    *,
    settings,
    export_settings,
    anchor: tuple[str, int],
    lead_s: float,
    work_dir: Path,
    hook: str,
    narration: Narration | None,
    source: str = "",
) -> dict:
    """Caption box + narrator per the GTA settings. Never raises."""
    if not settings.static_caption and narration is None:
        return {}
    try:
        done = apply(
            clip,
            caption_text=hook if settings.static_caption else "",
            narration=narration,
            anchor=anchor,
            work_dir=work_dir,
            voice=settings.voiceover_voice,
            speed=settings.voiceover_speed,
            duck_db=settings.voiceover_duck_db,
            outro=settings.voiceover_outro,
            lead_s=lead_s,
            loudness_lufs=settings.loudness_lufs,
            true_peak_db=settings.true_peak_db,
            export_settings=export_settings,
        )
        done["voiceover_source"] = source
        return done
    except (Exception, subprocess.SubprocessError) as exc:
        log.warning("Caption box / voiceover failed for %s: %s", clip.name, exc)
        return {"error": str(exc)[:300], "edits": []}
