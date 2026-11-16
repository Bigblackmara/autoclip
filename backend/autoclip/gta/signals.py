"""Raw signals for the Hype Curve: audio loudness, chat replay, viewer clips.

Each reader returns plain data (per-second arrays, timestamped messages) and
never raises for a missing or malformed source — a stream without chat replay
is still clippable from audio and transcript alone; the curve just records
which signals it had.
"""

from __future__ import annotations

import json
import logging
import math
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: dBFS reported for digital silence (log of zero).
SILENCE_DB = -90.0


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def wav_loudness_per_second(path: Path) -> np.ndarray:
    """RMS loudness of each whole second of a mono 16-bit WAV, in dBFS."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        if width != 2:
            raise ValueError(f"{path.name}: expected 16-bit PCM, got {8 * width}-bit")
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples[: len(samples) - len(samples) % channels].reshape(-1, channels)
        samples = samples.mean(axis=1)
    seconds = math.ceil(len(samples) / rate) if rate else 0
    if seconds == 0:
        return np.zeros(0, dtype=np.float32)
    padded = np.zeros(seconds * rate, dtype=np.float32)
    padded[: len(samples)] = samples.astype(np.float32) / 32768.0
    blocks = padded.reshape(seconds, rate)
    rms = np.sqrt(np.mean(blocks * blocks, axis=1))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms)
    return np.maximum(db, SILENCE_DB).astype(np.float32)


def loudness_for_parts(parts: list[Path], part_s: float) -> np.ndarray:
    """Per-second loudness over all analysis parts, in stream time.

    Each part is placed at ``index * part_s`` so a part ffmpeg cut a second
    short can't shift every later second.
    """
    chunks: list[tuple[int, np.ndarray]] = []
    for index, part in enumerate(parts):
        chunks.append((round(index * part_s), wav_loudness_per_second(part)))
    total = max((offset + len(values) for offset, values in chunks), default=0)
    out = np.full(total, SILENCE_DB, dtype=np.float32)
    for offset, values in chunks:
        out[offset : offset + len(values)] = values
    return out


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    t: float
    text: str


def _runs_text(message: dict) -> str:
    parts: list[str] = []
    for run in message.get("runs") or []:
        if not isinstance(run, dict):
            continue
        if "text" in run:
            parts.append(str(run["text"]))
        elif isinstance(run.get("emoji"), dict):
            emoji = run["emoji"]
            shortcuts = emoji.get("shortcuts") or []
            parts.append(str(shortcuts[0]) if shortcuts else str(emoji.get("emojiId", "")))
    return "".join(parts)


def parse_youtube_chat_line(line: str) -> list[ChatMessage]:
    """One JSON line of yt-dlp's ``live_chat.json`` into messages."""
    line = line.strip()
    if not line:
        return []
    try:
        data = json.loads(line)
    except ValueError:
        return []
    replay = data.get("replayChatItemAction") if isinstance(data, dict) else None
    if not isinstance(replay, dict):
        return []
    try:
        t = int(replay.get("videoOffsetTimeMsec", "")) / 1000.0
    except (TypeError, ValueError):
        return []
    out: list[ChatMessage] = []
    for action in replay.get("actions") or []:
        item = ((action or {}).get("addChatItemAction") or {}).get("item") or {}
        renderer = item.get("liveChatTextMessageRenderer") or item.get(
            "liveChatPaidMessageRenderer"
        )
        if not isinstance(renderer, dict):
            continue
        text = _runs_text(renderer.get("message") or {})
        if text:
            out.append(ChatMessage(t=t, text=text))
    return out


def read_youtube_chat(path: Path) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                messages.extend(parse_youtube_chat_line(line))
    except OSError:
        log.warning("Could not read chat replay %s.", path, exc_info=True)
        return []
    messages.sort(key=lambda m: m.t)
    return messages


def save_messages(path: Path, messages: Iterable[ChatMessage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps([round(message.t, 2), message.text]) + "\n")
    tmp.replace(path)


def load_messages(path: Path) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                t, text = json.loads(line)
                out.append(ChatMessage(t=float(t), text=str(text)))
            except (ValueError, TypeError):
                continue
    return out


#: Words that mean "chat is losing it". Lower-cased, matched per word.
HYPE_WORDS = frozenset(
    {
        "lmao",
        "lmfao",
        "lol",
        "lool",
        "loool",
        "kekw",
        "lul",
        "lulw",
        "omegalul",
        "icant",
        "omg",
        "wtf",
        "bruh",
        "💀",
        "😭",
        "😂",
        "🤣",
        ":skull:",
        ":sob:",
        ":joy:",
    }
)


def keyword_hits(text: str) -> tuple[int, int, int]:
    """(clip mentions, laugh/hype words, question-mark walls) in one message."""
    lowered = text.lower()
    words = [w.strip(".,!:;\"'()[]") for w in lowered.split()]
    clip = sum(1 for w in words if w in {"clip", "clipit", "clipped", "clipthat", "clip!"})
    if "clip it" in lowered or "clip that" in lowered:
        clip = max(clip, 1)
    hype = sum(1 for w in words if w in HYPE_WORDS)
    hype += sum(lowered.count(e) for e in ("💀", "😭", "😂", "🤣") if e not in words)
    stripped = lowered.replace(" ", "")
    wall = (
        1 if len(stripped) >= 3 and stripped.count("?") >= max(3, int(0.6 * len(stripped))) else 0
    )
    return clip, hype, wall


#: Chat reacts after the moment: messages are moved this much earlier so a
#: chat spike lines up with what caused it.
CHAT_LAG_S = 6.0


@dataclass
class ChatSeries:
    rate: np.ndarray
    clip: np.ndarray
    hype: np.ndarray
    walls: np.ndarray


def chat_series(
    messages: list[ChatMessage], seconds: int, *, lag_s: float = CHAT_LAG_S
) -> ChatSeries:
    """Per-second message counts and keyword counts, lag-corrected."""
    rate = np.zeros(seconds, dtype=np.float32)
    clip = np.zeros(seconds, dtype=np.float32)
    hype = np.zeros(seconds, dtype=np.float32)
    walls = np.zeros(seconds, dtype=np.float32)
    for message in messages:
        index = int(message.t - lag_s)
        if not 0 <= index < seconds:
            continue
        rate[index] += 1
        c, h, w = keyword_hits(message.text)
        clip[index] += c
        hype[index] += h
        walls[index] += w
    return ChatSeries(rate=rate, clip=clip, hype=hype, walls=walls)


# --------------------------------------------------------------------------
# Viewer clips (Twitch Helix, official API, optional credentials)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewerClip:
    offset_s: float
    duration_s: float
    views: int
    title: str = ""


HttpGet = Callable[[str, dict[str, str], dict[str, str]], dict]


def _http_get(url: str, params: dict[str, str], headers: dict[str, str]) -> dict:
    import httpx

    response = httpx.get(url, params=params, headers=headers, timeout=20.0)
    response.raise_for_status()
    return response.json()


def _http_post(url: str, params: dict[str, str]) -> dict:
    import httpx

    response = httpx.post(url, params=params, timeout=20.0)
    response.raise_for_status()
    return response.json()


TWITCH_CLIENT_ID_KEY = "gta_twitch_client_id"
TWITCH_CLIENT_SECRET_KEY = "gta_twitch_client_secret"


def twitch_credentials() -> tuple[str, str] | None:
    from ..config import get_secret

    client_id = get_secret(TWITCH_CLIENT_ID_KEY)
    secret = get_secret(TWITCH_CLIENT_SECRET_KEY)
    if client_id and secret:
        return client_id, secret
    return None


def twitch_viewer_clips(
    vod_id: str,
    channel_login: str,
    *,
    credentials: tuple[str, str],
    http_get: HttpGet = _http_get,
    http_post: Callable[[str, dict[str, str]], dict] = _http_post,
    max_pages: int = 20,
) -> list[ViewerClip]:
    """Viewer-made clips of one VOD, through the official Helix API.

    Helix lists clips per broadcaster; those whose ``video_id`` is this VOD
    carry ``vod_offset`` (the second in the VOD where the clip starts).
    """
    client_id, secret = credentials
    token = http_post(
        "https://id.twitch.tv/oauth2/token",
        {"client_id": client_id, "client_secret": secret, "grant_type": "client_credentials"},
    )["access_token"]
    headers = {"Client-Id": client_id, "Authorization": f"Bearer {token}"}

    users = http_get("https://api.twitch.tv/helix/users", {"login": channel_login}, headers)
    data = users.get("data") or []
    if not data:
        return []
    broadcaster = str(data[0]["id"])

    video = http_get("https://api.twitch.tv/helix/videos", {"id": vod_id}, headers)
    videos = video.get("data") or []
    params = {"broadcaster_id": broadcaster, "first": "100"}
    if videos and videos[0].get("created_at"):
        params["started_at"] = str(videos[0]["created_at"])

    out: list[ViewerClip] = []
    cursor = ""
    for _ in range(max_pages):
        page_params = dict(params)
        if cursor:
            page_params["after"] = cursor
        page = http_get("https://api.twitch.tv/helix/clips", page_params, headers)
        for item in page.get("data") or []:
            if str(item.get("video_id") or "") != vod_id or item.get("vod_offset") is None:
                continue
            out.append(
                ViewerClip(
                    offset_s=float(item["vod_offset"]),
                    duration_s=float(item.get("duration") or 30.0),
                    views=int(item.get("view_count") or 0),
                    title=str(item.get("title") or ""),
                )
            )
        cursor = str((page.get("pagination") or {}).get("cursor") or "")
        if not cursor:
            break
    out.sort(key=lambda c: c.offset_s)
    return out


def clips_series(clips: list[ViewerClip], seconds: int) -> np.ndarray:
    """How many viewer clips cover each second (view-weighted, log-damped)."""
    out = np.zeros(seconds, dtype=np.float32)
    for clip in clips:
        start = max(0, int(clip.offset_s))
        end = min(seconds, int(math.ceil(clip.offset_s + clip.duration_s)))
        if end > start:
            out[start:end] += 1.0 + math.log10(1 + max(0, clip.views)) / 3.0
    return out


def heatmap_series(heatmap: list[tuple[float, float, float]], seconds: int) -> np.ndarray:
    out = np.zeros(seconds, dtype=np.float32)
    for start, end, value in heatmap:
        a = max(0, int(start))
        b = min(seconds, int(math.ceil(end)))
        if b > a:
            out[a:b] = np.maximum(out[a:b], value)
    return out
