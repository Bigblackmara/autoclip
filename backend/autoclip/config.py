"""Settings persistence and secret storage.

Non-secret settings live in ``<root>/config.json`` as plain JSON so users can
hand-edit them. Secrets (LLM API keys, the HuggingFace token) are kept out of
that file and stored in the OS keyring — Credential Manager on Windows, Keychain
on macOS, Secret Service on Linux.

If no keyring backend is available (headless Linux, some Docker images), we fall
back to writing secrets into ``config.json`` with restrictive permissions and
set :attr:`Settings.insecure_secret_storage` so the UI and ``autoclip doctor``
can warn about it. Silently downgrading security without telling the user is the
thing we're trying to avoid.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from . import paths

log = logging.getLogger(__name__)

KEYRING_SERVICE = "autoclip"

#: Keyring service used before the rename. Read from as a fallback so keys
#: stored under the old name aren't silently lost; writes always use the new one.
LEGACY_KEYRING_SERVICE = "clipforge"

ProviderName = Literal["anthropic", "openai", "gemini", "ollama"]

#: Providers that authenticate with an API key. Ollama runs locally and needs none.
KEYED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "gemini")

#: Extra secrets that aren't tied to a provider.
HF_TOKEN_KEY = "huggingface_token"


class ProviderSettings(BaseModel):
    """Per-provider configuration. The API key itself is never stored here."""

    model: str = ""
    #: Only meaningful for the OpenAI-compatible provider. Pointing this at
    #: OpenRouter, Groq, DeepSeek, or a local LM Studio server is the supported
    #: way to use any other vendor.
    base_url: str | None = None


class WhisperSettings(BaseModel):
    #: tiny | base | small | medium | large-v3 — or a local path.
    model: str = "small"
    #: Force a CTranslate2 compute type. Empty means "let hardware detection pick".
    compute_type: str = ""
    #: ISO 639-1 code, or empty for auto-detection.
    language: str = ""
    #: Run WhisperX diarization to label words by speaker. Needs the
    #: ``diarization`` extra and a HuggingFace token.
    diarization: bool = False


class ClipSettings(BaseModel):
    min_duration_s: float = 20.0
    max_duration_s: float = 90.0
    max_clips: int = 10


class IngestSettings(BaseModel):
    #: yt-dlp format selector. Default caps at 1080p to keep downloads sane.
    ytdlp_format: str = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"
    #: Browser to pull YouTube cookies from ("chrome", "firefox", "edge", ...).
    #: As of 2026 most anonymous YouTube downloads hit a bot check that only
    #: browser cookies reliably clear, so this is a first-class setting.
    cookies_from_browser: str = ""
    #: Use YouTube's own captions (manual, else automatic, in the video's own
    #: language) as the transcript and skip Whisper. Off by default: automatic
    #: captions carry no punctuation, so clip edges lean on pauses alone and
    #: burned-in captions read as one lower-case run. Whisper is used whenever
    #: the captions are missing or look wrong.
    prefer_youtube_captions: bool = False
    #: Keep yt-dlp current: check PyPI at startup and daily, upgrade when a
    #: newer release is out, and once more when a YouTube download fails with
    #: an error that usually means yt-dlp is outdated (see ytdlp_update.py).
    ytdlp_auto_update: bool = True
    #: Largest file a Google Drive / Dropbox / OneDrive / direct link may be.
    link_max_mb: int = Field(default=20_000, ge=1, le=1_000_000)


class HousekeepingSettings(BaseModel):
    """Background clean-up of old work (see :mod:`autoclip.retention`)."""

    #: Delete sources, jobs and clips (rows and files) this many days after
    #: they were made. 0 keeps everything. Work still processing, and GTA
    #: clips with a pending or scheduled upload, are never deleted.
    retention_days: int = Field(default=0, ge=0, le=3650)


class DetectionSettings(BaseModel):
    """Audience signals and the judge pass around highlight detection."""

    #: Build the per-second audience curve: YouTube "Most replayed", comment
    #: timestamps, chapters, with audio energy as the fallback.
    audience_signals: bool = True
    #: Read timestamps viewers wrote in the top comments (one extra yt-dlp
    #: request per YouTube source; failures are ignored).
    fetch_comments: bool = True
    #: How many top comments to read.
    max_comments: int = Field(default=300, ge=0, le=5000)
    #: Use per-second loudness as a signal (the only one for uploads).
    audio_energy: bool = True
    #: Pass each window's audience peaks to the model as hints.
    audience_hints: bool = True
    #: Peaks listed per transcript window.
    hints_per_window: int = Field(default=4, ge=0, le=20)
    #: Share of the final score taken from audience data (heatmap/comments).
    audience_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    #: Share taken from the curve when it is audio energy alone (a weak signal).
    energy_weight: float = Field(default=0.12, ge=0.0, le=1.0)
    #: Second LLM pass with three yes/no tests (hook, context-free, payoff)
    #: that drops weak candidates. A failed judge keeps every candidate.
    judge: bool = True
    #: Candidates sent to the judge (at least twice ``clips.max_clips``).
    judge_max_candidates: int = Field(default=20, ge=1, le=200)
    #: Never let the judge leave fewer clips than this.
    judge_min_keep: int = Field(default=1, ge=0, le=100)


class ExportSettings(BaseModel):
    ratio: Literal["9:16", "1:1", "16:9"] = "9:16"
    caption_style: str = "bold_pop"
    #: Integrated loudness target in LUFS. -14 is the de-facto platform standard.
    loudness_lufs: float = -14.0
    #: Use h264_nvenc when the hardware supports it. Falls back to libx264.
    prefer_hardware_encoder: bool = True
    crf: int = 18
    #: Also write a .srt sidecar next to each exported clip.
    write_srt: bool = False
    #: Frame layout for 9:16 clips. "auto" stacks a detected facecam overlay
    #: above the gameplay (streamer clips) and otherwise crops as usual;
    #: "split" asks for the stacked layout whenever a facecam is found; "crop"
    #: always uses the speaker-following crop. Other ratios always crop.
    layout: Literal["auto", "split", "crop"] = "auto"
    #: How each shot is framed (normal mode; a facecam stack from ``layout``
    #: takes precedence). "auto" picks per shot: two people too far apart for
    #: one crop get a top/bottom split, shots with no faces (screen, slides,
    #: gameplay) are fitted over a blurred copy, the rest follow the speaker.
    #: "fill" always crops, "split" splits any two-person shot, "fit" fits
    #: every shot. See ``pipeline.reframe.FRAMINGS``.
    framing: Literal["auto", "fill", "split", "fit"] = "auto"

    # -- short-form editing (normal mode; see pipeline/edit.py) -------------
    #: Cut pauses: "natural" shortens pauses over ~0.6 s, "fast" nearly all,
    #: "off" keeps everything. Cuts land only on measured silences.
    pace: Literal["off", "natural", "fast"] = "natural"
    #: The clip's title as a headline over the first ~3 seconds.
    hook_title: bool = True
    #: Punch-in zooms at natural pauses: "subtle" every ~4-6 s, "dynamic" more
    #: often plus a quick zoom-out on the hook. Jump cuts always get a zoom step.
    motion: Literal["off", "subtle", "dynamic"] = "subtle"
    #: Draw 3-8 important words per clip bigger and in an accent colour.
    keyword_highlight: bool = True
    #: Background music: a track file, or a folder to pick one from per clip.
    music_enabled: bool = False
    music_path: str = ""
    #: Music level in percent (100 = the track's own level), before ducking.
    music_volume: int = Field(default=30, ge=0, le=200)
    #: Push the music down while someone is talking.
    music_duck: bool = True
    #: Watermark: "text" draws ``watermark_text`` as a logo, "image" overlays a PNG.
    watermark: Literal["off", "text", "image"] = "off"
    watermark_text: str = ""
    watermark_image: str = ""
    watermark_corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "top_right"
    watermark_opacity: float = Field(default=0.85, ge=0.05, le=1.0)
    #: Output frame rate, forced on every normal-mode export.
    fps: int = Field(default=30, ge=15, le=60)
    #: Measure loudness first and normalise linearly (lands on the target
    #: rather than near it). Off falls back to one-pass dynamic loudnorm.
    two_pass_loudnorm: bool = True
    #: Flash-forward opening: play the clip's punchiest 1-3 s line first
    #: (punched in), then the clip from the start. A clip's own toggle in
    #: the review page overrides this default.
    flash_forward: bool = False
    #: Language of burned captions, the hook headline and the post text.
    #: "as_spoken" keeps each language's own script except Hindi, which is
    #: written in Latin letters (Hinglish); "roman" writes every language in
    #: Latin letters; "english" translates (an LLM pass, timings kept).
    caption_language: Literal["as_spoken", "roman", "english"] = "as_spoken"


class Settings(BaseModel):
    """Top-level persisted settings."""

    active_provider: ProviderName = "anthropic"
    providers: dict[str, ProviderSettings] = Field(
        default_factory=lambda: {
            "anthropic": ProviderSettings(model="claude-sonnet-5"),
            "openai": ProviderSettings(model="gpt-4o"),
            "gemini": ProviderSettings(model="gemini-2.0-flash"),
            "ollama": ProviderSettings(model="", base_url="http://localhost:11434"),
        }
    )
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    clips: ClipSettings = Field(default_factory=ClipSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    detection: DetectionSettings = Field(default_factory=DetectionSettings)
    housekeeping: HousekeepingSettings = Field(default_factory=HousekeepingSettings)

    #: Set when secrets had to be written to config.json because no keyring
    #: backend was usable. Surfaced as a warning in the UI and in `doctor`.
    insecure_secret_storage: bool = False
    #: Plaintext secret fallback. Empty whenever the keyring is working.
    #: A private attribute so it never round-trips through the public schema —
    #: `save()` writes it back explicitly.
    _fallback_secrets: dict[str, str] = PrivateAttr(default_factory=dict)

    model_config = {"extra": "ignore"}

    def provider(self, name: str | None = None) -> ProviderSettings:
        """Return settings for ``name``, defaulting to the active provider."""
        key = name or self.active_provider
        return self.providers.setdefault(key, ProviderSettings())


def load() -> Settings:
    """Read settings from disk, falling back to defaults on a missing or bad file."""
    path = paths.config_path()
    if not path.exists():
        return Settings()
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s (%s); falling back to defaults.", path, exc)
        return Settings()

    fallback = raw.pop("_fallback_secrets", {}) or {}
    settings = Settings.model_validate(raw)
    settings._fallback_secrets = fallback
    return settings


def _foreign_sections(path: Path) -> dict[str, Any]:
    """Top-level keys of the existing file that :class:`Settings` does not own.

    config.json is shared: feature sections such as Live Watch's ``live``
    live beside the core settings without being part of this model
    (``extra="ignore"`` drops them on load). A save must carry them through
    untouched, or every Settings save would silently wipe them.
    ``_fallback_secrets`` is excluded on purpose: it is owned by the
    Settings object being saved, so a deleted fallback secret must not be
    resurrected from the old file.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    owned = set(Settings.model_fields) | {"_fallback_secrets"}
    return {key: value for key, value in raw.items() if key not in owned}


def save(settings: Settings) -> None:
    """Write settings to disk atomically, with owner-only permissions.

    Unknown top-level sections already in the file (e.g. ``live``) are
    preserved verbatim.
    """
    paths.ensure_layout()
    path = paths.config_path()

    payload = _foreign_sections(path)
    payload.update(settings.model_dump(mode="json"))
    if settings._fallback_secrets:
        payload["_fallback_secrets"] = settings._fallback_secrets

    # Write to a sibling temp file then replace, so an interrupted write can
    # never leave a truncated config behind.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _restrict_permissions(tmp)
    os.replace(tmp, path)


def _restrict_permissions(path: Path) -> None:
    """Best-effort chmod 600. A no-op where the platform doesn't support it."""
    with contextlib.suppress(OSError):  # platform dependent
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------


def _keyring():
    """Return a usable keyring module, or None if no backend works.

    Importing ``keyring`` succeeds even with no backend; the failure only shows
    up on first use, so we probe explicitly.
    """
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring
    except ImportError:  # pragma: no cover - keyring is a hard dependency
        return None

    try:
        if isinstance(keyring.get_keyring(), FailKeyring):
            return None
    except Exception:  # pragma: no cover - backend probing can raise anything
        return None
    return keyring


def get_secret(key: str, settings: Settings | None = None) -> str | None:
    """Return a stored secret, or None if unset.

    ``key`` is a provider name for API keys, or :data:`HF_TOKEN_KEY`.
    """
    env_override = os.environ.get(f"AUTOCLIP_{key.upper()}_KEY")
    if env_override:
        return env_override

    kr = _keyring()
    if kr is not None:
        for service in (KEYRING_SERVICE, LEGACY_KEYRING_SERVICE):
            try:
                value = kr.get_password(service, key)
                if value:
                    return value
            except Exception as exc:  # pragma: no cover - backend dependent
                log.warning("Keyring read failed for %s: %s", key, exc)

    settings = settings if settings is not None else load()
    return settings._fallback_secrets.get(key) or None


def set_secret(key: str, value: str, settings: Settings | None = None) -> bool:
    """Store a secret. Returns True if it went to the keyring, False if to disk.

    A False return is not an error — it means the caller should surface the
    insecure-storage warning to the user.
    """
    kr = _keyring()
    if kr is not None:
        try:
            kr.set_password(KEYRING_SERVICE, key, value)
            return True
        except Exception as exc:  # pragma: no cover - backend dependent
            log.warning("Keyring write failed for %s: %s", key, exc)

    settings = settings if settings is not None else load()
    settings._fallback_secrets[key] = value
    settings.insecure_secret_storage = True
    save(settings)
    return False


def delete_secret(key: str, settings: Settings | None = None) -> None:
    """Remove a secret from both the keyring and the plaintext fallback."""
    kr = _keyring()
    if kr is not None:
        # Backends raise when the entry is absent; deleting a missing secret is
        # not an error from the caller's point of view. Both services are cleared
        # so a delete can't leave a pre-rename copy behind to resurface.
        for service in (KEYRING_SERVICE, LEGACY_KEYRING_SERVICE):
            with contextlib.suppress(Exception):
                kr.delete_password(service, key)

    settings = settings if settings is not None else load()
    if settings._fallback_secrets.pop(key, None) is not None:
        settings.insecure_secret_storage = bool(settings._fallback_secrets)
        save(settings)
