"""AI usage per endpoint key: calls, tokens, audio seconds, rate-limit headroom.

Groq's free tier caps every key (requests/day, tokens/minute, audio
seconds/hour and /day) and reports what's left in ``x-ratelimit-*`` response
headers. Every call is recorded here; a key that hits a limit (a 429, or
headers saying it's nearly out) cools down until its reset, and the rotation
skips it meanwhile. Counters persist in ``<AUTOCLIP_HOME>/ai_usage.json`` and
"today" rolls over at UTC midnight.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)

_lock = threading.Lock()
_state: dict[str, Any] | None = None
_last_save = 0.0
#: Below this many tokens/minute left, a key rests until the minute resets.
MIN_TOKENS_LEFT = 1500
#: Unknown 429 reset: rest this long.
DEFAULT_COOLDOWN_S = 60.0
#: An endpoint saying "too many concurrent requests" / "in cooldown" rests this long.
BUSY_REST_S = 20.0
_COUNTERS = ("calls", "errors", "rate_limited", "tokens_in", "tokens_out", "audio_s")


def _path():
    from ..paths import root

    return root() / "ai_usage.json"


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _blank() -> dict[str, Any]:
    return dict.fromkeys(_COUNTERS, 0)


def _load() -> dict[str, Any]:
    global _state
    if _state is None:
        try:
            _state = json.loads(_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _state = {}
        _state.setdefault("keys", {})
    if _state.get("day") != _today():
        _state["day"] = _today()
        for entry in _state["keys"].values():
            entry["today"] = _blank()
    return _state


def _save(force: bool = False) -> None:
    global _last_save
    if not force and time.monotonic() - _last_save < 3.0:
        return
    _last_save = time.monotonic()
    try:
        path = _path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_state, indent=1), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        log.debug("could not save AI usage", exc_info=True)


def key_label(url: str, key: str, model: str = "") -> str:
    """'api.groq.com gpt-oss-120b …h3Q' — never the whole key.

    Groq limits each model separately (and transcription separately from
    chat), so each (key, model) pair is tracked and rests on its own.
    """
    parsed = urlparse(url or "")
    host = parsed.hostname or "default"
    what = " (whisper)" if "/audio/" in parsed.path else ""
    short = f" {model.rsplit('/', 1)[-1]}" if model else ""
    return f"{host}{what}{short} …{(key or '')[-4:]}"


def parse_duration(text: str) -> float | None:
    """Groq's reset values: '2m59.56s', '7.66s', '250ms', '1h2m'. None if unreadable."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    total = 0.0
    found = False
    for value, unit in re.findall(r"([\d.]+)(ms|h|m|s)", text):
        found = True
        total += float(value) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def _headers_dict(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    try:
        items = headers.items()
    except AttributeError:
        return {}
    return {
        str(k).lower(): str(v)
        for k, v in items
        if str(k).lower().startswith("x-ratelimit") or str(k).lower() == "retry-after"
    }


def record(
    url: str,
    key: str,
    *,
    kind: str = "chat",
    ok: bool = True,
    status: int = 200,
    tokens_in: int = 0,
    tokens_out: int = 0,
    audio_s: float = 0.0,
    headers: Any = None,
    error: str = "",
    model: str = "",
) -> None:
    """Count one call and update the key's headroom / cooldown."""
    label = key_label(url, key, model)
    limits = _headers_dict(headers)
    now = time.time()
    with _lock:
        state = _load()
        entry = state["keys"].setdefault(
            label, {"url": url, "today": _blank(), "total": _blank(), "limits": {}}
        )
        for bucket in (entry.setdefault("today", _blank()), entry.setdefault("total", _blank())):
            bucket["calls"] += 1
            bucket["errors"] += 0 if ok else 1
            bucket["rate_limited"] += 1 if status == 429 else 0
            bucket["tokens_in"] += int(tokens_in or 0)
            bucket["tokens_out"] += int(tokens_out or 0)
            bucket["audio_s"] = round(bucket["audio_s"] + float(audio_s or 0.0), 1)
        if limits:
            entry.setdefault("limits", {})[kind] = {**limits, "at": now}
        entry["last_at"] = now
        if not ok:
            entry["last_error"] = {"at": now, "kind": kind, "text": error[:300]}
        rest = _rest_seconds(status, limits)
        lowered = error.lower()
        stated = _stated_wait(lowered)
        if not ok and stated:
            # "try again in 12m23s" / "resets at 00:00 UTC": rest that long, not a minute.
            rest = max(rest, stated)
        if (
            not rest
            and status in (502, 503)
            and ("concurrent" in lowered or "cooldown" in lowered or "429" in lowered)
        ):
            # utim's upstream: "max 2 concurrent" / "in cooldown" — rest briefly.
            rest = BUSY_REST_S
        if not rest and not ok and ("timed out" in lowered or "timeout" in lowered):
            # The endpoint is likely still working on it and holding a slot.
            rest = BUSY_REST_S * 1.5
        if rest:
            until = now + rest
            if until > float(entry.get("cooldown_until") or 0):
                entry["cooldown_until"] = until
                entry["cooldown_kind"] = kind
                log.info("AI key %s resting %.0f s (%s)", label, rest, kind)
        _save(force=not ok)


def _rest_seconds(status: int, limits: dict[str, str]) -> float:
    """How long a key should rest after this answer (0 = fine)."""
    if status == 429:
        for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            value = parse_duration(limits.get(name, ""))
            if value:
                return min(value, 24 * 3600)
        return DEFAULT_COOLDOWN_S
    try:
        if int(limits.get("x-ratelimit-remaining-requests", "1")) <= 0:
            return parse_duration(limits.get("x-ratelimit-reset-requests", "")) or 300.0
        if int(limits.get("x-ratelimit-remaining-tokens", "999999")) < MIN_TOKENS_LEFT:
            return parse_duration(limits.get("x-ratelimit-reset-tokens", "")) or 30.0
    except ValueError:
        pass
    return 0.0


def _stated_wait(lowered: str) -> float:
    """Seconds until the reset an error message states (0 = none stated)."""
    m = re.search(r"try again in ((?:[\d.]+(?:ms|h|m|s))+)", lowered)
    if m:
        return parse_duration(m.group(1)) or 0.0
    if "00:00 utc" in lowered and ("daily" in lowered or "per day" in lowered):
        now = datetime.now(UTC)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() + 86400
        return max(60.0, midnight - now.timestamp() + 30)
    return 0.0


def resting(url: str, key: str, model: str = "") -> float:
    """Seconds this key (and model) still has to rest (0 = usable)."""
    with _lock:
        entry = _load()["keys"].get(key_label(url, key, model))
        if not entry:
            return 0.0
        return max(0.0, float(entry.get("cooldown_until") or 0) - time.time())


def snapshot() -> dict[str, Any]:
    """Everything the usage card shows."""
    with _lock:
        state = _load()
        now = time.time()
        keys = []
        for label, entry in sorted(state["keys"].items()):
            keys.append(
                {
                    "label": label,
                    "host": urlparse(entry.get("url") or "").hostname or "",
                    "today": entry.get("today") or _blank(),
                    "total": entry.get("total") or _blank(),
                    "limits": entry.get("limits") or {},
                    "resting_s": max(0.0, float(entry.get("cooldown_until") or 0) - now),
                    "resting_kind": entry.get("cooldown_kind") or "",
                    "last_at": entry.get("last_at"),
                    "last_error": entry.get("last_error"),
                }
            )
        return {"day": state.get("day"), "keys": keys, "now": now}


def reset_for_tests() -> None:
    global _state
    with _lock:
        _state = None
