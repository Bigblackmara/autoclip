"""OpenAI-compatible provider.

Deliberately the widest adapter in AutoClip. Because ``base_url`` is
configurable and the Chat Completions shape is a de-facto standard, this one
class also covers OpenRouter, Groq, DeepSeek, Together, Fireworks, vLLM, and a
local LM Studio server. "Bring any API key" is mostly this file.
"""

from __future__ import annotations

import itertools
import logging
import os
import threading

from .base import DetectionConfig, LLMProvider, ProviderError, ProviderStatus

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"

SUGGESTED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
]

#: Ready-made endpoints for the settings UI. Users can enter any other URL.
KNOWN_ENDPOINTS = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Groq": "https://api.groq.com/openai/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Together": "https://api.together.xyz/v1",
    "LM Studio (local)": "http://localhost:1234/v1",
}


#: Fallback when the configured endpoint fails: Groq, rotating through every key
#: stored (newline-separated) under the ``groq_fallback`` secret.
GROQ_URL = KNOWN_ENDPOINTS["Groq"]
GROQ_FALLBACK_SECRET = "groq_fallback"
GROQ_FALLBACK_MODEL = os.environ.get("AUTOCLIP_GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")
_groq_turn = itertools.count()
_groq_lock = threading.Lock()


def groq_fallback_keys() -> list[str]:
    try:
        from ..config import get_secret

        raw = get_secret(GROQ_FALLBACK_SECRET) or ""
    except Exception:  # pragma: no cover - keyring backend dependent
        return []
    return [k.strip() for k in raw.replace(",", " ").split() if k.strip()]


def groq_first() -> bool:
    """True when Groq is tried before the configured endpoint.

    Set by ``<AUTOCLIP_HOME>/ai_order.json`` ({"groq_first": true}); off when
    the file is missing.
    """
    import json

    from ..paths import root

    try:
        data = json.loads((root() / "ai_order.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(isinstance(data, dict) and data.get("groq_first"))


#: The primary AI endpoint (numes.ai, key stored under ``numes_key``). It takes
#: every call first; the Groq keys only step in when it fails. Its free model
#: answers only when streamed and takes ~60-90 s per call.
UTIM_URL = os.environ.get("AUTOCLIP_PRIMARY_URL", "https://api.numes.ai/api/v1")
UTIM_MODEL = os.environ.get("AUTOCLIP_PRIMARY_MODEL", "deepseek/deepseek-v4.1-flash:free")
UTIM_SECRET = "numes_key"
#: Calls in flight at once on the primary endpoint (2 measured to work).
UTIM_MAX_CONCURRENT = 2
UTIM_TIMEOUT_S = 150.0
_utim_slots = threading.BoundedSemaphore(UTIM_MAX_CONCURRENT)


def utim_key() -> str:
    try:
        from ..config import get_secret

        return (get_secret(UTIM_SECRET) or "").strip()
    except Exception:  # pragma: no cover - keyring backend dependent
        return ""


#: Groq models in order of preference. Each has its OWN daily token quota per
#: key (gpt-oss-120b: 200k tokens/day on the free tier), so falling through
#: them multiplies how long the free keys last.
GROQ_MODELS = [
    m.strip()
    for m in os.environ.get(
        "AUTOCLIP_GROQ_MODELS", f"{GROQ_FALLBACK_MODEL},qwen/qwen3.8-27b,openai/gpt-oss-20b"
    ).split(",")
    if m.strip()
]


def fast_endpoints() -> list[tuple[str, str, str]]:
    """The rotation: (base URL, model, key) for every Groq model x key, then utim."""
    keys = groq_fallback_keys()
    endpoints = [(GROQ_URL, model, k) for model in GROQ_MODELS for k in keys]
    key = utim_key()
    if key:
        endpoints.append((UTIM_URL, UTIM_MODEL, key))
    return endpoints


def _next_groq_order(keys: list) -> list:
    """Every key once, starting at the next one in turn (spreads the load)."""
    with _groq_lock:
        start = next(_groq_turn) % len(keys)
    return keys[start:] + keys[:start]


class OpenAIProvider(LLMProvider):
    name = "openai"
    requires_key = True

    def __init__(self, model: str = "", *, api_key: str | None = None, base_url: str | None = None):
        super().__init__(model or DEFAULT_MODEL, api_key=api_key, base_url=base_url)

    def _client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - openai is a core dep
            raise ProviderError("The openai package is not installed.", provider=self.name) from exc

        # Local servers ignore the key but the SDK still requires a non-empty
        # value, so supply a placeholder rather than failing the request.
        key = self.api_key or ("not-needed" if self._is_local() else None)
        if not key:
            raise ProviderError(
                "No API key is set for the OpenAI-compatible provider.",
                provider=self.name,
                hint="Add one with `autoclip config set-secret openai`.",
            )

        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        is_groq = "groq.com" in (self.base_url or "")
        if not self._is_local() and not is_groq and fast_endpoints():
            # With a fallback ready, don't sit on a dead endpoint for minutes:
            # one quick retry, then Groq takes the request.
            kwargs["max_retries"] = 1
            kwargs["timeout"] = 90
        return AsyncOpenAI(**kwargs)

    def _is_local(self) -> bool:
        return bool(self.base_url) and (
            "localhost" in self.base_url or "127.0.0.1" in self.base_url
        )

    def _is_groq(self) -> bool:
        return "groq.com" in (self.base_url or "")

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        keys = [] if self._is_groq() else fast_endpoints()
        if keys and groq_first():
            # Groq first (rotating keys); the configured endpoint is the backup.
            try:
                return await self._complete_groq(keys, system, user, config)
            except ProviderError as groq_error:
                log.warning("All Groq keys failed (%s); trying %s.", groq_error, self.base_url)
                try:
                    return await self._complete_with(
                        self._client(), self.model, system, user, config
                    )
                except ProviderError as exc:
                    raise ProviderError(
                        f"All {len(keys)} Groq keys failed ({groq_error}) and so did the "
                        f"backup endpoint ({exc}).",
                        provider=self.name,
                    ) from exc
        try:
            return await self._complete_with(self._client(), self.model, system, user, config)
        except ProviderError as primary_error:
            if not keys:
                raise
            log.warning(
                "%s failed (%s); falling back to Groq (%d keys).",
                self.base_url or "OpenAI",
                primary_error,
                len(keys),
            )
            try:
                return await self._complete_groq(keys, system, user, config)
            except ProviderError as last:
                raise ProviderError(
                    f"The AI endpoint failed ({primary_error}) and so did all {len(keys)} "
                    f"Groq keys ({last}).",
                    provider=self.name,
                ) from last

    async def _complete_groq(
        self, endpoints: list[tuple[str, str, str]], system: str, user: str, config: DetectionConfig
    ) -> str:
        """Try each fast endpoint once (Groq keys, utim), starting at the next in turn."""
        from openai import AsyncOpenAI

        from . import usage

        last: ProviderError | None = None
        # utim is unlimited, so it takes every call first; the Groq free-tier
        # keys (rotated) only step in when it fails, which saves their quota.
        unlimited = [e for e in endpoints if e[0] == UTIM_URL]
        limited = [e for e in endpoints if e[0] != UTIM_URL]
        order = list(unlimited)
        # Groq: best model first; every model starts at the same next key, so
        # the rotation moves one key per call (not one per model).
        models = list(dict.fromkeys(e[1] for e in limited))
        if models:
            first = [e for e in limited if e[1] == models[0]]
            keys = [e[2] for e in _next_groq_order(first)]
            for model in models:
                by_key = {e[2]: e for e in limited if e[1] == model}
                order += [by_key[k] for k in keys if k in by_key]
        ready = [e for e in order if usage.resting(e[0], e[2], e[1]) <= 0]
        if not ready:
            raise ProviderError(
                "Every fast AI key is resting after hitting its rate limit.", provider=self.name
            )
        for url, model, key in ready:
            is_utim = url == UTIM_URL
            if is_utim and not _utim_slots.acquire(blocking=False):
                continue  # 2 already in flight: utim would refuse a third
            client = AsyncOpenAI(
                api_key=key,
                base_url=url,
                max_retries=0,
                timeout=UTIM_TIMEOUT_S if is_utim else 60,
            )
            try:
                return await self._complete_with(client, model, system, user, config)
            except ProviderError as exc:
                log.warning("%s key ...%s failed: %s", url, key[-4:], exc)
                last = exc
            finally:
                if is_utim:
                    _utim_slots.release()
        if last is None:
            raise ProviderError("No fast AI endpoint was free.", provider=self.name)
        raise last

    async def _complete_with(
        self, client, model: str, system: str, user: str, config: DetectionConfig
    ) -> str:
        request = {
            "model": model,
            "temperature": config.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Only OpenAI's own endpoint reliably honours this; other vendors
            # 400 on it, so a rejected request is retried without it.
            "response_format": {"type": "json_object"},
        }
        # Each rejected optional parameter is dropped and the request retried.
        # Reasoning models (o-series, gpt-5) refuse any non-default
        # temperature with an "Unsupported value" 400; dropping only
        # response_format would fail the same way again, so a suggested model
        # like o4-mini could never produce a clip.
        from openai import AsyncOpenAI

        from . import usage

        tracked = isinstance(client, AsyncOpenAI)
        url, key = (str(client.base_url), str(client.api_key)) if tracked else ("", "")
        while True:
            try:
                if tracked and _streams(url):
                    # This endpoint only answers when streamed.
                    raw = await client.chat.completions.with_raw_response.create(
                        **request, stream=True
                    )
                    text, tokens = await _collect_stream(raw.parse())
                    usage.record(
                        url,
                        key,
                        model=model,
                        headers=raw.headers,
                        tokens_in=getattr(tokens, "prompt_tokens", 0) or 0,
                        tokens_out=getattr(tokens, "completion_tokens", 0) or 0,
                    )
                    if not text.strip():
                        raise ProviderError(
                            "The endpoint streamed an empty reply.", provider=self.name
                        )
                    return text
                if tracked:
                    raw = await client.chat.completions.with_raw_response.create(**request)
                    response = raw.parse()
                    tokens = getattr(response, "usage", None)
                    usage.record(
                        url,
                        key,
                        model=model,
                        headers=raw.headers,
                        tokens_in=getattr(tokens, "prompt_tokens", 0) or 0,
                        tokens_out=getattr(tokens, "completion_tokens", 0) or 0,
                    )
                else:
                    response = await client.chat.completions.create(**request)
                break
            except Exception as exc:
                rejected = _rejected_parameter(exc, request)
                if tracked:
                    reply = getattr(exc, "response", None)
                    usage.record(
                        url,
                        key,
                        model=model,
                        ok=False,
                        status=int(getattr(exc, "status_code", 0) or 0),
                        headers=getattr(reply, "headers", None),
                        error=str(exc),
                    )
                if rejected is None:
                    raise _translate(exc, self.name, model) from exc
                log.debug("Endpoint rejected %s; retrying without it.", rejected)
                del request[rejected]

        choices = response.choices or []
        if not choices:
            raise ProviderError("The endpoint returned no choices.", provider=self.name)
        return choices[0].message.content or ""

    async def health_check(self) -> ProviderStatus:
        if not self.api_key and not self._is_local():
            return ProviderStatus(
                name=self.name, available=False, detail="No API key set", models=SUGGESTED_MODELS
            )
        try:
            client = self._client()
            models = await client.models.list()
            names = sorted(m.id for m in models.data)[:50]
        except Exception as exc:
            return ProviderStatus(name=self.name, available=False, detail=str(exc)[:200])
        return ProviderStatus(
            name=self.name,
            available=True,
            detail=self.base_url or "https://api.openai.com/v1",
            models=names or SUGGESTED_MODELS,
        )


def _streams(url: str) -> bool:
    """Endpoints that must be streamed (numes' free model times out otherwise)."""
    return "numes.ai" in url


async def _collect_stream(stream) -> tuple[str, object | None]:
    """Join a streamed reply's text; also return its usage block if one came."""
    parts: list[str] = []
    tokens = None
    async for chunk in stream:
        if getattr(chunk, "usage", None):
            tokens = chunk.usage
        for choice in chunk.choices or []:
            delta = getattr(choice, "delta", None)
            if delta is not None and delta.content:
                parts.append(delta.content)
    return "".join(parts), tokens


def _is_unsupported_parameter(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "response_format" in lowered or "unsupported" in lowered or "unrecognized" in lowered


def _rejected_parameter(exc: Exception, request: dict) -> str | None:
    """The optional request parameter an error is complaining about, if any.

    Only parameters still present in ``request`` qualify, so the retry loop in
    ``_complete`` ends after at most one retry per optional parameter.
    """
    if not _is_unsupported_parameter(exc) and "temperature" not in str(exc).lower():
        return None
    lowered = str(exc).lower()
    if "temperature" in lowered and "temperature" in request:
        return "temperature"
    if "response_format" in request:
        return "response_format"
    return None


def _translate(exc: Exception, provider: str, model: str) -> ProviderError:
    message = str(exc)
    lowered = message.lower()

    if "401" in lowered or "invalid api key" in lowered or "incorrect api key" in lowered:
        return ProviderError(
            "The endpoint rejected the API key.",
            provider=provider,
            hint="Re-add it with `autoclip config set-secret openai`.",
        )
    if "429" in lowered or "rate limit" in lowered:
        return ProviderError(
            "Rate limit reached.",
            provider=provider,
            hint="Wait and retry, or switch providers.",
        )
    if "404" in lowered or "does not exist" in lowered or "model_not_found" in lowered:
        return ProviderError(
            f"The endpoint does not recognise the model '{model}'.",
            provider=provider,
            hint="Check the model name against your provider's catalogue.",
        )
    if "connection" in lowered or "connect" in lowered:
        return ProviderError(
            "Could not reach the endpoint.",
            provider=provider,
            hint="Check the base URL in settings, and that any local server is running.",
        )
    if "quota" in lowered or "billing" in lowered:
        return ProviderError("The account has no remaining quota.", provider=provider)

    return ProviderError(f"Request failed: {message}", provider=provider)
