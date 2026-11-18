"""Caption language: which language and script burned captions, the hook
headline and the post text are written in.

Three modes (``ExportSettings.caption_language``):

* ``as_spoken`` (default) — every language keeps its own script, except
  Hindi, which is written in Latin letters the way people type it in chat
  (Hinglish: "kya baat hai"). Whisper writes Hindi in Devanagari, so a
  rule-based romaniser converts it; no model call, so it cannot fail.
* ``roman`` — every language in Latin letters. Devanagari goes through the
  same rules; any other non-Latin script is transliterated by the LLM.
* ``english`` — translated into English by the LLM. Word timings are kept by
  translating whole caption lines and spreading each translated line's words
  over the span the original line was spoken in.

The romaniser is a port of CrestClip's ``hinglish.ts``. Spelling follows chat
habits rather than a formal standard: the silent "a" is dropped as Hindi
speakers drop it ("samajhna", not "samajhana"), long vowels are mostly written
short ("rakhi", "biwi" — but "baat", "kaam" in one-syllable words), व is "w"
("wala"), ड़ is "d" ("chhod do"), nasal vowels end in "n" ("main", "hoon").
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .transcript import Word

if TYPE_CHECKING:
    from ..providers.base import LLMProvider

log = logging.getLogger(__name__)

MODES = ("as_spoken", "roman", "english")

#: Languages written in Devanagari that "as_spoken" keeps in Devanagari.
#: Hindi (and an unknown language with Devanagari text) becomes Hinglish.
DEVANAGARI_NATIVE = frozenset({"mr", "ne", "sa", "kok", "mai", "bho", "new"})

# --------------------------------------------------------------------------
# Devanagari -> Hinglish
# --------------------------------------------------------------------------

_VOWELS = {
    "अ": "a",  # अ
    "आ": "aa",  # आ
    "इ": "i",  # इ
    "ई": "ee",  # ई
    "उ": "u",  # उ
    "ऊ": "oo",  # ऊ
    "ऋ": "ri",  # ऋ
    "ए": "e",  # ए
    "ऐ": "ai",  # ऐ
    "ओ": "o",  # ओ
    "औ": "au",  # औ
    "ऑ": "o",  # ऑ
    "ऍ": "e",  # ऍ
}
_SIGNS = {
    "ा": "aa",  # ा
    "ि": "i",  # ि
    "ी": "ee",  # ी
    "ु": "u",  # ु
    "ू": "oo",  # ू
    "ृ": "ri",  # ृ
    "े": "e",  # े
    "ै": "ai",  # ै
    "ो": "o",  # ो
    "ौ": "au",  # ौ
    "ॉ": "o",  # ॉ
    "ॅ": "e",  # ॅ
}
_CONSONANTS = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "n",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "n",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
    "ळ": "l",
    # With a nukta, as one character.
    "क़": "q",  # क़
    "ख़": "kh",  # ख़
    "ग़": "gh",  # ग़
    "ज़": "z",  # ज़
    "ड़": "d",  # ड़
    "ढ़": "dh",  # ढ़
    "फ़": "f",  # फ़
    "य़": "y",  # य़
}
#: With a nukta written as two characters (ड़ is "d" as typed in chat).
_NUKTA = {
    "क": "q",
    "ख": "kh",
    "ग": "gh",
    "ज": "z",
    "ड": "d",
    "ढ": "dh",
    "फ": "f",
}
_VIRAMA = "्"
_ANUSVARA = "ं"
_CHANDRABINDU = "ँ"
_VISARGA = "ः"
_NUKTA_SIGN = "़"
_DIGITS = "०१२३४५६७८९"
_DANDAS = ("।", "॥")  # । ॥

_DEVANAGARI = re.compile("[ऀ-ॿ]")
_DEVANAGARI_RUN = re.compile("[ऀ-ॿ]+")

#: Marks the inherent "a" until schwa deletion has run.
_INHERENT = "A"


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


@dataclass
class _Unit:
    """A consonant (with its vowel, or none after a virama) or a standalone vowel."""

    c: str | None
    #: A sign, the inherent "a" (``_INHERENT``), "" once deleted, None after a virama.
    v: str | None
    nasal: bool = False
    visarga: bool = False


def _spell(units: list[_Unit]) -> str:
    if not units:
        return ""
    # The silent "a": from the right, an inherent a is dropped at the end of a
    # word (after more than one unit), and in the middle when a vowel comes
    # before it and the next consonant carries a vowel of its own.
    for i in range(len(units) - 1, 0, -1):
        unit = units[i]
        if unit.v != _INHERENT or unit.nasal:
            continue
        prev = units[i - 1]
        if i == len(units) - 1:
            # Kept after a cluster ending in r/y/w/l ("mitra", "satya").
            if prev.v is not None or (unit.c or "") not in ("r", "y", "v", "l"):
                unit.v = ""
            continue
        nxt = units[i + 1]
        # Kept after a nasal: dropping it would stack three consonants.
        if prev.v not in (None, "") and not prev.nasal and nxt.c and nxt.v not in (None, ""):
            unit.v = ""

    syllables = sum(1 for u in units if u.v not in (None, ""))
    out: list[str] = []
    for i, unit in enumerate(units):
        vowel = "a" if unit.v == _INHERENT else (unit.v or "")
        last = i == len(units) - 1
        next_is_vowel = i + 1 < len(units) and not units[i + 1].c
        # Chat spelling: long ā is "aa" only in one-syllable words ("baat",
        # "kaam"), else "a"; ī is "i"; ū is "oo" in one-syllable words ("hoon").
        if vowel == "aa" and (syllables > 1 or last or next_is_vowel):
            vowel = "a"
        if vowel == "ee":
            vowel = "i"
        if vowel == "oo" and syllables > 1:
            vowel = "u"
        consonant = unit.c or ""
        # व is "w" ("wala", "biwi"), but "v" at the start before i ("vishwas").
        if consonant == "v" and not (i == 0 and unit.v in ("i", "ee")):
            consonant = "w"
        out.append(consonant + vowel)
        if unit.nasal:
            following = units[i + 1].c if i + 1 < len(units) else ""
            out.append("m" if following in ("p", "b", "bh", "m") else "n")
        if unit.visarga:
            out.append("h")
    return "".join(out)


def romanize_word(word: str) -> str:
    """One run of Devanagari in Hinglish spelling."""
    units: list[_Unit] = []
    out: list[str] = []

    def flush() -> None:
        out.append(_spell(units))
        units.clear()

    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in _CONSONANTS:
            consonant = _CONSONANTS[ch]
            if i + 1 < len(chars) and chars[i + 1] == _NUKTA_SIGN:
                consonant = _NUKTA.get(ch, consonant)
                i += 1
            units.append(_Unit(c=consonant, v=_INHERENT))
        elif ch in _VOWELS:
            units.append(_Unit(c=None, v=_VOWELS[ch]))
        elif ch in _SIGNS and units:
            units[-1].v = _SIGNS[ch]
        elif ch == _VIRAMA and units:
            units[-1].v = None
        elif ch in (_ANUSVARA, _CHANDRABINDU) and units:
            if units[-1].v == _INHERENT:
                units[-1].v = "a"
            units[-1].nasal = True
        elif ch == _VISARGA and units:
            units[-1].visarga = True
        elif ch in _DIGITS:
            flush()
            out.append(str(_DIGITS.index(ch)))
        elif ch in _DANDAS:
            flush()
            out.append(".")
        else:
            flush()
            out.append(ch)
        i += 1
    flush()
    return "".join(out)


def to_hinglish(text: str) -> str:
    """Devanagari in ``text`` written in Hinglish; everything else left as it is."""
    if not has_devanagari(text):
        return text
    return _DEVANAGARI_RUN.sub(lambda match: romanize_word(match.group(0)), text)


# --------------------------------------------------------------------------
# Script checks
# --------------------------------------------------------------------------


def base_language(code: str) -> str:
    """ "en-US" -> "en"; "" stays ""."""
    return re.split(r"[-_]", code.strip().lower())[0] if code else ""


def has_non_latin(text: str) -> bool:
    """Whether ``text`` holds letters outside the Latin script."""
    return any(ch.isalpha() and "LATIN" not in unicodedata.name(ch, "") for ch in text)


def hinglish_applies(language: str) -> bool:
    """ "as_spoken" romanises Devanagari unless the language keeps that script."""
    return base_language(language) not in DEVANAGARI_NATIVE


def writing_rule(mode: str) -> str:
    """Detection-prompt rule for the language of hooks and post text."""
    if mode == "english":
        return (
            "Write `hooks` and `description` in English, whatever language is spoken. "
            "`hook` stays a verbatim quote of the transcript."
        )
    if mode == "roman":
        return (
            "Write `hooks` and `description` in the language the video is spoken in, but in "
            "Latin (English) letters, romanised the way people type in chat (Hinglish for "
            'Hindi: "Ye sach hai?"). Never translate. `hook` stays a verbatim quote of the '
            "transcript."
        )
    return (
        "Write `hooks` and `description` in the language the video is spoken in. Hindi (and "
        "Hindi mixed with English) is written in Latin (English) letters the way people type "
        'it in chat, Hinglish ("Ye sach hai?"), never in Devanagari. Other languages use '
        "their own script; an English video gets English hooks. `hook` stays a verbatim "
        "quote of the transcript."
    )


# --------------------------------------------------------------------------
# Caption lines and retiming
# --------------------------------------------------------------------------

#: A translated line is at most this many source words.
LINE_MAX_WORDS = 12
#: A pause this long ends a line.
LINE_MAX_GAP_S = 0.6


def caption_lines(
    words: Sequence[Word], *, max_words: int = LINE_MAX_WORDS, max_gap_s: float = LINE_MAX_GAP_S
) -> list[tuple[int, int]]:
    """Inclusive (first, last) word-index ranges: sentences, split at pauses and length."""
    lines: list[tuple[int, int]] = []
    first = 0
    for index, word in enumerate(words):
        if index > first:
            gap = word.start - words[index - 1].end
            if gap > max_gap_s or index - first >= max_words:
                lines.append((first, index - 1))
                first = index
        if word.ends_sentence:
            lines.append((first, index))
            first = index + 1
    if first < len(words):
        lines.append((first, len(words) - 1))
    return [(a, b) for a, b in lines if a <= b]


def spread_words(text: str, start: float, end: float, speaker: str | None = None) -> list[Word]:
    """Words of ``text`` laid over ``start``-``end``, each in proportion to its length.

    A translated line has different words from the original, so per-word
    timings cannot be kept; the line itself stays exactly where it was spoken.
    """
    tokens = text.split()
    if not tokens:
        return []
    span = max(0.0, end - start)
    weights = [len(token) + 1 for token in tokens]
    total = float(sum(weights))
    out: list[Word] = []
    cursor = start
    for index, (token, weight) in enumerate(zip(tokens, weights, strict=True)):
        word_end = end if index == len(tokens) - 1 else cursor + span * weight / total
        out.append(
            Word(text=token, start=round(cursor, 3), end=round(word_end, 3), speaker=speaker)
        )
        cursor = word_end
    return out


# --------------------------------------------------------------------------
# The LLM pass (translate / transliterate whole lines)
# --------------------------------------------------------------------------

PROMPT_VERSION = "caption_language_v1"

#: Converts lines; returns the same number of lines, in order.
Converter = Callable[[list[str], str], list[str]]


class LanguageError(RuntimeError):
    """The model did not return usable converted lines."""


def _task(mode: str) -> str:
    if mode == "english":
        return (
            "Translate every line into natural, idiomatic, spoken English. Keep names, brands "
            "and numbers. Keep each line about as short as the original: these are captions."
        )
    return (
        "Write every line in Latin (English) letters, romanised the way people type in chat "
        '(Hinglish for Hindi: "main hota na", "kya baat hai"). Do NOT translate: keep the '
        "same words, only change the letters. Lines already in Latin letters stay as they are."
    )


def _parse_lines(raw: str, expected: int) -> list[str]:
    from ..providers.base import extract_json_object

    payload = extract_json_object(raw)
    lines = payload.get("lines")
    if not isinstance(lines, list):
        raise ValueError('"lines" must be a list of strings.')
    out = [
        " ".join(str(item.get("text", "") if isinstance(item, dict) else item).split())
        for item in lines
    ]
    if len(out) != expected:
        raise ValueError(f"Expected {expected} lines, got {len(out)}.")
    return out


async def convert_lines_async(
    provider: LLMProvider, lines: list[str], mode: str, *, language: str = ""
) -> list[str]:
    """Translate (``english``) or transliterate (``roman``) caption lines.

    One retry carries the validation error, as detection does.
    """
    from ..providers.base import DetectionConfig, load_prompt

    if not lines:
        return []
    system = load_prompt(PROMPT_VERSION)
    spoken = f" The video is spoken in '{language}'." if language else ""
    user = (
        f"{_task(mode)}{spoken}\n\n"
        f"Return exactly {len(lines)} lines, in the same order, as "
        '{"lines": ["...", ...]}.\n\n' + json.dumps({"lines": lines}, ensure_ascii=False)
    )
    config = DetectionConfig(prompt_version=PROMPT_VERSION, temperature=0.2)
    raw = await provider._complete(system, user, config)
    try:
        return _parse_lines(raw, len(lines))
    except ValueError as first:
        repair = (
            f"{user}\n\nYour previous response was unusable: {first}\n"
            "Respond again with ONLY the JSON object."
        )
        raw = await provider._complete(system, repair, config)
        try:
            return _parse_lines(raw, len(lines))
        except ValueError as second:
            raise LanguageError(
                f"{provider.name} returned unusable caption lines: {second}"
            ) from second


def run_blocking(coro: Any) -> Any:
    """Run a coroutine to completion from sync code, even inside a running loop.

    The pipeline's export stage is sync code called from an async runner, so
    ``asyncio.run`` there would fail; a short-lived thread gets its own loop.
    """
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=target, name="caption-language", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def provider_converter(
    provider: LLMProvider, *, language: str = "", cache_dir: Path | None = None
) -> Converter:
    """A sync converter backed by ``provider``, caching answers in ``cache_dir``.

    A re-export of the same clip in the same mode reuses the earlier answer
    instead of paying for it again.
    """

    def convert(lines: list[str], mode: str) -> list[str]:
        key = hashlib.sha1(
            json.dumps([mode, language, lines], ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20]
        cached = cache_dir / f"{mode}-{key}.json" if cache_dir is not None else None
        if cached is not None and cached.exists():
            try:
                stored = json.loads(cached.read_text(encoding="utf-8"))
                if isinstance(stored, list) and len(stored) == len(lines):
                    return [str(line) for line in stored]
            except (OSError, ValueError):
                pass
        out = run_blocking(convert_lines_async(provider, lines, mode, language=language))
        if cached is not None:
            try:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            except OSError:
                log.debug("Could not cache caption-language lines.", exc_info=True)
        return out

    return convert


# --------------------------------------------------------------------------
# Applying a mode
# --------------------------------------------------------------------------


@dataclass
class Localised:
    words: list[Word]
    hook: str = ""
    description: str = ""
    #: What was done, for logs ("hinglish", "translated", ...); empty = untouched.
    applied: list[str] = field(default_factory=list)


def _map_words(words: Sequence[Word], fn: Callable[[str], str]) -> list[Word]:
    return [Word(text=fn(w.text), start=w.start, end=w.end, speaker=w.speaker) for w in words]


def _needs_translation(words: Sequence[Word], extras: Sequence[str], language: str) -> bool:
    if base_language(language) == "en":
        return False
    texts = [w.text for w in words] + [t for t in extras if t]
    if not language:
        # Unknown language: plain ASCII text is taken to be English already.
        return any(not text.isascii() for text in texts)
    return bool(texts)


def localise(
    words: Sequence[Word],
    mode: str,
    *,
    language: str = "",
    hook: str = "",
    description: str = "",
    converter: Converter | None = None,
) -> Localised:
    """Apply a caption-language mode to caption words, the hook and the post text.

    Never raises: a failed model pass leaves the text in its rule-based form
    (Hinglish where it applies), which is still readable.
    """
    words = list(words)
    if mode not in MODES:
        mode = "as_spoken"

    def rules(text: str, *, force: bool = False) -> str:
        return to_hinglish(text) if force or hinglish_applies(language) else text

    if (
        mode == "english"
        and converter is not None
        and _needs_translation(words, (hook, description), language)
    ):
        try:
            return _convert(words, hook, description, "english", converter, only_non_latin=False)
        except Exception as exc:  # noqa: BLE001 - a failed pass must not fail the export
            log.warning("Caption translation failed (%s); keeping the spoken language.", exc)

    force = mode in ("roman", "english")
    out = Localised(
        words=_map_words(words, lambda t: rules(t, force=force)),
        hook=rules(hook, force=force),
        description=rules(description, force=force),
    )
    changed = (
        any(a.text != b.text for a, b in zip(words, out.words, strict=True))
        or out.hook != hook
        or out.description != description
    )
    if changed:
        out.applied.append("hinglish")
    if mode in ("roman", "english") and converter is not None:
        texts = [w.text for w in out.words] + [out.hook, out.description]
        if any(has_non_latin(text) for text in texts):
            try:
                converted = _convert(
                    out.words, out.hook, out.description, "roman", converter, only_non_latin=True
                )
                converted.applied = [*out.applied, *converted.applied]
                return converted
            except Exception as exc:  # noqa: BLE001
                log.warning("Caption romanisation failed (%s); keeping the script.", exc)
    return out


def _convert(
    words: list[Word],
    hook: str,
    description: str,
    mode: str,
    converter: Converter,
    *,
    only_non_latin: bool,
) -> Localised:
    """Send caption lines (plus hook and description) through ``converter`` and retime."""
    ranges = caption_lines(words)
    line_texts = [" ".join(w.text.strip() for w in words[a : b + 1]) for a, b in ranges]
    selected = [i for i, text in enumerate(line_texts) if not only_non_latin or has_non_latin(text)]
    extras: list[tuple[str, str]] = [
        (name, text)
        for name, text in (("hook", hook), ("description", description))
        if text.strip() and (not only_non_latin or has_non_latin(text))
    ]
    request = [line_texts[i] for i in selected] + [text for _, text in extras]
    if not request:
        return Localised(words=words, hook=hook, description=description)
    answer = converter(request, mode)
    if len(answer) != len(request):
        raise LanguageError(f"Expected {len(request)} lines, got {len(answer)}.")

    replaced = dict(zip(selected, answer[: len(selected)], strict=True))
    out_words: list[Word] = []
    for index, (a, b) in enumerate(ranges):
        if index not in replaced:
            out_words.extend(words[a : b + 1])
            continue
        spread = spread_words(
            replaced[index], words[a].start, words[b].end, speaker=words[a].speaker
        )
        # A line the model returned empty keeps its original words.
        out_words.extend(spread or words[a : b + 1])
    tail = dict(zip((name for name, _ in extras), answer[len(selected) :], strict=True))
    return Localised(
        words=out_words,
        hook=tail.get("hook") or hook,
        description=tail.get("description") or description,
        applied=["translated" if mode == "english" else "romanised"],
    )


def for_export(
    words: Sequence[Word],
    mode: str,
    *,
    language: str = "",
    hook: str = "",
    description: str = "",
    provider_name: str | None = None,
    settings: Any = None,
    cache_dir: Path | None = None,
) -> Localised:
    """:func:`localise` with a converter built from the job's provider when needed.

    "as_spoken" never calls a model. For "roman" and "english" a provider that
    cannot be built (no key, unknown name) leaves the rule-based result.
    """
    converter: Converter | None = None
    if mode in ("roman", "english"):
        try:
            from ..providers import build_provider

            provider = build_provider(provider_name or None, settings)
            converter = provider_converter(provider, language=language, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - captions still render without it
            log.warning("No provider for the caption-language pass (%s).", exc)
    return localise(
        words, mode, language=language, hook=hook, description=description, converter=converter
    )
