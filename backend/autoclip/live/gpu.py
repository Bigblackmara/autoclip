"""Shared heavy-compute resources for Live Watch.

Two process-wide singletons, used by BOTH the judge-phase transcription
(watcher) and the clip renders (clip_factory):

* :data:`GPU_SLOT` — one GPU-heavy stage at a time (Whisper transcription,
  the hardware encode). Without it a judge transcription and a render's
  transcription load two Whisper models into VRAM side by side.
* :data:`MODEL_CACHE` — one resident Whisper model, reused across calls.
  Loading a model per judged window costs seconds and a fresh VRAM
  allocation every time.

Both are threading primitives: every caller runs in a worker thread
(``asyncio.to_thread``), never on the event loop.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ..pipeline.transcribe import WhisperModelCache

#: One GPU-heavy stage at a time across judge transcription and renders.
GPU_SLOT = threading.BoundedSemaphore(1)

#: The Whisper model shared by judge transcription and renders.
MODEL_CACHE = WhisperModelCache(max_models=1)


@contextlib.contextmanager
def gpu_slot() -> Iterator[None]:
    """Hold the shared GPU slot for the duration of the block."""
    with GPU_SLOT:
        yield


def whisper_settings() -> Any:
    """The user's WhisperSettings (model, compute type, language)."""
    from .. import config

    return config.load().whisper


def transcribe(
    audio: Path,
    settings: Any = None,
    *,
    transcribe_fn: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Transcribe under the GPU slot with the shared model cache.

    ``settings`` defaults to the user's configured WhisperSettings — Live
    Watch must honour the same model/compute/language choices the batch
    pipeline does. ``transcribe_fn`` defaults to the pipeline stage, looked
    up at call time so tests that patch ``pipeline.transcribe.transcribe``
    are honoured.
    """
    if settings is None:
        settings = whisper_settings()
    if transcribe_fn is None:
        from ..pipeline import transcribe as transcribe_mod

        transcribe_fn = transcribe_mod.transcribe
    with gpu_slot():
        return transcribe_fn(audio, settings, model_cache=MODEL_CACHE, **kwargs)


__all__ = ["GPU_SLOT", "MODEL_CACHE", "gpu_slot", "transcribe", "whisper_settings"]
