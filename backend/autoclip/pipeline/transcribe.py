"""Transcribe stage — audio to word-level timestamps.

faster-whisper does the transcription; WhisperX (optional) adds speaker labels.
Word-level timing is non-negotiable: clip boundaries, caption animation, and
trim-handle snapping all derive from it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..config import WhisperSettings
from ..system import GPUInfo, report
from .transcript import Segment, Transcript, Word

log = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    """Transcription could not be completed."""


def resolve_compute(settings: WhisperSettings, gpu: GPUInfo | None = None) -> tuple[str, str]:
    """Return the ``(device, compute_type)`` to run Whisper with.

    An explicit ``settings.compute_type`` wins, but only if the device actually
    supports it. An override the backend will reject fails at model load with a
    message that points at cuDNN rather than at the setting, so an unsupported
    override is warned about and ignored rather than honoured into a crash.
    """
    gpu = gpu if gpu is not None else report().gpu
    automatic = gpu.compute_type

    override = settings.compute_type
    if not override:
        return gpu.device, automatic

    supported = gpu.supported_compute_types
    if supported and override not in supported:
        log.warning(
            "whisper.compute_type is set to %r, which %s does not support "
            "(supported: %s). Using %r instead.",
            override,
            gpu.device,
            ", ".join(sorted(supported)),
            automatic,
        )
        return gpu.device, automatic

    return gpu.device, override


def _model_load_error(exc: Exception, device: str, compute_type: str) -> TranscriptionError:
    """Turn a Whisper model-load failure into something actionable.

    The two causes look identical from the outside and need opposite fixes, so
    they're distinguished rather than lumped under one guess.
    """
    message = str(exc)
    lowered = message.lower()

    if "compute type" in lowered or "not support" in lowered:
        supported = report().gpu.supported_compute_types
        options = ", ".join(sorted(supported)) if supported else "unknown"
        return TranscriptionError(
            f"This device does not support the '{compute_type}' compute type.\n"
            f"Supported on {device}: {options}.\n\n"
            "Clear whisper.compute_type in config.json to let AutoClip choose, "
            "or set it to one of the supported values."
        )

    if device == "cuda" and ("cudnn" in lowered or "library" in lowered or "dll" in lowered):
        return TranscriptionError(
            f"Could not load Whisper on the GPU: {message}\n\n"
            "This is a missing CUDA runtime library. Install it with "
            "`uv pip install 'autoclip[gpu]'`, then retry — the job resumes "
            "from this stage."
        )

    return TranscriptionError(f"Could not load the Whisper model: {message}")


class WhisperModelCache:
    """Keeps loaded Whisper models alive between transcriptions (opt-in).

    Loading a model costs seconds and, on the GPU, a VRAM allocation; a
    caller that transcribes many short windows (Live Watch's judge) must not
    pay that per call. The batch pipeline does not pass a cache, so its
    one-model-per-job lifecycle (and VRAM release) is unchanged.

    Keys are ``(factory, model, device, compute_type)``; at most
    ``max_models`` stay resident (least recently used is dropped first).
    """

    def __init__(self, max_models: int = 1) -> None:
        import threading
        from collections import OrderedDict

        self.max_models = max(1, int(max_models))
        self._models: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()

    def loader(self, factory: Callable[..., object]) -> Callable[..., object]:
        """A drop-in for ``factory(model, device=..., compute_type=...)``."""

        def load(model: str, *, device: str, compute_type: str) -> object:
            key = (factory, model, device, compute_type)
            with self._lock:
                cached = self._models.get(key)
                if cached is not None:
                    self._models.move_to_end(key)
                    return cached
                instance = factory(model, device=device, compute_type=compute_type)
                self._models[key] = instance
                while len(self._models) > self.max_models:
                    self._models.popitem(last=False)
                return instance

        return load

    def evict(self, device: str | None = None) -> None:
        """Drop cached models (only ``device``'s when given)."""
        with self._lock:
            for key in [k for k in self._models if device is None or k[2] == device]:
                self._models.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)


def transcribe(
    audio: Path,
    settings: WhisperSettings | None = None,
    *,
    duration_s: float | None = None,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    model_cache: WhisperModelCache | None = None,
) -> Transcript:
    """Transcribe an audio file into a :class:`Transcript` with word timings.

    ``model_cache`` (optional) reuses a loaded model across calls; a GPU
    runtime failure evicts the cached CUDA model before the CPU retry.

    Preconditions:
        audio is a decodable audio file, ideally 16 kHz mono as produced by
        :func:`autoclip.pipeline.prepare.extract_audio`.
    """
    settings = settings or WhisperSettings()

    # Must happen before faster-whisper pulls in CTranslate2, which resolves its
    # CUDA dependencies at import.
    from ..cuda import ensure_cuda_libraries

    ensure_cuda_libraries()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - faster-whisper is a core dep
        raise TranscriptionError(
            "faster-whisper is not installed. Run `autoclip doctor` for details."
        ) from exc

    factory = WhisperModel if model_cache is None else model_cache.loader(WhisperModel)
    device, compute_type = resolve_compute(settings)
    run = {
        "audio": audio,
        "settings": settings,
        "duration_s": duration_s,
        "on_progress": on_progress,
        "cancelled": cancelled,
    }
    try:
        transcript = _run_whisper(factory, device=device, compute_type=compute_type, **run)
    except _GPURuntimeFailure as exc:
        if model_cache is not None:
            # A CUDA model that failed at encode would fail the same way next
            # time; only the CPU fallback may stay resident.
            model_cache.evict(device)
        # The GPU is present but its runtime is not (missing cuBLAS/cuDNN, a
        # driver mismatch, out of memory). The CPU produces the same transcript,
        # only slower — far better than failing a long job at 5 %.
        cpu_type = _cpu_compute_type()
        log.warning(
            "Whisper could not run on the GPU (%s). Transcribing on the CPU instead "
            "(compute_type=%s) — same result, slower. Run `autoclip doctor` to fix the GPU.",
            exc,
            cpu_type,
        )
        if on_progress:
            on_progress(0.0)
        transcript = _run_whisper(factory, device="cpu", compute_type=cpu_type, **run)

    if on_progress:
        on_progress(1.0)

    log.info(
        "Transcribed %d words in %d segments.", len(transcript.words), len(transcript.segments)
    )
    return transcript


class _GPURuntimeFailure(RuntimeError):
    """Whisper failed on CUDA for a reason the CPU does not share."""


#: Lower-cased fragments of CTranslate2 / CUDA errors that mean "this GPU
#: cannot run inference right now", as opposed to a bad file or a bad setting.
_GPU_RUNTIME_MARKERS = (
    "cublas",
    "cudnn",
    "cuda",
    "cannot be loaded",
    ".dll",
    "libcu",
    "out of memory",
)


def _is_gpu_runtime_failure(exc: BaseException) -> bool:
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _GPU_RUNTIME_MARKERS)


def _cpu_compute_type() -> str:
    """Best compute type for a CPU fallback, asked of CTranslate2 itself."""
    from ..system import query_compute_types

    return GPUInfo(accel="cpu", supported_compute_types=query_compute_types("cpu")).compute_type


def _run_whisper(
    whisper_model,
    *,
    audio: Path,
    settings: WhisperSettings,
    device: str,
    compute_type: str,
    duration_s: float | None,
    on_progress: Callable[[float], None] | None,
    cancelled: Callable[[], bool] | None,
) -> Transcript:
    """One full transcription attempt on one device.

    GPU-runtime failures — at model load *or* at the first encode, which is
    where a missing cuBLAS actually surfaces — are raised as
    :class:`_GPURuntimeFailure` so the caller can retry on the CPU.
    """
    log.info(
        "Transcribing with model=%s device=%s compute_type=%s",
        settings.model,
        device,
        compute_type,
    )

    try:
        model = whisper_model(settings.model, device=device, compute_type=compute_type)
    except Exception as exc:
        if device == "cuda" and _is_gpu_runtime_failure(exc):
            raise _GPURuntimeFailure(str(exc)) from exc
        raise _model_load_error(exc, device, compute_type) from exc

    transcript = Transcript(
        language=settings.language,
        model=settings.model,
        source="whisper",
    )
    try:
        segments_iter, info = model.transcribe(
            str(audio),
            language=settings.language or None,
            word_timestamps=True,
            vad_filter=True,
            # Trims long silences before decoding, which both speeds things up and
            # stops Whisper hallucinating text into empty audio.
            vad_parameters={"min_silence_duration_ms": 500},
        )

        total = duration_s or getattr(info, "duration", 0.0) or 0.0
        transcript.language = getattr(info, "language", "") or settings.language

        for segment in segments_iter:
            if cancelled is not None and cancelled():
                raise TranscriptionError("Transcription cancelled.")

            first_word = len(transcript.words)
            for word in getattr(segment, "words", None) or []:
                text = (word.word or "").strip()
                if not text:
                    continue
                transcript.words.append(
                    Word(text=text, start=float(word.start), end=float(word.end))
                )

            # A segment with no word timings still carries text worth keeping for
            # display, but it can't contribute to word-indexed clip boundaries.
            last_word = max(first_word, len(transcript.words) - 1)
            transcript.segments.append(
                Segment(
                    text=(segment.text or "").strip(),
                    start=float(segment.start),
                    end=float(segment.end),
                    first_word=first_word,
                    last_word=last_word,
                )
            )

            if on_progress and total:
                on_progress(min(1.0, float(segment.end) / total))
    except TranscriptionError:
        raise
    except Exception as exc:
        if device == "cuda" and _is_gpu_runtime_failure(exc):
            raise _GPURuntimeFailure(str(exc)) from exc
        raise

    if not transcript.words:
        raise TranscriptionError(
            "No speech was detected in this audio. If the file really does contain "
            "speech, try a larger Whisper model or check that the correct audio track "
            "was extracted."
        )

    return transcript


# --------------------------------------------------------------------------
# Diarization
# --------------------------------------------------------------------------


def diarize(
    audio: Path,
    transcript: Transcript,
    *,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Transcript:
    """Label each word with a speaker, in place.

    Uses pyannote via WhisperX. Returns the transcript unchanged (with a logged
    warning) if diarization isn't available — speaker labels improve reframing
    but are never required for the pipeline to complete.

    Preconditions:
        transcript already has word-level timings.
    """
    if not transcript.words:
        return transcript

    pipeline_cls = _load_diarization_pipeline()
    if pipeline_cls is None:
        log.warning(
            "WhisperX is not installed; skipping diarization. "
            "Install it with `uv pip install 'autoclip[diarization]'`."
        )
        return transcript

    if not hf_token:
        log.warning(
            "Diarization needs a HuggingFace token. Set one with "
            "`autoclip config set-secret huggingface_token`, and accept the pyannote "
            "model licences on huggingface.co. Continuing without speaker labels."
        )
        return transcript

    device = report().gpu.device
    try:
        pipeline = pipeline_cls(use_auth_token=hf_token, device=device)
        diarization = pipeline(str(audio), min_speakers=min_speakers, max_speakers=max_speakers)
    except Exception as exc:
        log.warning("Diarization failed (%s); continuing without speaker labels.", exc)
        return transcript

    turns = _diarization_turns(diarization)
    if not turns:
        return transcript

    _assign_speakers(transcript, turns)
    log.info("Diarization labelled %d speakers.", len(transcript.speakers))
    return transcript


def _load_diarization_pipeline():
    """Import WhisperX's diarization pipeline across its API reshuffles."""
    try:
        from whisperx.diarize import DiarizationPipeline

        return DiarizationPipeline
    except ImportError:
        pass
    try:
        from whisperx import DiarizationPipeline  # type: ignore[attr-defined]

        return DiarizationPipeline
    except (ImportError, AttributeError):
        return None


def _diarization_turns(diarization) -> list[tuple[str, float, float]]:
    """Normalise WhisperX output into ``(speaker, start, end)`` tuples.

    Recent versions return a pandas DataFrame; older ones return a pyannote
    ``Annotation``. Both are handled so the extra can be upgraded independently.
    """
    turns: list[tuple[str, float, float]] = []

    if hasattr(diarization, "itertuples"):  # DataFrame
        for row in diarization.itertuples():
            speaker = getattr(row, "speaker", None)
            start = getattr(row, "start", None)
            end = getattr(row, "end", None)
            if speaker is not None and start is not None and end is not None:
                turns.append((str(speaker), float(start), float(end)))
        return turns

    if hasattr(diarization, "itertracks"):  # pyannote Annotation
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((str(speaker), float(segment.start), float(segment.end)))

    return turns


def _assign_speakers(transcript: Transcript, turns: list[tuple[str, float, float]]) -> None:
    """Attach a speaker to every word by maximum temporal overlap.

    Overlap rather than midpoint containment: words straddling a turn boundary
    are common, and the speaker who covers more of the word is the better guess.
    """
    turns = sorted(turns, key=lambda t: t[1])

    for word in transcript.words:
        best_speaker: str | None = None
        best_overlap = 0.0
        for speaker, start, end in turns:
            if start > word.end:
                break
            overlap = min(word.end, end) - max(word.start, start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        word.speaker = best_speaker

    for segment in transcript.segments:
        segment.speaker = transcript.dominant_speaker(segment.first_word, segment.last_word)
