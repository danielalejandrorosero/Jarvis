"""Transcripción de audio a texto con faster-whisper.

Alcance de esta fase: transcribir un clip de audio ya grabado (después de detectar la wake
word). Nada de LLM ni TTS todavía (ADR-0004).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
LANGUAGE = "es"


def _register_cuda_dll_dirs() -> None:
    """Registrar los directorios `bin/` de los paquetes pip `nvidia-cublas-cu12` /
    `nvidia-cudnn-cu12`, si están instalados, para que ctranslate2 (backend de faster-whisper)
    encuentre `cublas64_12.dll` y `cudnn64_9.dll` en runtime.

    Sin esto, `device="cuda"` construye el modelo sin error (la construcción no toca cuBLAS) pero
    falla recién en `transcribe()` con "cublas64_12.dll is not found" — Windows no busca DLLs de
    dependencias en site-packages por default. No hace falta el instalador pesado de CUDA
    Toolkit: estos paquetes pip traen las DLLs necesarias.

    Usa `os.environ["PATH"]`, no `os.add_dll_directory()`: ese mecanismo solo cubre la carga de
    módulos de extensión de Python (.pyd), pero ctranslate2 es una librería nativa que resuelve
    sus propias dependencias (cuBLAS/cuDNN) vía el PATH real del proceso — confirmado
    empíricamente, `add_dll_directory` no alcanza acá. No-op silencioso si los paquetes no están
    instalados (entonces `device="cuda"` falla igual que antes, con el mismo error explícito —
    este helper no lo esconde).
    """
    dirs_to_add = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        bin_dir = os.path.join(spec.submodule_search_locations[0], "bin")
        if os.path.isdir(bin_dir):
            dirs_to_add.append(bin_dir)
    if dirs_to_add:
        os.environ["PATH"] = os.pathsep.join([*dirs_to_add, os.environ.get("PATH", "")])


def load_whisper_model(
    *, device: str = "cpu", compute_type: str | None = None
) -> WhisperModel:
    """Cargar el modelo Whisper.

    Default CPU porque siempre funciona sin dependencias extra. `device="cuda"` usa la GPU si
    `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` están instalados (`pip install`, no requiere el
    instalador de CUDA Toolkit) — confirmado funcionando en esta máquina con una RTX 3050.
    """
    if device == "cuda":
        _register_cuda_dll_dirs()
    resolved_compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    return WhisperModel(MODEL_SIZE, device=device, compute_type=resolved_compute_type)


def transcribe(
    audio: np.ndarray,
    *,
    model: WhisperModel,
    sample_rate: int = 16_000,
    language: str | None = LANGUAGE,
) -> str:
    """Transcribir audio int16 mono a texto plano. Concatena todos los segmentos detectados.

    `sample_rate` es solo una validación de contrato: WhisperModel.transcribe() no acepta un
    sampling_rate configurable cuando se le pasa un ndarray — asume el fijo del modelo (16kHz),
    tomado de `model.feature_extractor.sampling_rate`. Pasar audio a otra tasa lo transcribe mal
    en silencio, sin error — por eso esta función lo valida explícitamente en vez de confiar en
    que quien llama ya lo sabe.
    """
    if sample_rate != model.feature_extractor.sampling_rate:
        raise ValueError(
            f"sample_rate={sample_rate} no coincide con el sampling_rate del modelo "
            f"({model.feature_extractor.sampling_rate}); el audio se transcribiría mal en silencio."
        )
    audio_float = audio.astype(np.float32) / 32768.0
    segments, _info = model.transcribe(audio_float, language=language)
    return " ".join(segment.text.strip() for segment in segments).strip()
