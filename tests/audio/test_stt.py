"""Tests para el núcleo puro de transcripción de audio a texto (`transcribe`).

No cargan el modelo real de Whisper (`load_whisper_model`): `transcribe()` recibe un modelo
faster-whisper stub, controlado por el test, así que corren rápido, sin red y sin descargar
pesos del modelo (CI-safe).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pytest
from faster_whisper import WhisperModel

from jarvis.audio.stt import transcribe

SAMPLE_RATE = 16_000


class _FakeSegment:
    """Stub de un segmento de `faster_whisper`: solo expone `.text`, lo único que usa `transcribe()`."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeFeatureExtractor:
    def __init__(self, sampling_rate: int) -> None:
        self.sampling_rate = sampling_rate


class _FakeWhisperModel:
    """Stub de `faster_whisper.WhisperModel`.

    `.transcribe()` devuelve segmentos predefinidos y registra con qué audio/idioma fue
    llamado en `.calls`, para poder inspeccionarlo desde el test (spy) sin cargar un modelo real.
    """

    def __init__(self, segments: Iterable[_FakeSegment], sampling_rate: int = SAMPLE_RATE) -> None:
        self.feature_extractor = _FakeFeatureExtractor(sampling_rate)
        self._segments = segments
        self.calls: list[tuple[np.ndarray, str | None]] = []

    def transcribe(self, audio: np.ndarray, *, language: str | None = None) -> tuple[Iterable[_FakeSegment], None]:
        self.calls.append((audio, language))
        return self._segments, None


def _model_with(segments: Iterable[_FakeSegment], sampling_rate: int = SAMPLE_RATE) -> WhisperModel:
    # El stub no implementa la superficie completa de WhisperModel, solo lo que `transcribe()`
    # usa (`.feature_extractor.sampling_rate` y `.transcribe()`). Se castea para satisfacer la
    # firma de `transcribe(..., model: WhisperModel)`.
    return cast(WhisperModel, _FakeWhisperModel(segments, sampling_rate=sampling_rate))


def test_transcribe_joins_segment_texts_stripped_with_spaces() -> None:
    """Caso aceptado: varios segmentos con texto se concatenan con espacios, cada uno stripeado."""
    segments = [
        _FakeSegment("  Hola  "),
        _FakeSegment("cómo estás\n"),
        _FakeSegment("\tbien"),
    ]
    model = _model_with(segments)
    audio = np.zeros(SAMPLE_RATE, dtype=np.int16)

    result = transcribe(audio, model=model, sample_rate=SAMPLE_RATE)

    assert result == "Hola cómo estás bien"


def test_transcribe_returns_empty_string_when_no_segments_detected() -> None:
    """Silencio / sin voz detectada: `model.transcribe()` no produce segmentos, no es un error."""
    model = _model_with([])
    audio = np.zeros(SAMPLE_RATE, dtype=np.int16)

    result = transcribe(audio, model=model, sample_rate=SAMPLE_RATE)

    assert result == ""


def test_transcribe_raises_value_error_on_sample_rate_mismatch() -> None:
    """Caso rechazado: un sample_rate que no coincide con el del modelo debe fallar explícito,
    en vez de transcribir mal en silencio."""
    model = _model_with([_FakeSegment("no debería usarse")], sampling_rate=16_000)
    audio = np.zeros(8_000, dtype=np.int16)

    with pytest.raises(ValueError, match="sample_rate"):
        transcribe(audio, model=model, sample_rate=8_000)


def test_transcribe_normalizes_int16_audio_to_float32_unit_range() -> None:
    """El audio int16 se normaliza a float32 en aproximadamente [-1, 1] antes de pasarlo al modelo."""
    fake_model = _FakeWhisperModel([_FakeSegment("ok")])
    model = cast(WhisperModel, fake_model)
    audio = np.array([32767, -32768, 0], dtype=np.int16)

    transcribe(audio, model=model, sample_rate=SAMPLE_RATE)

    assert len(fake_model.calls) == 1
    called_audio, _language = fake_model.calls[0]
    assert called_audio.dtype == np.float32
    np.testing.assert_allclose(called_audio, [1.0, -1.0, 0.0], atol=1e-4)


def test_transcribe_passes_language_through_to_model() -> None:
    """El parámetro `language` (incluyendo el default "es") se reenvía tal cual a `model.transcribe()`."""
    fake_model = _FakeWhisperModel([_FakeSegment("hola")])
    model = cast(WhisperModel, fake_model)
    audio = np.zeros(4, dtype=np.int16)

    transcribe(audio, model=model, sample_rate=SAMPLE_RATE)
    transcribe(audio, model=model, sample_rate=SAMPLE_RATE, language=None)

    assert [language for _audio, language in fake_model.calls] == ["es", None]
