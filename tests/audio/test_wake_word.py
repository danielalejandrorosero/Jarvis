"""Tests para el núcleo puro de detección de wake word (`detect`).

No tocan el micrófono real (`iter_microphone_frames`) ni cargan el modelo ONNX real
(`load_model`): `detect()` recibe un modelo de openwakeword stub, controlado por el test,
así que corren rápido, sin red y sin hardware de audio (CI-safe).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from openwakeword.model import Model

from jarvis.audio import wake_word
from jarvis.audio.wake_word import (
    FRAME_SAMPLES,
    WAKEWORD_NAMES,
    Detection,
    detect,
    load_model,
)


class _FakeModel:
    """Stub de `openwakeword.model.Model`: `.predict()` devuelve scores predefinidos."""

    def __init__(self, predictions: list[dict[str, float]]) -> None:
        self._predictions = iter(predictions)

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        return next(self._predictions)


def _silence_frames(count: int) -> Iterator[np.ndarray]:
    """Frames mono int16 de silencio: el contenido no importa porque el modelo es un stub."""
    return iter(np.zeros(FRAME_SAMPLES, dtype=np.int16) for _ in range(count))


def _model_with(predictions: list[dict[str, float]]) -> Model:
    # El stub no implementa la superficie completa de Model, solo lo que `detect()` usa
    # (`.predict()`). Se castea para satisfacer la firma de `detect(..., model: Model)`.
    return cast(Model, _FakeModel(predictions))


# --- load_model (no descarga ni carga modelos reales: Model se mockea) ------------------------


def test_load_model_passes_all_wakeword_names_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_model()` debe pedirle a openWakeWord los tres modelos de `WAKEWORD_NAMES` a la vez,
    no solo el primario — es lo que hace que cualquiera de las tres frases dispare a JARVIS."""
    mock_model_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(wake_word, "Model", mock_model_cls)

    load_model()

    mock_model_cls.assert_called_once_with(
        wakeword_models=WAKEWORD_NAMES, inference_framework="onnx"
    )


def test_detect_yields_detection_when_score_crosses_threshold() -> None:
    """Caso aceptado: un frame cuyo score cruza el umbral produce exactamente una Detection."""
    frames = _silence_frames(3)
    model = _model_with(
        [
            {"hey_jarvis": 0.1},
            {"hey_jarvis": 0.9},
            {"hey_jarvis": 0.2},
        ]
    )

    hits = list(detect(frames, model=model, threshold=0.5))

    assert len(hits) == 1
    hit = hits[0]
    assert hit.wakeword == "hey_jarvis"
    assert hit.score == 0.9
    assert isinstance(hit, Detection)
    assert isinstance(hit.timestamp, dt.datetime)


def test_detect_yields_nothing_when_score_stays_below_threshold() -> None:
    """Caso rechazado: si ningún score cruza el umbral, no se emite ninguna Detection."""
    frames = _silence_frames(5)
    model = _model_with([{"hey_jarvis": 0.3} for _ in range(5)])

    hits = list(detect(frames, model=model, threshold=0.5))

    assert hits == []


def test_detect_yields_one_detection_per_crossing_without_debouncing() -> None:
    """Cruces múltiples producen múltiples Detection: `detect()` no deduplica ni debounce."""
    frames = _silence_frames(4)
    model = _model_with(
        [
            {"hey_jarvis": 0.9},
            {"hey_jarvis": 0.9},
            {"hey_jarvis": 0.1},
            {"hey_jarvis": 0.7},
        ]
    )

    hits = list(detect(frames, model=model, threshold=0.5))

    assert [hit.score for hit in hits] == [0.9, 0.9, 0.7]
    assert all(hit.wakeword == "hey_jarvis" for hit in hits)


def test_detect_uses_inclusive_threshold_comparison() -> None:
    """Un score exactamente igual al umbral cuenta como detección (`score >= threshold`)."""
    frames = _silence_frames(1)
    model = _model_with([{"hey_jarvis": 0.5}])

    hits = list(detect(frames, model=model, threshold=0.5))

    assert len(hits) == 1
    assert hits[0].score == 0.5


def test_detect_only_yields_keys_that_cross_threshold_in_multi_wakeword_frame() -> None:
    """Si `predict()` devuelve varias wakewords, solo las que cruzan el umbral se emiten."""
    frames = _silence_frames(1)
    model = _model_with([{"hey_jarvis": 0.9, "other_wakeword": 0.2}])

    hits = list(detect(frames, model=model, threshold=0.5))

    assert len(hits) == 1
    assert hits[0].wakeword == "hey_jarvis"


def test_detect_on_empty_frame_stream_yields_nothing() -> None:
    """Un stream de frames vacío no debe llamar a predict() ni emitir detecciones."""
    frames: Iterator[np.ndarray] = iter([])
    model = _model_with([])

    hits = list(detect(frames, model=model, threshold=0.5))

    assert hits == []


# --- system_audio (gate de falsos triggers por audio fuerte del sistema, ver `loopback.py`) --


class _FakeSystemAudioGate:
    """Stub de `loopback.SystemAudioGate`: `is_loud()` devuelve un valor fijo, sin abrir ningún
    stream real — cumple el contrato mínimo que `detect()` necesita."""

    def __init__(self, *, loud: bool) -> None:
        self._loud = loud

    def is_loud(self) -> bool:
        return self._loud


def test_detect_suppresses_detection_while_system_audio_is_loud() -> None:
    """Un score que normalmente cruzaría el umbral no produce Detection mientras
    `system_audio.is_loud()` sea True — el caso central de este cambio."""
    frames = _silence_frames(1)
    model = _model_with([{"hey_jarvis": 0.9}])

    hits = list(
        detect(
            frames,
            model=model,
            threshold=0.5,
            system_audio=_FakeSystemAudioGate(loud=True),
        )
    )

    assert hits == []


def test_detect_still_yields_detection_when_system_audio_is_not_loud() -> None:
    """Con el gate presente pero `is_loud() == False`, el comportamiento es el de siempre."""
    frames = _silence_frames(1)
    model = _model_with([{"hey_jarvis": 0.9}])

    hits = list(
        detect(
            frames,
            model=model,
            threshold=0.5,
            system_audio=_FakeSystemAudioGate(loud=False),
        )
    )

    assert len(hits) == 1
    assert hits[0].score == 0.9


def test_detect_calls_predict_even_on_frames_suppressed_by_system_audio() -> None:
    """`model.predict()` se llama en cada frame sin importar el gate: openWakeWord mantiene
    estado interno entre llamadas, así que saltear la llamada rompería las predicciones
    siguientes (ver docstring de `detect`). Un `_FakeModel` con menos predicciones que frames
    haría `StopIteration` si `predict()` no se llamara para el frame gateado."""
    frames = _silence_frames(2)
    model = _model_with([{"hey_jarvis": 0.9}, {"hey_jarvis": 0.9}])

    hits = list(
        detect(
            frames,
            model=model,
            threshold=0.5,
            system_audio=_FakeSystemAudioGate(loud=True),
        )
    )

    assert hits == []


# --- speech_detector (gate de falsos triggers por audio de sistema NO fuerte que igual se cuela
# al mic, ej. headset combinado con `SystemAudioMonitor` desactivado — ver docstring de `detect`) -


class _FakeSpeechDetector:
    """Stub de `wake_word.SpeechDetector`: `speech_probability()` devuelve un valor fijo, sin
    cargar ningún modelo Silero real — cumple el contrato mínimo que `detect()` necesita."""

    def __init__(self, *, probability: float) -> None:
        self._probability = probability
        self.calls = 0

    def speech_probability(self, chunk_int16: np.ndarray) -> float:
        self.calls += 1
        return self._probability


def test_detect_suppresses_detection_when_speech_probability_is_low() -> None:
    """Un score que normalmente cruzaría el umbral no produce Detection si `speech_detector`
    reporta una probabilidad de voz por debajo de `speech_probability_threshold` — el caso real
    confirmado en vivo (`data/jarvis-error.log`): audio de fondo (música/video) filtrándose al mic
    de un headset combinado, con el gate de `system_audio` desactivado para ese caso."""
    frames = _silence_frames(1)
    model = _model_with([{"hey_jarvis": 0.9}])

    hits = list(
        detect(
            frames,
            model=model,
            threshold=0.5,
            speech_detector=_FakeSpeechDetector(probability=0.1),
            speech_probability_threshold=0.5,
        )
    )

    assert hits == []


def test_detect_still_yields_detection_when_speech_probability_is_high() -> None:
    """Con el detector presente pero reportando probabilidad alta (voz real), el comportamiento
    es el de siempre."""
    frames = _silence_frames(1)
    model = _model_with([{"hey_jarvis": 0.9}])

    hits = list(
        detect(
            frames,
            model=model,
            threshold=0.5,
            speech_detector=_FakeSpeechDetector(probability=0.9),
            speech_probability_threshold=0.5,
        )
    )

    assert len(hits) == 1
    assert hits[0].score == 0.9


def test_detect_without_speech_detector_behaves_as_before() -> None:
    """Sin `speech_detector` (default `None`), el comportamiento es idéntico al de antes de este
    parámetro — una mejora opcional nunca tumba la detección si no se inyecta."""
    frames = _silence_frames(1)
    model = _model_with([{"hey_jarvis": 0.9}])

    hits = list(detect(frames, model=model, threshold=0.5))

    assert len(hits) == 1


def test_detect_calls_speech_probability_on_every_frame_regardless_of_gates() -> None:
    """`speech_probability()` se llama en cada frame sin importar el resultado de las otras
    gates (`system_audio`), mismo motivo que `model.predict()`: mantiene el buffer interno del
    detector sincronizado con el audio real en vez de desalinearlo salteando frames."""
    frames = _silence_frames(2)
    model = _model_with([{"hey_jarvis": 0.9}, {"hey_jarvis": 0.9}])
    fake_detector = _FakeSpeechDetector(probability=0.9)

    hits = list(
        detect(
            frames,
            model=model,
            threshold=0.5,
            system_audio=_FakeSystemAudioGate(loud=True),
            speech_detector=fake_detector,
        )
    )

    assert hits == []
    assert fake_detector.calls == 2
