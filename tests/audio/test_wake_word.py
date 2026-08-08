"""Tests para el núcleo puro de detección de wake word (`detect`).

No tocan el micrófono real (`iter_microphone_frames`) ni cargan el modelo ONNX real
(`load_model`): `detect()` recibe un modelo de openwakeword stub, controlado por el test,
así que corren rápido, sin red y sin hardware de audio (CI-safe).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import cast

import numpy as np
from openwakeword.model import Model

from jarvis.audio.wake_word import FRAME_SAMPLES, Detection, detect


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
