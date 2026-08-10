"""Tests para `jarvis.audio.speech_detector`.

El modelo real de Silero NO se prueba acá (necesitaría audio de voz real para tener sentido,
igual que este repo no testea la calidad de `gpt-4o-transcribe`/`gpt-live-transcribe`) — se
monkeypatchea `SileroVAD.process` con un stub controlado y se verifica solo la lógica propia de
`ChunkSpeechDetector`: el buffering de chunks de tamaño arbitrario en ventanas fijas de
`WINDOW_SAMPLES`, la agregación (máximo) cuando un chunk cubre más de una ventana, qué pasa
cuando un chunk no llega a completar ninguna, y la degradación de `load_speech_detector` cuando
el modelo no carga.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from jarvis.audio import speech_detector as speech_detector_module
from jarvis.audio.speech_detector import (
    WINDOW_SAMPLES,
    ChunkSpeechDetector,
    load_speech_detector,
)


class _FakeSileroVAD:
    """Doble de `silero_vad_lite.SileroVAD`: en vez de correr el modelo real, devuelve
    probabilidades pre-armadas en orden, una por cada llamada a `.process()` — permite controlar
    exactamente qué "escucha" el detector en cada ventana de 512 samples."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = iter(probabilities)
        self.calls: list[Any] = []

    def process(self, data: Any) -> float:
        self.calls.append(data)
        return next(self._probabilities)


def _install_fake_vad(
    monkeypatch: pytest.MonkeyPatch, probabilities: list[float]
) -> _FakeSileroVAD:
    fake = _FakeSileroVAD(probabilities)
    monkeypatch.setattr(speech_detector_module, "SileroVAD", lambda sample_rate: fake)
    return fake


def _silence_chunk(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.int16)


# --- ChunkSpeechDetector.speech_probability ----------------------------------------------------


def test_speech_probability_returns_zero_when_chunk_smaller_than_one_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un chunk que no llega a juntar ni una ventana de `WINDOW_SAMPLES` no dispara ninguna
    llamada al modelo — se guarda en el buffer para la próxima vez, no es "silencio confirmado"."""
    fake = _install_fake_vad(monkeypatch, [])
    detector = ChunkSpeechDetector()

    result = detector.speech_probability(_silence_chunk(WINDOW_SAMPLES - 1))

    assert result == 0.0
    assert fake.calls == []


def test_speech_probability_processes_exactly_one_full_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_fake_vad(monkeypatch, [0.73])
    detector = ChunkSpeechDetector()

    result = detector.speech_probability(_silence_chunk(WINDOW_SAMPLES))

    assert result == pytest.approx(0.73)
    assert len(fake.calls) == 1


def test_speech_probability_carries_leftover_samples_to_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un chunk típico del pipeline (~3200 samples) no es múltiplo exacto de `WINDOW_SAMPLES`
    (512) — el resto que no completa una ventana se guarda y se suma al próximo chunk, no se
    descarta."""
    fake = _install_fake_vad(monkeypatch, [0.1, 0.2])
    detector = ChunkSpeechDetector()

    # Primer chunk: una ventana completa + 100 samples sobrantes.
    first = detector.speech_probability(_silence_chunk(WINDOW_SAMPLES + 100))
    # Segundo chunk: 100 (sobrante anterior) + este chunk deben juntar otra ventana completa.
    second = detector.speech_probability(_silence_chunk(WINDOW_SAMPLES - 100))

    assert first == pytest.approx(0.1)
    assert second == pytest.approx(0.2)
    assert len(fake.calls) == 2


def test_speech_probability_returns_max_across_multiple_windows_in_one_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un chunk que cubre varias ventanas devuelve el MÁXIMO entre ellas, no el promedio ni la
    última — una ventana con voz real no debe diluirse entre ventanas de silencio de transición
    dentro del mismo chunk (ver docstring de `speech_probability`)."""
    fake = _install_fake_vad(monkeypatch, [0.1, 0.9, 0.2])
    detector = ChunkSpeechDetector()

    result = detector.speech_probability(_silence_chunk(WINDOW_SAMPLES * 3))

    assert result == pytest.approx(0.9)
    assert len(fake.calls) == 3


def test_speech_probability_result_is_a_plain_python_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`SileroVAD.process` (sin stubs de tipos) podría devolver un `numpy.float32` u otro tipo
    numérico — se normaliza a `float` de Python antes de devolver, así los llamadores (`vad.
    is_speech_chunk`) no tienen que lidiar con tipos numéricos ajenos."""
    _install_fake_vad(monkeypatch, [np.float32(0.5)])  # type: ignore[list-item]
    detector = ChunkSpeechDetector()

    result = detector.speech_probability(_silence_chunk(WINDOW_SAMPLES))

    assert type(result) is float


# --- load_speech_detector -----------------------------------------------------------------------


def test_load_speech_detector_returns_a_working_detector_when_model_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vad(monkeypatch, [0.42])

    detector = load_speech_detector()

    assert detector is not None
    assert detector.speech_probability(_silence_chunk(WINDOW_SAMPLES)) == pytest.approx(
        0.42
    )


def test_load_speech_detector_degrades_to_none_when_model_fails_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si `SileroVAD(...)` lanza (DLL/modelo ONNX faltante, lo que sea), `load_speech_detector`
    nunca propaga — degrada a `None` para que los llamadores (`pipeline.record_command`/
    `realtime_stt._capture_and_stream`) sigan funcionando con RMS solamente."""

    def _raise(sample_rate: int) -> None:
        raise RuntimeError("modelo no disponible")

    monkeypatch.setattr(speech_detector_module, "SileroVAD", _raise)

    detector = load_speech_detector()

    assert detector is None
