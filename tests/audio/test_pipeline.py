"""Tests para el núcleo puro del pipeline de voz (`jarvis.audio.pipeline`).

Cubre las funciones que no dependen de hardware/red (`chunk_rms`, `calibrate_thresholds`,
`normalize_gain`, `should_stop_recording`, `tee_frames`) sin mocks, y `measure_noise_floor` —
la única con I/O real (`sd.rec`/`sd.wait`, resolución de dispositivo) — mockeando esos puntos de
entrada explícitamente (mismo enfoque que el resto de `tests/audio/`: núcleo puro con
dependencias externas stubeadas, CI-safe, sin hardware ni red).

`record_command()` y `run()` no se testean acá: son wrappers finos de integración sobre
`sd.InputStream`/modelos reales sin lógica propia además de la ya cubierta por las funciones de
este archivo (mismo criterio que el repo ya aplica a funciones con forma de
`iter_microphone_frames`/`record_command`: integración, no unit test).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from jarvis.audio import pipeline
from jarvis.audio.pipeline import (
    MIN_SILENCE_RMS_THRESHOLD,
    NOISE_FLOOR_MULTIPLIER,
    NOISE_FLOOR_SUBCHUNKS,
    SAMPLE_RATE,
    TRAILING_SILENCE_SECONDS,
    calibrate_thresholds,
    chunk_rms,
    measure_noise_floor,
    normalize_gain,
    should_stop_recording,
    tee_frames,
)

# --- chunk_rms -----------------------------------------------------------------------------


def test_chunk_rms_of_all_zero_chunk_is_zero() -> None:
    """Silencio puro (todo ceros) da RMS 0."""
    chunk = np.zeros(100, dtype=np.int16)

    assert chunk_rms(chunk) == 0.0


def test_chunk_rms_of_constant_chunk_equals_its_absolute_value() -> None:
    """Un chunk de valor constante v tiene RMS == |v| (sqrt(mean(v^2)) == |v|)."""
    chunk = np.full(50, 1000, dtype=np.int16)

    assert chunk_rms(chunk) == pytest.approx(1000.0)


def test_chunk_rms_of_symmetric_alternating_signal_ignores_sign() -> None:
    """Una señal alternando +v/-v da RMS == v: el cuadrado borra el signo."""
    chunk = np.array([100, -100, 100, -100], dtype=np.int16)

    assert chunk_rms(chunk) == pytest.approx(100.0)


# --- calibrate_thresholds -------------------------------------------------------------------


def test_calibrate_thresholds_floor_wins_when_noise_floor_is_low() -> None:
    """Con un ambiente muy silencioso, el piso absoluto (`MIN_SILENCE_RMS_THRESHOLD`) gana
    sobre el múltiplo del piso de ruido — nunca se calibra por debajo de ese mínimo."""
    noise_floor = (
        5.0  # 5 * NOISE_FLOOR_MULTIPLIER (4.0) == 20, por debajo del piso de 40
    )

    result = calibrate_thresholds(noise_floor)

    assert result == MIN_SILENCE_RMS_THRESHOLD


def test_calibrate_thresholds_multiplier_wins_when_noise_floor_is_high() -> None:
    """Con un ambiente ruidoso, el umbral escala con el piso de ruido medido, no se queda
    pegado al mínimo absoluto."""
    noise_floor = (
        20.0  # 20 * NOISE_FLOOR_MULTIPLIER (4.0) == 80, por encima del piso de 40
    )

    result = calibrate_thresholds(noise_floor)

    assert result == pytest.approx(noise_floor * NOISE_FLOOR_MULTIPLIER)
    assert result > MIN_SILENCE_RMS_THRESHOLD


def test_calibrate_thresholds_at_the_tie_point_returns_the_floor() -> None:
    """Caso límite: cuando el múltiplo coincide exactamente con el piso, el resultado es ese
    valor (max() es determinista en el empate, no importa cuál "gana")."""
    noise_floor = MIN_SILENCE_RMS_THRESHOLD / NOISE_FLOOR_MULTIPLIER

    result = calibrate_thresholds(noise_floor)

    assert result == pytest.approx(MIN_SILENCE_RMS_THRESHOLD)


# --- normalize_gain --------------------------------------------------------------------------


def test_normalize_gain_boosts_quiet_but_real_audio_toward_target_peak() -> None:
    """Audio bajito pero por encima de `min_peak` se sube a ~90% del rango de int16."""
    audio = np.array([1000, -500, 200], dtype=np.int16)

    result = normalize_gain(audio, min_peak=40.0)

    target = int(0.9 * 32767)
    assert int(np.abs(result).max()) == target
    # La proporción entre samples se mantiene (mismo gain aplicado a todo el array).
    assert result[0] == pytest.approx(-2 * result[1], abs=1)


def test_normalize_gain_leaves_already_loud_audio_unchanged() -> None:
    """Audio cuyo pico ya alcanza (o supera) el target no se toca — evita clipping doble."""
    target = int(0.9 * 32767)
    audio = np.array([target, -target, 100], dtype=np.int16)

    result = normalize_gain(audio, min_peak=40.0)

    assert np.array_equal(result, audio)


def test_normalize_gain_leaves_loud_clipping_audio_unchanged() -> None:
    """Audio con pico por encima del target (ya fuerte de por sí) tampoco se modifica."""
    audio = np.array([32767, -32768, 0], dtype=np.int16)

    result = normalize_gain(audio, min_peak=40.0)

    assert np.array_equal(result, audio)


def test_normalize_gain_leaves_below_min_peak_audio_unchanged() -> None:
    """Audio por debajo de `min_peak` (piso de ruido calibrado) no se amplifica — amplificarlo
    solo fabricaría señal falsa para Whisper."""
    audio = np.array([10, -5, 3], dtype=np.int16)

    result = normalize_gain(audio, min_peak=40.0)

    assert np.array_equal(result, audio)


# --- should_stop_recording -------------------------------------------------------------------


def test_should_stop_recording_hard_cap_stops_regardless_of_speech_started() -> None:
    """El tope duro por tiempo corta la grabación incluso si nunca se detectó habla."""
    assert should_stop_recording(
        speech_started=False,
        silence_run_seconds=0.0,
        elapsed_seconds=4.0,
        max_seconds=4.0,
    )


def test_should_stop_recording_does_not_cut_on_silence_before_speech_started() -> None:
    """Sin habla detectada todavía, ni un silence_run enorme corta antes del tope duro — el
    usuario puede tardar en arrancar a hablar después de la wake word."""
    assert not should_stop_recording(
        speech_started=False,
        silence_run_seconds=100.0,
        elapsed_seconds=1.0,
        max_seconds=4.0,
    )


def test_should_stop_recording_cuts_on_trailing_silence_once_speech_started() -> None:
    """Una vez que hubo habla real, un silencio sostenido >= TRAILING_SILENCE_SECONDS corta
    la grabación antes del tope duro."""
    assert should_stop_recording(
        speech_started=True,
        silence_run_seconds=TRAILING_SILENCE_SECONDS,
        elapsed_seconds=2.0,
        max_seconds=4.0,
    )


def test_should_stop_recording_does_not_cut_before_trailing_silence_threshold() -> None:
    """Silencio todavía por debajo de TRAILING_SILENCE_SECONDS no corta, aunque ya haya
    habido habla."""
    assert not should_stop_recording(
        speech_started=True,
        silence_run_seconds=TRAILING_SILENCE_SECONDS - 0.1,
        elapsed_seconds=2.0,
        max_seconds=4.0,
    )


# --- tee_frames -------------------------------------------------------------------------------


def test_tee_frames_yields_each_frame_unchanged() -> None:
    """Cada frame que entra sale idéntico (mismo objeto) por el generador."""
    frames = [np.array([i], dtype=np.int16) for i in range(5)]
    buffer: deque[np.ndarray] = deque(maxlen=10)

    result = list(tee_frames(iter(frames), buffer))

    assert len(result) == len(frames)
    assert all(out is orig for out, orig in zip(result, frames, strict=True))


def test_tee_frames_buffer_is_bounded_to_most_recent_frames() -> None:
    """El buffer (deque de tamaño fijo) solo retiene los últimos `maxlen` frames, descartando
    los más viejos a medida que entran nuevos."""
    frames = [np.array([i], dtype=np.int16) for i in range(5)]
    buffer: deque[np.ndarray] = deque(maxlen=3)

    list(tee_frames(iter(frames), buffer))

    assert len(buffer) == 3
    assert [int(f[0]) for f in buffer] == [2, 3, 4]


def test_tee_frames_on_empty_stream_leaves_buffer_empty() -> None:
    """Un stream de frames vacío no agrega nada al buffer."""
    buffer: deque[np.ndarray] = deque(maxlen=3)

    result = list(tee_frames(iter([]), buffer))

    assert result == []
    assert len(buffer) == 0


# --- measure_noise_floor ----------------------------------------------------------------------


def _fake_rec_returning(audio: np.ndarray) -> object:
    """Fabrica un stub de `sd.rec` que ignora sus argumentos y devuelve `audio` reshapeado
    como (samples, 1) — la forma que `sounddevice.rec(..., channels=1)` produce."""

    def _rec(*_args: object, **_kwargs: object) -> np.ndarray:
        return audio.reshape(-1, 1)

    return _rec


def test_measure_noise_floor_uses_median_of_subchunks_not_dragged_by_loud_outlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regresión (el motivo de este cambio, ver docstring del módulo): un único sub-chunk
    ruidoso en medio de la ventana medida no debe inflar el piso de ruido reportado — la
    mediana de los RMS por sub-chunk lo ignora mientras siga siendo un outlier aislado."""
    device_sr = SAMPLE_RATE  # evita que `resample()` interpole (orig_sr == target_sr)
    sample_seconds = 1.0
    samples = int(sample_seconds * device_sr)
    subchunk_len = samples // NOISE_FLOOR_SUBCHUNKS
    assert samples % NOISE_FLOOR_SUBCHUNKS == 0  # asegura splits parejos para el test

    quiet = np.full(subchunk_len, 50, dtype=np.int16)
    loud = np.full(subchunk_len, 20000, dtype=np.int16)
    # Un solo sub-chunk (el del medio) ruidoso, el resto silencioso.
    audio = np.concatenate([quiet, quiet, loud, quiet, quiet])
    assert len(audio) == samples

    monkeypatch.setattr(pipeline, "resolve_input_device", lambda device: 0)
    monkeypatch.setattr(pipeline, "input_sample_rate", lambda device: device_sr)
    monkeypatch.setattr(pipeline.sd, "rec", _fake_rec_returning(audio))
    monkeypatch.setattr(pipeline.sd, "wait", lambda: None)

    result = measure_noise_floor(device=None, sample_seconds=sample_seconds)

    # La mediana de [50, 50, 20000, 50, 50] es 50 — el outlier no arrastra el resultado.
    assert result == pytest.approx(50.0)
    # Contraste explícito: el RMS de la ventana entera sí estaría muy por encima de esto,
    # confirmando que el enfoque por sub-chunk+mediana realmente cambia el resultado.
    assert chunk_rms(audio) > result * 10


def test_measure_noise_floor_passes_resolved_device_and_sample_rate_to_sd_rec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`measure_noise_floor` graba a la sample rate nativa del dispositivo resuelto, no a un
    valor fijo — parámetros correctos pasados a `sd.rec`."""
    device_sr = 48000
    sample_seconds = 1.0
    samples = int(sample_seconds * device_sr)
    audio = np.zeros(samples, dtype=np.int16)
    calls: list[dict[str, object]] = []

    def _rec(count: int, **kwargs: object) -> np.ndarray:
        calls.append({"count": count, **kwargs})
        return audio.reshape(-1, 1)

    monkeypatch.setattr(pipeline, "resolve_input_device", lambda device: 7)
    monkeypatch.setattr(pipeline, "input_sample_rate", lambda device: device_sr)
    monkeypatch.setattr(pipeline.sd, "rec", _rec)
    monkeypatch.setattr(pipeline.sd, "wait", lambda: None)

    measure_noise_floor(device=None, sample_seconds=sample_seconds)

    assert len(calls) == 1
    assert calls[0]["count"] == samples
    assert calls[0]["samplerate"] == device_sr
    assert calls[0]["device"] == 7
