"""Tests para `jarvis.audio.loopback.SystemAudioMonitor`.

No abren un stream WASAPI real: `sd.InputStream` se reemplaza por un stub controlado por el
test (mismo enfoque que `tests/audio/test_pipeline.py` usa para `sd.rec` en
`measure_noise_floor`), así que corren rápido, sin hardware de audio, CI-safe.
"""

from __future__ import annotations

import time
from typing import Any, Self

import numpy as np
import pytest

from jarvis.audio import loopback
from jarvis.audio.loopback import DEFAULT_LOUDNESS_THRESHOLD, SystemAudioMonitor

POLL_TIMEOUT_SECONDS = 2.0
POLL_INTERVAL_SECONDS = 0.01


def _poll_until(predicate: object, *, timeout: float = POLL_TIMEOUT_SECONDS) -> bool:
    """Esperar hasta que `predicate()` sea verdadero o pasen `timeout` segundos — evita un
    `sleep` fijo (flaky por definición) para sincronizar con el thread de fondo."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return bool(predicate())  # type: ignore[operator]


class _FakeInputStream:
    """Stub de `sd.InputStream`: `.read()` devuelve chunks de una lista fija, en orden; agotada
    la lista, repite el último para siempre (el test corta el thread con `.stop()` antes de que
    eso importe)."""

    def __init__(
        self, *_args: object, chunks: list[np.ndarray], **_kwargs: object
    ) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self.entered = False

    def __enter__(self) -> Self:
        self.entered = True
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
        elif self._chunks:
            chunk = self._chunks[-1]
        else:
            chunk = np.zeros(frames, dtype=np.int16)
        return chunk, False


def _patch_device_resolution(
    monkeypatch: pytest.MonkeyPatch, *, device_sr: int = 48_000, channels: int = 2
) -> None:
    monkeypatch.setattr(loopback, "resolve_output_device", lambda device: 3)
    monkeypatch.setattr(loopback, "output_sample_rate", lambda device: device_sr)
    monkeypatch.setattr(
        loopback.sd,
        "query_devices",
        lambda device: {"max_output_channels": channels},
    )
    monkeypatch.setattr(loopback.sd, "WasapiSettings", lambda **kwargs: kwargs)


def _patch_input_stream(
    monkeypatch: pytest.MonkeyPatch, chunks: list[np.ndarray]
) -> None:
    def _make_stream(*args: object, **kwargs: object) -> _FakeInputStream:
        return _FakeInputStream(*args, chunks=chunks, **kwargs)

    monkeypatch.setattr(loopback.sd, "InputStream", _make_stream)


# --- is_loud / current_level (lógica pura, sin thread) ---------------------------------------


def test_is_loud_true_when_level_at_or_above_threshold() -> None:
    """Umbral inclusivo: un nivel igual al threshold cuenta como "fuerte" (mismo criterio que
    `wake_word.detect()` usa para su propio umbral, `score >= threshold`)."""
    monitor = SystemAudioMonitor(threshold=500.0)
    with monitor._lock:
        monitor._level = 500.0

    assert monitor.is_loud() is True


def test_is_loud_false_when_level_below_threshold() -> None:
    monitor = SystemAudioMonitor(threshold=500.0)
    with monitor._lock:
        monitor._level = 499.0

    assert monitor.is_loud() is False


def test_is_loud_always_false_when_disabled_regardless_of_level() -> None:
    """Un monitor deshabilitado (no pudo abrir el loopback) nunca reporta "fuerte", ni con un
    nivel que en otras condiciones cruzaría el umbral — degradar en silencio, no en falso
    positivo."""
    monitor = SystemAudioMonitor(threshold=10.0)
    with monitor._lock:
        monitor._level = 999_999.0
    monitor._disabled = True

    assert monitor.is_loud() is False


def test_current_level_property_reflects_last_written_value() -> None:
    monitor = SystemAudioMonitor()
    with monitor._lock:
        monitor._level = 1234.5

    assert monitor.current_level == pytest.approx(1234.5)


def test_default_loudness_threshold_is_a_positive_finite_value() -> None:
    """Guarda contra un default accidentalmente roto (0, negativo, NaN) que dejaría el gate
    siempre activo o siempre inactivo."""
    assert 0.0 < DEFAULT_LOUDNESS_THRESHOLD < 32_768.0


# --- lifecycle: start()/stop() con InputStream stubeado ---------------------------------------


def test_start_updates_level_from_stream_and_is_loud_reflects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camino feliz: el thread de fondo lee chunks del stream y `is_loud()` termina reflejando
    un chunk fuerte."""
    _patch_device_resolution(monkeypatch)
    quiet = np.full(100, 10, dtype=np.int16)
    loud = np.full(100, 5000, dtype=np.int16)
    _patch_input_stream(monkeypatch, [quiet, quiet, loud])
    monitor = SystemAudioMonitor(threshold=500.0, chunk_seconds=0.01)

    monitor.start()
    try:
        assert _poll_until(monitor.is_loud)
    finally:
        monitor.stop()

    assert monitor._thread is None


def test_stop_is_idempotent_and_joins_the_background_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_device_resolution(monkeypatch)
    _patch_input_stream(monkeypatch, [np.zeros(100, dtype=np.int16)])
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    monitor.stop()
    monitor.stop()  # segunda llamada no debe lanzar ni bloquear

    assert monitor._thread is None


def test_start_is_idempotent_does_not_spawn_a_second_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_device_resolution(monkeypatch)
    _patch_input_stream(monkeypatch, [np.zeros(100, dtype=np.int16)])
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    first_thread = monitor._thread
    monitor.start()

    assert monitor._thread is first_thread
    monitor.stop()


# --- degradación ante fallos ------------------------------------------------------------------


def test_disables_gracefully_when_no_default_output_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin dispositivo de salida default, el monitor se deshabilita en vez de propagar la
    excepción al thread principal — `is_loud()` queda en False para siempre."""

    def _raise(device: int | None) -> int:
        raise RuntimeError("No hay dispositivo de salida de audio default")

    monkeypatch.setattr(loopback, "resolve_output_device", _raise)
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    assert _poll_until(lambda: monitor._disabled)
    monitor.stop()

    assert monitor.is_loud() is False


def test_disables_gracefully_when_input_stream_fails_to_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El stream de loopback puede fallar al abrirse (p.ej. conflicto de modo exclusivo) —
    `PortAudioError` se atrapa y deshabilita el monitor en vez de tumbar el thread con una
    excepción no manejada."""
    _patch_device_resolution(monkeypatch)

    def _raise_port_audio_error(*_args: object, **_kwargs: object) -> Any:
        raise loopback.sd.PortAudioError("dispositivo ocupado en modo exclusivo")

    monkeypatch.setattr(loopback.sd, "InputStream", _raise_port_audio_error)
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    assert _poll_until(lambda: monitor._disabled)
    monitor.stop()

    assert monitor.is_loud() is False


def test_disables_gracefully_when_wasapi_settings_rejects_loopback_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regresión de un bug real (no cubierto por `_patch_device_resolution`, que stubea
    `sd.WasapiSettings` como un `lambda **kwargs: kwargs` que acepta cualquier cosa): la build de
    `sounddevice` instalada puede no exponer el kwarg `loopback` en
    `WasapiSettings.__init__` — confirmado en vivo (`sounddevice==0.5.5`, ver
    `data/jarvis-error.log`: "TypeError: WasapiSettings.__init__() got an unexpected keyword
    argument 'loopback'", sin atrapar, tumbando el thread de fondo entero). Antes de este fix,
    ese `TypeError` no lo atrapaba nada en `_run()` (solo se atrapaba `sd.PortAudioError`) — acá
    se simula esa construcción real fallando, y el monitor debe degradarse igual que ante un
    `PortAudioError` al abrir el stream, no crashear el thread."""
    monkeypatch.setattr(loopback, "resolve_output_device", lambda device: 3)
    monkeypatch.setattr(loopback, "output_sample_rate", lambda device: 48_000)
    monkeypatch.setattr(
        loopback.sd, "query_devices", lambda device: {"max_output_channels": 2}
    )

    def _raise_type_error(**_kwargs: object) -> Any:
        raise TypeError(
            "WasapiSettings.__init__() got an unexpected keyword argument 'loopback'"
        )

    monkeypatch.setattr(loopback.sd, "WasapiSettings", _raise_type_error)
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    assert _poll_until(lambda: monitor._disabled)
    monitor.stop()

    assert monitor.is_loud() is False


# --- thread-safety -----------------------------------------------------------------------------


def test_concurrent_reads_during_writes_never_raise_or_see_a_torn_value() -> None:
    """El nivel se lee/escribe siempre bajo el mismo lock — un lector concurrente nunca ve una
    excepción ni un valor a medio escribir (acá "torn" no aplica a un float de Python, pero el
    lock es lo que garantiza que no haya una carrera de datos real)."""
    import threading

    monitor = SystemAudioMonitor(threshold=100.0)
    stop = threading.Event()
    errors: list[BaseException] = []

    def _writer() -> None:
        value = 0.0
        while not stop.is_set():
            with monitor._lock:
                monitor._level = value
            value = (value + 1.0) % 1000.0

    def _reader() -> None:
        try:
            for _ in range(2_000):
                monitor.is_loud()
                _ = monitor.current_level
        except BaseException as exc:  # noqa: BLE001 — se re-reporta explícitamente al test
            errors.append(exc)

    writer = threading.Thread(target=_writer, daemon=True)
    readers = [threading.Thread(target=_reader) for _ in range(4)]
    writer.start()
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join(timeout=5.0)
    stop.set()
    writer.join(timeout=1.0)

    assert errors == []
