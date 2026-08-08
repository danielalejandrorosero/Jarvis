"""Tests para `jarvis.audio.loopback.SystemAudioMonitor`.

No abren un stream WASAPI/COM real: tanto la resolución de device (`sounddevice`) como el
loopback en sí (`soundcard`) se reemplazan por stubs controlados por el test (mismo enfoque que
`tests/audio/test_pipeline.py` usa para `sd.rec` en `measure_noise_floor`), así que corren
rápido, sin hardware de audio, CI-safe. La única pieza que sí corre real es la inicialización de
COM del thread de fondo (`_init_com_for_this_thread`, sin mockear en los tests de lifecycle): es
una llamada de sistema barata, sin I/O de audio, y correrla real ejercita el mismo camino que
producción — la excepción es la sección dedicada a probar esa función en aislamiento, donde sí se
mockea `ctypes.windll.ole32` para cubrir las ramas de HRESULT sin depender del estado real de COM
del proceso de test.
"""

from __future__ import annotations

import time
import types
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


def test_enabled_false_disables_gate_without_starting_thread() -> None:
    """`enabled=False` (usado por `pipeline.run()` cuando el mic y la salida son el mismo
    headset combinado, ver `device.is_combined_headset`) deja `is_loud()` siempre `False` sin
    abrir ningún stream real — confirma el bug en vivo de esta noche: con el gate activo de
    más, un headset bloqueaba toda detección de wake word aunque el usuario hablara."""
    monitor = SystemAudioMonitor(enabled=False)

    monitor.start()

    assert monitor.is_loud() is False
    assert monitor._thread is None  # nunca se lanza el thread de fondo

    monitor.stop()  # no-op, no debe lanzar aunque nunca haya arrancado


def _float32_chunk_for_int16_level(
    level: float, *, frames: int = 100, channels: int = 1
) -> np.ndarray:
    """Chunk float32 en [-1.0, 1.0] (formato que devuelve `soundcard.Recorder.record`) cuyo RMS,
    tras escalar a int16 (`loopback._to_int16_scale`, lo que hace `_run_with_com_initialized`
    antes de comparar contra el threshold), da exactamente `level`."""
    return np.full((frames, channels), level / loopback._INT16_SCALE, dtype=np.float32)


class _FakeRecorder:
    """Stub del context manager que devuelve `soundcard`'s `_Microphone.recorder(...)`: `.record()`
    devuelve chunks de una lista fija, en orden; agotada la lista, repite el último para siempre
    (el test corta el thread con `.stop()` antes de que eso importe)."""

    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = list(chunks)
        self._index = 0
        self.entered = False

    def __enter__(self) -> Self:
        self.entered = True
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def record(self, numframes: int) -> np.ndarray:
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
        elif self._chunks:
            chunk = self._chunks[-1]
        else:
            chunk = np.zeros((numframes, 1), dtype=np.float32)
        return chunk


class _FakeLoopbackMic:
    """Stub de lo que devuelve `soundcard.get_microphone(..., include_loopback=True)`."""

    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = chunks
        self.recorder_kwargs: dict[str, object] | None = None

    def recorder(self, **kwargs: object) -> _FakeRecorder:
        self.recorder_kwargs = kwargs
        return _FakeRecorder(self._chunks)


def _patch_device_resolution(
    monkeypatch: pytest.MonkeyPatch, *, device_sr: int = 48_000
) -> None:
    monkeypatch.setattr(loopback, "resolve_output_device", lambda device: 3)
    monkeypatch.setattr(loopback, "output_sample_rate", lambda device: device_sr)
    monkeypatch.setattr(
        loopback.sd, "query_devices", lambda device: {"name": "Fake Output Device"}
    )


def _patch_recorder(
    monkeypatch: pytest.MonkeyPatch, chunks: list[np.ndarray]
) -> _FakeLoopbackMic:
    mic = _FakeLoopbackMic(chunks)
    monkeypatch.setattr(
        loopback.sc, "get_microphone", lambda *, id, include_loopback: mic
    )
    return mic


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


# --- lifecycle: start()/stop() con `soundcard` stubeado ---------------------------------------


def test_start_updates_level_from_stream_and_is_loud_reflects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camino feliz: el thread de fondo lee chunks del recorder y `is_loud()` termina reflejando
    un chunk fuerte."""
    _patch_device_resolution(monkeypatch)
    quiet = _float32_chunk_for_int16_level(10.0)
    loud = _float32_chunk_for_int16_level(5000.0)
    _patch_recorder(monkeypatch, [quiet, quiet, loud])
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
    _patch_recorder(monkeypatch, [np.zeros((100, 1), dtype=np.float32)])
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    monitor.stop()
    monitor.stop()  # segunda llamada no debe lanzar ni bloquear

    assert monitor._thread is None


def test_start_is_idempotent_does_not_spawn_a_second_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_device_resolution(monkeypatch)
    _patch_recorder(monkeypatch, [np.zeros((100, 1), dtype=np.float32)])
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


def test_disables_gracefully_when_recorder_fails_to_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El recorder de loopback puede fallar al abrirse (p.ej. conflicto de modo exclusivo,
    device desconectado) — `soundcard` reporta estos fallos como `RuntimeError` (ver
    `soundcard.mediafoundation._handle_error`); ese `RuntimeError` se atrapa y deshabilita el
    monitor en vez de tumbar el thread con una excepción no manejada."""
    _patch_device_resolution(monkeypatch)

    class _FailingRecorder:
        def __enter__(self) -> Self:
            raise RuntimeError(
                "Error 0x88890008"
            )  # AUDCLNT_E_DEVICE_IN_USE, ejemplo real

        def __exit__(self, *_exc: object) -> bool:
            return False

    class _FailingMic:
        def recorder(self, **_kwargs: object) -> _FailingRecorder:
            return _FailingRecorder()

    monkeypatch.setattr(
        loopback.sc, "get_microphone", lambda *, id, include_loopback: _FailingMic()
    )
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    assert _poll_until(lambda: monitor._disabled)
    monitor.stop()

    assert monitor.is_loud() is False


def test_disables_gracefully_when_no_matching_loopback_device_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`soundcard.get_microphone(id=..., include_loopback=True)` levanta `IndexError` si ningún
    device de loopback matchea el nombre resuelto por `sounddevice` (p.ej. el device default
    cambió entre la resolución y la búsqueda, o el nombre truncado por MME no matchea nada) —
    debe degradar igual que cualquier otro fallo al abrir el loopback, no tumbar el thread."""
    _patch_device_resolution(monkeypatch)

    def _raise_index_error(*, id: str, include_loopback: bool) -> Any:
        raise IndexError(f"no device with id {id!r}")

    monkeypatch.setattr(loopback.sc, "get_microphone", _raise_index_error)
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    assert _poll_until(lambda: monitor._disabled)
    monitor.stop()

    assert monitor.is_loud() is False


def test_disables_gracefully_when_com_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regresión de un bug real encontrado en vivo (no en logs previos — en la verificación
    manual de este mismo fix): `soundcard` habla COM, que Windows inicializa por thread, no una
    sola vez por proceso. `import soundcard` inicializa COM en el thread que hizo el import
    (normalmente el principal), pero NO en el thread de fondo propio de `SystemAudioMonitor` —
    sin `_init_com_for_this_thread()` ahí, cada llamada a `soundcard` desde `_run()` fallaba con
    `RuntimeError('Error 0x800401f0')` (`CO_E_NOTINITIALIZED`), con el monitor deshabilitándose
    en cada corrida pese a que la captura en sí ya funcionaba. Acá se simula que la
    inicialización de COM en sí falla, y el monitor debe degradarse igual que ante cualquier otro
    fallo, no tumbar el thread con una excepción no atrapada."""

    def _raise_oserror() -> bool:
        raise OSError("CoInitializeEx falló con HRESULT 0x8000ffff")

    monkeypatch.setattr(loopback, "_init_com_for_this_thread", _raise_oserror)
    monitor = SystemAudioMonitor(chunk_seconds=0.01)

    monitor.start()
    assert _poll_until(lambda: monitor._disabled)
    monitor.stop()

    assert monitor.is_loud() is False


# --- `_init_com_for_this_thread` en aislamiento (ramas de HRESULT) ----------------------------


class _FakeOle32:
    def __init__(self, hresult: int) -> None:
        self.hresult = hresult
        self.co_initialize_calls = 0

    def CoInitializeEx(self, _reserved: object, _coinit: int) -> int:
        self.co_initialize_calls += 1
        return self.hresult

    def CoUninitialize(self) -> None:
        pass


def _patch_ole32(monkeypatch: pytest.MonkeyPatch, hresult: int) -> _FakeOle32:
    fake_ole32 = _FakeOle32(hresult)
    monkeypatch.setattr(
        loopback.ctypes, "windll", types.SimpleNamespace(ole32=fake_ole32)
    )
    return fake_ole32


def test_init_com_for_this_thread_returns_true_on_s_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ole32(monkeypatch, 0x0)

    assert loopback._init_com_for_this_thread() is True


def test_init_com_for_this_thread_returns_true_on_s_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`S_FALSE` (1): COM ya estaba inicializado en este thread con el mismo modelo — sigue
    siendo "nosotros lo inicializamos" a efectos de este contrato (`True`), ver docstring de
    `_init_com_for_this_thread`."""
    _patch_ole32(monkeypatch, 0x1)

    assert loopback._init_com_for_this_thread() is True


def test_init_com_for_this_thread_returns_false_on_rpc_e_changed_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COM ya estaba inicializado en este thread con OTRO modelo de concurrencia por otra
    librería — usable, pero no lo inicializamos nosotros, así que no hay que desinicializarlo."""
    _patch_ole32(monkeypatch, 0x80010106)

    assert loopback._init_com_for_this_thread() is False


def test_init_com_for_this_thread_raises_oserror_on_unexpected_hresult(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ole32(monkeypatch, 0x8000FFFF)  # E_UNEXPECTED, HRESULT de falla arbitrario

    with pytest.raises(OSError):
        loopback._init_com_for_this_thread()


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
