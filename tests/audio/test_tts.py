"""Tests para `jarvis.audio.tts`.

`OpenAITTSClient`/`SapiTTSClient` pegan contra red/hardware de audio real y no son testeables
como unit test. Lo que sí es testeable, sin red ni reproducir audio (CI-safe), es
`load_default_tts_client()` (qué tipo devuelve) y `LockingTTSClient` (serialización de `.speak()`
sobre un `TTSClient` stub).
"""

from __future__ import annotations

import threading
import time

import pytest

from jarvis.audio.tts import LockingTTSClient, SapiTTSClient, load_default_tts_client


class _FakeTTSClient:
    """Stub de `TTSClient`: registra los textos con los que se llamó `.speak()` (spy)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def speak(self, text: str) -> None:
        self.calls.append(text)


class _RaisingTTSClient:
    """Stub de `TTSClient` cuyo `.speak()` siempre lanza la excepción dada."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def speak(self, text: str) -> None:
        raise self._exc


def test_load_default_tts_client_returns_a_sapi_tts_client() -> None:
    """`load_default_tts_client()` devuelve un `SapiTTSClient` directo (ADR-0013): local, gratis,
    sin cuenta ni crédito externo — construir la instancia es barato, `pyttsx3.init()` recién se
    llama dentro de `.speak()`, no en `__init__`."""
    client = load_default_tts_client()

    assert isinstance(client, SapiTTSClient)


# --- LockingTTSClient (belt-and-suspenders contra `speak()` concurrente, ver docstring) ----------
# Motivo: `TimerScheduler` (`jarvis.audio.timer_scheduler`) es el primer llamador de `speak()` que
# corre en un thread de fondo genuinamente independiente del loop principal — antes de eso, todo
# llamador (incluido `VoiceConfirmationChannel.ask()` vía `asyncio.to_thread`) quedaba serializado
# de hecho porque el loop principal esperaba (`await`) cada llamada antes de seguir.


def test_locking_tts_client_forwards_calls_to_inner() -> None:
    inner = _FakeTTSClient()
    client = LockingTTSClient(inner=inner)

    client.speak("hola jarvis")

    assert inner.calls == ["hola jarvis"]


def test_locking_tts_client_does_not_propagate_inner_exception_specially() -> None:
    """No es su trabajo atrapar excepciones del inner client — solo serializa el acceso; una
    excepción del inner sigue propagando tal cual. Este es el comportamiento relevante ahora que
    no hay `FallbackTTSClient` (ADR-0011): un fallo de la API de OpenAI se propaga de verdad hasta
    el llamador de `speak()`, no solo en teoría."""
    inner = _RaisingTTSClient(RuntimeError("boom"))
    client = LockingTTSClient(inner=inner)

    with pytest.raises(RuntimeError, match="boom"):
        client.speak("texto")


def test_locking_tts_client_serializes_concurrent_calls_from_multiple_threads() -> None:
    """El escenario real que motiva este wrapper: `TimerScheduler` puede llamar `speak()` desde
    su propio thread de fondo mientras el loop principal está a mitad de decir otra cosa. Con el
    lock, dos llamadas concurrentes nunca ejecutan `inner.speak()` al mismo tiempo — se ponen en
    cola."""
    overlap_detected = threading.Event()
    currently_speaking = threading.Event()

    class _SlowInnerClient:
        def speak(self, text: str) -> None:
            if currently_speaking.is_set():
                overlap_detected.set()
            currently_speaking.set()
            time.sleep(0.02)
            currently_speaking.clear()

    client = LockingTTSClient(inner=_SlowInnerClient())
    threads = [
        threading.Thread(target=client.speak, args=(f"texto {i}",)) for i in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not overlap_detected.is_set()
