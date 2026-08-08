"""Tests para `FallbackTTSClient.speak()` (la capa de recuperación de ADR-0004).

`OpenAITTSClient` y `SapiTTSClient` pegan contra red/hardware de audio real y no son testeables
como unit test (requieren la API de OpenAI y/o SAPI real). Lo que sí es testeable, y el punto
central de este módulo, es `FallbackTTSClient`: primario y fallback se reemplazan por stubs
controlados por el test, así que corre rápido, sin red y sin reproducir audio (CI-safe).
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from jarvis.audio.tts import (
    FallbackTTSClient,
    LockingTTSClient,
    TTSClient,
    load_default_tts_client,
)


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


def test_speak_calls_only_primary_when_primary_succeeds() -> None:
    """Caso aceptado: si el primario no falla, se lo llama con el texto correcto y el
    fallback no se toca en absoluto."""
    primary = _FakeTTSClient()
    fallback = _FakeTTSClient()
    client = FallbackTTSClient(primary=primary, fallback=fallback)

    client.speak("hola jarvis")

    assert primary.calls == ["hola jarvis"]
    assert fallback.calls == []


def test_speak_falls_back_when_primary_raises_runtime_error() -> None:
    """Caso rechazado: si el primario lanza (RuntimeError, no algo genérico ya esperado),
    el fallback se llama con el mismo texto."""
    primary = _RaisingTTSClient(RuntimeError("edge-tts endpoint no disponible"))
    fallback = _FakeTTSClient()
    client = FallbackTTSClient(primary=primary, fallback=fallback)

    client.speak("hola jarvis")

    assert fallback.calls == ["hola jarvis"]


def test_speak_falls_back_on_arbitrary_exception_type() -> None:
    """La captura es deliberadamente amplia (`except Exception`, documentado en el propio
    módulo como la capa de recuperación de ADR-0004): un tipo de excepción no relacionado con
    red/audio (ValueError) también dispara el fallback, no solo errores "esperables"."""
    primary = _RaisingTTSClient(ValueError("algo totalmente distinto"))
    fallback = _FakeTTSClient()
    client = FallbackTTSClient(primary=primary, fallback=fallback)

    client.speak("hola jarvis")

    assert fallback.calls == ["hola jarvis"]


def test_speak_does_not_propagate_primary_exception() -> None:
    """Caso límite: aunque el primario lance, `FallbackTTSClient.speak()` no debe lanzar —
    ese es el contrato central de "JARVIS nunca queda mudo"."""
    primary = _RaisingTTSClient(RuntimeError("boom"))
    fallback = _FakeTTSClient()
    client = FallbackTTSClient(primary=primary, fallback=fallback)

    # Si esto lanzara, el test fallaría con la excepción sin necesidad de pytest.raises.
    client.speak("no debería propagar")


def test_speak_does_not_call_fallback_when_fallback_would_also_fail_but_primary_succeeds() -> (
    None
):
    """Confirma que el éxito del primario evita tocar el fallback incluso si el fallback está
    configurado para fallar: la lógica no llama al fallback "por las dudas"."""
    primary = _FakeTTSClient()
    fallback = MagicMock(spec=TTSClient)
    fallback.speak.side_effect = RuntimeError("no debería ejecutarse nunca")
    client = FallbackTTSClient(primary=primary, fallback=fallback)

    client.speak("texto")

    fallback.speak.assert_not_called()


def test_load_default_tts_client_returns_a_fallback_tts_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_default_tts_client()` arma un `FallbackTTSClient` real (OpenAI TTS + SAPI), pero el
    test no invoca `.speak()` sobre él: construir las instancias es barato (sin I/O más allá de
    que el SDK de OpenAI exige una API key *presente* para instanciar el cliente, no que sea
    válida — `monkeypatch.setenv` alcanza, no hace falta una key real ni red), llamarlas no lo
    es."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")

    client = load_default_tts_client()

    assert isinstance(client, FallbackTTSClient)


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
    """No es su trabajo atrapar excepciones del inner client (eso es `FallbackTTSClient`) — solo
    serializa el acceso; una excepción del inner sigue propagando tal cual."""
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
