"""Tests para `jarvis.audio.realtime_stt` (transcripción en streaming vía la Realtime API de
Speechmatics, ADR-0012).

Sin red, sin micrófono real: `sounddevice.InputStream` se stubea con un fake que invoca el
callback registrado con chunks pre-armados (mismo enfoque que `tests/audio/test_pipeline.py` usa
para `sd.rec`), y `websockets.connect(...)` se stubea con un doble mínimo que replica solo la
superficie que el módulo bajo test usa de verdad (`.send()`, `.recv()`, iteración async) —
confirmado leyendo `websockets.asyncio.client.ClientConnection` instalado, no adivinado. Los
eventos de servidor son dicts JSON-serializables (el protocolo real de Speechmatics es JSON de
texto, no objetos tipados de SDK — a diferencia de la versión OpenAI de este módulo, acá no hay
tipos del lado del cliente que instanciar).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any, Self

import numpy as np

from jarvis.audio import realtime_stt
from jarvis.audio import stt as stt_module
from jarvis.audio.realtime_stt import SpeechmaticsCredentials, stream_transcribe_command

# `REALTIME_SAMPLE_RATE` (16kHz) == `SAMPLE_RATE` del VAD local: a diferencia de la versión
# OpenAI (24kHz fijo), acá el "device" de los tests se declara directamente a esa misma tasa para
# que el resample de `_capture_and_stream` sea un no-op exacto (ver `jarvis.audio.resample.
# resample`: `orig_sr == target_sr` devuelve el array tal cual) — y no hace falta, además, ningún
# resample aparte para el envío (ver docstring de `realtime_stt.py`).
DEVICE_SR = realtime_stt.REALTIME_SAMPLE_RATE
CHUNK_SAMPLES = int(realtime_stt.STREAM_CHUNK_SECONDS * DEVICE_SR)
LOUD_VALUE = (
    5000  # RMS muy por encima de cualquier `silence_threshold` razonable de los tests
)
SILENCE_THRESHOLD = 100.0
FAKE_API_KEY = (
    "test-speechmatics-key"  # nunca una clave real — solo para armar el header
)


def _fake_clock() -> Callable[[], float]:
    """Reloj falso de incremento fijo (`now_fn`), igual razonamiento que la versión OpenAI de
    este módulo: `_FakeInputStream` entrega todos los chunks de una, así que con el reloj real
    (`time.monotonic`) el tiempo transcurrido entre iteraciones sería ~0. Multiplica por un
    contador entero, no suma repetida — evita acumular error de redondeo justo en el límite de
    `TRAILING_SILENCE_SECONDS`."""
    state = {"count": 0}

    def _tick() -> float:
        state["count"] += 1
        return state["count"] * realtime_stt.STREAM_CHUNK_SECONDS

    return _tick


def _loud_chunk() -> np.ndarray:
    return np.full(CHUNK_SAMPLES, LOUD_VALUE, dtype=np.int16)


def _quiet_chunk() -> np.ndarray:
    return np.zeros(CHUNK_SAMPLES, dtype=np.int16)


def _recognition_started() -> dict[str, Any]:
    return {"message": "RecognitionStarted", "id": "session_1"}


def _partial_event(transcript: str) -> dict[str, Any]:
    return {
        "message": "AddPartialTranscript",
        "metadata": {"transcript": transcript, "start_time": 0.0, "end_time": 0.0},
        "results": [],
    }


def _final_event(transcript: str) -> dict[str, Any]:
    return {
        "message": "AddTranscript",
        "metadata": {"transcript": transcript, "start_time": 0.0, "end_time": 0.0},
        "results": [],
    }


def _end_of_transcript() -> dict[str, Any]:
    return {"message": "EndOfTranscript"}


def _error_event(reason: str) -> dict[str, Any]:
    return {"message": "Error", "type": "quota_exceeded", "reason": reason}


class _FakeConnection:
    """Doble de `websockets.asyncio.client.ClientConnection`: expone `.send()`/`.recv()` y es
    async-iterable, ambos leyendo de la MISMA secuencia de eventos pre-armados (igual que una
    conexión real, donde `recv()` y `async for` consumen del mismo stream entrante — confirmado
    en el código fuente instalado). Tras agotar los eventos, se queda "abierta" esperando más
    (`asyncio.Event().wait()`, nunca se completa) — igual que un WebSocket real que sigue vivo
    después del último mensaje: ejercita el `receive_task.cancel()` de `_capture_and_stream` en
    vez de depender de que el fake termine solo.
    """

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.sent_text: list[str] = []
        self.sent_binary: list[bytes] = []
        self._events = list(events)
        self._index = 0

    async def send(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            self.sent_binary.append(message)
        else:
            self.sent_text.append(message)

    async def recv(self) -> str:
        if self._index >= len(self._events):
            await asyncio.Event().wait()
        event = self._events[self._index]
        self._index += 1
        return json.dumps(event)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> str:
        return await self.recv()


class _FakeDroppedConnection(_FakeConnection):
    """Variante que simula una conexión WebSocket caída a nivel de transporte (red, cierre del
    servidor, timeout de ping/pong) tras agotar los eventos pre-armados: revienta con una
    excepción en vez de quedarse esperando — el escenario exacto del hallazgo de seguridad HIGH
    que motiva `_on_receive_task_done`/`TRANSCRIPTION_RESULT_TIMEOUT_SECONDS` en
    `realtime_stt._capture_and_stream`.
    """

    async def recv(self) -> str:
        if self._index >= len(self._events):
            raise ConnectionError("conexión perdida a nivel de transporte (simulado)")
        event = self._events[self._index]
        self._index += 1
        return json.dumps(event)


class _FakeConnect:
    """Doble de `websockets.connect(...)`: callable que registra los argumentos de conexión y
    devuelve un context manager async que entrega la `_FakeConnection` pre-armada — mismo patrón
    de doble mínimo que la versión OpenAI de este módulo usaba para `client.realtime.connect()`.
    """

    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection
        self.connect_calls: list[dict[str, Any]] = []

    def __call__(self, uri: str, **kwargs: Any) -> Self:
        self.connect_calls.append({"uri": uri, **kwargs})
        return self

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _FakeConnectRaises:
    """Doble de `websockets.connect(...)` que simula un fallo de conexión inicial (handshake,
    DNS, `401 Unauthorized`, etc.) — revienta al entrar al `async with`, antes de que exista
    ninguna `_FakeConnection` real."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __call__(self, uri: str, **kwargs: Any) -> Self:
        return self

    async def __aenter__(self) -> _FakeConnection:
        raise self._exc

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _fake_input_stream_factory(chunks: list[np.ndarray]) -> Any:
    class _FakeInputStream:
        def __init__(self, **kwargs: Any) -> None:
            self._callback = kwargs["callback"]

        def __enter__(self) -> Self:
            # Simula el modo callback de PortAudio: entrega todos los chunks pre-armados de una,
            # antes de que el bucle consumidor (`await audio_queue.get()`) llegue a pedir el
            # primero.
            for chunk in chunks:
                self._callback(chunk.reshape(-1, 1), len(chunk), None, 0)
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

    return _FakeInputStream


def _patch_audio(monkeypatch: Any, chunks: list[np.ndarray]) -> None:
    monkeypatch.setattr(realtime_stt, "resolve_input_device", lambda device: 3)
    monkeypatch.setattr(realtime_stt, "input_sample_rate", lambda device: DEVICE_SR)
    monkeypatch.setattr(
        realtime_stt.sd, "InputStream", _fake_input_stream_factory(chunks)
    )
    # Sin esto, `_capture_and_stream` cargaría el modelo real de Silero — los fixtures
    # `_loud_chunk`/`_quiet_chunk` son valores constantes de RMS, no audio con forma de voz real,
    # así que el modelo real los clasificaría a su manera y rompería estos tests, pensados para
    # controlar la detección solo por RMS (`None` degrada `vad.is_speech_chunk` a esa decisión).
    monkeypatch.setattr(realtime_stt, "load_speech_detector", lambda: None)


# --- _start_recognition_message ----------------------------------------------------------------


def test_start_recognition_message_disables_server_side_turn_detection() -> None:
    """Sin `conversation_config` en absoluto: el default documentado de
    `end_of_utterance_silence_trigger` es `0` ("desactiva la funcionalidad") — el corte de turno
    lo decide el VAD local, nunca el servidor (ver docstring del módulo)."""
    message = realtime_stt._start_recognition_message()

    assert "conversation_config" not in message["transcription_config"]


def test_start_recognition_message_uses_16khz_pcm_and_the_enhanced_model() -> None:
    message = realtime_stt._start_recognition_message()

    assert message["audio_format"] == {
        "type": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": realtime_stt.REALTIME_SAMPLE_RATE,
    }
    assert message["transcription_config"]["model"] == realtime_stt.TRANSCRIPTION_MODEL


def test_start_recognition_message_reuses_batch_language_and_skips_custom_vocab() -> (
    None
):
    """Mismo idioma que el camino batch (`jarvis.audio.stt.LANGUAGE`) y, deliberadamente, sin
    `additional_vocab` — costo de latencia documentado de hasta 15s por sesión (ver docstring del
    módulo), inaceptable para una sesión nueva por comando."""
    message = realtime_stt._start_recognition_message()

    assert message["transcription_config"]["language"] == stt_module.LANGUAGE
    assert "additional_vocab" not in message["transcription_config"]


def test_start_recognition_message_enables_partials_with_low_max_delay() -> None:
    message = realtime_stt._start_recognition_message()

    assert message["transcription_config"]["enable_partials"] is True
    assert (
        message["transcription_config"]["max_delay"] == realtime_stt.MAX_DELAY_SECONDS
    )


# --- stream_transcribe_command: happy path -------------------------------------------------------


def test_stream_transcribe_command_happy_path_accumulates_multiple_finals(
    monkeypatch: Any,
) -> None:
    """Caso aceptado: chunks transmitidos como binario a medida que llegan, `EndOfStream`
    disparado por el VAD local al detectar silencio sostenido tras habla real, y el texto final
    es la CONCATENACIÓN de los `AddTranscript` recibidos (dos tramos separados, no un único
    evento con el texto completo — a diferencia de la versión OpenAI, ver docstring del módulo),
    ignorando los `AddPartialTranscript` intermedios.

    7 chunks de silencio, no 6: con `STREAM_CHUNK_SECONDS=0.2` y `TRAILING_SILENCE_SECONDS=0.7`,
    3 chunks ya cruzarían el límite — un chunk extra de margen evita depender de una igualdad
    exacta de punto flotante justo en el borde.
    """
    chunks = [_loud_chunk(), _loud_chunk()] + [_quiet_chunk()] * 7
    events = [
        _recognition_started(),
        _partial_event("hola"),
        _final_event("Hola,"),
        _final_event("mundo."),
        _end_of_transcript(),
    ]
    connection = _FakeConnection(events)
    fake_connect = _FakeConnect(connection)
    monkeypatch.setattr(realtime_stt.websockets, "connect", fake_connect)
    _patch_audio(monkeypatch, chunks)

    text, speech_detected = asyncio.run(
        stream_transcribe_command(
            client=SpeechmaticsCredentials(api_key=FAKE_API_KEY),
            device=None,
            silence_threshold=SILENCE_THRESHOLD,
            max_duration=10.0,
            now_fn=_fake_clock(),
        )
    )

    assert (text, speech_detected) == ("Hola, mundo.", True)
    assert (
        json.loads(connection.sent_text[0]) == realtime_stt._start_recognition_message()
    )
    end_of_stream = json.loads(connection.sent_text[-1])
    assert end_of_stream["message"] == "EndOfStream"
    # Corta al 3er chunk de silencio (2 fuertes + 3 = 5), no consume los 9 provistos: el VAD
    # local ya cruzó `TRAILING_SILENCE_SECONDS` ahí.
    assert end_of_stream["last_seq_no"] == len(connection.sent_binary)
    assert len(connection.sent_binary) < len(chunks)
    assert fake_connect.connect_calls == [
        {
            "uri": realtime_stt.REALTIME_URL,
            "additional_headers": {"Authorization": f"Bearer {FAKE_API_KEY}"},
        }
    ]


# --- stream_transcribe_command: sin habla real ---------------------------------------------------


def test_stream_transcribe_command_no_speech_never_sends_end_of_stream(
    monkeypatch: Any,
) -> None:
    """El VAD local nunca cruza `silence_threshold`: se corta por `max_duration`, nunca se manda
    `EndOfStream`, y el resultado es `("", False)` — mismo contrato que
    `pipeline.record_command`/`speech_detected=False`."""
    chunks = [_quiet_chunk()] * 4  # de sobra para max_duration=0.5s
    connection = _FakeConnection([_recognition_started()])
    monkeypatch.setattr(realtime_stt.websockets, "connect", _FakeConnect(connection))
    _patch_audio(monkeypatch, chunks)

    text, speech_detected = asyncio.run(
        stream_transcribe_command(
            client=SpeechmaticsCredentials(api_key=FAKE_API_KEY),
            device=None,
            silence_threshold=SILENCE_THRESHOLD,
            max_duration=0.5,
            now_fn=_fake_clock(),
        )
    )

    assert (text, speech_detected) == ("", False)
    assert not any(
        json.loads(msg)["message"] == "EndOfStream" for msg in connection.sent_text
    )


# --- stream_transcribe_command: evento Error ------------------------------------------------------


def test_stream_transcribe_command_error_event_degrades_to_empty_string(
    monkeypatch: Any, caplog: Any
) -> None:
    """Un `Error` de sesión no debe propagar una excepción — se loguea a WARNING y se devuelve
    `""` (se descarta cualquier tramo `AddTranscript` ya acumulado), con `speech_detected=True`
    (sí hubo habla real y sí se mandó `EndOfStream`, la sesión en sí fue la que falló)."""
    chunks = [_loud_chunk()] + [_quiet_chunk()] * 7
    events = [
        _recognition_started(),
        _final_event("tramo parcial"),
        _error_event("boom"),
    ]
    connection = _FakeConnection(events)
    monkeypatch.setattr(realtime_stt.websockets, "connect", _FakeConnect(connection))
    _patch_audio(monkeypatch, chunks)

    with caplog.at_level("WARNING", logger="jarvis.audio.realtime_stt"):
        text, speech_detected = asyncio.run(
            stream_transcribe_command(
                client=SpeechmaticsCredentials(api_key=FAKE_API_KEY),
                device=None,
                silence_threshold=SILENCE_THRESHOLD,
                max_duration=10.0,
                now_fn=_fake_clock(),
            )
        )

    assert (text, speech_detected) == ("", True)
    assert any("error del servidor" in record.message for record in caplog.records)


# --- stream_transcribe_command: conexión caída sin evento terminal (HIGH, disponibilidad) --------


def test_stream_transcribe_command_dropped_connection_resolves_promptly_not_via_timeout(
    monkeypatch: Any, caplog: Any
) -> None:
    """Hallazgo de seguridad HIGH: si el WebSocket se cae a nivel de transporte SIN mandar nunca
    `EndOfTranscript`/`Error`, `_drain_events` no tiene ninguna oportunidad de resolver
    `result_future` por sí solo — sin el `add_done_callback` de `_on_receive_task_done`,
    `await result_future` colgaría hasta `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS`. Se mide
    wall-clock real (no `_fake_clock`, que solo gobierna el reloj interno del VAD) y se afirma que
    el test termina muy por debajo del timeout — la prueba de que resuelve el callback, no el
    backstop.
    """
    chunks = [_loud_chunk()] + [_quiet_chunk()] * 7
    connection = _FakeDroppedConnection([_recognition_started()])
    monkeypatch.setattr(realtime_stt.websockets, "connect", _FakeConnect(connection))
    _patch_audio(monkeypatch, chunks)

    started = time.monotonic()
    with caplog.at_level("WARNING", logger="jarvis.audio.realtime_stt"):
        text, speech_detected = asyncio.run(
            stream_transcribe_command(
                client=SpeechmaticsCredentials(api_key=FAKE_API_KEY),
                device=None,
                silence_threshold=SILENCE_THRESHOLD,
                max_duration=10.0,
                now_fn=_fake_clock(),
            )
        )
    elapsed_wall = time.monotonic() - started

    assert (text, speech_detected) == ("", True)
    assert elapsed_wall < realtime_stt.TRANSCRIPTION_RESULT_TIMEOUT_SECONDS / 2
    assert any(
        "receive_task terminó con una excepción" in record.message
        for record in caplog.records
    )


# --- stream_transcribe_command: fallo de conexión inicial -----------------------------------------


def test_stream_transcribe_command_initial_connection_failure_degrades_without_raising(
    monkeypatch: Any, caplog: Any
) -> None:
    """Pedido explícito para este módulo (a diferencia de la versión OpenAI, que dejaba propagar
    esto): CUALQUIER fallo antes o durante el establecimiento de la sesión (acá, un fallo de
    handshake/autenticación) degrada a `("", False)` en vez de propagar una excepción — nunca
    llega a tocar `sounddevice` ni a abrir el micrófono."""
    monkeypatch.setattr(
        realtime_stt.websockets,
        "connect",
        _FakeConnectRaises(ConnectionRefusedError("handshake rechazado (simulado)")),
    )

    with caplog.at_level("WARNING", logger="jarvis.audio.realtime_stt"):
        text, speech_detected = asyncio.run(
            stream_transcribe_command(
                client=SpeechmaticsCredentials(api_key=FAKE_API_KEY),
                device=None,
                silence_threshold=SILENCE_THRESHOLD,
                max_duration=10.0,
                now_fn=_fake_clock(),
            )
        )

    assert (text, speech_detected) == ("", False)
    assert any(
        "no se pudo completar la sesión de streaming" in record.message
        for record in caplog.records
    )


def test_stream_transcribe_command_rejected_start_recognition_degrades_without_raising(
    monkeypatch: Any, caplog: Any
) -> None:
    """Mismo contrato que el fallo de conexión inicial, pero para un `Error` recibido ANTES de
    `RecognitionStarted` (`StartRecognition` rechazado — config inválida, clave inválida ya
    pasado el handshake HTTP): `_wait_for_recognition_started` lo convierte en una excepción
    clara, y `stream_transcribe_command` la atrapa igual que cualquier otro fallo de sesión."""
    connection = _FakeConnection([_error_event("invalid transcription_config")])
    monkeypatch.setattr(realtime_stt.websockets, "connect", _FakeConnect(connection))

    with caplog.at_level("WARNING", logger="jarvis.audio.realtime_stt"):
        text, speech_detected = asyncio.run(
            stream_transcribe_command(
                client=SpeechmaticsCredentials(api_key=FAKE_API_KEY),
                device=None,
                silence_threshold=SILENCE_THRESHOLD,
                max_duration=10.0,
                now_fn=_fake_clock(),
            )
        )

    assert (text, speech_detected) == ("", False)
    assert not connection.sent_binary  # nunca llegó a tocar el micrófono


def test_wait_for_recognition_started_tolerates_info_before_ack() -> None:
    """`Info` (ej. `recognition_quality`) puede llegar antes de `RecognitionStarted` — confirmado
    en la documentación real ("se manda inmediatamente después del handshake") — y no debe
    abortar la espera."""
    connection = _FakeConnection(
        [{"message": "Info", "type": "recognition_quality"}, _recognition_started()]
    )

    asyncio.run(realtime_stt._wait_for_recognition_started(connection))  # type: ignore[arg-type]

    assert connection._index == 2
