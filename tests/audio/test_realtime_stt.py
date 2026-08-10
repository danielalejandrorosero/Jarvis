"""Tests para `jarvis.audio.realtime_stt` (transcripción en streaming vía la Realtime API de
OpenAI, `gpt-live-transcribe`).

Sin red, sin micrófono real: `sounddevice.InputStream` se stubea con un fake que invoca el
callback registrado con chunks pre-armados (mismo enfoque que `tests/audio/test_pipeline.py`
usa para `sd.rec`), y `client.realtime.connect(...)` se stubea con dobles mínimos que replican
solo la superficie que el módulo bajo test usa de verdad (`AsyncRealtimeConnection`/
`AsyncRealtimeConnectionManager`, confirmado leyendo el SDK instalado —
`openai/resources/realtime/realtime.py`): `session.update`, `input_audio_buffer.append`/
`.commit`, e iteración async del propio `connection` para los eventos del servidor. Los eventos
de servidor son instancias reales de los tipos del SDK (`ConversationItemInputAudioTranscription
*Event`), no `MagicMock`, porque `_drain_events` los distingue con `isinstance()`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, Self, cast

import numpy as np
from openai import AsyncOpenAI
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
    UsageTranscriptTextUsageDuration,
)
from openai.types.realtime.conversation_item_input_audio_transcription_delta_event import (
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    ConversationItemInputAudioTranscriptionFailedEvent,
    Error,
)

from jarvis.audio import realtime_stt
from jarvis.audio import stt as stt_module
from jarvis.audio.realtime_stt import stream_transcribe_command

DEVICE_SR = (
    24_000  # == REALTIME_SAMPLE_RATE, así el resample de envío es un no-op exacto (ver
)
# `jarvis.audio.resample.resample`: `orig_sr == target_sr` devuelve el array tal cual).
CHUNK_SAMPLES = int(realtime_stt.STREAM_CHUNK_SECONDS * DEVICE_SR)
LOUD_VALUE = (
    5000  # RMS muy por encima de cualquier `silence_threshold` razonable de los tests
)


def _fake_clock() -> Callable[[], float]:
    """Reloj falso de incremento fijo (`now_fn`, ver `realtime_stt._capture_and_stream`): cada
    llamada avanza exactamente `STREAM_CHUNK_SECONDS`, imitando el paso de tiempo real sin
    necesitar `asyncio.sleep` — `_FakeInputStream` entrega todos los chunks de una, así que con
    el reloj real (`time.monotonic`) el tiempo transcurrido entre iteraciones sería ~0.

    Multiplica por un contador entero (`count * STREAM_CHUNK_SECONDS`), no suma repetida
    (`now += STREAM_CHUNK_SECONDS`): sumar el mismo float muchas veces acumula error de redondeo
    call a call (confirmado en vivo: after 9 sumas de 0.2 el resultado no era 1.8 exacto), lo que
    hacía que `silence_run >= TRAILING_SILENCE_SECONDS` fallara por un margen infinitesimal justo
    en el límite y colgara el test para siempre esperando un chunk que no iba a llegar más.
    Multiplicar recalcula cada valor desde cero a partir de un entero limpio, sin heredar el
    redondeo del valor anterior."""
    state = {"count": 0}

    def _tick() -> float:
        state["count"] += 1
        return state["count"] * realtime_stt.STREAM_CHUNK_SECONDS

    return _tick


SILENCE_THRESHOLD = 100.0


def _loud_chunk() -> np.ndarray:
    # Valor constante: el resample (lineal) de un array constante reproduce el mismo valor
    # exacto sin importar la tasa de origen/destino — evita imprecisión de punto flotante en las
    # comparaciones de RMS del VAD (`chunk_rms`/`is_speech_chunk`) entre 24kHz y 16kHz.
    return np.full(CHUNK_SAMPLES, LOUD_VALUE, dtype=np.int16)


def _quiet_chunk() -> np.ndarray:
    return np.zeros(CHUNK_SAMPLES, dtype=np.int16)


class _FakeSessionResource:
    def __init__(self, outer: _FakeConnection) -> None:
        self._outer = outer

    async def update(self, *, session: dict[str, Any]) -> None:
        self._outer.session_updates.append(session)


class _FakeInputAudioBufferResource:
    def __init__(self, outer: _FakeConnection) -> None:
        self._outer = outer

    async def append(self, *, audio: str) -> None:
        self._outer.appended_audio.append(audio)

    async def commit(self) -> None:
        self._outer.committed = True


class _FakeConnection:
    """Doble de `AsyncRealtimeConnection`: expone `session`/`input_audio_buffer` y es
    async-iterable sobre `events`. Tras agotar `events`, se queda "abierta" esperando más
    (`asyncio.Event().wait()`, nunca se completa) — igual que una conexión WebSocket real que
    sigue viva después de la última transcripción: ejercita el `receive_task.cancel()` de
    `_capture_and_stream` en vez de depender de que el fake termine solo.
    """

    def __init__(self, events: list[object]) -> None:
        self.session_updates: list[dict[str, Any]] = []
        self.appended_audio: list[str] = []
        self.committed = False
        self._events = events
        self.session = _FakeSessionResource(self)
        self.input_audio_buffer = _FakeInputAudioBufferResource(self)

    async def __aiter__(self) -> Any:
        for event in self._events:
            yield event
        await asyncio.Event().wait()


class _FakeDroppedConnection(_FakeConnection):
    """Variante de `_FakeConnection` que simula una conexión WebSocket caída a nivel de
    transporte (red, cierre del lado del servidor, timeout de ping/pong de `websockets`): el
    `async for` sobre la conexión revienta con una excepción de golpe, sin mandar nunca un
    evento `.completed`/`.failed`/`error` propio antes de morir — el escenario exacto del
    hallazgo de seguridad HIGH que motiva `_on_receive_task_done`/
    `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS` en `realtime_stt._capture_and_stream`: sin esos dos
    mecanismos, `await result_future` colgaría para siempre.
    """

    async def __aiter__(self) -> Any:
        for event in self._events:
            yield event
        raise ConnectionError("conexión perdida a nivel de transporte (simulado)")


class _FakeConnectionManager:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _FakeRealtimeResource:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection
        self.connect_calls: list[dict[str, object]] = []

    def connect(self, **kwargs: object) -> _FakeConnectionManager:
        self.connect_calls.append(kwargs)
        return _FakeConnectionManager(self._connection)


class _FakeClient:
    def __init__(self, realtime: _FakeRealtimeResource) -> None:
        self.realtime = realtime


def _make_client(
    events: list[object],
) -> tuple[AsyncOpenAI, _FakeConnection, _FakeRealtimeResource]:
    connection = _FakeConnection(events)
    realtime_resource = _FakeRealtimeResource(connection)
    # El doble no implementa la superficie completa de `AsyncOpenAI`, solo `.realtime.connect()`
    # — mismo patrón de cast que `tests/audio/test_stt.py` usa para el cliente sync.
    return (
        cast(AsyncOpenAI, _FakeClient(realtime_resource)),
        connection,
        realtime_resource,
    )


class _FakeInputStream:
    def __init__(self, chunks: list[np.ndarray], **kwargs: Any) -> None:
        self._chunks = chunks
        self._callback = kwargs["callback"]

    def __enter__(self) -> Self:
        # Simula el modo callback de PortAudio: entrega todos los chunks pre-armados de una,
        # antes de que el bucle consumidor (`await audio_queue.get()`) llegue a pedir el primero
        # — ver docstring de `jarvis.audio.realtime_stt` sobre el puente
        # `call_soon_threadsafe`/`asyncio.Queue`.
        for chunk in self._chunks:
            self._callback(chunk.reshape(-1, 1), len(chunk), None, 0)
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _fake_input_stream_factory(chunks: list[np.ndarray]) -> Any:
    def _factory(**kwargs: Any) -> _FakeInputStream:
        return _FakeInputStream(chunks, **kwargs)

    return _factory


def _completed_event(
    transcript: str,
) -> ConversationItemInputAudioTranscriptionCompletedEvent:
    return ConversationItemInputAudioTranscriptionCompletedEvent(
        content_index=0,
        event_id="evt_completed",
        item_id="item_1",
        transcript=transcript,
        type="conversation.item.input_audio_transcription.completed",
        usage=UsageTranscriptTextUsageDuration(seconds=1.0, type="duration"),
    )


def _delta_event(delta: str) -> ConversationItemInputAudioTranscriptionDeltaEvent:
    return ConversationItemInputAudioTranscriptionDeltaEvent(
        event_id="evt_delta",
        item_id="item_1",
        type="conversation.item.input_audio_transcription.delta",
        delta=delta,
    )


def _failed_event(message: str) -> ConversationItemInputAudioTranscriptionFailedEvent:
    return ConversationItemInputAudioTranscriptionFailedEvent(
        content_index=0,
        error=Error(message=message),
        event_id="evt_failed",
        item_id="item_1",
        type="conversation.item.input_audio_transcription.failed",
    )


# --- _session_config -------------------------------------------------------------------------


def test_session_config_disables_server_side_turn_detection() -> None:
    """`turn_detection: null` — el corte de grabación lo decide el VAD local, nunca el servidor
    (ver docstring del módulo, ADR pendiente)."""
    config = realtime_stt._session_config()

    assert config["type"] == "transcription"
    assert config["audio"]["input"]["turn_detection"] is None


def test_session_config_uses_24khz_pcm_and_the_live_model() -> None:
    """Única tasa soportada por la Realtime API (`RealtimeAudioFormatsParam.AudioPCM.rate:
    Literal[24000]`, confirmado en el SDK instalado) y el modelo pedido explícitamente."""
    config = realtime_stt._session_config()

    audio_input = config["audio"]["input"]
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["transcription"]["model"] == realtime_stt.REALTIME_MODEL


def test_session_config_reuses_batch_language_and_no_prompt_hint() -> None:
    """Mismo idioma que el camino batch (`jarvis.audio.stt.LANGUAGE`) y, deliberadamente, sin
    `prompt` — reusa el hallazgo de alucinación ya documentado en `stt.PROMPT` (`None`)."""
    config = realtime_stt._session_config()

    transcription = config["audio"]["input"]["transcription"]
    assert transcription["languages"] == [stt_module.LANGUAGE]
    assert "prompt" not in transcription


def test_session_config_omits_delay_confirmed_as_a_no_op_for_this_model() -> None:
    """`delay` se probó en vivo contra la API real y el servidor lo acepta pero lo ignora por
    completo para `gpt-live-transcribe` (`session.updated` siempre devuelve `delay: null` sin
    importar el valor mandado — ver docstring de `_session_config`) — no es una perilla real acá,
    no debe aparecer sugiriendo un control que no existe."""
    config = realtime_stt._session_config()

    assert "delay" not in config["audio"]["input"]["transcription"]


def test_session_config_includes_real_jarvis_vocabulary_as_keywords() -> None:
    """`keywords`, a diferencia de `delay`, sí es real para este modelo (confirmado contra el
    SDK: "Supported by gpt-transcribe and gpt-live-transcribe") — vocabulario propio de JARVIS,
    no una lista vacía ni genérica."""
    config = realtime_stt._session_config()

    keywords = config["audio"]["input"]["transcription"]["keywords"]
    assert keywords == list(realtime_stt.TRANSCRIPTION_KEYWORDS)
    assert "Alexa" in keywords


# --- stream_transcribe_command: happy path ----------------------------------------------------


def test_stream_transcribe_command_happy_path_streams_chunks_and_returns_final_transcript(
    monkeypatch: Any,
) -> None:
    """Caso aceptado: chunks transmitidos a medida que llegan, `commit` disparado por el VAD
    local al detectar silencio sostenido tras habla real, y el texto final es el de
    `.completed` — no el de ningún `.delta` parcial.

    7 chunks de silencio, no 6: con `STREAM_CHUNK_SECONDS=0.2` y `TRAILING_SILENCE_SECONDS=1.2`,
    6 chunks caen justo en el límite exacto (6 × 0.2 = 1.2). `_fake_clock`/`_capture_and_stream`
    ya no acumulan error de redondeo (ver sus respectivos docstrings — reloj por multiplicación,
    `silence_started_at` ancla a un timestamp fijo en vez de sumar deltas), pero un test que
    depende de una igualdad exacta de punto flotante (`>=` justo en el límite) sigue siendo
    frágil por principio — un chunk de margen de más evita testear exactamente sobre el borde.
    """
    chunks = [_loud_chunk(), _loud_chunk()] + [_quiet_chunk()] * 7
    events: list[object] = [_delta_event("hola"), _completed_event("hola mundo")]
    client, connection, realtime_resource = _make_client(events)
    monkeypatch.setattr(realtime_stt, "resolve_input_device", lambda device: 3)
    monkeypatch.setattr(realtime_stt, "input_sample_rate", lambda device: DEVICE_SR)
    monkeypatch.setattr(
        realtime_stt.sd, "InputStream", _fake_input_stream_factory(chunks)
    )
    # Sin esto, `_capture_and_stream` cargaría el modelo real de Silero — los fixtures
    # `_loud_chunk`/`_quiet_chunk` son valores constantes de RMS, no audio con forma de voz real,
    # así que el modelo real los clasificaría a su manera (probablemente como "no es voz") y
    # rompería estos tests, que están pensados para controlar la detección solo por RMS (`None`
    # degrada `vad.is_speech_chunk` a esa decisión, ver su docstring).
    monkeypatch.setattr(realtime_stt, "load_speech_detector", lambda: None)

    text, speech_detected = asyncio.run(
        stream_transcribe_command(
            client=client,
            device=None,
            silence_threshold=SILENCE_THRESHOLD,
            max_duration=10.0,
            now_fn=_fake_clock(),
        )
    )

    assert (text, speech_detected) == ("hola mundo", True)
    assert connection.committed is True
    assert (
        len(connection.appended_audio) == 8
    )  # corta al octavo de los 9 chunks provistos (2 fuertes + 7 de silencio) — el VAD local ya
    # cruzó `TRAILING_SILENCE_SECONDS` ahí, no hace falta consumir el último
    # `extra_query={"intent": "transcription"}`, sin `model=` — confirmado en vivo contra la API
    # real (ver docstring de `stream_transcribe_command`): pasar el modelo de transcripción
    # directo a `connect()` falla con `invalid_model`, y pasar un modelo "realtime" normal abre
    # una sesión conversacional que rechaza el `session.update` de tipo "transcription".
    assert realtime_resource.connect_calls == [
        {"extra_query": {"intent": "transcription"}}
    ]
    assert connection.session_updates[0]["audio"]["input"]["turn_detection"] is None


# --- stream_transcribe_command: sin habla real ------------------------------------------------


def test_stream_transcribe_command_no_speech_never_commits_or_calls_stt(
    monkeypatch: Any,
) -> None:
    """El VAD local nunca cruza `silence_threshold`: se corta por `max_duration`, nunca se manda
    `input_audio_buffer.commit`, y el resultado es `("", False)` — mismo contrato que
    `pipeline.record_command`/`speech_detected=False`, ninguna alucinación posible porque nunca
    se le pide un resultado final a la sesión."""
    chunks = [
        _quiet_chunk()
    ] * 4  # de sobra para max_duration=0.5s (corta al 3er chunk)
    client, connection, _realtime_resource = _make_client([])
    monkeypatch.setattr(realtime_stt, "resolve_input_device", lambda device: 3)
    monkeypatch.setattr(realtime_stt, "input_sample_rate", lambda device: DEVICE_SR)
    monkeypatch.setattr(
        realtime_stt.sd, "InputStream", _fake_input_stream_factory(chunks)
    )
    # Sin esto, `_capture_and_stream` cargaría el modelo real de Silero — los fixtures
    # `_loud_chunk`/`_quiet_chunk` son valores constantes de RMS, no audio con forma de voz real,
    # así que el modelo real los clasificaría a su manera (probablemente como "no es voz") y
    # rompería estos tests, que están pensados para controlar la detección solo por RMS (`None`
    # degrada `vad.is_speech_chunk` a esa decisión, ver su docstring).
    monkeypatch.setattr(realtime_stt, "load_speech_detector", lambda: None)

    text, speech_detected = asyncio.run(
        stream_transcribe_command(
            client=client,
            device=None,
            silence_threshold=SILENCE_THRESHOLD,
            max_duration=0.5,
            now_fn=_fake_clock(),
        )
    )

    assert (text, speech_detected) == ("", False)
    assert connection.committed is False
    assert len(connection.appended_audio) == 3


# --- stream_transcribe_command: evento .failed -------------------------------------------------


def test_stream_transcribe_command_failed_event_degrades_to_empty_string(
    monkeypatch: Any, caplog: Any
) -> None:
    """`.failed` no debe propagar una excepción — se loguea a WARNING y se devuelve `""`, con
    `speech_detected=True` (sí hubo habla real y sí se confirmó, la sesión de transcripción en
    sí fue la que falló).

    7 chunks de silencio, no 6 — mismo motivo que el test de happy path: evitar depender de una
    igualdad exacta de punto flotante justo en `TRAILING_SILENCE_SECONDS`."""
    chunks = [_loud_chunk()] + [_quiet_chunk()] * 7
    events: list[object] = [_failed_event("boom")]
    client, connection, _realtime_resource = _make_client(events)
    monkeypatch.setattr(realtime_stt, "resolve_input_device", lambda device: 3)
    monkeypatch.setattr(realtime_stt, "input_sample_rate", lambda device: DEVICE_SR)
    monkeypatch.setattr(
        realtime_stt.sd, "InputStream", _fake_input_stream_factory(chunks)
    )
    # Sin esto, `_capture_and_stream` cargaría el modelo real de Silero — los fixtures
    # `_loud_chunk`/`_quiet_chunk` son valores constantes de RMS, no audio con forma de voz real,
    # así que el modelo real los clasificaría a su manera (probablemente como "no es voz") y
    # rompería estos tests, que están pensados para controlar la detección solo por RMS (`None`
    # degrada `vad.is_speech_chunk` a esa decisión, ver su docstring).
    monkeypatch.setattr(realtime_stt, "load_speech_detector", lambda: None)

    with caplog.at_level("WARNING", logger="jarvis.audio.realtime_stt"):
        text, speech_detected = asyncio.run(
            stream_transcribe_command(
                client=client,
                device=None,
                silence_threshold=SILENCE_THRESHOLD,
                max_duration=10.0,
                now_fn=_fake_clock(),
            )
        )

    assert (text, speech_detected) == ("", True)
    assert connection.committed is True
    assert len(connection.appended_audio) == 7
    assert any("transcripción falló" in record.message for record in caplog.records)


# --- stream_transcribe_command: conexión caída sin evento terminal (HIGH, disponibilidad) -----


def test_stream_transcribe_command_dropped_connection_resolves_promptly_not_via_timeout(
    monkeypatch: Any, caplog: Any
) -> None:
    """Hallazgo de seguridad HIGH (`security-reviewer`): si el WebSocket se cae a nivel de
    transporte (red, cierre del servidor, timeout de ping/pong de `websockets`) SIN mandar nunca
    un evento `.completed`/`.failed`/`error` propio, `_drain_events` no tiene ninguna oportunidad
    de resolver `result_future` por sí solo — antes de este fix, `await result_future` colgaba
    para siempre (congelando el loop de voz completo de `pipeline.run()`, sin ninguna excepción
    que su propio `except Exception` pudiera atrapar: no hay excepción, solo un await pendiente
    para siempre). Este test simula exactamente ese escenario (`_FakeDroppedConnection`: el
    `async for` revienta sin evento terminal) y confirma que el resultado degrada a `("", True)`
    sin bloquear.

    Se mide wall-clock real (`time.monotonic`, no `_fake_clock` — ese solo gobierna el reloj
    interno del VAD, no cuánto tarda realmente `await result_future`) y se afirma que el test
    termina muy por debajo de `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS`: la prueba de que quien
    resuelve rápido es el `add_done_callback` de `receive_task` (`_on_receive_task_done`), no el
    timeout como único mecanismo — si el callback no existiera y solo quedara el timeout como
    backstop, este test tardaría los 10s completos en vez de resolver casi de inmediato.
    """
    chunks = [_loud_chunk()] + [_quiet_chunk()] * 7
    connection = _FakeDroppedConnection([])
    realtime_resource = _FakeRealtimeResource(connection)
    client = cast(AsyncOpenAI, _FakeClient(realtime_resource))
    monkeypatch.setattr(realtime_stt, "resolve_input_device", lambda device: 3)
    monkeypatch.setattr(realtime_stt, "input_sample_rate", lambda device: DEVICE_SR)
    monkeypatch.setattr(
        realtime_stt.sd, "InputStream", _fake_input_stream_factory(chunks)
    )
    # Sin esto, `_capture_and_stream` cargaría el modelo real de Silero — ver el mismo comentario
    # en los tests de arriba.
    monkeypatch.setattr(realtime_stt, "load_speech_detector", lambda: None)

    started = time.monotonic()
    with caplog.at_level("WARNING", logger="jarvis.audio.realtime_stt"):
        text, speech_detected = asyncio.run(
            stream_transcribe_command(
                client=client,
                device=None,
                silence_threshold=SILENCE_THRESHOLD,
                max_duration=10.0,
                now_fn=_fake_clock(),
            )
        )
    elapsed_wall = time.monotonic() - started

    assert (text, speech_detected) == ("", True)
    assert connection.committed is True
    # Muy por debajo del timeout de backstop: si el `add_done_callback` no existiera y el test
    # dependiera solo de `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS`, esto tardaría los 10s completos.
    assert elapsed_wall < realtime_stt.TRANSCRIPTION_RESULT_TIMEOUT_SECONDS / 2
    assert any(
        "receive_task terminó con una excepción" in record.message
        for record in caplog.records
    )
