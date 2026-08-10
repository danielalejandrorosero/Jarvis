"""Transcripción de audio en streaming vía la Realtime API de OpenAI (`gpt-live-transcribe`,
julio 2026) — WebSocket, no REST: el objetivo es que JARVIS reciba texto MIENTRAS el usuario
todavía está hablando, en vez de esperar (a) que el VAD local detecte silencio Y (b) un
round-trip completo de red a `client.audio.transcriptions.create()` (el camino batch existente,
`jarvis.audio.stt.transcribe`, que sigue intacto — ver más abajo).

Este módulo es ADITIVO, no un reemplazo: `jarvis.audio.stt.transcribe()` (batch, `gpt-4o-
transcribe`) sigue siendo el camino usado por `VoiceConfirmationChannel.ask()`
(`jarvis.audio.pipeline`) para confirmaciones CONFIRM (sí/no habladas antes de una acción
irreversible o de impacto real, ADR-0004/0005) — decisión deliberada, no un olvido: en ese punto
la latencia importa menos que la fiabilidad probada del camino batch (`temperature=0.0`,
`logprobs` de observabilidad), y `gpt-live-transcribe` no devuelve logprobs en absoluto (ver más
abajo), así que introducirlo ahí cambiaría el perfil de fiabilidad justo en el único punto donde
una transcripción mal interpretada ("sí" en vez de "no") tiene consecuencias de seguridad reales.
Solo la captura de comandos generales (`run()`, la mayoría del tráfico de voz, sin ese mismo peso
de seguridad) migra a este módulo.

Diseño clave: `turn_detection: null` en la sesión — se desactiva a propósito la detección de
turnos del lado del servidor. Se evaluó y se descartó dejar que la Realtime API decida esto en
la nube (`server_vad`/`semantic_vad`): confirmado en vivo contra la API real que
`gpt-live-transcribe` en modo `type: "transcription"` los rechaza a ambos ("Turn detection is
not supported for this transcription model") — no es opcional, es una limitación actual y
documentada de este modelo/modo específico. El corte real (`input_audio_buffer.commit`) lo
dispara exclusivamente el VAD local (`jarvis.audio.vad`: `chunk_rms`, `is_speech_chunk`,
`should_stop_recording`, más `measure_noise_floor`/`calibrate_thresholds` en `pipeline.py`,
y `jarvis.audio.speech_detector` para la parte de "¿esto suena a voz humana?" — ver su
docstring), exactamente como en el camino batch (`pipeline.record_command`). Por esto mismo
`jarvis.audio.vad` se separó de `jarvis.audio.pipeline` (ver docstring de ese módulo): este
archivo necesita esas mismas funciones sin poder importar `pipeline.py` (que sí importa este
módulo, para invocarlo desde `run()` — importar en la otra dirección sería un ciclo).

Sin score de confianza (`logprobs`/`avg_logprob`) en este modo: a diferencia de
`jarvis.audio.stt.transcribe()`, `gpt-live-transcribe` no los expone. Regresión conocida y
aceptada explícitamente (discutida con el usuario): ese logging era solo observabilidad, nunca
estuvo conectado a ningún filtro activo (ver docstring de `stt.transcribe`), así que perderlo no
quita ninguna protección real hoy.

Sin `reduce_background_noise`/`normalize_gain` (`jarvis.audio.pipeline`) en este camino, a
propósito: las dos operan sobre el clip COMPLETO ya grabado (resta espectral contra un perfil de
ruido calculado sobre toda la muestra, o AGC a partir del pico de todo el clip) — aplicarlas por
cada chunk chico que se transmite en vivo exigiría acumular todo el audio antes de mandar nada
(perdiendo exactamente la ventaja de latencia que motiva este módulo) o reprocesar cada chunk de
forma aislada, con artefactos de borde en cada ventana espectral independiente. Si en uso real
aparecen alucinaciones nuevas atribuibles a ruido de fondo sin filtrar, revisar esto — no se
intenta preservar ninguna de las dos acá.

Puente sounddevice (bloqueante, basado en el thread/callback de PortAudio) → asyncio: a
diferencia de `pipeline.record_command`/`iter_microphone_frames` (`sd.InputStream` en modo
BLOQUEANTE, `stream.read()`), acá se usa el modo CALLBACK de `sounddevice`
(`sd.InputStream(callback=...)`, confirmado en el código fuente instalado de `sounddevice`, no
asumido): PortAudio llama al callback en su propio thread nativo cada vez que hay un chunk nuevo
listo, nunca en el thread/loop de asyncio. El callback empuja una copia del chunk a una
`asyncio.Queue` vía `loop.call_soon_threadsafe(...)` — puente estándar entre un thread bloqueante
y un loop de asyncio (`Queue.put_nowait` no es thread-safe por sí solo, pero programarlo con
`call_soon_threadsafe` sobre el loop dueño de la queue sí lo es). El bucle principal de este
módulo, corriendo en el loop de asyncio, consume esa queue con `await queue.get()` mientras manda
cada chunk ya resampleado a 24kHz al WebSocket (`input_audio_buffer.append`) y corre el VAD local
sobre una copia a 16kHz del mismo chunk — mismo `resample()` de siempre, dos veces con destinos
distintos: uno para que el VAD compare contra `silence_threshold` (calibrado en dominio de
16kHz, igual que en `pipeline.py`) y otro para cumplir el único sample rate que soporta la
Realtime API (24kHz).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime.audio_transcription_param import AudioTranscriptionParam
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    ConversationItemInputAudioTranscriptionCompletedEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_delta_event import (
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    ConversationItemInputAudioTranscriptionFailedEvent,
)
from openai.types.realtime.realtime_error_event import RealtimeErrorEvent
from openai.types.realtime.realtime_transcription_session_audio_input_param import (
    RealtimeTranscriptionSessionAudioInputParam,
)
from openai.types.realtime.realtime_transcription_session_create_request_param import (
    RealtimeTranscriptionSessionCreateRequestParam,
)

from jarvis.audio.device import input_sample_rate, resolve_input_device
from jarvis.audio.loopback import SystemAudioGate
from jarvis.audio.resample import resample
from jarvis.audio.speech_detector import (
    SPEECH_PROBABILITY_THRESHOLD,
    load_speech_detector,
)
from jarvis.audio.stt import LANGUAGE, PROMPT
from jarvis.audio.vad import chunk_rms, is_speech_chunk, should_stop_recording
from jarvis.audio.wake_word import SAMPLE_RATE

logger = logging.getLogger(__name__)

REALTIME_MODEL = "gpt-live-transcribe"
REALTIME_SAMPLE_RATE = (
    24_000  # única tasa que soporta el input de audio de la Realtime API
)
# (confirmado en el SDK instalado: `RealtimeAudioFormatsParam.AudioPCM.rate: Literal[24000]`).
STREAM_CHUNK_SECONDS = (
    0.2  # mismo tamaño de ventana de VAD que `pipeline.CHUNK_SECONDS` — no se
)
# importa esa constante de `pipeline.py` (crearía el import circular que motiva este módulo, ver
# su docstring); se repite el valor acá como constante propia, standalone.

TRANSCRIPTION_RESULT_TIMEOUT_SECONDS = (
    10.0  # backstop duro tras `input_audio_buffer.commit()` — mismo orden de magnitud
)
# que los timeouts de red ya existentes en el resto del codebase para un único round-trip
# (`tools.search.REQUEST_TIMEOUT_SECONDS`/`tools.weather.REQUEST_TIMEOUT_SECONDS`, ambos 10.0).
# En el camino feliz, el resultado llega en un round-trip corto (el audio ya se transmitió en
# vivo mientras se grababa; el commit solo cierra el turno). Sin este timeout, si la conexión
# WebSocket se cae a nivel de transporte (red, cierre del servidor, timeout de ping/pong de
# `websockets`) SIN mandar un evento `.completed`/`.failed`/`error` propio antes de morir,
# `await result_future` en `_capture_and_stream` colgaría para siempre — congelando todo el loop
# de voz de `pipeline.run()` (hallazgo de seguridad HIGH, disponibilidad). Ver también el
# `add_done_callback` en `_capture_and_stream`, que apunta a resolver esto de forma inmediata
# (no recién al vencer el timeout) — este valor es el backstop duro para el caso en que ese
# callback tenga algún hueco no previsto.

# Vocabulario real de JARVIS para `keywords` (ver `_session_config`) — palabras de activación,
# verbos de comando frecuentes, y apps/servicios que el usuario pide seguido en uso real esta
# sesión. No exhaustivo a propósito (`CLAUDE.md`: sin abstracciones prematuras) — si en el uso
# real aparecen otras palabras mal transcriptas de forma recurrente, agregar acá, no antes.
TRANSCRIPTION_KEYWORDS = (
    "Alexa",
    "Jarvis",
    "Mycroft",
    "Daniel",
    "temporizador",
    "recordatorio",
    "reproduce",
    "pausa",
    "volumen",
    "campeón",
    "Discord",
    "YouTube",
    "Spotify",
    "League of Legends",
)


def load_realtime_client() -> AsyncOpenAI:
    """Cliente async de la Realtime API. Lee `OPENAI_API_KEY` del entorno (cargado por
    `jarvis.config.load_dotenv()`), igual que `jarvis.audio.stt.load_stt_client` — mismo patrón
    cliente-sync/cliente-async que ya separa el resto de este codebase entre rutas bloqueantes
    (batch) y asíncronas.
    """
    return AsyncOpenAI()


def _session_config() -> RealtimeTranscriptionSessionCreateRequestParam:
    """Configuración de la sesión de transcripción: `turn_detection: null` (ver docstring del
    módulo — el corte lo decide el VAD local, nunca el servidor), PCM16 a 24kHz (única tasa
    soportada), mismo idioma que el camino batch (`jarvis.audio.stt.LANGUAGE`).

    Sin `delay`: se probó (`"low"`, ya en desuso) y se confirmó en vivo, contra la API real, que
    el servidor lo acepta sin error pero lo IGNORA por completo para `gpt-live-transcribe`
    (`session.updated` siempre devuelve `delay: null` sin importar qué valor se mande) — el
    propio SDK instalado documenta por qué: ese campo "Only supported with `gpt-realtime-whisper`
    in GA Realtime sessions", no el modelo que usa este módulo. No es una perilla real acá,
    incluirlo solo sugiere un control que no existe.

    `keywords` sí es real para este modelo (confirmado contra el SDK: "Supported by
    `gpt-transcribe` and `gpt-live-transcribe`") y, a diferencia de `prompt`, no es la misma
    mecánica que causó la alucinación documentada en `jarvis.audio.stt.PROMPT` — investigado
    esta noche: OpenAI lo rediseñó a propósito frente al viejo prompt de Whisper (que el decoder
    trataba como contexto a continuar, de ahí la alucinación); `keywords` son "hints, no salida
    obligatoria — el transcript solo debe incluir una keyword si el audio realmente la contiene"
    (documentación oficial). Vocabulario real de JARVIS: nombre del usuario, palabras de
    activación/dormir, verbos de comando frecuentes y apps que pide seguido — no una lista
    exhaustiva, la que ya se ve en `SYSTEM_PROMPT`/uso real de esta sesión. `prompt` se deja sin
    usar (sigue siendo texto libre "a continuar", más parecido al mecanismo que sí alucinaba).
    """
    transcription: AudioTranscriptionParam = {
        "model": REALTIME_MODEL,
        "languages": [LANGUAGE],
        "keywords": list(TRANSCRIPTION_KEYWORDS),
    }
    if PROMPT is not None:
        transcription["prompt"] = PROMPT
    audio_input: RealtimeTranscriptionSessionAudioInputParam = {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": transcription,
        "turn_detection": None,
    }
    return {"type": "transcription", "audio": {"input": audio_input}}


async def _send_chunk(
    connection: AsyncRealtimeConnection, chunk: np.ndarray, *, orig_sr: int
) -> None:
    """Resamplear `chunk` (int16 mono) a `REALTIME_SAMPLE_RATE` y mandarlo al buffer de entrada
    del WebSocket como PCM16 base64 — el formato exacto que espera `input_audio_buffer.append`
    (confirmado en el SDK instalado, no en documentación externa: el campo se llama `audio` y es
    un `str`, no bytes crudos).
    """
    resampled = resample(chunk, orig_sr=orig_sr, target_sr=REALTIME_SAMPLE_RATE)
    payload = base64.b64encode(resampled.astype(np.int16).tobytes()).decode("ascii")
    await connection.input_audio_buffer.append(audio=payload)


async def _drain_events(
    connection: AsyncRealtimeConnection, result_future: asyncio.Future[str]
) -> None:
    """Consumir eventos del servidor durante toda la sesión, resolviendo `result_future` apenas
    llega el resultado final del turno de transcripción — nunca antes, para no devolver texto
    parcial como si fuera el definitivo.

    Tres desenlaces posibles, los tres resuelven `result_future` (nunca la dejan colgada):
    `.completed` con el texto final, o `.failed`/un evento `error` de nivel de conexión con
    cadena vacía — mismo contrato de "degradar sin lanzar" que ya usa el resto del pipeline ante
    audio ambiguo (ver `pipeline.record_command`/`speech_detected`). Los `.delta` (texto parcial
    mientras el usuario sigue hablando — la razón de ser de este módulo) solo se loguean a INFO
    para observabilidad; ningún llamador depende todavía de consumirlos en vivo.
    """
    async for event in connection:
        if isinstance(event, ConversationItemInputAudioTranscriptionDeltaEvent):
            if event.delta:
                logger.info(
                    "stream_transcribe_command(): parcial en vivo: %r", event.delta
                )
        elif isinstance(event, ConversationItemInputAudioTranscriptionCompletedEvent):
            if not result_future.done():
                result_future.set_result(event.transcript)
            return
        elif isinstance(event, ConversationItemInputAudioTranscriptionFailedEvent):
            logger.warning(
                "stream_transcribe_command(): transcripción falló: %s",
                event.error.message,
            )
            if not result_future.done():
                result_future.set_result("")
            return
        elif isinstance(event, RealtimeErrorEvent):
            logger.warning(
                "stream_transcribe_command(): error de conexión: %s", event.error
            )
            if not result_future.done():
                result_future.set_result("")
            return


async def _capture_and_stream(
    connection: AsyncRealtimeConnection,
    *,
    device: int | None,
    silence_threshold: float,
    max_duration: float,
    pre_roll: np.ndarray | None,
    system_audio: SystemAudioGate | None,
    now_fn: Callable[[], float] = time.monotonic,
) -> tuple[str, bool]:
    """Capturar el mic en modo callback (ver docstring del módulo), transmitiendo cada chunk al
    WebSocket a medida que llega y decidiendo con el VAD local (`jarvis.audio.vad`) cuándo cortar
    y confirmar (`input_audio_buffer.commit`) — o, si nunca hubo habla real, cortar sin confirmar
    en absoluto (mismo contrato que `pipeline.record_command`/`speech_detected`: nada que
    transcribir, nada que enviar de más).

    `now_fn` (default `time.monotonic`, reloj real) es inyectable a propósito: los tests
    (`tests/audio/test_realtime_stt.py`) entregan todos los chunks pre-armados de una, sin pausas
    reales entre ellos (`_FakeInputStream`), así que con el reloj real el tiempo transcurrido
    apenas avanzaría entre iteraciones — un reloj falso de incremento fijo (mismo valor que
    `STREAM_CHUNK_SECONDS`) les deja seguir siendo deterministas y rápidos sin simular pausas
    reales con `asyncio.sleep`.
    """
    resolved_device = resolve_input_device(device)
    device_sr = input_sample_rate(resolved_device)
    chunk_samples = int(STREAM_CHUNK_SECONDS * device_sr)
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def _on_audio(
        indata: np.ndarray, _frames: int, _time_info: Any, status: sd.CallbackFlags
    ) -> None:
        # Corre en el thread nativo de PortAudio, nunca en el loop de asyncio — ver docstring
        # del módulo. `indata` es un array `(frames, channels)`; con `channels=1` la columna 0
        # es el único canal.
        if status:
            logger.warning("stream_transcribe_command(): status del stream: %s", status)
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata[:, 0].copy())

    result_future: asyncio.Future[str] = loop.create_future()
    receive_task = asyncio.create_task(_drain_events(connection, result_future))

    def _on_receive_task_done(task: asyncio.Task[None]) -> None:
        # Backstop rápido para una conexión que se cae a nivel de transporte (red, cierre del
        # lado del servidor, timeout de ping/pong de `websockets`) SIN mandar nunca un evento
        # `.completed`/`.failed`/`error` propio: en ese caso el `async for event in connection`
        # de `_drain_events` termina con una excepción (o el generador simplemente se acaba) sin
        # haber tocado `result_future` — nada más lo resolvería. Este callback corre apenas
        # `receive_task` termina, de la forma que sea, así que `result_future` se resuelve de
        # inmediato en vez de depender solo de `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS` como único
        # mecanismo de recuperación (ese timeout sigue como backstop duro por si este callback
        # tuviera algún hueco no previsto — los dos mecanismos coexisten a propósito).
        if result_future.done() or task.cancelled():
            # Ya resuelto por `_drain_events` (camino normal), o cancelado desde el `finally` de
            # más abajo porque `result_future` ya estaba resuelto — nada que hacer.
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "stream_transcribe_command(): receive_task terminó con una excepción sin "
                "resolver ningún resultado (conexión caída a nivel de transporte?): %r",
                exc,
            )
        result_future.set_result("")

    receive_task.add_done_callback(_on_receive_task_done)
    try:
        if pre_roll is not None and len(pre_roll) > 0:
            # `pre_roll` ya viene resampleado a `SAMPLE_RATE` (16kHz — ver
            # `pipeline.PRE_ROLL_FRAMES`/`iter_microphone_frames`), no a la tasa nativa del
            # device: `orig_sr` acá es `SAMPLE_RATE`, no `device_sr`. Se manda antes de abrir el
            # stream de captura, igual que `record_command` lo antepone al audio grabado.
            await _send_chunk(connection, pre_roll, orig_sr=SAMPLE_RATE)

        speech_started = False
        silence_run = 0.0
        elapsed = 0.0
        start_time = now_fn()
        last_now = start_time
        silence_started_at: float | None = None
        detector = load_speech_detector()
        with sd.InputStream(
            samplerate=device_sr,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            device=resolved_device,
            callback=_on_audio,
        ):
            while True:
                raw_chunk = await audio_queue.get()
                vad_chunk = resample(
                    raw_chunk, orig_sr=device_sr, target_sr=SAMPLE_RATE
                )
                # Reloj real (delta contra `start_time`/`silence_started_at`), no un incremento
                # fijo por iteración sumado repetidamente: a diferencia del camino batch
                # (`pipeline.record_command`, donde `stream.read()` bloquea durante
                # aproximadamente `CHUNK_SECONDS` reales por el propio hardware de audio), acá
                # cada iteración además espera un round-trip de red (`_send_chunk`) de duración
                # variable — un incremento fijo subestimaría el tiempo real transcurrido bajo
                # latencia de red, dejando `max_duration`/el corte por silencio tardar mucho más
                # en la práctica que lo configurado (bug real, en vivo, encontrado esta sesión:
                # una grabación que debía cortar en segundos siguió streameando >40s).
                #
                # `silence_run` se calcula como una única resta contra `silence_started_at`, no
                # como una suma repetida de deltas (`silence_run += tick`) — sumar muchos deltas
                # de punto flotante acumula error de redondeo (confirmado en vivo: tras varias
                # sumas de ~0.2, el acumulado quedó por debajo de 1.2 por un margen infinitesimal
                # y `should_stop_recording` nunca cruzó el umbral — colgado esperando un chunk que
                # nunca iba a llegar). `silence_started_at` se ancla a `last_now` (el `now` del
                # chunk ANTERIOR, el último con habla real), no al `now` de este mismo chunk —
                # así la duración de este primer chunk silencioso también cuenta como silencio
                # (mismo criterio que la versión original por acumulación, donde el primer tick
                # de silencio ya sumaba de entrada), no arranca la cuenta en cero recién en el
                # chunk siguiente.
                now = now_fn()
                elapsed = now - start_time
                rms = chunk_rms(vad_chunk)
                system_is_loud = system_audio is not None and system_audio.is_loud()
                speech_probability = (
                    detector.speech_probability(vad_chunk)
                    if detector is not None
                    else None
                )
                is_speech = is_speech_chunk(
                    rms,
                    min_rms_floor=silence_threshold,
                    speech_probability=speech_probability,
                    speech_probability_threshold=SPEECH_PROBABILITY_THRESHOLD,
                    system_is_loud=system_is_loud,
                )
                if is_speech:
                    speech_started = True
                    silence_started_at = None
                    silence_run = 0.0
                elif speech_started:
                    if silence_started_at is None:
                        silence_started_at = last_now
                    silence_run = now - silence_started_at
                last_now = now
                logger.info(
                    "stream_transcribe_command(): chunk rms=%.1f umbral=%.1f prob_voz=%s "
                    "habla=%s speech_started=%s silencio_acumulado=%.2fs transcurrido=%.2fs",
                    rms,
                    silence_threshold,
                    f"{speech_probability:.2f}"
                    if speech_probability is not None
                    else "N/A",
                    is_speech,
                    speech_started,
                    silence_run,
                    elapsed,
                )
                await _send_chunk(connection, raw_chunk, orig_sr=device_sr)
                if should_stop_recording(
                    speech_started=speech_started,
                    silence_run_seconds=silence_run,
                    elapsed_seconds=elapsed,
                    max_seconds=max_duration,
                ):
                    break

        if not speech_started:
            # Mismo motivo que `pipeline.record_command`/`speech_detected=False`: nunca cruzó el
            # umbral de silencio, así que no hay audio real que confirmar — cortar limpio, sin
            # `input_audio_buffer.commit()`, es la mitigación real contra alucinaciones sobre
            # silencio puro, no un ahorro de red incidental.
            return "", False

        await connection.input_audio_buffer.commit()
        try:
            text = await asyncio.wait_for(
                result_future, timeout=TRANSCRIPTION_RESULT_TIMEOUT_SECONDS
            )
        except TimeoutError:
            # Backstop duro (ver `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS`): ni `_drain_events` ni
            # `_on_receive_task_done` resolvieron nada dentro del plazo — degradar sin lanzar,
            # mismo contrato que `.failed`/`error` acá arriba (hubo habla real, sí se confirmó el
            # turno, pero la sesión de transcripción nunca devolvió un resultado utilizable).
            logger.warning(
                "stream_transcribe_command(): sin resultado de transcripción %.0fs después de "
                "commit() — degradando a cadena vacía en vez de colgar el loop de voz.",
                TRANSCRIPTION_RESULT_TIMEOUT_SECONDS,
            )
            text = ""
        return text.strip(), True
    finally:
        if not receive_task.done():
            receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receive_task


async def stream_transcribe_command(
    *,
    client: AsyncOpenAI,
    device: int | None,
    silence_threshold: float,
    max_duration: float,
    pre_roll: np.ndarray | None = None,
    system_audio: SystemAudioGate | None = None,
    now_fn: Callable[[], float] = time.monotonic,
) -> tuple[str, bool]:
    """Grabar y transcribir el comando dicho después de la wake word en streaming, vía la
    Realtime API de OpenAI (`gpt-live-transcribe`) — reemplazo, en `pipeline.run()`, del combo
    `record_command()` + `jarvis.audio.stt.transcribe()` para la captura de comandos generales
    (ver docstring del módulo para qué NO migra y por qué).

    Devuelve `(texto, speech_detected)` — mismo contrato que `record_command`: `speech_detected`
    es `True` solo si el VAD local (`jarvis.audio.vad`) detectó habla real durante la ventana de
    grabación. Con `speech_detected=False`, `texto` es siempre `""` y nunca se llegó a confirmar
    (`input_audio_buffer.commit`) nada en la sesión — el llamador puede seguir tratando ese caso
    exactamente igual que antes (no hay nada que transcribir).

    `silence_threshold`/`max_duration`/`pre_roll`/`system_audio` tienen el mismo significado
    exacto que en `pipeline.record_command` — pensado como reemplazo directo del mismo call site,
    no una API nueva a aprender.

    `client.realtime.connect(extra_query={"intent": "transcription"})`, sin `model=` — confirmado
    en vivo contra la API real de OpenAI (no adivinado): pasar `model=REALTIME_MODEL`
    (`"gpt-live-transcribe"`) directo a `connect()` falla con `invalid_model` ("is a transcription
    model and cannot be used as the realtime session model"; ese modelo solo es válido dentro de
    `audio.input.transcription.model`, ver `_session_config`). Pasar cualquier modelo "realtime"
    normal (ej. `gpt-realtime-mini`) sí conecta, pero abre una sesión `type: "realtime"`
    (conversacional) que después rechaza el `session.update` de tipo `"transcription"`
    ("Passing a transcription session update to a realtime session is not allowed"). El query
    param `intent=transcription` (sin `model`) es lo que hace que `connect()` abra directamente
    una sesión `type: "transcription"` desde el principio — verificado con un smoke test real de
    punta a punta (conectar, `session.update`, mandar audio sintético, `commit`, recibir
    `conversation.item.input_audio_transcription.completed`) antes de este fix, ver docstring del
    módulo.
    """
    async with client.realtime.connect(
        extra_query={"intent": "transcription"}
    ) as connection:
        await connection.session.update(session=_session_config())
        return await _capture_and_stream(
            connection,
            device=device,
            silence_threshold=silence_threshold,
            max_duration=max_duration,
            pre_roll=pre_roll,
            system_audio=system_audio,
            now_fn=now_fn,
        )
