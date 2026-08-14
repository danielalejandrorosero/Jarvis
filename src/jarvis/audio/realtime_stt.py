"""Transcripción de audio en streaming vía la Realtime API de Speechmatics (WebSocket) — ADR-0012,
reemplazo de la Realtime API de OpenAI (`gpt-live-transcribe`, ADR-0007) como proveedor de STT
para la captura de comandos generales. Investigación con fuentes (no un cambio de gusto): un
benchmark independiente en español latinoamericano (dominio médico conversacional) midió 7.3% WER
para Speechmatics contra 9.5-9.8% para GPT-4o/Whisper/Deepgram en el mismo corpus — la mejor
evidencia encontrada para este idioma/variante específica, con el usuario señalando el STT como
cuello de botella real de la experiencia de voz.

Este módulo sigue siendo ADITIVO, no un reemplazo total: `jarvis.audio.stt.transcribe()` (batch,
`gpt-4o-transcribe`, OpenAI) sigue siendo el camino usado por `VoiceConfirmationChannel.ask()`
(`jarvis.audio.pipeline`) para confirmaciones CONFIRM (sí/no habladas antes de una acción
irreversible o de impacto real, ADR-0004/0005) — decisión deliberada, no un olvido, y NO migra:
mismo motivo ya documentado (fiabilidad probada + observabilidad de confianza vía `logprobs`, que
Speechmatics tampoco expone en este modo). Solo la captura de comandos generales (`run()`, la
mayoría del tráfico de voz) migra a Speechmatics.

Protocolo real usado, confirmado en vivo contra la documentación oficial
(docs.speechmatics.com/api-ref/realtime-transcription-websocket,
docs.speechmatics.com/speech-to-text/realtime/{quickstart,input,turn-detection},
docs.speechmatics.com/get-started/authentication) y contra el código fuente del SDK oficial
(`speechmatics-rt`, github.com/speechmatics/speechmatics-python-sdk) — no adivinado:

- WebSocket crudo (`websockets`, ya dependencia dura del repo — ver más abajo por qué no se suma
  `speechmatics-rt` como dependencia nueva), no REST: `wss://global.rt.speechmatics.com/v2`
  (endpoint recomendado por Speechmatics para menor latencia, enruta automáticamente a la región
  más cercana — hay endpoints regionales fijos si hiciera falta residencia de datos, no es el
  caso acá).
- Autenticación: header `Authorization: Bearer <SPEECHMATICS_API_KEY>` en el handshake HTTP de la
  conexión WebSocket — la clave larga-duración directa, SIN el flujo de token JWT temporal.
  Confirmado explícitamente en la documentación de autenticación: los tokens JWT temporales son
  "para llamadas del lado del cliente" (browser, para no exponer la clave larga-duración a un
  usuario final); "para llamadas del lado del servidor, tu clave de API debe proveerse en el
  header de la petición WebSocket" tal cual. JARVIS corre nativo en la máquina del usuario, nunca
  expone la clave a un navegador — el caso de uso que el JWT temporal existe para mitigar no
  aplica acá, así que agregar ese flujo (una llamada HTTP extra para pedir un token de 60s antes
  de cada conexión) sería una capa de complejidad sin ningún beneficio de seguridad real en este
  contexto.
- Mensajes cliente→servidor: `StartRecognition` (JSON, config de la sesión), `AddAudio` (binario
  crudo — PCM16 tal cual, SIN el envoltorio base64+JSON que exige `input_audio_buffer.append` de
  OpenAI; confirmado en la referencia de API: "el único mensaje que el Cliente manda al Servidor
  como binario es un chunk de audio"), `EndOfStream` (JSON, declara que no hay más audio — incluye
  `last_seq_no`, la cantidad total de `AddAudio` mandados).
- Mensajes servidor→cliente relevantes acá: `RecognitionStarted` (ack de `StartRecognition`),
  `AddPartialTranscript`/`AddTranscript` (parcial/final, ambos con el texto en
  `metadata.transcript`), `EndOfTranscript` (el servidor terminó de mandar todos los
  `AddTranscript` pendientes tras nuestro `EndOfStream`), `Error`/`Warning`.

Diferencia real de diseño, no trivial, frente a la versión OpenAI — hay que acumular texto, no
esperar un único evento con el texto completo del turno: a diferencia de `gpt-live-transcribe`
(un solo evento `.completed` con la transcripción completa del turno), Speechmatics es streaming
de verdad — manda `AddTranscript` PROGRESIVAMENTE a medida que reconoce tramos de audio (según
`max_delay`), no solo al final. Confirmado en el propio ejemplo oficial de la documentación: un
usuario diciendo dos oraciones seguidas produce DOS eventos `AddTranscript` separados, uno por
oración, mientras la sesión sigue abierta. Este módulo acumula el `metadata.transcript` de cada
`AddTranscript` recibido (`_drain_events`) y recién arma el resultado final, uniendo los tramos con
un espacio, cuando llega `EndOfTranscript` (que el servidor manda una vez que procesó todo el
audio pendiente después de nuestro `EndOfStream`) — el equivalente real de "fin del turno" acá,
no ningún evento individual de transcripción.

Corte de turno: sigue siendo 100% el VAD local (`jarvis.audio.vad`: `chunk_rms`,
`is_speech_chunk`, `should_stop_recording`, más `jarvis.audio.speech_detector` para "¿esto suena a
voz humana?"), nunca el servidor — igual que con OpenAI. A diferencia de `gpt-live-transcribe`
(que RECHAZA turn detection de servidor por completo para este modo, sin alternativa), Speechmatics
sí tiene un mecanismo de servidor equivalente y hay que desactivarlo explícitamente, no asumir que
está apagado: `transcription_config.conversation_config.end_of_utterance_silence_trigger`
("detecta cuándo un hablante dejó de hablar" vía timings de palabras del lado del servidor, y
manda un mensaje `EndOfUtterance` al vencer el silencio configurado). Confirmado en la
documentación real: el default de ese campo es `0`, y "un valor de 0 desactiva la funcionalidad"
— así que NO enviar `conversation_config` en absoluto dentro de `StartRecognition` (como hace este
módulo) deja esa funcionalidad apagada por default del lado del servidor, sin que este módulo
tenga que declarar nada para lograrlo. También existe `ForceEndOfUtterance` (mensaje que el
cliente puede mandar para forzar el corte del lado del servidor en el instante que decida) —
evaluado y descartado: no hace falta, porque acá cada comando abre y cierra su propia sesión
(`EndOfStream` ya cierra el turno actual de forma explícita), así que no hay ningún beneficio sobre
simplemente mandar `EndOfStream` cuando el VAD local decide cortar.

Un solo resample por chunk, no dos con destinos distintos: a diferencia de la Realtime API de
OpenAI (fija a 24kHz, confirmado en su SDK), Speechmatics acepta cualquier `sample_rate` entero en
`audio_format` (confirmado en la documentación real, sin límite mínimo/máximo documentado). Este
módulo elige 16kHz (`jarvis.audio.wake_word.SAMPLE_RATE`, el mismo dominio que ya usa el VAD local)
para que el resample que ya hace falta para el VAD (`resample(..., target_sr=SAMPLE_RATE)`) sea,
sin ningún resample adicional, exactamente el mismo array que se manda por el WebSocket — 16kHz es
además la tasa estándar de la industria para reconocimiento de voz (ver el propio quickstart
oficial de Speechmatics, que usa 16kHz), no un compromiso de calidad.

Sin `additional_vocab` (el equivalente de `keywords`/`TRANSCRIPTION_KEYWORDS` de la versión
OpenAI) — regresión real, aceptada explícitamente, no un descuido: la documentación oficial de
Speechmatics advierte que usar `additional_vocab` introduce "un retraso de hasta 15 segundos"
al iniciar la sesión (degradación de latencia + uso de memoria), dependiendo del tamaño de la
lista. Como este módulo abre una sesión NUEVA por cada comando (no una conexión persistente para
toda la corrida de `pipeline.run()`), pagar ese costo en cada "Alexa, ..." destruiría por completo
el objetivo de baja latencia que motiva usar streaming en primer lugar — mucho peor que la mejora
marginal de reconocimiento que ese vocabulario daba con OpenAI (ahí no tenía ningún costo de
latencia documentado). Si en el futuro se justifica (ej. si `pipeline.py` pasara a reusar una
única conexión de larga duración en vez de una por comando), reevaluar.

Sin `domain="medical"`: el benchmark que motivó esta migración (ver arriba) midió WER sobre un
corpus del dominio médico, pero eso es una característica del corpus de evaluación, no algo que
JARVIS deba pedirle a la API — JARVIS es un asistente de propósito general, así que no se solicita
ningún `domain` especializado.

Sin score de confianza (`logprobs`/`avg_logprob`): igual que con la versión OpenAI, Speechmatics
no expone algo equivalente en este modo — misma regresión ya aceptada y documentada (nunca estuvo
conectada a ningún filtro activo).

Sin `reduce_background_noise`/`normalize_gain` en este camino, mismo motivo que la versión
anterior (operan sobre el clip COMPLETO ya grabado, no por chunk transmitido en vivo — ver
`jarvis.audio.pipeline`).

Puente sounddevice (bloqueante, callback de PortAudio) → asyncio: sin cambios respecto a la
versión OpenAI — `sd.InputStream(callback=...)` empuja cada chunk a una `asyncio.Queue` vía
`loop.call_soon_threadsafe(...)`, consumida por el loop principal de este módulo. Ver el detalle
completo en el historial de este módulo (git log) si hace falta, no se repite acá.

`websockets`, no `speechmatics-rt` (el SDK oficial): evaluado y descartado, con criterio, no por
default. `speechmatics-rt` (confirmado en PyPI/GitHub, `websockets>=10.0` como única dependencia
real) es un cliente async prolijo (`AsyncClient`, `start_session`/`send_audio`/`stop_session`,
handlers de evento vía `.on(ServerMessageType.X, callback)`), pero su modelo de eventos solo
soporta callbacks SÍNCRONOS y no expone públicamente la tarea interna de recepción
(`_recv_task`/`_closed_evt` son atributos privados) — este módulo necesita enganchar
`add_done_callback` sobre esa tarea para resolver el resultado de inmediato si la conexión se cae
sin avisar (ver `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS`/`_on_receive_task_done`, el mismo hallazgo
de seguridad HIGH que motivó ese mecanismo en la versión OpenAI), y depender de un atributo interno
no público de una librería de terceros para una garantía de seguridad es justo el tipo de
fragilidad que este repo evita. Hablar el protocolo documentado directamente contra `websockets`
(ya dependencia dura, agregada originalmente para la Realtime API de OpenAI — ver `pyproject.toml`)
da control total sobre la tarea de recepción sin sumar una dependencia nueva ni depender de
internals ajenos, y el protocolo de Speechmatics es simple (JSON de texto + binario crudo) — no
hay ganancia real de mantenibilidad en la capa de abstracción que da el SDK para este caso puntual.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd
import websockets
from websockets.asyncio.client import ClientConnection

from jarvis.audio.device import input_sample_rate, resolve_input_device
from jarvis.audio.loopback import SystemAudioGate
from jarvis.audio.resample import resample
from jarvis.audio.speech_detector import (
    SPEECH_PROBABILITY_THRESHOLD,
    load_speech_detector,
)
from jarvis.audio.stt import LANGUAGE
from jarvis.audio.vad import chunk_rms, is_speech_chunk, should_stop_recording
from jarvis.audio.wake_word import SAMPLE_RATE

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://global.rt.speechmatics.com/v2"  # enruta a la región más cercana —
# confirmado en la documentación oficial de endpoints soportados; hay endpoints regionales fijos
# (`eu`/`us`) si en algún momento hiciera falta residencia de datos, no es un requisito de JARVIS.
TRANSCRIPTION_MODEL = (
    "enhanced"  # no "standard" (el otro valor no deprecado): la propia
)
# documentación de Speechmatics usa "enhanced" en todos sus ejemplos de voice AI/turn detection, y
# el SDK oficial (`speechmatics-rt`) lo eligió como default cuando no se especifica nada — la
# migración completa se hizo por WER más bajo (ver docstring del módulo), así que no tiene sentido
# pedir el modelo de menor calidad acá. Existe también "melia-1" (próxima generación,
# multilingüe/code-switching) — no elegido: no hay benchmark propio ni de terceros consultado
# todavía que lo respalde para español latinoamericano específicamente, a diferencia de "enhanced".
REALTIME_SAMPLE_RATE = (
    SAMPLE_RATE  # 16kHz — mismo dominio que el VAD local, ver docstring del
)
# módulo (Speechmatics acepta cualquier sample_rate entero, a diferencia de OpenAI).
STREAM_CHUNK_SECONDS = (
    0.2  # mismo tamaño de ventana de VAD que `pipeline.CHUNK_SECONDS` — no se
)
# importa esa constante de `pipeline.py` (import circular, ver docstring del módulo); se repite el
# valor acá como constante propia, standalone.
MAX_DELAY_SECONDS = (
    0.7  # mínimo permitido por la API (rango documentado: 0.7-4, default 4) —
)
# cuánto tarda el servidor en confirmar un tramo de audio como transcripción FINAL una vez que deja
# de haber palabras nuevas ahí. El propio quickstart oficial de Speechmatics para voice AI usa este
# mismo valor mínimo — tiene sentido acá: el objetivo entero de este módulo es baja latencia, así
# que se pide el mínimo permitido en vez del default (4s, pensado para dictado/subtitulado, no
# para un asistente de voz esperando una respuesta).
RECOGNITION_STARTED_TIMEOUT_SECONDS = (
    5.0  # cuánto esperar el ack `RecognitionStarted` antes de
)
# abortar — documentado explícitamente ("después de recibir un mensaje RecognitionStarted, podés
# empezar a mandar audio"): mandar `AddAudio` antes de ese ack no está definido por el protocolo,
# así que este módulo lo espera explícitamente (igual que el propio SDK oficial de Speechmatics)
# en vez de asumir que el servidor lo va a aceptar de todos modos.
TRANSCRIPTION_RESULT_TIMEOUT_SECONDS = (
    10.0  # backstop duro tras `EndOfStream` — mismo orden de magnitud
)
# que los timeouts de red ya existentes en el resto del codebase para un único round-trip
# (`tools.search.REQUEST_TIMEOUT_SECONDS`/`tools.weather.REQUEST_TIMEOUT_SECONDS`, ambos 10.0), y
# el mismo valor que usaba la versión OpenAI de este módulo. En el camino feliz, `EndOfTranscript`
# llega en un round-trip corto (el audio ya se transmitió en vivo mientras se grababa; `EndOfStream`
# solo cierra el turno). Sin este timeout, si la conexión WebSocket se cae a nivel de transporte
# SIN mandar `EndOfTranscript`/`Error` antes de morir, `await result_future` colgaría para siempre
# — congelando todo el loop de voz de `pipeline.run()` (mismo hallazgo de seguridad HIGH que ya
# motivó esto en la versión OpenAI). Ver también `_on_receive_task_done`, que apunta a resolver
# esto de inmediato (no recién al vencer el timeout) — los dos mecanismos coexisten a propósito.


@dataclass(frozen=True)
class SpeechmaticsCredentials:
    """Credenciales + endpoint para abrir una sesión de streaming contra la Realtime API de
    Speechmatics. No hay un objeto "cliente" pesado que instanciar acá (a diferencia de
    `AsyncOpenAI()` en la versión anterior): este módulo habla WebSocket crudo directamente contra
    el protocolo documentado (ver docstring del módulo), así que lo único que hace falta llevar de
    `load_realtime_client()` a `stream_transcribe_command()` es la clave y el endpoint.
    """

    api_key: str
    url: str = REALTIME_URL


def load_realtime_client() -> SpeechmaticsCredentials:
    """Lee `SPEECHMATICS_API_KEY` del entorno (cargado por `jarvis.config.load_dotenv()`), mismo
    patrón que `jarvis.audio.stt.load_stt_client()`. Falla rápido y explícito (`RuntimeError`, sin
    incluir nunca el valor de la clave en el mensaje — `.claude/rules/security.md`, "Higiene de
    secretos") si la variable no está seteada, en vez de esperar a que la primera conexión
    WebSocket falle recién al hablar (401 genérico, más difícil de diagnosticar).
    """
    api_key = os.environ.get("SPEECHMATICS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SPEECHMATICS_API_KEY no está seteada — revisar .env "
            "(cargado por jarvis.config.load_dotenv())."
        )
    return SpeechmaticsCredentials(api_key=api_key)


def _start_recognition_message() -> dict[str, Any]:
    """Mensaje `StartRecognition`: PCM16 crudo a `REALTIME_SAMPLE_RATE`, mismo idioma que el
    camino batch (`jarvis.audio.stt.LANGUAGE`), sin `conversation_config` (deja
    `end_of_utterance_silence_trigger` en su default `0` — detección de turno de servidor
    desactivada, ver docstring del módulo) y sin `additional_vocab` (ver docstring del módulo,
    costo de latencia documentado de hasta 15s por sesión).
    """
    return {
        "message": "StartRecognition",
        "audio_format": {
            "type": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": REALTIME_SAMPLE_RATE,
        },
        "transcription_config": {
            "language": LANGUAGE,
            "model": TRANSCRIPTION_MODEL,
            "enable_partials": True,
            "max_delay": MAX_DELAY_SECONDS,
        },
    }


async def _wait_for_recognition_started(ws: ClientConnection) -> None:
    """Esperar el ack `RecognitionStarted` antes de mandar cualquier `AddAudio` (ver
    `RECOGNITION_STARTED_TIMEOUT_SECONDS`). Tolera cualquier cantidad de mensajes `Info`/`Warning`
    antes del ack — confirmado en la documentación real que `Info` (ej. `recognition_quality`,
    `concurrent_session_usage`) se manda "inmediatamente después de completado el handshake", en
    cualquier momento antes de `RecognitionStarted`. Un `Error` acá (StartRecognition rechazado —
    config inválida, clave inválida ya pasado el handshake HTTP) aborta con una excepción clara en
    vez de colgarse esperando un ack que nunca va a llegar; `stream_transcribe_command` la atrapa
    junto con cualquier otro fallo de conexión/protocolo (ver su docstring).
    """

    async def _wait() -> None:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                continue
            event = json.loads(raw)
            message_type = event.get("message")
            if message_type == "RecognitionStarted":
                return
            if message_type == "Error":
                raise RuntimeError(
                    "Speechmatics rechazó StartRecognition: "
                    f"{event.get('reason', 'sin detalle')}"
                )

    await asyncio.wait_for(_wait(), timeout=RECOGNITION_STARTED_TIMEOUT_SECONDS)


async def _drain_events(
    ws: ClientConnection, result_future: asyncio.Future[str]
) -> None:
    """Consumir eventos del servidor durante toda la sesión, acumulando cada tramo final
    (`AddTranscript.metadata.transcript`) y resolviendo `result_future` recién con
    `EndOfTranscript` — el equivalente real de "fin del turno" acá, no un evento individual (ver
    docstring del módulo: Speechmatics manda `AddTranscript` progresivamente, no un único evento
    con el texto completo). Los `AddPartialTranscript` (texto parcial mientras el usuario sigue
    hablando) solo se loguean a INFO para observabilidad, igual que los `.delta` de la versión
    OpenAI — ningún llamador depende todavía de consumirlos en vivo.

    Tres desenlaces posibles, los tres resuelven `result_future` (nunca la dejan colgada):
    `EndOfTranscript` con el texto acumulado, o `Error` con cadena vacía (se descarta cualquier
    tramo ya acumulado — un error de sesión no garantiza que lo acumulado hasta ahí sea
    confiable) — mismo contrato de "degradar sin lanzar" que ya usa el resto del pipeline ante
    audio ambiguo.
    """
    transcript_parts: list[str] = []
    async for raw in ws:
        if isinstance(raw, bytes):
            logger.warning(
                "stream_transcribe_command(): mensaje binario inesperado del servidor, "
                "ignorado (el protocolo solo define AddAudio como binario, y ese es "
                "cliente→servidor)."
            )
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "stream_transcribe_command(): mensaje no-JSON del servidor, ignorado: %r",
                raw,
            )
            continue
        message_type = event.get("message")
        if message_type == "AddPartialTranscript":
            partial = event.get("metadata", {}).get("transcript", "")
            if partial:
                logger.info("stream_transcribe_command(): parcial en vivo: %r", partial)
        elif message_type == "AddTranscript":
            final = event.get("metadata", {}).get("transcript", "")
            if final:
                transcript_parts.append(final)
        elif message_type == "EndOfTranscript":
            if not result_future.done():
                result_future.set_result(" ".join(transcript_parts).strip())
            return
        elif message_type == "Error":
            logger.warning(
                "stream_transcribe_command(): error del servidor: %s",
                event.get("reason", "sin detalle"),
            )
            if not result_future.done():
                result_future.set_result("")
            return
        elif message_type == "Warning":
            logger.warning(
                "stream_transcribe_command(): warning del servidor: %s",
                event.get("reason", "sin detalle"),
            )
        # `RecognitionStarted` ya se consumió en `_wait_for_recognition_started` antes de que
        # este loop arranque. `AudioAdded`/`Info`/`EndOfUtterance`/`AudioEvent*`/`Speakers*` no
        # aplican a este módulo (no se pide diarización, eventos de audio, ni end-of-utterance de
        # servidor — ver docstring del módulo) y se ignoran en silencio si igual llegaran.


async def _capture_and_stream(
    ws: ClientConnection,
    *,
    device: int | None,
    silence_threshold: float,
    max_duration: float,
    pre_roll: np.ndarray | None,
    system_audio: SystemAudioGate | None,
    now_fn: Callable[[], float] = time.monotonic,
) -> tuple[str, bool]:
    """Capturar el mic en modo callback, transmitiendo cada chunk al WebSocket a medida que llega
    y decidiendo con el VAD local (`jarvis.audio.vad`) cuándo cortar y mandar `EndOfStream` — o,
    si nunca hubo habla real, cortar sin mandarlo en absoluto (mismo contrato que
    `pipeline.record_command`/`speech_detected`: nada que transcribir, nada que confirmar de más).

    `now_fn` (default `time.monotonic`, reloj real) es inyectable a propósito: los tests
    (`tests/audio/test_realtime_stt.py`) entregan todos los chunks pre-armados de una, sin pausas
    reales entre ellos, así que con el reloj real el tiempo transcurrido apenas avanzaría entre
    iteraciones — un reloj falso de incremento fijo les deja seguir siendo deterministas y rápidos
    sin simular pausas reales con `asyncio.sleep`.
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
    receive_task = asyncio.create_task(_drain_events(ws, result_future))

    def _on_receive_task_done(task: asyncio.Task[None]) -> None:
        # Backstop rápido para una conexión que se cae a nivel de transporte (red, cierre del
        # lado del servidor, timeout de ping/pong de `websockets`) SIN mandar nunca
        # `EndOfTranscript`/`Error`: en ese caso el `async for raw in ws` de `_drain_events`
        # termina con una excepción sin haber tocado `result_future` — nada más lo resolvería.
        # Este callback corre apenas `receive_task` termina, de la forma que sea, así que
        # `result_future` se resuelve de inmediato en vez de depender solo de
        # `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS` como único mecanismo de recuperación (ese
        # timeout sigue como backstop duro por si este callback tuviera algún hueco no previsto
        # — los dos mecanismos coexisten a propósito, mismo hallazgo de seguridad HIGH que
        # motivó esto en la versión OpenAI de este módulo).
        if result_future.done() or task.cancelled():
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
    chunks_sent = 0
    try:
        if pre_roll is not None and len(pre_roll) > 0:
            # `pre_roll` ya viene resampleado a `SAMPLE_RATE` (16kHz — ver
            # `pipeline.PRE_ROLL_FRAMES`/`iter_microphone_frames`), que es exactamente
            # `REALTIME_SAMPLE_RATE`: a diferencia de la versión OpenAI (24kHz fijo, un resample
            # aparte solo para mandar), acá no hace falta ningún resample para el envío — ver
            # docstring del módulo.
            await ws.send(pre_roll.astype(np.int16).tobytes())
            chunks_sent += 1

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
                # fijo por iteración sumado repetidamente — mismo razonamiento que la versión
                # OpenAI de este módulo (bug real, en vivo, encontrado esa sesión: un incremento
                # fijo subestima el tiempo real bajo latencia de red variable).
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
                # `vad_chunk` (int16, ya a 16kHz) se manda tal cual — mismo array que ya usó el
                # VAD de arriba, sin ningún resample adicional (ver docstring del módulo/
                # comentario de `pre_roll`).
                await ws.send(vad_chunk.astype(np.int16).tobytes())
                chunks_sent += 1
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
            # `EndOfStream`, es la mitigación real contra alucinaciones sobre silencio puro, no
            # un ahorro de red incidental.
            return "", False

        await ws.send(
            json.dumps({"message": "EndOfStream", "last_seq_no": chunks_sent})
        )
        try:
            text = await asyncio.wait_for(
                result_future, timeout=TRANSCRIPTION_RESULT_TIMEOUT_SECONDS
            )
        except TimeoutError:
            # Backstop duro (ver `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS`): ni `_drain_events` ni
            # `_on_receive_task_done` resolvieron nada dentro del plazo — degradar sin lanzar,
            # mismo contrato que `Error` acá arriba (hubo habla real, sí se mandó `EndOfStream`,
            # pero la sesión nunca devolvió `EndOfTranscript`).
            logger.warning(
                "stream_transcribe_command(): sin resultado de transcripción %.0fs después de "
                "EndOfStream — degradando a cadena vacía en vez de colgar el loop de voz.",
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
    client: SpeechmaticsCredentials,
    device: int | None,
    silence_threshold: float,
    max_duration: float,
    pre_roll: np.ndarray | None = None,
    system_audio: SystemAudioGate | None = None,
    now_fn: Callable[[], float] = time.monotonic,
) -> tuple[str, bool]:
    """Grabar y transcribir el comando dicho después de la wake word en streaming, vía la
    Realtime API de Speechmatics — reemplazo, en `pipeline.run()`, del combo anterior (OpenAI,
    ADR-0007) para la captura de comandos generales (ver docstring del módulo para qué NO migra y
    por qué, y para las diferencias de protocolo frente a la versión anterior).

    Devuelve `(texto, speech_detected)` — mismo contrato que antes: `speech_detected` es `True`
    solo si el VAD local (`jarvis.audio.vad`) detectó habla real durante la ventana de grabación.
    Con `speech_detected=False`, `texto` es siempre `""` y nunca se llegó a mandar `EndOfStream`
    en la sesión — el llamador puede seguir tratando ese caso exactamente igual que antes.

    `silence_threshold`/`max_duration`/`pre_roll`/`system_audio` tienen el mismo significado
    exacto que en `pipeline.record_command`/la versión anterior de este módulo.

    A diferencia de la versión OpenAI (que dejaba propagar sin atrapar un fallo de conexión
    inicial — `client.realtime.connect()` — confiando en el `except Exception` de nivel de turno
    de `pipeline.run()` como red de seguridad), esta versión atrapa CUALQUIER excepción de la
    sesión completa (conexión, autenticación, handshake, `StartRecognition` rechazado, error de
    protocolo) acá mismo y degrada a `("", False)` en vez de propagar — pedido explícito para este
    módulo: "degradar ante cualquier fallo (conexión, error del servidor, timeout), nunca
    lanzar". El resultado funcional para `pipeline.run()` es equivalente en ambos diseños (el
    turno se pierde, el loop de voz sigue vivo), pero acá queda expresado como el mismo tipo de
    resultado "no se detectó habla real" en vez de como una excepción de turno genérica.
    """
    try:
        async with websockets.connect(
            client.url,
            additional_headers={"Authorization": f"Bearer {client.api_key}"},
        ) as ws:
            await ws.send(json.dumps(_start_recognition_message()))
            await _wait_for_recognition_started(ws)
            return await _capture_and_stream(
                ws,
                device=device,
                silence_threshold=silence_threshold,
                max_duration=max_duration,
                pre_roll=pre_roll,
                system_audio=system_audio,
                now_fn=now_fn,
            )
    except Exception as exc:  # noqa: BLE001 — frontera de recuperación explícitamente
        # documentada (`.claude/rules/python.md`): cualquier fallo de conexión/autenticación/
        # protocolo antes o durante la sesión degrada a "no se detectó habla real" en vez de
        # propagar (ver docstring de esta función). `asyncio.CancelledError`/`KeyboardInterrupt`
        # no son `Exception` (heredan de `BaseException` desde Python 3.8) — nunca se atrapan acá.
        logger.warning(
            "stream_transcribe_command(): no se pudo completar la sesión de streaming "
            "(conexión, autenticación, o error de protocolo) — degradando a cadena vacía en vez "
            "de propagar: %r",
            exc,
        )
        return "", False
