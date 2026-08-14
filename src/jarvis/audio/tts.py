"""Texto a voz detrás de una interfaz swappable.

Default: SAPI local vía `pyttsx3` (ADR-0013, revierte ADR-0011 sobre cuál es el default —
ADR-0011 sigue vigente en la parte que importa: nunca hay un fallback en cascada, un solo
`TTSClient` a la vez, sin degradación silenciosa a mitad de turno). Motivo del cambio: en uso
real, la API de OpenAI (`gpt-4o-mini-tts`, la voz "bonita") depende de crédito de cuenta que se
agotó más de una vez y dejó a JARVIS completamente mudo — y el usuario, en vivo, dijo
explícitamente que no le importa la calidad de la voz ("para qué quiero que me hablen bonito"),
que lo que sí le importa es que el transcriptor (STT) entienda bien, ya resuelto aparte
(Speechmatics, ADR-0012). SAPI es local, gratis, sin red ni cuenta — nunca depende de crédito
externo. `OpenAITTSClient` se deja definida (interfaz `TTSClient` sigue siendo swappable a
propósito) por si en el futuro se quiere volver a la voz neural, pero no es el default.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Protocol

import pyttsx3
from openai import OpenAI
from playsound3 import playsound

DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
# "nova" sonaba muy marcada al inglés en español (confirmado en vivo, pedido explícito del
# usuario). Se probó "cedar" (una de las voces que OpenAI recomienda por calidad) pero resultó
# sonar masculina — el usuario pidió explícitamente una voz femenina. "shimmer" es, junto con
# "nova", una de las dos voces que toda la documentación confirma como femenina (las voces
# nuevas — marin/cedar/coral/sage/ash/ballad/verse — no tienen género documentado en ningún
# lado, no vale la pena arriesgar otra adivinanza); confirmada en vivo con el usuario que suena
# femenina y menos marcada al inglés que nova.
DEFAULT_OPENAI_TTS_VOICE = "shimmer"
# OpenAI adapta el idioma automáticamente al texto de entrada (no hace falta un voice_id
# específico de español, a diferencia de edge-tts) — pero por default el acento sigue sonando
# marcado al inglés/estadounidense (confirmado en vivo, "shimmer" solo no alcanzó). El modelo
# `gpt-4o-mini-tts` es "steerable": acepta `instructions` para dirigir cómo hablar (acento, tono),
# no solo qué decir. Con esta instrucción explícita de acento colombiano/latinoamericano, sin
# inglés, el usuario confirmó en vivo que suena mucho mejor.
DEFAULT_OPENAI_TTS_INSTRUCTIONS = (
    "Hablá en español latinoamericano neutro, con acento colombiano natural. Nunca uses acento "
    "ni entonación en inglés/estadounidense. Voz cálida, femenina, natural, como una persona "
    "real hablando español de nacimiento."
)


class TTSClient(Protocol):
    def speak(self, text: str) -> None:
        """Reproducir `text` en voz, bloqueando hasta que termina."""
        ...


class OpenAITTSClient:
    """Primario: voz neural vía la API de OpenAI."""

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        model: str = DEFAULT_OPENAI_TTS_MODEL,
        voice: str = DEFAULT_OPENAI_TTS_VOICE,
        instructions: str = DEFAULT_OPENAI_TTS_INSTRUCTIONS,
    ) -> None:
        self._client = client if client is not None else OpenAI()
        self._model = model
        self._voice = voice
        self._instructions = instructions

    def speak(self, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "speech.mp3"
            with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=self._voice,
                input=text,
                instructions=self._instructions,
            ) as response:
                response.stream_to_file(str(out_path))
            playsound(str(out_path))


class SapiTTSClient:
    """Voz local de Windows vía SAPI (`pyttsx3`) — sin red, sin cuenta, sin crédito que se pueda
    agotar. Default desde ADR-0013 (ver docstring del módulo)."""

    def speak(self, text: str) -> None:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


def load_default_tts_client() -> TTSClient:
    """Cliente TTS por defecto: `SapiTTSClient` (ADR-0013) — local, gratis, nunca depende de
    crédito externo. `OpenAITTSClient` sigue disponible como alternativa swappable, no es el
    default."""
    return SapiTTSClient()


class LockingTTSClient:
    """Envuelve otro `TTSClient` serializando llamadas a `.speak()` con un `threading.Lock`.

    Hasta la introducción de `jarvis.audio.timer_scheduler.TimerScheduler`, todo llamador de
    `speak()` en este codebase quedaba, de hecho, serializado — `VoiceConfirmationChannel.ask()`
    lo llama vía `asyncio.to_thread` desde un thread aparte, pero siempre *dentro* de un
    `await` que el loop de `dispatch_turn`/`run()` espera antes de seguir, así que dos llamadas a
    `speak()` nunca corrían de verdad al mismo tiempo. `TimerScheduler` rompe ese supuesto: es un
    thread de fondo genuinamente independiente que puede llamar `tts.speak()` en cualquier
    momento, incluso mientras el loop principal está a mitad de decir otra cosa.

    Riesgo real y concreto con el default actual (`SapiTTSClient`, ADR-0013): crea un
    `pyttsx3.init()` nuevo por llamada, y SAPI vía COM en Windows no está garantizado thread-safe
    sin inicialización explícita de apartment por thread — dos `speak()` concurrentes cayendo en
    SAPI podrían interferir entre sí de formas no obvias. Sin haber podido confirmar esto en vivo
    (haría falta forzar dos llamadas simultáneas), este wrapper es la mitigación barata: un lock
    global sobre la reproducción real hace que "concurrente" se convierta en "en cola, uno después
    del otro" — nunca simultáneo — a costo de que un anuncio de timer/recordatorio pueda esperar a
    que termine de hablar lo que esté sonando en ese momento, que es exactamente el comportamiento
    esperado (dos voces superpuestas serían peor experiencia que una cola corta). Si el default
    volviera a `OpenAITTSClient` en el futuro, el riesgo es menor pero no nulo (escribe a un
    archivo temporal *propio* por llamada, sin colisión de paths, pero `playsound3.playsound()` no
    documenta explícitamente ser reentrante) — el lock se queda de cualquier forma, no es
    específico de un solo `TTSClient`.

    Usado en `jarvis.audio.pipeline.run()` envolviendo el `TTSClient` que ya devuelve
    `load_default_tts_client()` — nunca cambia el tipo que devuelve esa función, la envoltura se
    aplica en el punto de uso.
    """

    def __init__(self, *, inner: TTSClient, lock: threading.Lock | None = None) -> None:
        self._inner = inner
        self._lock = lock if lock is not None else threading.Lock()

    def speak(self, text: str) -> None:
        with self._lock:
            self._inner.speak(text)
