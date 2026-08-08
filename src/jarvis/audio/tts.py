"""Texto a voz detrás de una interfaz swappable (ADR-0004).

Primario: API de OpenAI (`gpt-4o-mini-tts`) — mismo proveedor que ya usamos para STT (misma
`OPENAI_API_KEY`, sin cuenta nueva). Reemplaza a `edge-tts` (medido en vivo: generación de
~8.7s en una respuesta corta, el cuello de botella real de latencia de todo el pipeline —
`edge-tts` es un endpoint no oficial de Microsoft, sin garantía de rendimiento). Fallback local
obligatorio: SAPI vía `pyttsx3`, siempre disponible en Windows, sin red — JARVIS nunca queda
mudo por depender de algo externo, sea cual sea el primario.
"""

from __future__ import annotations

import sys
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
    """Fallback local: SAPI vía `pyttsx3`. Siempre disponible en Windows, sin red ni dependencia externa."""

    def speak(self, text: str) -> None:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


class FallbackTTSClient:
    """Intenta el primario; si falla por lo que sea, cae al fallback local.

    Excepción amplia deliberada: esta es exactamente la "capa de recuperación explícitamente
    documentada" que `.claude/rules/python.md` permite como excepción a "nunca except Exception
    silencioso" — es el punto central del contrato de ADR-0004 (nunca mudo). No silenciosa del
    todo: avisa por stderr cuándo cayó al fallback, para que el fallo del primario sea visible
    aunque no sea fatal.
    """

    def __init__(self, *, primary: TTSClient, fallback: TTSClient) -> None:
        self._primary = primary
        self._fallback = fallback

    def speak(self, text: str) -> None:
        try:
            self._primary.speak(text)
        except Exception as exc:  # noqa: BLE001 — fallback documentado (ADR-0004), no descuido
            print(
                f"TTS primario falló ({exc!r}), usando fallback local.", file=sys.stderr
            )
            self._fallback.speak(text)


def load_default_tts_client() -> TTSClient:
    """Cliente TTS con el contrato de ADR-0004: OpenAI TTS primario, SAPI de fallback."""
    return FallbackTTSClient(primary=OpenAITTSClient(), fallback=SapiTTSClient())


class LockingTTSClient:
    """Envuelve otro `TTSClient` serializando llamadas a `.speak()` con un `threading.Lock`.

    Hasta la introducción de `jarvis.audio.timer_scheduler.TimerScheduler`, todo llamador de
    `speak()` en este codebase quedaba, de hecho, serializado — `VoiceConfirmationChannel.ask()`
    lo llama vía `asyncio.to_thread` desde un thread aparte, pero siempre *dentro* de un
    `await` que el loop de `dispatch_turn`/`run()` espera antes de seguir, así que dos llamadas a
    `speak()` nunca corrían de verdad al mismo tiempo. `TimerScheduler` rompe ese supuesto: es un
    thread de fondo genuinamente independiente que puede llamar `tts.speak()` en cualquier
    momento, incluso mientras el loop principal está a mitad de decir otra cosa.

    No hay evidencia concreta de que las implementaciones de `TTSClient` de este módulo se rompan
    con llamadas concurrentes (`OpenAITTSClient.speak` escribe a un archivo temporal *propio* por
    llamada — `tempfile.TemporaryDirectory()` por invocación, sin colisión de paths — y
    `playsound3.playsound()` bloquea hasta terminar su propia reproducción; probablemente ambas
    llamadas simplemente reproducirían audio superpuesto en vez de fallar). El riesgo real y
    concreto es distinto: `SapiTTSClient.speak()` crea un `pyttsx3.init()` nuevo por llamada, y
    SAPI vía COM en Windows no está garantizado thread-safe sin inicialización explícita de
    apartment por thread — dos `speak()` concurrentes cayendo al fallback local podrían
    interferir entre sí de formas no obvias. Sin haber podido confirmar esto en vivo (haría falta
    forzar el fallback dos veces en simultáneo), este wrapper es la mitigación barata: un lock
    global sobre la reproducción real hace que "concurrente" se convierta en "en cola, uno
    después del otro" — nunca simultáneo — a costo de que un anuncio de timer/recordatorio pueda
    esperar a que termine de hablar lo que esté sonando en ese momento, que es exactamente el
    comportamiento esperado (dos voces superpuestas serían peor experiencia que una cola corta).

    Usado en `jarvis.audio.pipeline.run()` envolviendo el `TTSClient` que ya devuelve
    `load_default_tts_client()` — nunca cambia el tipo que devuelve esa función (no rompe
    `test_load_default_tts_client_returns_a_fallback_tts_client`), la envoltura se aplica en el
    punto de uso.
    """

    def __init__(self, *, inner: TTSClient, lock: threading.Lock | None = None) -> None:
        self._inner = inner
        self._lock = lock if lock is not None else threading.Lock()

    def speak(self, text: str) -> None:
        with self._lock:
            self._inner.speak(text)
