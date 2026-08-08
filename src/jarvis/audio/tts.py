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
from pathlib import Path
from typing import Protocol

import pyttsx3
from openai import OpenAI
from playsound3 import playsound

DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
# "nova" sonaba muy marcada al inglés en español (confirmado en vivo, pedido explícito del
# usuario). "cedar" es una de las dos voces que OpenAI recomienda para casos donde la calidad
# importa (la otra es "marin") y salió más rápida en la comparación en vivo (~6.9s vs ~8.9s de
# marin con el mismo texto) — empate roto por velocidad, no hay forma de que yo "escuche" cuál
# suena mejor.
DEFAULT_OPENAI_TTS_VOICE = "cedar"
# OpenAI adapta el idioma automáticamente al texto de entrada (no hace falta un voice_id
# específico de español, a diferencia de edge-tts).


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
    ) -> None:
        self._client = client if client is not None else OpenAI()
        self._model = model
        self._voice = voice

    def speak(self, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "speech.mp3"
            with self._client.audio.speech.with_streaming_response.create(
                model=self._model, voice=self._voice, input=text
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
