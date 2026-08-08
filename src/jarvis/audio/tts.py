"""Texto a voz detrás de una interfaz swappable (ADR-0004).

Primario: `edge-tts` (calidad neural, voz "es-CO-SalomeNeural" — femenina, español colombiano,
pedido explícito del usuario al renombrar la identidad hablada a "Alexa") — usa un endpoint no
oficial de Microsoft, funcional y gratis pero puede romperse o bloquearse sin aviso porque no es
una API soportada. Fallback local obligatorio: SAPI vía `pyttsx3`, siempre disponible en Windows,
sin red — JARVIS nunca queda mudo por depender de algo externo.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Protocol

import edge_tts
import pyttsx3
from playsound3 import playsound

DEFAULT_EDGE_VOICE = "es-CO-SalomeNeural"


class TTSClient(Protocol):
    def speak(self, text: str) -> None:
        """Reproducir `text` en voz, bloqueando hasta que termina."""
        ...


class EdgeTTSClient:
    """Primario: voz neural vía `edge-tts`."""

    def __init__(self, *, voice: str = DEFAULT_EDGE_VOICE) -> None:
        self._voice = voice

    def speak(self, text: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "speech.mp3"
            communicate = edge_tts.Communicate(text, voice=self._voice)
            asyncio.run(communicate.save(str(out_path)))
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
    """Cliente TTS con el contrato de ADR-0004: edge-tts primario, SAPI de fallback."""
    return FallbackTTSClient(primary=EdgeTTSClient(), fallback=SapiTTSClient())
