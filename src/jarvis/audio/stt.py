"""Transcripción de audio a texto vía la API de OpenAI (`gpt-4o-transcribe`).

Alcance de esta fase: transcribir un clip de audio ya grabado (después de detectar la wake
word). Nada de LLM ni TTS todavía (ADR-0004).

Reemplaza el Whisper local (`faster-whisper`): en pruebas en vivo, la API entendió correctamente
audio que el modelo local ("medium", GPU) devolvió vacío — más robusta ante audio no perfecto.
Depende de red y de `OPENAI_API_KEY` (ver `.env.example`).
"""

from __future__ import annotations

import io
import wave

import numpy as np
from openai import Omit, OpenAI

MODEL = "gpt-4o-transcribe"
LANGUAGE = "es"
# Hint de vocabulario: la API lo usa para sesgar ortografía/reconocimiento de palabras
# ambiguas, no cambia el idioma de salida. Confirmado en vivo que "Jarvis" se transcribe mal
# de formas distintas cada vez ("Jorvis", "Garbis", "Garrechu") — es un nombre propio en
# inglés dentro de una frase en español, justo el caso que este hint corrige.
PROMPT = "JARVIS es un asistente de voz."


def load_stt_client() -> OpenAI:
    """Cliente de transcripción. Lee `OPENAI_API_KEY` del entorno (cargado por `config.load_dotenv()`)."""
    return OpenAI()


def transcribe(
    audio: np.ndarray,
    *,
    client: OpenAI,
    sample_rate: int = 16_000,
    language: str | None = LANGUAGE,
) -> str:
    """Transcribir audio int16 mono a texto plano vía la API de OpenAI.

    Empaqueta el audio como WAV en memoria (la API espera un archivo, no un array crudo) y le
    pone `.name` porque el SDK de OpenAI usa la extensión del nombre para el content-type.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    buffer.seek(0)
    buffer.name = "audio.wav"
    result = client.audio.transcriptions.create(
        model=MODEL,
        file=buffer,
        language=language if language is not None else Omit(),
        prompt=PROMPT,
    )
    return result.text.strip()
