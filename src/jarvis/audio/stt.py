"""Transcripción de audio a texto vía la API de OpenAI (`gpt-4o-transcribe`).

Alcance de esta fase: transcribir un clip de audio ya grabado (después de detectar la wake
word). Nada de LLM ni TTS todavía (ADR-0004).

Reemplaza el Whisper local (`faster-whisper`): en pruebas en vivo, la API entendió correctamente
audio que el modelo local ("medium", GPU) devolvió vacío — más robusta ante audio no perfecto.
Depende de red y de `OPENAI_API_KEY` (ver `.env.example`).
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np
from openai import Omit, OpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-transcribe"
LANGUAGE = "es"
# Decoding determinístico: 0.0 minimiza la aleatoriedad del muestreo interno del modelo, que es
# justamente lo que produce alucinaciones "creativas" sobre audio ambiguo/ruidoso (texto en otro
# idioma, frases inventadas) — mismo mecanismo, a nivel de API, que motivó filtrar transcripciones
# en script no latino en `jarvis.audio.pipeline._contains_non_latin_script`. Sin este parámetro
# antes, se usaba lo que sea que la API tome por default (no documentado como 0.0).
TEMPERATURE = 0.0
# Sin hint de vocabulario (`prompt`) a propósito — se probaron dos versiones (una oración
# completa "JARVIS es un asistente de voz.", después una lista de palabras "Alexa, Hey Mycroft,
# Daniel") y las dos terminaron alucinadas de vuelta como si fueran lo que dijo el usuario en
# audio ambiguo/silencio (`Dijiste: 'JARVIS es un asistente de voz.'`, después
# `Dijiste: 'Alexa, Hey Mycroft, Daniel'`, ambas con el usuario en silencio real) — y esas
# transcripciones fantasma llegaron a contaminar la memoria (el LLM guardó "el usuario llama a
# su asistente JARVIS" a partir de eso). Cualquier texto en `prompt` corre ese riesgo, sea
# oración o lista de palabras; "Alexa"/"Daniel" además no son ambiguos como sí lo era "Jarvis"
# (múltiples transcripciones erróneas distintas), así que el hint ya no compensa el riesgo.
PROMPT: str | None = None


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

    Pide `logprobs` por token (`include=["logprobs"]`, soportado por `gpt-4o-transcribe` con
    `response_format="json"` — confirmado inspeccionando el SDK instalado, no documentación
    externa) y loguea el promedio a nivel INFO (ver `logging.basicConfig` en
    `jarvis.audio.pipeline.run()` — sin eso, INFO/DEBUG son no-ops silenciosos), sin filtrar
    nada todavía en base a esto: no hay
    todavía un umbral calibrado contra valores reales de esta cuenta/modelo (mismo criterio que
    `DEFAULT_LOUDNESS_THRESHOLD` en `jarvis.audio.loopback` — no se fija un número sin medir
    primero). El objetivo es juntar evidencia real la próxima vez que haya una alucinación tipo
    "Der Kontrollenduetum" (alfabeto latino, no la agarra el filtro de script no latino de
    `jarvis.audio.pipeline`) para poder calibrar un filtro de confianza real más adelante.
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
        prompt=PROMPT if PROMPT is not None else Omit(),
        temperature=TEMPERATURE,
        response_format="json",
        include=["logprobs"],
    )
    if result.logprobs:
        avg_logprob = sum(
            lp.logprob for lp in result.logprobs if lp.logprob is not None
        ) / len(result.logprobs)
        logger.info(
            "transcribe(): confianza promedio (logprob) = %.3f sobre %d tokens — %r",
            avg_logprob,
            len(result.logprobs),
            result.text,
        )
    return result.text.strip()
