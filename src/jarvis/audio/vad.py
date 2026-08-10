"""Voice Activity Detection (VAD) local y puro de JARVIS: decidir si un chunk de audio es habla
real y cuándo cortar una grabación por silencio sostenido.

Extraído de `jarvis.audio.pipeline` (que originalmente definía estas funciones junto al resto del
pipeline batch) para que `jarvis.audio.realtime_stt` (transcripción en streaming vía la Realtime
API de OpenAI, `gpt-live-transcribe`) pueda reusar exactamente la misma lógica de corte sin crear
un import circular: `pipeline.py` importa `realtime_stt.py` para invocar la transcripción en vivo
desde `run()`, así que `realtime_stt.py` no puede importar de vuelta desde `pipeline.py`. Este
módulo no depende de audio (`sounddevice`) ni de red — son funciones puras sobre arrays de numpy
y floats, deliberadamente sin estado ni I/O, para que ambos módulos (y sus tests) puedan
importarlas sin ciclo.

`jarvis.audio.pipeline` sigue re-exportando `chunk_rms`/`is_speech_chunk`/`should_stop_recording`/
`TRAILING_SILENCE_SECONDS` (import simple, sin redefinir nada) para no romper ningún uso existente
que ya las busca en `jarvis.audio.pipeline` — este movimiento es de implementación, no un cambio
de contrato público ni de comportamiento.

`is_speech_chunk` ya NO decide solo por RMS: el RMS por sí solo confunde ruido fuerte no-vocal
(clics de mouse, efectos de sonido de un juego) con voz real — solo mide volumen, no si *suena*
a voz humana. Confirmado en vivo, en producción: 30% de los chunks de una sesión jugando League
of Legends se marcaban como "voz" por RMS, la mayoría ruido del juego. El RMS ahora es un
pre-filtro barato (`min_rms_floor` — chunks muy por debajo del piso de ruido calibrado ni
siquiera necesitan más análisis); la decisión real la toma `speech_probability`, calculada por
`jarvis.audio.speech_detector.ChunkSpeechDetector` (Silero VAD, modelo neuronal entrenado para
esto — NO vive en este módulo, tiene estado real y carga un modelo, rompería la pureza de este
archivo). `speech_probability=None` degrada a la decisión de solo-RMS de antes de este cambio
(nunca tumba la captura de voz por una falla al cargar el modelo).
"""

from __future__ import annotations

import numpy as np

TRAILING_SILENCE_SECONDS = (
    1.2  # cuánto silencio sostenido después de hablar antes de cortar solo
)


def chunk_rms(chunk: np.ndarray) -> float:
    """RMS de un chunk de audio int16 — mide qué tan fuerte es la señal en ese instante."""
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def is_speech_chunk(
    rms: float,
    *,
    min_rms_floor: float,
    speech_probability: float | None,
    speech_probability_threshold: float,
    system_is_loud: bool,
) -> bool:
    """Decidir si un chunk cuenta como habla real.

    Tres condiciones, todas deben cumplirse (AND, no OR):

    1. El sistema no está sonando fuerte en ese instante — audio del sistema (juego, música)
       filtrándose al mic puede subir su RMS/probabilidad sin que el usuario esté hablando;
       mientras el sistema suena fuerte, ese chunk nunca cuenta como habla, sin importar qué
       digan las otras dos señales (`loopback.py`: no reemplaza cancelación de eco real, es un
       gate binario aceptado como alcance reducido).
    2. RMS por encima de `min_rms_floor` — pre-filtro barato: un chunk muy por debajo del piso de
       ruido calibrado (`calibrate_thresholds`) es silencio real, ni vale la pena preguntarle al
       modelo de voz.
    3. `speech_probability` (si está disponible) por encima de `speech_probability_threshold` —
       la decisión real. Si `speech_probability` es `None` (el detector de voz, ver
       `jarvis.audio.speech_detector`, no cargó o está deshabilitado), esta condición se omite y
       se degrada a decidir solo por RMS — nunca debe tumbar la captura de voz completa porque
       una mejora opcional no está disponible.
    """
    if system_is_loud:
        return False
    if rms < min_rms_floor:
        return False
    if speech_probability is None:
        return True
    return speech_probability >= speech_probability_threshold


def should_stop_recording(
    *,
    speech_started: bool,
    silence_run_seconds: float,
    elapsed_seconds: float,
    max_seconds: float,
) -> bool:
    """Decidir si cortar la grabación del comando: por tope duro (`max_seconds`), o por
    silencio sostenido (`TRAILING_SILENCE_SECONDS`) una vez que ya hubo habla real.

    Nunca corta antes del tope solo por silencio si todavía no se detectó habla — el usuario
    puede tardar un instante en arrancar a hablar después de la wake word, y cortar ahí sería
    peor que esperar los `max_seconds` completos como antes.
    """
    if elapsed_seconds >= max_seconds:
        return True
    return speech_started and silence_run_seconds >= TRAILING_SILENCE_SECONDS
