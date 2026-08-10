"""Detector de voz real vía Silero VAD — reemplaza el RMS/volumen como criterio de "¿esto es
voz?" en `jarvis.audio.pipeline`/`jarvis.audio.realtime_stt`.

Por qué existe: el VAD por RMS (`jarvis.audio.vad.is_speech_chunk`, calibrado contra el piso de
ruido al arrancar) confunde ruido fuerte no-vocal (clics de mouse, efectos de sonido de un juego)
con voz real — solo mide volumen, no si *suena* a voz humana. Confirmado en vivo, en producción,
esta sesión: de 2951 chunks medidos jugando League of Legends, 875 (30%) se marcaron como "voz"
por RMS, la mayoría ruido del juego, no el usuario hablando. Se evaluó dejar que la Realtime API
de OpenAI decida esto en la nube (`server_vad`/`semantic_vad`) — descartado: confirmado en vivo
contra la API real que `gpt-live-transcribe` en modo `type: "transcription"` rechaza turn
detection por completo ("Turn detection is not supported for this transcription model"). Hay que
resolverlo local.

Dependencia: `silero-vad-lite` (PyPI, paquete de un solo mantenedor, no el oficial del equipo
Silero) — deliberado, no un compromiso a la baja: el paquete OFICIAL `silero-vad` declara
`torch>=1.12.0`/`torchaudio>=0.12.0` como dependencias duras (verificado contra el JSON real de
PyPI antes de decidir, no asumido) — exactamente el tipo de dependencia pesada que este repo
viene evitando toda la sesión (mismo motivo que se descartó `noisereduce` por arrastrar
`matplotlib` para la reducción de ruido de `pipeline.py`). `silero-vad-lite` tiene CERO
dependencias pip en runtime — bundlea ONNX Runtime como extensión compilada (wheels Windows
`cp312-win_amd64` confirmados). Trade-off aceptado: menos "oficial", mucho más liviano.

`jarvis.audio.vad` documenta explícitamente que es un módulo de funciones puras, sin estado ni
I/O, a propósito (evita un ciclo de import entre `pipeline.py`/`realtime_stt.py`). Este módulo
NO vive ahí: carga un modelo real y mantiene un buffer con estado entre llamadas, así que rompería
esa invariante — vive aparte, y sus llamadores pasan la probabilidad ya calculada a
`vad.is_speech_chunk` como un float más, no un objeto con estado.
"""

from __future__ import annotations

import logging

import numpy as np
from silero_vad_lite import SileroVAD

logger = logging.getLogger(__name__)

SAMPLE_RATE = (
    16_000  # el único otro valor soportado por SileroVAD es 8000 -- no lo usamos, el
)
# resto del pipeline ya normaliza todo a 16kHz (jarvis.audio.wake_word.SAMPLE_RATE)

# Ventana fija que exige `SileroVAD.process()` — no un valor elegido, es un requisito del modelo
# (32ms a 16kHz), confirmado contra el paquete instalado (`SileroVAD(16000).window_size_samples`).
WINDOW_SAMPLES = 512

# Umbral estándar de Silero (su propia documentación lo usa como default) — a diferencia del
# umbral RMS de `vad.calibrate_thresholds`, este NO se deriva de una calibración de ambiente: la
# probabilidad que devuelve el modelo ya viene normalizada por su propio entrenamiento, no
# depende de cuán ruidoso esté el cuarto en esta sesión particular.
SPEECH_PROBABILITY_THRESHOLD = 0.5


class ChunkSpeechDetector:
    """Adapta `SileroVAD` (que exige exactamente `WINDOW_SAMPLES` por llamada) a los chunks de
    tamaño arbitrario que ya produce el pipeline (~3200 samples a 16kHz, `CHUNK_SECONDS`/
    `STREAM_CHUNK_SECONDS` en `pipeline.py`/`realtime_stt.py` — no es múltiplo exacto de 512).

    Mantiene un buffer interno de samples sobrantes entre llamadas a `speech_probability` — cada
    instancia es para UNA sesión de grabación (una llamada a `record_command`/
    `_capture_and_stream`), no compartida entre comandos distintos: `SileroVAD` no expone ningún
    método de reset de estado interno (verificado contra la API instalada, no asumido — es un
    modelo con memoria temporal tipo RNN, es esperable que SÍ tenga estado entre `.process()`),
    así que la forma segura de garantizar que el contexto de un comando no se filtre al
    siguiente es construir una instancia nueva (y por lo tanto un `SileroVAD` nuevo) por sesión,
    no reusar una sola para todo el proceso como sí hace `wake_word.load_model()` con su modelo.
    """

    def __init__(self) -> None:
        self._vad = SileroVAD(SAMPLE_RATE)
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)

    def speech_probability(self, chunk_int16: np.ndarray) -> float:
        """Probabilidad de voz (0.0-1.0) para `chunk_int16` (mono, int16, a `SAMPLE_RATE`).

        `chunk_int16` casi nunca mide un múltiplo exacto de `WINDOW_SAMPLES` — se acumula en un
        buffer interno y se procesa una ventana de `WINDOW_SAMPLES` a la vez; el resto que no
        llega a completar una ventana se guarda para la próxima llamada. Devuelve la probabilidad
        MÁXIMA entre las ventanas completas de esta llamada (no el promedio): un chunk de 0.2s
        puede contener una ventana con voz real y otras con silencio de transición — el máximo
        evita que esas ventanas silenciosas diluyan una detección real, mismo espíritu que
        `is_speech_chunk` ya aplica hoy con un único valor de RMS por chunk.

        Devuelve `0.0` si el chunk no alcanza a completar ni una ventana todavía (se guarda en el
        buffer para la próxima llamada) — no hay decisión de voz/silencio que tomar todavía, no
        es lo mismo que "silencio confirmado".
        """
        float_chunk = chunk_int16.astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, float_chunk])
        probabilities: list[float] = []
        while len(self._buffer) >= WINDOW_SAMPLES:
            window = self._buffer[:WINDOW_SAMPLES]
            self._buffer = self._buffer[WINDOW_SAMPLES:]
            probabilities.append(float(self._vad.process(memoryview(window.data))))
        return max(probabilities) if probabilities else 0.0


def load_speech_detector() -> ChunkSpeechDetector | None:
    """Construye un `ChunkSpeechDetector` nuevo para una sesión de grabación, o `None` si el
    modelo no se pudo cargar (DLL/modelo ONNX faltante, lo que sea) — nunca lanza. `None` le dice
    a `vad.is_speech_chunk` que degrade a RMS-solo (comportamiento de antes de este cambio) en
    vez de tumbar la captura de voz completa por una falla en una mejora opcional.
    """
    try:
        return ChunkSpeechDetector()
    except Exception:
        logger.exception(
            "No se pudo cargar el detector de voz (Silero VAD); se degrada a RMS solamente."
        )
        return None
