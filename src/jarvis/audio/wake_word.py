"""Prototipo de detección de wake word ("Hey Jarvis", "Alexa", "Hey Mycroft") sobre el
micrófono real.

Alcance de esta fase: solo detección. Nada de STT, LLM ni TTS todavía (ADR-0004).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from jarvis.audio.device import input_sample_rate, resolve_input_device
from jarvis.audio.loopback import SystemAudioGate
from jarvis.audio.resample import fit_frame, resample
from jarvis.audio.speech_detector import SPEECH_PROBABILITY_THRESHOLD

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1280  # 80ms a 16kHz: tamaño de chunk recomendado por openWakeWord

# "alexa" primero/primaria (pedido explícito del usuario). "hey_jarvis" y "hey_mycroft" son
# modelos pretrained adicionales de openWakeWord (sin entrenar nada custom — eso queda para una
# fase futura con Python 3.10 + deps viejas + GPU), agregados para que cualquiera de las tres
# frases dispare a JARVIS. `Model(wakeword_models=[...])` con varias entradas hace que
# `model.predict()` devuelva un dict keyed por wakeword, y `detect()` ya itera ese dict
# (`predictions.items()`), así que no necesitó cambios para soportar múltiples wakewords. Las
# tres comparten `DEFAULT_THRESHOLD` (calibrado en vivo solo contra "hey_jarvis" — ver abajo);
# "alexa" no tiene calibración propia todavía, si en el uso real resulta muy sensible o muy poco
# sensible hay que ajustarla con `scripts/diagnose_wakeword.py`.
WAKEWORD_NAMES = ["alexa", "hey_jarvis", "hey_mycroft"]

# Calibrado empíricamente (fase 1 y recalibrado en fase 4, docs/decisions/), no elegido al azar.
# Con el modelo pretrained "hey_jarvis", el score varía sesión a sesión: fase 1 llegó a 0.4916
# de máximo; una sesión posterior (fase 4) topeó en 0.2997, con la mayoría de las tomas entre
# 0.05 y 0.2 en ambas. El default de openWakeWord (0.5) nunca cruzó en la práctica, y 0.3
# tampoco fue confiable en la segunda sesión. 0.25 prioriza no quedar mudo por sobre el riesgo
# de falsos positivos — aceptable mientras ninguna acción real esté atada a la detección
# (ADR-0004). Si en una fase futura se ata una acción real a esto, revisar este valor junto con
# el contrato de confirmación verbal (silencio/ambigüedad = denegar), y considerar entrenar un
# modelo de wake word con la voz real del usuario en vez de seguir bajando el umbral genérico.
# "alexa" y "hey_mycroft" reusan este mismo umbral por ahora: no fueron calibrados individualmente
# en vivo todavía, así que puede que necesiten su propio valor más adelante (más falsos positivos
# o negativos que "hey_jarvis" con este mismo threshold no está descartado).
DEFAULT_THRESHOLD = 0.25


@dataclass(frozen=True)
class Detection:
    """Una detección de wake word por encima del umbral configurado."""

    wakeword: str
    score: float
    timestamp: dt.datetime


class SpeechDetector(Protocol):
    """Contrato mínimo que `detect()` necesita de un detector de voz (Silero VAD,
    `jarvis.audio.speech_detector.ChunkSpeechDetector`) — mismo espíritu que `SystemAudioGate`
    (`jarvis.audio.loopback`): tipar el parámetro sin acoplar `detect()` a cargar un modelo real
    ni abrir ningún recurso en sus tests, un fake que solo implementa `speech_probability` alcanza.
    """

    def speech_probability(self, chunk_int16: np.ndarray) -> float: ...


def load_model() -> Model:
    """Cargar los modelos de `WAKEWORD_NAMES`. Requiere haber corrido
    `openwakeword.utils.download_models(WAKEWORD_NAMES)` al menos una vez."""
    return Model(wakeword_models=WAKEWORD_NAMES, inference_framework="onnx")


def iter_microphone_frames(
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
    device: int | None = None,
    duration: float | None = None,
) -> Generator[np.ndarray, None, None]:
    """Generar frames mono int16 a `sample_rate` desde el dispositivo de entrada dado (o el
    default), resampleados desde la tasa nativa del dispositivo.

    Grabamos a la tasa nativa del device (no `sample_rate` directo): confirmado en vivo que el
    endpoint WASAPI del micrófono de esta máquina no acepta pedir 16kHz (falla con
    "Invalid sample rate"; está fijado a 48000Hz en Windows), así que grabamos a 48000 y
    resampleamos en software (`resample.py`) al `sample_rate` que espera el modelo.

    Si `duration` (segundos) está seteado, deja de generar frames pasado ese tiempo en vez de
    correr para siempre — necesario para probarlo sin depender de Ctrl+C manual.

    Tipado como `Generator`, no `Iterator`: quien llama necesita poder cerrar el stream
    explícitamente con `.close()` antes de que termine solo (ver `pipeline.py`, que corta la
    escucha para abrir un stream de grabación separado sin dejar dos streams abiertos a la vez).
    """
    resolved_device = resolve_input_device(device)
    device_sr = input_sample_rate(resolved_device)
    native_frame_samples = round(frame_samples * device_sr / sample_rate)
    deadline = time.monotonic() + duration if duration is not None else None
    with sd.InputStream(
        samplerate=device_sr,
        channels=1,
        dtype="int16",
        blocksize=native_frame_samples,
        device=resolved_device,
    ) as stream:
        while deadline is None or time.monotonic() < deadline:
            frame, _overflowed = stream.read(native_frame_samples)
            resampled = resample(
                frame.reshape(-1), orig_sr=device_sr, target_sr=sample_rate
            )
            yield fit_frame(resampled, frame_samples)


def detect(
    frames: Iterator[np.ndarray],
    *,
    model: Model,
    threshold: float = DEFAULT_THRESHOLD,
    system_audio: SystemAudioGate | None = None,
    speech_detector: SpeechDetector | None = None,
    speech_probability_threshold: float = SPEECH_PROBABILITY_THRESHOLD,
) -> Iterator[Detection]:
    """Consumir frames de audio y emitir una Detection cada vez que se cruza el umbral.

    `system_audio`, si se pasa, es una inyección de dependencia opcional (`SystemAudioGate`,
    `jarvis.audio.loopback`): mientras reporte `is_loud() == True`, no se emite Detection en ese
    frame aunque el score cruce el umbral — evita falsos triggers por audio fuerte del sistema
    (juego, música) filtrándose al mic (no reemplaza cancelación de eco real, ver docstring de
    `loopback.py`). Se sigue llamando a `model.predict()` en cada frame sin importar el gate:
    openWakeWord mantiene una ventana deslizante de estado interno entre llamadas, así que
    saltear frames enteros rompería las predicciones siguientes en vez de solo silenciarlas.
    Sin `system_audio` (default `None`, como en todos los tests existentes), el comportamiento
    es idéntico al de antes de este parámetro.

    `speech_detector` (opcional, `jarvis.audio.speech_detector.ChunkSpeechDetector` vía
    `load_speech_detector()`) cierra un hueco real de `system_audio`: confirmado en vivo
    (`data/jarvis-error.log`, sesión del 2026-08-13, ~20:13-20:15) — con el mic y la salida
    siendo el mismo headset combinado, `SystemAudioMonitor` desactiva el gate por completo
    (`device.is_combined_headset`, asumiendo que un headset no tiene filtración acústica real de
    salida a entrada), pero en este hardware SÍ la hay: mientras sonaba música/video de fondo
    (YouTube, abierto por un turno anterior — "te abrí YouTube en el navegador"/"reproduce Space
    Zone en YouTube"), el wake word se disparó solo, repetidamente, con score=1.00, cada vez que
    se reabría una ventana de escucha — sin que el usuario dijera nada (`Dijiste: ''` en
    `data/jarvis.log` en cada ciclo) y sin que JARVIS estuviera hablando (`tts.speak()` no se
    llamó ninguna vez en ese tramo). El RMS medido en esos ciclos (`data/jarvis-error.log`, los
    primeros chunks de `record_command` justo después de cada disparo falso) se quedó bajo
    (~5-300, muy por debajo del rango de voz real medido en esta sesión, 300-9000) — consistente
    con audio de sistema filtrándose bajito al mic (sea acústicamente o vía sidetone de hardware
    del propio headset), no con el usuario hablando fuerte cerca del mic. Ni "pausar mientras
    `tts.speak()` está activo" ni "umbral más alto justo después de abrir una URL" (los dos
    candidatos evaluados antes de este) cubren este caso: el audio de fondo sigue sonando minutos
    después de abrirse la URL, mucho más allá de cualquier ventana transitoria razonable, y JARVIS
    no estaba hablando en ningún momento del tramo con falsos triggers.
    En vez de tocar `loopback.py`/`is_combined_headset` (revertir el gate a "siempre activo"
    reintroduciría el bug que motivó la excepción de headset en primer lugar: `SystemAudioMonitor`
    calibrado contra el nivel del loopback de SALIDA, no de lo que se filtra al mic, reportaba
    `is_loud()=True` casi todo el tiempo con juego/música de fondo y bloqueaba el wake word por
    completo incluso diciendo "Alexa" activamente — cambiar ese trade-off es una decisión de
    producto, no algo a decidir de forma incidental acá), este chequeo es independiente y
    ortogonal: usa `jarvis.audio.speech_detector` (Silero VAD), el mismo discriminador "¿esto
    suena a voz humana o no?" que ya filtra ruido de juego en `record_command`/
    `stream_transcribe_command` (`jarvis.audio.vad.is_speech_chunk`) — acá se aplica en el único
    punto del pipeline de audio que todavía no lo tenía. Igual que `system_audio`, se llama a
    `speech_probability()` en CADA frame sin importar el resultado de las otras gates: mantiene el
    buffer interno del detector sincronizado con el audio real (saltear frames lo desalinearía).
    Sin `speech_detector` (default `None`, como en todos los tests existentes salvo los que lo
    inyectan explícitamente), el comportamiento es idéntico al de antes de este parámetro — mismo
    criterio de degradación que `is_speech_chunk` con `speech_probability=None`: una mejora
    opcional que no tumba la detección si el modelo no cargó.
    """
    for frame in frames:
        predictions = model.predict(frame)
        speech_probability = (
            speech_detector.speech_probability(frame)
            if speech_detector is not None
            else None
        )
        if system_audio is not None and system_audio.is_loud():
            continue
        if (
            speech_probability is not None
            and speech_probability < speech_probability_threshold
        ):
            continue
        for wakeword, score in predictions.items():
            if score >= threshold:
                yield Detection(
                    wakeword=wakeword,
                    score=float(score),
                    timestamp=dt.datetime.now(dt.UTC),
                )


def run(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: int | None = None,
    duration: float | None = None,
) -> None:
    """Escuchar el micrófono imprimiendo cada detección de wake word.

    Corre hasta Ctrl+C, o hasta que pasen `duration` segundos si se especifica.
    """
    model = load_model()
    frames = iter_microphone_frames(device=device, duration=duration)
    print(
        "Escuchando... decí 'Hey Jarvis', 'Alexa' o 'Hey Mycroft' "
        f"(Ctrl+C para salir, umbral={threshold})",
        file=sys.stderr,
    )
    try:
        for hit in detect(frames, model=model, threshold=threshold):
            local_time = hit.timestamp.astimezone()
            print(
                f"[{local_time.isoformat(timespec='seconds')}] "
                f"wake word detectada: {hit.wakeword} (score={hit.score:.2f})"
            )
    except KeyboardInterrupt:
        print("Detenido.", file=sys.stderr)
    print("Fin de la escucha.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototipo de detección de wake word para JARVIS"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Índice del dispositivo de entrada (ver sounddevice.query_devices())",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Segundos a escuchar antes de cortar solo (default: infinito, Ctrl+C)",
    )
    args = parser.parse_args()
    run(threshold=args.threshold, device=args.device, duration=args.duration)


if __name__ == "__main__":
    main()
