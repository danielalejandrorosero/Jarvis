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

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from jarvis.audio.device import input_sample_rate, resolve_input_device
from jarvis.audio.loopback import SystemAudioGate
from jarvis.audio.resample import fit_frame, resample

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
    """
    for frame in frames:
        predictions = model.predict(frame)
        if system_audio is not None and system_audio.is_loud():
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
