"""Prototipo de detección de wake word ("Hey Jarvis") sobre el micrófono real.

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

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1280  # 80ms a 16kHz: tamaño de chunk recomendado por openWakeWord
WAKEWORD_NAME = "hey_jarvis"

# Calibrado empíricamente (fase 1, docs/decisions/, sesión de diagnóstico en vivo), no elegido
# al azar. Con el modelo pretrained "hey_jarvis" y la voz/pronunciación del usuario de este
# proyecto, el score máximo observado en ~8 tomas reales (dos micrófonos, distintos volúmenes)
# fue 0.4916, con la mayoría de las tomas entre 0.05 y 0.2. El default de openWakeWord (0.5)
# nunca cruzó en la práctica. 0.3 es un punto medio: por debajo del mejor caso real, por encima
# del piso de ruido — prioriza no quedar mudo por sobre el riesgo de falsos positivos, aceptable
# en esta fase porque ninguna acción real está atada todavía a la detección (ADR-0004). Si en la
# próxima fase se ata una acción real a esto, revisar este valor junto con el contrato de
# confirmación verbal (silencio/ambigüedad = denegar).
DEFAULT_THRESHOLD = 0.3


@dataclass(frozen=True)
class Detection:
    """Una detección de wake word por encima del umbral configurado."""

    wakeword: str
    score: float
    timestamp: dt.datetime


def load_model() -> Model:
    """Cargar el modelo hey_jarvis. Requiere haber corrido download_models() al menos una vez."""
    return Model(wakeword_models=[WAKEWORD_NAME], inference_framework="onnx")


def iter_microphone_frames(
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_samples: int = FRAME_SAMPLES,
    device: int | None = None,
    duration: float | None = None,
) -> Generator[np.ndarray, None, None]:
    """Generar frames mono int16 desde el dispositivo de entrada dado (o el default).

    Si `duration` (segundos) está seteado, deja de generar frames pasado ese tiempo en vez de
    correr para siempre — necesario para probarlo sin depender de Ctrl+C manual.

    Tipado como `Generator`, no `Iterator`: quien llama necesita poder cerrar el stream
    explícitamente con `.close()` antes de que termine solo (ver `pipeline.py`, que corta la
    escucha para abrir un stream de grabación separado sin dejar dos streams abiertos a la vez).
    """
    deadline = time.monotonic() + duration if duration is not None else None
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=frame_samples,
        device=device,
    ) as stream:
        while deadline is None or time.monotonic() < deadline:
            frame, _overflowed = stream.read(frame_samples)
            yield frame.reshape(-1)


def detect(
    frames: Iterator[np.ndarray],
    *,
    model: Model,
    threshold: float = DEFAULT_THRESHOLD,
) -> Iterator[Detection]:
    """Consumir frames de audio y emitir una Detection cada vez que se cruza el umbral."""
    for frame in frames:
        predictions = model.predict(frame)
        for wakeword, score in predictions.items():
            if score >= threshold:
                yield Detection(wakeword=wakeword, score=float(score), timestamp=dt.datetime.now(dt.UTC))


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
        f"Escuchando... decí 'Hey Jarvis' (Ctrl+C para salir, umbral={threshold})",
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
    parser = argparse.ArgumentParser(description="Prototipo de detección de wake word para JARVIS")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device", type=int, default=None, help="Índice del dispositivo de entrada (ver sounddevice.query_devices())"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Segundos a escuchar antes de cortar solo (default: infinito, Ctrl+C)"
    )
    args = parser.parse_args()
    run(threshold=args.threshold, device=args.device, duration=args.duration)


if __name__ == "__main__":
    main()
