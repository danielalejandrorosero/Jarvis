"""Pipeline integrado: wake word → grabar comando → transcribir → responder con el LLM.

Alcance de esta fase: detectar + transcribir + una respuesta de texto del LLM (DeepSeek). Nada
de TTS todavía — la respuesta solo se imprime por consola (ADR-0004). Sin tools ni acciones
reales: el LLM solo conversa, no ejecuta nada sobre el sistema (esa capa es la próxima fase).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

from jarvis.audio.stt import load_whisper_model, transcribe
from jarvis.audio.wake_word import (
    DEFAULT_THRESHOLD,
    SAMPLE_RATE,
    detect,
    iter_microphone_frames,
)
from jarvis.audio.wake_word import load_model as load_wake_word_model
from jarvis.config import load_dotenv
from jarvis.llm.client import LLMClient, load_deepseek_client_from_env

COMMAND_WINDOW_SECONDS = 4.0
COOLDOWN_SECONDS = 1.5  # evita que el eco de la grabación o repetir la frase retriggeree al toque
SYSTEM_PROMPT = (
    "Sos JARVIS, un asistente personal por voz. Respondés corto y directo, en español, "
    "porque tu respuesta se va a leer en una consola por ahora — nada de listas largas ni "
    "formato markdown. Todavía no podés ejecutar ninguna acción real sobre la computadora, "
    "solo conversar."
)


def record_command(*, device: int | None, duration: float = COMMAND_WINDOW_SECONDS) -> np.ndarray:
    """Grabar un clip de audio de duración fija — la orden dicha después de la wake word."""
    audio: np.ndarray = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    return np.asarray(audio.reshape(-1))


def run(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: int | None = None,
    duration: float | None = None,
    stt_device: str = "cuda",
) -> None:
    """Escuchar la wake word; al detectarla, grabar un comando y transcribirlo.

    Corre hasta Ctrl+C, o hasta que pasen `duration` segundos totales si se especifica.

    `stt_device` default "cuda" a nivel de esta app (no de la librería `stt.py`, que por
    defecto usa CPU): confirmado funcionando en esta máquina con una RTX 3050 + los paquetes
    pip `nvidia-cublas-cu12`/`nvidia-cudnn-cu12`. Si no están instalados o la GPU falla, pasar
    `--stt-device cpu` explícitamente.
    """
    load_dotenv()
    wake_model = load_wake_word_model()
    whisper_model = load_whisper_model(device=stt_device)
    llm: LLMClient = load_deepseek_client_from_env()
    deadline = time.monotonic() + duration if duration is not None else None
    print(
        f"Escuchando... decí 'Hey Jarvis' (Ctrl+C para salir, umbral={threshold})",
        file=sys.stderr,
    )
    try:
        while deadline is None or time.monotonic() < deadline:
            remaining = (deadline - time.monotonic()) if deadline is not None else None
            frames = iter_microphone_frames(device=device, duration=remaining)
            hit = next(detect(frames, model=wake_model, threshold=threshold), None)
            frames.close()  # cierra el stream de escucha antes de abrir el de grabación
            if hit is None:
                break  # se acabó la ventana de duration sin detectar nada más
            print(
                f"Wake word detectada (score={hit.score:.2f}). Escuchando comando...",
                file=sys.stderr,
            )
            command_audio = record_command(device=device)
            text = transcribe(command_audio, model=whisper_model)
            print(f"Dijiste: {text!r}")
            if not text.strip():
                # Whisper puede "alucinar" frases sobre silencio puro (confirmado en esta fase);
                # no tiene sentido gastar una llamada al LLM sobre texto vacío.
                print("(nada que responder, no se detectó habla real)", file=sys.stderr)
            else:
                reply = llm.complete(text, system=SYSTEM_PROMPT)
                print(f"JARVIS: {reply}")
            time.sleep(COOLDOWN_SECONDS)
    except KeyboardInterrupt:
        print("Detenido.", file=sys.stderr)
    print("Fin de la escucha.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline JARVIS: wake word + transcripción")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device", type=int, default=None, help="Índice del dispositivo de entrada (ver sounddevice.query_devices())"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Segundos totales de escucha (default: infinito, Ctrl+C)"
    )
    parser.add_argument(
        "--stt-device", type=str, default="cuda", choices=["cuda", "cpu"], help="Dispositivo para Whisper (default: cuda)"
    )
    args = parser.parse_args()
    run(threshold=args.threshold, device=args.device, duration=args.duration, stt_device=args.stt_device)


if __name__ == "__main__":
    main()
