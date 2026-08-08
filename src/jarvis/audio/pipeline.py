"""Pipeline integrado: wake word → grabar comando → transcribir → LLM → hablar la respuesta.

Alcance de esta fase: el loop de conversación por voz completo (ADR-0004). Sin tools ni
acciones reales todavía: el LLM solo conversa, no ejecuta nada sobre el sistema — esa capa,
con el modelo SAFE/CONFIRM/DANGEROUS de `.claude/rules/security.md`, es la próxima fase.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd

from jarvis.audio.stt import load_whisper_model, transcribe
from jarvis.audio.tts import TTSClient, load_default_tts_client
from jarvis.audio.wake_word import (
    DEFAULT_THRESHOLD,
    SAMPLE_RATE,
    detect,
    iter_microphone_frames,
)
from jarvis.audio.wake_word import load_model as load_wake_word_model
from jarvis.config import load_dotenv
from jarvis.llm.client import LLMClient, load_deepseek_client_from_env

COMMAND_WINDOW_SECONDS = 4.0  # tope duro: si nunca hay silencio, no graba para siempre
COOLDOWN_SECONDS = 1.5  # evita que el eco de la grabación o repetir la frase retriggeree al toque
AGC_MIN_PEAK = 50  # por debajo de esto asumimos piso de ruido, no habla real: no amplificar
AGC_TARGET_PEAK_RATIO = 0.9  # a qué % del rango de int16 apuntamos al subir volumen
CHUNK_SECONDS = 0.2  # tamaño de chunk para detectar silencio mientras se graba el comando
SILENCE_RMS_THRESHOLD = 150.0  # piso de ruido medido ~0.5-7, habla real ~600-6000+ (fase 1)
TRAILING_SILENCE_SECONDS = 1.2  # cuánto silencio sostenido después de hablar antes de cortar solo
SYSTEM_PROMPT = (
    "Sos JARVIS, un asistente personal por voz. Respondés corto y directo, en español, porque "
    "tu respuesta se lee en voz alta — nada de listas, markdown, ni símbolos que no se puedan "
    "pronunciar. Todavía no podés ejecutar ninguna acción real sobre la computadora, solo "
    "conversar."
)


def normalize_gain(audio: np.ndarray) -> np.ndarray:
    """AGC simple: si el audio vino bajito pero con señal real, lo sube a ~90% del rango de
    int16 antes de mandarlo a transcribir. Nunca baja volumen ya fuerte (evita clipping doble),
    y no toca audio por debajo de `AGC_MIN_PEAK` — eso es piso de ruido, no habla; amplificarlo
    solo fabricaría "señal" falsa para Whisper, que ya sabemos que alucina sobre silencio
    (fase 2). Motivo de existir: en vivo se midió audio real de voz con picos tan bajos como
    ~1300/32767 — muy por debajo de lo que veníamos asumiendo como "hablar fuerte".
    """
    peak = int(np.abs(audio).max())
    max_int16 = 32767
    target = int(AGC_TARGET_PEAK_RATIO * max_int16)
    if peak < AGC_MIN_PEAK or peak >= target:
        return audio
    gain = target / peak
    boosted = audio.astype(np.float64) * gain
    return np.clip(boosted, -max_int16, max_int16).astype(np.int16)


def chunk_rms(chunk: np.ndarray) -> float:
    """RMS de un chunk de audio int16 — mide qué tan fuerte es la señal en ese instante."""
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


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


def record_command(*, device: int | None, max_duration: float = COMMAND_WINDOW_SECONDS) -> np.ndarray:
    """Grabar el comando dicho después de la wake word, cortando solo tras un silencio
    sostenido en vez de esperar siempre `max_duration` completos.

    Medido en vivo: la mayoría de los comandos duran 1-2s, no los 4s fijos que se grababan
    antes — eso agregaba hasta 3s de silencio muerto por interacción. Esto le pone fin a esa
    espera innecesaria sin sacrificar el tope duro para cuando alguien habla más largo.
    """
    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    speech_started = False
    silence_run = 0.0
    elapsed = 0.0
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=chunk_samples,
        device=device,
    ) as stream:
        while True:
            chunk, _overflowed = stream.read(chunk_samples)
            chunk = chunk.reshape(-1)
            chunks.append(chunk)
            elapsed += CHUNK_SECONDS
            if chunk_rms(chunk) >= SILENCE_RMS_THRESHOLD:
                speech_started = True
                silence_run = 0.0
            elif speech_started:
                silence_run += CHUNK_SECONDS
            if should_stop_recording(
                speech_started=speech_started,
                silence_run_seconds=silence_run,
                elapsed_seconds=elapsed,
                max_seconds=max_duration,
            ):
                break
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    return normalize_gain(audio)


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
    tts: TTSClient = load_default_tts_client()
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
                if reply.strip():
                    tts.speak(reply)
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
