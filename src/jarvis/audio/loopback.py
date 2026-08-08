"""Monitor de audio del sistema vía loopback WASAPI: mide qué tan fuerte es lo que se está
reproduciendo (juego, música, llamada, etc.) para poder "gatear" el micrófono mientras eso suena
fuerte — ver el wiring en `jarvis.audio.wake_word.detect()` y `jarvis.audio.pipeline`.

Esto NO es cancelación de eco acústico (AEC): no resta la señal de salida de la de entrada ni las
sincroniza. Es deliberadamente más simple — un gate binario "¿el sistema está sonando fuerte
ahora mismo?" — aceptado a propósito como alcance reducido: no se puede hablar por encima de
audio fuerte del PC y ser escuchado, a cambio de evitar el esfuerzo de una AEC real (descartada
explícitamente para esta fase).

Captura "lo que se reproduce" abriendo un `sd.InputStream` contra el device de SALIDA default con
`extra_settings=sd.WasapiSettings(loopback=True)` — el truco estándar de WASAPI/`sounddevice`
para loopback en Windows (no existe una API separada de "grabar el output" en PortAudio).
"""

from __future__ import annotations

import sys
import threading
from typing import Protocol

import numpy as np
import sounddevice as sd

from jarvis.audio.device import output_sample_rate, resolve_output_device

CHUNK_SECONDS = (
    0.15  # ventana de medición: corta para reaccionar rápido sin sobrecargar el
)
# thread de fondo con lecturas constantes.

# Umbral de RMS (int16) por encima del cual se considera "el sistema está sonando fuerte".
# Sin datos en vivo todavía — a diferencia de `NOISE_FLOOR_MULTIPLIER` en `pipeline.py`, que sí
# se calibró contra grabaciones reales del mic (fase 4) — este es un punto de partida razonado,
# no medido: claramente por encima del silencio/ruido de fondo de un output idle (~0 cuando no
# suena nada) pero por debajo de música/juego a volumen normal-alto en parlantes. Mismo orden de
# magnitud que `MIN_SILENCE_RMS_THRESHOLD` del mic (40.0) sería demasiado sensible acá porque el
# loopback captura la señal de reproducción completa, no una voz a distancia — de ahí un umbral
# un orden de magnitud mayor. Revisar en vivo (`.claude/rules/testing.md`) igual que ese valor,
# apenas haya un caso real de juego/música sonando para medir contra silencio real.
DEFAULT_LOUDNESS_THRESHOLD = 500.0


class SystemAudioGate(Protocol):
    """Contrato mínimo que `wake_word.detect()`/`pipeline.record_command()` necesitan del
    monitor de audio del sistema. Existe para que esos sitios puedan tipar el parámetro sin
    importar `SystemAudioMonitor` (ni abrir un stream real) en sus tests — un fake que solo
    implementa `is_loud()` alcanza.
    """

    def is_loud(self) -> bool: ...


def _chunk_rms(chunk: np.ndarray) -> float:
    """RMS de un chunk de audio int16 (mismo cálculo que `pipeline.chunk_rms`; no se importa
    desde ahí para no acoplar este módulo, standalone, a `pipeline.py`)."""
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


class SystemAudioMonitor:
    """Corre loopback WASAPI del device de salida default en un thread de fondo, manteniendo una
    medida rodante (RMS del último chunk leído) de lo que se está reproduciendo ahora mismo.

    `is_loud()` es una lectura barata (un lock corto, sin I/O) pensada para llamarse desde los
    hot paths de `detect()`/`record_command()` sin bloquearlos.
    """

    def __init__(
        self,
        *,
        device: int | None = None,
        threshold: float = DEFAULT_LOUDNESS_THRESHOLD,
        chunk_seconds: float = CHUNK_SECONDS,
    ) -> None:
        self._device = device
        self._threshold = threshold
        self._chunk_seconds = chunk_seconds
        self._level = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Se pone en True si no hay device de salida o el stream de loopback no pudo abrirse
        # (p.ej. conflicto de modo exclusivo) — desde ahí `is_loud()` siempre devuelve False en
        # vez de tumbar JARVIS por una feature de robustez nice-to-have (ver docstring del
        # módulo).
        self._disabled = False

    def start(self) -> None:
        """Arrancar el thread de fondo. Idempotente: no hace nada si ya está corriendo."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Señalizar el fin del thread de fondo y esperar a que cierre el stream. Idempotente."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def is_loud(self) -> bool:
        """`True` si el nivel medido más reciente supera el umbral configurado."""
        if self._disabled:
            return False
        with self._lock:
            return self._level >= self._threshold

    @property
    def current_level(self) -> float:
        with self._lock:
            return self._level

    def _run(self) -> None:
        try:
            resolved_device = resolve_output_device(self._device)
            device_sr = output_sample_rate(resolved_device)
            channels = max(
                1, int(sd.query_devices(resolved_device)["max_output_channels"])
            )
        except (RuntimeError, sd.PortAudioError) as exc:
            self._disable(exc)
            return
        chunk_samples = max(1, int(self._chunk_seconds * device_sr))
        try:
            with sd.InputStream(
                samplerate=device_sr,
                channels=channels,
                dtype="int16",
                blocksize=chunk_samples,
                device=resolved_device,
                extra_settings=sd.WasapiSettings(loopback=True),
            ) as stream:
                while not self._stop_event.is_set():
                    chunk, _overflowed = stream.read(chunk_samples)
                    level = _chunk_rms(chunk.reshape(-1))
                    with self._lock:
                        self._level = level
        except sd.PortAudioError as exc:
            self._disable(exc)

    def _disable(self, exc: Exception) -> None:
        self._disabled = True
        print(
            f"Monitor de audio del sistema deshabilitado (loopback WASAPI falló: {exc!r}). "
            "El gate de audio fuerte no va a activarse — JARVIS sigue funcionando sin él.",
            file=sys.stderr,
        )
