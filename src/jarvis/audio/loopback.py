"""Monitor de audio del sistema vía loopback WASAPI: mide qué tan fuerte es lo que se está
reproduciendo (juego, música, llamada, etc.) para poder "gatear" el micrófono mientras eso suena
fuerte — ver el wiring en `jarvis.audio.wake_word.detect()` y `jarvis.audio.pipeline`.

Esto NO es cancelación de eco acústico (AEC): no resta la señal de salida de la de entrada ni las
sincroniza. Es deliberadamente más simple — un gate binario "¿el sistema está sonando fuerte
ahora mismo?" — aceptado a propósito como alcance reducido: no se puede hablar por encima de
audio fuerte del PC y ser escuchado, a cambio de evitar el esfuerzo de una AEC real (descartada
explícitamente para esta fase).

Captura "lo que se reproduce" con la librería `soundcard` (`sc.get_microphone(id=...,
include_loopback=True)`), NO con `sounddevice`/PortAudio. Esto no es un cambio cosmético de API:
`sounddevice.WasapiSettings(loopback=True)` nunca fue una API real de PortAudio/`sounddevice` en
ninguna versión — confirmado en vivo contra `sounddevice==0.5.5` instalado acá (`WasapiSettings`
solo acepta `exclusive`/`auto_convert`/`explicit_sample_format`; no hay flag de loopback en el
enum `PaWasapiFlags` de PortAudio) y confirmado además por el mantenedor de `sounddevice` en
https://github.com/spatialaudio/python-sounddevice/issues/281 (feature request de loopback WASAPI
todavía abierto: PortAudio vanilla no lo soporta, hace falta un DLL de PortAudio parcheado
-como el que usa Audacity- o una librería que hable WASAPI directo). `soundcard` (mismo autor que
`sounddevice`, `pip install soundcard`) sí implementa loopback real: en Windows habla WASAPI
directo vía Media Foundation/COM, sin pasar por PortAudio, así que no hereda esta limitación.
Reutilizamos `resolve_output_device`/`output_sample_rate` (`sounddevice`, sin tocar — esa parte sí
funciona bien) solo para resolver *qué* device de salida y a qué sample rate; el stream de
captura en sí lo abre `soundcard`.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import warnings
from typing import Protocol

import numpy as np
import soundcard as sc
import sounddevice as sd

from jarvis.audio.device import output_sample_rate, resolve_output_device

CHUNK_SECONDS = (
    0.15  # ventana de medición: corta para reaccionar rápido sin sobrecargar el
)
# thread de fondo con lecturas constantes.

# Umbral de RMS (escala int16) por encima del cual se considera "el sistema está sonando fuerte".
#
# Por qué NO es un umbral calibrado en runtime como `calibrate_thresholds()` (mic, `pipeline.py`):
# ese patrón mide el ambiente al arrancar asumiendo que todavía no pasó lo que se quiere excluir
# ("no hables todavía" — asunción razonable porque el usuario recién arrancó JARVIS y no dijo
# nada). Ese supuesto NO se sostiene acá: el audio del sistema (juego, música) puede estar sonando
# fuerte desde antes de que JARVIS arranque y seguir así indefinidamente — no hay un momento
# garantizado de "silencio" al iniciar el monitor. Calibrar contra el nivel del momento del
# arranque arriesga tomar audio fuerte como si fuera el piso de silencio (exactamente el escenario
# de esta noche: JARVIS se reinicia con el juego ya sonando), lo que subiría el umbral en vez de
# bajarlo — el mismo bug de nuevo, por otro camino. Por eso se usa un valor fijo, documentado
# contra mediciones reales, en vez de recalibrar cada corrida.
#
# Mediciones en vivo (`SystemAudioMonitor` real, no simulado, hoy, mismo hardware que corre
# JARVIS) — tres corridas independientes, todas contra el device de salida default (auriculares
# Bluetooth) mientras el usuario jugaba League of Legends con audio de juego/música sonando:
#   corrida 1 (~18s): rms min=67.7   max=105.1  (sin filtrar silencios)
#   corrida 2 (~35s): rms min=39.7   max=105.1  p50=58.9  p95=86.4  p99=96.8
#   corrida 3 (~15s, ya con el fix de COM aplicado, end-to-end vía la clase real): min=42.1
#     max=103.1 mean=65.5 p50=63.5 p95=89.5
# → el piso real de "juego sonando" en este sistema, a este volumen, converge en ~40-105, nunca
# por debajo de ~40 en ninguna corrida.
# Baseline de silencio real (misma corrida 3, mismo instante): loopback contra un device de
# salida WASAPI DISTINTO al default activo (el Realtek "Speaker" interno, que Windows no le manda
# audio del juego porque el default activo es el Bluetooth) — mismo mecanismo de captura, cero
# reproducción real: rms = 0.0 constante en las ~12s medidas. Silencio digital real, no ruido de
# fondo del driver.
# → con un piso de 0.0 y un suelo real de "algo sonando" en ~40, un umbral de 30.0 cae claramente
# entre ambos: muy por encima del piso de silencio medido (margen de 30 unidades sobre un piso de
# 0.0, sin necesitar asumir dither/ruido de driver que no se observó) y por debajo del mínimo real
# de audio de juego observado en las tres corridas (~40), así que dispara con contenido real
# reproduciéndose en vez de nunca disparar (bug original, con el 500.0 previo) ni disparar con
# silencio real. `MIN_LOUDNESS_RMS_THRESHOLD` no se introduce como piso absoluto separado (a
# diferencia de `MIN_SILENCE_RMS_THRESHOLD` del mic) porque acá no hay calibración en runtime que
# necesite un piso — el valor ya es fijo y medido.
# Verificación final end-to-end con este umbral ya aplicado (no solo `current_level`, `is_loud()`
# en sí, vía la clase real): 0/60 polls en `True` contra el device idle (silencio real, ~6s) y
# 100/100 polls en `True` contra el device default con el juego sonando (~10s) — confirma que el
# gate efectivamente prende con audio real y no prende con silencio real, no solo que los números
# de RMS "se ven razonables".
# Sigue siendo una calibración de una sola sesión/volumen/juego, no exhaustiva (falta variar
# aplicaciones, volúmenes del sistema, y confirmar que un pico real de combate no se escapa muy
# por encima de 105 sin que sea necesario — un valor por encima del pico también dispararía igual
# de bien). Revisar de nuevo si en el futuro aparece un caso real donde el gate falle en cualquier
# dirección (nunca dispara con audio fuerte real, o dispara con silencio/audio bajito real).
DEFAULT_LOUDNESS_THRESHOLD = 30.0


class SystemAudioGate(Protocol):
    """Contrato mínimo que `wake_word.detect()`/`pipeline.record_command()` necesitan del
    monitor de audio del sistema. Existe para que esos sitios puedan tipar el parámetro sin
    importar `SystemAudioMonitor` (ni abrir un stream real) en sus tests — un fake que solo
    implementa `is_loud()` alcanza.
    """

    def is_loud(self) -> bool: ...


def _chunk_rms(chunk: np.ndarray) -> float:
    """RMS de un chunk de audio en escala int16 (mismo cálculo que `pipeline.chunk_rms`; no se
    importa desde ahí para no acoplar este módulo, standalone, a `pipeline.py`). El chunk debe
    venir ya escalado a int16 — ver `_to_int16_scale` para chunks float32 de `soundcard`."""
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


# `soundcard` devuelve muestras float32 normalizadas en [-1.0, 1.0] (a diferencia del
# `sd.InputStream(dtype="int16")` que usaba este módulo antes de migrar a `soundcard` — ver
# docstring del módulo). Escalamos a int16 acá, en el borde de este módulo, para que
# `DEFAULT_LOUDNESS_THRESHOLD` y `_chunk_rms` sean comparables 1:1 con el resto del código de
# audio de JARVIS (mic vía `sounddevice`, siempre en escala int16 — ver `pipeline.chunk_rms`).
_INT16_SCALE = 32767.0


def _to_int16_scale(chunk: np.ndarray) -> np.ndarray:
    return chunk * _INT16_SCALE


# `soundcard` habla COM (WASAPI vía Media Foundation) por debajo. COM se inicializa por thread de
# Windows, no una sola vez por proceso: `import soundcard` inicializa COM en el thread que hizo el
# import (normalmente el principal), pero eso NO cubre el thread de fondo propio de
# `SystemAudioMonitor` (`threading.Thread` separado) — confirmado en vivo: sin este init, cada
# llamada a `soundcard` desde `_run()` fallaba con `RuntimeError('Error 0x800401f0')`
# (`CO_E_NOTINITIALIZED`) y el monitor se deshabilitaba en cada corrida, con `is_loud()` en False
# para siempre pese a que la captura en sí ya funcionaba. `COINIT_MULTITHREADED` (no STA): mismo
# modelo que usa `soundcard._COMLibrary` en cualquier Windows que no sea 8 (ver
# `soundcard/mediafoundation.py`), correcto para un thread de fondo dedicado como este.
_COINIT_MULTITHREADED = 0x0
_S_OK = 0
_S_FALSE = 1
_RPC_E_CHANGED_MODE = 0x80010106


def _init_com_for_this_thread() -> bool:
    """Inicializar COM para el thread desde el que se llama. Devuelve `True` si este llamado fue
    el que lo inicializó (hay que liberarlo con `CoUninitialize` al terminar); `False` si el
    thread ya tenía COM inicializado por otra vía (nada que liberar acá)."""
    hr = ctypes.windll.ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    hr_unsigned = hr & 0xFFFFFFFF
    if hr_unsigned in (_S_OK, _S_FALSE):
        return True
    if hr_unsigned == _RPC_E_CHANGED_MODE:
        # Ya inicializado en este thread con otro modelo de concurrencia (p.ej. STA, por otra
        # librería) — sigue siendo usable, pero no lo inicializamos nosotros, así que no hay que
        # desinicializarlo nosotros tampoco.
        return False
    raise OSError(f"CoInitializeEx falló con HRESULT {hr_unsigned:#010x}")


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
            owns_com = _init_com_for_this_thread()
        except OSError as exc:
            self._disable(exc)
            return
        try:
            self._run_with_com_initialized()
        finally:
            if owns_com:
                ctypes.windll.ole32.CoUninitialize()

    def _run_with_com_initialized(self) -> None:
        try:
            resolved_device = resolve_output_device(self._device)
            device_sr = output_sample_rate(resolved_device)
            device_name = str(sd.query_devices(resolved_device)["name"])
            # `sounddevice`/`resolve_output_device` solo se usan para resolver *qué* device de
            # salida y su sample rate nativo (eso funciona bien — ver `device.py`); el device
            # loopback en sí lo resuelve `soundcard` por nombre, matcheando contra el mismo
            # endpoint WASAPI (ver docstring del módulo sobre por qué `sounddevice` no puede
            # abrir el loopback él mismo).
            loopback_mic = sc.get_microphone(id=device_name, include_loopback=True)
        except (RuntimeError, sd.PortAudioError, IndexError) as exc:
            self._disable(exc)
            return
        chunk_samples = max(1, int(self._chunk_seconds * device_sr))
        try:
            # `soundcard` emite `SoundcardRuntimeWarning` cuando WASAPI reporta un frame de audio
            # perdido en el buffer interno (visto en vivo: "data discontinuity in recording",
            # confirmado como comportamiento esperado/no-fatal de la librería bajo carga, no un
            # bug de este código — ver issues de `bastibe/SoundCard` sobre buffer underruns).
            # `warnings.filterwarnings` muta el filtro global de `warnings` del proceso entero
            # (no hay forma de scoparlo solo a este thread/loop sin `catch_warnings()`, que la
            # propia documentación de `warnings` marca como no thread-safe si se mantiene abierto
            # mientras corren otros threads — este monitor vive mientras JARVIS corre, así que
            # mantenerlo abierto todo ese tiempo sería igual de global en la práctica). Se acepta
            # el filtro global acá porque `SoundcardRuntimeWarning` es una clase específica de
            # una sola librería para un solo escenario (frame perdido en loopback) — no hay
            # ningún otro código en este proceso al que le importe verla, y `is_loud()` es un
            # gate de nivel aproximado (no captura de alta fidelidad), así que perder un frame
            # ocasional no afecta su corrección. Sin silenciar, el warning spamea
            # `jarvis-error.log` en cada ocurrencia, degradando la señal real de ese log.
            warnings.filterwarnings(
                "ignore", category=sc.mediafoundation.SoundcardRuntimeWarning
            )
            with loopback_mic.recorder(
                samplerate=device_sr, blocksize=chunk_samples
            ) as recorder:
                while not self._stop_event.is_set():
                    chunk = recorder.record(numframes=chunk_samples)
                    level = _chunk_rms(_to_int16_scale(chunk.reshape(-1)))
                    with self._lock:
                        self._level = level
        except (RuntimeError, OSError) as exc:
            # `soundcard` reporta fallos de COM/WASAPI (device desconectado, conflicto de modo
            # exclusivo, endpoint que dejó de existir) como `RuntimeError` de forma uniforme (ver
            # `soundcard.mediafoundation._handle_error`) — no hay un tipo de excepción propio
            # como `sd.PortAudioError` acá. Mismo contrato que antes: degradar en silencio
            # (`_disabled=True`, `is_loud()` siempre `False`) en vez de tumbar este thread de
            # fondo con un traceback crudo — ver docstring de la clase.
            self._disable(exc)

    def _disable(self, exc: Exception) -> None:
        self._disabled = True
        print(
            f"Monitor de audio del sistema deshabilitado (loopback WASAPI falló: {exc!r}). "
            "El gate de audio fuerte no va a activarse — JARVIS sigue funcionando sin él.",
            file=sys.stderr,
        )
