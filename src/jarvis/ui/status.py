"""Contrato de estado compartido entre `jarvis.audio.pipeline.run()` (único escritor) y
`jarvis.ui.overlay` (único lector, proceso separado) — ver ADR-0008.

Formato deliberadamente mínimo: un archivo JSON plano (`DEFAULT_STATUS_PATH`,
`data/status.json`, mismo directorio de estado en runtime que `jarvis.memory.store`/
`jarvis.tools.screenshot`) con tres campos — `state`, `last_text`, `timestamp` — sin versión de
esquema, sin lista de eventos, sin nada más elaborado. Se evaluaron alternativas (socket local,
memoria compartida) y se descartaron por sobre-ingeniería para el caso de uso real: un archivo
sondeado cada ~200ms por un único lector es indistinguible en latencia percibida de un socket para
una ventana que se actualiza a ojo humano, y no exige que ambos procesos estén vivos al mismo
tiempo para no romperse (`jarvis.ui.overlay` puede arrancar antes, después, o nunca, sin que
`run()` note la diferencia) — ver ADR-0008 para el detalle completo de la comparación.

`write_status()` es la única función que toca este archivo desde el lado de `run()`, y nunca
lanza: un fallo acá (disco lleno, permisos, alguna carrera rara con el propio directorio `data/`)
jamás puede tumbar un turno de voz real — mismo principio que ya aplica el resto de `run()` (ver
el `except Exception` de última línea del loop principal en `jarvis.audio.pipeline`) y que
`.claude/rules/python.md` reconoce como frontera de recuperación explícita, no un
`except Exception` silencioso genérico. El overlay es una capa de presentación completamente
opcional, sin autoridad sobre el pipeline (`.claude/rules/architecture.md`): que no se pueda
escribir su archivo de estado no es motivo para perder un comando de voz real.

`read_status()`/`is_stale()` son las dos funciones puras que usa `jarvis.ui.overlay` — separadas a
propósito del código de widgets de Tk para que sean testeables sin abrir una ventana real (mismo
criterio que el repo ya aplica en otros módulos: núcleo puro cubierto por tests, integración con
hardware/UI real no testeada directamente).

`StatusHeartbeat` existe por un hallazgo real, confirmado en vivo contra el proceso real de
`run()` durante la verificación de esta feature (no una hipótesis de diseño): escribir
`write_status()` solo en cada transición real dejaba el archivo sin tocar durante los minutos que
`run()` pasa bloqueado esperando la wake word (`next(detect(...))`, `jarvis.audio.wake_word`) —
el estado `idle` es, con diferencia, el más duradero de los cuatro. Pasados
`DEFAULT_STALE_AFTER_SECONDS`, el overlay empezaba a mostrarse como "desconectado" mientras
`run()` seguía vivo y funcionando con total normalidad, exactamente el falso negativo que el punto
4 del pedido original quería evitar (mostrar stale como si fuera un fallo real). `StatusHeartbeat`
resuelve esto reescribiendo el último estado conocido cada `HEARTBEAT_INTERVAL_SECONDS` desde un
hilo de fondo — mismo patrón de lifecycle `start()`/`stop()` idempotente que ya usan los otros
tres servicios de fondo de `run()` (`jarvis.audio.loopback.SystemAudioMonitor`,
`jarvis.league.lcu_monitor.LCUAutoAcceptMonitor`, `jarvis.audio.timer_scheduler.TimerScheduler`):
un hilo sin relación con ninguna decisión del LLM, que nunca puede tumbar un turno real porque
delega toda escritura a `write_status()` (que ya nunca lanza).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Relativo al directorio desde el que se corre JARVIS, mismo criterio que
# `jarvis.memory.store.DEFAULT_DB_PATH`/`jarvis.tools.screenshot.DEFAULT_SCREENSHOT_DIR` — no un
# path absoluto hardcodeado. Nivel de módulo y monkeypatcheable para que los tests nunca escriban
# bajo el `data/` real del repo.
DEFAULT_STATUS_PATH = Path("data/status.json")

# Tope de cuánto texto se persiste/muestra — esta es una ventana de estado "de un vistazo", no un
# log de conversación completo (ya cubierto por `jarvis.memory.store.conversation_turns`). Se
# trunca acá, en la escritura, no solo en el overlay al mostrarlo — así el archivo en sí nunca
# crece sin límite si algún texto fuera inusualmente largo.
MAX_LAST_TEXT_LENGTH = 200

# Un snapshot más viejo que esto se considera "el proceso de `run()` no está corriendo, o se
# colgó" — ver docstring de `is_stale`. Bastante por encima del intervalo de sondeo del overlay
# (`jarvis.ui.overlay.POLL_INTERVAL_MS`) y de cualquier pausa normal entre transiciones de estado
# de un turno real, pero lo bastante chico para que "JARVIS se cerró" se refleje en la ventana en
# segundos, no minutos.
DEFAULT_STALE_AFTER_SECONDS = 5.0

# Cada cuánto `StatusHeartbeat` reescribe el último estado conocido, incluso sin ninguna
# transición real de por medio. Bastante por debajo de `DEFAULT_STALE_AFTER_SECONDS` (menos de la
# mitad) para dejar margen real ante una demora puntual del hilo de fondo (contención del GIL con
# el resto de `run()`, o del propio disco) sin que un snapshot legítimamente fresco se muestre
# como stale por un pelo.
HEARTBEAT_INTERVAL_SECONDS = 2.0


class StatusState(str, Enum):
    """Los cuatro estados que `run()` puede reportar. `str, Enum` (no `Enum` simple) para que
    `state.value` sea directamente serializable a JSON sin conversión manual — mismo patrón que
    `jarvis.security.policy.RiskLevel` no usa (ese es un `Enum` común porque nunca se serializa a
    JSON), pero acá sí hace falta.

    No incluye un estado "desconectado"/"stale": eso no es algo que `run()` pueda reportar sobre
    sí mismo (si el proceso no está vivo o el archivo quedó viejo, no hay quién escriba ese
    estado) — es una conclusión que el propio overlay deriva de `is_stale()` sobre el
    `timestamp`, nunca un valor que aparezca en el archivo.
    """

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class StatusSnapshot:
    """Resultado de una lectura exitosa de `read_status()` — inmutable, sin lógica propia."""

    state: StatusState
    last_text: str
    timestamp: float


def write_status(
    state: StatusState,
    last_text: str = "",
    *,
    path: str | Path = DEFAULT_STATUS_PATH,
) -> None:
    """Escribir el estado actual del pipeline de voz a `path`, en un formato JSON plano.

    Escritura atómica (escribir a un archivo temporal y `os.replace()` sobre el destino) para que
    `read_status()`, corriendo en el otro proceso y sondeando cada ~200ms, nunca vea un JSON a
    medio escribir — `os.replace()` en Windows reemplaza el destino de forma atómica incluso si ya
    existe (a diferencia de `Path.rename()`, que en Windows falla si el destino existe).

    Nunca lanza (ver docstring del módulo): `OSError` cubre fallos reales de I/O (disco lleno,
    permisos, `data/` no creable en algún escenario raro); `TypeError`/`ValueError` cubren un
    fallo de serialización que hoy no debería poder pasar (`last_text` siempre es `str`, `state`
    siempre un `StatusState` válido) y `AttributeError` cubre un `state` inválido (ej. no un
    `StatusState`, sin `.value`) — el contrato de esta función es "nunca lanza", no "nunca
    debería lanzar en la práctica", así que el truncado de `last_text` y el armado del payload
    están dentro del mismo `try` que la escritura, no antes. Un error real se loguea a nivel
    WARNING (visible en `jarvis-error.log`, ver `run()`) y se descarta, nunca se re-lanza.
    """
    resolved_path = Path(path)
    tmp_path: Path | None = None
    try:
        truncated_text = last_text[:MAX_LAST_TEXT_LENGTH]
        payload = {
            "state": state.value,
            "last_text": truncated_text,
            "timestamp": time.time(),
        }
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        # Nombre de archivo temporal único por escritura (sufijo aleatorio de `mkstemp`, con
        # `O_EXCL` para garantizar que no colisiona con otro ya existente), no un `.tmp` fijo:
        # `write_status()` puede ser llamado concurrentemente desde el thread principal de `run()`
        # y desde el thread de fondo de `StatusHeartbeat` apuntando al mismo `path` — con un
        # nombre fijo, dos escrituras concurrentes pueden pisarse el archivo temporal entre sí y
        # hacer que `os.replace()` falle con `PermissionError`/`WinError 32` en Windows. `mkstemp`
        # en el mismo directorio que `resolved_path` garantiza unicidad y mantiene `os.replace()`
        # en el mismo volumen (requisito para que siga siendo atómico).
        fd, tmp_name = tempfile.mkstemp(
            dir=resolved_path.parent,
            prefix=resolved_path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(json.dumps(payload))
        os.replace(tmp_path, resolved_path)
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        logger.warning("No se pudo escribir el status del overlay (%s): %r", path, exc)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def read_status(path: str | Path = DEFAULT_STATUS_PATH) -> StatusSnapshot | None:
    """Leer y parsear `path`. Devuelve `None` si el archivo no existe, no es JSON válido, o le
    falta algún campo esperado — nunca lanza.

    `None` es un resultado esperado y frecuente, no un error a loguear ni propagar: el overlay
    puede arrancar antes de que `run()` haya escrito su primer estado, o `run()` puede no estar
    corriendo en absoluto — en ambos casos el llamador (`jarvis.ui.overlay`) trata `None` igual
    que un snapshot obsoleto (ver `is_stale`): se muestra como desconectado, no como un fallo.
    """
    resolved_path = Path(path)
    try:
        raw = resolved_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return StatusSnapshot(
            state=StatusState(data["state"]),
            last_text=str(data["last_text"]),
            timestamp=float(data["timestamp"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def is_stale(
    timestamp: float,
    *,
    now: float,
    max_age_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> bool:
    """`True` si `timestamp` (segundos, `time.time()`) es más viejo que `max_age_seconds`
    respecto de `now` — función pura, `now` siempre inyectado explícitamente (nunca
    `time.time()` adentro) para que el test no dependa del reloj real de la máquina que corre la
    suite, mismo criterio que `jarvis.audio.pipeline._current_time_line`.

    Un `timestamp` en el futuro (`now < timestamp`, ej. reloj del sistema ajustado hacia atrás
    entre la escritura y la lectura) nunca cuenta como stale — solo el paso del tiempo hacia
    adelante indica que `run()` dejó de actualizar el archivo.
    """
    return (now - timestamp) > max_age_seconds


class StatusHeartbeat:
    """Mantiene `path` fresco para `jarvis.ui.overlay`, incluso durante esperas largas sin
    ninguna transición de estado real (ver docstring del módulo para el hallazgo en vivo que
    motiva esto). `update()` es el único método que le importa al llamador (`run()`): escribe
    inmediatamente (mismo efecto que llamar a `write_status()` directo) y además recuerda el
    último `(state, last_text)` para que el hilo de fondo lo siga reescribiendo mientras no llegue
    una transición nueva.

    `start()`/`stop()` son idempotentes (llamar dos veces seguidas no crea un segundo hilo ni
    lanza si ya estaba parado) — mismo contrato que `SystemAudioMonitor`/`LCUAutoAcceptMonitor`/
    `TimerScheduler`. El hilo es `daemon=True`: si `run()` termina sin pasar por `stop()` (ej. un
    `SIGKILL` externo), no debe quedar colgado el proceso.
    """

    def __init__(
        self,
        *,
        path: str | Path = DEFAULT_STATUS_PATH,
        interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self._path = path
        self._interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._state = StatusState.IDLE
        self._last_text = ""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def update(self, state: StatusState, last_text: str = "") -> None:
        """Escribir `state`/`last_text` ahora mismo, y recordarlos como el último estado conocido
        para que el hilo de fondo los siga reescribiendo hasta la próxima transición."""
        with self._lock:
            self._state = state
            self._last_text = last_text
        write_status(state, last_text, path=self._path)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="jarvis-status-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 1.0)
        self._thread = None

    def _run(self) -> None:
        # `Event.wait(timeout)` en vez de `time.sleep(timeout)`: duerme el mismo tiempo, pero
        # `stop()` lo despierta al instante en vez de esperar hasta el próximo tick — mismo
        # patrón que ya usan los otros hilos de fondo de `run()` para un `stop()` responsivo.
        while not self._stop_event.wait(self._interval_seconds):
            with self._lock:
                state, last_text = self._state, self._last_text
            write_status(state, last_text, path=self._path)
