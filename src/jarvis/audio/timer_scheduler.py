"""Scheduler de fondo para timers y recordatorios: la primera fuente de habla *proactiva* de
JARVIS (post-ADR-0005).

Todo tool existente hasta ahora solo *responde* a un turno de voz — `Tool.execute()` devuelve un
string que el dispatch loop (`jarvis.audio.pipeline.dispatch_turn`) mete en la respuesta de ESE
turno. Un timer/recordatorio necesita anunciarse cuando vence, sin que haya ningún turno en curso
— el usuario puede estar en silencio, hablando de otra cosa, o el propio JARVIS puede estar en
medio de otra tarea. Eso no puede vivir en el resultado de un `execute()`: necesita un componente
independiente, con su propio hilo de vida, igual que `jarvis.audio.loopback.SystemAudioMonitor`
(medición continua de audio del sistema en un thread de fondo) — mismo patrón de
`start()`/`stop()` idempotentes, un solo thread para toda la corrida, construido una vez en
`jarvis.audio.pipeline.run()`.

`TimerScheduler` NO es un `Tool` ni pasa por `PolicyEngine`: hablar por TTS para anunciar algo que
el propio sistema programó (a pedido explícito y ya autorizado del usuario, vía
`TimerTool`/`ReminderTool`, ambos SAFE) no es una acción nueva que necesite reclasificarse — es la
misma clase de "hablar sin que se lo pidan en este instante" que ya hace `run()` al arrancar
("Alexa activa y funcionando correctamente.", `tts.speak()` directo, sin tool ni policy de por
medio). El límite de capas de `.claude/rules/architecture.md` ("el planner nunca llama a Windows
directo") no aplica acá: este componente no es el planner, es infraestructura del lado de
audio/pipeline, análoga a `SystemAudioMonitor`.

Dos fuentes de items pendientes, con ciclos de vida distintos (ver docstring de
`jarvis.memory.store` para la justificación completa):

- **Timers**: puramente en memoria (`PendingTimer`, lista protegida por `_lock`) — efímeros,
  minutos de vida típica, no sobreviven un reinicio de JARVIS (aceptado como degradación menor).
- **Recordatorios**: persistidos en SQLite (`jarvis.memory.store.reminders`, vía `save_reminder`/
  `list_due_reminders`/`delete_reminder`) — pueden tener horas o días de vida, así que sí
  necesitan sobrevivir un reinicio.

`poll_once()` es el corazón puro y testeable de este módulo: revisa ambas fuentes contra un `now`
inyectado (nunca `time.sleep` real en los tests, ver `tests/audio/test_timer_scheduler.py`),
anuncia lo vencido vía `tts.speak()`, y lo descarta (memoria) o lo borra (DB) para que nunca se
anuncie dos veces. `_run()` es el wrapper de thread de fondo que llama a `poll_once()` en un loop
con `threading.Event.wait(poll_interval)` en vez de `time.sleep` — permite que `stop()` interrumpa
la espera al instante en vez de tener que esperar hasta el próximo tick, mismo motivo que
`SystemAudioMonitor` usa un `Event` para su loop.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis.audio.tts import TTSClient
from jarvis.memory.store import DEFAULT_DB_PATH as MEMORY_DEFAULT_DB_PATH
from jarvis.memory.store import delete_reminder, list_due_reminders

# Intervalo de sondeo del thread de fondo: suficientemente seguido para que un timer/recordatorio
# se anuncie con una demora imperceptible para una interacción de voz (unos segundos de margen,
# no minutos), sin sondear tan seguido que el thread consuma CPU de más por algo que no cambia
# la mayor parte del tiempo (`SELECT` a SQLite incluido en cada poll de recordatorios).
DEFAULT_POLL_INTERVAL_SECONDS = 2.0

# Tope duro sobre el total de timers *pendientes* a la vez (hallazgo MEDIUM de `security-reviewer`,
# análogo a `MAX_STORED_REMINDERS` en `jarvis.memory.store` — mismo orden de magnitud, elegido a
# propósito para no reinventar el criterio ya establecido ahí). A diferencia de `reminders`, los
# timers no tienen ninguna poda existente: solo se descartan cuando vencen y `poll_once()` los
# anuncia (`_announce_due_timers`), así que un `_timers` sin tope crecería sin límite en una
# corrida larga (horas/días, mismo perfil que `SystemAudioMonitor`/`LCUAutoAcceptMonitor`) si el
# usuario (o una alucinación del LLM) pide `set_timer` repetidamente sin que ninguno llegue a
# cumplirse. `schedule_timer` rechaza devolviendo `None` en vez de podar los más viejos como
# `save_fact`/`save_reminder`: descartar en silencio un timer que el usuario cree que sigue
# pendiente es peor que rechazar el nuevo pedido con un mensaje claro (mismo principio que ya
# aplica `save_reminder` al rechazar un `due_at` inválido en vez de fallar en silencio más tarde).
MAX_PENDING_TIMERS = 100


@dataclass
class PendingTimer:
    """Un timer en memoria (ver docstring del módulo: nunca persiste a disco), expuesto en
    lectura para que un tool de cancelación (`jarvis.tools.cancel_timer`) pueda resolver un
    pedido en lenguaje natural ("cancelá el timer de la pasta", "cancelá mi último timer")
    contra la lista de timers *actualmente* pendientes, sin que el LLM tenga que conocer ni decir
    un id interno — mismo problema y misma solución que la resolución por nombre de
    `close_app`/`lock_lol_champion` (ver `.claude/rules/agents.md` y esos módulos)."""

    id: int
    label: str | None
    fire_at: datetime


class TimerScheduler:
    """Sondea timers en memoria y recordatorios persistidos, anunciando por TTS los que ya
    vencieron.

    Construido una vez por corrida de `run()` (`jarvis.audio.pipeline`), con el mismo `tts` que
    usa el resto del pipeline — `.start()` cerca de donde ya arranca `system_audio.start()`,
    `.stop()` en el mismo `finally`.
    """

    def __init__(
        self,
        *,
        tts: TTSClient,
        db_path: str | Path = MEMORY_DEFAULT_DB_PATH,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._tts = tts
        self._db_path = db_path
        self._poll_interval = poll_interval_seconds
        self._timers: list[PendingTimer] = []
        self._next_timer_id = 1
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def schedule_timer(
        self, *, seconds: float, label: str | None = None, now: datetime | None = None
    ) -> int | None:
        """Registrar un timer en memoria que vence en `seconds` a partir de `now` (default:
        ahora mismo, UTC). Devuelve un id local (solo único dentro de este `TimerScheduler`, no
        persistido) — usado por `cancel_timer` para cancelarlo antes de que venza (el llamador
        típico, `jarvis.tools.timer.TimerTool`, descarta este valor de retorno: la resolución de
        "qué timer cancelar" a partir de lenguaje natural vive en `jarvis.tools.cancel_timer`,
        contra `list_pending_timers`, no acá).

        Devuelve `None` en vez de registrar nada si ya hay `MAX_PENDING_TIMERS` timers pendientes
        — el llamador (`jarvis.tools.timer.TimerTool.execute`) es responsable de convertir ese
        `None` en un mensaje claro para el usuario, en vez de confirmar un timer que en realidad
        nunca se registró.

        `now` inyectable (no siempre `datetime.now(UTC)` hardcodeado adentro) por el mismo motivo
        que el resto de este módulo: tests determinísticos con un reloj falso, sin `time.sleep`.
        """
        resolved_now = now if now is not None else datetime.now(UTC)
        fire_at = resolved_now + timedelta(seconds=seconds)
        with self._lock:
            if len(self._timers) >= MAX_PENDING_TIMERS:
                return None
            timer_id = self._next_timer_id
            self._next_timer_id += 1
            self._timers.append(PendingTimer(id=timer_id, label=label, fire_at=fire_at))
        return timer_id

    def cancel_timer(self, timer_id: int) -> bool:
        """Cancelar el timer pendiente `timer_id` (el id devuelto por `schedule_timer`), sin
        anunciarlo — a diferencia de `_announce_due_timers`, acá se descarta en silencio porque
        cancelar es justamente lo contrario de "avisame cuando se cumpla".

        Devuelve `True` si había un timer con ese id (y quedó cancelado), `False` si no existía
        (ya venció y se anunció, ya se canceló antes, o nunca existió) — idempotente, mismo
        contrato que `jarvis.memory.store.delete_reminder`. El llamador (`jarvis.tools.
        cancel_timer.CancelTimerTool`) es responsable de resolver un pedido en lenguaje natural
        a un `timer_id` concreto ANTES de llamar a esto (ver `list_pending_timers`) — este método
        no hace fuzzy-matching, solo opera sobre un id ya resuelto.
        """
        with self._lock:
            remaining = [timer for timer in self._timers if timer.id != timer_id]
            cancelled = len(remaining) != len(self._timers)
            self._timers = remaining
        return cancelled

    def list_pending_timers(self) -> list[PendingTimer]:
        """Devolver una copia de los timers en memoria que todavía no vencieron — para que un
        tool de cancelación pueda resolver "el timer de la pasta"/"mi último timer"/"todos" contra
        el estado actual sin tener que exponer `_timers` (protegido por `_lock`) directamente.

        Copia superficial (`list(...)`, no la lista interna): quien llama a esto puede iterar o
        filtrar sin arriesgar una mutación concurrente de la lista real mientras el poll loop de
        fondo corre en su propio thread.
        """
        with self._lock:
            return list(self._timers)

    @property
    def pending_timer_count(self) -> int:
        """Cuántos timers en memoria siguen sin vencer — expuesto para tests."""
        with self._lock:
            return len(self._timers)

    def start(self) -> None:
        """Arrancar el thread de fondo. Idempotente: no hace nada si ya está corriendo (mismo
        contrato que `SystemAudioMonitor.start()`)."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Señalizar el fin del thread de fondo y esperar a que termine. Idempotente. Vuelve casi
        al instante aunque `poll_interval_seconds` sea grande: `_run()` espera con
        `threading.Event.wait()`, que `_stop_event.set()` interrumpe de inmediato en vez de tener
        que esperar al próximo tick."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def poll_once(self, *, now: datetime | None = None) -> None:
        """Revisar timers en memoria y recordatorios persistidos contra `now` (default: ahora
        mismo, UTC) y anunciar por TTS los que ya vencieron. Método público y sin efectos de
        thread/tiempo real — el punto de entrada que testea
        `tests/audio/test_timer_scheduler.py` inyectando `now` y un `tts` espía, sin
        `threading.Timer` ni sleeps reales.
        """
        resolved_now = now if now is not None else datetime.now(UTC)
        self._announce_due_timers(resolved_now)
        self._announce_due_reminders(resolved_now)

    def _announce_due_timers(self, now: datetime) -> None:
        with self._lock:
            due = [timer for timer in self._timers if timer.fire_at <= now]
        for timer in due:
            if timer.label:
                message = f"¡Se cumplió tu timer de {timer.label}!"
            else:
                message = "¡Se cumplió tu timer!"
            # Anunciar ANTES de descartar el timer de la lista en memoria — mismo orden que
            # `_announce_due_reminders` con `delete_reminder`: `tts.speak()` puede lanzar de
            # verdad en el camino normal (sin fallback local, ver `jarvis.audio.tts`/ADR-0011, un
            # fallo de la API de OpenAI se propaga tal cual), y `_run()` más abajo atrapa esa
            # excepción a nivel de poll — es preferible reintentar el anuncio en el próximo poll a
            # perder el timer sin haberlo dicho nunca.
            self._tts.speak(message)
            with self._lock:
                self._timers = [t for t in self._timers if t.id != timer.id]

    def _announce_due_reminders(self, now: datetime) -> None:
        for reminder in list_due_reminders(now, db_path=self._db_path):
            self._tts.speak(f"Recordatorio: {reminder.text}")
            # Borrar DESPUÉS de hablar, no antes — mismo razonamiento que `_announce_due_timers`.
            delete_reminder(reminder.id, db_path=self._db_path)

    def _run(self) -> None:
        # `Event.wait(timeout)` devuelve `True` apenas `_stop_event.set()` se llama, sin esperar
        # el resto del intervalo — mismo motivo que documenta `stop()`. Mientras no se pida parar,
        # devuelve `False` tras `poll_interval` segundos y el loop vuelve a sondear.
        while not self._stop_event.wait(self._poll_interval):
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 — último nivel de defensa de este thread de
                # fondo: un error puntual (ej. la DB de recordatorios bloqueada un instante por
                # otra escritura concurrente) no puede tumbar el scheduler entero silenciosamente
                # — se loguea y el próximo poll simplemente reintenta, mismo espíritu que el
                # `except Exception` documentado de `run()` en `jarvis.audio.pipeline`.
                print(f"Error en el poll de TimerScheduler: {exc!r}", file=sys.stderr)
