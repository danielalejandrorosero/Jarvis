"""Tool para poner un timer de cuenta regresiva (ADR-0005, extendida con anuncio proactivo —
ver `jarvis.audio.timer_scheduler`).

Distinto de todos los tools anteriores en una dimensión clave: `execute()` acá NO hace el trabajo
completo del pedido. Confirma que el timer quedó registrado ("Listo, timer de 10 minutos
puesto.") — el aviso real de que el timer se cumplió pasa por `TimerScheduler`, un componente de
fondo completamente separado del ciclo tool-call → respuesta de este turno (ver docstring de ese
módulo para el porqué). Este tool solo *registra* el timer con el scheduler compartido, inyectado
por constructor — mismo patrón que `RememberTool` recibe `db_path`, acá se recibe la instancia de
`TimerScheduler` en sí, porque el estado que hay que mutar (la lista de timers pendientes) vive en
memoria de proceso, no en un archivo que se pueda re-abrir por path.

SAFE bajo `.claude/rules/security.md`: registrar una notificación futura no muta nada visible del
sistema (ni archivos, ni procesos, ni configuración) — es, si acaso, menos invasivo que
`remember_fact` (que sí persiste a disco); acá lo único que se crea es una entrada efímera en
memoria que desaparece sola al vencer o al reiniciar JARVIS.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from jarvis.audio.timer_scheduler import TimerScheduler
from jarvis.tools.base import RiskLevel, Tool

# Tope duro de duración de un timer: 24 horas. Un pedido más largo que esto ya no es un "timer" en
# el sentido de cocina/pomodoro que motiva este tool — es, en la práctica, un recordatorio
# (`ReminderTool`), que sí persiste a disco y sobrevive un reinicio. Evita además que un timer
# quede en memoria indefinidamente si el usuario (o una alucinación del LLM sobre la duración)
# pide algo desproporcionado.
MAX_TIMER_SECONDS = 24 * 60 * 60

MAX_LABEL_LENGTH = 100


def _format_duration(seconds: int) -> str:
    """Frase corta y natural para la confirmación hablada: minutos si es un múltiplo exacto de
    60 (el caso común, "10 minutos" en vez de "600 segundos"), segundos si no."""
    if seconds % 60 == 0:
        minutes = seconds // 60
        unit = "minuto" if minutes == 1 else "minutos"
        return f"{minutes} {unit}"
    unit = "segundo" if seconds == 1 else "segundos"
    return f"{seconds} {unit}"


class TimerTool(Tool):
    """Pone un timer de cuenta regresiva que JARVIS anuncia por voz apenas se cumple."""

    name = "set_timer"
    description = (
        "Pone un timer de cuenta regresiva (ej. 'poné un timer de 10 minutos', 'avisame en 5 "
        "minutos'). JARVIS lo anuncia en voz alta apenas se cumple, sin que el usuario tenga que "
        "preguntar. Usar para duraciones cortas relativas a ahora (minutos, alguna hora) — para "
        "un aviso en un momento específico del día o más adelante en el tiempo (ej. 'a las 5', "
        "'mañana'), usar set_reminder en cambio."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "integer",
                "description": (
                    "Duración del timer en segundos. Convertí vos mismo minutos/horas a "
                    "segundos (ej. 'timer de 10 minutos' -> 600)."
                ),
            },
            "label": {
                "type": "string",
                "description": (
                    "Etiqueta corta opcional para identificar el timer al anunciarlo (ej. "
                    "'la pasta', 'el lavarropas'). Omitir si el usuario no dio ningún contexto."
                ),
            },
        },
        "required": ["seconds"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def __init__(self, *, scheduler: TimerScheduler) -> None:
        # Instancia compartida construida una vez en `jarvis.audio.pipeline.run()` (mismo
        # `TimerScheduler` que corre el poll loop de fondo) — nunca una instancia propia de este
        # tool, o el timer registrado acá nunca sería sondeado por nadie.
        self._scheduler = scheduler

    async def execute(self, **kwargs: Any) -> str:
        seconds = kwargs.get("seconds")
        # `bool` es subclase de `int` en Python — sin este chequeo explícito, `seconds=True`
        # pasaría el `isinstance(seconds, int)` y se interpretaría como un timer de 1 segundo.
        if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds <= 0:
            return "No se especificó una duración válida para el timer."
        if seconds > MAX_TIMER_SECONDS:
            return (
                f"La duración máxima de un timer es {MAX_TIMER_SECONDS // 3600} horas; para "
                "algo más largo usá un recordatorio en su lugar."
            )

        label = kwargs.get("label")
        if isinstance(label, str) and label.strip():
            label = label.strip()[:MAX_LABEL_LENGTH]
        else:
            label = None

        # `schedule_timer` es una operación en memoria (lock corto, sin I/O) — no hace falta
        # `asyncio.to_thread` acá, a diferencia de `save_fact`/`save_reminder` (sqlite3
        # bloqueante). Devuelve `None` en vez de un id si ya se alcanzó `MAX_PENDING_TIMERS`
        # (hallazgo MEDIUM de `security-reviewer`, ver `jarvis.audio.timer_scheduler`) — nunca hay
        # que confirmarle al usuario un timer que en realidad no quedó registrado.
        timer_id = self._scheduler.schedule_timer(seconds=float(seconds), label=label)
        await asyncio.sleep(
            0
        )  # ceder el loop, mismo motivo que cualquier `async def` trivial

        if timer_id is None:
            return (
                "No pude poner el timer: ya hay demasiados timers pendientes. Esperá a que "
                "alguno se cumpla, o cancelá alguno, antes de pedir uno nuevo."
            )

        duration_phrase = _format_duration(seconds)
        if label:
            return f"Listo, timer de {duration_phrase} puesto para {label}."
        return f"Listo, timer de {duration_phrase} puesto."
