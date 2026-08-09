"""Tool para cancelar un timer pendiente antes de que se cumpla (`jarvis.tools.timer.TimerTool`).

Gap explícito dejado fuera de alcance cuando se construyó `TimerTool`/`TimerScheduler`: no había
ninguna forma de cancelar un timer una vez puesto. `TimerScheduler.schedule_timer` ya devolvía un
id (`jarvis.audio.timer_scheduler`) pero nada lo usaba — este tool es el primer consumidor, junto
a `TimerScheduler.cancel_timer`/`list_pending_timers` (agregados junto con este módulo).

El problema de UX real: el usuario dice "cancelá el timer de la pasta" o "cancelá mi último
timer" o "cancelá todos los timers" — nunca un id interno, porque el LLM nunca ve esos ids en la
conversación (no hay forma de que los pida de vuelta de forma confiable). Mismo problema, mismo
tipo de solución, que `close_app` (resolución de nombre de proceso por fuzzy-match) y
`lock_lol_champion` (resolución de nombre de campeón): este tool resuelve `target` (texto en
lenguaje natural) contra la lista de timers *pendientes en este momento*
(`TimerScheduler.list_pending_timers`), no contra un identificador que el usuario tendría que
recordar.

Dos formas de resolver `target` en `CancelTimerTool`, en este orden de precedencia:

1. Una palabra de "más reciente" (`_MOST_RECENT_WORDS`) → cancela el que se puso más
   recientemente (mayor id — los ids de `TimerScheduler` son estrictamente crecientes en orden de
   creación, ver `schedule_timer`), sin importar su `label`.
2. Cualquier otro texto → se interpreta como la etiqueta que el usuario le dio al timer al
   crearlo, y se resuelve con el mismo algoritmo de tres pasos que `jarvis.tools.close_app.
   _find_best_match` (exacto primero, después substring por cobertura relativa, después
   `difflib` como último recurso) — deliberadamente el mismo orden, no una variación nueva (ese
   orden existe por un fix concreto de `security-reviewer` sobre `open_app.py`: un match exacto
   nunca puede perder contra un substring más corto pero no relacionado). Solo compara contra
   timers que SÍ tienen `label` — uno sin etiqueta no puede resolverse por nombre, solo por
   "último".

Deliberadamente proporcionado, no un parser de referencias completo (`CLAUDE.md`: "sin
abstracciones prematuras") — label-fuzzy-match + "más reciente" cubre el uso de voz real sin
construir NLU de fechas/referencias.

`CancelTimerTool` es SAFE bajo `.claude/rules/security.md`: cancelar UN timer puntual es, si
acaso, MENOS invasivo que ponerlo (`TimerTool`, ya SAFE) — descarta una única entrada efímera en
memoria de JARVIS, sin ningún efecto sobre archivos, procesos o configuración real del sistema, y
totalmente reversible (el usuario puede volver a pedir el mismo timer al instante si se equivocó).

Historia (ver ADR-0006 para el razonamiento completo, ya aplicado esta misma noche a
`jarvis.tools.lol_champion_select`): originalmente este mismo tool también aceptaba `target`
igual a "todos"/"todas" y cancelaba TODOS los timers pendientes de una sola vez, clasificado SAFE
junto con el resto. Un `security-reviewer` marcó que ese radio de impacto (borrar TODO lo
pendiente en una sola frase, sin ninguna oportunidad de decir que no) no es equivalente al de
cancelar un único timer identificado — sobre todo porque `target` es texto libre transcripto por
voz: un "todo"/"todos" mal escuchado en una frase que en realidad pedía cancelar un timer puntual
podría silenciosamente vaciar todo lo pendiente, y el usuario recién lo notaría cuando un
recordatorio/timer esperado nunca suena. Mismo patrón que `PickChampionTool` en ADR-0006: se
parte en dos tools en vez de evaluar riesgo condicional por parámetro (`Tool.risk` es estático a
propósito, ADR-0005) — `CancelTimerTool` se queda SAFE y pierde la rama "todos" por completo (ya
no reconoce esa palabra en `target`, ni falta que la reconozca); `CancelAllTimersTool`
(`cancel_all_timers`), CONFIRM y sin parámetros (cancela explícitamente todo lo pendiente, nada
que resolver en lenguaje natural), requiere confirmación hablada antes de ejecutarse —
`describe()` nombra cuántos timers hay pendientes ahora mismo, no un genérico "¿confirmás
cancelar todo?" (mismo motivo que `LockChampionTool.describe`/`SetLobbyQueueTool.describe`: el
usuario tiene que poder decir que no con la información real, no una frase vacía).
"""

from __future__ import annotations

import asyncio
import difflib
from typing import Any, ClassVar

from jarvis.audio.timer_scheduler import PendingTimer, TimerScheduler
from jarvis.tools.base import RiskLevel, Tool

# Mismo cutoff que `jarvis.tools.close_app`/`jarvis.tools.open_app` — mismo balance entre tolerar
# typos/variaciones de cómo el usuario dijo la etiqueta y no volverse tan laxo que cualquier
# palabra corta matchee con cualquier timer pendiente.
_FUZZY_MATCH_CUTOFF = 0.45

_MOST_RECENT_WORDS = frozenset(
    {"ultimo", "último", "ultima", "última", "reciente", "recien", "recién"}
)

_MISSING_TARGET_MESSAGE = "No se especificó qué timer cancelar."
_NO_PENDING_TIMERS_MESSAGE = "No hay ningún timer pendiente para cancelar."
_RACE_LOST_MESSAGE = (
    "No pude cancelar ese timer: se cumplió justo antes de que llegara a cancelarlo."
)


def _cancel_all_pending_timers_sync(scheduler: TimerScheduler) -> str:
    """Cuerpo compartido de `CancelAllTimersTool.execute` — separado a nivel de módulo por el
    mismo motivo que `_select_champion_sync` en `jarvis.tools.lol_champion_select` (ADR-0006):
    la clase wrapper es lo único que debería duplicarse entre tools hermanos, no la lógica."""
    pending = scheduler.list_pending_timers()
    if not pending:
        return _NO_PENDING_TIMERS_MESSAGE
    cancelled_count = sum(1 for timer in pending if scheduler.cancel_timer(timer.id))
    if cancelled_count == 0:
        return _NO_PENDING_TIMERS_MESSAGE
    unit = "timer" if cancelled_count == 1 else "timers"
    return f"Listo, cancelé {cancelled_count} {unit}."


def _describe_cancel_all_timers(scheduler: TimerScheduler) -> str:
    """Frase natural para el prompt de confirmación de `CancelAllTimersTool.describe` — nombra
    cuántos timers hay pendientes AHORA (`pending_timer_count`, lectura en memoria, sin I/O real)
    en vez de un genérico "¿confirmás cancelar todo?", mismo criterio de especificidad que
    `LockChampionTool.describe`/`SetLobbyQueueTool.describe`."""
    count = scheduler.pending_timer_count
    if count == 0:
        return "cancelar todos los timers pendientes (no hay ninguno pendiente ahora mismo)"
    unit = "timer" if count == 1 else "timers"
    return f"cancelar TODOS los {count} {unit} pendientes"


def _normalize(text: str) -> str:
    return text.strip().lower()


def _contains_any_word(text: str, trigger_words: frozenset[str]) -> bool:
    """`True` si alguna palabra de `text` (normalizada, sin puntuación) está en `trigger_words` —
    mismo helper que `_contains_any_word` en `jarvis.audio.pipeline` (duplicado a propósito, ver
    `_escape_untrusted` en ese módulo para el mismo razonamiento: es un detalle interno de este
    tool, no un contrato compartido)."""
    normalized = text.strip().lower()
    words = [word.strip(".,!¡¿?;:") for word in normalized.split()]
    return any(word in trigger_words for word in words if word)


def _find_best_label_match(
    query: str, timers: list[PendingTimer]
) -> PendingTimer | None:
    """Encontrar el timer pendiente cuya `label` mejor matchea `query` — mismo algoritmo en tres
    pasos que `jarvis.tools.close_app._find_best_match` (ver docstring del módulo), aplicado
    sobre etiquetas de timer en vez de nombres de proceso. Solo considera timers con `label` no
    vacío."""
    labeled_timers = [timer for timer in timers if timer.label]
    if not labeled_timers:
        return None

    normalized_query = _normalize(query)
    timers_by_label: dict[str, PendingTimer] = {}
    for timer in labeled_timers:
        assert timer.label is not None  # filtrado arriba
        key = _normalize(timer.label)
        # Ante etiquetas duplicadas, se conserva la primera — cualquiera de los dos timers con esa
        # etiqueta es un target razonable para cancelar.
        timers_by_label.setdefault(key, timer)

    if normalized_query in timers_by_label:
        return timers_by_label[normalized_query]

    substring_matches = [
        label
        for label in timers_by_label
        if normalized_query in label or label in normalized_query
    ]
    if substring_matches:

        def _coverage(label: str) -> float:
            shorter = min(len(label), len(normalized_query))
            longer = max(len(label), len(normalized_query))
            return shorter / longer if longer else 0.0

        best_label = max(
            substring_matches, key=lambda label: (_coverage(label), -len(label))
        )
        return timers_by_label[best_label]

    close = difflib.get_close_matches(
        normalized_query, timers_by_label.keys(), n=1, cutoff=_FUZZY_MATCH_CUTOFF
    )
    if not close:
        return None
    return timers_by_label[close[0]]


def _confirm_cancelled(timer: PendingTimer) -> str:
    if timer.label:
        return f"Listo, cancelé el timer de {timer.label}."
    return "Listo, cancelé el timer."


class CancelTimerTool(Tool):
    """Cancela UN timer pendiente puntual, resolviendo el pedido en lenguaje natural contra los
    timers actualmente en memoria (`TimerScheduler.list_pending_timers`) — nunca "todos", ver
    `CancelAllTimersTool` para eso (ADR-0006, docstring del módulo)."""

    name = "cancel_timer"
    description = (
        "Cancela un timer puntual que ya fue puesto y todavía no se cumplió (ej. 'cancelá el "
        "timer de la pasta', 'cancelá mi último timer'). Usar cuando el usuario pide cancelar, "
        "parar o sacar UN timer en curso — no confundir con set_timer (poner uno nuevo), con "
        "cancel_reminder (para recordatorios en vez de timers), ni con cancel_all_timers (cuando "
        "el usuario pide cancelar TODOS los timers pendientes de una vez)."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "A qué timer se refiere el usuario: la etiqueta que le dio al ponerlo (ej. "
                    "'la pasta'), o la palabra 'último' si pidió cancelar el más reciente sin "
                    "dar una etiqueta."
                ),
            }
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def __init__(self, *, scheduler: TimerScheduler) -> None:
        # Misma instancia compartida que `TimerTool` (`jarvis.audio.pipeline.run()`) — nunca una
        # propia, o este tool cancelaría timers de un scheduler que nadie sondea.
        self._scheduler = scheduler

    async def execute(self, **kwargs: Any) -> str:
        target = kwargs.get("target")
        if not isinstance(target, str) or not target.strip():
            return _MISSING_TARGET_MESSAGE

        # `list_pending_timers`/`cancel_timer` son operaciones en memoria (lock corto, sin I/O) —
        # no hace falta `asyncio.to_thread`, mismo criterio que `TimerTool.execute`.
        pending = self._scheduler.list_pending_timers()
        if not pending:
            return _NO_PENDING_TIMERS_MESSAGE

        if _contains_any_word(target, _MOST_RECENT_WORDS):
            most_recent = max(pending, key=lambda timer: timer.id)
            await asyncio.sleep(0)
            if not self._scheduler.cancel_timer(most_recent.id):
                return _RACE_LOST_MESSAGE
            return _confirm_cancelled(most_recent)

        match = _find_best_label_match(target, pending)
        if match is None:
            return f"No encontré ningún timer pendiente que coincida con '{target}'."
        await asyncio.sleep(0)
        if not self._scheduler.cancel_timer(match.id):
            return _RACE_LOST_MESSAGE
        return _confirm_cancelled(match)


class CancelAllTimersTool(Tool):
    """Cancela TODOS los timers pendientes de una sola vez (ADR-0006) — CONFIRM: requiere
    confirmación explícita por voz antes de ejecutarse, `describe()` nombra cuántos timers hay
    pendientes ahora mismo (ver docstring del módulo)."""

    name = "cancel_all_timers"
    description = (
        "Cancela TODOS los timers pendientes de una sola vez (ej. 'cancelá todos los timers', "
        "'borrá todos mis timers'). Usar únicamente cuando el usuario pide explícitamente "
        "cancelar TODOS a la vez, nunca para cancelar uno puntual (para eso está cancel_timer). "
        "Requiere confirmación explícita del usuario antes de ejecutarse."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    risk = RiskLevel.CONFIRM

    def __init__(self, *, scheduler: TimerScheduler) -> None:
        # Misma instancia compartida que `TimerTool`/`CancelTimerTool` — nunca una propia.
        self._scheduler = scheduler

    def describe(self, kwargs: dict[str, Any]) -> str:
        return _describe_cancel_all_timers(self._scheduler)

    async def execute(self, **kwargs: Any) -> str:
        # Igual que `CancelTimerTool.execute`: operación en memoria (lock corto, sin I/O), no
        # hace falta `asyncio.to_thread` — solo ceder el loop, mismo motivo que
        # `TimerTool.execute`.
        result = _cancel_all_pending_timers_sync(self._scheduler)
        await asyncio.sleep(0)
        return result
