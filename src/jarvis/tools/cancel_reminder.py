"""Tool para cancelar un recordatorio pendiente antes de que se cumpla
(`jarvis.tools.reminder.ReminderTool`).

Mismo gap y mismo tipo de solución que `jarvis.tools.cancel_timer.CancelTimerTool` (ver docstring
de ese módulo), aplicado sobre `reminders` (`jarvis.memory.store`) en vez de los timers en memoria
de `TimerScheduler`. Tool separado, no combinado con `CancelTimerTool`, por el mismo motivo que
`TimerTool`/`ReminderTool` ya son tools separados: fuentes de datos y ciclos de vida distintos
(memoria efímera vs SQLite persistido, ver docstring de `jarvis.memory.store`) — combinarlos
forzaría un solo tool a hablarle a dos backends con dos formas de identificar filas (`timer_id`
local vs `reminder_id` de SQLite), sin ganar nada real en la experiencia de voz (el usuario ya
distingue "timer" de "recordatorio" al pedirlos, así que distinguirlos también al cancelar es
consistente, no una carga extra).

Mismas dos formas de resolver `target` en `CancelReminderTool` (más reciente / texto-fuzzy) que
`CancelTimerTool`, contra el `text` de cada `Reminder` en vez de una `label` opcional — a
diferencia de un timer, un recordatorio siempre tiene texto no vacío (`ReminderTool.execute`
rechaza `text` vacío antes de guardar), así que acá no hace falta filtrar por "tiene o no tiene
etiqueta" antes de intentar el fuzzy-match. "Más reciente" se resuelve por `id` más alto — los ids
de `reminders` son autoincrementales en SQLite, así que el mayor id es el creado más
recientemente, mismo criterio que `CancelTimerTool` usa sobre los ids de `TimerScheduler`.

`CancelReminderTool` es SAFE bajo `.claude/rules/security.md`: mismo razonamiento que
`ReminderTool` (SAFE) y `CancelTimerTool` (SAFE) — mutar el store SQLite interno de JARVIS
borrando UNA fila puntual de `reminders` es reversible (el usuario puede volver a pedir el mismo
recordatorio) y de impacto acotado a un anuncio hablado que JARVIS mismo iba a hacer, no un
archivo/proceso/configuración real del usuario.

Historia (ver ADR-0006 para el razonamiento completo, ya aplicado esta misma noche a
`jarvis.tools.lol_champion_select` y a `jarvis.tools.cancel_timer`): originalmente este mismo tool
también aceptaba `target` igual a "todos"/"todas" y borraba TODAS las filas de `reminders` de una
sola vez, clasificado SAFE. Un `security-reviewer` marcó que esto viola textualmente
`.claude/rules/security.md`: la excepción SAFE para el store SQLite propio de JARVIS "no se
extiende a operaciones de borrado masivo o irreversibles sobre ese mismo store (eso es CONFIRM o
DANGEROUS según impacto)" — borrar TODOS los recordatorios pendientes en una sola frase, sin
ninguna oportunidad de decir que no, es exactamente ese caso, agravado porque `target` es texto
libre transcripto por voz (un "todo"/"todos" mal escuchado en un pedido que en realidad quería
cancelar un recordatorio puntual vaciaría todo lo pendiente en silencio, y el usuario recién lo
notaría cuando un recordatorio esperado nunca suena). Mismo patrón de ADR-0006: se parte en dos
tools en vez de evaluar riesgo condicional por parámetro — `CancelReminderTool` se queda SAFE y
pierde la rama "todos" por completo; `CancelAllRemindersTool` (`cancel_all_reminders`), CONFIRM y
sin parámetros, requiere confirmación hablada antes de ejecutarse — `describe()` nombra cuántos
recordatorios hay pendientes ahora mismo, no un genérico "¿confirmás cancelar todo?" (mismo
motivo que `LockChampionTool.describe`/`SetLobbyQueueTool.describe`).
"""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path
from typing import Any, ClassVar

from jarvis.memory.store import (
    DEFAULT_DB_PATH,
    Reminder,
    delete_reminder,
    list_reminders,
)
from jarvis.tools.base import RiskLevel, Tool

# Mismo cutoff que `jarvis.tools.cancel_timer`/`jarvis.tools.close_app`.
_FUZZY_MATCH_CUTOFF = 0.45

_MOST_RECENT_WORDS = frozenset(
    {"ultimo", "último", "ultima", "última", "reciente", "recien", "recién"}
)

_MISSING_TARGET_MESSAGE = "No se especificó qué recordatorio cancelar."
_NO_PENDING_REMINDERS_MESSAGE = "No hay ningún recordatorio pendiente para cancelar."


async def _cancel_all_pending_reminders(db_path: str | Path) -> str:
    """Cuerpo compartido de `CancelAllRemindersTool.execute` — separado a nivel de módulo por el
    mismo motivo que `_select_champion_sync` en `jarvis.tools.lol_champion_select` (ADR-0006): la
    clase wrapper es lo único que debería duplicarse entre tools hermanos, no la lógica.

    Async (no `_sync`, a diferencia de su equivalente en `cancel_timer.py`) porque, a diferencia
    de un `TimerScheduler` en memoria, esto es I/O de archivo bloqueante (sqlite3, stdlib) —
    `asyncio.to_thread` por cada llamada, mismo patrón que `CancelReminderTool.execute`."""
    pending = await asyncio.to_thread(list_reminders, db_path=db_path)
    if not pending:
        return _NO_PENDING_REMINDERS_MESSAGE
    for reminder in pending:
        await asyncio.to_thread(delete_reminder, reminder.id, db_path=db_path)
    unit = "recordatorio" if len(pending) == 1 else "recordatorios"
    return f"Listo, cancelé {len(pending)} {unit}."


def _describe_cancel_all_reminders(db_path: str | Path) -> str:
    """Frase natural para el prompt de confirmación de `CancelAllRemindersTool.describe` —
    nombra cuántos recordatorios hay pendientes AHORA en vez de un genérico "¿confirmás cancelar
    todo?", mismo criterio de especificidad que `LockChampionTool.describe`/
    `SetLobbyQueueTool.describe`.

    Síncrono por contrato de `Tool.describe()` (`base.py`) — corre `list_reminders` (sqlite3,
    stdlib) directo, sin `asyncio.to_thread`, misma excepción puntual y documentada a
    `.claude/rules/python.md` que `CloseAppTool.describe`: una única lectura local rápida, al
    mismo costo que ya se paga dentro de `execute()`."""
    count = len(list_reminders(db_path=db_path))
    if count == 0:
        return (
            "cancelar todos los recordatorios pendientes (no hay ninguno pendiente "
            "ahora mismo)"
        )
    unit = "recordatorio" if count == 1 else "recordatorios"
    return f"cancelar TODOS los {count} {unit} pendientes"


def _normalize(text: str) -> str:
    return text.strip().lower()


def _contains_any_word(text: str, trigger_words: frozenset[str]) -> bool:
    """Mismo helper que `jarvis.tools.cancel_timer._contains_any_word` (duplicado a propósito,
    ver docstring de ese módulo)."""
    normalized = text.strip().lower()
    words = [word.strip(".,!¡¿?;:") for word in normalized.split()]
    return any(word in trigger_words for word in words if word)


def _find_best_text_match(query: str, reminders: list[Reminder]) -> Reminder | None:
    """Encontrar el recordatorio cuyo `text` mejor matchea `query` — mismo algoritmo en tres
    pasos que `jarvis.tools.cancel_timer._find_best_label_match`/`jarvis.tools.close_app.
    _find_best_match` (exacto, después substring por cobertura relativa, después `difflib`)."""
    if not reminders:
        return None

    normalized_query = _normalize(query)
    reminders_by_text: dict[str, Reminder] = {}
    for reminder in reminders:
        key = _normalize(reminder.text)
        # Ante textos duplicados, se conserva el primero (el próximo a vencer — `list_reminders`
        # ya devuelve por `due_at` ascendente).
        reminders_by_text.setdefault(key, reminder)

    if normalized_query in reminders_by_text:
        return reminders_by_text[normalized_query]

    substring_matches = [
        text
        for text in reminders_by_text
        if normalized_query in text or text in normalized_query
    ]
    if substring_matches:

        def _coverage(text: str) -> float:
            shorter = min(len(text), len(normalized_query))
            longer = max(len(text), len(normalized_query))
            return shorter / longer if longer else 0.0

        best_text = max(
            substring_matches, key=lambda text: (_coverage(text), -len(text))
        )
        return reminders_by_text[best_text]

    close = difflib.get_close_matches(
        normalized_query, reminders_by_text.keys(), n=1, cutoff=_FUZZY_MATCH_CUTOFF
    )
    if not close:
        return None
    return reminders_by_text[close[0]]


class CancelReminderTool(Tool):
    """Cancela UN recordatorio pendiente puntual, resolviendo el pedido en lenguaje natural
    contra los recordatorios actualmente guardados (`jarvis.memory.store.list_reminders`) — nunca
    "todos", ver `CancelAllRemindersTool` para eso (ADR-0006, docstring del módulo)."""

    name = "cancel_reminder"
    description = (
        "Cancela un recordatorio puntual que ya fue programado y todavía no se cumplió (ej. "
        "'cancelá el recordatorio de llamar a mamá', 'cancelá mi último recordatorio'). Usar "
        "cuando el usuario pide cancelar, borrar o sacar UN recordatorio pendiente — no "
        "confundir con set_reminder (programar uno nuevo), con cancel_timer (para timers en vez "
        "de recordatorios), ni con cancel_all_reminders (cuando el usuario pide cancelar TODOS "
        "los recordatorios pendientes de una vez)."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "A qué recordatorio se refiere el usuario: el texto que le dio al "
                    "programarlo (ej. 'llamar a mamá'), o la palabra 'último' si pidió cancelar "
                    "el más reciente sin dar más detalle."
                ),
            }
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def __init__(self, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path

    async def execute(self, **kwargs: Any) -> str:
        target = kwargs.get("target")
        if not isinstance(target, str) or not target.strip():
            return _MISSING_TARGET_MESSAGE

        # `list_reminders`/`delete_reminder` son I/O de archivo bloqueante (sqlite3, stdlib) —
        # `asyncio.to_thread`, mismo patrón que `ReminderTool.execute`. Ventana de carrera
        # aceptada y documentada (mismo espíritu que `CloseAppTool.describe`): entre este
        # `list_reminders` y el `delete_reminder` de más abajo, `TimerScheduler` (en su propio
        # thread de fondo, ver `jarvis.audio.timer_scheduler`) podría anunciar y borrar el mismo
        # recordatorio si venció justo en el medio — `delete_reminder` es idempotente (no lanza
        # sobre un id ya borrado), así que el peor caso es confirmar la cancelación de un
        # recordatorio que en realidad ya se anunció un instante antes, no un error ni un estado
        # inconsistente.
        pending = await asyncio.to_thread(list_reminders, db_path=self._db_path)
        if not pending:
            return _NO_PENDING_REMINDERS_MESSAGE

        if _contains_any_word(target, _MOST_RECENT_WORDS):
            most_recent = max(pending, key=lambda reminder: reminder.id)
            await asyncio.to_thread(
                delete_reminder, most_recent.id, db_path=self._db_path
            )
            return f"Listo, cancelé el recordatorio: {most_recent.text}."

        match = _find_best_text_match(target, pending)
        if match is None:
            return f"No encontré ningún recordatorio pendiente que coincida con '{target}'."
        await asyncio.to_thread(delete_reminder, match.id, db_path=self._db_path)
        return f"Listo, cancelé el recordatorio: {match.text}."


class CancelAllRemindersTool(Tool):
    """Cancela TODOS los recordatorios pendientes de una sola vez (ADR-0006) — CONFIRM: requiere
    confirmación explícita por voz antes de ejecutarse, `describe()` nombra cuántos recordatorios
    hay pendientes ahora mismo (ver docstring del módulo)."""

    name = "cancel_all_reminders"
    description = (
        "Cancela TODOS los recordatorios pendientes de una sola vez (ej. 'cancelá todos los "
        "recordatorios', 'borrá todos mis recordatorios'). Usar únicamente cuando el usuario "
        "pide explícitamente cancelar TODOS a la vez, nunca para cancelar uno puntual (para eso "
        "está cancel_reminder). Requiere confirmación explícita del usuario antes de ejecutarse."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    # Pedido explícito del usuario: sacar todos los confirmadores salvo SystemPowerTool (ver
    # close_app.py para el motivo completo — confirmación hablada rota en la práctica por
    # audio de juego durante la ventana de escucha). Downgrade deliberado a SAFE.
    risk = RiskLevel.SAFE

    def __init__(self, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path

    def describe(self, kwargs: dict[str, Any]) -> str:
        return _describe_cancel_all_reminders(self._db_path)

    async def execute(self, **kwargs: Any) -> str:
        return await _cancel_all_pending_reminders(self._db_path)
