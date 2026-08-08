"""Tool para programar un recordatorio que JARVIS anuncia por voz más adelante (ADR-0005,
extendida con anuncio proactivo — ver `jarvis.audio.timer_scheduler`).

Mismo espíritu que `jarvis.tools.timer.TimerTool` (confirma que quedó programado, el anuncio real
lo hace `TimerScheduler` de forma independiente a este turno) pero con una diferencia de diseño
central: **quién calcula el tiempo**. `TimerTool` recibe segundos directos — trivial, sin
ambigüedad. Un recordatorio en cambio parte de lenguaje natural con fecha/hora ("a las 5",
"mañana a la mañana", "en 20 minutos") que hay que convertir a un instante concreto.

Dos formas de resolver esto: (a) que este tool reciba el string crudo y lo parsee él mismo
("a las 5"), o (b) que el LLM haga la conversión y este tool reciba un número ya calculado. Se
eligió (b) — el LLM ya es bueno en aritmética de fechas cuando se le dice la hora actual (por eso
`jarvis.audio.pipeline._build_system_prompt` inyecta la hora local de cada turno), y un parser de
fecha en lenguaje natural en español, robusto contra la variedad real de cómo la gente pide esto
("a las 5", "en veinte", "mañana a la mañana", "el viernes que viene") es un problema mucho más
grande y frágil que aritmética de segundos — más ambiguo de testear, más fácil de fallar en
silencio con una interpretación equivocada de una hora AM/PM. Pedirle al LLM que devuelva
`seconds_from_now` (un entero, sin ambigüedad de zona horaria ni de formato) mueve el trabajo
difícil (NLU de fecha/hora) a donde ya se resuelve bien (el LLM en el prompt), y deja acá solo
aritmética de reloj (`datetime.now(UTC) + timedelta(seconds=...)`), que es trivial de validar y
testear.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from jarvis.memory.store import DEFAULT_DB_PATH, save_reminder
from jarvis.tools.base import RiskLevel, Tool

# Tope duro de cuánto en el futuro puede estar un recordatorio: 1 año. No es una expectativa real
# de uso (nadie pide "recordame algo en 300 días" por voz en la práctica) — es una red de
# seguridad contra una alucinación del LLM en la conversión de fecha (ej. interpretar mal un año)
# que dejaría un recordatorio "perdido" tan lejos en el futuro que en la práctica nunca se anuncia
# ni el usuario recuerda haberlo pedido.
MAX_SECONDS_FROM_NOW = 365 * 24 * 60 * 60


class ReminderTool(Tool):
    """Programa un recordatorio que JARVIS anuncia por voz cuando llega el momento."""

    name = "set_reminder"
    description = (
        "Programa un recordatorio para un momento futuro (ej. 'recordame llamar a mamá a las "
        "5', 'recordame en 20 minutos sacar la comida del horno', 'recordame mañana a la mañana "
        "pagar la factura'). JARVIS lo anuncia en voz alta apenas llega el momento, incluso si "
        "el usuario no está hablando con vos en ese momento. Para una cuenta regresiva simple "
        "relativa a ahora sin un momento específico en mente (ej. 'timer de 10 minutos'), usar "
        "set_timer en cambio."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Qué recordarle al usuario, en sus palabras, corto y en tercera persona o "
                    "imperativo (ej. 'llamar a mamá', 'sacar la comida del horno')."
                ),
            },
            "seconds_from_now": {
                "type": "integer",
                "description": (
                    "En cuántos segundos, contados desde AHORA MISMO, corresponde avisar. "
                    "Convertí vos la fecha/hora que pidió el usuario ('a las 5', 'mañana a la "
                    "mañana', 'en 20 minutos') a esta cantidad de segundos usando la hora "
                    "actual que te dieron en tus instrucciones — nunca dejes ese cálculo para "
                    "después ni le pidas al usuario que lo haga él."
                ),
            },
        },
        "required": ["text", "seconds_from_now"],
        "additionalProperties": False,
    }
    # SAFE por el mismo razonamiento que `RememberTool`: lo que muta es el store SQLite interno
    # de JARVIS (`jarvis.memory.store`), no un archivo/proceso/configuración real del usuario;
    # reversible (borrar la fila) y de impacto acotado a un anuncio hablado propio de JARVIS.
    risk = RiskLevel.SAFE

    def __init__(self, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path

    async def execute(self, **kwargs: Any) -> str:
        text = kwargs.get("text")
        if not isinstance(text, str) or not text.strip():
            return "No se especificó qué recordar."
        stripped_text = text.strip()

        seconds_from_now = kwargs.get("seconds_from_now")
        # `bool` es subclase de `int` — mismo chequeo explícito que `TimerTool.execute`.
        if (
            not isinstance(seconds_from_now, int)
            or isinstance(seconds_from_now, bool)
            or seconds_from_now <= 0
        ):
            return "No se especificó un momento válido para el recordatorio."
        if seconds_from_now > MAX_SECONDS_FROM_NOW:
            return "Ese recordatorio queda demasiado lejos en el futuro; pedime algo más cercano."

        due_at = (datetime.now(UTC) + timedelta(seconds=seconds_from_now)).isoformat()
        # `save_reminder` es I/O de archivo bloqueante (sqlite3, stdlib) — `asyncio.to_thread`,
        # mismo patrón que `RememberTool.execute`.
        await asyncio.to_thread(
            save_reminder, stripped_text, due_at, db_path=self._db_path
        )
        return f"Listo, te voy a recordar: {stripped_text}."
