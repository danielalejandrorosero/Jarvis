"""Tercer tool de JARVIS: guardar un hecho sobre el usuario en memoria persistente (ADR-0004,
"Persistencia: SQLite"; `jarvis.memory.store`).

Sin lectura explícita como tool: el recall es ambiental, no a pedido — `jarvis.audio.pipeline`
inyecta los hechos guardados en el system prompt de cada turno (ver `_build_system_prompt`), así
que no hace falta un `recall_facts`/`get_memories` tool para que el LLM "sepa" lo que ya
aprendió. Este módulo solo cubre la escritura.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from jarvis.memory.store import DEFAULT_DB_PATH, save_fact
from jarvis.tools.base import RiskLevel, Tool


class RememberTool(Tool):
    """Guarda un hecho o preferencia sobre el usuario para recordarlo en conversaciones
    futuras."""

    name = "remember_fact"
    description = (
        "Guarda un hecho, preferencia o hábito del usuario para recordarlo en conversaciones "
        "futuras (ej. 'el usuario prefiere respuestas cortas', 'suele pedir abrir tal app a la "
        "tarde'). Usar cuando el usuario pide explícitamente que recuerdes algo, o cuando dice "
        "algo claramente relevante para el futuro."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "El hecho a recordar, como frase corta en español, en tercera persona "
                    "(ej. 'el usuario prefiere que las respuestas sean cortas')."
                ),
            }
        },
        "required": ["content"],
        "additionalProperties": False,
    }
    # SAFE, no CONFIRM: a diferencia de `WeatherTool`/`SearchTool` este tool sí escribe (muta
    # estado), así que la clasificación no es tan obvia a primera vista — el juicio concreto es
    # que lo que muta es el propio store SQLite interno de JARVIS (`jarvis.memory.store`,
    # `data/jarvis.db`), no un archivo real del usuario, software instalado, ni un proceso del
    # sistema (los ejemplos de CONFIRM en `.claude/rules/security.md`). Es reversible (borrar la
    # fila o el archivo), de impacto acotado a lo que el propio LLM puede leer de vuelta en el
    # próximo turno, y no tiene efecto visible fuera de JARVIS — encaja en SAFE, no en
    # "modificar configuración"/CONFIRM.
    risk = RiskLevel.SAFE

    def __init__(self, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        # Configurable (no hardcodeado a `DEFAULT_DB_PATH` dentro de `execute`) por el mismo
        # motivo que `jarvis.memory.store` expone `db_path` como parámetro: tests aislados
        # (`tmp_path`) sin tocar el `data/jarvis.db` real, y coherencia con
        # `dispatch_turn(memory_db_path=...)` (`jarvis.audio.pipeline`), que lee de la misma DB
        # a la que este tool escribe.
        self._db_path = db_path

    async def execute(self, **kwargs: Any) -> str:
        content = kwargs.get("content")
        if not isinstance(content, str) or not content.strip():
            return "No se especificó qué recordar."
        stripped = content.strip()
        # `save_fact` es I/O de archivo bloqueante (sqlite3, stdlib, sin variante async) —
        # `asyncio.to_thread` evita bloquear el loop de asyncio que orquesta el tool-call
        # (`.claude/rules/python.md`).
        await asyncio.to_thread(save_fact, stripped, db_path=self._db_path)
        return f"Listo, lo voy a recordar: {stripped}"
