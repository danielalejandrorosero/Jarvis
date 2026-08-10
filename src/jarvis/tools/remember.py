"""Tercer tool de JARVIS: guardar un hecho sobre el usuario en memoria persistente (ADR-0004,
"Persistencia: SQLite"; `jarvis.memory.store`).

Recall ambiental por defecto, sin tool de lectura para esto: `jarvis.audio.pipeline` inyecta los
últimos `DEFAULT_LIST_LIMIT` hechos guardados en el system prompt de cada turno (ver
`_build_system_prompt`), así que el LLM "sabe" lo reciente sin pedirlo explícitamente. Lo que este
módulo sí agrega ahora es el embedding de cada hecho al guardarlo (`jarvis.memory.embeddings`,
API de OpenAI) — necesario para que `jarvis.tools.recall_memory.RecallMemoryTool` pueda encontrar
por significado algo dicho hace tiempo, que ya se cayó de esa ventana de recencia.

Hallazgo MEDIUM de `security-reviewer`: el embedding se calculaba sobre `content` ANTES de
truncarlo — `save_fact` (`jarvis.memory.store`) trunca a `MAX_CONTENT_LENGTH` recién al guardar,
así que un `content` de largo arbitrario (ej. el LLM "recordando" un fragmento entero de una
página web) se mandaba entero, sin capear, a la API de embeddings de OpenAI, y encima el
embedding resultante representaba un texto más largo que el que realmente queda guardado y se
reinyecta — un desajuste semántico, no solo de costo. `execute()` ahora trunca a
`MAX_CONTENT_LENGTH` antes de llamar a `embed_text`, así el embedding representa exactamente el
mismo texto que `save_fact` termina persistiendo (la truncación de `save_fact` sobre ese mismo
texto ya truncado es, en la práctica, un no-op — se deja igual, en vez de asumir que este tool
truncó "bien", porque `jarvis.memory.store` es la única fuente de verdad sobre ese límite).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar

from openai import OpenAI

from jarvis.memory.embeddings import embed_text
from jarvis.memory.store import DEFAULT_DB_PATH, MAX_CONTENT_LENGTH, save_fact
from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        *,
        db_path: str | Path = DEFAULT_DB_PATH,
        embedding_client: OpenAI | None = None,
    ) -> None:
        # Configurable (no hardcodeado a `DEFAULT_DB_PATH` dentro de `execute`) por el mismo
        # motivo que `jarvis.memory.store` expone `db_path` como parámetro: tests aislados
        # (`tmp_path`) sin tocar el `data/jarvis.db` real, y coherencia con
        # `dispatch_turn(memory_db_path=...)` (`jarvis.audio.pipeline`), que lee de la misma DB
        # a la que este tool escribe.
        self._db_path = db_path
        # `embedding_client`, si se pasa, es el mismo `OpenAI()` que ya usa el STT (ver
        # `jarvis.audio.pipeline.run` — `load_stt_client()`) — no una instancia nueva. `None` por
        # defecto (no `OpenAI()` construido acá): construirlo sin API key configurada lanza en el
        # constructor mismo, y los tests existentes de este tool no necesitan un cliente real para
        # cubrir el guardado en sí.
        self._embedding_client = embedding_client

    async def execute(self, **kwargs: Any) -> str:
        content = kwargs.get("content")
        if not isinstance(content, str) or not content.strip():
            return "No se especificó qué recordar."
        # Truncado ANTES de embeber, no después (hallazgo MEDIUM, ver docstring del módulo). Corte
        # simple por índice, sin el indicador "…" que sí usa `jarvis.memory.store._truncate`: así
        # `len(stripped) <= MAX_CONTENT_LENGTH` siempre se cumple acá, y el truncado que hace
        # `save_fact` sobre este mismo texto es un no-op garantizado — el embedding y el `content`
        # finalmente persistido quedan bit a bit idénticos, sin el riesgo de una segunda pasada de
        # truncado (que podría, en el caso límite de un texto ya en el límite, cortar de nuevo y
        # dejar embedding/contenido desalineados otra vez). Se pierde el indicador visual de corte
        # en el caso raro (>500 caracteres) — aceptable: ese indicador es cosmético, la consistencia
        # embedding↔contenido es lo que motivó el hallazgo.
        stripped = content.strip()[:MAX_CONTENT_LENGTH]
        embedding: list[float] | None = None
        if self._embedding_client is not None:
            try:
                embedding = await asyncio.to_thread(
                    embed_text, stripped, client=self._embedding_client
                )
            except Exception:
                # Frontera de recuperación explícita (`.claude/rules/python.md` lo permite acá):
                # una falla de red/API de embeddings no puede impedir guardar el hecho en sí, que
                # es la funcionalidad central de este tool — la búsqueda semántica
                # (`RecallMemoryTool`) es una mejora sobre eso, no un requisito para que funcione.
                logger.exception(
                    "No se pudo calcular el embedding de %r; se guarda sin él.",
                    stripped,
                )
        # `save_fact` es I/O de archivo bloqueante (sqlite3, stdlib, sin variante async) —
        # `asyncio.to_thread` evita bloquear el loop de asyncio que orquesta el tool-call
        # (`.claude/rules/python.md`). `stripped` ya está acotado a `MAX_CONTENT_LENGTH` arriba,
        # así que el truncado interno de `save_fact` sobre este mismo valor es un no-op.
        await asyncio.to_thread(
            save_fact, stripped, db_path=self._db_path, embedding=embedding
        )
        return f"Listo, lo voy a recordar: {stripped}"
