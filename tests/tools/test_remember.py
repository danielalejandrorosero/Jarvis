"""Tests para `RememberTool` (`jarvis.tools.remember`).

Sin red ni mocks de `sqlite3`: local, stdlib, rápido — un archivo SQLite real en `tmp_path`, mismo
enfoque que `tests/memory/test_store.py`. `RememberTool` toma `db_path` en el constructor (ver
docstring de `__init__`), así que cada test instancia el tool apuntando a una DB de `tmp_path` sin
tocar el `data/jarvis.db` real del repo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jarvis.memory.store import list_facts
from jarvis.tools.base import RiskLevel
from jarvis.tools.remember import RememberTool


def test_remember_tool_declares_safe_risk() -> None:
    """Escribe en el store SQLite propio de JARVIS, no en un archivo/proceso real del usuario —
    ver el comentario de `risk` en `jarvis.tools.remember` para el razonamiento completo."""
    assert RememberTool.risk is RiskLevel.SAFE


def test_execute_saves_fact_and_confirms(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    result = asyncio.run(tool.execute(content="el usuario prefiere respuestas cortas"))

    assert "el usuario prefiere respuestas cortas" in result
    assert list_facts(db_path=db_path) == ["el usuario prefiere respuestas cortas"]


def test_execute_strips_whitespace_before_saving(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    asyncio.run(tool.execute(content="  hecho con espacios  "))

    assert list_facts(db_path=db_path) == ["hecho con espacios"]


def test_execute_rejects_missing_content_without_touching_the_store(
    tmp_path: Path,
) -> None:
    """Caso rechazado: sin `content` (o vacío), `execute()` no escribe nada en el store y
    devuelve un mensaje de error claro."""
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    result = asyncio.run(tool.execute())

    assert "No se especificó qué recordar" in result
    assert not db_path.exists()


def test_execute_rejects_blank_content_without_touching_the_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    result = asyncio.run(tool.execute(content="   "))

    assert "No se especificó qué recordar" in result
    assert not db_path.exists()
