"""Tests para `ReminderTool` (`jarvis.tools.reminder`).

Mismo enfoque que `tests/tools/test_remember.py`: sin red ni mocks de `sqlite3`, un archivo
SQLite real en `tmp_path`. `ReminderTool` calcula `due_at` a partir de `seconds_from_now` y la
hora actual real (`datetime.now(UTC)`) — los tests verifican la relación (el `due_at` guardado
cae dentro de una ventana razonable alrededor de "ahora + seconds_from_now"), no un timestamp
exacto, para no acoplar el test al instante preciso en que corre.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis.memory.store import list_reminders
from jarvis.tools.base import RiskLevel
from jarvis.tools.reminder import MAX_SECONDS_FROM_NOW, ReminderTool


def test_reminder_tool_declares_safe_risk() -> None:
    """Escribe en el store SQLite propio de JARVIS, no en un archivo/proceso real del usuario —
    mismo razonamiento que `RememberTool` (`jarvis.tools.remember`)."""
    assert ReminderTool.risk is RiskLevel.SAFE


def test_execute_saves_reminder_and_confirms(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)
    before = datetime.now(UTC)

    result = asyncio.run(tool.execute(text="llamar a mamá", seconds_from_now=300))

    after = datetime.now(UTC)
    assert "llamar a mamá" in result
    (reminder,) = list_reminders(db_path=db_path)
    assert reminder.text == "llamar a mamá"
    due_at = datetime.fromisoformat(reminder.due_at)
    assert before + timedelta(seconds=300) <= due_at <= after + timedelta(seconds=300)


def test_execute_strips_whitespace_from_text_before_saving(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    asyncio.run(tool.execute(text="  sacar la comida  ", seconds_from_now=60))

    (reminder,) = list_reminders(db_path=db_path)
    assert reminder.text == "sacar la comida"


# --- casos rechazados: nunca llegan a `save_reminder` ----------------------------------------


def test_execute_rejects_missing_text_without_touching_the_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(seconds_from_now=60))

    assert "No se especificó qué recordar" in result
    assert not db_path.exists()


def test_execute_rejects_blank_text_without_touching_the_store(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(text="   ", seconds_from_now=60))

    assert "No se especificó qué recordar" in result
    assert not db_path.exists()


def test_execute_rejects_missing_seconds_from_now(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(text="llamar a mamá"))

    assert "No se especificó un momento válido" in result
    assert not db_path.exists()


def test_execute_rejects_zero_seconds_from_now(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(text="llamar a mamá", seconds_from_now=0))

    assert "No se especificó un momento válido" in result
    assert not db_path.exists()


def test_execute_rejects_negative_seconds_from_now(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(text="llamar a mamá", seconds_from_now=-5))

    assert "No se especificó un momento válido" in result
    assert not db_path.exists()


def test_execute_rejects_non_numeric_seconds_from_now(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(
        tool.execute(text="llamar a mamá", seconds_from_now="en un rato")
    )

    assert "No se especificó un momento válido" in result
    assert not db_path.exists()


def test_execute_rejects_boolean_seconds_from_now_despite_being_an_int_subclass(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(text="llamar a mamá", seconds_from_now=True))

    assert "No se especificó un momento válido" in result
    assert not db_path.exists()


def test_execute_rejects_seconds_from_now_beyond_max(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = ReminderTool(db_path=db_path)

    result = asyncio.run(
        tool.execute(text="llamar a mamá", seconds_from_now=MAX_SECONDS_FROM_NOW + 1)
    )

    assert "demasiado lejos" in result
    assert not db_path.exists()
