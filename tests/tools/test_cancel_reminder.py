"""Tests para `CancelReminderTool`/`CancelAllRemindersTool` (`jarvis.tools.cancel_reminder`).

Mismo enfoque que `tests/tools/test_reminder.py`: sin red ni mocks de `sqlite3`, un archivo
SQLite real en `tmp_path` (vía `jarvis.memory.store.save_reminder` para poblar el estado inicial).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis.memory.store import list_reminders, save_reminder
from jarvis.tools.base import RiskLevel
from jarvis.tools.cancel_reminder import CancelAllRemindersTool, CancelReminderTool

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _due_at(offset_seconds: int) -> str:
    return (_NOW + timedelta(seconds=offset_seconds)).isoformat()


def test_cancel_reminder_tool_declares_safe_risk() -> None:
    """Mismo razonamiento que `ReminderTool` (SAFE) — ver docstring del módulo."""
    assert CancelReminderTool.risk is RiskLevel.SAFE


def test_execute_cancels_by_exact_text_match(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("llamar a mamá", _due_at(300), db_path=db_path)
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(target="llamar a mamá"))

    assert "cancelé el recordatorio: llamar a mamá" in result
    assert list_reminders(db_path=db_path) == []


def test_execute_cancels_by_fuzzy_text_match(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("sacar la comida del horno", _due_at(300), db_path=db_path)
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(target="la comida"))

    assert "cancelé el recordatorio: sacar la comida del horno" in result
    assert list_reminders(db_path=db_path) == []


def test_execute_cancels_most_recent_when_target_is_ultimo(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("uno", _due_at(300), db_path=db_path)
    save_reminder("dos", _due_at(600), db_path=db_path)
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(target="mi último recordatorio"))

    assert "cancelé el recordatorio: dos" in result
    remaining = list_reminders(db_path=db_path)
    assert [reminder.text for reminder in remaining] == ["uno"]


def test_execute_does_not_special_case_todos_anymore(tmp_path: Path) -> None:
    """ADR-0006 (aplicado hoy, ver docstring del módulo): `CancelReminderTool` perdió la rama
    "todos" — ahora "todos" se resuelve como cualquier otro texto libre contra los recordatorios
    pendientes (fuzzy-match), sin ningún efecto especial de vaciar todo lo pendiente. Cancelar
    TODO de una vez ahora es exclusivamente `CancelAllRemindersTool` (CONFIRM)."""
    # Textos que no colisionan por substring con "todos" (a diferencia de, ej., "dos") — el
    # objetivo de este test es que "todos" no dispare ningún manejo especial, no ejercitar el
    # fuzzy-match en sí (ya cubierto por `test_execute_cancels_by_fuzzy_text_match`).
    db_path = tmp_path / "jarvis.db"
    save_reminder("llamar a mamá", _due_at(300), db_path=db_path)
    save_reminder("sacar la comida del horno", _due_at(600), db_path=db_path)
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(target="todos"))

    assert "No encontré ningún recordatorio pendiente" in result
    assert len(list_reminders(db_path=db_path)) == 2


def test_execute_reports_a_clear_message_when_nothing_is_pending(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(target="lo que sea"))

    assert "No hay ningún recordatorio pendiente" in result


def test_execute_reports_a_clear_message_when_nothing_matches(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("llamar a mamá", _due_at(300), db_path=db_path)
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(target="algo completamente distinto"))

    assert "No encontré ningún recordatorio pendiente" in result
    assert len(list_reminders(db_path=db_path)) == 1


def test_execute_rejects_missing_target_without_touching_the_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("llamar a mamá", _due_at(300), db_path=db_path)
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute())

    assert "No se especificó qué recordatorio cancelar" in result
    assert len(list_reminders(db_path=db_path)) == 1


def test_execute_rejects_blank_target_without_touching_the_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("llamar a mamá", _due_at(300), db_path=db_path)
    tool = CancelReminderTool(db_path=db_path)

    result = asyncio.run(tool.execute(target="   "))

    assert "No se especificó qué recordatorio cancelar" in result
    assert len(list_reminders(db_path=db_path)) == 1


def test_cancel_all_reminders_tool_declares_confirm_risk() -> None:
    """ADR-0006 y `.claude/rules/security.md` (excepción SAFE del store propio de JARVIS no
    cubre borrado masivo) — ver docstring del módulo."""
    assert CancelAllRemindersTool.risk is RiskLevel.CONFIRM


def test_cancel_all_reminders_execute_cancels_every_pending_reminder(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("uno", _due_at(300), db_path=db_path)
    save_reminder("dos", _due_at(600), db_path=db_path)
    tool = CancelAllRemindersTool(db_path=db_path)

    result = asyncio.run(tool.execute())

    assert "cancelé 2 recordatorios" in result
    assert list_reminders(db_path=db_path) == []


def test_cancel_all_reminders_execute_uses_singular_phrasing_for_one_reminder(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("uno", _due_at(300), db_path=db_path)
    tool = CancelAllRemindersTool(db_path=db_path)

    result = asyncio.run(tool.execute())

    assert "cancelé 1 recordatorio" in result
    assert list_reminders(db_path=db_path) == []


def test_cancel_all_reminders_execute_reports_a_clear_message_when_nothing_is_pending(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = CancelAllRemindersTool(db_path=db_path)

    result = asyncio.run(tool.execute())

    assert "No hay ningún recordatorio pendiente" in result


def test_cancel_all_reminders_describe_names_the_pending_count(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("uno", _due_at(300), db_path=db_path)
    save_reminder("dos", _due_at(600), db_path=db_path)
    tool = CancelAllRemindersTool(db_path=db_path)

    description = tool.describe({})

    assert "TODOS los 2 recordatorios pendientes" in description


def test_cancel_all_reminders_describe_uses_singular_phrasing_for_one_reminder(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    save_reminder("uno", _due_at(300), db_path=db_path)
    tool = CancelAllRemindersTool(db_path=db_path)

    description = tool.describe({})

    assert "TODOS los 1 recordatorio pendientes" in description


def test_cancel_all_reminders_describe_reports_zero_pending(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = CancelAllRemindersTool(db_path=db_path)

    description = tool.describe({})

    assert "no hay ninguno pendiente ahora mismo" in description
