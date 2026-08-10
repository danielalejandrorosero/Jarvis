"""Tests para `CancelSystemPowerTool` (`jarvis.tools.cancel_system_power`).

`run_power_command` (importado desde `jarvis.tools.system_power`) se reemplaza por un stub a
nivel de módulo — mismo patrón que `tests/tools/test_system_power.py` — ningún test invoca
`shutdown.exe` de verdad.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.tools import cancel_system_power as cancel_system_power_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.cancel_system_power import CancelSystemPowerTool


def _install_fake_power_command(
    monkeypatch: pytest.MonkeyPatch, *, outcome: tuple[bool, str] = (True, "")
) -> list[list[str]]:
    calls: list[list[str]] = []

    def _fake_run_power_command(command: list[str]) -> tuple[bool, str]:
        calls.append(command)
        return outcome

    monkeypatch.setattr(
        cancel_system_power_module, "run_power_command", _fake_run_power_command
    )
    return calls


def test_cancel_system_power_tool_declares_safe_risk() -> None:
    """Cancelar es puramente protector — nunca deja el sistema en peor estado que antes de la
    confirmación original de `system_power` — ver el docstring del módulo para el razonamiento
    completo (mismo criterio que `CancelTimerTool`/`CancelReminderTool`/`CancelQueueTool`)."""
    assert CancelSystemPowerTool.risk is RiskLevel.SAFE


def test_execute_runs_shutdown_a(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_power_command(monkeypatch)
    tool = CancelSystemPowerTool()

    result = asyncio.run(tool.execute())

    assert calls == [["shutdown", "/a"]]
    assert "Listo" in result


def test_execute_reports_when_nothing_was_pending_to_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`shutdown /a` falla con su propio mensaje cuando no hay ningún apagado/reinicio
    programado — caso esperado, no debería sonar como un error grave."""
    _install_fake_power_command(
        monkeypatch,
        outcome=(False, "No se ha programado ningún apagado del sistema."),
    )
    tool = CancelSystemPowerTool()

    result = asyncio.run(tool.execute())

    assert "No pude cancelar" in result
    assert "No se ha programado ningún apagado del sistema." in result
    assert "Listo" not in result
