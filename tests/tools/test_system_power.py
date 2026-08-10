"""Tests para `SystemPowerTool` (`jarvis.tools.system_power`, ADR-0005).

`run_power_command`/`_suspend_system` se reemplazan por stubs monkeypatcheados a nivel de módulo
(mismo patrón que `_terminate_process` en `test_close_app.py`/`_press_media_key` en
`test_media_control.py`) — ningún test de este archivo apaga, reinicia ni suspende la máquina
real donde corre la suite.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import pytest

from jarvis.tools import system_power as system_power_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.system_power import SystemPowerTool


def _install_fake_power_command(
    monkeypatch: pytest.MonkeyPatch, *, outcome: tuple[bool, str] = (True, "")
) -> list[list[str]]:
    calls: list[list[str]] = []

    def _fake_run_power_command(command: list[str]) -> tuple[bool, str]:
        calls.append(command)
        return outcome

    monkeypatch.setattr(
        system_power_module, "run_power_command", _fake_run_power_command
    )
    return calls


def _install_fake_suspend(
    monkeypatch: pytest.MonkeyPatch, *, outcome: bool = True
) -> list[bool]:
    calls: list[bool] = []

    def _fake_suspend_system() -> bool:
        calls.append(True)
        return outcome

    monkeypatch.setattr(system_power_module, "_suspend_system", _fake_suspend_system)
    return calls


def test_system_power_tool_declares_confirm_risk() -> None:
    """Reversible (prender/despertar de nuevo) y de impacto acotado a esta única máquina — mismo
    techo que 'terminar procesos' (`close_app`), ejemplo explícito de CONFIRM en
    `.claude/rules/security.md`. Nunca SAFE (muta estado real de forma visible), nunca DANGEROUS
    (`PolicyEngine` no ejecuta DANGEROUS bajo ningún flujo, lo que dejaría el comando inutilizable
    por voz sin ganar seguridad real — ver docstring del módulo)."""
    assert SystemPowerTool.risk is RiskLevel.CONFIRM


@pytest.mark.parametrize(
    ("action", "phrase"),
    [
        ("shutdown", "apagar la computadora"),
        ("restart", "reiniciar la computadora"),
        ("sleep", "suspender la computadora"),
    ],
)
def test_describe_returns_natural_phrase_per_action(action: str, phrase: str) -> None:
    tool = SystemPowerTool()

    assert tool.describe({"action": action}) == phrase


def test_describe_falls_back_to_default_for_invalid_action() -> None:
    """Sin una acción reconocida, `describe()` no inventa una frase — cae al default genérico de
    `Tool.describe()` (nombre + kwargs), igual que cualquier otro tool sin override aplicable."""
    tool = SystemPowerTool()

    assert tool.describe({"action": "hibernate"}) == "system_power (action='hibernate')"


@pytest.mark.parametrize(
    ("action", "expected_command"),
    [
        ("shutdown", ["shutdown", "/s", "/t", "15"]),
        ("restart", ["shutdown", "/r", "/t", "15"]),
    ],
)
def test_execute_runs_expected_shutdown_command_per_action(
    monkeypatch: pytest.MonkeyPatch, action: str, expected_command: list[str]
) -> None:
    calls = _install_fake_power_command(monkeypatch)
    suspend_calls = _install_fake_suspend(monkeypatch)
    tool = SystemPowerTool()

    result = asyncio.run(tool.execute(action=action))

    assert calls == [expected_command]
    assert suspend_calls == []
    assert "Listo" in result


def test_execute_sleep_calls_suspend_system_not_shutdown_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    power_calls = _install_fake_power_command(monkeypatch)
    suspend_calls = _install_fake_suspend(monkeypatch)
    tool = SystemPowerTool()

    result = asyncio.run(tool.execute(action="sleep"))

    assert suspend_calls == [True]
    assert power_calls == []
    assert "Listo" in result
    assert "suspend" in result.lower()


def test_execute_rejects_unknown_action_without_touching_os_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso rechazado: una acción fuera del enum no dispara ningún comando de apagado/reinicio ni
    ninguna suspensión."""
    power_calls = _install_fake_power_command(monkeypatch)
    suspend_calls = _install_fake_suspend(monkeypatch)
    tool = SystemPowerTool()

    result = asyncio.run(tool.execute(action="hibernate"))

    assert "acción de energía válida" in result
    assert power_calls == []
    assert suspend_calls == []


def test_execute_rejects_missing_action(monkeypatch: pytest.MonkeyPatch) -> None:
    power_calls = _install_fake_power_command(monkeypatch)
    suspend_calls = _install_fake_suspend(monkeypatch)
    tool = SystemPowerTool()

    result = asyncio.run(tool.execute())

    assert "acción de energía válida" in result
    assert power_calls == []
    assert suspend_calls == []


@pytest.mark.parametrize("action", ["shutdown", "restart"])
def test_execute_degrades_on_shutdown_command_failure(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """Si `shutdown.exe` falla (código de salida distinto de 0, o `OSError` al lanzarlo),
    `execute()` no afirma falsamente que apagó/reinició la máquina."""
    _install_fake_power_command(monkeypatch, outcome=(False, "acceso denegado"))
    tool = SystemPowerTool()

    result = asyncio.run(tool.execute(action=action))

    assert "No pude" in result
    assert "acceso denegado" in result
    assert "Listo" not in result


def test_execute_degrades_on_suspend_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_suspend_system` devolviendo `False` (falla reportada por Win32, sin excepción) degrada a
    un mensaje de error claro en vez de afirmar éxito."""
    _install_fake_suspend(monkeypatch, outcome=False)
    tool = SystemPowerTool()

    result = asyncio.run(tool.execute(action="sleep"))

    assert "No pude" in result
    assert "Listo" not in result


def test_execute_degrades_on_suspend_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si la llamada WinAPI lanza (`OSError`, ej. `powrprof.dll` inaccesible), `execute()` nunca
    deja que la excepción se propague — degrada a un mensaje de error claro."""

    def _fake_suspend_system() -> bool:
        raise OSError("boom")

    monkeypatch.setattr(system_power_module, "_suspend_system", _fake_suspend_system)
    tool = SystemPowerTool()

    result = asyncio.run(tool.execute(action="sleep"))

    assert "No pude" in result
    assert "OSError" in result


# --- ventana de consola oculta (bug real, en vivo) ----------------------------------------------
# `run_power_command` invoca `shutdown.exe` — sin `creationflags=CREATE_NO_WINDOW`, Windows abre
# una ventana de consola visible cada vez (JARVIS corre vía `pythonw.exe`, sin consola propia) —
# mismo bug real encontrado en vivo con el mismo patrón en `jarvis.league.lcu_monitor`/
# `jarvis.tools.close_app`/`jarvis.tools.system_info`.


def test_run_power_command_hides_console_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _FakeCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(*_args: object, **kwargs: Any) -> _FakeCompletedProcess:
        captured_kwargs.update(kwargs)
        return _FakeCompletedProcess()

    monkeypatch.setattr(system_power_module.subprocess, "run", _fake_run)

    system_power_module.run_power_command(["shutdown", "/s", "/t", "5"])

    assert captured_kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW


def test_run_power_command_reports_nonzero_exit_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_power_command` en sí (sin mockear a nivel de tool) reporta éxito/fracaso según el
    código de salida real — se prueba con un stub de `subprocess.run` para no invocar
    `shutdown.exe` de verdad."""

    class _FakeCompletedProcess:
        def __init__(self, returncode: int, stderr: str = "", stdout: str = "") -> None:
            self.returncode = returncode
            self.stderr = stderr
            self.stdout = stdout

    def _fake_run(*_args: object, **_kwargs: object) -> _FakeCompletedProcess:
        return _FakeCompletedProcess(returncode=1, stderr="acceso denegado")

    monkeypatch.setattr(system_power_module.subprocess, "run", _fake_run)

    success, detail = system_power_module.run_power_command(
        ["shutdown", "/s", "/t", "5"]
    )

    assert success is False
    assert detail == "acceso denegado"
