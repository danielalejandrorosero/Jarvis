"""Tests para `SystemInfoTool` (`jarvis.tools.system_info`, ADR-0005).

`_cpu_usage_percent`/`_memory_stats`/`_raw_gpu_stats` se reemplazan por stubs — ningún test de
este archivo mide hardware real ni invoca `nvidia-smi` de verdad.
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from jarvis.tools import system_info as system_info_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.system_info import SystemInfoTool, _parse_gpu_line


def _install_fake_stats(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cpu_percent: float = 42.0,
    memory: tuple[float, float, float] = (8.0, 16.0, 50.0),
    gpu_line: str | None = None,
) -> None:
    monkeypatch.setattr(system_info_module, "_cpu_usage_percent", lambda: cpu_percent)
    monkeypatch.setattr(system_info_module, "_memory_stats", lambda: memory)
    monkeypatch.setattr(system_info_module, "_raw_gpu_stats", lambda: gpu_line)


def test_system_info_tool_declares_safe_risk() -> None:
    """ "Información del sistema" está listada explícitamente como ejemplo SAFE en
    `.claude/rules/security.md` — solo lectura, no muta estado."""
    assert SystemInfoTool.risk is RiskLevel.SAFE


def test_execute_reports_cpu_and_ram_without_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin GPU disponible (`_raw_gpu_stats` -> `None`), el tool degrada a reportar solo CPU/RAM,
    sin fallar ni mencionar GPU."""
    _install_fake_stats(
        monkeypatch, cpu_percent=42.0, memory=(8.0, 16.0, 50.0), gpu_line=None
    )
    tool = SystemInfoTool()

    result = asyncio.run(tool.execute())

    assert "CPU al 42" in result
    assert "8.0 de 16.0 GB" in result
    assert "50" in result
    assert "GPU" not in result


def test_execute_reports_gpu_when_nvidia_smi_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_stats(
        monkeypatch,
        cpu_percent=10.0,
        memory=(4.0, 32.0, 12.5),
        gpu_line="35, 2048, 4096, 60",
    )
    tool = SystemInfoTool()

    result = asyncio.run(tool.execute())

    assert "GPU al 35" in result
    assert "2.0 de 4.0 GB" in result
    assert "60 grados" in result


def test_execute_degrades_when_nvidia_smi_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso central pedido: máquina sin GPU NVIDIA (o `nvidia-smi` no está en el PATH) —
    `_raw_gpu_stats` en sí (sin mockear) devuelve `None` en vez de lanzar, y `execute()` reporta
    igual CPU/RAM."""

    def _fake_run(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("nvidia-smi no encontrado")

    monkeypatch.setattr(system_info_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(system_info_module, "_cpu_usage_percent", lambda: 5.0)
    monkeypatch.setattr(system_info_module, "_memory_stats", lambda: (1.0, 8.0, 12.5))
    tool = SystemInfoTool()

    result = asyncio.run(tool.execute())

    assert "CPU al 5" in result
    assert "GPU" not in result


def test_raw_gpu_stats_returns_none_on_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`nvidia-smi` presente pero devolviendo un código de error (ej. driver en mal estado) —
    también degrada a `None`, no lanza."""

    class _FakeCompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "error"

    monkeypatch.setattr(
        system_info_module.subprocess,
        "run",
        lambda *args, **kwargs: _FakeCompletedProcess(),
    )

    assert system_info_module._raw_gpu_stats() is None


def test_raw_gpu_stats_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    monkeypatch.setattr(system_info_module.subprocess, "run", _fake_run)

    assert system_info_module._raw_gpu_stats() is None


def test_parse_gpu_line_rejects_malformed_csv() -> None:
    assert _parse_gpu_line("not,enough") is None


def test_raw_gpu_stats_hides_console_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug real, en vivo: sin `creationflags=CREATE_NO_WINDOW`, `nvidia-smi.exe` abre una
    ventana de consola visible cada vez (JARVIS corre vía `pythonw.exe`, sin consola propia) —
    mismo patrón encontrado en `jarvis.league.lcu_monitor`/`jarvis.tools.close_app`."""
    captured_kwargs: dict[str, object] = {}

    class _FakeCompletedProcess:
        returncode = 1
        stdout = ""
        stderr = ""

    def _fake_run(*_args: object, **kwargs: object) -> _FakeCompletedProcess:
        captured_kwargs.update(kwargs)
        return _FakeCompletedProcess()

    monkeypatch.setattr(system_info_module.subprocess, "run", _fake_run)

    system_info_module._raw_gpu_stats()

    assert captured_kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW
    assert _parse_gpu_line("a, b, c, d") is None


def test_parse_gpu_line_parses_well_formed_csv() -> None:
    assert _parse_gpu_line("35, 2048, 4096, 60") == (35.0, 2048.0, 4096.0, 60.0)
