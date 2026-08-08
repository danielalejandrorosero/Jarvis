"""Tests para `VolumeControlTool` (`jarvis.tools.volume_control`, ADR-0005).

`_press_volume_key` se reemplaza por un stub que registra el código de tecla en vez de llamar a
`ctypes.windll.user32.keybd_event` — mismo enfoque que `test_media_control.py`. Ningún test de
este archivo presiona una tecla real.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.tools import volume_control as volume_control_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.volume_control import VolumeControlTool


def _install_fake_press(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    pressed: list[int] = []

    def _fake_press(vk_code: int) -> None:
        pressed.append(vk_code)

    monkeypatch.setattr(volume_control_module, "_press_volume_key", _fake_press)
    return pressed


def test_volume_control_tool_declares_safe_risk() -> None:
    """Un paso de volumen, trivialmente reversible y sin persistencia — SAFE (ver docstring del
    módulo)."""
    assert VolumeControlTool.risk is RiskLevel.SAFE


@pytest.mark.parametrize(
    ("action", "expected_vk"),
    [
        ("up", volume_control_module.VK_VOLUME_UP),
        ("down", volume_control_module.VK_VOLUME_DOWN),
        ("mute", volume_control_module.VK_VOLUME_MUTE),
    ],
)
def test_execute_presses_expected_key_per_action(
    monkeypatch: pytest.MonkeyPatch, action: str, expected_vk: int
) -> None:
    pressed = _install_fake_press(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(action=action))

    assert pressed == [expected_vk]
    assert result != ""


def test_execute_rejects_unknown_action_without_pressing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso rechazado: una acción fuera del enum no dispara ningún evento de tecla."""
    pressed = _install_fake_press(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(action="max"))

    assert "acción de volumen válida" in result
    assert pressed == []


def test_execute_rejects_missing_action(monkeypatch: pytest.MonkeyPatch) -> None:
    pressed = _install_fake_press(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute())

    assert "acción de volumen válida" in result
    assert pressed == []


def test_execute_degrades_on_keybd_event_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si la llamada WinAPI falla (`OSError`), `execute()` nunca deja que la excepción se
    propague — degrada a un mensaje de error claro."""

    def _fake_press(vk_code: int) -> None:
        raise OSError("boom")

    monkeypatch.setattr(volume_control_module, "_press_volume_key", _fake_press)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(action="up"))

    assert "No pude ajustar el volumen" in result
