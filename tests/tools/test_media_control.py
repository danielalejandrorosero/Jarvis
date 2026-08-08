"""Tests para `MediaControlTool` (`jarvis.tools.media_control`, ADR-0005).

`_press_media_key` se reemplaza por un stub que registra el código de tecla en vez de llamar a
`ctypes.windll.user32.keybd_event` — mismo enfoque de monkeypatch sobre atributos de módulo que
`test_open_app.py` usa para `_launch_shortcut`. Ningún test de este archivo presiona una tecla
real.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.tools import media_control as media_control_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.media_control import MediaControlTool


def _install_fake_press(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    pressed: list[int] = []

    def _fake_press(vk_code: int) -> None:
        pressed.append(vk_code)

    monkeypatch.setattr(media_control_module, "_press_media_key", _fake_press)
    return pressed


def test_media_control_tool_declares_safe_risk() -> None:
    """Un evento momentáneo de tecla multimedia, sin persistencia — SAFE (ver docstring del
    módulo)."""
    assert MediaControlTool.risk is RiskLevel.SAFE


@pytest.mark.parametrize(
    ("action", "expected_vk"),
    [
        ("play_pause", media_control_module.VK_MEDIA_PLAY_PAUSE),
        ("next", media_control_module.VK_MEDIA_NEXT_TRACK),
        ("previous", media_control_module.VK_MEDIA_PREV_TRACK),
        ("stop", media_control_module.VK_MEDIA_STOP),
    ],
)
def test_execute_presses_expected_key_per_action(
    monkeypatch: pytest.MonkeyPatch, action: str, expected_vk: int
) -> None:
    pressed = _install_fake_press(monkeypatch)
    tool = MediaControlTool()

    result = asyncio.run(tool.execute(action=action))

    assert pressed == [expected_vk]
    assert result != ""


def test_execute_rejects_unknown_action_without_pressing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso rechazado: una acción fuera del enum no dispara ningún evento de tecla."""
    pressed = _install_fake_press(monkeypatch)
    tool = MediaControlTool()

    result = asyncio.run(tool.execute(action="rewind"))

    assert "acción de reproducción válida" in result
    assert pressed == []


def test_execute_rejects_missing_action(monkeypatch: pytest.MonkeyPatch) -> None:
    pressed = _install_fake_press(monkeypatch)
    tool = MediaControlTool()

    result = asyncio.run(tool.execute())

    assert "acción de reproducción válida" in result
    assert pressed == []


def test_execute_degrades_on_keybd_event_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si la llamada WinAPI falla (`OSError`), `execute()` nunca deja que la excepción se
    propague — degrada a un mensaje de error claro."""

    def _fake_press(vk_code: int) -> None:
        raise OSError("boom")

    monkeypatch.setattr(media_control_module, "_press_media_key", _fake_press)
    tool = MediaControlTool()

    result = asyncio.run(tool.execute(action="play_pause"))

    assert "No pude ejecutar el control de reproducción" in result
