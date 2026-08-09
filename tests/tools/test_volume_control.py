"""Tests para `VolumeControlTool` (`jarvis.tools.volume_control`, ADR-0005 + nivel exacto).

`_press_volume_key`/`_set_volume_level` se reemplazan por stubs que registran la llamada en vez de
tocar `ctypes.windll.user32.keybd_event`/la Core Audio API real (`pycaw`) — mismo enfoque que
`test_media_control.py`. Ningún test de este archivo presiona una tecla real ni llama a `pycaw`.
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


def _install_fake_set_level(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    levels_set: list[int] = []

    def _fake_set_level(level: int) -> None:
        levels_set.append(level)

    monkeypatch.setattr(volume_control_module, "_set_volume_level", _fake_set_level)
    return levels_set


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


# --- level (nivel exacto vía pycaw / Core Audio API) ------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 30, 50, 99, 100])
def test_execute_sets_exact_level_within_range(
    monkeypatch: pytest.MonkeyPatch, level: int
) -> None:
    levels_set = _install_fake_set_level(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(level=level))

    assert levels_set == [level]
    assert f"{level}%" in result


def test_execute_level_takes_precedence_over_action_when_both_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pressed = _install_fake_press(monkeypatch)
    levels_set = _install_fake_set_level(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(action="up", level=42))

    assert levels_set == [42]
    assert pressed == []
    assert "42%" in result


@pytest.mark.parametrize("level", [-1, 101, 1000, -50])
def test_execute_rejects_out_of_range_level_without_touching_audio_api(
    monkeypatch: pytest.MonkeyPatch, level: int
) -> None:
    levels_set = _install_fake_set_level(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(level=level))

    assert "entre 0 y 100" in result
    assert levels_set == []


@pytest.mark.parametrize("level", ["30", 30.5, None, [30], "treinta"])
def test_execute_rejects_non_integer_level_without_touching_audio_api(
    monkeypatch: pytest.MonkeyPatch, level: object
) -> None:
    levels_set = _install_fake_set_level(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(level=level))

    if level is None:
        # `level=None` se trata igual que no haberlo pasado: cae al flujo de `action`, que acá
        # tampoco viene, así que el mensaje esperado es el de acción inválida, no el de rango.
        assert "acción de volumen válida" in result
    else:
        assert "entre 0 y 100" in result
    assert levels_set == []


def test_execute_rejects_boolean_level_despite_being_an_int_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`bool` es subclase de `int` en Python — mismo chequeo explícito que `TimerTool`/
    `ReminderTool` para que `level=True` no se cuele como un volumen de 1%."""
    levels_set = _install_fake_set_level(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(level=True))

    assert "entre 0 y 100" in result
    assert levels_set == []


def test_execute_degrades_on_core_audio_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un fallo de la Core Audio API vía `pycaw`/`comtypes` (ej. `comtypes.COMError`, que no
    hereda de `OSError`) tampoco puede escapar de `execute()`."""

    def _fake_set_level(level: int) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(volume_control_module, "_set_volume_level", _fake_set_level)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(level=50))

    assert "No pude fijar el volumen" in result


# --- backward-compat: action sigue funcionando exactamente igual sin level --------------------


def test_execute_action_still_works_without_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pressed = _install_fake_press(monkeypatch)
    levels_set = _install_fake_set_level(monkeypatch)
    tool = VolumeControlTool()

    result = asyncio.run(tool.execute(action="mute"))

    assert pressed == [volume_control_module.VK_VOLUME_MUTE]
    assert levels_set == []
    assert "silencié el audio" in result
