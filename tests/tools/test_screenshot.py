"""Tests para `ScreenshotTool` (`jarvis.tools.screenshot`, ADR-0005).

`_capture_screen` se reemplaza por un stub que devuelve una imagen fabricada en vez de llamar a
`PIL.ImageGrab.grab()`, y `DEFAULT_SCREENSHOT_DIR` se reemplaza por un directorio bajo `tmp_path`
— mismo enfoque que `test_open_app.py` reemplaza `START_MENU_DIRECTORIES`. Ningún test de este
archivo captura la pantalla real ni escribe fuera de `tmp_path`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from jarvis.tools import screenshot as screenshot_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.screenshot import ScreenshotTool


def _install_fake_capture(monkeypatch: pytest.MonkeyPatch) -> Image.Image:
    fake_image = Image.new("RGB", (2, 2))
    monkeypatch.setattr(screenshot_module, "_capture_screen", lambda: fake_image)
    return fake_image


def test_screenshot_tool_declares_safe_risk() -> None:
    """Captura y guarda un archivo propio de JARVIS, sin mutar estado del sistema ni requerir
    privilegios — SAFE (ver docstring del módulo)."""
    assert ScreenshotTool.risk is RiskLevel.SAFE


def test_execute_saves_screenshot_under_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target_dir = tmp_path / "screenshots"
    monkeypatch.setattr(screenshot_module, "DEFAULT_SCREENSHOT_DIR", target_dir)
    _install_fake_capture(monkeypatch)
    tool = ScreenshotTool()

    result = asyncio.run(tool.execute())

    saved_files = list(target_dir.glob("screenshot-*.png"))
    assert len(saved_files) == 1
    assert str(saved_files[0]) in result
    assert "Captura de pantalla guardada" in result


def test_execute_creates_directory_if_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El subdirectorio de destino no existe todavía — `execute()` lo crea en vez de fallar."""
    target_dir = tmp_path / "nested" / "screenshots"
    assert not target_dir.exists()
    monkeypatch.setattr(screenshot_module, "DEFAULT_SCREENSHOT_DIR", target_dir)
    _install_fake_capture(monkeypatch)
    tool = ScreenshotTool()

    asyncio.run(tool.execute())

    assert target_dir.is_dir()
    assert list(target_dir.glob("screenshot-*.png"))


def test_execute_degrades_when_directory_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso de fallo al guardar: el directorio de destino no se puede crear (acá simulado como un
    archivo ya existente donde se esperaba un directorio) — `execute()` nunca deja propagar la
    excepción, degrada a un mensaje de error."""
    blocking_file = tmp_path / "screenshots"
    blocking_file.write_text("no soy un directorio")
    monkeypatch.setattr(screenshot_module, "DEFAULT_SCREENSHOT_DIR", blocking_file)
    _install_fake_capture(monkeypatch)
    tool = ScreenshotTool()

    result = asyncio.run(tool.execute())

    assert "No pude tomar la captura de pantalla" in result
