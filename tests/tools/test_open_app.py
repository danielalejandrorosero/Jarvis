"""Tests para `OpenAppTool` (`jarvis.tools.open_app`, ADR-0005).

Sin filesystem real: `START_MENU_DIRECTORIES` se reemplaza por directorios fabricados bajo
`tmp_path` con archivos `.lnk` vacíos (alcanza para probar la lógica de matching, no hace falta un
acceso directo real). `_launch_shortcut` se reemplaza por un stub que registra la invocación en
vez de llamar a `os.startfile` — mismo enfoque de monkeypatch sobre atributos de módulo que
`test_weather.py`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jarvis.tools import open_app as open_app_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.open_app import OpenAppTool


def _make_shortcut(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.lnk"
    path.write_text("")
    return path


def _install_fake_start_menu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, names: list[str]
) -> Path:
    start_menu = tmp_path / "Start Menu" / "Programs"
    for name in names:
        _make_shortcut(start_menu, name)
    monkeypatch.setattr(open_app_module, "START_MENU_DIRECTORIES", [start_menu])
    return start_menu


def _install_fake_launch(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    launched: list[Path] = []

    def _fake_launch(path: Path) -> None:
        launched.append(path)

    monkeypatch.setattr(open_app_module, "_launch_shortcut", _fake_launch)
    return launched


def test_open_app_tool_declares_safe_risk() -> None:
    """ "Abrir aplicaciones" está listado explícitamente como ejemplo SAFE en
    `.claude/rules/security.md` — no muta estado persistente."""
    assert OpenAppTool.risk is RiskLevel.SAFE


def test_execute_opens_exact_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso aceptado: nombre exacto (case-insensitive) encuentra su `.lnk` y lo lanza."""
    _install_fake_start_menu(
        monkeypatch, tmp_path, names=["Calculadora", "Google Chrome"]
    )
    launched = _install_fake_launch(monkeypatch)
    tool = OpenAppTool()

    result = asyncio.run(tool.execute(app_name="calculadora"))

    assert result == "Abriendo Calculadora."
    assert len(launched) == 1
    assert launched[0].stem == "Calculadora"


def test_execute_opens_fuzzy_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso aceptado: nombre con typo (ni exacto ni substring) resuelve vía
    `difflib.get_close_matches` en vez de fallar."""
    _install_fake_start_menu(
        monkeypatch, tmp_path, names=["Calculadora", "Bloc de notas"]
    )
    launched = _install_fake_launch(monkeypatch)
    tool = OpenAppTool()

    result = asyncio.run(tool.execute(app_name="calculdora"))

    assert result == "Abriendo Calculadora."
    assert len(launched) == 1
    assert launched[0].stem == "Calculadora"


def test_execute_prefers_exact_match_over_shorter_substring_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regresión del finding 4 de `security-reviewer`: un `.lnk` cuyo nombre coincide letra por
    letra con lo pedido debe ganar siempre, aunque exista otro shortcut más corto (y no
    relacionado) que también matchea por substring. Antes del fix, `min(..., key=len(stem))`
    hacía que "Microsoft Store" (exacto) perdiera contra un "Store" corto que también contenía
    la palabra pedida."""
    _install_fake_start_menu(
        monkeypatch,
        tmp_path,
        names=["Microsoft Store", "Store"],
    )
    launched = _install_fake_launch(monkeypatch)
    tool = OpenAppTool()

    result = asyncio.run(tool.execute(app_name="Microsoft Store"))

    assert result == "Abriendo Microsoft Store."
    assert len(launched) == 1
    assert launched[0].stem == "Microsoft Store"


def test_execute_reports_no_match_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso límite: sin ningún candidato razonable, `execute()` devuelve un mensaje claro en vez
    de lanzar algo al azar o una excepción."""
    _install_fake_start_menu(monkeypatch, tmp_path, names=["Calculadora"])
    launched = _install_fake_launch(monkeypatch)
    tool = OpenAppTool()

    result = asyncio.run(
        tool.execute(app_name="Aplicación Completamente Inexistente Xyz123")
    )

    assert "No encontré una aplicación llamada" in result
    assert launched == []


def test_execute_rejects_missing_app_name_without_filesystem_walk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso rechazado: sin `app_name` (o vacío), `execute()` no recorre el filesystem ni intenta
    lanzar nada."""
    _install_fake_start_menu(monkeypatch, tmp_path, names=["Calculadora"])
    launched = _install_fake_launch(monkeypatch)
    tool = OpenAppTool()

    result = asyncio.run(tool.execute())

    assert "nombre de aplicación válido" in result
    assert launched == []


def test_execute_rejects_blank_app_name() -> None:
    """Caso rechazado: `app_name` de solo espacios se trata igual que ausente."""
    tool = OpenAppTool()

    result = asyncio.run(tool.execute(app_name="   "))

    assert "nombre de aplicación válido" in result


def test_execute_rejects_overlong_app_name() -> None:
    """Caso rechazado: nombre irrazonablemente largo no dispara ninguna búsqueda en disco."""
    tool = OpenAppTool()

    result = asyncio.run(tool.execute(app_name="x" * 500))

    assert "longitud válida" in result
