"""Tests para `CloseAppTool` (`jarvis.tools.close_app`, ADR-0005) — primer tool CONFIRM.

Sin procesos reales: `_list_running_process_names`/`_terminate_process` se reemplazan por stubs
monkeypatcheados a nivel de módulo (mismo patrón que `_launch_shortcut`/
`_find_chrome_executable` en `test_open_app.py`/`test_open_url.py`) — la suite nunca lista
procesos reales ni mata nada.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

import pytest

from jarvis.tools import close_app as close_app_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.close_app import CloseAppTool


def _install_fake_process_list(
    monkeypatch: pytest.MonkeyPatch, *, names: list[str]
) -> None:
    monkeypatch.setattr(
        close_app_module, "_list_running_process_names", lambda: list(names)
    )


def _install_fake_terminate(
    monkeypatch: pytest.MonkeyPatch, *, outcome: tuple[bool, str] | None = None
) -> list[str]:
    """Reemplaza `_terminate_process`; registra los nombres de imagen que se intentaron cerrar.

    `outcome`, si se pasa, fija el `(éxito, mensaje)` devuelto por cada llamada — default: éxito,
    mensaje genérico coherente con el nombre pedido.
    """
    calls: list[str] = []

    def _fake_terminate(image_name: str) -> tuple[bool, str]:
        calls.append(image_name)
        if outcome is not None:
            return outcome
        display = close_app_module._strip_exe(image_name)
        return True, f"Listo, cerré {display}."

    monkeypatch.setattr(close_app_module, "_terminate_process", _fake_terminate)
    return calls


def test_close_app_tool_declares_confirm_risk() -> None:
    """ "Terminar procesos" está listado explícitamente como ejemplo CONFIRM en
    `.claude/rules/security.md` — nunca SAFE, nunca DANGEROUS."""
    assert CloseAppTool.risk is RiskLevel.CONFIRM


def test_describe_names_the_resolved_process_not_the_raw_app_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 7 de `security-reviewer` (segunda ronda): el prompt de confirmación tiene que
    nombrar el proceso REALMENTE resuelto (ej. 'chrome.exe'), no el texto crudo que pidió el
    usuario — y avisar que el cierre es forzado y afecta todas las ventanas."""
    _install_fake_process_list(monkeypatch, names=["chrome.exe", "Discord.exe"])
    tool = CloseAppTool()

    description = tool.describe({"app_name": "Chrome"})

    assert description == (
        "cerrar chrome.exe (forzado, todas las ventanas abiertas, sin guardar cambios)"
    )


def test_describe_resolves_fuzzy_match_before_confirming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El prompt de confirmación refleja el resultado de la MISMA resolución fuzzy-match que usa
    `execute()` — si `app_name` no matchea exacto pero sí resuelve por `difflib`, el usuario
    escucha el proceso real que se va a cerrar, no la palabra que dijo."""
    _install_fake_process_list(monkeypatch, names=["Discord.exe", "notepad.exe"])
    tool = CloseAppTool()

    description = tool.describe({"app_name": "discrod"})

    assert description == (
        "cerrar Discord.exe (forzado, todas las ventanas abiertas, sin guardar cambios)"
    )


def test_describe_reports_no_match_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si nada matchea, `describe()` no explota ni inventa un target — da una frase honesta de
    que no hay nada real para confirmar, en vez del nombre pedido como si existiera."""
    _install_fake_process_list(monkeypatch, names=["chrome.exe"])
    tool = CloseAppTool()

    description = tool.describe({"app_name": "Aplicación Inexistente Xyz123"})

    assert "no encontré ninguna aplicación abierta" in description
    assert "Aplicación Inexistente Xyz123" in description


def test_describe_falls_back_to_tool_name_without_app_name() -> None:
    tool = CloseAppTool()

    assert tool.describe({}) == "close_app"


def test_execute_closes_exact_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso aceptado: nombre exacto (case-insensitive, sin `.exe`) encuentra su proceso y lo
    cierra vía taskkill."""
    _install_fake_process_list(monkeypatch, names=["chrome.exe", "Discord.exe"])
    calls = _install_fake_terminate(monkeypatch)
    tool = CloseAppTool()

    result = asyncio.run(tool.execute(app_name="chrome"))

    assert result == "Listo, cerré chrome."
    assert calls == ["chrome.exe"]


def test_execute_closes_fuzzy_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso aceptado: nombre con typo (ni exacto ni substring) resuelve vía
    `difflib.get_close_matches`, mismo fallback que `open_app.py`."""
    _install_fake_process_list(monkeypatch, names=["Discord.exe", "notepad.exe"])
    calls = _install_fake_terminate(monkeypatch)
    tool = CloseAppTool()

    result = asyncio.run(tool.execute(app_name="discrod"))

    assert result == "Listo, cerré Discord."
    assert calls == ["Discord.exe"]


def test_execute_prefers_exact_match_over_shorter_substring_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regresión del mismo finding que `open_app.py` (finding 4 de `security-reviewer`): un
    proceso cuyo nombre coincide letra por letra con lo pedido gana siempre, aunque exista otro
    proceso más corto que también matchea por substring."""
    _install_fake_process_list(monkeypatch, names=["Steam.exe", "SteamService.exe"])
    calls = _install_fake_terminate(monkeypatch)
    tool = CloseAppTool()

    result = asyncio.run(tool.execute(app_name="Steam"))

    assert result == "Listo, cerré Steam."
    assert calls == ["Steam.exe"]


def test_execute_reports_no_match_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso rechazado: sin ningún candidato razonable, `execute()` devuelve un mensaje claro y
    nunca llama a taskkill."""
    _install_fake_process_list(monkeypatch, names=["chrome.exe"])
    calls = _install_fake_terminate(monkeypatch)
    tool = CloseAppTool()

    result = asyncio.run(
        tool.execute(app_name="Aplicación Completamente Inexistente Xyz123")
    )

    assert "No encontré ninguna aplicación abierta llamada" in result
    assert calls == []


@pytest.mark.parametrize(
    "process_name", ["explorer.exe", "winlogon.exe", "svchost.exe", "python.exe"]
)
def test_execute_refuses_denylisted_process_even_on_exact_match(
    monkeypatch: pytest.MonkeyPatch, process_name: str
) -> None:
    """Un proceso denylisteado (crítico del sistema, o JARVIS mismo) se rechaza incluso con un
    match exacto y aunque `PolicyEngine` ya haya recibido confirmación de voz — taskkill nunca se
    invoca."""
    _install_fake_process_list(monkeypatch, names=[process_name])
    calls = _install_fake_terminate(monkeypatch)
    tool = CloseAppTool()

    app_name = close_app_module._strip_exe(process_name)
    result = asyncio.run(tool.execute(app_name=app_name))

    assert "No puedo cerrar" in result
    assert calls == []


def test_execute_refuses_denylisted_process_reached_via_fuzzy_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hallazgo LOW de `security-reviewer`: la parametrización anterior solo probaba el camino de
    match EXACTO llegando a un proceso denylisteado. Acá el denylist se alcanza vía el fallback de
    `difflib` (ni exacto ni substring: 'explorador' no es substring de 'explorer' ni viceversa) —
    un futuro refactor de `_find_best_match` que cambie cómo fluyen los matches no puede pasar por
    alto el chequeo del denylist sin que este test lo note."""
    _install_fake_process_list(monkeypatch, names=["explorer.exe"])
    calls = _install_fake_terminate(monkeypatch)
    tool = CloseAppTool()

    result = asyncio.run(tool.execute(app_name="explorador"))

    assert "No puedo cerrar" in result
    assert calls == []


def test_execute_reports_taskkill_failure_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cuando `taskkill` falla (código de salida distinto de 0 — proceso ya cerrado, permisos
    insuficientes, etc.), `execute()` no afirma falsamente que cerró la app."""
    _install_fake_process_list(monkeypatch, names=["chrome.exe"])
    calls = _install_fake_terminate(
        monkeypatch, outcome=(False, "No pude cerrar chrome: acceso denegado.")
    )
    tool = CloseAppTool()

    result = asyncio.run(tool.execute(app_name="chrome"))

    assert result == "No pude cerrar chrome: acceso denegado."
    assert "cerré" not in result
    assert calls == ["chrome.exe"]


def test_execute_rejects_missing_app_name_without_listing_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso rechazado: sin `app_name` (o vacío), `execute()` no lista procesos ni intenta cerrar
    nada."""

    def _fail_if_called() -> list[str]:
        raise AssertionError("no debería listar procesos sin un app_name válido")

    monkeypatch.setattr(
        close_app_module, "_list_running_process_names", _fail_if_called
    )
    calls = _install_fake_terminate(monkeypatch)
    tool = CloseAppTool()

    result = asyncio.run(tool.execute())

    assert "nombre de aplicación válido" in result
    assert calls == []


def test_execute_rejects_blank_app_name() -> None:
    """Caso rechazado: `app_name` de solo espacios se trata igual que ausente."""
    tool = CloseAppTool()

    result = asyncio.run(tool.execute(app_name="   "))

    assert "nombre de aplicación válido" in result


def test_execute_rejects_overlong_app_name() -> None:
    """Caso rechazado: nombre irrazonablemente largo no dispara ninguna búsqueda de procesos."""
    tool = CloseAppTool()

    result = asyncio.run(tool.execute(app_name="x" * 500))

    assert "longitud válida" in result


def test_terminate_process_reports_success_from_taskkill_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_terminate_process` en sí (sin mockear) reporta éxito/fracaso según el código de salida
    real de `taskkill` — se prueba con un stub de `subprocess.run` para no invocar taskkill de
    verdad."""

    class _FakeCompletedProcess:
        def __init__(self, returncode: int, stderr: str = "", stdout: str = "") -> None:
            self.returncode = returncode
            self.stderr = stderr
            self.stdout = stdout

    def _fake_run(*_args: object, **_kwargs: object) -> _FakeCompletedProcess:
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(close_app_module.subprocess, "run", _fake_run)

    success, message = close_app_module._terminate_process("chrome.exe")

    assert success is True
    assert message == "Listo, cerré chrome."


# --- ventana de consola oculta (bug real, en vivo) ----------------------------------------------
# `_list_running_process_names`/`_terminate_process` invocan `tasklist`/`taskkill` — sin
# `creationflags=CREATE_NO_WINDOW`, Windows abre una ventana de consola visible cada vez (JARVIS
# corre vía `pythonw.exe`, sin consola propia) — mismo bug real encontrado en vivo con el mismo
# patrón en `jarvis.league.lcu_monitor`.


def test_list_running_process_names_hides_console_window(
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

    monkeypatch.setattr(close_app_module.subprocess, "run", _fake_run)

    close_app_module._list_running_process_names()

    assert captured_kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW


def test_terminate_process_hides_console_window(
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

    monkeypatch.setattr(close_app_module.subprocess, "run", _fake_run)

    close_app_module._terminate_process("chrome.exe")

    assert captured_kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW
