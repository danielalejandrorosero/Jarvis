"""Tests para `OpenFileTool` (`jarvis.tools.open_file`, ADR-0005).

`_open_path` se reemplaza por un stub que registra la invocación en vez de llamar a
`os.startfile` — mismo enfoque de monkeypatch sobre atributos de módulo que
`test_open_app.py`/`test_screenshot.py`. `DEFAULT_DATA_DIR` se reemplaza por un directorio bajo
`tmp_path` para que la resolución "relativa a data/" (y el chequeo de jaula de rutas) nunca
toquen el `data/` real del repo. Los casos con rutas "relativas al directorio de trabajo actual"
usan `monkeypatch.chdir(tmp_path)` para el mismo motivo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from jarvis.tools import open_file as open_file_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.open_file import OpenFileTool


def _install_fake_open(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    opened: list[Path] = []

    def _fake_open(path: Path) -> None:
        opened.append(path)

    monkeypatch.setattr(open_file_module, "_open_path", _fake_open)
    return opened


def test_open_file_tool_declares_safe_risk() -> None:
    """Abrir con el programa asociado por defecto es equivalente a doble-click en el Explorador
    — mismo razonamiento que `OpenAppTool`/`OpenUrlTool` (ver docstring del módulo)."""
    assert OpenFileTool.risk is RiskLevel.SAFE


def test_execute_opens_file_relative_to_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso principal: una ruta relativa tal cual el LLM la vio en un resultado de tool anterior
    (ej. lo que devuelve `ScreenshotTool`) resuelve contra el directorio de trabajo actual, que
    cae dentro de `DEFAULT_DATA_DIR` (relativa al mismo cwd) — dentro de la jaula."""
    monkeypatch.chdir(tmp_path)
    screenshots_dir = tmp_path / "data" / "screenshots"
    screenshots_dir.mkdir(parents=True)
    target = screenshots_dir / "screenshot-20260101-120000.png"
    target.write_bytes(b"fake-png")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(
        tool.execute(path="data/screenshots/screenshot-20260101-120000.png")
    )

    # `_resolve_within_data_dir` devuelve la ruta ya resuelta (absoluta, canónica) vía
    # `Path.resolve()`, no la relativa cruda — ver docstring del módulo.
    assert opened == [target.resolve()]
    assert "Abriendo" in result


def test_execute_opens_absolute_path_within_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso aceptado: ruta absoluta, pero solo si cae dentro de `DEFAULT_DATA_DIR` — la jaula de
    rutas aplica igual a rutas absolutas que a relativas (Finding 4)."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path)
    target = tmp_path / "informe.pdf"
    target.write_bytes(b"fake-pdf")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(target)))

    assert opened == [target.resolve()]
    assert "Abriendo" in result


def test_execute_resolves_relative_path_under_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso aceptado: el LLM recuerda solo la parte final de la ruta (sin el prefijo 'data/'),
    resuelve contra `DEFAULT_DATA_DIR` como fallback."""
    (tmp_path / "somewhere_else").mkdir()
    monkeypatch.chdir(tmp_path / "somewhere_else")
    fake_data_dir = tmp_path / "data"
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", fake_data_dir)
    screenshots_dir = fake_data_dir / "screenshots"
    screenshots_dir.mkdir(parents=True)
    target = screenshots_dir / "screenshot-20260101-120000.png"
    target.write_bytes(b"fake-png")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(
        tool.execute(path="screenshots/screenshot-20260101-120000.png")
    )

    assert opened == [target.resolve()]
    assert "Abriendo" in result


def test_execute_reports_file_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso rechazado: la ruta cae dentro de la jaula (`data/`) pero no existe en disco — mensaje
    de 'no encontrado', nunca una excepción cruda."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path / "data")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path="data/no/existe/nada.png"))

    assert "No encontré el archivo" in result
    assert opened == []


def test_execute_degrades_when_startfile_raises_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso de fallo al abrir: el archivo existe (dentro de la jaula, extensión permitida) pero
    `os.startfile` levanta `OSError` (ej. sin programa asociado) — `execute()` nunca deja
    propagar la excepción, degrada a un mensaje."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path)
    target = tmp_path / "archivo.txt"
    target.write_bytes(b"contenido")

    def _raise(path: Path) -> None:
        raise OSError("no hay programa asociado")

    monkeypatch.setattr(open_file_module, "_open_path", _raise)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(target)))

    assert "no pude abrirlo" in result


def test_execute_rejects_missing_path_without_filesystem_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso rechazado: sin `path` (o vacío), `execute()` no intenta resolver ni abrir nada."""
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute())

    assert "ruta de archivo válida" in result
    assert opened == []


def test_execute_rejects_blank_path() -> None:
    """Caso rechazado: `path` de solo espacios se trata igual que ausente."""
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path="   "))

    assert "ruta de archivo válida" in result


def test_execute_rejects_overlong_path() -> None:
    """Caso rechazado: ruta irrazonablemente larga no dispara ninguna resolución en disco."""
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path="x" * 5000))

    assert "demasiado larga" in result


# --- Finding 4: jaula de rutas (path-jail) --------------------------------------------------


def test_execute_rejects_path_outside_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Caso rechazado: ruta absoluta que existe, con una extensión perfectamente inocente, pero
    fuera de `DEFAULT_DATA_DIR` — la jaula de rutas la rechaza igual (Finding 4). Este es
    exactamente el caso que la versión anterior del tool aceptaba sin restricción."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path / "data")
    outside = tmp_path / "fuera_de_data" / "informe.pdf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"fake-pdf")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(outside)))

    assert "carpeta de datos de JARVIS" in result
    assert opened == []


def test_execute_rejects_unc_path_without_any_filesystem_or_network_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 1 (crítico): una ruta UNC (`\\\\host\\share\\...`) es "absoluta" para
    `Path.is_absolute()`, y llamar a `.exists()`/`.resolve()` sobre ella dispara negociación de
    sesión SMB (autenticación NTLM incluida) contra `host`, solo por preguntar si existe — antes
    de que `os.startfile` llegue a ejecutarse. Se espía `Path.exists` y `Path.resolve` para
    demostrar, sin disparar una llamada de red real, que la jaula de rutas rechaza la ruta UNC
    puramente por comparación de strings — ninguno de los dos métodos se invoca jamás durante
    este `execute()`."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path / "data")
    opened = _install_fake_open(monkeypatch)

    exists_calls: list[Path] = []
    resolve_calls: list[Path] = []
    original_exists = Path.exists
    original_resolve = Path.resolve

    def _spy_exists(self: Path, *args: object, **kwargs: object) -> bool:
        exists_calls.append(self)
        return original_exists(self, *args, **kwargs)  # type: ignore[arg-type]

    def _spy_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        resolve_calls.append(self)
        return original_resolve(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "exists", _spy_exists)
    monkeypatch.setattr(Path, "resolve", _spy_resolve)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=r"\\attacker-host\share\payload.png"))

    assert "carpeta de datos de JARVIS" in result
    assert opened == []
    assert exists_calls == []
    assert resolve_calls == []


def test_execute_rejects_forward_slash_unc_style_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Misma familia que el UNC clásico: `//host/share/...` (doble slash) también designa un
    recurso de red en Windows y debe rechazarse por la misma vía léxica, sin I/O."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path / "data")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path="//attacker-host/share/payload.png"))

    assert "carpeta de datos de JARVIS" in result
    assert opened == []


# --- Finding 2: bypass de punto final --------------------------------------------------------


def test_execute_rejects_trailing_dot_bypass_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 2 (alto): pedir `"evil.exe."` (con punto final) en vez de `"evil.exe"` no debe
    evadir el filtro de extensión. Windows ignora el punto final al resolver la ruta real contra
    el filesystem, así que sin la normalización explícita (o sin que `Path.resolve()` revele el
    nombre canónico) el string crudo `"evil.exe."` tiene `suffix == ""`. Acá se confirma que el
    resultado es el mismo rechazo por extensión que pedir `"evil.exe"` directamente."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path)
    target = tmp_path / "evil.exe"
    target.write_bytes(b"MZ")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(target) + "."))

    assert "No abro ese tipo de archivo" in result
    assert opened == []


def test_execute_rejects_trailing_spaces_bypass_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Misma familia que el punto final: espacios finales en el último componente también los
    ignora Windows al resolver contra el filesystem real."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path)
    target = tmp_path / "evil.exe"
    target.write_bytes(b"MZ")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(target) + "   "))

    assert "No abro ese tipo de archivo" in result
    assert opened == []


def test_normalize_trailing_dots_and_spaces_strips_last_component() -> None:
    """Unit test directo de la función de normalización (defensa en profundidad explícita,
    pedida por `security-reviewer` — ver docstring del módulo)."""
    normalize = open_file_module._normalize_trailing_dots_and_spaces

    assert normalize("data/evil.exe.") == str(Path("data/evil.exe"))
    assert normalize("data/evil.exe   ") == str(Path("data/evil.exe"))
    assert normalize("data/captura.png") == "data/captura.png"  # no-op, sin punto final


# --- Finding 3: allowlist de extensiones ------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "programa.exe",
        "script.bat",
        "script.ps1",
        "script.vbs",
        "atajo.lnk",
        "instalador.msi",
        "lanzador.appref-ms",
        "clickonce.application",
        "carpeta.scf",
        "atajo_internet.url",
        "paquete.msix",
        "paquete.appx",
        "diagnostico.diagcab",
        "documento.docx",  # no ejecutable, pero tampoco está en el allowlist
        "archivo_sin_extension_conocida.xyz",
    ],
)
def test_execute_rejects_extensions_not_on_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filename: str
) -> None:
    """Decisión de diseño explícita (ver docstring del módulo): allowlist, no denylist. Se
    rechaza cualquier extensión que no esté explícitamente permitida, incluyendo ejecutables
    clásicos, lanzadores de app (ClickOnce, MSIX/APPX), `.scf`/`.url` (vectores de auth forzada /
    redirección arbitraria) y tipos de documento comunes que simplemente no están en el
    allowlist — aunque el archivo exista y caiga dentro de la jaula de rutas."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path)
    target = tmp_path / filename
    target.write_bytes(b"contenido")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(target)))

    assert "No abro ese tipo de archivo" in result
    assert opened == []


@pytest.mark.parametrize(
    "filename",
    [
        "screenshot-20260101-120000.png",
        "foto.jpg",
        "foto.jpeg",
        "animacion.gif",
        "imagen.bmp",
        "imagen.webp",
        "documento.pdf",
        "notas.txt",
        "jarvis.log",
        "datos.csv",
        "config.json",
    ],
)
def test_execute_opens_allowlisted_extensions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filename: str
) -> None:
    """Caso aceptado: todas las extensiones del allowlist (lo que JARVIS efectivamente genera
    bajo `data/` hoy — capturas `.png` de `ScreenshotTool`, `data/jarvis.log`/`jarvis-error.log`
    de `scripts/start_jarvis.ps1` — más los tipos de "solo visor" más comunes) abren sin
    problema dentro de la jaula de rutas."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path)
    target = tmp_path / filename
    target.write_bytes(b"contenido")
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(target)))

    assert opened == [target.resolve()]
    assert "Abriendo" in result


def test_execute_opens_directory_without_requiring_allowlisted_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El allowlist de extensiones aplica solo a archivos — abrir una carpeta (ej. la propia
    `data/screenshots/`) sigue soportado, como ya anunciaba la descripción del tool ("archivo o
    carpeta"), porque una carpeta no tiene una extensión ejecutable que Windows pueda disparar."""
    monkeypatch.setattr(open_file_module, "DEFAULT_DATA_DIR", tmp_path)
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    opened = _install_fake_open(monkeypatch)
    tool = OpenFileTool()

    result = asyncio.run(tool.execute(path=str(screenshots_dir)))

    assert opened == [screenshots_dir.resolve()]
    assert "Abriendo" in result
