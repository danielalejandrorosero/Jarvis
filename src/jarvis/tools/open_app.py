"""Tool para abrir aplicaciones instaladas por nombre (ADR-0005).

Windows no expone una API única para "abrir una app por nombre difuso" — el enfoque estándar (el
mismo que usa Windows Search por debajo) es buscar accesos directos `.lnk` en los directorios de
Start Menu:

- `%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs` (por usuario)
- `%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs` (todos los usuarios)

...recursivamente, hacer fuzzy-match del nombre pedido contra los nombres de archivo (sin
extensión), y lanzar el mejor match vía `os.startfile()` — la forma estándar de "doble-clickear"
un archivo/acceso directo desde Python: resuelve el destino del `.lnk` sin que haga falta parsear
el binario del shortcut.

SAFE bajo `.claude/rules/security.md` — "abrir aplicaciones" está en la lista explícita de
ejemplos SAFE (lectura/consulta, no muta estado persistente; solo lanza un proceso ya instalado
por el usuario, igual que hacer doble-click en el Start Menu).

Matching: `difflib.get_close_matches` sobre el nombre de archivo (sin `.lnk`, en minúsculas), más
un paso previo de substring case-insensitive, en vez de una tabla de alias hardcodeada por app —
mantiene el tool general en vez de app-específico. Límite conocido: abreviaturas/siglas muy cortas
o muy distintas de su nombre completo ("LoL" vs "League of Legends") pueden no superar el cutoff
de similaridad porque no comparten suficientes caracteres en secuencia; en ese caso el tool
devuelve "no encontré" en vez de adivinar (ver `_find_best_match`).

Ante múltiples candidatos plausibles, se lanza el de mejor score (exacto primero siempre; si no
hay exacto, mayor cobertura relativa entre los matches por substring; si no hay substring, el
resultado top de `difflib`) en vez de listar opciones para desambiguar — pedir "¿cuál de estos
querías?" necesitaría una ronda de confirmación fuera del alcance de este tool; con una interfaz
de voz, abrir el candidato más plausible es preferible a forzar al usuario a repetirse.
"""

from __future__ import annotations

import difflib
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

SHORTCUT_EXTENSION = ".lnk"

MIN_APP_NAME_LENGTH = 1
MAX_APP_NAME_LENGTH = 100

# Cutoff de similaridad para difflib.get_close_matches — 0.0 (cualquier cosa matchea) a 1.0
# (exacto). 0.45 tolera variaciones razonables (typos, orden de palabras parcial) sin volverse tan
# laxo que cualquier nombre corto matchee con cualquier accesorio del Start Menu.
_FUZZY_MATCH_CUTOFF = 0.45


def _default_start_menu_directories() -> list[Path]:
    """Directorios de Start Menu por defecto: por-usuario (`%APPDATA%`) y todos-los-usuarios
    (`%ProgramData%`). Función, no valor calculado a import-time, para no depender de las
    variables de entorno del proceso en el momento del `import` — se evalúa una vez al definir
    `START_MENU_DIRECTORIES` más abajo, que es lo que los tests monkeypatchean."""
    directories: list[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        directories.append(
            Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        )
    programdata = os.environ.get("ProgramData")
    if programdata:
        directories.append(
            Path(programdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        )
    return directories


# Nivel de módulo, monkeypatcheable por tests (mismo patrón que `weather_module.httpx` en
# `test_weather.py`) — así la suite nunca recorre el Start Menu real ni lanza `os.startfile`.
START_MENU_DIRECTORIES: list[Path] = _default_start_menu_directories()


def _launch_shortcut(path: Path) -> None:
    """Lanzar un acceso directo (o cualquier archivo) como si el usuario le hubiera hecho
    doble-click. `os.startfile` es asíncrono del lado de Windows: devuelve el control apenas el
    proceso fue *invocado*, no espera a que la app termine de abrir ni reporta si el lanzamiento
    en sí tuvo éxito más allá de que el acceso directo existía — por eso `OpenAppTool.execute()`
    nunca afirma "se abrió", solo "se está abriendo"."""
    os.startfile(str(path))


def _iter_shortcuts(directories: Sequence[Path]) -> list[Path]:
    """Recorrer `directories` recursivamente y devolver todos los `.lnk` encontrados. Un único
    recorrido por llamada a `execute()`, sin cache ni índice — los directorios de Start Menu no
    son lo bastante grandes como para justificar más que eso en esta fase."""
    shortcuts: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        try:
            for path in directory.rglob(f"*{SHORTCUT_EXTENSION}"):
                if path.is_file():
                    shortcuts.append(path)
        except OSError:
            # Directorio no accesible (permisos, link roto) — se salta, no aborta el resto de la
            # búsqueda en los otros directorios.
            logger.warning(
                "No se pudo recorrer '%s' buscando accesos directos.", directory
            )
    return shortcuts


def _find_best_match(app_name: str, shortcuts: Sequence[Path]) -> Path | None:
    """Encontrar el `.lnk` cuyo nombre de archivo mejor matchea `app_name` (ver límites y
    justificación del enfoque en el docstring del módulo)."""
    if not shortcuts:
        return None

    normalized_target = app_name.strip().lower()
    names_by_stem: dict[str, Path] = {}
    for shortcut in shortcuts:
        # En caso de nombres duplicados entre directorios (usuario y todos-los-usuarios), se
        # conserva el primero encontrado — no hay una razón para preferir uno sobre otro acá.
        names_by_stem.setdefault(shortcut.stem.lower(), shortcut)

    # Match exacto primero, siempre — un `.lnk` cuyo nombre coincide letra por letra con lo
    # pedido nunca debe perder contra un match por substring más corto pero no relacionado
    # (hallazgo de `security-reviewer`, finding 4: "Microsoft Store" pedido explícitamente no
    # puede resolver a un shortcut corto tipo "Store" solo porque su stem es más breve).
    if normalized_target in names_by_stem:
        return names_by_stem[normalized_target]

    substring_matches = [
        stem
        for stem in names_by_stem
        if normalized_target in stem or stem in normalized_target
    ]
    if substring_matches:
        # Entre los que matchean por substring (sin match exacto disponible), preferir mayor
        # cobertura relativa — la razón entre el más corto y el más largo de (target, stem) — en
        # vez de simplemente el stem más corto: un match casi exacto por longitud ("Chrome" vs
        # "Google Chrome") debe ganarle a uno mucho más corto y solo parcialmente relacionado
        # ("Chrome" vs "Chrome Remote Desktop Host" no debería perder contra un "Chrome.txt"
        # accidental de 6 caracteres). Empate en cobertura: preferir el stem más corto, mismo
        # criterio que antes.
        def _coverage(stem: str) -> float:
            shorter = min(len(stem), len(normalized_target))
            longer = max(len(stem), len(normalized_target))
            return shorter / longer if longer else 0.0

        best_stem = max(
            substring_matches, key=lambda stem: (_coverage(stem), -len(stem))
        )
        return names_by_stem[best_stem]

    close = difflib.get_close_matches(
        normalized_target, names_by_stem.keys(), n=1, cutoff=_FUZZY_MATCH_CUTOFF
    )
    if not close:
        return None
    return names_by_stem[close[0]]


class OpenAppTool(Tool):
    """Abre una aplicación instalada, buscándola por nombre entre los accesos directos del Start
    Menu de Windows."""

    name = "open_app"
    description = (
        "Abre una aplicación instalada en Windows a partir de su nombre en lenguaje natural "
        "(ej. 'Chrome', 'calculadora', 'League of Legends'). Usar cuando el usuario pide abrir, "
        "iniciar o ejecutar un programa."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": (
                    "Nombre de la aplicación tal como lo dijo el usuario, ej. 'League of "
                    "Legends' o 'calculadora'."
                ),
            }
        },
        "required": ["app_name"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        app_name = kwargs.get("app_name")
        if not isinstance(app_name, str) or not app_name.strip():
            return "No se especificó un nombre de aplicación válido para abrir."
        app_name = app_name.strip()
        if len(app_name) < MIN_APP_NAME_LENGTH or len(app_name) > MAX_APP_NAME_LENGTH:
            return "El nombre de aplicación no tiene una longitud válida."

        shortcuts = _iter_shortcuts(START_MENU_DIRECTORIES)
        match = _find_best_match(app_name, shortcuts)
        if match is None:
            return f"No encontré una aplicación llamada '{app_name}'."

        try:
            _launch_shortcut(match)
        except OSError as exc:
            return (
                f"Encontré '{match.stem}' pero no pude abrirlo "
                f"({exc.__class__.__name__})."
            )

        return f"Abriendo {match.stem}."
