"""Tool para abrir un archivo (o carpeta) local por su ruta, con el programa asociado por
defecto de Windows (ADR-0005).

Origen concreto de este tool: `ScreenshotTool` guarda una captura y devuelve su ruta en el
resultado del tool (`"Captura de pantalla guardada en data/screenshots/screenshot-....png."`),
que queda persistido en `conversation_turns` y se reinyecta como contexto en turnos futuros
(`jarvis.audio.pipeline._build_system_prompt`). Hasta este tool, JARVIS no tenía forma de
completar "Alexa, sacá una captura" → "abrí la captura": `OpenAppTool` solo resuelve nombres de
programas instalados contra accesos directos del Start Menu, y `OpenUrlTool` solo abre
`http`/`https`. Ninguno de los dos puede abrir una ruta de archivo local arbitraria.

Mecanismo: `os.startfile(path)`, la misma API que ya usa `open_app._launch_shortcut` — es
"doble-click" a nivel de shell (`ShellExecuteW` por debajo), no requiere ninguna dependencia
nueva.

## Path-jail: solo `data/` (hallazgo crítico de `security-reviewer`, corregido acá)

Versión anterior de este tool: `_resolve_existing_path` aceptaba *cualquier* ruta absoluta del
filesystem que el LLM lograra nombrar (tras eso, sí, filtraba extensiones ejecutables). Eso abría
varios problemas que `security-reviewer` bloqueó en revisión:

1. **(Crítico) Autenticación SMB forzada vía ruta UNC.** Una ruta UNC (`\\\\host\\share\\x.png`)
   es "absoluta" para `Path.is_absolute()`. Llamar a `Path.exists()` — o, peor, `Path.resolve()`
   (que en Windows usa `_getfinalpathname`, es decir `GetFinalPathNameByHandleW` sobre un handle
   abierto con `CreateFileW`) — sobre esa ruta dispara negociación de sesión SMB contra `host`,
   NTLM incluido, **solo por preguntar si existe**, sin que `os.startfile` llegue a ejecutarse
   nunca. Es la técnica documentada de "forced authentication" / NTLM relay. `open_url.py` ya
   tiene una validación de host análoga (`_is_host_allowed`) para el caso HTTP; este tool no tenía
   ningún análogo para el caso filesystem.
2. **(Alto) El filtro de extensiones ejecutables (denylist) se evadía con un punto final.**
   `Path("evil.exe.").suffix` es `""` (no `".exe"`) a nivel de string de Python, pero Win32
   (`CreateFileW`, y por lo tanto `os.startfile`/`ShellExecuteW`) descarta puntos/espacios finales
   del último componente de la ruta al resolverla contra el filesystem real — comportamiento
   documentado de Windows, el mismo mecanismo detrás de bypasses de Mark-of-the-Web como
   CVE-2022-41049. Si `evil.exe` existe junto a `evil.exe.`, pedir `"evil.exe."` resolvía (gracias
   a que el propio Windows ya ignora el punto final al nivel de syscall — confirmado en este
   entorno: `Path("...evil.exe.").exists()` da `True` cuando solo existe `evil.exe`) y ejecutaba
   el binario real, evadiendo el filtro por completo.
3. **(Medio) El denylist era estructuralmente incompleto** contra otras extensiones que
   `ShellExecuteW` *ejecuta* en vez de "abrir para ver" (`.appref-ms`/`.application` — lanzadores
   ClickOnce, `.scf` — históricamente usado para el mismo truco de autenticación SMB forzada del
   punto 1 vía su campo de ícono, `.url` — accesos directos de Internet cuyo campo `URL=` puede
   apuntar a cualquier esquema, `.msix`/`.appx` — lanzadores de paquetes de app, `.diagcab` —
   disparadores de scripts de diagnóstico con historial de CVEs). Mantener un denylist exhaustivo
   contra esa superficie es una carrera perdida de antemano.
4. **(Medio) No había jaula de rutas**: el tool podía resolver a cualquier lugar del filesystem
   alcanzable por el proceso de JARVIS, no solo `data/` — el único lugar donde JARVIS realmente
   genera contenido propio (capturas de pantalla, hoy; potencialmente más adelante), que es el caso
   de uso real que motivó este tool.

Corrección, siguiendo la recomendación explícita de `security-reviewer`: restringir la resolución
a rutas que caigan (tras normalizar) dentro de `DEFAULT_DATA_DIR` cierra el punto 4
*estructuralmente*, y de paso cierra el punto 1 sin necesitar ninguna lógica de detección de host
— una ruta UNC nunca puede caer "dentro de `data/`", así que un único chequeo de contención
elimina el vector de autenticación SMB forzada, y además reduce el radio de impacto de los puntos
2-3 al propio directorio de datos de JARVIS. Ver `_is_within_directory`/`_candidate_paths`/
`_resolve_within_data_dir` para la implementación.

**Importante sobre cómo se implementa el chequeo de contención**: nunca se llama a
`Path.resolve()` (ni a `Path.exists()`, ni a ningún otro método que abra un handle) sobre una ruta
candidata *antes* de saber que esa ruta cae dentro de `data/` — porque `Path.resolve()` en Windows
es exactamente el API (`_getfinalpathname`) que dispara el problema del punto 1 para una ruta UNC.
El chequeo de contención (`_is_within_directory`) usa `Path.absolute()` (que según su propia
documentación "no realiza normalización ni resolución de symlinks", solo antepone el directorio de
trabajo actual — sin tocar el filesystem) más `os.path.normpath` (manipulación de string pura) —
ninguno de los dos toca disco ni red. Recién *después* de confirmar por esa vía puramente léxica
que una ruta cae dentro de `data/` se le aplica `.exists()` y, si existe, `.resolve()` (para
defensa en profundidad extra contra symlinks/`..` que pudieran escapar de `data/` — improbable,
pero barato de chequear ya que en ese punto la ruta es local y de bajo riesgo).

## Allowlist en vez de denylist de extensiones

Reemplaza `EXECUTABLE_EXTENSIONS` (denylist) por `ALLOWED_EXTENSIONS` (allowlist): dado que el
único generador de contenido bajo `data/` hoy es `ScreenshotTool` (`.png`, ver
`jarvis.tools.screenshot.DEFAULT_SCREENSHOT_DIR`) y `data/jarvis.log`/`data/jarvis-error.log`
(`scripts/start_jarvis.ps1`), la lista se arma con esos formatos confirmados más los tipos de
"solo visor" más comunes que JARVIS podría llegar a generar o que el usuario podría depositar en
`data/` a mano (imágenes, PDF, texto plano, CSV, JSON) — nunca ejecutables ni lanzadores de
ningún tipo. Con la jaula de rutas ya en su lugar, esta lista ya no es la única barrera (punto 4
del análisis de arriba reduce el radio de impacto de cualquier hueco acá), pero sigue siendo
estructuralmente más segura que un denylist: lo desconocido se rechaza por defecto en vez de
aceptarse por defecto.

Además, la comparación de extensión se hace sobre `resolved.suffix` — la ruta ya resuelta con
`Path.resolve()` tras confirmar que existe — no sobre el string crudo que mandó el LLM.
`Path.resolve()` consulta el nombre canónico real en disco (vía `GetFinalPathNameByHandleW`), que
ya reflaja el comportamiento de Windows de ignorar puntos/espacios finales (confirmado en este
entorno: para un archivo real `evil.exe`, `Path("evil.exe.").resolve().suffix` da `.exe`, no
`""`) — así que el punto 2 del análisis de arriba queda cerrado incluso sin lógica adicional. Aun
así, `_normalize_trailing_dots_and_spaces` se aplica también al string de entrada, *antes* de
construir candidatos, como defensa en profundidad explícita y documentada (pedido concreto de
`security-reviewer`) — no por necesidad estricta dado lo anterior, sino para que la intención sea
explícita en el código y no dependa implícitamente de un detalle de implementación de
`Path.resolve()`.

## Resolución de ruta

El LLM no tiene memorizada una ruta absoluta exacta de forma confiable — tiene el texto que vio
en un resultado de tool o en el historial de conversación (ej. lo que devolvió `ScreenshotTool`,
que es una ruta *relativa* al directorio desde el que corre JARVIS: `Path("data/screenshots") /
nombre`, igual que `jarvis.tools.screenshot.DEFAULT_SCREENSHOT_DIR`). `_candidate_paths` prueba,
en orden, hasta encontrar una que exista *y* caiga dentro de `DEFAULT_DATA_DIR`:

1. Ruta absoluta tal cual, si `path` ya lo es (`Path.is_absolute()`).
2. Ruta relativa tal cual, contra el directorio de trabajo actual — cubre el caso normal: el LLM
   repite literalmente la ruta que le devolvió `ScreenshotTool` (`data/screenshots/screenshot-
   ....png`), que ya es relativa al mismo directorio desde el que corre JARVIS.
3. Ruta relativa contra `DEFAULT_DATA_DIR` (`data/`, mismo directorio de estado en tiempo de
   ejecución que usa `jarvis.memory.store`/`jarvis.tools.screenshot`) — cubre el caso en que el
   LLM recuerda solo la parte final (ej. `screenshots/screenshot-....png` sin el prefijo `data/`).

Si ninguna candidata cae dentro de `data/`, `execute()` rechaza sin haber tocado el filesystem. Si
al menos una cae dentro pero ninguna existe, `execute()` devuelve un mensaje de "no encontrado".

## Alcance: por qué se bloquean ejecutables/scripts

A diferencia de `OpenAppTool` (limitado a accesos directos ya presentes en el Start Menu — una
lista curada de software que el usuario mismo instaló) y de `OpenUrlTool` (limitado a esquemas
`http`/`https`, sin poder disparar manejadores de protocolo arbitrarios), este tool recibe una
ruta de filesystem construida por el LLM a partir de *cualquier* texto que haya visto —
incluyendo, en teoría, contenido reinyectado desde `<web_data>`, `<remembered_facts>` o
`<conversation_history>` (ver el framing de prompt-injection ya documentado en
`jarvis.audio.pipeline` para esas tres fuentes). `os.startfile` sobre un `.exe`/`.bat`/`.cmd`/
`.ps1`/etc. no "abre para ver" — *ejecuta* ese programa. La jaula de rutas a `data/` ya acota
drásticamente qué se puede llegar a nombrar (nada fuera de lo que JARVIS mismo escribe ahí), y el
allowlist de extensiones acota además *qué tipo* de contenido dentro de esa carpeta se puede abrir
— nunca algo que Windows trate como ejecutable.

El resto (imágenes, PDFs, texto, CSV, JSON, logs, carpetas) queda SAFE bajo
`.claude/rules/security.md` con el mismo razonamiento que `OpenAppTool`/`OpenUrlTool`: abrir algo
con su programa asociado por defecto es exactamente lo que hace el usuario al doble-clickearlo en
el Explorador — no muta estado persistente del sistema ni del usuario más allá de lanzar el
visor/editor que ya está asociado a ese tipo de archivo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

# Mismo directorio de estado en tiempo de ejecución que `jarvis.memory.store.DEFAULT_DB_PATH` y
# `jarvis.tools.screenshot.DEFAULT_SCREENSHOT_DIR` — relativo al directorio desde el que corre
# JARVIS, no un path absoluto hardcodeado. Nivel de módulo y monkeypatcheable (mismo patrón que
# esos dos) para que los tests nunca resuelvan contra el `data/` real del repo. Es también, ahora,
# la jaula de rutas: ver docstring del módulo, sección "Path-jail".
DEFAULT_DATA_DIR = Path("data")

MAX_PATH_LENGTH = (
    4096  # generoso; consistencia con MAX_APP_NAME_LENGTH/MAX_URL_LENGTH, no es
)
# en sí mismo una mitigación de seguridad (ver docstring de `open_url.py` para el mismo criterio).

# Allowlist de extensiones, no denylist — ver docstring del módulo, sección "Allowlist en vez de
# denylist", para el razonamiento y qué genera JARVIS hoy bajo `data/`. Se aplica solo a archivos
# (nunca a carpetas — ver `OpenFileTool.execute`).
ALLOWED_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".webp",
        ".pdf",
        ".txt",
        ".log",
        ".csv",
        ".json",
    }
)

_OUTSIDE_DATA_DIR_MESSAGE = (
    "Solo puedo abrir archivos dentro de la carpeta de datos de JARVIS (data/)."
)
_EXTENSION_REJECTION_MESSAGE = (
    "No abro ese tipo de archivo por seguridad — solo abro imágenes, PDFs, texto, CSV, JSON "
    "o logs dentro de la carpeta de datos de JARVIS."
)
_LENGTH_REJECTION_MESSAGE = "Esa ruta de archivo es demasiado larga."
_NOT_FOUND_MESSAGE_TEMPLATE = "No encontré el archivo '{path}'."


def _open_path(path: Path) -> None:
    """Abrir `path` con su programa asociado por defecto, como si el usuario le hubiera hecho
    doble-click en el Explorador. Nivel de módulo, monkeypatcheable (mismo patrón que
    `_launch_shortcut` en `open_app.py`) — la suite de tests nunca lanza nada real."""
    os.startfile(str(path))


def _normalize_trailing_dots_and_spaces(path_str: str) -> str:
    """Quita puntos/espacios finales del último componente de `path_str` (ej. `"evil.exe."` →
    `"evil.exe"`), replicando a nivel de string lo que Win32 (`CreateFileW`) ya hace de forma
    silenciosa al resolver una ruta contra el filesystem real — ver docstring del módulo, sección
    "Allowlist en vez de denylist", para por qué esto es defensa en profundidad explícita y no la
    única barrera. Pura manipulación de string (`Path.name`/`Path.with_name`), sin ningún I/O.
    No-op si el último componente no termina en punto/espacio, o si tras quitarlos no queda nada
    (ej. un nombre formado solo por puntos) — en ese caso se deja `path_str` sin tocar y la
    resolución/allowlist posteriores lo rechazan por otra vía."""
    candidate = Path(path_str)
    name = candidate.name
    stripped_name = name.rstrip(" .")
    if not stripped_name or stripped_name == name:
        return path_str
    return str(candidate.with_name(stripped_name))


def _candidate_paths(path_str: str, data_dir: Path) -> list[Path]:
    """Construye, en orden, las rutas candidatas para `path_str` — ver docstring del módulo,
    sección "Resolución de ruta". Solo hace joins de `Path` (manipulación de string), nunca toca
    disco ni red — seguro de llamar sobre cualquier `path_str`, incluida una ruta UNC."""
    candidate = Path(path_str)
    if candidate.is_absolute():
        return [candidate]
    return [Path.cwd() / candidate, data_dir / candidate]


def _is_within_directory(candidate: Path, directory: Path) -> bool:
    """`True` si `candidate` cae dentro de `directory` (o es `directory` mismo), comparados de
    forma puramente léxica — sin ningún I/O de filesystem ni de red.

    Usa `Path.absolute()` (que, según su propia documentación, "no realiza normalización ni
    resolución de symlinks", solo antepone el directorio de trabajo actual) más
    `os.path.normpath` (manipulación de string pura) en vez de `Path.resolve()`. Esto es
    deliberado, no un descuido: `Path.resolve()` en Windows usa `_getfinalpathname`
    (`GetFinalPathNameByHandleW` sobre un handle abierto con `CreateFileW`), que abre el archivo
    de verdad — y para una ruta UNC eso dispara negociación de sesión SMB (autenticación NTLM
    incluida) contra el host nombrado en la ruta, el ataque de "forced authentication" del
    Finding 1 documentado en el docstring del módulo. Esta función tiene que poder decidir que una
    ruta UNC cae fuera de `data/` sin tocar la red en absoluto, así que nunca puede depender de
    `.resolve()` para ese primer chequeo — ver `_resolve_within_data_dir` para dónde sí se usa
    `.resolve()` (solo después de que este chequeo léxico ya aprobó la ruta y se confirmó que
    existe localmente)."""
    normalized_candidate = Path(os.path.normpath(str(candidate.absolute())))
    normalized_directory = Path(os.path.normpath(str(directory.absolute())))
    return (
        normalized_candidate == normalized_directory
        or normalized_candidate.is_relative_to(normalized_directory)
    )


def _has_any_candidate_within_data_dir(path_str: str, data_dir: Path) -> bool:
    """`True` si al menos una de las rutas candidatas para `path_str` cae dentro de `data_dir`.
    Puramente léxico (ver `_is_within_directory`) — cero I/O, así que puede (y debe) evaluarse
    antes de cualquier `.exists()`/`.resolve()`. Esta es la jaula de rutas: una ruta UNC, o
    cualquier ruta absoluta/relativa que no resuelva bajo `data/`, siempre da `False` acá, sin
    haber tocado el filesystem ni la red."""
    return any(
        _is_within_directory(candidate, data_dir)
        for candidate in _candidate_paths(path_str, data_dir)
    )


def _resolve_within_data_dir(path_str: str, data_dir: Path) -> Path | None:
    """Entre las candidatas que ya caen dentro de `data_dir` (chequeo léxico, ver
    `_is_within_directory`), devuelve la primera que existe en disco, resuelta con
    `Path.resolve()` — usado acá recién porque en este punto la candidata ya está confirmada
    local y dentro de `data/`, así que abrir un handle sobre ella no es una ruta atacante-
    controlada hacia un host arbitrario. `Path.resolve()` sirve de defensa en profundidad extra
    (además de nombre canónico real, ver docstring del módulo) contra un symlink/`..` que
    pudiera escapar de `data_dir` — se re-verifica la contención sobre la ruta ya resuelta antes
    de aceptarla. `None` si ninguna candidata dentro de la jaula existe."""
    for candidate in _candidate_paths(path_str, data_dir):
        if not _is_within_directory(candidate, data_dir):
            continue
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if _is_within_directory(resolved, data_dir.resolve()):
            return resolved
    return None


class OpenFileTool(Tool):
    """Abre un archivo o carpeta local dentro de la carpeta de datos de JARVIS (`data/`), con el
    programa asociado por defecto de Windows."""

    name = "open_local_file"
    description = (
        "Abre un archivo o carpeta (no un sitio web) con el programa asociado por defecto de "
        "Windows, a partir de su ruta. Usar cuando el usuario pide abrir, ver o mostrar algo "
        "que JARVIS ya generó o guardó antes (ej. 'abrí la captura', 'mostrame esa captura de "
        "pantalla') — la ruta suele ser la que devolvió otra herramienta (ej. screenshot) en un "
        "resultado anterior o en el historial de la conversación. Solo puede abrir archivos "
        "dentro de la carpeta de datos de JARVIS (donde JARVIS guarda lo que genera). No sirve "
        "para abrir aplicaciones por nombre (usá open_app) ni sitios web (usá open_url), y no "
        "abre archivos ejecutables ni scripts."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Ruta del archivo o carpeta a abrir, absoluta o relativa (ej. "
                    "'data/screenshots/screenshot-20260101-120000.png'), tal como apareció en "
                    "un resultado de herramienta o en el historial de la conversación."
                ),
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        path_arg = kwargs.get("path")
        if not isinstance(path_arg, str) or not path_arg.strip():
            return "No se especificó una ruta de archivo válida para abrir."
        path_str = path_arg.strip()

        if len(path_str) > MAX_PATH_LENGTH:
            return _LENGTH_REJECTION_MESSAGE

        path_str = _normalize_trailing_dots_and_spaces(path_str)

        # Chequeo léxico, cero I/O — ver docstring de `_has_any_candidate_within_data_dir`. Esto
        # es lo que rechaza una ruta UNC (Finding 1) antes de que cualquier `.exists()`/
        # `.resolve()` la toque.
        if not _has_any_candidate_within_data_dir(path_str, DEFAULT_DATA_DIR):
            return _OUTSIDE_DATA_DIR_MESSAGE

        resolved = _resolve_within_data_dir(path_str, DEFAULT_DATA_DIR)
        if resolved is None:
            return _NOT_FOUND_MESSAGE_TEMPLATE.format(path=path_str)

        if resolved.is_file() and resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
            return _EXTENSION_REJECTION_MESSAGE

        try:
            _open_path(resolved)
        except OSError as exc:
            return f"Encontré '{resolved}' pero no pude abrirlo ({exc.__class__.__name__})."

        return f"Abriendo {resolved}."
