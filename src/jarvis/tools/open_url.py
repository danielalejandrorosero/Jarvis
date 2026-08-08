"""Tool para abrir sitios web en el navegador (ADR-0005).

Fuerza Chrome específicamente (pedido explícito del usuario: nunca Edge, aunque sea el navegador
default de Windows en la máquina), con fallback al navegador default si Chrome no está instalado
— ver `_find_chrome_executable`/`_open_in_browser`.

Distinto de `jarvis.tools.search.SearchTool`: `SearchTool` busca en la web y devuelve texto para
que el LLM lo lea en voz alta; este tool abre una URL *visualmente*, para que el usuario la mire
(ej. "abrí YouTube", "buscá tal canción en YouTube"). El LLM arma la URL completa él mismo (ver
`SYSTEM_PROMPT` en `jarvis.audio.pipeline`) — este tool no mantiene una tabla de sitio → URL; solo
abre, de forma segura, lo que se le pida.

SAFE bajo `.claude/rules/security.md` — "abrir URLs" está en la lista explícita de ejemplos SAFE,
igual que "abrir aplicaciones" (mismo razonamiento que ya justifica `OpenAppTool`: no muta estado
persistente, es equivalente a que el usuario haga click en un enlace).

Validación antes de abrir (el punto central de este tool, no un detalle accesorio): solo se abren
URLs `http`/`https` con un host presente. Windows registra manejadores de URI-scheme custom para
muchas apps instaladas (`steam://`, algunos `ms-*:`) que pueden dispararse con argumentos
influenciados por el atacante — "URI-scheme argument injection" es una clase de ataque real y
documentada. Restringir a `http`/`https` cierra esa superficie por completo para este tool, y
además es, sencillamente, lo que "una URL a un sitio web" significa para la funcionalidad pedida
(nadie pidió lanzar manejadores de protocolo arbitrarios registrados en el sistema). Se valida con
`urllib.parse.urlsplit` (stdlib) en vez de un regex ad-hoc — parsear la URL es más robusto que
intentar reconocer esquemas peligrosos por patrón.

Hallazgo de `security-reviewer` (medio, corregido acá): sin restricción de host, `open_url` le da
al LLM la primera capacidad de este codebase de disparar una petición de red a un destino que él
mismo elige, con datos arbitrarios en la query string. Encadenado con el riesgo de prompt
injection ya aceptado en `search.py`/`remember.py` (contenido de `<web_data>` o
`<remembered_facts>` intentando manipular al LLM), eso abre dos escenarios concretos: exfiltrar
contexto de la conversación embebido como parámetro hacia un dominio del atacante, o navegar el
navegador del usuario contra una dirección interna/local (`localhost`, IPs privadas, endpoints de
metadata tipo `169.254.169.254`) que confíe en el origen de la petición. `_is_host_allowed`
deniega hosts que resuelven a loopback/privado/link-local o al literal `localhost` — no es un
SSRF-proofing completo (no resuelve DNS para detectar DNS rebinding), pero cierra el caso directo
sin agregar una dependencia nueva ni complejidad de resolución de DNS que esta fase no justifica.
`MAX_URL_LENGTH` es consistencia con `OpenAppTool.MAX_APP_NAME_LENGTH`, no una mitigación de
seguridad autónoma (`security-reviewer` confirmó que una URL larga no da RCE, `webbrowser.open`
en Windows resuelve a `ShellExecuteW`, sin shell de por medio).
"""

from __future__ import annotations

import ipaddress
import logging
import subprocess
import webbrowser
import winreg
from typing import Any, ClassVar
from urllib.parse import urlsplit

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_URL_LENGTH = (
    2000  # generoso para query strings de búsqueda legítimas, ver docstring del módulo
)
_LOCALHOST_NAMES = frozenset({"localhost"})
# Pedido explícito del usuario: siempre Chrome, nunca Edge — aunque Edge sea el navegador
# default de Windows en esta máquina. Se busca vía el registro (App Paths), el mecanismo
# estándar que usa el propio Windows para resolver "chrome.exe" sin asumir una ruta de
# instalación fija (Program Files vs Program Files (x86), instalación por-máquina vs por-usuario).
_CHROME_APP_PATHS_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
)

_REJECTION_MESSAGE = "No puedo abrir esa URL: solo abro sitios web (http/https)."
_HOST_REJECTION_MESSAGE = (
    "No puedo abrir esa URL: apunta a una dirección local o interna."
)
_LENGTH_REJECTION_MESSAGE = "No puedo abrir esa URL: es demasiado larga."


def _find_chrome_executable() -> str | None:
    """Ruta al ejecutable de Chrome, o `None` si no está instalado. Prueba primero la clave de
    máquina (instalación para todos los usuarios) y después la de usuario actual — mismo orden
    que usa Windows internamente al resolver `chrome.exe` para `ShellExecute`."""
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, _CHROME_APP_PATHS_KEY) as key:
                return winreg.QueryValue(key, None)
        except OSError:
            continue
    return None


def _open_in_browser(url: str) -> None:
    """Abrir `url` en Chrome específicamente (pedido del usuario: nunca Edge), con fallback al
    navegador default de Windows si Chrome no está instalado en esta máquina.

    Se lanza Chrome vía `subprocess.Popen` con `url` como argumento propio de la lista (no
    interpolado en un string de shell) — a diferencia de registrar un `webbrowser.GenericBrowser`
    con un comando tipo `"chrome.exe" %s`, que internamente pasa por `os.system` (un shell real);
    con `Popen([chrome_path, url])` no hay shell de por medio, así que no hay riesgo de inyección
    de comando aunque `url` viniera de una fuente no del todo confiable (ya validada de todos
    modos por `_is_valid_web_url`/`_is_host_allowed` antes de llegar acá). Pasarle una URL a
    Chrome como argumento reusa una ventana ya abierta (nueva pestaña) si Chrome ya está
    corriendo, igual que `new=0` en `webbrowser.open`.

    Fire-and-forget del lado del sistema operativo, igual que `os.startfile` en `open_app.py`:
    devuelve el control apenas se lanzó el proceso, sin esperar a que la página cargue ni
    confirmar éxito — por eso `OpenUrlTool.execute()` nunca afirma que la página "se abrió", solo
    que "se está abriendo". Función de nivel de módulo, monkeypatcheable por tests (mismo patrón
    que `_launch_shortcut` en `open_app.py`) para que la suite nunca abra un navegador real.
    """
    chrome_path = _find_chrome_executable()
    if chrome_path is not None:
        subprocess.Popen([chrome_path, url])  # lista de args, sin shell=True
        return
    logger.warning("Chrome no encontrado, usando el navegador default de Windows.")
    webbrowser.open(url, new=0)


def _is_valid_web_url(url: str) -> bool:
    """`True` solo si `url` parsea a un esquema `http`/`https` (case-insensitive) con un host
    presente — ver docstring del módulo para la justificación de por qué se restringe así."""
    split = urlsplit(url)
    return split.scheme.lower() in ALLOWED_SCHEMES and bool(split.netloc)


def _is_host_allowed(url: str) -> bool:
    """`True` salvo que el host de `url` sea `localhost` o una dirección IP loopback/privada/
    link-local — ver docstring del módulo (hallazgo de `security-reviewer`) para el escenario que
    esto cierra. Si el host no es una IP literal (el caso normal, un dominio como youtube.com), no
    se resuelve DNS acá — se permite; esto no es SSRF-proofing completo, es la barrera contra el
    caso directo (URL con una IP/localhost literal escrita por el LLM)."""
    hostname = urlsplit(url).hostname
    if hostname is None:
        return False
    if hostname.lower() in _LOCALHOST_NAMES:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return True  # no es una IP literal (dominio normal) — no se resuelve DNS acá
    return not (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
    )


class OpenUrlTool(Tool):
    """Abre una URL de un sitio web en el navegador por defecto."""

    name = "open_url"
    description = (
        "Abre un sitio web en el navegador para que el usuario lo vea (ej. 'abrí YouTube', "
        "'buscá tal canción en YouTube', 'abrí Wikipedia'). Recibe la URL completa "
        "(https://...) ya armada por vos, incluyendo parámetros de búsqueda si corresponde. "
        "Solo funcionan URLs http/https — no uses esto para buscar información a reportar por "
        "voz, para eso usá search_web."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "URL completa a abrir, ej. 'https://www.youtube.com' o "
                    "'https://www.youtube.com/results?search_query=nombre+de+la+cancion'."
                ),
            }
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url")
        if not isinstance(url, str) or not url.strip():
            return "No se especificó una URL válida para abrir."
        url = url.strip()

        if len(url) > MAX_URL_LENGTH:
            return _LENGTH_REJECTION_MESSAGE

        if not _is_valid_web_url(url):
            return _REJECTION_MESSAGE

        if not _is_host_allowed(url):
            return _HOST_REJECTION_MESSAGE

        try:
            _open_in_browser(url)
        except (OSError, webbrowser.Error) as exc:
            return f"No pude abrir {url} ({exc.__class__.__name__})."

        return f"Abriendo {url} en el navegador."
