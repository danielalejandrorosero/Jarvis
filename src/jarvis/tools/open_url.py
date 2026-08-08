"""Tool para abrir sitios web en el navegador por defecto (ADR-0005).

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
import webbrowser
from typing import Any, ClassVar
from urllib.parse import urlsplit

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_URL_LENGTH = (
    2000  # generoso para query strings de búsqueda legítimas, ver docstring del módulo
)
_LOCALHOST_NAMES = frozenset({"localhost"})

_REJECTION_MESSAGE = "No puedo abrir esa URL: solo abro sitios web (http/https)."
_HOST_REJECTION_MESSAGE = (
    "No puedo abrir esa URL: apunta a una dirección local o interna."
)
_LENGTH_REJECTION_MESSAGE = "No puedo abrir esa URL: es demasiado larga."


def _open_in_browser(url: str) -> None:
    """Abrir `url` en el navegador por defecto. `webbrowser.open` es asíncrono/fire-and-forget
    del lado del sistema operativo, igual que `os.startfile` en `open_app.py`: devuelve el
    control apenas se invocó al navegador, sin esperar a que la página cargue ni confirmar que el
    lanzamiento tuvo éxito — por eso `OpenUrlTool.execute()` nunca afirma que la página "se
    abrió", solo que "se está abriendo". Función de nivel de módulo, monkeypatcheable por tests
    (mismo patrón que `_launch_shortcut` en `open_app.py`) para que la suite nunca abra un
    navegador real.

    `new=0` explícito (pedido del usuario): reusar una ventana del navegador ya abierta en vez
    de forzar una nueva — es el default de `webbrowser.open`, pero se deja explícito para que no
    se rompa sin querer en un refactor futuro. Como usa el navegador default de Windows (no una
    instancia separada), ya usa la sesión/cuenta en la que el usuario esté logeado — no hay
    "sesión propia" de JARVIS que pudiera divergir de eso.
    """
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
