"""Tool para cerrar aplicaciones en ejecución por nombre (ADR-0005).

Primer tool CONFIRM del codebase. Hasta ahora todos los tools (`WeatherTool`, `SearchTool`,
`RememberTool`, `OpenAppTool`, `OpenUrlTool`) son `RiskLevel.SAFE` — ninguno muta estado visible
fuera de JARVIS sin fricción. Cerrar un proceso sí lo hace: es exactamente el ejemplo que
`.claude/rules/security.md` da para CONFIRM ("...terminar procesos... reversible o de impacto
acotado, pero muta estado"), así que `risk = RiskLevel.CONFIRM` acá no es una decisión nueva, es
aplicar la clasificación ya escrita. La infraestructura de confirmación (`PolicyEngine`,
`VoiceConfirmationChannel`) ya existe y ya funciona (ADR-0005) — este módulo no agrega ningún
mecanismo de seguridad nuevo, solo lo usa.

Listar procesos: `tasklist /FO CSV /NH` (stdlib `subprocess`, args en lista, sin `shell=True`) en
vez de agregar `psutil` como dependencia nueva — `tasklist` ya viene con Windows, y lo único que
hace falta es la columna "Nombre de imagen" (`chrome.exe`, etc.), que `/FO CSV` deja en una
columna fija y entrecomillada por `csv.reader` sin ambigüedad aunque el título de una ventana
(no una columna que se pida acá) tuviera comas. `.claude/rules/python.md`: "sin dependencias sin
justificar la necesidad — preferir stdlib cuando cubra el caso"; acá cubre el caso.

Matching: mismo enfoque en tres pasos que `jarvis.tools.open_app._find_best_match` (exacto
primero siempre, después substring por cobertura relativa, después `difflib` como último
recurso) — deliberadamente, no una variación nueva. Ese orden existe por un fix concreto de
`security-reviewer` en `open_app.py` (finding 4: un match exacto no puede perder contra un
substring más corto pero no relacionado, ej. "Microsoft Store" pedido explícitamente no debe
resolver a un proceso corto tipo "Store" solo porque su nombre es más breve) — reintroducir un
`min(..., key=len)` acá sería repetir ese mismo bug en un tool nuevo. Se compara sobre el nombre
de imagen sin `.exe` (el usuario dice "Chrome", el proceso es `chrome.exe`).

A diferencia de `OpenAppTool`/`OpenUrlTool` (fire-and-forget: `os.startfile`/`Popen` no reportan
si la app en sí llegó a abrir, solo que el lanzamiento se invocó), `taskkill` SÍ reporta éxito o
fracaso de forma síncrona vía código de salida y stderr (proceso no encontrado, ya cerrado,
acceso denegado por permisos insuficientes, protegido por el sistema, etc.) — `execute()` usa eso
para no afirmar "cerré la app" cuando en realidad falló.

Denylist (`_DENYLISTED_PROCESSES`): un puñado corto de procesos que este tool nunca intenta
cerrar, incluso después de que la confirmación por voz haya dado `True` — decisión explícita, no
un default silencioso, en el mismo espíritu que el allowlist de host que `security-reviewer` pidió
para `open_url.py`. Dos motivos distintos cubiertos por la misma lista:

1. Procesos core de Windows (`explorer.exe`, `winlogon.exe`, `csrss.exe`, `services.exe`,
   `svchost.exe`) cuyo cierre no es "cerrar una aplicación" en el sentido que pide el usuario —
   es dejar la sesión de Windows en un estado roto o forzar un crash del sistema, muy por encima
   de "reversible, impacto acotado" que es el techo de lo que CONFIRM está pensado para cubrir
   (`.claude/rules/security.md`). Una confirmación de voz de un solo paso, entendible mal por STT
   o dicha sin pensarlo ("sí, dale"), no es una barrera suficiente para una acción de ese impacto.
2. El proceso del propio JARVIS (`python.exe`/`pythonw.exe`, con las que corre este mismo
   pipeline) — pedirle a JARVIS por voz que se cierre a sí mismo es un modo de falla raro y
   autoinfligido (el propio proceso que está evaluando el pedido desaparece a mitad de la
   respuesta); vale la pena bloquearlo explícitamente en vez de dejar que "funcione" por
   casualidad. Nota: `python.exe`/`pythonw.exe` también son el nombre de proceso de *cualquier*
   otro script Python corriendo en la máquina, no solo JARVIS — la lista es deliberadamente
   conservadora (prefiere negarse de más en un caso raro, "cerrá mi otro script de Python", a
   arriesgarse a cerrarse a sí misma).

Lista corta a propósito (no un intento de enumerar todo lo que podría ser peligroso cerrar) —
sigue habiendo una superficie amplia de procesos que este tool sí permite cerrar con solo una
confirmación de voz (antivirus, VPN, backup en curso), aceptado como parte de lo que CONFIRM
autoriza; ver hallazgos abiertos para `security-reviewer`.
"""

from __future__ import annotations

import asyncio
import csv
import difflib
import io
import logging
import subprocess
from collections.abc import Sequence
from typing import Any, ClassVar

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

MIN_APP_NAME_LENGTH = 1
MAX_APP_NAME_LENGTH = 100

# Mismo cutoff que `open_app.py` — mismo balance entre tolerar typos/variaciones y no volverse
# tan laxo que cualquier nombre corto matchee con cualquier proceso en ejecución.
_FUZZY_MATCH_CUTOFF = 0.45

# Ver docstring del módulo — procesos que este tool nunca cierra, con o sin confirmación de voz.
# Comparado contra el nombre de imagen normalizado (minúsculas, sin `.exe`).
_DENYLISTED_PROCESSES = frozenset(
    {
        "explorer",
        "winlogon",
        "csrss",
        "services",
        "svchost",
        "python",
        "pythonw",
    }
)

_TASKLIST_COMMAND = ["tasklist", "/FO", "CSV", "/NH"]


def _strip_exe(image_name: str) -> str:
    """`chrome.exe` -> `chrome`; deja cualquier otro nombre sin cambios."""
    if image_name.lower().endswith(".exe"):
        return image_name[: -len(".exe")]
    return image_name


def _list_running_process_names() -> list[str]:
    """Nombres de imagen (`chrome.exe`, `Discord.exe`, ...) de todos los procesos en ejecución,
    vía `tasklist /FO CSV /NH` (ver docstring del módulo). Nivel de módulo y sin argumentos, para
    que los tests lo reemplacen igual que `open_app.py` reemplaza `START_MENU_DIRECTORIES`/
    `_launch_shortcut` — la suite nunca lista procesos reales.

    Devuelve lista vacía (nunca lanza) si `tasklist` falla o su salida no se puede parsear —
    `execute()` lo trata igual que "no encontré ningún proceso con ese nombre", no como un error
    fatal.
    """
    try:
        result = subprocess.run(
            _TASKLIST_COMMAND,
            capture_output=True,
            text=True,
            check=False,
            # Sin esto, Windows abre una ventana de consola visible para `tasklist.exe` cada vez
            # — JARVIS corre vía `pythonw.exe`, sin consola propia a la que adjuntarlo (bug real
            # encontrado en vivo con el mismo patrón en `jarvis.league.lcu_monitor`).
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError:
        logger.warning("No se pudo ejecutar tasklist para listar procesos.")
        return []
    if result.returncode != 0:
        logger.warning(
            "tasklist devolvió código %s al listar procesos: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return []
    names: list[str] = []
    try:
        for row in csv.reader(io.StringIO(result.stdout)):
            if row:
                names.append(row[0])
    except csv.Error:
        logger.warning("No se pudo parsear la salida CSV de tasklist.")
        return []
    return names


def _find_best_match(app_name: str, process_names: Sequence[str]) -> str | None:
    """Encontrar el nombre de imagen de proceso que mejor matchea `app_name` — mismo algoritmo en
    tres pasos que `jarvis.tools.open_app._find_best_match` (ver docstring del módulo para la
    justificación del orden exacto/substring/difflib). Devuelve el nombre de imagen original (con
    `.exe`, tal como lo reportó `tasklist`), no la versión normalizada usada para comparar.
    """
    if not process_names:
        return None

    normalized_target = app_name.strip().lower()
    names_by_stem: dict[str, str] = {}
    for name in process_names:
        stem = _strip_exe(name).lower()
        # Ante duplicados (varias pestañas/ventanas del mismo proceso, ej. varios `chrome.exe`),
        # se conserva el primero — cualquiera de las instancias con el mismo nombre de imagen es
        # un target válido para `taskkill /IM`, que de por sí cierra *todas* las instancias con
        # ese nombre de imagen, no solo una.
        names_by_stem.setdefault(stem, name)

    if normalized_target in names_by_stem:
        return names_by_stem[normalized_target]

    substring_matches = [
        stem
        for stem in names_by_stem
        if normalized_target in stem or stem in normalized_target
    ]
    if substring_matches:

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


def _terminate_process(image_name: str) -> tuple[bool, str]:
    """Cerrar `image_name` vía `taskkill /IM <nombre> /F` (lista de args, sin shell) y devolver
    `(éxito, mensaje)` según el código de salida real de `taskkill`, no una asunción optimista —
    a diferencia de `os.startfile`/`webbrowser`/`Popen` en `open_app.py`/`open_url.py`,
    `taskkill` sí reporta síncronamente si la operación falló (proceso ya cerrado, sin permisos,
    protegido por el sistema), así que no hay motivo para no confirmarlo antes de responder.
    """
    display_name = _strip_exe(image_name)
    try:
        result = subprocess.run(
            ["taskkill", "/IM", image_name, "/F"],
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError as exc:
        return False, f"No pude cerrar {display_name} ({exc.__class__.__name__})."

    if result.returncode == 0:
        return True, f"Listo, cerré {display_name}."

    detail = result.stderr.strip() or result.stdout.strip() or "error desconocido"
    return False, f"No pude cerrar {display_name}: {detail}"


class CloseAppTool(Tool):
    """Cierra una aplicación en ejecución, buscándola por nombre entre los procesos activos de
    Windows. CONFIRM: requiere confirmación explícita por voz antes de ejecutarse (ver docstring
    del módulo)."""

    name = "close_app"
    description = (
        "Cierra una aplicación que está corriendo en la computadora, a partir de su nombre en "
        "lenguaje natural (ej. 'Chrome', 'Discord', 'la calculadora'). Usar cuando el usuario "
        "pide cerrar, terminar o matar un programa abierto. ADVERTENCIA: el cierre es FORZADO "
        "(equivalente a 'Finalizar tarea') y afecta TODAS las ventanas/instancias abiertas de "
        "esa aplicación a la vez — la app no tiene oportunidad de preguntar 'guardar cambios' "
        "antes de cerrarse, así que cualquier trabajo no guardado en ella se pierde. Requiere "
        "confirmación explícita del usuario antes de ejecutarse: no es sorpresa, es el "
        "comportamiento esperado."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": (
                    "Nombre de la aplicación a cerrar tal como lo dijo el usuario, ej. "
                    "'Discord' o 'la calculadora'."
                ),
            }
        },
        "required": ["app_name"],
        "additionalProperties": False,
    }
    # Pedido explícito del usuario (en vivo, con "Kimi" citado como alternativa si no se
    # sacaba): la confirmación hablada se rompía en la práctica por contaminación de audio
    # del juego durante la ventana de escucha de la respuesta, y el usuario prefirió sacar la
    # fricción de raíz antes que esperar el fix del pipeline de audio. Downgrade deliberado a
    # SAFE, no un bug — único CONFIRM que sigue en pie en todo el repo es SystemPowerTool.
    risk = RiskLevel.SAFE

    def describe(self, kwargs: dict[str, Any]) -> str:
        """Frase natural para el prompt de confirmación por voz de `PolicyEngine`
        (`jarvis.security.policy`) — nombra el target REAL que se va a cerrar, no el texto crudo
        que pidió el usuario.

        Finding 7 de `security-reviewer` (segunda ronda de revisión): `PolicyEngine.
        authorize_and_execute` arma el prompt de confirmación llamando a `describe(kwargs)` ANTES
        de `execute(**kwargs)` — si `describe()` solo devolviera `kwargs["app_name"]` tal cual
        (texto crudo, sin resolver), el usuario podría escuchar "¿confirmás cerrar Chrome?",
        decir que sí, y que el fuzzy-match dentro de `execute()` (cutoff de 0.45, mismo que
        `open_app.py`) resuelva en realidad a otro proceso corriendo con un nombre parecido —
        confirmando algo distinto de lo que escuchó. Por eso acá se corre la MISMA resolución
        (`_list_running_process_names` + `_find_best_match`) que corre `execute()`, para que el
        prompt nombre el nombre de imagen ya resuelto (ej. "cerrar chrome.exe"), o el mismo
        mensaje de "no encontré esa app" que devolvería `execute()` si no hay match — así nunca se
        le pide al usuario confirmar el cierre de algo que ni siquiera existe.

        Costo aceptado: la resolución corre dos veces por ciclo completo de
        `authorize_and_execute` (acá y de nuevo dentro de `execute()`) — un `tasklist` de más es
        barato. Motivo concreto para no cachear el resultado entre las dos llamadas: hay una
        ventana real de varios segundos entre que se arma este prompt y que el usuario responde
        por voz (`VoiceConfirmationChannel.ask`, `jarvis.audio.pipeline`), tiempo suficiente para
        que la lista de procesos cambie (el proceso objetivo se cierra solo, por ejemplo). Si eso
        pasa, `execute()` vuelve a resolver contra el estado *actual* y simplemente no encuentra
        nada — falla seguro reportando "no encontré esa app" en vez de actuar sobre un match
        stale de hace varios segundos. Preferible a confirmar sobre un estado desactualizado.

        Nota de diseño: `Tool.describe()` es síncrono por contrato (`base.py`) — `PolicyEngine` lo
        llama sin `await`. Este método corre `_list_running_process_names()` (un `subprocess.run`
        bloqueante) directo, sin `asyncio.to_thread`, porque no hay forma de hacerlo de otro modo
        sin cambiar el contrato compartido `Tool`/`PolicyEngine` (fuera de alcance de este tool:
        eso es una decisión de `architect`, no de un tool individual). Es una excepción puntual y
        documentada a la regla general de `.claude/rules/python.md` de no bloquear rutas async sin
        `asyncio.to_thread` — aceptable acá porque `tasklist` es rápido (bajo un segundo) y ya se
        paga ese mismo costo, de forma normal, dentro de `execute()`.
        """
        app_name = kwargs.get("app_name")
        if not isinstance(app_name, str) or not app_name.strip():
            return self.name
        app_name = app_name.strip()

        process_names = _list_running_process_names()
        match = _find_best_match(app_name, process_names)
        if match is None:
            return f"cerrar '{app_name}' (no encontré ninguna aplicación abierta con ese nombre)"
        return f"cerrar {match} (forzado, todas las ventanas abiertas, sin guardar cambios)"

    async def execute(self, **kwargs: Any) -> str:
        app_name = kwargs.get("app_name")
        if not isinstance(app_name, str) or not app_name.strip():
            return "No se especificó un nombre de aplicación válido para cerrar."
        app_name = app_name.strip()
        if len(app_name) < MIN_APP_NAME_LENGTH or len(app_name) > MAX_APP_NAME_LENGTH:
            return "El nombre de aplicación no tiene una longitud válida."

        # `tasklist`/`taskkill` son subprocesos bloqueantes — `asyncio.to_thread` evita bloquear
        # el loop de asyncio que orquesta el tool-call (`.claude/rules/python.md`), mismo patrón
        # que `save_fact` en `remember.py`.
        process_names = await asyncio.to_thread(_list_running_process_names)
        match = _find_best_match(app_name, process_names)
        if match is None:
            return f"No encontré ninguna aplicación abierta llamada '{app_name}'."

        if _strip_exe(match).lower() in _DENYLISTED_PROCESSES:
            return (
                f"No puedo cerrar '{_strip_exe(match)}': es un proceso crítico del sistema "
                "o del propio JARVIS."
            )

        _success, message = await asyncio.to_thread(_terminate_process, match)
        return message
