"""Tool para configurar una página de runas de League of Legends vía la LCU API.

Endpoint (`POST /lol-perks/v1/pages`, cuerpo `{name, primaryStyleId, subStyleId,
selectedPerkIds, current}`): documentación no oficial pero ampliamente usada y estable desde hace
años, la misma referencia que ya cita `jarvis.league.lcu_monitor` (hextechdocs.dev, "How to set
runes using LCU"). **Aclaración honesta**: esta sesión de trabajo no tiene acceso a un navegador/
fetch de red para releer esa página en vivo — el shape de acá viene de conocimiento de
entrenamiento sobre esa misma documentación (ampliamente replicado por proyectos open-source de
auto-runas), no de una verificación fresca ahora mismo. Antes de depender de esto en producción,
vale la pena un chequeo puntual contra un cliente real (el usuario ya confirmó que hoy está en
partida, no en selección de campeón, así que no hay sesión de LCU real disponible para probar esto
en vivo en este momento).

Reutiliza `jarvis.league.lcu_monitor.connect_to_lcu` para la conexión (lockfile, auth, certificado
autofirmado) — ver docstring de ese módulo. No reimplementa nada de eso acá.

## Decisión de diseño: de dónde salen las runas (punto 1 del pedido)

Dos formas de resolver "poné las runas de tal campeón": (a) un preset local que el usuario va
guardando por voz con otro tool ("guardá estas runas para Jinx"), consultado por nombre de
campeón; o (b) que el propio LLM, con su conocimiento de entrenamiento sobre builds razonables
por campeón, complete directamente los IDs de estilo/runas como parámetros de esta tool, sin
flujo de guardado aparte.

Se eligió (b) para esta primera versión. Motivo: (a) requiere un tool nuevo de "guardar preset"
completo (validación, upsert por campeón, otra tabla en `jarvis.memory.store`) para resolver un
problema que hoy no tiene ningún usuario real todavía — cero presets guardados el primer día,
así que el camino feliz de (a) en la práctica sería casi siempre "no tengo runas guardadas para
ese campeón, decímelas" de todos modos, no muy distinto de simplemente pedírselas al LLM
directamente. (b) cubre el caso de uso pedido ("Alexa, poné las runas de Jinx") sin esa
inversión, a costa de una limitación real y documentada:

**Limitación**: las runas que arma el LLM son tan buenas como su conocimiento de entrenamiento
sobre el meta de League — pueden no reflejar el parche/meta actual (los caminos de runas más
fuertes por campeón cambian con cada parche, y el LLM no tiene forma de saber en qué parche está
el usuario ahora mismo salvo que se lo digan). Es una degradación aceptada para esta primera
versión: sigue siendo mejor que no tener el tool, y no cierra la puerta a agregar un preset
guardado por el usuario (opción (a)) como una iteración futura si esto no alcanza en la práctica.

Clasificación: **SAFE** (`.claude/rules/security.md`) — mismo razonamiento que
`jarvis.league.lcu_monitor.LCUAutoAcceptMonitor`: interactúa solo con la API local que el propio
cliente del juego expone para este caso de uso exacto (el usuario cambiando sus runas antes de la
partida), sin mutar estado de Windows; equivalente a que el usuario lo hiciera él mismo con clicks
en la pantalla de runas del cliente.

## Exclusión por modo de juego: Arena (corrección en vivo del usuario)

Arena reemplazó las páginas de runas pre-partida por Augments, elegidos durante la propia
partida (yunques/prompts in-game), no por `/lol-perks/v1/pages` — el usuario lo confirmó
explícitamente, no es una suposición de este código. `_set_runes_sync` consulta
`jarvis.league.game_mode.detect_game_mode` antes de intentar el POST y se niega con un mensaje
claro si el modo detectado es Arena, en vez de crear una página de runas que el juego va a
ignorar. Si `detect_game_mode` devuelve `None` (no se pudo determinar el modo — red, parseo, o
queue ID no reconocido, ver advertencia en `jarvis.league.game_mode`), este tool también se niega
con un mensaje propio en vez de asumir que NO es Arena y proceder de todos modos — mismo criterio
fail-closed que `jarvis.tools.lol_champion_select.PreviewChampionTool`/`LockChampionTool`.

El usuario también mencionó "ARAM Chaos" como otro modo sin runas pre-partida — **gap conocido,
sin implementar**: no hay un `gameMode`/queue ID confirmado para ese modo distinto del de ARAM
tradicional (que SÍ sigue usando runas normales, y no puede bloquearse por error), y no hay forma
de verificarlo sin acceso a red en esta sesión de trabajo. Ver
`jarvis.league.game_mode` para el detalle completo de esta limitación.

## Borrado de página al liberar cupo: solo páginas propias de JARVIS (hallazgo de seguridad)

`_delete_one_editable_page` (ver su docstring) solo borra páginas cuyo `name` empieza con
`JARVIS_PAGE_NAME_PREFIX` — el mismo prefijo que `execute()` usa para nombrar las páginas que
crea. `isDeletable: true` no alcanza como criterio: lo tienen todas las páginas propias del
usuario, incluidas las que armó y afinó a mano, y borrar la primera que aparezca sin distinguir
de quién es arriesgaba borrar trabajo del usuario sin avisarle. Si el usuario está en el límite
de páginas y ninguna es de JARVIS, el tool no borra nada y le pide que libere cupo manualmente.
"""

from __future__ import annotations

import asyncio
from enum import Enum, auto
from typing import Any, ClassVar

import httpx

from jarvis.league.game_mode import ARENA_GAME_MODE, detect_game_mode
from jarvis.league.lcu_monitor import connect_to_lcu
from jarvis.tools.base import RiskLevel, Tool

LOL_PERKS_PAGES_ENDPOINT = "/lol-perks/v1/pages"

# Prefijo con el que `execute()` nombra toda página que crea (ver `resolved_page_name`) — también
# el único criterio que `_delete_one_editable_page` usa para decidir qué página puede borrar al
# liberar cupo. Nunca se borra una página ajena solo porque esté marcada `isDeletable: true`: esa
# marca la tienen TODAS las páginas propias del usuario (las suyas hechas a mano incluidas), no
# solo las que creó JARVIS — ver hallazgo de seguridad que motivó este filtro.
JARVIS_PAGE_NAME_PREFIX = "JARVIS: "

MIN_CHAMPION_NAME_LENGTH = 1
MAX_CHAMPION_NAME_LENGTH = 100
# Una página de runas real lleva 9 IDs (keystone + 3 del árbol primario + 2 del secundario + 3
# fragmentos), pero se acepta un rango algo más laxo acá — la validación estricta del shape
# correcto la hace el propio League Client al recibir el POST (`_extract_error_detail` reporta su
# respuesta), no hace falta duplicar esa regla de negocio en este tool.
MIN_SELECTED_PERKS = 3
MAX_SELECTED_PERKS = 12


def _extract_error_detail(response: httpx.Response) -> str:
    """Mensaje de error legible a partir de una respuesta de error de la LCU API — el cuerpo
    suele traer `{"message": "..."}`; si no se puede parsear, se cae al código de estado."""
    try:
        data = response.json()
    except ValueError:
        return f"código {response.status_code}"
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
    return f"código {response.status_code}"


class _DeleteOutcome(Enum):
    """Resultado de `_delete_one_editable_page` — tres casos, no un booleano, porque el llamador
    necesita distinguir "no encontré nada para borrar" (sigue con el error original del POST) de
    "encontré páginas borrables pero ninguna es mía" (fail closed: no borra nada ajeno y le avisa
    al usuario en vez de intentar adivinar)."""

    DELETED = auto()
    BLOCKED_NOT_OWNED = auto()
    NONE_AVAILABLE = auto()


def _delete_one_editable_page(client: httpx.Client) -> _DeleteOutcome:
    """Liberar un cupo de página de runas personalizada, si hace falta: el cliente solo permite
    un número limitado de páginas propias (`isDeletable: true`; las páginas preset del juego no
    lo son), así que crear una nueva puede fallar por "límite alcanzado" si ya no queda cupo.

    **Nunca borra una página que no haya creado JARVIS**: `isDeletable: true` lo tienen TODAS las
    páginas propias del usuario, incluidas las que armó y afinó a mano — no hay forma de saber
    desde ese campo solo si una página en particular es reemplazable. Por eso acá se filtra
    también por `name` empezando con `JARVIS_PAGE_NAME_PREFIX` (el mismo prefijo que `execute()`
    usa al crear sus propias páginas) antes de considerar una página candidata a borrado. Si hay
    páginas borrables pero ninguna tiene ese prefijo, se devuelve `BLOCKED_NOT_OWNED` sin borrar
    nada — el usuario está en el límite con páginas que no le pertenecen a JARVIS, y borrar una a
    ciegas sería el problema que este chequeo existe para evitar.

    Se borra como mucho UNA página, y se reintenta el POST una sola vez del lado del llamador.
    Nunca lanza: cualquier fallo de red/parseo se trata como `NONE_AVAILABLE`, el llamador sigue
    con el error original del POST.
    """
    try:
        response = client.get(LOL_PERKS_PAGES_ENDPOINT)
    except httpx.HTTPError:
        return _DeleteOutcome.NONE_AVAILABLE
    if response.status_code != httpx.codes.OK:
        return _DeleteOutcome.NONE_AVAILABLE
    try:
        pages = response.json()
    except ValueError:
        return _DeleteOutcome.NONE_AVAILABLE
    if not isinstance(pages, list):
        return _DeleteOutcome.NONE_AVAILABLE

    found_non_jarvis_deletable_page = False
    for page in pages:
        if not isinstance(page, dict) or not page.get("isDeletable"):
            continue
        name = page.get("name")
        if not isinstance(name, str) or not name.startswith(JARVIS_PAGE_NAME_PREFIX):
            found_non_jarvis_deletable_page = True
            continue
        page_id = page.get("id")
        if not isinstance(page_id, int):
            continue
        try:
            delete_response = client.delete(f"{LOL_PERKS_PAGES_ENDPOINT}/{page_id}")
        except httpx.HTTPError:
            return _DeleteOutcome.NONE_AVAILABLE
        if delete_response.status_code < httpx.codes.BAD_REQUEST:
            return _DeleteOutcome.DELETED
        return _DeleteOutcome.NONE_AVAILABLE

    if found_non_jarvis_deletable_page:
        return _DeleteOutcome.BLOCKED_NOT_OWNED
    return _DeleteOutcome.NONE_AVAILABLE


def _set_runes_sync(
    *,
    champion_name: str,
    primary_style_id: int,
    sub_style_id: int,
    selected_perk_ids: list[int],
    page_name: str,
) -> str:
    """Cuerpo bloqueante de `SetRunesTool.execute` — corre en un thread aparte
    (`asyncio.to_thread`, ver `execute`), igual que `CloseAppTool`/`OpenAppTool` con sus
    subprocesos bloqueantes."""
    client = connect_to_lcu()
    if client is None:
        return "League of Legends no está corriendo ahora mismo."

    body = {
        "name": page_name,
        "primaryStyleId": primary_style_id,
        "subStyleId": sub_style_id,
        "selectedPerkIds": selected_perk_ids,
        "current": True,
    }
    try:
        with client:
            game_mode = detect_game_mode(client)
            if game_mode is None:
                return (
                    "No pude determinar el modo de juego actual para saber si "
                    "corresponde configurar runas ahora mismo."
                )
            if game_mode == ARENA_GAME_MODE:
                return (
                    "Arena no usa páginas de runas: los Augments que elegís durante la "
                    "partida reemplazan las runas pre-partida en ese modo."
                )
            response = client.post(LOL_PERKS_PAGES_ENDPOINT, json=body)
            # Probablemente el límite de páginas propias — liberar una (solo si es de JARVIS) y
            # reintentar una vez.
            if response.status_code >= httpx.codes.BAD_REQUEST:
                outcome = _delete_one_editable_page(client)
                if outcome is _DeleteOutcome.DELETED:
                    response = client.post(LOL_PERKS_PAGES_ENDPOINT, json=body)
                elif outcome is _DeleteOutcome.BLOCKED_NOT_OWNED:
                    return (
                        "Ya tenés el máximo de páginas de runas y ninguna es mía para "
                        "reemplazar — borrá una manualmente desde el cliente e intentá "
                        "de nuevo."
                    )
            if response.status_code >= httpx.codes.BAD_REQUEST:
                detail = _extract_error_detail(response)
                return f"No pude configurar las runas de {champion_name}: {detail}."
    except httpx.HTTPError as exc:
        return (
            "No pude conectar con League Client para configurar las runas "
            f"({exc.__class__.__name__})."
        )
    return f"Listo, configuré las runas de {champion_name}."


class SetRunesTool(Tool):
    """Configura una página de runas en League of Legends a partir de los IDs de estilo/runas
    que indique el LLM (ver docstring del módulo para la decisión de diseño y su limitación)."""

    name = "set_lol_runes"
    description = (
        "Configura una página de runas en League of Legends (menú de runas del cliente). Solo "
        "funciona mientras League of Legends está corriendo — antes de una partida, en el "
        "lobby o en selección de campeón. No aplica en el modo Arena (ese modo usa Augments "
        "durante la partida en vez de runas pre-partida) — si el usuario está en Arena, este "
        "tool va a avisarte que no corresponde en vez de intentarlo. Vos (el LLM) elegís "
        "primary_style_id, sub_style_id y selected_perk_ids a partir de tu propio conocimiento "
        "de builds razonables para el campeón pedido; esto puede no reflejar el meta del parche "
        "actual del juego, avisale al usuario si te parece relevante. Usar cuando el usuario "
        "pida poner/configurar runas para un campeón."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "champion_name": {
                "type": "string",
                "description": "Nombre del campeón para el que son estas runas, ej. 'Jinx'.",
            },
            "primary_style_id": {
                "type": "integer",
                "description": (
                    "ID del árbol de runas primario (ej. 8000 Precisión, 8100 Dominación, 8200 "
                    "Brujería, 8300 Inspiración, 8400 Determinación)."
                ),
            },
            "sub_style_id": {
                "type": "integer",
                "description": "ID del árbol de runas secundario (mismos valores posibles que primary_style_id).",
            },
            "selected_perk_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "IDs de las runas elegidas, en orden: keystone, las 3 runas menores del "
                    "árbol primario, las 2 runas del árbol secundario, y los 3 fragmentos "
                    "(stat shards) — 9 IDs en total."
                ),
            },
            "page_name": {
                "type": "string",
                "description": "Nombre para la página de runas (opcional; por defecto el nombre del campeón).",
            },
        },
        "required": [
            "champion_name",
            "primary_style_id",
            "sub_style_id",
            "selected_perk_ids",
        ],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        champion_name = kwargs.get("champion_name")
        if not isinstance(champion_name, str) or not champion_name.strip():
            return "No se especificó un campeón válido para configurar las runas."
        champion_name = champion_name.strip()
        if not (
            MIN_CHAMPION_NAME_LENGTH <= len(champion_name) <= MAX_CHAMPION_NAME_LENGTH
        ):
            return "El nombre de campeón no tiene una longitud válida."

        primary_style_id = kwargs.get("primary_style_id")
        sub_style_id = kwargs.get("sub_style_id")
        if (
            not isinstance(primary_style_id, int)
            or isinstance(primary_style_id, bool)
            or primary_style_id <= 0
        ):
            return "No se especificó un árbol de runas primario válido."
        if (
            not isinstance(sub_style_id, int)
            or isinstance(sub_style_id, bool)
            or sub_style_id <= 0
        ):
            return "No se especificó un árbol de runas secundario válido."

        selected_perk_ids = kwargs.get("selected_perk_ids")
        if not isinstance(selected_perk_ids, list) or not (
            MIN_SELECTED_PERKS <= len(selected_perk_ids) <= MAX_SELECTED_PERKS
        ):
            return "No se especificó una lista válida de runas para la página."
        if not all(
            isinstance(perk_id, int) and not isinstance(perk_id, bool) and perk_id > 0
            for perk_id in selected_perk_ids
        ):
            return "No se especificó una lista válida de runas para la página."

        page_name = kwargs.get("page_name")
        resolved_page_name = (
            page_name.strip()
            if isinstance(page_name, str) and page_name.strip()
            else f"{JARVIS_PAGE_NAME_PREFIX}{champion_name}"
        )

        # `httpx.Client` (síncrono) + descubrimiento de lockfile son bloqueantes —
        # `asyncio.to_thread` evita bloquear el loop de asyncio (`.claude/rules/python.md`),
        # mismo patrón que `CloseAppTool.execute`.
        return await asyncio.to_thread(
            _set_runes_sync,
            champion_name=champion_name,
            primary_style_id=int(primary_style_id),
            sub_style_id=int(sub_style_id),
            selected_perk_ids=[int(perk_id) for perk_id in selected_perk_ids],
            page_name=resolved_page_name,
        )
