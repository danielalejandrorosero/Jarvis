"""Tool para elegir (previsualizar/confirmar) un campeón durante la selección de campeón de
League of Legends, vía la LCU API.

Endpoints usados, ninguno verificado contra un fetch de red en vivo en esta sesión (sin acceso a
internet acá — igual aclaración honesta que en `jarvis.tools.lol_runes`/
`jarvis.tools.lol_summoner_spells`; el usuario está en partida, no en selección de campeón,
mientras se escribe esto, así que tampoco hay sesión de LCU real para probar en vivo):

- `GET /lol-champ-select/v1/session`: estado completo de la selección de campeón actual —
  `localPlayerCellId` (qué jugador sos vos) y `actions` (lista de rondas, cada una una lista de
  acciones `{id, actorCellId, championId, type, completed, ...}`). Documentación no oficial
  ampliamente usada y replicada por proyectos open-source de asistentes de selección de campeón.
- `PATCH /lol-champ-select/v1/session/actions/{action_id}`, cuerpo `{"championId": <id>,
  "completed": <bool>}`: `completed=false` previsualiza (hover) el campeón sin confirmarlo,
  `completed=true` lo confirma/bloquea — mismo endpoint, el campo `completed` es la diferencia
  entre las dos.
- `GET /lol-gameflow/v1/session`: estado general de la partida en curso, incluyendo
  `gameData.queue.gameMode`/`gameData.queue.id` — se usa acá únicamente para *detectar el modo de
  juego actual*, ver más abajo.
- `GET /lol-game-data/assets/v1/champion-summary.json`: listado de campeones que el propio
  cliente sirve localmente (sin depender de una fuente de datos externa ni del conocimiento de
  entrenamiento del LLM sobre qué campeones existen o sus IDs — a diferencia de
  `jarvis.tools.lol_runes`, acá SÍ hay una fuente de datos local confiable y siempre al día con
  el parche actual, así que no hace falta la misma limitación de "puede no reflejar lo último").

## Scoping: por qué esto no aplica a todos los modos de juego

ARAM asigna un campeón al azar por jugador (con intercambio entre compañeros como mecánica
aparte, no elección libre) — ahí no tiene sentido "elegir libremente" un campeón por nombre;
intentarlo de todos modos devolvería, en el mejor caso, un error de la LCU API poco claro para
leer en voz alta, y en el peor, una acción que el modo ni siquiera soporta de la forma en que
este tool asume.

**Corrección en vivo del usuario** (no una suposición de este código): Arena SÍ soporta elegir
campeón — a diferencia de ARAM, tiene una mecánica propia de elección ("Valentía"/"Bravery" en el
cliente: reroll entre un pool ofrecido), pero sigue siendo una decisión del jugador, no una
asignación fija. Por eso Arena NO está en el set de modos sin elección libre acá — ver
`jarvis.league.game_mode` para el detalle completo de qué modo excluye qué tool y por qué.
Aclaración honesta: no se verificó en vivo que el flujo de "Valentía" de Arena exponga sus
acciones a través del mismo shape de `/lol-champ-select/v1/session` (`actions[].type == "pick"`,
`actorCellId`/`completed`) que asume `_find_active_pick_action` — si Arena expone esto distinto,
este tool podría no encontrar ninguna acción activa y degradar a "no tenés una elección de
campeón disponible" en vez de funcionar; no hay forma de confirmar esto sin una sesión de Arena
real, que no está disponible en esta sesión de trabajo.

Detección de modo: delegada a `jarvis.league.game_mode.detect_game_mode` (compartida con
`jarvis.tools.lol_runes`/`jarvis.tools.lol_summoner_spells`, cada uno con su propio criterio de
qué modo excluir) — señal primaria estructural (`gameData.queue.gameMode`), con ID de cola como
fallback documentado. `_detect_free_pick_support` devuelve `None` (distinto de `False`) cuando no
se pudo determinar el modo por ninguna de las dos vías, para que `execute()` nunca reporte "no
podés elegir en este modo" cuando en realidad lo que pasó fue que no se pudo consultar el estado
de la partida — son dos mensajes distintos, uno no debe disfrazarse del otro.

Clasificación (ver ADR-0006 para el razonamiento completo): originalmente un único
`PickChampionTool` con un booleano `lock`, clasificado SAFE para ambos casos. Un
`security-reviewer` marcó que `lock=true` (confirma el pick — en la mayoría de las colas se
puede deshacer por trade con un compañero, pero no siempre, y compromete a 4 personas reales) no
es equivalente en radio de impacto a `lock=false` (hover, trivialmente reversible), y `architect`
resolvió partir el tool en dos en vez de evaluar riesgo condicional por parámetro (ADR-0005
declara `Tool.risk` estático a propósito):

- `PreviewChampionTool` (`preview_lol_champion`) — hover únicamente (`completed=false`). SAFE,
  mismo razonamiento que `jarvis.tools.lol_runes.SetRunesTool`/
  `jarvis.tools.lol_summoner_spells.SetSummonerSpellsTool`: interactúa solo con la API local que
  el propio cliente expone para este caso de uso, equivalente a que el usuario haga click en un
  campeón durante la selección, sin confirmar nada.
- `LockChampionTool` (`lock_lol_champion`) — confirma/bloquea el pick (`completed=true`).
  CONFIRM: requiere confirmación explícita por voz antes de ejecutarse (`describe()` nombra el
  campeón real que se va a lockear, resuelto contra el listado de campeones, no el texto crudo
  dictado por el usuario — mismo motivo que el finding 7 de `security-reviewer` en
  `jarvis.tools.close_app`).

Lógica de matching de nombre, resolución del `actionId` activo, conexión/detección de modo LCU:
compartida a nivel de módulo entre ambos tools — la duplicación aceptable es la clase wrapper
delgada (`execute()`), no la lógica, mismo patrón ya usado entre `jarvis.tools.open_app`/
`jarvis.tools.close_app` (ADR-0006).
"""

from __future__ import annotations

import asyncio
import difflib
from typing import Any, ClassVar

import httpx

from jarvis.league.game_mode import ARAM_GAME_MODE, detect_game_mode
from jarvis.league.lcu_monitor import connect_to_lcu
from jarvis.tools.base import RiskLevel, Tool

CHAMP_SELECT_SESSION_ENDPOINT = "/lol-champ-select/v1/session"
CHAMP_SELECT_ACTION_ENDPOINT_TEMPLATE = (
    "/lol-champ-select/v1/session/actions/{action_id}"
)
CHAMPION_SUMMARY_ENDPOINT = "/lol-game-data/assets/v1/champion-summary.json"

MIN_CHAMPION_NAME_LENGTH = 1
MAX_CHAMPION_NAME_LENGTH = 100

# Mismo cutoff que `jarvis.tools.open_app`/`jarvis.tools.close_app` — mismo balance entre tolerar
# variaciones de nombre dictadas por voz y no volverse tan laxo que cualquier nombre corto
# matchee con cualquier campeón.
_FUZZY_MATCH_CUTOFF = 0.45


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return f"código {response.status_code}"
    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
    return f"código {response.status_code}"


def _detect_free_pick_support(client: httpx.Client) -> bool | None:
    """`True` si el modo de juego actual soporta elegir campeón libremente, `False` si se
    detectó ARAM (el único modo, de los que este tool conoce, que estructuralmente no lo
    soporta — Arena SÍ soporta elegir, ver docstring del módulo), `None` si no se pudo
    determinar el modo (`jarvis.league.game_mode.detect_game_mode`) — ver docstring del módulo
    para por qué estos tres casos no pueden colapsarse en un booleano.
    """
    mode = detect_game_mode(client)
    if mode is None:
        return None
    return mode != ARAM_GAME_MODE


def _find_active_pick_action(session: dict[str, Any]) -> dict[str, Any] | None:
    """Encontrar la acción de tipo "pick" del jugador local que todavía no se confirmó — la
    única acción sobre la que tiene sentido hacer hover/lock ahora mismo. `None` si no hay
    ninguna (todavía no le toca elegir, ya eligió y quedó confirmado, o la sesión no tiene el
    shape esperado)."""
    local_cell_id = session.get("localPlayerCellId")
    if not isinstance(local_cell_id, int):
        return None
    actions = session.get("actions")
    if not isinstance(actions, list):
        return None
    for round_actions in actions:
        if not isinstance(round_actions, list):
            continue
        for action in round_actions:
            if not isinstance(action, dict):
                continue
            if (
                action.get("actorCellId") == local_cell_id
                and action.get("type") == "pick"
                and not action.get("completed")
            ):
                return action
    return None


def _find_best_champion_match(
    champion_name: str, candidates: dict[str, tuple[str, int]]
) -> tuple[str, int] | None:
    """Mismo algoritmo en tres pasos (exacto, substring por cobertura relativa, difflib) que
    `jarvis.tools.open_app._find_best_match`/`jarvis.tools.close_app._find_best_match` —
    duplicado a propósito, no un contrato compartido (ver docstring de esos módulos).

    `candidates` mapea nombre/alias normalizado (minúsculas) -> (nombre canónico, `championId`).
    Devuelve el par `(nombre canónico, championId)` del mejor match, no solo el id — usado tanto
    para el PATCH real (`_select_champion_sync`) como para nombrar el campeón real en el prompt
    de confirmación de `LockChampionTool.describe` (ver `_resolve_champion`)."""
    if not candidates:
        return None
    normalized_target = champion_name.strip().lower()

    if normalized_target in candidates:
        return candidates[normalized_target]

    substring_matches = [
        label
        for label in candidates
        if normalized_target in label or label in normalized_target
    ]
    if substring_matches:

        def _coverage(label: str) -> float:
            shorter = min(len(label), len(normalized_target))
            longer = max(len(label), len(normalized_target))
            return shorter / longer if longer else 0.0

        best_label = max(
            substring_matches, key=lambda label: (_coverage(label), -len(label))
        )
        return candidates[best_label]

    close = difflib.get_close_matches(
        normalized_target, candidates.keys(), n=1, cutoff=_FUZZY_MATCH_CUTOFF
    )
    if not close:
        return None
    return candidates[close[0]]


def _fetch_champion_candidates(
    client: httpx.Client,
) -> dict[str, tuple[str, int]] | None:
    """GET del listado de campeones que el cliente sirve localmente (siempre al día con el
    parche actual — ver docstring del módulo), indexado por nombre/alias normalizado (minúsculas)
    -> (nombre canónico, `championId`). `None` ante cualquier fallo de red/parseo — distinto de
    un dict vacío (conectó bien pero sin candidatos)."""
    try:
        response = client.get(CHAMPION_SUMMARY_ENDPOINT)
    except httpx.HTTPError:
        return None
    if response.status_code != httpx.codes.OK:
        return None
    try:
        champions = response.json()
    except ValueError:
        return None
    if not isinstance(champions, list):
        return None

    candidates: dict[str, tuple[str, int]] = {}
    for entry in champions:
        if not isinstance(entry, dict):
            continue
        champion_id = entry.get("id")
        # El listado incluye una entrada "None"/placeholder con id <= 0 — se descarta, nunca es
        # un campeón real que alguien pueda pedir por voz.
        if not isinstance(champion_id, int) or champion_id <= 0:
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        display_name = name.strip()
        for label in (entry.get("name"), entry.get("alias")):
            if isinstance(label, str) and label.strip():
                candidates.setdefault(
                    label.strip().lower(), (display_name, champion_id)
                )

    return candidates


def _resolve_champion(
    client: httpx.Client, champion_name: str
) -> tuple[str, int] | None:
    """Resolver `champion_name` (texto libre, potencialmente mal transcripto por voz) al nombre
    canónico y `championId` del mejor match. `None` ante cualquier fallo de red/parseo o si no
    hay ningún match razonable. Usado por `_select_champion_sync` (para el PATCH real, con
    `lock` fijo por tool) y por `LockChampionTool.describe` (para nombrar el campeón real en el
    prompt de confirmación — ver su docstring)."""
    candidates = _fetch_champion_candidates(client)
    if candidates is None:
        return None
    return _find_best_champion_match(champion_name, candidates)


def _select_champion_sync(*, champion_name: str, lock: bool) -> str:
    """Cuerpo bloqueante compartido por `PreviewChampionTool.execute`
    (`lock=False`)/`LockChampionTool.execute` (`lock=True`) — corre en un thread aparte, mismo
    patrón que `jarvis.tools.lol_runes._set_runes_sync`. Único punto donde vive la secuencia
    completa (conectar, leer sesión, detectar modo, encontrar acción activa, resolver campeón,
    PATCH) — ver docstring del módulo (ADR-0006): la clase wrapper es lo único que se duplica
    entre los dos tools, no esta lógica."""
    client = connect_to_lcu()
    if client is None:
        return "League of Legends no está corriendo ahora mismo."

    try:
        with client:
            session_response = client.get(CHAMP_SELECT_SESSION_ENDPOINT)
            if session_response.status_code != httpx.codes.OK:
                return "No estás en una selección de campeón activa ahora mismo."
            try:
                session = session_response.json()
            except ValueError:
                return "No pude leer el estado de la selección de campeón."
            if not isinstance(session, dict):
                return "No pude leer el estado de la selección de campeón."

            supports_free_pick = _detect_free_pick_support(client)
            if supports_free_pick is False:
                return (
                    "No podés elegir campeón libremente en este modo de juego — ARAM asigna el "
                    "campeón de otra forma, no por selección libre."
                )
            if supports_free_pick is None:
                return (
                    "No pude determinar el modo de juego actual para saber si podés elegir "
                    "campeón libremente ahora mismo."
                )

            action = _find_active_pick_action(session)
            if action is None:
                return "No tenés una elección de campeón disponible en este momento."

            match = _resolve_champion(client, champion_name)
            if match is None:
                return f"No encontré ningún campeón llamado '{champion_name}'."
            _display_name, champion_id = match

            action_id = action.get("id")
            if not isinstance(action_id, int):
                return "No pude identificar la acción de selección de campeón."

            patch_response = client.patch(
                CHAMP_SELECT_ACTION_ENDPOINT_TEMPLATE.format(action_id=action_id),
                json={"championId": champion_id, "completed": lock},
            )
            if patch_response.status_code >= httpx.codes.BAD_REQUEST:
                detail = _extract_error_detail(patch_response)
                return f"No pude seleccionar a {champion_name}: {detail}."
    except httpx.HTTPError as exc:
        return (
            "No pude conectar con League Client para elegir el campeón "
            f"({exc.__class__.__name__})."
        )

    verb = "Confirmé" if lock else "Preseleccioné"
    return f"{verb} a {champion_name}."


def _describe_lock_target(champion_name: str) -> str:
    """Frase natural para el prompt de confirmación por voz de `LockChampionTool.describe`
    (`jarvis.security.policy.PolicyEngine` arma el prompt final llamando a esto ANTES de
    `execute()`, mismo contrato que `CloseAppTool.describe` — ver su docstring).

    Resuelve `champion_name` contra el listado real de campeones (mismo camino que
    `_select_champion_sync`, vía `_resolve_champion`) para que el prompt nombre el campeón que
    realmente se va a lockear, no el texto crudo dictado por el usuario — mismo motivo que el
    finding 7 de `security-reviewer` en `jarvis.tools.close_app`: el fuzzy-match de
    `_find_best_champion_match` (cutoff 0.45, deliberadamente permisivo para tolerar
    transcripciones garbladas — ver docstring del módulo) podría resolver a un campeón distinto
    del que el usuario cree estar confirmando. Sin este paso, el usuario podría escuchar
    "¿confirmás lockear a Yasuo?", decir que sí, y que en realidad se lockee otro campeón por un
    match ambiguo — exactamente el escenario que motivó partir este tool en ADR-0006.

    A diferencia de `_select_champion_sync`, esto NO valida sesión de selección activa ni modo de
    juego — solo resuelve el nombre del campeón. Si League no está corriendo, no hay lockfile, o
    no hay ningún match razonable, cae de vuelta a nombrar el texto crudo del usuario con una
    aclaración en vez de fallar: el usuario igual necesita poder escuchar un prompt y decir que
    no, y `execute()` va a reportar el motivo específico del fallo si de todos modos confirma.

    Síncrono por contrato de `Tool.describe()` (`base.py`) — mismo patrón y misma excepción
    puntual y documentada a `.claude/rules/python.md` (bloquear una ruta async sin
    `asyncio.to_thread`) que `CloseAppTool.describe`: una única llamada HTTP local, rápida, al
    mismo costo que ya se paga dentro de `execute()`.
    """
    client = connect_to_lcu()
    if client is None:
        return f"'{champion_name}' (no pude conectar con League Client para confirmar cuál es)"

    try:
        with client:
            match = _resolve_champion(client, champion_name)
    except httpx.HTTPError:
        match = None

    if match is None:
        return (
            f"'{champion_name}' (no pude confirmar que ese campeón exista o esté "
            "disponible ahora mismo)"
        )
    display_name, _champion_id = match
    return display_name


class PreviewChampionTool(Tool):
    """Previsualiza (hover) un campeón durante una selección de campeón activa, sin confirmarlo,
    cuando el modo de juego actual soporta elección libre (ver docstring del módulo)."""

    name = "preview_lol_champion"
    description = (
        "Previsualiza (hover) un campeón durante la selección de campeón de League of Legends, "
        "SIN confirmarlo — reversible con solo hacer hover a otro campeón después. Solo "
        "funciona durante una selección de campeón activa Y en modos con elección de campeón "
        "propia (Ranked, Normales, y también Arena vía su mecánica de 'Valentía') — en ARAM el "
        "campeón se asigna al azar y este tool va a avisarte que no aplica en vez de "
        "intentarlo. Usar cuando el usuario pida elegir, previsualizar o hoverear un campeón "
        "SIN decir explícitamente que lo quiere confirmar/lockear (para eso está "
        "lock_lol_champion)."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "champion_name": {
                "type": "string",
                "description": "Nombre del campeón a previsualizar, ej. 'Jinx' o 'Kai'Sa'.",
            },
        },
        "required": ["champion_name"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        champion_name = kwargs.get("champion_name")
        if not isinstance(champion_name, str) or not champion_name.strip():
            return "No se especificó un campeón válido para elegir."
        champion_name = champion_name.strip()
        if not (
            MIN_CHAMPION_NAME_LENGTH <= len(champion_name) <= MAX_CHAMPION_NAME_LENGTH
        ):
            return "El nombre de campeón no tiene una longitud válida."

        return await asyncio.to_thread(
            _select_champion_sync, champion_name=champion_name, lock=False
        )


class LockChampionTool(Tool):
    """Confirma/bloquea un campeón durante una selección de campeón activa, cuando el modo de
    juego actual soporta elección libre (ver docstring del módulo). CONFIRM: requiere
    confirmación explícita por voz antes de ejecutarse (ADR-0006) — `describe()` nombra el
    campeón real que se va a lockear."""

    name = "lock_lol_champion"
    description = (
        "Confirma/bloquea (lock) un campeón durante la selección de campeón de League of "
        "Legends — a diferencia de preview_lol_champion, esto compromete la elección para toda "
        "la partida y no siempre se puede deshacer (depende de que un compañero acepte un "
        "trade). Solo funciona durante una selección de campeón activa Y en modos con elección "
        "de campeón propia (Ranked, Normales, y también Arena vía su mecánica de 'Valentía') — "
        "en ARAM el campeón se asigna al azar y este tool va a avisarte que no aplica en vez de "
        "intentarlo. Usar únicamente cuando el usuario pida explícitamente confirmar, lockear o "
        "bloquear un campeón — requiere confirmación explícita del usuario antes de ejecutarse."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "champion_name": {
                "type": "string",
                "description": "Nombre del campeón a confirmar/lockear, ej. 'Jinx' o 'Kai'Sa'.",
            },
        },
        "required": ["champion_name"],
        "additionalProperties": False,
    }
    # Pedido explícito del usuario: sacar todos los confirmadores salvo SystemPowerTool (ver
    # close_app.py para el motivo completo — confirmación hablada rota en la práctica por
    # audio de juego durante la ventana de escucha). Downgrade deliberado a SAFE: lockear el
    # campeón equivocado ya no tiene red de seguridad hablada.
    risk = RiskLevel.SAFE

    def describe(self, kwargs: dict[str, Any]) -> str:
        """Ver `_describe_lock_target` para el razonamiento completo (por qué resuelve el
        nombre real del campeón en vez de citar el texto crudo del usuario)."""
        champion_name = kwargs.get("champion_name")
        if not isinstance(champion_name, str) or not champion_name.strip():
            return self.name
        champion_name = champion_name.strip()
        target = _describe_lock_target(champion_name)
        return f"lockear a {target} (confirma tu elección para toda la partida)"

    async def execute(self, **kwargs: Any) -> str:
        champion_name = kwargs.get("champion_name")
        if not isinstance(champion_name, str) or not champion_name.strip():
            return "No se especificó un campeón válido para elegir."
        champion_name = champion_name.strip()
        if not (
            MIN_CHAMPION_NAME_LENGTH <= len(champion_name) <= MAX_CHAMPION_NAME_LENGTH
        ):
            return "El nombre de campeón no tiene una longitud válida."

        return await asyncio.to_thread(
            _select_champion_sync, champion_name=champion_name, lock=True
        )
