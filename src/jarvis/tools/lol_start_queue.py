"""Tools para arrancar la búsqueda de partida (matchmaking) de League of Legends vía la LCU API.

Endpoints usados:

- `GET /lol-gameflow/v1/gameflow-phase` (mismo endpoint que ya usa
  `jarvis.league.lcu_monitor.LCUAutoAcceptMonitor._poll_gameflow` para su propio polling — se
  reutiliza la misma constante `GAMEFLOW_PHASE_ENDPOINT` de ese módulo en vez de redeclararla):
  devuelve un string plano (`response.json()` es directamente `"Lobby"`, `"Matchmaking"`,
  `"ReadyCheck"`, etc. — no un objeto), se consulta acá antes de intentar nada, para dar un
  mensaje claro y específico según en qué fase esté el usuario en vez de un error genérico de la
  LCU API.
- `POST /lol-lobby/v2/lobby`, cuerpo `{"queueId": <id>}` — solo usado por `SetLobbyQueueTool`:
  arma un lobby nuevo para esa cola, o reemplaza el que ya exista. Documentación no oficial
  ampliamente replicada (mismo criterio de confianza relativa que
  `jarvis.tools.lol_runes`/`jarvis.tools.lol_champion_select`, no verificado en vivo esta
  sesión).
- `GET /lol-gameflow/v1/session` — vía `jarvis.league.game_mode.detect_game_mode` (compartido
  con `jarvis.tools.lol_runes`/`jarvis.tools.lol_summoner_spells`/
  `jarvis.tools.lol_champion_select`), usado por `SetLobbyQueueTool` como lectura de confirmación
  DESPUÉS del `POST /lol-lobby/v2/lobby` de arriba — ver "Finding 2" más abajo para el porqué.
- `POST /lol-lobby/v2/lobby/matchmaking/search`, cuerpo vacío: arranca la búsqueda de partida
  para el lobby en el que el usuario ya está. **Verificado en vivo esta noche contra el cliente
  real del usuario** (no documentación no oficial sin confirmar, a diferencia de la salvedad que
  sí aplica a `jarvis.tools.lol_runes`/`jarvis.tools.lol_summoner_spells`/
  `jarvis.tools.lol_champion_select`): un script puntual confirmó que este POST con cuerpo vacío
  devuelve `204` y dispara la transición de fase `Lobby` → `Matchmaking` → `ReadyCheck` como se
  espera. `CancelQueueTool` (más abajo) usa el `DELETE` del mismo endpoint — no verificado en
  vivo de la misma forma, pero es la operación inversa documentada del mismo par de endpoints, así
  que se aplica el mismo criterio de confianza relativa que el resto de los tools de League.

## Historia: por qué esto son dos tools, no uno con un parámetro opcional (ADR-0006)

Versión original de `StartQueueTool` (sin elegir cola): arrancar la búsqueda de partida para el
lobby en el que el usuario ya estaba sentado, igual que el botón "BUSCAR PARTIDA" del cliente —
nunca crea ni reemplaza un lobby.

Gap real encontrado en vivo: sin forma de elegir cola, el tool no podía resolver "cambiate de
Arena a Ranked Solo/Dúo y buscá" — solo podía re-buscar en la cola del lobby en el que el usuario
ya estaba. Un parámetro opcional `queue_type` se agregó para cerrar ese gap, con
`POST /lol-lobby/v2/lobby` (arma o reemplaza el lobby actual) antes del POST de búsqueda —
clasificado SAFE en ese momento bajo el argumento de que armar/reemplazar un lobby es equivalente
a que el usuario haga click en esa cola dentro del cliente.

Un `security-reviewer` marcó esto como hallazgo bloqueante (finding 1, HIGH): a diferencia de
click manual del usuario, el tool no verifica ni avisa si había compañeros ya invitados al lobby
que se está reemplazando — una mutación no trivialmente reversible desde la perspectiva de esos
compañeros, estructuralmente el mismo problema que motivó partir `PickChampionTool` en
`PreviewChampionTool`/`LockChampionTool` (ADR-0006: "un parámetro que cambia el radio de impacto
real de la acción... se parte el tool en dos, cada uno con su propio `RiskLevel` fijo" — no se
evalúa riesgo condicional por parámetro). Mismo remedio acá:

- `StartQueueTool` (`start_lol_queue`) — vuelve a su scope original: SAFE, **sin** parámetro
  `queue_type`, busca en el lobby ya armado sin tocar la cola. No crea ni reemplaza ningún lobby.
- `SetLobbyQueueTool` (`set_lol_lobby_queue`) — nuevo tool, CONFIRM: arma o reemplaza el lobby
  actual con una cola concreta (`queue_type`, obligatorio acá) y, si la cola resultante confirma
  coincidir con la pedida (ver "Finding 2"), arranca la búsqueda en el mismo paso — un solo
  `describe()`/confirmación para la intención completa del usuario ("cambiate a Ranked y buscá"),
  en vez de pedir confirmación dos veces para una sola intención. `describe()` avisa
  explícitamente que compañeros ya invitados al lobby actual podrían verse afectados (mismo
  motivo que el finding 7 de `security-reviewer` en `jarvis.tools.close_app`/ADR-0006 en
  `jarvis.tools.lol_champion_select`: el usuario tiene que poder escuchar el prompt y decir que
  no con la información real relevante, no un genérico "¿confirmás cambiar de cola?").

Lógica compartida a nivel de módulo entre ambos tools (resolución de `queue_type`, consulta de
fase, extracción de detalle de error): la duplicación aceptable es la clase wrapper delgada
(`execute()`/`describe()`), no la lógica — mismo patrón que
`jarvis.tools.open_app`/`jarvis.tools.close_app` y
`jarvis.tools.lol_champion_select.PreviewChampionTool`/`LockChampionTool` (ADR-0006).

## Finding 2 (MEDIUM): por qué `SetLobbyQueueTool` relee el modo resultante antes de buscar

`jarvis.league.game_mode` documenta el `queueId` de Arena como un conjunto incierto de tres
valores históricos (`ARENA_QUEUE_IDS`) porque Riot lo cambió más de una vez; acá se pide `1700`
(el más reciente de los tres conocidos) al crear el lobby, sin confirmación en vivo de que sea el
vigente ahora mismo. Si Riot reasignó `1700` a una cola *distinta mente válida* (no simplemente
inválida), la LCU no rechaza el `POST /lol-lobby/v2/lobby` con un 400 limpio — arma en silencio un
lobby de un modo equivocado, sin ningún error que reportar.

Por eso, después de que el `POST /lol-lobby/v2/lobby` confirma éxito, `SetLobbyQueueTool` hace una
lectura de confirmación (`jarvis.league.game_mode.detect_game_mode`, mismo endpoint que ya usan
`jarvis.tools.lol_runes`/`jarvis.tools.lol_summoner_spells`/`jarvis.tools.lol_champion_select`
para detectar modo) y compara el `gameMode` resultante contra `_EXPECTED_GAME_MODE[queue_type]`
antes de arrancar la búsqueda — si no coincide, se reporta un mensaje claro de discrepancia en vez
de confirmar éxito sobre un modo equivocado. `_EXPECTED_GAME_MODE` agrupa `ranked_solo_duo`/
`normal_draft`/`normal_blind` bajo el mismo `"CLASSIC"` (las tres son colas 5v5 de Grieta del
Invocador con el mismo `gameMode` interno — este chequeo no puede distinguirlas entre sí, ninguna
de las tres tiene la ambigüedad de `queueId` que sí tiene Arena, ver `game_mode.py`), así que el
chequeo es proporcional al problema real: confirma que no se terminó en un modo totalmente
distinto del pedido (ARAM, Arena, Grieta), no una reconciliación exhaustiva por `queueId`.
`"CLASSIC"` como valor de `gameMode` para colas de Grieta del Invocador es conocimiento de
entrenamiento sobre la API pública de Riot (match-v5), mismo nivel de confianza que `"CHERRY"`
para Arena en `game_mode.py` — no verificado en vivo esta sesión.

## Caso "ya está buscando partida" (`Matchmaking`)

Si se pide `set_lol_lobby_queue` mientras ya hay una búsqueda en curso, el tool se niega con un
mensaje explícito pidiendo cancelar primero (`cancel_lol_queue`), en vez de mandar el
`POST /lol-lobby/v2/lobby` de todos modos — la propia UI del cliente deshabilita cambiar de lobby
mientras se está buscando (el único control activo en ese momento es cancelar), así que lo más
probable es que ese POST devuelva un error de la LCU API confuso de leer en voz alta en vez de
funcionar; no verificado en vivo esta sesión, pero degradar a un mensaje claro y accionable es
preferible a intentarlo a ciegas y improvisar sobre un error sin confirmar.

## Por qué no espera el ready-check

`LCUAutoAcceptMonitor` (`jarvis.league.lcu_monitor`) ya es responsable de aceptar el ready-check
en segundo plano — corre siempre, independiente de que el usuario haya usado estos tools o
apretado el botón él mismo. Ambos tools acá son acciones de voz síncronas (`execute()` la llama
el usuario y espera una respuesta hablada en segundos, no en minutos) — su trabajo termina en
cuanto el último `POST` se confirma, no deben quedarse sondeando `gameflow-phase` a la espera de
que aparezca gente para la partida.

Clasificación: `StartQueueTool` es **SAFE** (`.claude/rules/security.md`) — mismo razonamiento
que `jarvis.tools.lol_runes.SetRunesTool`/`jarvis.league.lcu_monitor.LCUAutoAcceptMonitor`:
interactúa solo con la API local que el propio cliente expone para este caso de uso exacto,
equivalente a que el usuario apriete el botón "Buscar partida" él mismo. Reversible:
`CancelQueueTool` (o el propio cliente) puede cancelar la búsqueda antes de que aparezca un
ready-check. `SetLobbyQueueTool` es **CONFIRM** (ver "Historia" arriba, ADR-0006).
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import httpx

from jarvis.league.game_mode import ARAM_GAME_MODE, ARENA_GAME_MODE, detect_game_mode
from jarvis.league.lcu_monitor import GAMEFLOW_PHASE_ENDPOINT, connect_to_lcu
from jarvis.tools.base import RiskLevel, Tool

LOBBY_ENDPOINT = "/lol-lobby/v2/lobby"
MATCHMAKING_SEARCH_ENDPOINT = "/lol-lobby/v2/lobby/matchmaking/search"

LOBBY_PHASE = "Lobby"
NO_LOBBY_PHASE = "None"
MATCHMAKING_PHASE = "Matchmaking"

# `gameMode` interno que reporta la LCU para las colas 5v5 tradicionales de Grieta del Invocador
# — mismo valor para `ranked_solo_duo`/`normal_draft`/`normal_blind` (ver "Finding 2" en el
# docstring del módulo para el porqué de agruparlas).
#
# Residual conocido, no bloqueante (confirmado por security-reviewer): al chequear solo
# `gameMode` y no `queueId`, el read-back de `_set_lobby_queue_sync` NO puede distinguir estas
# tres colas entre sí — si se pide `ranked_solo_duo` y la LCU arma `normal_draft` por el motivo
# que sea, el chequeo ve `"CLASSIC" == "CLASSIC"` y confirma éxito igual, iniciando la búsqueda en
# la cola equivocada sin avisar. Aceptado porque, a diferencia de Arena, estos tres `queueId`
# están verificados contra la fuente oficial de Riot (sin la incertidumbre que motivó el finding
# original) y el radio de impacto de "quedó en Grieta pero en la cola equivocada" es menor que el
# de Arena/ARAM. Si en el futuro esto importa, `game_mode.detect_game_mode` ya parsea `queueId`
# internamente pero no lo expone — extenderlo para comparar por `queueId` exacto cerraría esta
# brecha más fina.
_CLASSIC_GAME_MODE = "CLASSIC"

# Valores de `queue_type` que aceptan `StartQueueTool`/`SetLobbyQueueTool` -> (`queueId`
# resuelto, nombre legible para los mensajes por voz, `gameMode` esperado tras crear el lobby —
# ver "Finding 2"). `queueId` verificados contra la fuente estática oficial de Riot
# (`https://static.developer.riotgames.com/docs/lol/queues.json`) el mismo día que se agregó este
# parámetro, salvo Arena: ese endpoint no cubre modos rotativos/de temporada como Arena, así que
# `1700` viene de `jarvis.league.game_mode.ARENA_QUEUE_IDS` (el conjunto de IDs históricos que ese
# módulo ya documenta con la misma advertencia de "Riot cambió el ID de Arena más de una vez") —
# se usa `1700` como el que se pide al crear el lobby porque es el más reciente de los tres
# conocidos, no por una confirmación en vivo puntual de que sea el vigente ahora mismo.
_QUEUE_TYPES: dict[str, tuple[int, str, str]] = {
    "ranked_solo_duo": (420, "Ranked Solo/Dúo", _CLASSIC_GAME_MODE),
    "normal_draft": (400, "Normales (selección por turnos)", _CLASSIC_GAME_MODE),
    "normal_blind": (430, "Normales (selección a ciegas)", _CLASSIC_GAME_MODE),
    "aram": (450, "ARAM", ARAM_GAME_MODE),
    "arena": (1700, "Arena", ARENA_GAME_MODE),
}

# Mensajes específicos por fase distinta de `Lobby` — un mensaje genérico ("no se puede ahora")
# sería técnicamente correcto pero mucho menos útil por voz que decirle al usuario exactamente en
# qué fase está y por qué. `None` (fase "None": el usuario no está en ningún lobby/partida) tiene
# su propio mensaje porque la causa y la solución son distintas ("armá un lobby primero") de estar
# ya en una fase de partida en curso.
_PHASE_MESSAGES: dict[str, str] = {
    "None": "No estás en ningún lobby ahora mismo — armá uno primero desde el cliente.",
    "Matchmaking": "Ya estás buscando partida — no hace falta arrancar de nuevo.",
    "ReadyCheck": "Ya se encontró una partida y hay un ready check esperando — no hace falta buscar de nuevo.",
    "ChampSelect": "Ya estás en la selección de campeón — la búsqueda de partida ya terminó.",
    "InProgress": "Ya estás en una partida en curso ahora mismo.",
    "Reconnect": "Tenés una partida en curso a la que reconectarte — la búsqueda no aplica ahora.",
    "WaitingForStats": "La partida ya terminó y estás esperando las estadísticas — no hace falta buscar ahora.",
    "PreEndOfGame": "La partida ya terminó — no hace falta buscar ahora.",
    "EndOfGame": "La partida ya terminó — no hace falta buscar ahora.",
    "GameStart": "La partida ya está arrancando — no hace falta buscar de nuevo.",
    "FailedToLaunch": "Hubo un problema lanzando la partida anterior — revisá el cliente antes de buscar de nuevo.",
    "TerminatedInError": "La partida anterior terminó con un error — revisá el cliente antes de buscar de nuevo.",
    "CheckedIntoTournament": "Estás registrado en un torneo ahora mismo — la búsqueda normal no aplica.",
}


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


def _resolve_queue_type(queue_type: str) -> tuple[int, str, str] | None:
    """Resolver un `queue_type` (string libre pasado por el LLM) contra `_QUEUE_TYPES`. `None`
    si no matchea ningún valor conocido — el llamador trata eso como error de validación y nunca
    llega a conectar con la LCU API (ver docstring del módulo: valores inválidos se rechazan antes
    de la red, no con un error de la LCU)."""
    return _QUEUE_TYPES.get(queue_type.strip().lower())


def _query_gameflow_phase(client: httpx.Client) -> str | None:
    """Consultar la fase actual de partida. `None` ante cualquier fallo de red/parseo o si la
    respuesta no es el string plano esperado — el llamador trata `None` como "no pude determinar
    la fase", distinto de una fase conocida pero no `Lobby`."""
    try:
        response = client.get(GAMEFLOW_PHASE_ENDPOINT)
    except httpx.HTTPError:
        return None
    if response.status_code != httpx.codes.OK:
        return None
    try:
        phase = response.json()
    except ValueError:
        return None
    if not isinstance(phase, str) or not phase.strip():
        return None
    return phase.strip()


def _start_queue_sync() -> str:
    """Cuerpo bloqueante de `StartQueueTool.execute` — corre en un thread aparte, mismo patrón
    que `jarvis.tools.lol_runes._set_runes_sync`. Nunca crea ni reemplaza un lobby: busca
    únicamente en el lobby en el que el usuario ya esté sentado (ver docstring del módulo, para
    cambiar de cola primero está `SetLobbyQueueTool`/`set_lol_lobby_queue`)."""
    client = connect_to_lcu()
    if client is None:
        return "League of Legends no está corriendo ahora mismo."

    try:
        with client:
            phase = _query_gameflow_phase(client)
            if phase is None:
                return "No pude determinar en qué fase del juego estás ahora mismo."
            if phase != LOBBY_PHASE:
                return _PHASE_MESSAGES.get(
                    phase,
                    f"No podés buscar partida ahora mismo — estás en la fase '{phase}'.",
                )

            response = client.post(MATCHMAKING_SEARCH_ENDPOINT)
            if response.status_code >= httpx.codes.BAD_REQUEST:
                detail = _extract_error_detail(response)
                return f"No pude arrancar la búsqueda de partida: {detail}."
    except httpx.HTTPError as exc:
        return (
            "No pude conectar con League Client para arrancar la búsqueda de partida "
            f"({exc.__class__.__name__})."
        )
    return "Listo, buscando partida."


def _set_lobby_queue_sync(*, queue_type: str) -> str:
    """Cuerpo bloqueante de `SetLobbyQueueTool.execute` — corre en un thread aparte, mismo patrón
    que `_start_queue_sync`. Arma o reemplaza el lobby actual con la cola pedida, confirma (ver
    "Finding 2" en el docstring del módulo) que el modo resultante coincide con el pedido, y solo
    entonces arranca la búsqueda — un único paso CONFIRM'd para la intención completa del
    usuario, ver docstring del módulo ("Historia")."""
    resolved = _resolve_queue_type(queue_type)
    if resolved is None:
        valid_values = ", ".join(sorted(_QUEUE_TYPES))
        return (
            f"No reconozco el tipo de cola '{queue_type}' — los valores válidos son: "
            f"{valid_values}."
        )
    queue_id, queue_label, expected_mode = resolved

    client = connect_to_lcu()
    if client is None:
        return "League of Legends no está corriendo ahora mismo."

    try:
        with client:
            phase = _query_gameflow_phase(client)
            if phase is None:
                return "No pude determinar en qué fase del juego estás ahora mismo."
            if phase == MATCHMAKING_PHASE:
                return (
                    "Ya estás buscando partida con la cola actual — cancelá la "
                    f"búsqueda primero (cancelar la cola) antes de cambiar a {queue_label}."
                )
            if phase not in (LOBBY_PHASE, NO_LOBBY_PHASE):
                return _PHASE_MESSAGES.get(
                    phase,
                    f"No podés cambiar de cola ahora mismo — estás en la fase '{phase}'.",
                )

            lobby_response = client.post(LOBBY_ENDPOINT, json={"queueId": queue_id})
            if lobby_response.status_code >= httpx.codes.BAD_REQUEST:
                detail = _extract_error_detail(lobby_response)
                return f"No pude armar el lobby de {queue_label}: {detail}."

            resulting_mode = detect_game_mode(client)
            if resulting_mode is None:
                return (
                    f"Armé un lobby pero no pude confirmar que sea de {queue_label} — "
                    "revisá el cliente antes de buscar partida."
                )
            if resulting_mode != expected_mode:
                return (
                    f"El lobby que se armó no coincide con {queue_label} (la cola resultante "
                    "no es la que pediste) — no arranqué la búsqueda, revisá el cliente."
                )

            response = client.post(MATCHMAKING_SEARCH_ENDPOINT)
            if response.status_code >= httpx.codes.BAD_REQUEST:
                detail = _extract_error_detail(response)
                return f"No pude arrancar la búsqueda de partida: {detail}."
    except httpx.HTTPError as exc:
        return (
            "No pude conectar con League Client para armar el lobby y arrancar la búsqueda "
            f"({exc.__class__.__name__})."
        )
    return f"Listo, armé el lobby de {queue_label} y arranqué la búsqueda de partida."


def _cancel_queue_sync() -> str:
    """Cuerpo bloqueante de `CancelQueueTool.execute` — mismo patrón que `_start_queue_sync`, sin
    el chequeo de fase previo: cancelar una búsqueda que no existe es una operación idempotente
    (el propio cliente la trata como no-op o devuelve un error claro), no hace falta duplicar acá
    la misma lógica de fases que `_start_queue_sync`."""
    client = connect_to_lcu()
    if client is None:
        return "League of Legends no está corriendo ahora mismo."

    try:
        with client:
            response = client.delete(MATCHMAKING_SEARCH_ENDPOINT)
            if response.status_code >= httpx.codes.BAD_REQUEST:
                detail = _extract_error_detail(response)
                return f"No pude cancelar la búsqueda de partida: {detail}."
    except httpx.HTTPError as exc:
        return (
            "No pude conectar con League Client para cancelar la búsqueda de partida "
            f"({exc.__class__.__name__})."
        )
    return "Listo, cancelé la búsqueda de partida."


class StartQueueTool(Tool):
    """Arranca la búsqueda de partida (matchmaking) de League of Legends para el lobby en el que
    el usuario ya esté sentado — no lo crea ni le cambia la cola (ver docstring del módulo, para
    eso está `SetLobbyQueueTool`)."""

    name = "start_lol_queue"
    description = (
        "Arranca la búsqueda de partida (botón 'Buscar partida') en League of Legends, para el "
        "lobby en el que el usuario ya esté armado (cualquier cola: Ranked, Normales, ARAM, "
        "Arena, etc.), sin tocar la cola actual ni crear/reemplazar ningún lobby — si no está "
        "en ningún lobby, o ya está buscando/en selección/en partida, devuelve un mensaje claro "
        "explicando por qué no corresponde en vez de intentarlo. Para armar o cambiar la cola "
        "antes de buscar (ej. 'cambiate a ranked y buscá') usar set_lol_lobby_queue en vez de "
        "este tool. Usar cuando el usuario pida buscar partida o arrancar a jugar sin mencionar "
        "una cola concreta."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(_start_queue_sync)


class SetLobbyQueueTool(Tool):
    """Arma o reemplaza el lobby actual de League of Legends con una cola concreta y arranca la
    búsqueda de partida — CONFIRM (ADR-0006): reemplazar el lobby actual puede afectar a
    compañeros ya invitados a él, ver docstring del módulo."""

    name = "set_lol_lobby_queue"
    description = (
        "Arma o reemplaza el lobby actual de League of Legends con una cola concreta "
        "('ranked_solo_duo', 'normal_draft', 'normal_blind', 'aram' o 'arena') y, si la cola "
        "resultante confirma coincidir con la pedida, arranca la búsqueda de partida en el "
        "mismo paso. Funciona incluso si el usuario no está en ningún lobby todavía, salvo que "
        "ya esté buscando partida (ahí hay que cancelar la búsqueda primero con "
        "cancel_lol_queue) o esté en una fase donde no se puede tocar el lobby (selección de "
        "campeón, partida en curso, etc.). Usar cuando el usuario pida cambiar o elegir "
        "explícitamente una cola concreta (ej. 'cambiate a ARAM y buscá', 'armá una de Arena'). "
        "Requiere confirmación explícita del usuario antes de ejecutarse, porque reemplazar el "
        "lobby actual puede afectar a compañeros ya invitados a él."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "queue_type": {
                "type": "string",
                "enum": sorted(_QUEUE_TYPES),
                "description": (
                    "Cola a la que armar/cambiar el lobby antes de buscar partida. Valores: "
                    "'ranked_solo_duo' (Ranked Solo/Dúo 5v5), 'normal_draft' (Normales, "
                    "selección por turnos), 'normal_blind' (Normales, selección a ciegas), "
                    "'aram' (ARAM), 'arena' (Arena)."
                ),
            },
        },
        "required": ["queue_type"],
        "additionalProperties": False,
    }
    # Pedido explícito del usuario: sacar todos los confirmadores salvo SystemPowerTool (ver
    # close_app.py para el motivo completo — confirmación hablada rota en la práctica por
    # audio de juego durante la ventana de escucha, exactamente el caso real que motivó el
    # pedido). Downgrade deliberado a SAFE: puede reemplazar un lobby ya armado sin avisar.
    risk = RiskLevel.SAFE

    def describe(self, kwargs: dict[str, Any]) -> str:
        """Frase natural para el prompt de confirmación por voz de `PolicyEngine`
        (`jarvis.security.policy`) — a diferencia de `LockChampionTool.describe`/
        `CloseAppTool.describe`, `queue_type` es un enum cerrado (ya restringido por el schema de
        function-calling, `parameters["properties"]["queue_type"]["enum"]`), no texto libre
        dictado por voz que haga falta resolver por fuzzy-match contra un estado en vivo — así
        que esto no necesita conectarse a la LCU API, solo mapear el valor recibido contra
        `_QUEUE_TYPES` (misma tabla estática que usa `_set_lobby_queue_sync`).

        Avisa explícitamente sobre el riesgo real de la acción (compañeros ya invitados al lobby
        actual) — mismo motivo que el finding que originó este tool (ver "Historia" en el
        docstring del módulo): el usuario tiene que poder decir que no con la información
        relevante, no un genérico "¿confirmás cambiar de cola?"."""
        queue_type = kwargs.get("queue_type")
        if not isinstance(queue_type, str) or not queue_type.strip():
            return self.name
        resolved = _resolve_queue_type(queue_type)
        if resolved is None:
            return f"armar un lobby con la cola '{queue_type}' (no reconozco ese tipo de cola)"
        _queue_id, queue_label, _expected_mode = resolved
        return (
            f"armar el lobby de {queue_label} y arrancar la búsqueda de partida (si tenías "
            "compañeros invitados en el lobby actual, esto podría afectarlos)"
        )

    async def execute(self, **kwargs: Any) -> str:
        queue_type = kwargs.get("queue_type")
        if not isinstance(queue_type, str) or not queue_type.strip():
            return "No se especificó un tipo de cola válido."

        return await asyncio.to_thread(
            _set_lobby_queue_sync, queue_type=queue_type.strip()
        )


class CancelQueueTool(Tool):
    """Cancela una búsqueda de partida de League of Legends en curso."""

    name = "cancel_lol_queue"
    description = (
        "Cancela una búsqueda de partida de League of Legends que esté en curso ahora mismo "
        "(equivalente al botón de cancelar búsqueda del cliente). Usar cuando el usuario pida "
        "cancelar, parar o dejar de buscar partida."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(_cancel_queue_sync)
