"""Tool para arrancar la búsqueda de partida (matchmaking) de League of Legends vía la LCU API,
cuando el usuario ya está sentado en un lobby (no arma un lobby desde cero — sin selección de
`queueId` ni lógica de creación de lobby, ver más abajo).

Endpoints usados:

- `GET /lol-gameflow/v1/gameflow-phase` (mismo endpoint que ya usa
  `jarvis.league.lcu_monitor.LCUAutoAcceptMonitor._poll_gameflow` para su propio polling — se
  reutiliza la misma constante `GAMEFLOW_PHASE_ENDPOINT` de ese módulo en vez de redeclararla):
  devuelve un string plano (`response.json()` es directamente `"Lobby"`, `"Matchmaking"`,
  `"ReadyCheck"`, etc. — no un objeto), se consulta acá antes de intentar nada, para dar un
  mensaje claro y específico según en qué fase esté el usuario en vez de un error genérico de la
  LCU API.
- `POST /lol-lobby/v2/lobby/matchmaking/search`, cuerpo vacío: arranca la búsqueda de partida
  para el lobby en el que el usuario ya está. **Verificado en vivo esta noche contra el cliente
  real del usuario** (no documentación no oficial sin confirmar, a diferencia de la salvedad que
  sí aplica a `jarvis.tools.lol_runes`/`jarvis.tools.lol_summoner_spells`/
  `jarvis.tools.lol_champion_select`): un script puntual confirmó que este POST con cuerpo vacío
  devuelve `204` y dispara la transición de fase `Lobby` → `Matchmaking` → `ReadyCheck` como se
  espera. `CancelQueueTool` (más abajo) usa el `DELETE` del mismo endpoint — no verificado en
  vivo de la misma forma, pero es la operación inversa documentada del mismo par de endpoints, así
  que se aplica el mismo criterio de confianza relativa que el resto de los tools de League.

## Scoping: por qué esto no crea un lobby

Elegir cola (Ranked Solo/Dúo, Normales, ARAM, Arena, etc. — cada una con su propio `queueId`) y
crear el lobby correspondiente (`POST /lol-lobby/v2/lobby`, cuerpo con `queueId`) es una decisión
de producto distinta y más amplia (qué colas están disponibles, mapeo nombre-hablado → `queueId`,
qué hacer si el usuario ya tiene gente invitada a un lobby existente de otra cola) que no fue
pedida para este tool — el pedido concreto fue "arrancar a buscar partida estando ya en un
lobby", igual que el botón "BUSCAR PARTIDA" que el usuario mostró en su propio cliente (lobby de
Arena 3x6 armado, listo para buscar). Si más adelante se pide "armá una partida de ARAM desde
cero", eso es un tool nuevo (o una extensión explícita de este), no algo a inferir acá.

## Por qué no espera el ready-check

`LCUAutoAcceptMonitor` (`jarvis.league.lcu_monitor`) ya es responsable de aceptar el ready-check
en segundo plano — corre siempre, independiente de que el usuario haya usado este tool o
apretado el botón él mismo. Este tool es una acción de voz síncrona (`execute()` la llama el
usuario y espera una respuesta hablada en segundos, no en minutos) — su trabajo termina en cuanto
el `POST` de arranque de búsqueda se confirma, no debe quedarse sondeando `gameflow-phase` a la
espera de que aparezca gente para la partida.

Clasificación: **SAFE** (`.claude/rules/security.md`) — mismo razonamiento que
`jarvis.tools.lol_runes.SetRunesTool`/`jarvis.league.lcu_monitor.LCUAutoAcceptMonitor`: interactúa
solo con la API local que el propio cliente expone para este caso de uso exacto, equivalente a que
el usuario apriete el botón "Buscar partida" él mismo. Reversible: `CancelQueueTool` (o el propio
cliente) puede cancelar la búsqueda antes de que aparezca un ready-check.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import httpx

from jarvis.league.lcu_monitor import GAMEFLOW_PHASE_ENDPOINT, connect_to_lcu
from jarvis.tools.base import RiskLevel, Tool

MATCHMAKING_SEARCH_ENDPOINT = "/lol-lobby/v2/lobby/matchmaking/search"

LOBBY_PHASE = "Lobby"

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
    que `jarvis.tools.lol_runes._set_runes_sync`."""
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
    """Arranca la búsqueda de partida (matchmaking) para un lobby de League of Legends en el que
    el usuario ya está — no crea el lobby ni elige la cola (ver docstring del módulo)."""

    name = "start_lol_queue"
    description = (
        "Arranca la búsqueda de partida (botón 'Buscar partida') en League of Legends, cuando "
        "el usuario ya está en un lobby armado (cualquier cola: Ranked, Normales, ARAM, Arena, "
        "etc.). No arma un lobby ni elige la cola — si el usuario no está en ningún lobby ahora "
        "mismo, o ya está buscando/en selección/en partida, este tool te va a devolver un "
        "mensaje claro explicando por qué no corresponde en vez de intentarlo. Usar cuando el "
        "usuario, estando ya en un lobby, pida buscar partida, arrancar a jugar o empezar la "
        "cola."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(_start_queue_sync)


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
