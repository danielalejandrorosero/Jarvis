"""Detección del modo de juego actual de League of Legends a partir de la LCU API, compartida
por los tres tools de League invocados por voz (`jarvis.tools.lol_runes`,
`jarvis.tools.lol_summoner_spells`, `jarvis.tools.lol_champion_select`) — cada uno necesita saber
"¿estoy en Arena?" / "¿estoy en ARAM?" para decidir si una acción tiene sentido en el modo actual
(runas y hechizos de invocador se comportan distinto en Arena; elegir campeón libremente no
aplica en ARAM tradicional), así que la consulta y el parseo del estado de partida vive en un solo
lugar en vez de triplicarse.

**Corrección en vivo del usuario, no conocimiento de entrenamiento sin verificar** (importante:
las reglas de exclusión por modo de acá vienen de lo que el usuario confirmó explícitamente
durante esta sesión, no de una suposición genérica sobre League):

- Arena (`gameMode` interno `"CHERRY"` — el nombre que usa la propia API de Riot, confirmado por
  conocimiento de entrenamiento, con la salvedad de abajo) reemplazó las páginas de runas
  pre-partida por Augments elegidos durante la partida (yunques/prompts in-game, no
  `/lol-perks/v1/pages`) — así que `SetRunesTool` tiene que negarse en Arena. Arena tampoco tiene
  selección de hechizos de invocador como paso de champ-select — así que `SetSummonerSpellsTool`
  también se niega en Arena. Elegir campeón SÍ aplica en Arena (mecánica de "Valentía"/"Bravery":
  reroll entre un pool, pero sigue siendo una elección del jugador) — `PickChampionTool` NO se
  niega en Arena, a diferencia de los otros dos tools.
- ARAM tradicional (`gameMode` `"ARAM"`) sigue usando páginas de runas normales (el campeón puede
  venir asignado/rerolleado, pero las runas se configuran igual que siempre para el campeón que
  te toque) — `SetRunesTool` NO se niega acá. Elegir campeón libremente NO aplica (asignación
  aleatoria) — `PickChampionTool` sí se niega acá.
- "ARAM Chaos" (mencionado por el usuario: tampoco usa runas normales) — **gap conocido, sin
  implementar**: no tengo un `gameMode`/queue ID confirmado para este modo distinto del de ARAM
  tradicional, y adivinar uno arriesga bloquear ARAM normal por error si el ID adivinado
  resultara ser incorrecto (viola el requisito explícito de no romper ARAM normal). `detect_
  game_mode` NO distingue "ARAM Chaos" de `"ARAM"` — reportado como riesgo/limitación abierta,
  no silenciado.

**Advertencia de verificación** (aplica a todo lo demás en este módulo también): esta sesión de
trabajo no tuvo acceso a un fetch de red para releer la documentación de la LCU API en vivo. El
string `"CHERRY"` para Arena y los IDs de cola de más abajo vienen de conocimiento de
entrenamiento sobre documentación no oficial ampliamente replicada (proyectos open-source de
LCU), no de una verificación fresca — y Riot cambió los queue IDs de Arena más de una vez desde
su beta 2023-2024. Antes de confiar en el fallback por queue ID en producción, vale un chequeo
puntual contra un cliente real.
"""

from __future__ import annotations

from typing import Any

import httpx

GAMEFLOW_SESSION_ENDPOINT = "/lol-gameflow/v1/session"

ARENA_GAME_MODE = "CHERRY"
ARAM_GAME_MODE = "ARAM"

# Fallback por ID de cola cuando el campo `gameMode` no está presente/reconocido en la respuesta
# — ver advertencia de verificación en el docstring del módulo.
ARENA_QUEUE_IDS = frozenset({1700, 1710, 1720})
ARAM_QUEUE_IDS = frozenset({450})


def detect_game_mode(client: httpx.Client) -> str | None:
    """Devolver el `gameMode` normalizado (mayúsculas) reportado por
    `/lol-gameflow/v1/session` (ej. `"ARAM"`, `"CLASSIC"`, `"CHERRY"`), o `None` si no se pudo
    determinar: falla de red/parseo, la respuesta no tiene el shape esperado, o el campo
    `gameMode` está ausente y el ID de cola tampoco se reconoce en `ARENA_QUEUE_IDS`/
    `ARAM_QUEUE_IDS`. Nunca lanza — cualquier tool llamador decide cómo degradar ante `None` según
    su propio caso (ver cada uno).
    """
    try:
        response = client.get(GAMEFLOW_SESSION_ENDPOINT)
    except httpx.HTTPError:
        return None
    if response.status_code != httpx.codes.OK:
        return None
    try:
        data: Any = response.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    game_data = data.get("gameData")
    if not isinstance(game_data, dict):
        return None
    queue = game_data.get("queue")

    game_mode: Any = None
    queue_id: Any = None
    if isinstance(queue, dict):
        game_mode = queue.get("gameMode")
        queue_id = queue.get("id")
    if not isinstance(game_mode, str):
        fallback_mode = game_data.get("gameMode")
        game_mode = fallback_mode if isinstance(fallback_mode, str) else None

    if isinstance(game_mode, str) and game_mode.strip():
        return game_mode.strip().upper()
    if isinstance(queue_id, int):
        if queue_id in ARENA_QUEUE_IDS:
            return ARENA_GAME_MODE
        if queue_id in ARAM_QUEUE_IDS:
            return ARAM_GAME_MODE
    return None
