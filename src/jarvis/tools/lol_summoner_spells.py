"""Tool para configurar los hechizos de invocador durante la selección de campeón de League of
Legends, vía la LCU API.

Endpoint: `PATCH /lol-champ-select/v1/session/my-selection`, cuerpo `{"spell1Id": <id>,
"spell2Id": <id>}` — `spell1Id` corresponde a la tecla D, `spell2Id` a la tecla F (convención del
propio cliente). Documentación no oficial pero ampliamente usada (replicada por numerosos
proyectos open-source de selección de campeón vía LCU); igual que en `jarvis.tools.lol_runes`,
esta sesión de trabajo no tiene acceso a fetch de red para releerla en vivo ahora mismo — el shape
viene de conocimiento de entrenamiento sobre esa misma documentación, no de una verificación
fresca. El usuario está en partida (no en selección de campeón) mientras se escribe esto, así que
tampoco hay una sesión de LCU real disponible ahora para probarlo en vivo.

Los IDs de hechizo de invocador en sí (`_SUMMONER_SPELL_IDS`) sí son datos mucho más estables y
ampliamente documentados (Data Dragon `summoner.json`, sin cambios en años: Flash=4, Ignite=14,
etc.) — no dependen de conocimiento de "meta actual" como las runas, así que se resuelven acá por
nombre con un diccionario fijo, no delegado al LLM.

Solo tiene sentido durante una selección de campeón activa (`GET /lol-champ-select/v1/session`
como chequeo de fase, antes de intentar el PATCH) — fuera de esa fase, `execute()` devuelve un
mensaje claro en vez de intentar la acción.

## Exclusión por modo de juego: Arena (corrección en vivo del usuario)

Arena no tiene un paso de selección de hechizos de invocador como parte de su champ-select —
confirmado explícitamente por el usuario, no una suposición de este código. `_set_summoner_
spells_sync` consulta `jarvis.league.game_mode.detect_game_mode` una vez confirmada la sesión de
champ-select activa, y se niega con un mensaje claro si el modo detectado es Arena, en vez de
intentar un PATCH que ese modo no soporta de la forma en que este tool asume. Ningún otro modo
(incluida ARAM tradicional) está excluido acá. Si `detect_game_mode` devuelve `None` (no se pudo
determinar el modo), este tool también se niega con un mensaje propio en vez de asumir que NO es
Arena y proceder de todos modos — mismo criterio fail-closed que
`jarvis.tools.lol_champion_select.PickChampionTool`/`jarvis.tools.lol_runes.SetRunesTool`.

Clasificación: **SAFE** — mismo razonamiento que `jarvis.tools.lol_runes.SetRunesTool`: API local
que el propio cliente expone para este caso de uso, equivalente a que el usuario elija sus
hechizos con clicks en la pantalla de selección de campeón.
"""

from __future__ import annotations

import asyncio
import unicodedata
from typing import Any, ClassVar

import httpx

from jarvis.league.game_mode import ARENA_GAME_MODE, detect_game_mode
from jarvis.league.lcu_monitor import connect_to_lcu
from jarvis.tools.base import RiskLevel, Tool

CHAMP_SELECT_SESSION_ENDPOINT = "/lol-champ-select/v1/session"
MY_SELECTION_ENDPOINT = "/lol-champ-select/v1/session/my-selection"

# IDs estables de Data Dragon (`summoner.json`) — sin cambios de un patch a otro desde hace años,
# a diferencia de los IDs de runas/estilos (esos sí dependen de balance/meta). Claves en español e
# inglés, normalizadas (minúsculas, sin acentos) por `_normalize` antes de comparar.
_SUMMONER_SPELL_IDS: dict[str, int] = {
    "flash": 4,
    "destello": 4,
    "ignite": 14,
    "ignicion": 14,
    "heal": 7,
    "curacion": 7,
    "curar": 7,
    "barrier": 21,
    "barrera": 21,
    "exhaust": 3,
    "debilitar": 3,
    "cleanse": 1,
    "purificar": 1,
    "purificacion": 1,
    "ghost": 6,
    "fantasma": 6,
    "teleport": 12,
    "teleportacion": 12,
    "smite": 11,
    "golpe certero": 11,
    "clarity": 13,
    "clarividencia": 13,
    "mark": 32,
    "marca": 32,
    "snowball": 32,
}


def _normalize(text: str) -> str:
    """Minúsculas y sin acentos (`unicodedata`, sin dependencia nueva) — así "ignición" e
    "ignicion" (dictado por voz, a veces sin tilde) resuelven al mismo ID."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _resolve_spell_id(name: str) -> int | None:
    return _SUMMONER_SPELL_IDS.get(_normalize(name))


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


def _set_summoner_spells_sync(
    *, spell1_name: str, spell1_id: int, spell2_name: str, spell2_id: int
) -> str:
    """Cuerpo bloqueante de `SetSummonerSpellsTool.execute` — corre en un thread aparte, mismo
    patrón que `jarvis.tools.lol_runes._set_runes_sync`."""
    client = connect_to_lcu()
    if client is None:
        return "League of Legends no está corriendo ahora mismo."

    try:
        with client:
            session_response = client.get(CHAMP_SELECT_SESSION_ENDPOINT)
            if session_response.status_code != httpx.codes.OK:
                return "No estás en una selección de campeón activa ahora mismo."
            game_mode = detect_game_mode(client)
            if game_mode is None:
                return (
                    "No pude determinar el modo de juego actual para saber si "
                    "corresponde configurar los hechizos de invocador ahora mismo."
                )
            if game_mode == ARENA_GAME_MODE:
                return (
                    "En Arena no se pueden cambiar los hechizos de invocador: ese modo no "
                    "tiene ese paso en la selección de campeón."
                )
            response = client.patch(
                MY_SELECTION_ENDPOINT,
                json={"spell1Id": spell1_id, "spell2Id": spell2_id},
            )
            if response.status_code >= httpx.codes.BAD_REQUEST:
                detail = _extract_error_detail(response)
                return f"No pude configurar los hechizos de invocador: {detail}."
    except httpx.HTTPError as exc:
        return (
            "No pude conectar con League Client para configurar los hechizos de invocador "
            f"({exc.__class__.__name__})."
        )
    return f"Listo, configuré {spell1_name} y {spell2_name}."


class SetSummonerSpellsTool(Tool):
    """Configura los dos hechizos de invocador durante una selección de campeón activa."""

    name = "set_lol_summoner_spells"
    description = (
        "Configura los dos hechizos de invocador (ej. Destello, Ignición, Curación) durante la "
        "selección de campeón de League of Legends. Solo funciona mientras hay una selección de "
        "campeón activa en curso — si el usuario todavía no entró a partida o ya terminó de "
        "elegir, avisale que no se puede ahora mismo en vez de asumir que funcionó. No aplica "
        "en el modo Arena (no tiene paso de hechizos de invocador) — si el usuario está en "
        "Arena, este tool va a avisarte que no corresponde. Usar cuando el usuario pida poner/"
        "cambiar sus hechizos de invocador."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "spell1": {
                "type": "string",
                "description": (
                    "Hechizo para la tecla D, en español o inglés (ej. 'destello', 'flash', "
                    "'ignición', 'curación', 'barrera', 'debilitar', 'purificar', 'fantasma', "
                    "'teleportación', 'golpe certero', 'clarividencia', 'marca')."
                ),
            },
            "spell2": {
                "type": "string",
                "description": "Hechizo para la tecla F, mismas opciones que spell1.",
            },
        },
        "required": ["spell1", "spell2"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        spell1_name = kwargs.get("spell1")
        spell2_name = kwargs.get("spell2")
        if not isinstance(spell1_name, str) or not spell1_name.strip():
            return "No se especificó un hechizo de invocador válido para la tecla D."
        if not isinstance(spell2_name, str) or not spell2_name.strip():
            return "No se especificó un hechizo de invocador válido para la tecla F."
        spell1_name = spell1_name.strip()
        spell2_name = spell2_name.strip()

        spell1_id = _resolve_spell_id(spell1_name)
        if spell1_id is None:
            return f"No reconozco el hechizo de invocador '{spell1_name}'."
        spell2_id = _resolve_spell_id(spell2_name)
        if spell2_id is None:
            return f"No reconozco el hechizo de invocador '{spell2_name}'."
        if spell1_id == spell2_id:
            return "No podés llevar el mismo hechizo de invocador dos veces."

        return await asyncio.to_thread(
            _set_summoner_spells_sync,
            spell1_name=spell1_name,
            spell1_id=spell1_id,
            spell2_name=spell2_name,
            spell2_id=spell2_id,
        )
