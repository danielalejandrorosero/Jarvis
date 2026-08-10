"""Tests para `PreviewChampionTool`/`LockChampionTool` (`jarvis.tools.lol_champion_select`,
ADR-0006 — partidos desde el `PickChampionTool` original).

Nunca toca un League Client real: `jarvis.league.lcu_monitor.connect_to_lcu` se monkeypatchea a
`None` (League no corriendo) o a un `FakeLCUToolClient` controlado por cada test — ver
`tests/tools/_lcu_fake_client.py`.

El caso de detección de modo ARAM (`test_execute_refuses_in_aram_with_realistic_session_payload`)
usa un payload de `/lol-gameflow/v1/session` inventado pero realista (mismo shape documentado en
`jarvis.league.game_mode`) en vez de monkeypatchear `detect_game_mode` directamente — ejercita el
parseo real, no solo el punto de integración.

La mayoría de los casos de `execute()` (LoL no corriendo, sin selección activa, ARAM, modo
indeterminado, sin acción activa, campeón desconocido, validación de input) son idénticos entre
los dos tools porque comparten `_select_champion_sync` (ver docstring del módulo) — se
parametrizan sobre ambas clases en vez de duplicar cada test dos veces. Lo que sí es específico
de cada tool: qué manda el PATCH (`completed=False`/`True`), el verbo de la respuesta
("Preseleccioné"/"Confirmé"), el `risk` declarado, y `LockChampionTool.describe()`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jarvis.league.game_mode import GAMEFLOW_SESSION_ENDPOINT
from jarvis.tools import lol_champion_select as pick_module
from jarvis.tools.base import RiskLevel, Tool
from jarvis.tools.lol_champion_select import (
    CHAMP_SELECT_ACTION_ENDPOINT_TEMPLATE,
    CHAMP_SELECT_SESSION_ENDPOINT,
    CHAMPION_SUMMARY_ENDPOINT,
    LockChampionTool,
    PreviewChampionTool,
)
from tests.tools._lcu_fake_client import FakeLCUResponse, FakeLCUToolClient

_CHAMPION_SUMMARY = [
    {"id": -1, "name": "None", "alias": "None"},
    {"id": 222, "name": "Jinx", "alias": "Jinx"},
    {"id": 145, "name": "Kai'Sa", "alias": "Kaisa"},
]

_ACTIVE_PICK_SESSION = {
    "localPlayerCellId": 3,
    "actions": [
        [
            {
                "id": 5,
                "actorCellId": 3,
                "championId": 0,
                "type": "pick",
                "completed": False,
            },
            {
                "id": 6,
                "actorCellId": 4,
                "championId": 0,
                "type": "pick",
                "completed": False,
            },
        ]
    ],
}

_NO_ACTIVE_ACTION_SESSION = {
    "localPlayerCellId": 3,
    "actions": [
        [
            {
                "id": 5,
                "actorCellId": 3,
                "championId": 222,
                "type": "pick",
                "completed": True,
            }
        ]
    ],
}

_CLASSIC_GAMEFLOW_SESSION = {
    "phase": "ChampSelect",
    "gameData": {"queue": {"id": 400, "gameMode": "CLASSIC"}},
}

_ARAM_GAMEFLOW_SESSION = {
    "phase": "ChampSelect",
    "gameData": {"queue": {"id": 450, "gameMode": "ARAM"}},
}

# Los dos tools comparten `_select_champion_sync` — parametrizar sobre la clase evita duplicar
# cada test de validación/estado compartido dos veces (ver docstring del módulo).
_TOOL_CLASSES: tuple[type[Tool], ...] = (PreviewChampionTool, LockChampionTool)


def _install_fake_connect(
    monkeypatch: pytest.MonkeyPatch, client: object | None
) -> None:
    monkeypatch.setattr(pick_module, "connect_to_lcu", lambda: client)


def _action_endpoint(action_id: int) -> str:
    return CHAMP_SELECT_ACTION_ENDPOINT_TEMPLATE.format(action_id=action_id)


def test_preview_champion_tool_declares_safe_risk() -> None:
    assert PreviewChampionTool.risk is RiskLevel.SAFE


def test_lock_champion_tool_declares_safe_risk() -> None:
    """Downgrade deliberado pedido explícitamente por el usuario — ver close_app.py."""
    assert LockChampionTool.risk is RiskLevel.SAFE


def test_preview_and_lock_tools_have_distinct_names() -> None:
    assert PreviewChampionTool.name == "preview_lol_champion"
    assert LockChampionTool.name == "lock_lol_champion"


def test_execute_previews_champion_without_confirming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_ACTIVE_PICK_SESSION
            ),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CLASSIC_GAMEFLOW_SESSION
            ),
            ("GET", CHAMPION_SUMMARY_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CHAMPION_SUMMARY
            ),
            ("PATCH", _action_endpoint(5)): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = PreviewChampionTool()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert result == "Preseleccioné a Jinx."
    method, path, body = client.calls[-1]
    assert (method, path) == ("PATCH", _action_endpoint(5))
    assert body == {"championId": 222, "completed": False}


def test_execute_locks_champion(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_ACTIVE_PICK_SESSION
            ),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CLASSIC_GAMEFLOW_SESSION
            ),
            ("GET", CHAMPION_SUMMARY_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CHAMPION_SUMMARY
            ),
            ("PATCH", _action_endpoint(5)): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = LockChampionTool()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert result == "Confirmé a Jinx."
    _method, _path, body = client.calls[-1]
    assert body == {"championId": 222, "completed": True}


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
def test_execute_reports_when_league_is_not_running(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool]
) -> None:
    _install_fake_connect(monkeypatch, None)
    tool = tool_cls()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert result == "League of Legends no está corriendo ahora mismo."


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
def test_execute_reports_when_not_in_champ_select(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool]
) -> None:
    client = FakeLCUToolClient(
        {("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(status_code=404)}
    )
    _install_fake_connect(monkeypatch, client)
    tool = tool_cls()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert result == "No estás en una selección de campeón activa ahora mismo."
    assert all(call[0] != "PATCH" for call in client.calls)


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
def test_execute_refuses_in_aram_with_realistic_session_payload(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool]
) -> None:
    """Caso pedido explícitamente: payload inventado-pero-realista de
    `/lol-gameflow/v1/session` representando ARAM — ejercita el parseo real de
    `jarvis.league.game_mode.detect_game_mode`, no un mock del resultado."""
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_ACTIVE_PICK_SESSION
            ),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_ARAM_GAMEFLOW_SESSION
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = tool_cls()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert "ARAM" in result
    assert "elegir campeón libremente" in result
    assert all(call[0] != "PATCH" for call in client.calls)


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
def test_execute_allows_arena(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool]
) -> None:
    """Corrección en vivo del usuario: a diferencia de ARAM, Arena SÍ soporta elegir campeón
    (mecánica de 'Valentía') — no debe negarse."""
    arena_gameflow_session = {
        "phase": "ChampSelect",
        "gameData": {"queue": {"id": 1700, "gameMode": "CHERRY"}},
    }
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_ACTIVE_PICK_SESSION
            ),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=arena_gameflow_session
            ),
            ("GET", CHAMPION_SUMMARY_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CHAMPION_SUMMARY
            ),
            ("PATCH", _action_endpoint(5)): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = tool_cls()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert result.endswith("a Jinx.")


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
def test_execute_reports_undetermined_mode_distinctly_from_refusal(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool]
) -> None:
    """No poder determinar el modo (gameflow no responde) es un mensaje distinto de "sabemos que
    este modo no soporta elección libre" — no deben confundirse."""
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_ACTIVE_PICK_SESSION
            ),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(status_code=500),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = tool_cls()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert result == (
        "No pude determinar el modo de juego actual para saber si podés elegir "
        "campeón libremente ahora mismo."
    )
    assert all(call[0] != "PATCH" for call in client.calls)


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
def test_execute_reports_no_active_action(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool]
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_NO_ACTIVE_ACTION_SESSION
            ),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CLASSIC_GAMEFLOW_SESSION
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = tool_cls()

    result = asyncio.run(tool.execute(champion_name="Jinx"))

    assert result == "No tenés una elección de campeón disponible en este momento."


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
def test_execute_reports_unknown_champion(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool]
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_ACTIVE_PICK_SESSION
            ),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CLASSIC_GAMEFLOW_SESSION
            ),
            ("GET", CHAMPION_SUMMARY_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CHAMPION_SUMMARY
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = tool_cls()

    result = asyncio.run(tool.execute(champion_name="Campeón Que No Existe"))

    assert result == "No encontré ningún campeón llamado 'Campeón Que No Existe'."
    assert all(call[0] != "PATCH" for call in client.calls)


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"champion_name": ""},
        {"champion_name": "   "},
    ],
)
def test_execute_rejects_invalid_params_without_connecting(
    monkeypatch: pytest.MonkeyPatch, tool_cls: type[Tool], kwargs: dict[str, object]
) -> None:
    connected = {"called": False}
    monkeypatch.setattr(
        pick_module, "connect_to_lcu", lambda: connected.update(called=True)
    )
    tool = tool_cls()

    result = asyncio.run(tool.execute(**kwargs))

    assert isinstance(result, str) and result
    assert not connected["called"]


def test_lock_champion_tool_describe_names_the_resolved_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`describe()` tiene que nombrar el campeón real (resuelto contra el listado de campeones),
    no el texto crudo del usuario — ver docstring de `_describe_lock_target` (ADR-0006, mismo
    motivo que el finding 7 de `security-reviewer` en `close_app.py`). Acá el input ("Kaisa", el
    alias sin apóstrofe, típico de una transcripción por voz) resuelve a un nombre canónico
    distinto ("Kai'Sa") — si `describe()` devolviera el texto crudo, este test no lo detectaría.
    """
    client = FakeLCUToolClient(
        {
            ("GET", CHAMPION_SUMMARY_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CHAMPION_SUMMARY
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = LockChampionTool()

    description = tool.describe({"champion_name": "Kaisa"})

    assert "Kai'Sa" in description
    assert "lockear" in description.lower()


def test_lock_champion_tool_describe_falls_back_when_league_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_connect(monkeypatch, None)
    tool = LockChampionTool()

    description = tool.describe({"champion_name": "Jinx"})

    assert "Jinx" in description
    assert "lockear" in description.lower()


def test_lock_champion_tool_describe_falls_back_when_champion_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", CHAMPION_SUMMARY_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=_CHAMPION_SUMMARY
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = LockChampionTool()

    description = tool.describe({"champion_name": "Campeón Que No Existe"})

    assert "Campeón Que No Existe" in description
    assert "lockear" in description.lower()


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"champion_name": ""}, {"champion_name": "   "}, {"champion_name": 1}],
)
def test_lock_champion_tool_describe_handles_missing_or_invalid_name(
    kwargs: dict[str, Any],
) -> None:
    tool = LockChampionTool()

    description = tool.describe(kwargs)

    assert description == tool.name
