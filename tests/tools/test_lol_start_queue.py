"""Tests para `StartQueueTool`/`SetLobbyQueueTool`/`CancelQueueTool`
(`jarvis.tools.lol_start_queue`).

Nunca toca un League Client real: `jarvis.league.lcu_monitor.connect_to_lcu` se monkeypatchea a
`None` (League no corriendo) o a un `FakeLCUToolClient` (`tests.tools._lcu_fake_client`)
controlado por cada test — mismo patrón que `test_lol_runes.py`/`test_lol_champion_select.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.league.game_mode import ARENA_QUEUE_IDS, GAMEFLOW_SESSION_ENDPOINT
from jarvis.league.lcu_monitor import GAMEFLOW_PHASE_ENDPOINT
from jarvis.tools import lol_start_queue as start_queue_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.lol_start_queue import (
    LOBBY_ENDPOINT,
    MATCHMAKING_SEARCH_ENDPOINT,
    CancelQueueTool,
    SetLobbyQueueTool,
    StartQueueTool,
)
from tests.tools._lcu_fake_client import FakeLCUResponse, FakeLCUToolClient


def _install_fake_connect(
    monkeypatch: pytest.MonkeyPatch, client: object | None
) -> None:
    monkeypatch.setattr(start_queue_module, "connect_to_lcu", lambda: client)


def _gameflow_session_response(game_mode: str) -> FakeLCUResponse:
    """Respuesta fake de `GET /lol-gameflow/v1/session` con el shape mínimo que
    `jarvis.league.game_mode.detect_game_mode` necesita para resolver `game_mode` como
    `gameMode`."""
    return FakeLCUResponse(
        status_code=200,
        json_body={"gameData": {"queue": {"gameMode": game_mode}}},
    )


def test_start_queue_tool_declares_safe_risk() -> None:
    assert StartQueueTool.risk is RiskLevel.SAFE


def test_set_lobby_queue_tool_declares_safe_risk() -> None:
    """Downgrade deliberado pedido explícitamente por el usuario — ver close_app.py."""
    assert SetLobbyQueueTool.risk is RiskLevel.SAFE


def test_cancel_queue_tool_declares_safe_risk() -> None:
    assert CancelQueueTool.risk is RiskLevel.SAFE


def test_start_queue_tool_has_no_queue_type_parameter() -> None:
    """Finding 1 (ADR-0006): `start_lol_queue` vuelve a su scope original — no acepta ningún
    parámetro, en particular no `queue_type` (eso es responsabilidad exclusiva de
    `set_lol_lobby_queue`, CONFIRM)."""
    assert StartQueueTool.parameters["properties"] == {}


def test_execute_starts_search_when_in_lobby(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Lobby"
            ),
            ("POST", MATCHMAKING_SEARCH_ENDPOINT): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = StartQueueTool()

    result = asyncio.run(tool.execute())

    assert result == "Listo, buscando partida."
    assert ("POST", MATCHMAKING_SEARCH_ENDPOINT, None) in client.calls


def test_execute_ignores_unexpected_kwargs_like_queue_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`start_lol_queue` no declara `queue_type` en su schema — si de todos modos llega en
    `kwargs` (ej. el LLM lo alucina), `execute()` lo ignora en vez de fallar, porque nunca lo lee
    del `kwargs`."""
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Lobby"
            ),
            ("POST", MATCHMAKING_SEARCH_ENDPOINT): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = StartQueueTool()

    result = asyncio.run(tool.execute(queue_type="arena"))

    assert result == "Listo, buscando partida."
    assert all(call[1] != LOBBY_ENDPOINT for call in client.calls)


def test_execute_reports_when_league_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_connect(monkeypatch, None)
    tool = StartQueueTool()

    result = asyncio.run(tool.execute())

    assert result == "League of Legends no está corriendo ahora mismo."


def test_execute_reports_when_phase_cannot_be_determined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLCUToolClient(
        {("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(status_code=500)}
    )
    _install_fake_connect(monkeypatch, client)
    tool = StartQueueTool()

    result = asyncio.run(tool.execute())

    assert result == "No pude determinar en qué fase del juego estás ahora mismo."
    assert all(call[0] != "POST" for call in client.calls)


@pytest.mark.parametrize(
    ("phase", "expected_substring"),
    [
        ("None", "armá uno primero"),
        ("Matchmaking", "Ya estás buscando partida"),
        ("ReadyCheck", "ready check"),
        ("ChampSelect", "selección de campeón"),
        ("InProgress", "partida en curso"),
    ],
)
def test_execute_refuses_with_specific_message_per_phase(
    monkeypatch: pytest.MonkeyPatch, phase: str, expected_substring: str
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=phase
            )
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = StartQueueTool()

    result = asyncio.run(tool.execute())

    assert expected_substring in result
    assert all(call[0] != "POST" for call in client.calls)


def test_execute_refuses_with_generic_message_for_unrecognized_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="SomeFuturePhase"
            )
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = StartQueueTool()

    result = asyncio.run(tool.execute())

    assert "SomeFuturePhase" in result
    assert all(call[0] != "POST" for call in client.calls)


def test_execute_reports_failure_detail_when_post_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Lobby"
            ),
            ("POST", MATCHMAKING_SEARCH_ENDPOINT): FakeLCUResponse(
                status_code=400, json_body={"message": "no active lobby"}
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = StartQueueTool()

    result = asyncio.run(tool.execute())

    assert result == "No pude arrancar la búsqueda de partida: no active lobby."


def test_cancel_execute_cancels_search(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeLCUToolClient(
        {("DELETE", MATCHMAKING_SEARCH_ENDPOINT): FakeLCUResponse(status_code=204)}
    )
    _install_fake_connect(monkeypatch, client)
    tool = CancelQueueTool()

    result = asyncio.run(tool.execute())

    assert result == "Listo, cancelé la búsqueda de partida."
    assert ("DELETE", MATCHMAKING_SEARCH_ENDPOINT, None) in client.calls


def test_cancel_execute_reports_when_league_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_connect(monkeypatch, None)
    tool = CancelQueueTool()

    result = asyncio.run(tool.execute())

    assert result == "League of Legends no está corriendo ahora mismo."


def test_cancel_execute_reports_failure_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeLCUToolClient(
        {
            ("DELETE", MATCHMAKING_SEARCH_ENDPOINT): FakeLCUResponse(
                status_code=404, json_body={"message": "no search in progress"}
            )
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = CancelQueueTool()

    result = asyncio.run(tool.execute())

    assert result == ("No pude cancelar la búsqueda de partida: no search in progress.")


# --- `SetLobbyQueueTool` (`set_lol_lobby_queue`): armar/cambiar cola, CONFIRM (ADR-0006) -------

_QUEUE_TYPE_CASES = [
    ("ranked_solo_duo", 420, "CLASSIC"),
    ("normal_draft", 400, "CLASSIC"),
    ("normal_blind", 430, "CLASSIC"),
    ("aram", 450, "ARAM"),
    ("arena", 1700, "CHERRY"),
]


def test_arena_queue_type_id_is_a_known_arena_queue_id() -> None:
    """El `queueId` que se pide para `queue_type="arena"` tiene que seguir siendo uno de los IDs
    de Arena que `jarvis.league.game_mode` ya monitorea — si ese módulo actualiza el conjunto
    (Riot cambia el ID vigente) y este tool no se actualiza en paralelo, esta aserción falla en
    vez de quedar desincronizado en silencio."""
    arena_queue_id = next(
        queue_id
        for queue_type, queue_id, _mode in _QUEUE_TYPE_CASES
        if queue_type == "arena"
    )
    assert arena_queue_id in ARENA_QUEUE_IDS


@pytest.mark.parametrize(
    ("queue_type", "expected_queue_id"), [(q, qid) for q, qid, _ in _QUEUE_TYPE_CASES]
)
def test_describe_names_the_queue_and_warns_about_teammates(
    queue_type: str, expected_queue_id: int
) -> None:
    tool = SetLobbyQueueTool()

    description = tool.describe({"queue_type": queue_type})

    assert "compañeros invitados" in description
    assert "arrancar la búsqueda" in description


def test_describe_falls_back_to_generic_for_missing_queue_type() -> None:
    tool = SetLobbyQueueTool()

    assert tool.describe({}) == tool.name


def test_describe_reports_unrecognized_queue_type() -> None:
    tool = SetLobbyQueueTool()

    description = tool.describe({"queue_type": "clash"})

    assert "clash" in description
    assert "no reconozco" in description.lower()


@pytest.mark.parametrize(
    ("queue_type", "expected_queue_id", "expected_mode"), _QUEUE_TYPE_CASES
)
def test_execute_creates_lobby_then_confirms_mode_then_searches(
    monkeypatch: pytest.MonkeyPatch,
    queue_type: str,
    expected_queue_id: int,
    expected_mode: str,
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Lobby"
            ),
            ("POST", LOBBY_ENDPOINT): FakeLCUResponse(status_code=200),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): _gameflow_session_response(
                expected_mode
            ),
            ("POST", MATCHMAKING_SEARCH_ENDPOINT): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type=queue_type))

    assert "Listo" in result
    assert ("POST", LOBBY_ENDPOINT, {"queueId": expected_queue_id}) in client.calls
    # El lobby se arma, LUEGO se confirma el modo, y solo entonces se arranca la búsqueda.
    lobby_call_index = client.calls.index(
        ("POST", LOBBY_ENDPOINT, {"queueId": expected_queue_id})
    )
    mode_check_index = client.calls.index(("GET", GAMEFLOW_SESSION_ENDPOINT, None))
    search_call_index = client.calls.index(("POST", MATCHMAKING_SEARCH_ENDPOINT, None))
    assert lobby_call_index < mode_check_index < search_call_index


def test_execute_works_with_no_existing_lobby(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fase `"None"` (usuario sin ningún lobby armado) no se rechaza cuando se pide una cola
    concreta — la LCU API arma un lobby nuevo igual."""
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="None"
            ),
            ("POST", LOBBY_ENDPOINT): FakeLCUResponse(status_code=200),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): _gameflow_session_response("CLASSIC"),
            ("POST", MATCHMAKING_SEARCH_ENDPOINT): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type="ranked_solo_duo"))

    assert "Listo" in result
    assert ("POST", LOBBY_ENDPOINT, {"queueId": 420}) in client.calls


def test_execute_with_invalid_queue_type_never_touches_lcu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_if_called() -> object:
        raise AssertionError(
            "connect_to_lcu no debería llamarse con un queue_type inválido"
        )

    monkeypatch.setattr(start_queue_module, "connect_to_lcu", _fail_if_called)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type="clash"))

    assert "clash" in result
    assert "ranked_solo_duo" in result


def test_execute_missing_queue_type_never_touches_lcu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_if_called() -> object:
        raise AssertionError("connect_to_lcu no debería llamarse sin queue_type")

    monkeypatch.setattr(start_queue_module, "connect_to_lcu", _fail_if_called)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute())

    assert "No se especificó" in result


def test_execute_while_already_searching_asks_to_cancel_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso límite explícito: el usuario ya está en `Matchmaking` (búsqueda en curso) y pide
    cambiar de cola. En vez de mandar `POST /lol-lobby/v2/lobby` a ciegas (la UI del cliente
    deshabilita eso mientras se busca — ver docstring del módulo), el tool se niega con un
    mensaje claro pidiendo cancelar primero, sin tocar ningún endpoint de POST."""
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Matchmaking"
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type="aram"))

    assert "cancelá la búsqueda" in result.lower() or "cancelá" in result.lower()
    assert all(call[0] != "POST" for call in client.calls)


@pytest.mark.parametrize(
    "phase",
    ["ChampSelect", "InProgress", "ReadyCheck"],
)
def test_execute_refuses_in_phases_that_cannot_change_lobby(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=phase
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type="arena"))

    assert result != "Listo, buscando partida."
    assert all(call[0] != "POST" for call in client.calls)


def test_execute_reports_failure_when_lobby_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Lobby"
            ),
            ("POST", LOBBY_ENDPOINT): FakeLCUResponse(
                status_code=400, json_body={"message": "invalid queue"}
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type="normal_draft"))

    assert (
        result
        == "No pude armar el lobby de Normales (selección por turnos): invalid queue."
    )
    # No debe intentar buscar partida si no se pudo armar el lobby primero.
    assert all(
        call[0] != "POST" or call[1] != MATCHMAKING_SEARCH_ENDPOINT
        for call in client.calls
    )


# --- Finding 2 (MEDIUM): discrepancia entre la cola pedida y la cola resultante ------------


def test_execute_reports_mismatch_when_resulting_mode_differs_from_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso concreto de Finding 2: se pide Arena (`queueId=1700`), el `POST` de creación de
    lobby "tiene éxito" (200), pero la lectura de confirmación posterior
    (`GET /lol-gameflow/v1/session`, vía `detect_game_mode`) reporta un `gameMode` distinto del
    esperado para Arena (`"CHERRY"`) — ej. porque Riot reasignó `1700` a otra cola vigente. El
    tool tiene que reportar la discrepancia en vez de confirmar éxito sobre un modo equivocado, y
    no debe arrancar la búsqueda de partida en ese modo no pedido."""
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Lobby"
            ),
            ("POST", LOBBY_ENDPOINT): FakeLCUResponse(status_code=200),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): _gameflow_session_response("CLASSIC"),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type="arena"))

    assert "no coincide" in result.lower()
    assert "Arena" in result
    assert all(call[1] != MATCHMAKING_SEARCH_ENDPOINT for call in client.calls)


def test_execute_reports_when_resulting_mode_cannot_be_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si la lectura de confirmación posterior a la creación del lobby falla (red, parseo, shape
    inesperado — `detect_game_mode` devuelve `None`), el tool no asume éxito ni intenta buscar
    partida: reporta que no pudo confirmar el modo resultante."""
    client = FakeLCUToolClient(
        {
            ("GET", GAMEFLOW_PHASE_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body="Lobby"
            ),
            ("POST", LOBBY_ENDPOINT): FakeLCUResponse(status_code=200),
            ("GET", GAMEFLOW_SESSION_ENDPOINT): FakeLCUResponse(status_code=500),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetLobbyQueueTool()

    result = asyncio.run(tool.execute(queue_type="ranked_solo_duo"))

    assert "no pude confirmar" in result.lower()
    assert all(call[1] != MATCHMAKING_SEARCH_ENDPOINT for call in client.calls)
