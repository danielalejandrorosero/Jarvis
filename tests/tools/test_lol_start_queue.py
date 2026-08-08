"""Tests para `StartQueueTool`/`CancelQueueTool` (`jarvis.tools.lol_start_queue`).

Nunca toca un League Client real: `jarvis.league.lcu_monitor.connect_to_lcu` se monkeypatchea a
`None` (League no corriendo) o a un `FakeLCUToolClient` (`tests.tools._lcu_fake_client`)
controlado por cada test — mismo patrón que `test_lol_runes.py`/`test_lol_champion_select.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.league.lcu_monitor import GAMEFLOW_PHASE_ENDPOINT
from jarvis.tools import lol_start_queue as start_queue_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.lol_start_queue import (
    MATCHMAKING_SEARCH_ENDPOINT,
    CancelQueueTool,
    StartQueueTool,
)
from tests.tools._lcu_fake_client import FakeLCUResponse, FakeLCUToolClient


def _install_fake_connect(
    monkeypatch: pytest.MonkeyPatch, client: object | None
) -> None:
    monkeypatch.setattr(start_queue_module, "connect_to_lcu", lambda: client)


def test_start_queue_tool_declares_safe_risk() -> None:
    assert StartQueueTool.risk is RiskLevel.SAFE


def test_cancel_queue_tool_declares_safe_risk() -> None:
    assert CancelQueueTool.risk is RiskLevel.SAFE


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
