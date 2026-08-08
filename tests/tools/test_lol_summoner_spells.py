"""Tests para `SetSummonerSpellsTool` (`jarvis.tools.lol_summoner_spells`).

Nunca toca un League Client real: `jarvis.league.lcu_monitor.connect_to_lcu` se monkeypatchea a
`None` (League no corriendo) o a un `FakeLCUToolClient` controlado por cada test — ver
`tests/tools/_lcu_fake_client.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.tools import lol_summoner_spells as spells_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.lol_summoner_spells import (
    CHAMP_SELECT_SESSION_ENDPOINT,
    MY_SELECTION_ENDPOINT,
    SetSummonerSpellsTool,
)
from tests.tools._lcu_fake_client import FakeLCUResponse, FakeLCUToolClient


def _install_fake_connect(
    monkeypatch: pytest.MonkeyPatch, client: object | None
) -> None:
    monkeypatch.setattr(spells_module, "connect_to_lcu", lambda: client)


def _install_fake_game_mode(monkeypatch: pytest.MonkeyPatch, mode: str | None) -> None:
    monkeypatch.setattr(spells_module, "detect_game_mode", lambda _client: mode)


def test_set_summoner_spells_tool_declares_safe_risk() -> None:
    assert SetSummonerSpellsTool.risk is RiskLevel.SAFE


def test_execute_sets_spells_during_active_champ_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_game_mode(monkeypatch, "CLASSIC")
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(status_code=200),
            ("PATCH", MY_SELECTION_ENDPOINT): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="flash", spell2="ignite"))

    assert result == "Listo, configuré flash y ignite."
    method, path, body = client.calls[-1]
    assert (method, path) == ("PATCH", MY_SELECTION_ENDPOINT)
    assert body == {"spell1Id": 4, "spell2Id": 14}


def test_execute_accepts_spanish_names_without_accents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_game_mode(monkeypatch, "CLASSIC")
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(status_code=200),
            ("PATCH", MY_SELECTION_ENDPOINT): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="destello", spell2="ignicion"))

    assert result == "Listo, configuré destello y ignicion."
    _method, _path, body = client.calls[-1]
    assert body == {"spell1Id": 4, "spell2Id": 14}


def test_execute_reports_when_league_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_connect(monkeypatch, None)
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="flash", spell2="ignite"))

    assert result == "League of Legends no está corriendo ahora mismo."


def test_execute_reports_when_not_in_champ_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLCUToolClient(
        {("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(status_code=404)}
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="flash", spell2="ignite"))

    assert result == "No estás en una selección de campeón activa ahora mismo."
    assert all(call[0] != "PATCH" for call in client.calls)


def test_execute_refuses_in_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrección en vivo del usuario: Arena no tiene paso de selección de hechizos de
    invocador — el tool se niega en vez de intentar un PATCH que ese modo no soporta."""
    _install_fake_game_mode(monkeypatch, "CHERRY")
    client = FakeLCUToolClient(
        {("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(status_code=200)}
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="flash", spell2="ignite"))

    assert "Arena" in result
    assert all(call[0] != "PATCH" for call in client.calls)


def test_execute_refuses_when_game_mode_cannot_be_determined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`detect_game_mode` devolviendo `None` (red, parseo, o queue ID no reconocido) no debe
    tratarse como "no es Arena, seguir adelante" — el tool se niega con un mensaje propio, mismo
    criterio fail-closed que `PickChampionTool` (`jarvis.tools.lol_champion_select`)."""
    _install_fake_game_mode(monkeypatch, None)
    client = FakeLCUToolClient(
        {("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(status_code=200)}
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="flash", spell2="ignite"))

    assert result == (
        "No pude determinar el modo de juego actual para saber si corresponde "
        "configurar los hechizos de invocador ahora mismo."
    )
    assert all(call[0] != "PATCH" for call in client.calls)


def test_execute_allows_normal_aram(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_game_mode(monkeypatch, "ARAM")
    client = FakeLCUToolClient(
        {
            ("GET", CHAMP_SELECT_SESSION_ENDPOINT): FakeLCUResponse(status_code=200),
            ("PATCH", MY_SELECTION_ENDPOINT): FakeLCUResponse(status_code=204),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="flash", spell2="heal"))

    assert result == "Listo, configuré flash y heal."


def test_execute_rejects_unrecognized_spell_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected = {"called": False}

    def _fail_if_called() -> None:
        connected["called"] = True
        raise AssertionError(
            "no debería intentar conectar con un hechizo no reconocido"
        )

    monkeypatch.setattr(spells_module, "connect_to_lcu", lambda: _fail_if_called())
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="hechizo-inventado", spell2="flash"))

    assert "hechizo-inventado" in result
    assert not connected["called"]


def test_execute_rejects_duplicate_spells(monkeypatch: pytest.MonkeyPatch) -> None:
    connected = {"called": False}

    def _fail_if_called() -> None:
        connected["called"] = True
        raise AssertionError("no debería intentar conectar con hechizos duplicados")

    monkeypatch.setattr(spells_module, "connect_to_lcu", lambda: _fail_if_called())
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(spell1="flash", spell2="destello"))

    assert result == "No podés llevar el mismo hechizo de invocador dos veces."
    assert not connected["called"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spell1": "", "spell2": "flash"},
        {"spell1": "flash", "spell2": ""},
        {"spell2": "flash"},
        {"spell1": "flash"},
    ],
)
def test_execute_rejects_missing_or_blank_spells(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, str]
) -> None:
    connected = {"called": False}
    monkeypatch.setattr(
        spells_module, "connect_to_lcu", lambda: connected.update(called=True)
    )
    tool = SetSummonerSpellsTool()

    result = asyncio.run(tool.execute(**kwargs))

    assert isinstance(result, str) and result
    assert not connected["called"]
