"""Tests para `SetRunesTool` (`jarvis.tools.lol_runes`).

Nunca toca un League Client real: `jarvis.league.lcu_monitor.connect_to_lcu` se monkeypatchea
para devolver `None` (League no corriendo) o un `FakeLCUToolClient` (`tests.tools.
_lcu_fake_client`) controlado por cada test — mismo espíritu que `tests/league/test_lcu_monitor.py`
usa para `_build_client`.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.tools import lol_runes as lol_runes_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.lol_runes import (
    JARVIS_PAGE_NAME_PREFIX,
    LOL_PERKS_PAGES_ENDPOINT,
    SetRunesTool,
)
from tests.tools._lcu_fake_client import FakeLCUResponse, FakeLCUToolClient

_VALID_KWARGS = {
    "champion_name": "Jinx",
    "primary_style_id": 8000,
    "sub_style_id": 8100,
    "selected_perk_ids": [8005, 9111, 9104, 8014, 8017, 8299, 5008, 5008, 5002],
}


def _install_fake_connect(
    monkeypatch: pytest.MonkeyPatch, client: object | None
) -> None:
    monkeypatch.setattr(lol_runes_module, "connect_to_lcu", lambda: client)


def _install_fake_game_mode(monkeypatch: pytest.MonkeyPatch, mode: str | None) -> None:
    monkeypatch.setattr(lol_runes_module, "detect_game_mode", lambda _client: mode)


def test_set_runes_tool_declares_safe_risk() -> None:
    """Mismo razonamiento que `LCUAutoAcceptMonitor` (`jarvis.league.lcu_monitor`): API local que
    el propio cliente expone para este caso de uso, sin mutar estado de Windows."""
    assert SetRunesTool.risk is RiskLevel.SAFE


def test_execute_creates_page_and_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_game_mode(monkeypatch, "CLASSIC")
    client = FakeLCUToolClient(
        {
            ("POST", LOL_PERKS_PAGES_ENDPOINT): FakeLCUResponse(status_code=200),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert result == "Listo, configuré las runas de Jinx."
    assert client.closed
    method, path, body = client.calls[-1]
    assert (method, path) == ("POST", LOL_PERKS_PAGES_ENDPOINT)
    assert body["primaryStyleId"] == 8000
    assert body["subStyleId"] == 8100
    assert body["current"] is True


def test_execute_reports_when_league_is_not_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_connect(monkeypatch, None)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert result == "League of Legends no está corriendo ahora mismo."


def test_execute_refuses_in_arena(monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrección en vivo del usuario: Arena reemplaza runas pre-partida por Augments in-game —
    el tool se niega en vez de crear una página que el juego va a ignorar."""
    _install_fake_game_mode(monkeypatch, "CHERRY")
    client = FakeLCUToolClient({})
    _install_fake_connect(monkeypatch, client)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert "Arena" in result
    assert "Augments" in result
    # Nunca se intentó el POST de la página — la negativa corta el flujo antes de la llamada.
    assert all(call[0] != "POST" for call in client.calls)


def test_execute_refuses_when_game_mode_cannot_be_determined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`detect_game_mode` devolviendo `None` (red, parseo, o queue ID no reconocido) no debe
    tratarse como "no es Arena, seguir adelante" — el tool se niega con un mensaje propio, mismo
    criterio fail-closed que `PickChampionTool` (`jarvis.tools.lol_champion_select`)."""
    _install_fake_game_mode(monkeypatch, None)
    client = FakeLCUToolClient({})
    _install_fake_connect(monkeypatch, client)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert result == (
        "No pude determinar el modo de juego actual para saber si corresponde "
        "configurar runas ahora mismo."
    )
    assert all(call[0] != "POST" for call in client.calls)


def test_execute_allows_normal_aram(monkeypatch: pytest.MonkeyPatch) -> None:
    """ARAM tradicional sigue usando runas normales — no debe bloquearse (a diferencia de
    Arena/"ARAM Chaos", ver docstring del módulo)."""
    _install_fake_game_mode(monkeypatch, "ARAM")
    client = FakeLCUToolClient(
        {("POST", LOL_PERKS_PAGES_ENDPOINT): FakeLCUResponse(status_code=200)}
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert result == "Listo, configuré las runas de Jinx."


def test_execute_deletes_jarvis_owned_page_and_retries_on_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si el primer POST falla (límite de páginas propias alcanzado), el tool libera una página
    existente marcada como borrable Y creada por JARVIS (prefijo `JARVIS_PAGE_NAME_PREFIX`) y
    reintenta una vez. Nunca borra una página ajena solo por estar marcada `isDeletable`."""
    _install_fake_game_mode(monkeypatch, "CLASSIC")
    post_calls = {"count": 0}

    def _post_handler(_body: object) -> FakeLCUResponse:
        post_calls["count"] += 1
        if post_calls["count"] == 1:
            return FakeLCUResponse(
                status_code=400, json_body={"message": "limit reached"}
            )
        return FakeLCUResponse(status_code=200)

    client = FakeLCUToolClient(
        {
            ("POST", LOL_PERKS_PAGES_ENDPOINT): _post_handler,
            ("GET", LOL_PERKS_PAGES_ENDPOINT): FakeLCUResponse(
                status_code=200,
                json_body=[
                    {"id": 1, "isDeletable": False},
                    {
                        "id": 2,
                        "isDeletable": True,
                        "name": f"{JARVIS_PAGE_NAME_PREFIX}Yasuo",
                    },
                ],
            ),
            ("DELETE", f"{LOL_PERKS_PAGES_ENDPOINT}/2"): FakeLCUResponse(
                status_code=204
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert result == "Listo, configuré las runas de Jinx."
    assert post_calls["count"] == 2
    assert ("DELETE", f"{LOL_PERKS_PAGES_ENDPOINT}/2", None) in client.calls


def test_execute_refuses_to_delete_page_not_owned_by_jarvis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hallazgo de seguridad (HIGH): si el primer POST falla por límite de páginas y las únicas
    páginas borrables disponibles NO tienen el prefijo `JARVIS_PAGE_NAME_PREFIX` (son páginas
    hechas a mano por el usuario), el tool no borra nada — fail closed — y le pide al usuario que
    libere cupo manualmente, en vez de arriesgarse a borrar una página que no es suya."""
    _install_fake_game_mode(monkeypatch, "CLASSIC")
    post_calls = {"count": 0}

    def _post_handler(_body: object) -> FakeLCUResponse:
        post_calls["count"] += 1
        return FakeLCUResponse(status_code=400, json_body={"message": "limit reached"})

    client = FakeLCUToolClient(
        {
            ("POST", LOL_PERKS_PAGES_ENDPOINT): _post_handler,
            ("GET", LOL_PERKS_PAGES_ENDPOINT): FakeLCUResponse(
                status_code=200,
                json_body=[
                    {"id": 1, "isDeletable": False},
                    {"id": 2, "isDeletable": True, "name": "Mi build afinada"},
                    {"id": 3, "isDeletable": True, "name": "Otra página mía"},
                ],
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert result == (
        "Ya tenés el máximo de páginas de runas y ninguna es mía para reemplazar — "
        "borrá una manualmente desde el cliente e intentá de nuevo."
    )
    # Solo un intento de POST — nunca se reintenta porque nunca se borró nada.
    assert post_calls["count"] == 1
    assert all(call[0] != "DELETE" for call in client.calls)


def test_execute_reports_failure_detail_when_post_keeps_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_game_mode(monkeypatch, "CLASSIC")
    client = FakeLCUToolClient(
        {
            ("POST", LOL_PERKS_PAGES_ENDPOINT): FakeLCUResponse(
                status_code=400, json_body={"message": "invalid perk ids"}
            ),
            ("GET", LOL_PERKS_PAGES_ENDPOINT): FakeLCUResponse(
                status_code=200, json_body=[]
            ),
        }
    )
    _install_fake_connect(monkeypatch, client)
    tool = SetRunesTool()

    result = asyncio.run(tool.execute(**_VALID_KWARGS))

    assert result == "No pude configurar las runas de Jinx: invalid perk ids."


@pytest.mark.parametrize(
    "overrides",
    [
        {"champion_name": ""},
        {"champion_name": "   "},
        {"primary_style_id": "8000"},
        {"primary_style_id": True},
        {"primary_style_id": -1},
        {"sub_style_id": 0},
        {"selected_perk_ids": []},
        {"selected_perk_ids": ["not", "ints"]},
        {"selected_perk_ids": [1, 2, True]},
    ],
)
def test_execute_rejects_invalid_params_without_connecting(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    connected = {"called": False}

    def _fail_if_called() -> None:
        connected["called"] = True
        raise AssertionError("no debería intentar conectar con parámetros inválidos")

    monkeypatch.setattr(lol_runes_module, "connect_to_lcu", lambda: _fail_if_called())
    tool = SetRunesTool()
    kwargs = {**_VALID_KWARGS, **overrides}

    result = asyncio.run(tool.execute(**kwargs))

    assert not connected["called"]
    assert isinstance(result, str) and result
