"""Fake de `httpx.Client` compartido por los tests de los tools de League
(`test_lol_runes.py`, `test_lol_summoner_spells.py`, `test_lol_champion_select.py`) — sin prefijo
`test_`, así pytest no lo colecciona como archivo de tests (`pythonpath = ["src"]`,
`testpaths = ["tests"]` en `pyproject.toml`, default `python_files = "test_*.py"`).

A diferencia del `_FakeLCUClient` de `tests/league/test_lcu_monitor.py` (que solo necesita
`get`/`post` para el flujo fijo de ready-check), estos tres tools llaman a una variedad más
amplia de endpoints (`GET`/`POST`/`PATCH`/`DELETE`, varios paths distintos por tool) — un fake
ruteado por `(método, path)` evita escribir un stub artesanal por test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self


class FakeLCUResponse:
    def __init__(self, *, status_code: int = 200, json_body: Any = None) -> None:
        self.status_code = status_code
        self._json_body = json_body

    def json(self) -> Any:
        return self._json_body


class FakeLCUToolClient:
    """Stub de `httpx.Client` ruteado por `(método, path)`. `handlers` mapea cada endpoint a una
    `FakeLCUResponse` fija o a un callable `(json_body) -> FakeLCUResponse` cuando la respuesta
    depende del cuerpo de la request (ej. verificar qué se mandó a hacer un POST/PATCH). Un
    endpoint no registrado en `handlers` devuelve 404 en vez de lanzar `KeyError` — mismo espíritu
    que un servidor real respondiendo "no encontrado" a una ruta que no existe.

    `calls` registra cada invocación `(método, path, json_body)` en orden, para que los tests
    puedan verificar qué se llamó y con qué cuerpo sin necesidad de un mock de aserciones aparte.
    """

    def __init__(
        self,
        handlers: dict[
            tuple[str, str], FakeLCUResponse | Callable[[Any], FakeLCUResponse]
        ],
    ) -> None:
        self._handlers = handlers
        self.calls: list[tuple[str, str, Any]] = []
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.closed = True
        return False

    def _handle(self, method: str, path: str, json: Any = None) -> FakeLCUResponse:
        self.calls.append((method, path, json))
        handler = self._handlers.get((method, path))
        if handler is None:
            return FakeLCUResponse(status_code=404, json_body={"message": "not found"})
        if callable(handler):
            return handler(json)
        return handler

    def get(self, path: str) -> FakeLCUResponse:
        return self._handle("GET", path)

    def post(self, path: str, json: Any = None) -> FakeLCUResponse:
        return self._handle("POST", path, json)

    def patch(self, path: str, json: Any = None) -> FakeLCUResponse:
        return self._handle("PATCH", path, json)

    def delete(self, path: str) -> FakeLCUResponse:
        return self._handle("DELETE", path)
