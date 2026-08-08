"""Tests para `WeatherTool` (`jarvis.tools.weather`, ADR-0005).

Sin red real: `httpx.AsyncClient` se reemplaza por un stub cuyo `.get()` devuelve respuestas
fabricadas en memoria — mismo enfoque que el resto del repo (núcleo puro con dependencia externa
stubeada), acá aplicado a un cliente async. Sin `pytest-asyncio` instalado (no está entre las
dev deps), las corrutinas se corren con `asyncio.run()` directamente en tests sync.
"""

from __future__ import annotations

import asyncio
from typing import Self

import pytest

from jarvis.tools import weather as weather_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.weather import WeatherTool


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeAsyncClient:
    def __init__(
        self, *, geocoding: dict[str, object], forecast: dict[str, object]
    ) -> None:
        self._geocoding = geocoding
        self._forecast = forecast

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get(self, url: str, *, params: dict[str, object]) -> _FakeResponse:
        if url == weather_module.GEOCODING_URL:
            return _FakeResponse(self._geocoding)
        assert url == weather_module.FORECAST_URL
        return _FakeResponse(self._forecast)


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    geocoding: dict[str, object],
    forecast: dict[str, object],
) -> None:
    def _ctor(*args: object, **kwargs: object) -> _FakeAsyncClient:
        return _FakeAsyncClient(geocoding=geocoding, forecast=forecast)

    monkeypatch.setattr(weather_module.httpx, "AsyncClient", _ctor)


def test_weather_tool_declares_safe_risk() -> None:
    """El primer tool real de JARVIS es SAFE — solo lectura, sin key, sin mutar estado."""
    assert WeatherTool.risk is RiskLevel.SAFE


def test_execute_returns_temperature_and_condition_for_known_city(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso aceptado: geocoding resuelve la ciudad, forecast devuelve clima actual, `execute()`
    arma una frase corta en español con temperatura y condición."""
    _install_fake_client(
        monkeypatch,
        geocoding={
            "results": [{"name": "Madrid", "latitude": 40.4, "longitude": -3.7}]
        },
        forecast={"current_weather": {"temperature": 21.5, "weathercode": 0}},
    )
    tool = WeatherTool()

    result = asyncio.run(tool.execute(city="Madrid"))

    assert result == "En Madrid hace 21.5 grados, despejado."


def test_execute_reports_city_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caso límite: sin resultados de geocoding, `execute()` devuelve un mensaje claro en vez
    de lanzar una excepción o alucinar datos."""
    _install_fake_client(monkeypatch, geocoding={"results": []}, forecast={})
    tool = WeatherTool()

    result = asyncio.run(tool.execute(city="Ciudad Inexistente Xyz123"))

    assert "No encontré la ciudad" in result


def test_execute_rejects_missing_city_without_network_call() -> None:
    """Caso rechazado: sin `city` (o con un valor no-string/vacío), `execute()` no intenta
    ninguna llamada de red y devuelve un mensaje de error claro."""
    tool = WeatherTool()

    result = asyncio.run(tool.execute())

    assert "ciudad válida" in result
