"""Primer tool real de JARVIS: consulta de clima vía Open-Meteo (ADR-0005).

Open-Meteo no requiere API key — geocoding (nombre de ciudad → lat/lon) + forecast (lat/lon →
clima actual). SAFE: solo lectura contra una API pública, sin mutar estado ni exponer
credenciales. "Qué hora es" queda deliberadamente fuera (se resuelve con `zoneinfo`, sin red) y
búsqueda web general queda fuera de esta fase — ver ADR-0005.

Usa `httpx` (ya transitivo vía `openai`, sin dependencia nueva) en modo async nativo: no hace
falta `asyncio.to_thread` porque el I/O de red ya es no bloqueante de por sí.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from jarvis.tools.base import RiskLevel, Tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 10.0

# Subconjunto de códigos WMO (https://open-meteo.com/en/docs) suficiente para una descripción
# corta en español hablada — no pretende cubrir el listado completo de ~100 códigos.
_WEATHER_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "despejado",
    1: "mayormente despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "con niebla",
    48: "con niebla escarchada",
    51: "con llovizna ligera",
    53: "con llovizna moderada",
    55: "con llovizna intensa",
    61: "con lluvia ligera",
    63: "con lluvia moderada",
    65: "con lluvia intensa",
    71: "con nevadas ligeras",
    73: "con nevadas moderadas",
    75: "con nevadas intensas",
    80: "con chubascos ligeros",
    81: "con chubascos moderados",
    82: "con chubascos intensos",
    95: "con tormenta eléctrica",
}


def _describe_weather_code(code: int) -> str:
    return _WEATHER_CODE_DESCRIPTIONS.get(code, f"código de clima {code}")


async def _geocode(
    client: httpx.AsyncClient, city: str
) -> tuple[str, float, float] | None:
    """Resolver un nombre de ciudad a `(nombre_canónico, latitud, longitud)`, o `None` si la
    API de geocoding no encuentra ningún resultado."""
    response = await client.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "es", "format": "json"},
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    results: list[Any] = data.get("results") or []
    if not results:
        return None
    first: dict[str, Any] = results[0]
    return str(first["name"]), float(first["latitude"]), float(first["longitude"])


async def _fetch_current_weather(
    client: httpx.AsyncClient, *, name: str, latitude: float, longitude: float
) -> str:
    """Consultar el clima actual para unas coordenadas y devolver una frase corta en español."""
    response = await client.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
        },
    )
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    current: dict[str, Any] | None = data.get("current_weather")
    if not current:
        return f"No hay datos de clima disponibles para {name} ahora mismo."
    temperature = current.get("temperature")
    code = current.get("weathercode")
    condition = (
        _describe_weather_code(int(code))
        if code is not None
        else "condición desconocida"
    )
    return f"En {name} hace {temperature} grados, {condition}."


class WeatherTool(Tool):
    """Consulta el clima actual de una ciudad."""

    name = "get_weather"
    description = (
        "Devuelve el clima actual (temperatura y condición) de una ciudad. Usar cuando el "
        "usuario pregunta por el clima, la temperatura o el tiempo de un lugar."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Nombre de la ciudad, ej. 'Buenos Aires' o 'Madrid'.",
            }
        },
        "required": ["city"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        city = kwargs.get("city")
        if not isinstance(city, str) or not city.strip():
            return "No se especificó una ciudad válida para consultar el clima."
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                location = await _geocode(client, city)
                if location is None:
                    return f"No encontré la ciudad '{city}'."
                name, latitude, longitude = location
                return await _fetch_current_weather(
                    client, name=name, latitude=latitude, longitude=longitude
                )
        except httpx.HTTPError as exc:
            return f"No pude consultar el clima de '{city}' ahora mismo ({exc.__class__.__name__})."
