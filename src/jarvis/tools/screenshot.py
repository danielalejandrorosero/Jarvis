"""Tool para capturar la pantalla completa y guardarla como archivo de imagen (ADR-0005).

Dependencia nueva: `Pillow` (`PIL.ImageGrab`). Stdlib no tiene ninguna forma de capturar pantalla
en Windows sin GDI vía `ctypes` a mano (crear un device context compatible, un bitmap, hacer
`BitBlt`, convertir el bitmap a bytes de imagen) — decenas de líneas de boilerplate de bajo nivel
para algo que `ImageGrab.grab()` resuelve en una llamada. Pillow es una librería de imágenes
extremadamente estándar, activamente mantenida y de bajo riesgo — mismo criterio de justificación
que ya se aplicó a `httpx`/`psutil` en este repo (`.claude/rules/python.md`).

Ubicación de guardado: `data/screenshots/`, subdirectorio de `data/` — el directorio de estado en
tiempo de ejecución ya establecido en este repo (gitignored; usado por `jarvis.memory.store` para
la DB de SQLite y por `scripts/start_jarvis.ps1` para logs). Se crea si no existe. Nombre de
archivo con timestamp (`screenshot-<YYYYmmdd-HHMMSS>.png`) para no pisar capturas anteriores en
llamadas sucesivas.

SAFE bajo `.claude/rules/security.md`: tomar una captura de pantalla y guardarla es una acción de
usuario sin privilegios, exactamente análoga a usar la tecla "Imprimir pantalla" o la herramienta
de recortes de Windows — no lee ni expone nada que el usuario no pueda ya ver en su propia
pantalla en ese instante, no muta configuración ni estado del sistema, solo escribe un archivo
nuevo dentro del directorio de datos propio de JARVIS (no sobrescribe ni toca nada existente). El
tool no abre ni muestra el archivo, solo lo guarda y confirma la ubicación — el usuario decide si
quiere mirarlo.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, ImageGrab

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

# Relativo al directorio desde el que se corre JARVIS, mismo criterio que
# `jarvis.memory.store.DEFAULT_DB_PATH` — no un path absoluto hardcodeado. Nivel de módulo y
# monkeypatcheable (mismo patrón que `START_MENU_DIRECTORIES` en `open_app.py`) para que los
# tests nunca escriban bajo el `data/` real del repo.
DEFAULT_SCREENSHOT_DIR = Path("data/screenshots")

_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def _capture_screen() -> Image.Image:
    """Captura la pantalla completa como una imagen en memoria. Nivel de módulo, monkeypatcheable
    — la suite de tests nunca dispara una captura real."""
    return ImageGrab.grab()


def _save_screenshot(image: Image.Image, directory: Path) -> Path:
    """Guarda `image` como PNG dentro de `directory` (creándolo si hace falta) con un nombre
    único basado en timestamp, y devuelve la ruta final. Nivel de módulo, monkeypatcheable —
    separado de `_capture_screen` para poder probar el caso de fallo al guardar (ej. directorio
    no creable) sin depender de una captura real."""
    directory.mkdir(parents=True, exist_ok=True)
    # UTC, igual que las marcas de tiempo de `jarvis.memory.store` — el nombre solo necesita ser
    # único y ordenable, no reflejar la hora local del usuario.
    timestamp = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)
    path = directory / f"screenshot-{timestamp}.png"
    image.save(path)
    return path


class ScreenshotTool(Tool):
    """Captura la pantalla completa y la guarda como archivo de imagen en `data/screenshots/`."""

    name = "screenshot"
    description = (
        "Captura la pantalla completa de la computadora y la guarda como archivo de imagen. "
        "Usar cuando el usuario pide tomar una captura de pantalla, un screenshot, o guardar lo "
        "que se ve en la pantalla ahora. No muestra ni abre la imagen, solo la guarda."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        try:
            image = await asyncio.to_thread(_capture_screen)
            path = await asyncio.to_thread(
                _save_screenshot, image, DEFAULT_SCREENSHOT_DIR
            )
        except OSError as exc:
            return f"No pude tomar la captura de pantalla ({exc.__class__.__name__})."

        return f"Captura de pantalla guardada en {path}."
