"""Tool para ajustar el volumen general del sistema (ADR-0005).

Mismo enfoque que `jarvis.tools.media_control`: simula las teclas de volumen virtuales
universales de Windows (`VK_VOLUME_UP`, `VK_VOLUME_DOWN`, `VK_VOLUME_MUTE`) vía `ctypes` llamando
a `user32.keybd_event` — sin dependencia nueva, `ctypes` es stdlib. Ver el docstring de
`media_control.py` para la justificación completa de por qué `keybd_event` (más simple) y no
`SendInput` (más moderna, pero sin ninguna ventaja real para este caso de uso).

Deliberadamente de pasos GRUESOS, no de precisión porcentual: cada llamada sube/baja el volumen
lo mismo que una única pulsación de la tecla física de volumen (típicamente un puñado de puntos
porcentuales, según lo que Windows tenga configurado), igual que si el usuario tocara esa tecla
del teclado. Fijar un volumen exacto (ej. "poné el volumen al 50%") necesitaría la Core Audio API
de Windows (vía `pycaw` u otra librería equivalente) — una dependencia bastante más pesada, no
justificada para el pedido actual ("subí/bajá/silenciá el volumen"). Si más adelante se pide
precisión porcentual, es una decisión de scope nueva, no una extensión trivial de este módulo.

SAFE bajo `.claude/rules/security.md`: ajustar el volumen es un evento momentáneo y trivialmente
reversible (otra pulsación lo deshace), sin persistencia — exactamente igual que presionar la
tecla física de volumen del teclado. No muta estado más allá del nivel de volumen en sí, que
Windows ya expone al usuario sin fricción de por sí (bandeja del sistema, teclas físicas).
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, ClassVar

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

# Virtual-key codes de las teclas de volumen universales de Windows (tabla de Virtual-Key Codes
# de la Win32 API) — los mismos códigos que genera una tecla física de teclado multimedia.
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF

_KEYEVENTF_KEYUP = (
    0x0002  # flag de keybd_event: este evento es un "soltar", no un "presionar"
)

_ACTION_TO_VK: dict[str, int] = {
    "up": VK_VOLUME_UP,
    "down": VK_VOLUME_DOWN,
    "mute": VK_VOLUME_MUTE,
}

_ACTION_LABELS: dict[str, str] = {
    "up": "subí el volumen",
    "down": "bajé el volumen",
    "mute": "silencié el audio",
}

_INVALID_ACTION_MESSAGE = (
    "No se especificó una acción de volumen válida (esperaba 'up', 'down' o 'mute')."
)


def _press_volume_key(vk_code: int) -> None:
    """Simula presionar y soltar `vk_code` vía `user32.keybd_event` — mismo patrón que
    `_press_media_key` en `media_control.py` (nivel de módulo, monkeypatcheable, dos llamadas
    key-down/key-up porque así modela `keybd_event` un evento de tecla completo)."""
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_code, 0, 0, 0)
    user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)


class VolumeControlTool(Tool):
    """Sube, baja o silencia el volumen general del sistema, en pasos del mismo tamaño que una
    tecla física de volumen del teclado (ver docstring del módulo — no fija un porcentaje
    exacto)."""

    name = "volume_control"
    description = (
        "Ajusta el volumen general de la computadora en un paso, igual que presionar una tecla "
        "física de volumen del teclado (no fija un porcentaje exacto). Usar cuando el usuario "
        "pide subir, bajar o silenciar/mutear el volumen."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["up", "down", "mute"],
                "description": (
                    "Acción a realizar: 'up' (subir un paso), 'down' (bajar un paso), 'mute' "
                    "(silenciar o des-silenciar, alterna igual que la tecla física)."
                ),
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action")
        if not isinstance(action, str) or action not in _ACTION_TO_VK:
            return _INVALID_ACTION_MESSAGE

        try:
            _press_volume_key(_ACTION_TO_VK[action])
        except OSError as exc:
            return f"No pude ajustar el volumen ({exc.__class__.__name__})."

        return f"Listo, {_ACTION_LABELS[action]}."
