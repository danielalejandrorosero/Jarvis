"""Tool para controlar la reproducción multimedia activa del sistema (ADR-0005).

Simula las teclas multimedia virtuales universales de Windows (`VK_MEDIA_PLAY_PAUSE`,
`VK_MEDIA_NEXT_TRACK`, `VK_MEDIA_PREV_TRACK`, `VK_MEDIA_STOP`) vía `ctypes` llamando a
`user32.keybd_event` — la misma API Win32 que usaría un teclado multimedia físico al presionar una
de esas teclas. `ctypes` es stdlib, así que esto no agrega ninguna dependencia nueva.

`keybd_event` en vez de la API más moderna `SendInput`: para un evento de tecla simple (no
sintético-de-mouse, no parte de una secuencia compleja de touch/pen), ambas producen exactamente
el mismo resultado a nivel de driver — `keybd_event` es una línea por press/release contra el
boilerplate de structs (`INPUT`, `KEYBDINPUT`, unions) que pide `SendInput`. `keybd_event` sigue
soportada por Windows (no deprecated para este caso; solo "superseded" para escenarios más
complejos que este tool no necesita), así que no hay motivo para pagar la complejidad extra acá.

Las teclas multimedia virtuales las intercepta el sistema operativo (el "session manager"
multimedia de Windows decide qué aplicación las recibe), no la ventana con foco — por eso este
tool controla lo que esté sonando en cualquier app (Spotify, una pestaña de YouTube, cualquier
reproductor), sin necesidad de saber cuál es ni de haberla abierto. Es justo lo que lo distingue
de `OpenUrlTool`/`OpenAppTool`: esos solo controlan lo que JARVIS mismo lanzó; este tool controla
"lo que esté sonando ahora", sea lo que sea.

SAFE bajo `.claude/rules/security.md`: presionar una tecla multimedia es un evento momentáneo, sin
persistencia — no muta ningún estado de Windows más allá de lo que la app que esté reproduciendo
decida hacer con ese evento (pausar, cambiar de pista), exactamente igual que si el usuario
presionara la tecla física del teclado. No instala, no escribe archivos, no cierra procesos.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, ClassVar

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

# Virtual-key codes de las teclas multimedia universales de Windows (tabla de Virtual-Key Codes
# de la Win32 API) — los mismos códigos que genera una tecla física de teclado multimedia.
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

_KEYEVENTF_KEYUP = (
    0x0002  # flag de keybd_event: este evento es un "soltar", no un "presionar"
)

_ACTION_TO_VK: dict[str, int] = {
    "play_pause": VK_MEDIA_PLAY_PAUSE,
    "next": VK_MEDIA_NEXT_TRACK,
    "previous": VK_MEDIA_PREV_TRACK,
    "stop": VK_MEDIA_STOP,
}

_ACTION_LABELS: dict[str, str] = {
    "play_pause": "play/pausa",
    "next": "siguiente pista",
    "previous": "pista anterior",
    "stop": "detuve la reproducción",
}

_INVALID_ACTION_MESSAGE = (
    "No se especificó una acción de reproducción válida (esperaba 'play_pause', "
    "'next', 'previous' o 'stop')."
)


def _press_media_key(vk_code: int) -> None:
    """Simula presionar y soltar `vk_code` vía `user32.keybd_event` (nivel de módulo,
    monkeypatcheable — mismo patrón que `_launch_shortcut` en `open_app.py`, así la suite de
    tests nunca envía una tecla real). Dos llamadas (`0` = key-down, `_KEYEVENTF_KEYUP` =
    key-up) porque `keybd_event` modela cada mitad del evento por separado, igual que hardware
    real: sin la segunda llamada, Windows seguiría viendo la tecla como presionada.
    """
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_code, 0, 0, 0)
    user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)


class MediaControlTool(Tool):
    """Controla la reproducción multimedia activa en el sistema (play/pausa, siguiente, anterior,
    detener) simulando las teclas multimedia universales de Windows."""

    name = "media_control"
    description = (
        "Controla la reproducción multimedia que esté sonando en la computadora, en cualquier "
        "aplicación (Spotify, una pestaña de YouTube, cualquier reproductor), sin necesidad de "
        "saber cuál es. Usar cuando el usuario pide pausar, reanudar, pasar a la siguiente "
        "canción/pista, volver a la anterior, o detener lo que está sonando."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play_pause", "next", "previous", "stop"],
                "description": (
                    "Acción a realizar: 'play_pause' (pausar si está sonando, reanudar si está "
                    "pausado), 'next' (siguiente pista), 'previous' (pista anterior), 'stop' "
                    "(detener la reproducción)."
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
            _press_media_key(_ACTION_TO_VK[action])
        except OSError as exc:
            return (
                "No pude ejecutar el control de reproducción "
                f"({exc.__class__.__name__})."
            )

        return f"Listo, {_ACTION_LABELS[action]}."
