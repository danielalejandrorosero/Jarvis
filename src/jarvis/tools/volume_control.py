"""Tool para ajustar el volumen general del sistema (ADR-0005, extendido con nivel exacto).

Dos formas de ajustar el volumen, según lo que pida el usuario:

- Pasos GRUESOS (`action`: `up`/`down`/`mute`) — mismo enfoque que `jarvis.tools.media_control`:
  simula las teclas de volumen virtuales universales de Windows (`VK_VOLUME_UP`, `VK_VOLUME_DOWN`,
  `VK_VOLUME_MUTE`) vía `ctypes` llamando a `user32.keybd_event` — sin dependencia nueva, `ctypes`
  es stdlib. Ver el docstring de `media_control.py` para la justificación completa de por qué
  `keybd_event` (más simple) y no `SendInput` (más moderna, pero sin ninguna ventaja real para
  este caso de uso). Cada llamada sube/baja el volumen lo mismo que una única pulsación de la
  tecla física de volumen (típicamente un puñado de puntos porcentuales, según lo que Windows
  tenga configurado), igual que si el usuario tocara esa tecla del teclado.
- Nivel EXACTO (`level`, 0-100) — pedido explícito del usuario ("poné el volumen en 30%"), que un
  paso grueso no puede resolver de forma confiable (no hay manera de saber cuántos "pasos" hacen
  falta para llegar a un porcentaje exacto, ni sumar deriva de redondeo). Esto sí necesita la Core
  Audio API de Windows (`IAudioEndpointVolume::SetMasterVolumeLevelScalar`, vía COM) — no hay
  equivalente en stdlib. Se usa `pycaw` (dependencia nueva, ver `pyproject.toml`), la librería
  estándar y ampliamente usada del ecosistema Python-Windows específicamente para envolver esta
  API de Core Audio (evita escribir a mano el boilerplate de interfaces COM/`comtypes` que ya
  resuelve `pycaw`); `comtypes` (que `pycaw` usa por debajo) ya era una dependencia transitiva
  existente en este repo (vía `pyttsx3`), así que el costo real incremental es solo `pycaw` en sí,
  una librería chica y de un único propósito.

`level` tiene precedencia sobre `action` cuando ambos vienen presentes en la misma llamada (no
debería pasar en la práctica — el LLM elige uno u otro según el pedido — pero definir un orden
explícito evita comportamiento ambiguo si igual ocurriera).

SAFE bajo `.claude/rules/security.md`, en ambos casos: ajustar el volumen (por paso o a un nivel
exacto) es un evento momentáneo y trivialmente reversible (otro ajuste lo deshace, incluyendo
sencillamente mover el mismo control deslizante que el usuario ya tiene disponible en la bandeja
del sistema), sin persistencia — no muta estado más allá del nivel de volumen en sí, que Windows
ya expone al usuario sin fricción de por sí.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import POINTER, cast
from typing import Any, ClassVar

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

MIN_VOLUME_LEVEL = 0
MAX_VOLUME_LEVEL = 100

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
    "No se especificó una acción de volumen válida ni un nivel exacto (esperaba 'action': "
    "'up', 'down' o 'mute', o 'level': un entero entre 0 y 100)."
)

_INVALID_LEVEL_MESSAGE = (
    f"El nivel de volumen tiene que ser un número entero entre {MIN_VOLUME_LEVEL} y "
    f"{MAX_VOLUME_LEVEL}."
)


def _press_volume_key(vk_code: int) -> None:
    """Simula presionar y soltar `vk_code` vía `user32.keybd_event` — mismo patrón que
    `_press_media_key` en `media_control.py` (nivel de módulo, monkeypatcheable, dos llamadas
    key-down/key-up porque así modela `keybd_event` un evento de tecla completo)."""
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_code, 0, 0, 0)
    user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)


def _set_volume_level(level: int) -> None:
    """Fijar el volumen general del sistema a `level` por ciento exacto, vía
    `IAudioEndpointVolume::SetMasterVolumeLevelScalar` (Core Audio API de Windows, envuelta por
    `pycaw`) sobre el dispositivo de salida por defecto (`AudioUtilities.GetSpeakers()`).

    Nivel de módulo y sin importar nada de `pycaw`/`comtypes` en la firma — monkeypatcheable
    igual que `_press_volume_key`, así la suite de tests nunca llama a la Core Audio API real.
    """
    speakers = AudioUtilities.GetSpeakers()
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(level / 100.0, None)


class VolumeControlTool(Tool):
    """Sube, baja o silencia el volumen general del sistema en un paso (`action`), o lo fija a un
    porcentaje exacto (`level`) — ver docstring del módulo."""

    name = "volume_control"
    description = (
        "Ajusta el volumen general de la computadora. Dos formas, según lo que pida el usuario: "
        "'action' (subir/bajar un paso, igual que una tecla física de volumen, o silenciar/mutear) "
        "cuando no especifica un número; 'level' (un entero de 0 a 100) cuando pide un porcentaje "
        "exacto (ej. 'poné el volumen en 30%', 'subilo al 100'). Si se especifican ambos, se usa "
        "'level'."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["up", "down", "mute"],
                "description": (
                    "Acción a realizar: 'up' (subir un paso), 'down' (bajar un paso), 'mute' "
                    "(silenciar o des-silenciar, alterna igual que la tecla física). Ignorado si "
                    "se especifica 'level'."
                ),
            },
            "level": {
                "type": "integer",
                "minimum": MIN_VOLUME_LEVEL,
                "maximum": MAX_VOLUME_LEVEL,
                "description": (
                    "Nivel exacto de volumen deseado, en porcentaje (0 a 100). Usar cuando el "
                    "usuario pide un número concreto en vez de solo subir/bajar/silenciar."
                ),
            },
        },
        "required": [],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        if "level" in kwargs and kwargs["level"] is not None:
            level = kwargs["level"]
            # `bool` es subclase de `int` en Python — mismo chequeo explícito que `TimerTool`/
            # `ReminderTool` (`jarvis.tools.timer`/`jarvis.tools.reminder`) para que `level=True`
            # no se cuele como un volumen de 1%.
            if (
                not isinstance(level, int)
                or isinstance(level, bool)
                or not (MIN_VOLUME_LEVEL <= level <= MAX_VOLUME_LEVEL)
            ):
                return _INVALID_LEVEL_MESSAGE
            try:
                _set_volume_level(level)
            except Exception as exc:  # noqa: BLE001 — frontera con la Core Audio API vía
                # `pycaw`/`comtypes`: un fallo real de COM (`comtypes.COMError`, no una subclase
                # de `OSError`) no puede dejarse escapar de `execute()`, mismo principio que el
                # resto de este módulo degrada cualquier fallo de la capa Windows a un mensaje
                # claro en vez de propagar la excepción.
                return f"No pude fijar el volumen ({exc.__class__.__name__})."
            return f"Listo, volumen en {level}%."

        action = kwargs.get("action")
        if not isinstance(action, str) or action not in _ACTION_TO_VK:
            return _INVALID_ACTION_MESSAGE

        try:
            _press_volume_key(_ACTION_TO_VK[action])
        except OSError as exc:
            return f"No pude ajustar el volumen ({exc.__class__.__name__})."

        return f"Listo, {_ACTION_LABELS[action]}."
