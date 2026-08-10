"""Tool para cancelar un apagado/reinicio pendiente antes de que ocurra de verdad — pensado sobre
todo para el que `jarvis.tools.system_power.SystemPowerTool` acaba de programar, pero (ver más
abajo) no exclusivamente ese.

Gap encontrado por `security-reviewer` sobre `SystemPowerTool`: `shutdown /t 15` (ver
`_SHUTDOWN_DELAY_SECONDS` en `system_power.py` para el porqué de ese número) deja una ventana de
15 segundos entre la confirmación hablada y el apagado real — pensada como margen de seguridad
ante, por ejemplo, un "no" que el STT transcribió mal como "sí" — pero sin ningún comando de voz
para aprovechar ese margen, la ventana no servía de nada en la práctica. Este tool es ese comando:
"Alexa, cancelá el apagado" dentro de esos 15 segundos, y `shutdown.exe` nunca llega a ejecutarse.

Reusa `run_power_command` de `system_power.py` (el mismo wrapper monkeypatcheable sobre
`subprocess.run(..., creationflags=CREATE_NO_WINDOW)`) en vez de duplicar el mecanismo — `shutdown
/a` es, en los hechos, el mismo binario con otro flag.

SAFE bajo `.claude/rules/security.md`, no CONFIRM: cancelar un apagado pendiente es, si acaso,
MENOS invasivo que programarlo (`SystemPowerTool`, CONFIRM) — es una acción puramente protectora,
nunca deja el sistema en un estado peor que el que ya tenía antes de la confirmación original.
Exigir una segunda confirmación hablada para cancelar sería contraproducente: el punto de este
tool es poder abortar rápido dentro de una ventana de solo 15 segundos, no agregar más fricción en
el momento exacto en que el usuario está tratando de frenar algo. Mismo criterio ya aplicado en
este repo a `CancelTimerTool`/`CancelReminderTool`/`CancelQueueTool` (cancelar algo puntual es
SAFE; la versión "cancelar TODO" es la que sube a CONFIRM, ADR-0006 — acá no aplica esa distinción
porque `shutdown /a` solo puede cancelar un apagado/reinicio pendiente a la vez a nivel de Windows,
no una lista acotada de los que JARVIS mismo programó: `shutdown.exe` no distingue quién programó
el apagado pendiente, así que este tool cancela igual de bien uno disparado por
`SystemPowerTool` que uno disparado por cualquier otro mecanismo del sistema (ej. un reinicio
forzado por Windows Update que coincida en el tiempo) — sigue siendo SAFE en cualquier caso, porque
cancelar un apagado no programado por el usuario nunca deja la máquina peor de lo que ya estaba.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from jarvis.tools.base import RiskLevel, Tool
from jarvis.tools.system_power import run_power_command

_CANCEL_COMMAND = ["shutdown", "/a"]


class CancelSystemPowerTool(Tool):
    """Cancela cualquier apagado o reinicio pendiente antes de que ocurra (típicamente uno
    programado por `system_power`, pero `shutdown /a` cancela cualquier apagado/reinicio pendiente
    de Windows, sin importar quién lo haya disparado)."""

    name = "cancel_system_power"
    description = (
        "Cancela un apagado o reinicio de la computadora que ya fue confirmado y está por "
        "ocurrir (dentro de la breve ventana de margen antes de que se ejecute de verdad). Usar "
        "cuando el usuario pide cancelar, frenar o detener un apagado/reinicio que acaba de "
        "confirmar, por ejemplo si se arrepiente o si fue un error. No hace nada si no hay "
        "ningún apagado/reinicio pendiente en este momento."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        success, detail = await asyncio.to_thread(run_power_command, _CANCEL_COMMAND)
        if not success:
            # `shutdown /a` falla con un mensaje propio de Windows cuando no hay nada programado
            # para cancelar (caso esperado, no un error real) — se lo pasamos tal cual al usuario
            # en vez de inventar una distinción que `run_power_command` no expone.
            return f"No pude cancelar: {detail}."
        return "Listo, cancelé el apagado/reinicio pendiente."
