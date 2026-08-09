"""Policy engine: único punto de paso entre el loop de dispatch y `Tool.execute` (ADR-0005).

Implementa la capa SECURITY/POLICY de `.claude/rules/architecture.md`: la única con autoridad
para decidir si una acción se ejecuta, se confirma o se rechaza. La clasificación de riesgo vive
en el `Tool` (`jarvis.tools.base.RiskLevel`); el comportamiento por nivel vive acá, centralizado,
para que sea cumplible por construcción — no depende de que el dispatch loop "se acuerde" de
chequear.

Este módulo no importa nada de audio/hardware (`sounddevice`, etc.): `ConfirmationChannel` es un
`Protocol` que `pipeline.py` implementa reusando TTS/STT, sin que la policy dependa de audio
directamente.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)


class ConfirmationChannel(Protocol):
    """Cómo se le pide confirmación al usuario para una acción CONFIRM.

    Contrato de ADR-0004 ("silencio o timeout ⇒ denegar por defecto"): la implementación debe
    devolver `False` ante silencio, timeout o transcripción vacía — nunca `True` por defecto.
    Esa lógica vive en quien implementa el `Protocol` (audio en `pipeline.py`), no acá.
    """

    async def ask(self, prompt: str) -> bool:
        """Pedir confirmación verbal para `prompt`; `True` solo ante una respuesta afirmativa
        explícita."""
        ...


class PolicyEngine:
    """Autoriza (o no) la ejecución de un `Tool` según su `RiskLevel`.

    SAFE se ejecuta sin fricción. CONFIRM exige una confirmación explícita vía
    `ConfirmationChannel` antes de ejecutar; una negativa (incluida la denegación por defecto de
    silencio/timeout) nunca ejecuta. DANGEROUS nunca se ejecuta desde acá, con o sin
    confirmación — no hay ningún código path que lo permita, por diseño
    (`.claude/rules/security.md`): la única vía para una acción DANGEROUS es ejecución manual del
    usuario fuera del agente.
    """

    def __init__(self, confirmation: ConfirmationChannel) -> None:
        self._confirmation = confirmation

    async def authorize_and_execute(
        self,
        tool: Tool,
        kwargs: dict[str, Any],
        *,
        on_executed: Callable[[], None] | None = None,
    ) -> str:
        """Autorizar `tool` según su `RiskLevel` y, si corresponde, ejecutarlo.

        `on_executed`, si se pasa, se llama sin argumentos una única vez, y únicamente después de
        que `tool.execute()` ya devolvió con éxito — nunca antes, y nunca en el caso CONFIRM
        denegado ni en el caso DANGEROUS (esos branches no llegan a llamar `tool.execute()`, así
        que tampoco a `on_executed`). Existe para que un llamador (`jarvis.audio.pipeline.
        dispatch_turn`, para `jarvis.memory.store.save_tool_call`) pueda saber con certeza que una
        llamada llegó a ejecutarse de verdad, sin tener que inferirlo parseando el texto de
        retorno de este método (que es igual de opaco para una llamada SAFE, una CONFIRM
        aprobada, una CONFIRM denegada o una DANGEROUS rechazada).

        Hallazgo de `security-reviewer`: `on_executed` corre DESPUÉS de que la acción real ya
        ocurrió (incluida una CONFIRM ya irreversible, ej. `lock_lol_champion`) — si lanzara sin
        capturar (ej. `save_tool_call` contra SQLite bajo lock contention con el thread de fondo
        de `TimerScheduler`), la excepción se propagaría antes de `return result` y el usuario se
        quedaría sin ninguna confirmación hablada de una acción que sí se ejecutó. Por eso se
        captura acá, ampliamente a propósito (frontera de una capa de recuperación documentada,
        `.claude/rules/python.md`): un fallo del logging de `on_executed` nunca debe ocultar que
        la acción principal ya se completó.
        """
        if tool.risk is RiskLevel.SAFE:
            result = await tool.execute(**kwargs)
            self._run_on_executed(on_executed)
            return result

        if tool.risk is RiskLevel.CONFIRM:
            approved = await self._confirmation.ask(
                f"¿Confirmás esta acción: {tool.describe(kwargs)}? Decí que sí o que no."
            )
            if not approved:
                return f"Acción '{tool.name}' cancelada: no se confirmó."
            result = await tool.execute(**kwargs)
            self._run_on_executed(on_executed)
            return result

        # RiskLevel.DANGEROUS — nunca se ejecuta desde el loop automatizado, ni con
        # confirmación de un solo paso (`.claude/rules/security.md`). No hay branch que llame
        # a `tool.execute()` acá, deliberadamente.
        return (
            f"Acción '{tool.name}' clasificada como DANGEROUS: JARVIS no la ejecuta "
            "automáticamente. Requiere ejecución manual fuera del agente."
        )

    @staticmethod
    def _run_on_executed(on_executed: Callable[[], None] | None) -> None:
        """Invocar `on_executed` (si se pasó) sin dejar que un fallo suyo se propague por encima
        de un `return result` de una acción que ya se ejecutó de verdad — ver docstring de
        `authorize_and_execute` para el escenario concreto que motiva esto."""
        if on_executed is None:
            return
        try:
            on_executed()
        except Exception:
            # Frontera de recuperación documentada: la acción principal ya se ejecutó, un fallo
            # del hook de logging no debe ocultar eso.
            logger.exception(
                "on_executed() falló después de una ejecución exitosa del tool "
                "(la acción real ya se completó, esto solo afecta el registro/log)."
            )
