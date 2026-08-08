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

from typing import Any, Protocol

from jarvis.tools.base import RiskLevel, Tool


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

    async def authorize_and_execute(self, tool: Tool, kwargs: dict[str, Any]) -> str:
        if tool.risk is RiskLevel.SAFE:
            return await tool.execute(**kwargs)

        if tool.risk is RiskLevel.CONFIRM:
            approved = await self._confirmation.ask(
                f"¿Confirmás esta acción: {tool.describe(kwargs)}? Decí que sí o que no."
            )
            if not approved:
                return f"Acción '{tool.name}' cancelada: no se confirmó."
            return await tool.execute(**kwargs)

        # RiskLevel.DANGEROUS — nunca se ejecuta desde el loop automatizado, ni con
        # confirmación de un solo paso (`.claude/rules/security.md`). No hay branch que llame
        # a `tool.execute()` acá, deliberadamente.
        return (
            f"Acción '{tool.name}' clasificada como DANGEROUS: JARVIS no la ejecuta "
            "automáticamente. Requiere ejecución manual fuera del agente."
        )
