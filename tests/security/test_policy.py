"""Tests para `PolicyEngine` (`jarvis.security.policy`, ADR-0005).

`PolicyEngine` es el único punto de paso entre el dispatch loop y `Tool.execute`
(`.claude/rules/architecture.md`). Estos tests verifican la garantía dura de
`.claude/rules/security.md`: SAFE ejecuta sin fricción, CONFIRM solo ejecuta con aprobación
explícita del `ConfirmationChannel` (una negativa nunca ejecuta), y DANGEROUS nunca ejecuta bajo
ninguna circunstancia — ni siquiera si el canal de confirmación aprobara.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from jarvis.security.policy import PolicyEngine
from jarvis.tools.base import RiskLevel, Tool


class _SafeTool(Tool):
    name = "safe_tool"
    description = "tool de prueba SAFE"
    parameters: ClassVar[dict[str, Any]] = {}
    risk = RiskLevel.SAFE

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, **kwargs: Any) -> str:
        self.executed = True
        return "safe executed"


class _ConfirmTool(Tool):
    name = "confirm_tool"
    description = "tool de prueba CONFIRM"
    parameters: ClassVar[dict[str, Any]] = {}
    risk = RiskLevel.CONFIRM

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, **kwargs: Any) -> str:
        self.executed = True
        return "confirm executed"


class _DangerousTool(Tool):
    name = "dangerous_tool"
    description = "tool de prueba DANGEROUS"
    parameters: ClassVar[dict[str, Any]] = {}
    risk = RiskLevel.DANGEROUS

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, **kwargs: Any) -> str:
        self.executed = True
        return "dangerous executed"


class _FixedConfirmationChannel:
    """`ConfirmationChannel` de prueba: siempre responde `answer`, y registra qué se le
    preguntó — para poder verificar que SAFE/DANGEROUS ni siquiera lo consultan."""

    def __init__(self, *, answer: bool) -> None:
        self.answer = answer
        self.asked_prompts: list[str] = []

    async def ask(self, prompt: str) -> bool:
        self.asked_prompts.append(prompt)
        return self.answer


def test_safe_tool_executes_without_asking_confirmation() -> None:
    """Caso aceptado: SAFE ejecuta directo, sin pasar por el `ConfirmationChannel`."""
    tool = _SafeTool()
    confirmation = _FixedConfirmationChannel(answer=False)
    policy = PolicyEngine(confirmation)

    result = asyncio.run(policy.authorize_and_execute(tool, {}))

    assert tool.executed
    assert result == "safe executed"
    assert confirmation.asked_prompts == []


def test_confirm_tool_executes_when_confirmation_channel_approves() -> None:
    """Caso aceptado: CONFIRM ejecuta solo después de una aprobación explícita del canal."""
    tool = _ConfirmTool()
    confirmation = _FixedConfirmationChannel(answer=True)
    policy = PolicyEngine(confirmation)

    result = asyncio.run(policy.authorize_and_execute(tool, {}))

    assert tool.executed
    assert result == "confirm executed"
    assert len(confirmation.asked_prompts) == 1


def test_confirm_tool_never_executes_when_confirmation_channel_denies() -> None:
    """Caso rechazado: si el canal de confirmación deniega (incluye la denegación por defecto
    de silencio/timeout de ADR-0004, que vive en la implementación del canal, no acá), el tool
    nunca ejecuta."""
    tool = _ConfirmTool()
    confirmation = _FixedConfirmationChannel(answer=False)
    policy = PolicyEngine(confirmation)

    result = asyncio.run(policy.authorize_and_execute(tool, {}))

    assert not tool.executed
    assert "cancelada" in result


def test_dangerous_tool_never_executes_even_when_confirmation_channel_approves() -> (
    None
):
    """Garantía dura de `.claude/rules/security.md`: DANGEROUS nunca ejecuta desde el loop
    automatizado, con o sin confirmación — ni siquiera se consulta al canal."""
    tool = _DangerousTool()
    confirmation = _FixedConfirmationChannel(answer=True)
    policy = PolicyEngine(confirmation)

    result = asyncio.run(policy.authorize_and_execute(tool, {}))

    assert not tool.executed
    assert "DANGEROUS" in result
    assert confirmation.asked_prompts == []


def test_confirm_prompt_includes_the_concrete_arguments_not_just_the_tool_name() -> (
    None
):
    """Regresión del hallazgo #3 de `security-reviewer`: el prompt de confirmación tiene que
    dejarle saber al usuario *qué* argumentos concretos va a ejecutar el tool, no solo su
    nombre — si no, un usuario podría confirmar a ciegas una acción sobre un target distinto al
    que pidió (p.ej. una transcripción mal extraída). Vía el default de `Tool.describe()`."""
    tool = _ConfirmTool()
    confirmation = _FixedConfirmationChannel(answer=True)
    policy = PolicyEngine(confirmation)

    asyncio.run(policy.authorize_and_execute(tool, {"target": "informe_final.pdf"}))

    assert len(confirmation.asked_prompts) == 1
    prompt = confirmation.asked_prompts[0]
    assert "informe_final.pdf" in prompt
    assert tool.name in prompt


# --- `on_executed` (jarvis.memory.store.tool_call_log, ver jarvis.audio.pipeline.dispatch_turn) -
# el callback tiene que reflejar exactamente "llegó a ejecutarse de verdad", sin que el llamador
# tenga que inferirlo del texto de retorno (SAFE ejecuta siempre; CONFIRM solo si aprueba; DANGEROUS
# nunca).


def test_on_executed_is_called_once_for_a_safe_tool() -> None:
    tool = _SafeTool()
    confirmation = _FixedConfirmationChannel(answer=False)
    policy = PolicyEngine(confirmation)
    calls: list[None] = []

    asyncio.run(
        policy.authorize_and_execute(tool, {}, on_executed=lambda: calls.append(None))
    )

    assert len(calls) == 1


def test_on_executed_is_called_once_for_an_approved_confirm_tool() -> None:
    tool = _ConfirmTool()
    confirmation = _FixedConfirmationChannel(answer=True)
    policy = PolicyEngine(confirmation)
    calls: list[None] = []

    asyncio.run(
        policy.authorize_and_execute(tool, {}, on_executed=lambda: calls.append(None))
    )

    assert len(calls) == 1


def test_on_executed_is_never_called_for_a_denied_confirm_tool() -> None:
    """El caso central que motiva este callback: una llamada CONFIRM denegada nunca llega a
    `Tool.execute()`, así que `on_executed` tampoco se dispara — sin que el llamador tenga que
    parsear el texto `"...cancelada: no se confirmó."` para saberlo."""
    tool = _ConfirmTool()
    confirmation = _FixedConfirmationChannel(answer=False)
    policy = PolicyEngine(confirmation)
    calls: list[None] = []

    asyncio.run(
        policy.authorize_and_execute(tool, {}, on_executed=lambda: calls.append(None))
    )

    assert calls == []


def test_on_executed_is_never_called_for_a_dangerous_tool_even_if_channel_would_approve() -> (
    None
):
    tool = _DangerousTool()
    confirmation = _FixedConfirmationChannel(answer=True)
    policy = PolicyEngine(confirmation)
    calls: list[None] = []

    asyncio.run(
        policy.authorize_and_execute(tool, {}, on_executed=lambda: calls.append(None))
    )

    assert calls == []


def test_authorize_and_execute_without_on_executed_behaves_like_before() -> None:
    """`on_executed=None` (el default) preserva el comportamiento exacto de antes de este
    parámetro — no rompe a ningún llamador existente que no lo pase."""
    tool = _SafeTool()
    confirmation = _FixedConfirmationChannel(answer=False)
    policy = PolicyEngine(confirmation)

    result = asyncio.run(policy.authorize_and_execute(tool, {}))

    assert tool.executed
    assert result == "safe executed"


def test_on_executed_failure_never_masks_a_completed_safe_action() -> None:
    """Regresión del hallazgo de `security-reviewer`: si `on_executed` lanza (ej. `save_tool_call`
    contra SQLite bajo lock contention), la acción ya se ejecutó de verdad y el resultado tiene
    que llegar igual al usuario — un fallo del hook de logging no puede tumbar `return result`."""
    tool = _SafeTool()
    confirmation = _FixedConfirmationChannel(answer=False)
    policy = PolicyEngine(confirmation)

    def _raising_on_executed() -> None:
        raise RuntimeError("save_tool_call: database is locked")

    result = asyncio.run(
        policy.authorize_and_execute(tool, {}, on_executed=_raising_on_executed)
    )

    assert tool.executed
    assert result == "safe executed"


def test_on_executed_failure_never_masks_a_completed_confirm_action() -> None:
    """Mismo caso que el anterior pero para CONFIRM ya aprobado — el escenario concreto que
    motivó el hallazgo (ej. `lock_lol_champion`, una acción sin deshacer garantizado): el usuario
    tiene que enterarse igual de que la acción se ejecutó, aunque el logging haya fallado."""
    tool = _ConfirmTool()
    confirmation = _FixedConfirmationChannel(answer=True)
    policy = PolicyEngine(confirmation)

    def _raising_on_executed() -> None:
        raise RuntimeError("save_tool_call: database is locked")

    result = asyncio.run(
        policy.authorize_and_execute(tool, {}, on_executed=_raising_on_executed)
    )

    assert tool.executed
    assert result == "confirm executed"
