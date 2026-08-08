"""Tests para el contrato base `Tool` (`jarvis.tools.base`, ADR-0005)."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from jarvis.tools.base import RiskLevel, Tool


def test_tool_subclass_without_risk_raises_type_error_at_definition() -> None:
    """Caso rechazado: una subclase de `Tool` que no declara `risk` falla al definirse (import
    time), no hereda silenciosamente SAFE (`.claude/rules/security.md`)."""
    with pytest.raises(TypeError, match="risk"):

        class ToolWithoutRisk(Tool):
            name = "no_risk"
            description = "tool sin risk declarado"
            parameters: ClassVar[dict[str, Any]] = {}

            async def execute(self, **kwargs: Any) -> str:
                return "nunca debería instanciarse"


def test_tool_subclass_with_risk_defines_normally() -> None:
    """Caso aceptado: una subclase que sí declara `risk` se define sin error y expone el nivel
    declarado como atributo de clase."""

    class ToolWithRisk(Tool):
        name = "has_risk"
        description = "tool con risk declarado"
        parameters: ClassVar[dict[str, Any]] = {}
        risk = RiskLevel.SAFE

        async def execute(self, **kwargs: Any) -> str:
            return "ok"

    assert ToolWithRisk.risk is RiskLevel.SAFE


def test_describe_default_includes_name_and_all_arguments() -> None:
    """`Tool.describe()` (usado por `PolicyEngine` en el prompt CONFIRM/DANGEROUS) debe mostrar
    tanto el nombre del tool como sus argumentos concretos — no alcanza con el nombre solo
    (hallazgo de `security-reviewer`: el usuario tiene que saber sobre qué target concreto va a
    actuar antes de confirmar)."""

    class ToolWithArgs(Tool):
        name = "delete_file"
        description = "tool de prueba"
        parameters: ClassVar[dict[str, Any]] = {}
        risk = RiskLevel.CONFIRM

        async def execute(self, **kwargs: Any) -> str:
            return "ok"

    description = ToolWithArgs().describe({"path": "informe_final.pdf"})

    assert "delete_file" in description
    assert "informe_final.pdf" in description


def test_describe_default_with_no_arguments_is_just_the_name() -> None:
    """Caso límite: sin `kwargs`, el default no agrega paréntesis vacíos ni ruido."""

    class ToolWithoutArgs(Tool):
        name = "no_args_tool"
        description = "tool de prueba"
        parameters: ClassVar[dict[str, Any]] = {}
        risk = RiskLevel.CONFIRM

        async def execute(self, **kwargs: Any) -> str:
            return "ok"

    assert ToolWithoutArgs().describe({}) == "no_args_tool"
