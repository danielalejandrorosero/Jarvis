"""Contrato base de un tool invocable por el LLM (ADR-0005).

`Tool` es deliberadamente una `ABC`, no un `Protocol` como `LLMClient`/`TTSClient`: acá el
patrón es un registro heterogéneo de N tools que comparten generación de schema y clasificación
de riesgo, no un backend único intercambiable (ver ADR-0005, "Decisión").

Ningún tool decide si está autorizado a correr — `risk` solo lo *declara*; quien decide es
`PolicyEngine` (`jarvis.security.policy`), siempre antes de `execute()`
(`.claude/rules/architecture.md`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar


class RiskLevel(Enum):
    """Clasificación de riesgo de una acción (`.claude/rules/security.md`)."""

    SAFE = "safe"
    CONFIRM = "confirm"
    DANGEROUS = "dangerous"


class Tool(ABC):
    """Una capacidad concreta que el LLM puede invocar vía function-calling.

    `name`/`description`/`parameters` alimentan el schema de function-calling que ve el LLM.
    `risk` es un atributo de clase *obligatorio*, sin valor por defecto acá: una subclase que no
    lo declare falla al definirse (`__init_subclass__`), en vez de heredar silenciosamente un
    nivel SAFE. Es la lección concreta de ADR-0002 (`Read` sin cubrir sobre `.env`) aplicada al
    propio runtime de JARVIS — un tool nuevo no puede quedar sin clasificar por omisión.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]
    risk: ClassVar[RiskLevel]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "risk"):
            raise TypeError(
                f"{cls.__name__} debe declarar `risk: RiskLevel` como atributo de clase, sin "
                "valor por defecto — ver `.claude/rules/security.md`."
            )

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Ejecutar el tool y devolver el resultado como texto plano para el LLM.

        Solo se llama después de que `PolicyEngine` autorizó la ejecución — nunca directamente
        desde el loop de dispatch.
        """
        ...

    def describe(self, kwargs: dict[str, Any]) -> str:
        """Descripción legible en una línea de esta invocación concreta (nombre + argumentos),
        usada por `PolicyEngine` en el prompt de confirmación CONFIRM/DANGEROUS — para que el
        usuario sepa *qué* va a ejecutarse, no solo el nombre del tool (hallazgo de
        `security-reviewer` en la revisión de ADR-0005: el template que copian los tools
        futuros no puede confirmar a ciegas).

        Default genérico (nombre + `clave=valor`); los tools que se clasifiquen CONFIRM/
        DANGEROUS deberían sobreescribir esto con una frase natural para lectura por voz (ej.
        "abrir el archivo informe.pdf" en vez de "open_file (path='informe.pdf')").
        """
        if not kwargs:
            return self.name
        rendered_args = ", ".join(
            f"{key}={value!r}" for key, value in sorted(kwargs.items())
        )
        return f"{self.name} ({rendered_args})"
