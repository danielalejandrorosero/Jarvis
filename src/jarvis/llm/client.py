"""Cliente LLM detrás de una interfaz swappable (ADR-0004).

DeepSeek es el proveedor actual, pero el resto del código no debe depender de su SDK
directamente — solo de `LLMClient`. Motivo registrado en el ADR: DeepSeek anunció una suba de
precios significativa sin fecha ni cifra confirmada; si hay que migrar de proveedor, el cambio
queda contenido acá.
"""

from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI

DEEPSEEK_BASE_URL_DEFAULT = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


class LLMClient(Protocol):
    """Contrato mínimo que cualquier proveedor de LLM debe cumplir."""

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Devolver la respuesta del modelo a `prompt`, dado un system prompt opcional."""
        ...


class DeepSeekClient:
    """Implementación de `LLMClient` sobre la API de DeepSeek (formato compatible con OpenAI)."""

    def __init__(self, *, client: OpenAI, model: str = DEEPSEEK_MODEL) -> None:
        self._client = client
        self._model = model

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(model=self._model, messages=messages)  # type: ignore[arg-type]
        content = response.choices[0].message.content
        return content.strip() if content else ""


def load_deepseek_client_from_env() -> DeepSeekClient:
    """Construir un `DeepSeekClient` leyendo `DEEPSEEK_API_KEY`/`DEEPSEEK_BASE_URL` de `os.environ`.

    Asume que `jarvis.config.load_dotenv()` ya corrió (o que las vars ya están seteadas de otra
    forma). Nunca imprime ni loguea el valor de la API key (`.claude/rules/security.md`).
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY no está seteada. Corré jarvis.config.load_dotenv() primero, o "
            "exportá la variable manualmente."
        )
    base_url = os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL_DEFAULT)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return DeepSeekClient(client=client)
