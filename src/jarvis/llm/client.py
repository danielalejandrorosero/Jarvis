"""Cliente LLM detrás de una interfaz swappable (ADR-0004, extendida por ADR-0005).

DeepSeek es el proveedor actual, pero el resto del código no debe depender de su SDK
directamente — solo de `LLMClient`. Motivo registrado en el ADR: DeepSeek anunció una suba de
precios significativa sin fecha ni cifra confirmada; si hay que migrar de proveedor, el cambio
queda contenido acá.

ADR-0005: `complete()` cambia de contrato (no aditivo) para soportar tool-calling — de
`complete(prompt, system=...) -> str` a `complete(messages, tools=...) -> LLMResult`. `messages`
usa el formato de mensajes por rol (`role`/`content`, más `tool_calls`/`tool_call_id` cuando
aplica) que es el estándar de facto entre proveedores compatibles con OpenAI — no es
DeepSeek-específico; lo DeepSeek-específico (traducir `ToolSchema` al `tools=` del SDK, parsear
`response.tool_calls`) vive únicamente en `DeepSeekClient`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL_DEFAULT = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


@dataclass(frozen=True)
class ToolSchema:
    """Descripción de un tool, agnóstica de proveedor, para function-calling.

    Mismo shape que los atributos públicos de `jarvis.tools.base.Tool` (`name`/`description`/
    `parameters`), pero sin importar esa clase — mantiene `llm/client.py` usable de forma
    independiente de `jarvis.tools`. Quien arma la lista de `ToolSchema` a partir de los `Tool`
    registrados es el dispatch loop (`pipeline.py`), no este módulo.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """Solicitud estructurada del LLM para invocar un tool: id de la llamada (necesario para
    correlacionar la respuesta `role: tool` siguiente), nombre del tool y argumentos ya
    parseados a dict.

    `arguments` es *siempre* un `dict[str, Any]` — nunca `None`, una lista, ni ningún otro tipo
    — para que `tool.execute(**kwargs)` nunca reciba algo no desempaquetable. Si el LLM devolvió
    JSON inválido, o JSON válido pero no un objeto (p.ej. una lista o un string), `arguments`
    queda vacío y `arguments_error` describe qué falló: el dispatch loop (`pipeline.py`) debe
    chequear `arguments_error` antes de autorizar/ejecutar y, si no es `None`, devolverle ese
    error al LLM como mensaje `role: tool` en vez de intentar ejecutar el tool — nunca se deja
    propagar una excepción de parseo hasta `run()`.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    arguments_error: str | None = None


@dataclass(frozen=True)
class LLMResult:
    """Resultado de `LLMClient.complete()`.

    Si `tool_call` no es `None`, el LLM pidió invocar un tool y `text` no es la respuesta
    final del turno (puede venir vacío) — el dispatch loop debe ejecutar el tool, devolver su
    resultado como mensaje `role: tool`, y llamar a `complete()` de nuevo para obtener el texto
    final. Si `tool_call` es `None`, `text` es la respuesta final para hablar por TTS.
    """

    text: str
    tool_call: ToolCall | None


class LLMClient(Protocol):
    """Contrato mínimo que cualquier proveedor de LLM debe cumplir."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolSchema] | None = None,
    ) -> LLMResult:
        """Continuar la conversación `messages` (lista de dicts `role`/`content`, con
        `tool_calls`/`tool_call_id` cuando aplica) y devolver texto final o una solicitud de
        tool-call estructurada.
        """
        ...


def _parse_tool_arguments(
    raw_arguments: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Parsear el string de argumentos crudo que devuelve el SDK a un `dict`, sin nunca lanzar.

    Un LLM puede devolver JSON inválido, o JSON válido que no es un objeto (una lista, un
    string, `null`) — ninguno de esos casos debe propagar una excepción hasta `dispatch_turn()`/
    `run()` (`run()` solo atrapa `KeyboardInterrupt`, así que cualquier excepción acá tumbaría
    todo el proceso de JARVIS). Devuelve `({}, "descripción del error")` en vez de lanzar; el
    caller decide qué hacer con el error (`ToolCall.arguments_error`).
    """
    if not raw_arguments:
        return {}, None
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return {}, f"JSON inválido en los argumentos del tool: {exc}"
    if not isinstance(parsed, dict):
        return {}, (
            "Los argumentos del tool deben ser un objeto JSON, se recibió "
            f"{type(parsed).__name__}."
        )
    return parsed, None


def _log_reasoning_content(message: Any) -> None:
    """Loguear (nunca hablar ni mostrar) el razonamiento interno del modelo, si el proveedor lo
    devolvió en un campo separado de `content`.

    Motivo: los modelos "reasoner"/"thinking" de DeepSeek devuelven su cadena de razonamiento en
    `reasoning_content`, un campo propio de la API de DeepSeek que no existe en el schema de
    OpenAI — pero como `openai.OpenAI` (el SDK que usamos para hablar con la API, compatible con
    OpenAI) modela sus respuestas con `pydantic` y `extra="allow"` (`openai._models.BaseModel`),
    ese campo extra queda accesible vía atributo en vez de perderse. Nunca se mezcla con `content`
    en ningún punto de este módulo — `text` de `LLMResult` siempre sale de `message.content` en
    limpio — así que esta función es puramente defensiva/de diagnóstico: cubre el caso de que el
    alias de modelo configurado (`DEEPSEEK_MODEL`) empiece a devolver razonamiento en este campo
    server-side (p.ej. si DeepSeek migra qué modelo concreto resuelve `deepseek-chat`), sin que
    haga falta ningún cambio de código para que ese contenido siga sin llegar nunca a TTS.

    Bug real que motivó esto (no hipotético): JARVIS llegó a hablarle al usuario un párrafo de
    análisis en tercera persona ("El usuario está respondiendo...") seguido recién después de la
    respuesta real — pero para el modelo/momento en que ocurrió, `reasoning_content` no estaba
    presente en la respuesta (confirmado leyendo este mismo método: solo toca `message.content`,
    nunca ningún otro campo); el texto de análisis venía ya mezclado *dentro* de `content`. La
    causa real de ese caso es una instrucción faltante en `SYSTEM_PROMPT`
    (`jarvis.audio.pipeline`), no un bug de extracción acá — pero como el proveedor sí soporta un
    campo separado para otros modelos/variantes, esta función cierra ese otro vector por completo
    en vez de asumir que nunca va a pasar.
    """
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        logger.debug("(razonamiento, no hablado) %s", reasoning_content)


def _to_openai_tool_param(schema: ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


class DeepSeekClient:
    """Implementación de `LLMClient` sobre la API de DeepSeek (formato compatible con OpenAI)."""

    def __init__(self, *, client: OpenAI, model: str = DEEPSEEK_MODEL) -> None:
        self._client = client
        self._model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolSchema] | None = None,
    ) -> LLMResult:
        create_kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            create_kwargs["tools"] = [_to_openai_tool_param(schema) for schema in tools]
        response = self._client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        _log_reasoning_content(message)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            first_call = tool_calls[0]
            arguments, arguments_error = _parse_tool_arguments(
                first_call.function.arguments
            )
            content = message.content
            return LLMResult(
                text=content.strip() if content else "",
                tool_call=ToolCall(
                    id=first_call.id,
                    name=first_call.function.name,
                    arguments=arguments,
                    arguments_error=arguments_error,
                ),
            )
        content = message.content
        return LLMResult(text=content.strip() if content else "", tool_call=None)


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
