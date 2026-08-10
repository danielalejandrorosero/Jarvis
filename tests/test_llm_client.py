"""Tests para `DeepSeekClient.complete()` y `load_deepseek_client_from_env()`.

No hacen ninguna llamada de red real ni construyen un `openai.OpenAI` real contra la API de
DeepSeek: `DeepSeekClient` recibe un stub controlado por el test cuyo
`.chat.completions.create()` está mockeado (mismo enfoque que
`tests/audio/test_wake_word.py`/`test_stt.py`: núcleo puro con dependencia externa stubeada).

ADR-0005: `complete()` cambió de contrato — de `complete(prompt, system=...) -> str` a
`complete(messages, tools=...) -> LLMResult`. Estos tests cubren el nuevo contrato: texto plano
sin tools, y la detección/parseo de un `tool_call` cuando el provider lo devuelve.

Regla dura (`.claude/rules/security.md`, "Higiene de secretos"): nunca imprimir ni incluir en
asserts el valor real de `DEEPSEEK_API_KEY`. El valor usado en el test de éxito de
`load_deepseek_client_from_env()` es un placeholder inventado, no una key real.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jarvis.llm.client import (
    MAX_COMPLETION_TOKENS,
    TEMPERATURE,
    DeepSeekClient,
    LLMResult,
    ToolCall,
    ToolSchema,
    load_deepseek_client_from_env,
)


def _fake_response(
    content: str | None,
    *,
    tool_calls: list[MagicMock] | None = None,
    reasoning_content: str | None = None,
) -> MagicMock:
    """Construye un stub de `ChatCompletion` con `.choices[0].message.content = content` y,
    opcionalmente, `.choices[0].message.tool_calls`.

    `reasoning_content=None` (default) simula el caso normal: la API no devolvió ese campo extra
    (`_fake_response` lo setea explícito, en vez de dejar que `MagicMock` autogenere un atributo
    truthy en cualquier acceso, que rompería la premisa de estos tests) — solo los tests que
    ejercitan `reasoning_content` lo pasan explícito, simulando un modelo "reasoner"/"thinking"
    que sí separa razonamiento de respuesta final (ver `DeepSeekClient._log_reasoning_content` /
    `jarvis.llm.client._log_reasoning_content`)."""
    response = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    response.choices[0].message.reasoning_content = reasoning_content
    return response


def _fake_tool_call(*, call_id: str, name: str, arguments: str) -> MagicMock:
    """Stub de un `tool_call` individual tal como lo devuelve el SDK de `openai`."""
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


def _client_with(response: MagicMock) -> tuple[DeepSeekClient, MagicMock]:
    """Devuelve un `DeepSeekClient` cuyo `._client.chat.completions.create()` es un mock que
    siempre devuelve `response`, junto con ese mock para poder inspeccionar la llamada (spy)."""
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.return_value = response
    return DeepSeekClient(client=fake_openai_client), fake_openai_client


def test_complete_returns_stripped_text_when_no_tool_call() -> None:
    """Caso aceptado: sin `tool_calls` en la respuesta, `complete()` devuelve un `LLMResult`
    con el texto stripeado y `tool_call=None`."""
    client, _fake = _client_with(_fake_response("  some reply  \n"))

    result = client.complete([{"role": "user", "content": "hola"}])

    assert result == LLMResult(text="some reply", tool_call=None)


def test_complete_sends_messages_through_unchanged() -> None:
    """La lista de `messages` pasada por el caller llega tal cual a `create()` — `complete()`
    no la reconstruye ni le agrega/saca mensajes."""
    client, fake_openai_client = _client_with(_fake_response("ok"))
    messages = [
        {"role": "system", "content": "some persona"},
        {"role": "user", "content": "hola"},
    ]

    client.complete(messages)

    _args, kwargs = fake_openai_client.chat.completions.create.call_args
    assert kwargs["messages"] == messages


def test_complete_sends_deterministic_temperature() -> None:
    """`temperature` (`jarvis.llm.client.TEMPERATURE`) se manda siempre — el default de la API
    (1.0) es demasiado alto para selección de tool confiable, ver comentario junto a la
    constante."""
    client, fake_openai_client = _client_with(_fake_response("ok"))

    client.complete([{"role": "user", "content": "hola"}])

    _args, kwargs = fake_openai_client.chat.completions.create.call_args
    assert kwargs["temperature"] == TEMPERATURE
    assert "tools" not in kwargs


def test_complete_sends_a_hard_output_token_cap() -> None:
    """`max_tokens` (`jarvis.llm.client.MAX_COMPLETION_TOKENS`) se manda siempre — defensa de
    raíz contra una respuesta que se extiende sin límite (ver comentario junto a la constante):
    estructuralmente imposible que el modelo genere más de esto en una sola llamada, sin importar
    qué tan mal siga la instrucción de `SYSTEM_PROMPT` de responder corto."""
    client, fake_openai_client = _client_with(_fake_response("ok"))

    client.complete([{"role": "user", "content": "hola"}])

    _args, kwargs = fake_openai_client.chat.completions.create.call_args
    assert kwargs["max_tokens"] == MAX_COMPLETION_TOKENS


def test_complete_passes_tool_schemas_in_openai_wire_format() -> None:
    """Cuando se pasan `tools`, `complete()` los traduce al formato `{"type": "function", ...}`
    que espera el SDK de `openai` — la traducción es responsabilidad de `DeepSeekClient`, no del
    caller."""
    client, fake_openai_client = _client_with(_fake_response("ok"))
    schema = ToolSchema(
        name="get_weather",
        description="clima de una ciudad",
        parameters={"type": "object", "properties": {}},
    )

    client.complete([{"role": "user", "content": "hola"}], tools=[schema])

    _args, kwargs = fake_openai_client.chat.completions.create.call_args
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "clima de una ciudad",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_complete_returns_empty_text_when_content_is_none_and_no_tool_call() -> None:
    """Caso rechazado/límite: si la API devuelve `content=None` sin `tool_calls`, `complete()`
    devuelve `text=""`, no `None`, y no lanza excepción."""
    client, _fake = _client_with(_fake_response(None))

    result = client.complete([{"role": "user", "content": "hola"}])

    assert result == LLMResult(text="", tool_call=None)


def test_complete_parses_tool_call_when_present() -> None:
    """Caso aceptado: si la respuesta trae `tool_calls`, `complete()` devuelve un `LLMResult`
    con un `ToolCall` parseado (id, nombre, argumentos ya deserializados de JSON), no texto."""
    tool_call = _fake_tool_call(
        call_id="call_123", name="get_weather", arguments='{"city": "Madrid"}'
    )
    client, _fake = _client_with(_fake_response(None, tool_calls=[tool_call]))

    result = client.complete([{"role": "user", "content": "clima en Madrid"}])

    assert result == LLMResult(
        text="",
        tool_call=ToolCall(
            id="call_123", name="get_weather", arguments={"city": "Madrid"}
        ),
    )


def test_complete_parses_tool_call_with_empty_arguments_as_empty_dict() -> None:
    """Caso límite: `arguments=""` (tool sin parámetros) se parsea como `{}`, no lanza
    `json.JSONDecodeError`."""
    tool_call = _fake_tool_call(call_id="call_456", name="no_args_tool", arguments="")
    client, _fake = _client_with(_fake_response(None, tool_calls=[tool_call]))

    result = client.complete([{"role": "user", "content": "hola"}])

    assert result.tool_call == ToolCall(
        id="call_456", name="no_args_tool", arguments={}
    )


def test_complete_degrades_gracefully_on_invalid_json_arguments_instead_of_raising() -> (
    None
):
    """Regresión del hallazgo #2 de `security-reviewer`: `arguments` con JSON inválido no debe
    propagar `json.JSONDecodeError` — `complete()` devuelve un `ToolCall` con `arguments={}` y
    `arguments_error` describiendo el problema, para que el dispatch loop lo pueda degradar a un
    mensaje de error en vez de tumbar el proceso."""
    tool_call = _fake_tool_call(
        call_id="call_bad_json", name="get_weather", arguments="{not valid json"
    )
    client, _fake = _client_with(_fake_response(None, tool_calls=[tool_call]))

    result = client.complete([{"role": "user", "content": "clima"}])

    assert result.tool_call is not None
    assert result.tool_call.arguments == {}
    assert result.tool_call.arguments_error is not None
    assert "inválido" in result.tool_call.arguments_error


def test_complete_degrades_gracefully_when_arguments_parse_to_non_dict() -> None:
    """Regresión del hallazgo #2: JSON válido pero que no es un objeto (acá, una lista) tampoco
    debe llegar como `arguments` a `Tool.execute(**kwargs)` — `**` sobre una lista lanzaría
    `TypeError` sin este chequeo. `complete()` degrada a `arguments={}` +
    `arguments_error` en vez de dejar pasar un valor no desempaquetable."""
    tool_call = _fake_tool_call(
        call_id="call_list_args", name="get_weather", arguments='["Madrid"]'
    )
    client, _fake = _client_with(_fake_response(None, tool_calls=[tool_call]))

    result = client.complete([{"role": "user", "content": "clima"}])

    assert result.tool_call is not None
    assert result.tool_call.arguments == {}
    assert result.tool_call.arguments_error is not None
    assert "objeto JSON" in result.tool_call.arguments_error


def test_complete_parsed_tool_call_has_no_arguments_error_on_valid_dict() -> None:
    """Caso aceptado (contraste con los dos anteriores): argumentos válidos no disparan
    `arguments_error` — `None` es el valor por defecto, no un string vacío ni otro sentinel."""
    tool_call = _fake_tool_call(
        call_id="call_ok", name="get_weather", arguments='{"city": "Madrid"}'
    )
    client, _fake = _client_with(_fake_response(None, tool_calls=[tool_call]))

    result = client.complete([{"role": "user", "content": "clima"}])

    assert result.tool_call is not None
    assert result.tool_call.arguments_error is None


def test_complete_never_includes_reasoning_content_in_returned_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regresión de bug confirmado en vivo: JARVIS llegó a hablarle al usuario un párrafo de
    razonamiento interno ("El usuario está respondiendo... Debo pedir aclaración...") antes de la
    respuesta real. Si el proveedor separa ese razonamiento del texto final en un campo propio
    (`reasoning_content` — DeepSeek lo usa para sus modelos "reasoner"/"thinking", y el SDK de
    `openai` lo preserva como atributo extra vía `pydantic`/`extra=\"allow\"`, aunque no sea parte
    del schema oficial de OpenAI), `complete()` nunca debe mezclarlo con `text` — se descarta del
    resultado y se loguea aparte (nivel debug, nunca print/TTS) para debugging."""
    caplog.set_level("DEBUG", logger="jarvis.llm.client")
    client, _fake = _client_with(
        _fake_response(
            "Me alegro de que te haya quedado bien, Daniel.",
            reasoning_content=(
                "El usuario está respondiendo 'Ese está mejor' refiriéndose "
                "probablemente a algo anterior..."
            ),
        )
    )

    result = client.complete([{"role": "user", "content": "Ese está mejor."}])

    assert result == LLMResult(
        text="Me alegro de que te haya quedado bien, Daniel.", tool_call=None
    )
    assert "El usuario está respondiendo" not in result.text
    assert any(
        "razonamiento" in record.message
        and "El usuario está respondiendo" in record.message
        for record in caplog.records
    )


def test_complete_omits_reasoning_log_when_field_absent() -> None:
    """Caso normal (sin modelo `reasoner`): sin `reasoning_content` en la respuesta, `complete()`
    no loguea nada — no hay ruido de diagnóstico cuando no hay nada que reportar."""
    client, _fake = _client_with(_fake_response("respuesta normal"))

    result = client.complete([{"role": "user", "content": "hola"}])

    assert result == LLMResult(text="respuesta normal", tool_call=None)


def test_load_deepseek_client_from_env_raises_runtime_error_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso rechazado: sin `DEEPSEEK_API_KEY` seteada, la factory falla explícito en vez de
    construir un cliente inválido."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        load_deepseek_client_from_env()

    # Solo se verifica que hay un mensaje de error, nunca se imprime/compara un valor de key.
    assert str(exc_info.value)


def test_load_deepseek_client_from_env_constructs_client_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caso aceptado de la factory: con la key seteada, arma un `DeepSeekClient` envolviendo un
    `openai.OpenAI` — sin pegarle a la red real, `openai.OpenAI` se mockea en el punto donde
    `jarvis.llm.client` lo importa."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder-key-for-test")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    fake_openai_instance = MagicMock()

    def _fake_openai_ctor(*, api_key: str, base_url: str) -> Any:
        assert base_url == "https://api.deepseek.com"
        return fake_openai_instance

    monkeypatch.setattr("jarvis.llm.client.OpenAI", _fake_openai_ctor)

    client = load_deepseek_client_from_env()

    assert isinstance(client, DeepSeekClient)
