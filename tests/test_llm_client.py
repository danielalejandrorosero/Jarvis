"""Tests para `DeepSeekClient.complete()` y `load_deepseek_client_from_env()`.

No hacen ninguna llamada de red real ni construyen un `openai.OpenAI` real contra la API de
DeepSeek: `DeepSeekClient` recibe un stub controlado por el test cuyo
`.chat.completions.create()` está mockeado (mismo enfoque que
`tests/audio/test_wake_word.py`/`test_stt.py`: núcleo puro con dependencia externa stubeada).

Regla dura (`.claude/rules/security.md`, "Higiene de secretos"): nunca imprimir ni incluir en
asserts el valor real de `DEEPSEEK_API_KEY`. El valor usado en el test de éxito de
`load_deepseek_client_from_env()` es un placeholder inventado, no una key real.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from jarvis.llm.client import DeepSeekClient, load_deepseek_client_from_env


def _fake_response(content: str | None) -> MagicMock:
    """Construye un stub de `ChatCompletion` con `.choices[0].message.content = content`."""
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def _client_with(response: MagicMock) -> tuple[DeepSeekClient, MagicMock]:
    """Devuelve un `DeepSeekClient` cuyo `._client.chat.completions.create()` es un mock que
    siempre devuelve `response`, junto con ese mock para poder inspeccionar la llamada (spy)."""
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.return_value = response
    return DeepSeekClient(client=fake_openai_client), fake_openai_client


def test_complete_returns_stripped_content() -> None:
    """Caso aceptado: el contenido de la respuesta se devuelve stripeado."""
    client, _fake = _client_with(_fake_response("  some reply  \n"))

    result = client.complete("hola")

    assert result == "some reply"


def test_complete_without_system_sends_only_user_message() -> None:
    """`system=None` (default): la lista de `messages` pasada a `create()` tiene un solo
    mensaje, el de usuario, sin mensaje de sistema."""
    client, fake_openai_client = _client_with(_fake_response("ok"))

    client.complete("hola")

    _args, kwargs = fake_openai_client.chat.completions.create.call_args
    messages: list[dict[str, str]] = kwargs["messages"]
    assert messages == [{"role": "user", "content": "hola"}]


def test_complete_with_system_sends_system_then_user_message() -> None:
    """`system="some persona"`: `messages` lleva el mensaje de sistema primero y el de
    usuario segundo, con los `role`/`content` correctos."""
    client, fake_openai_client = _client_with(_fake_response("ok"))

    client.complete("hola", system="some persona")

    _args, kwargs = fake_openai_client.chat.completions.create.call_args
    messages: list[dict[str, str]] = kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "some persona"},
        {"role": "user", "content": "hola"},
    ]


def test_complete_returns_empty_string_when_content_is_none() -> None:
    """Caso rechazado/límite: si la API devuelve `content=None` (ocurre en la práctica),
    `complete()` devuelve `""`, no `None`, y no lanza excepción."""
    client, _fake = _client_with(_fake_response(None))

    result = client.complete("hola")

    assert result == ""


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
