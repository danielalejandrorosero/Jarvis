"""Tests para `OpenUrlTool` (`jarvis.tools.open_url`, ADR-0005).

`_open_in_browser` se reemplaza por un stub que registra la invocación en vez de llamar a
`webbrowser.open` — mismo enfoque de monkeypatch sobre atributos de módulo que
`test_open_app.py` usa para `_launch_shortcut`. Ningún test de este archivo abre un navegador
real.
"""

from __future__ import annotations

import asyncio

import pytest

from jarvis.tools import open_url as open_url_module
from jarvis.tools.base import RiskLevel
from jarvis.tools.open_url import OpenUrlTool


def _install_fake_open(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []

    def _fake_open(url: str) -> None:
        opened.append(url)

    monkeypatch.setattr(open_url_module, "_open_in_browser", _fake_open)
    return opened


def test_open_url_tool_declares_safe_risk() -> None:
    """ "Abrir URLs" está listado explícitamente como ejemplo SAFE en
    `.claude/rules/security.md`, igual que "abrir aplicaciones"."""
    assert OpenUrlTool.risk is RiskLevel.SAFE


def test_execute_opens_valid_https_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute(url="https://www.youtube.com"))

    assert result == "Abriendo https://www.youtube.com en el navegador."
    assert opened == ["https://www.youtube.com"]


def test_execute_opens_valid_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute(url="http://example.com"))

    assert result == "Abriendo http://example.com en el navegador."
    assert opened == ["http://example.com"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Windows/System32",
        "javascript:alert(1)",
        "steam://run/123",
        "no-es-una-url-en-absoluto",
    ],
)
def test_execute_rejects_disallowed_schemes(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Solo http/https — cualquier otro esquema (o ninguno) se rechaza sin abrir nada."""
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute(url=url))

    assert result == "No puedo abrir esa URL: solo abro sitios web (http/https)."
    assert opened == []


def test_execute_rejects_missing_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute())

    assert "URL válida" in result
    assert opened == []


def test_execute_rejects_blank_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute(url="   "))

    assert "URL válida" in result
    assert opened == []


@pytest.mark.parametrize("url", ["https://", "https:///path-sin-host"])
def test_execute_rejects_url_without_host(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Esquema http/https válido pero sin host (`netloc` vacío) — se rechaza igual."""
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute(url=url))

    assert result == "No puedo abrir esa URL: solo abro sitios web (http/https)."
    assert opened == []


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://LOCALHOST:8080/admin",
        "http://127.0.0.1/",
        "http://127.0.0.1:9000/",
        "http://192.168.1.1/",
        "http://10.0.0.5/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/",
    ],
)
def test_execute_rejects_local_and_private_hosts(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Hallazgo de security-reviewer: localhost/loopback/privado/link-local se rechaza, aunque
    el esquema sea válido — cierra el caso directo de SSRF/exfiltración vía navegación."""
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute(url=url))

    assert result == "No puedo abrir esa URL: apunta a una dirección local o interna."
    assert opened == []


def test_execute_allows_normal_public_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un dominio normal (no una IP literal) se permite — no se resuelve DNS para chequear a
    qué IP apunta, solo se bloquea el caso directo de una IP/localhost literal en la URL."""
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()

    result = asyncio.run(tool.execute(url="https://www.youtube.com/watch?v=abc"))

    assert result == "Abriendo https://www.youtube.com/watch?v=abc en el navegador."
    assert opened == ["https://www.youtube.com/watch?v=abc"]


def test_execute_rejects_overly_long_url(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _install_fake_open(monkeypatch)
    tool = OpenUrlTool()
    long_url = "https://example.com/?q=" + ("a" * 3000)

    result = asyncio.run(tool.execute(url=long_url))

    assert result == "No puedo abrir esa URL: es demasiado larga."
    assert opened == []
