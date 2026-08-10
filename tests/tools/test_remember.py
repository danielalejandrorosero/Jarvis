"""Tests para `RememberTool` (`jarvis.tools.remember`).

Sin red ni mocks de `sqlite3`: local, stdlib, rápido — un archivo SQLite real en `tmp_path`, mismo
enfoque que `tests/memory/test_store.py`. `RememberTool` toma `db_path` en el constructor (ver
docstring de `__init__`), así que cada test instancia el tool apuntando a una DB de `tmp_path` sin
tocar el `data/jarvis.db` real del repo.

`embedding_client` (opcional) se cubre con un stub mínimo (`_FakeEmbeddingClient`) — misma forma
que `tests/memory/test_embeddings.py` — sin red real.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from jarvis.memory.store import (
    MAX_CONTENT_LENGTH,
    list_facts,
    list_facts_with_embeddings,
)
from jarvis.tools.base import RiskLevel
from jarvis.tools.remember import RememberTool


class _FakeEmbeddingData:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embedding: list[float]) -> None:
        self.data = [_FakeEmbeddingData(embedding)]


class _FakeEmbeddingsResource:
    def __init__(self, *, embedding: list[float] | None, raises: bool) -> None:
        self._embedding = embedding
        self._raises = raises
        self.received_inputs: list[str] = []

    def create(self, *, model: str, input: str) -> _FakeEmbeddingResponse:
        self.received_inputs.append(input)
        if self._raises:
            raise RuntimeError("falla simulada de la API de embeddings")
        assert self._embedding is not None
        return _FakeEmbeddingResponse(self._embedding)


class _FakeEmbeddingClient:
    def __init__(
        self, *, embedding: list[float] | None = None, raises: bool = False
    ) -> None:
        self.embeddings = _FakeEmbeddingsResource(embedding=embedding, raises=raises)


def test_remember_tool_declares_safe_risk() -> None:
    """Escribe en el store SQLite propio de JARVIS, no en un archivo/proceso real del usuario —
    ver el comentario de `risk` en `jarvis.tools.remember` para el razonamiento completo."""
    assert RememberTool.risk is RiskLevel.SAFE


def test_execute_saves_fact_and_confirms(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    result = asyncio.run(tool.execute(content="el usuario prefiere respuestas cortas"))

    assert "el usuario prefiere respuestas cortas" in result
    assert list_facts(db_path=db_path) == ["el usuario prefiere respuestas cortas"]


def test_execute_strips_whitespace_before_saving(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    asyncio.run(tool.execute(content="  hecho con espacios  "))

    assert list_facts(db_path=db_path) == ["hecho con espacios"]


def test_execute_rejects_missing_content_without_touching_the_store(
    tmp_path: Path,
) -> None:
    """Caso rechazado: sin `content` (o vacío), `execute()` no escribe nada en el store y
    devuelve un mensaje de error claro."""
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    result = asyncio.run(tool.execute())

    assert "No se especificó qué recordar" in result
    assert not db_path.exists()


def test_execute_rejects_blank_content_without_touching_the_store(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    result = asyncio.run(tool.execute(content="   "))

    assert "No se especificó qué recordar" in result
    assert not db_path.exists()


# --- embedding_client (búsqueda semántica, `jarvis.memory.embeddings`) --------------------------


def test_execute_without_embedding_client_saves_fact_without_one(
    tmp_path: Path,
) -> None:
    """Sin `embedding_client` (default `None`, comportamiento de antes de esta feature), el
    hecho se guarda igual, sin embedding — no intenta ninguna llamada de red."""
    db_path = tmp_path / "jarvis.db"
    tool = RememberTool(db_path=db_path)

    asyncio.run(tool.execute(content="hecho sin cliente de embeddings"))

    assert list_facts(db_path=db_path) == ["hecho sin cliente de embeddings"]
    assert list_facts_with_embeddings(db_path=db_path) == []


def test_execute_with_embedding_client_saves_the_computed_embedding(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    client = _FakeEmbeddingClient(embedding=[0.1, 0.2, 0.3])
    tool = RememberTool(db_path=db_path, embedding_client=client)  # type: ignore[arg-type]

    asyncio.run(tool.execute(content="a Daniel le gusta el azul"))

    assert list_facts_with_embeddings(db_path=db_path) == [
        ("a Daniel le gusta el azul", [0.1, 0.2, 0.3])
    ]


def test_execute_saves_fact_even_when_embedding_api_call_fails(
    tmp_path: Path,
) -> None:
    """Caso central: una falla de red/API de embeddings (cuota, timeout, lo que sea) nunca
    impide guardar el hecho en sí — degrada a guardarlo sin embedding, no pierde el dato."""
    db_path = tmp_path / "jarvis.db"
    client = _FakeEmbeddingClient(raises=True)
    tool = RememberTool(db_path=db_path, embedding_client=client)  # type: ignore[arg-type]

    result = asyncio.run(tool.execute(content="hecho a pesar del error"))

    assert "hecho a pesar del error" in result
    assert list_facts(db_path=db_path) == ["hecho a pesar del error"]
    assert list_facts_with_embeddings(db_path=db_path) == []


def test_execute_truncates_content_before_embedding_it(tmp_path: Path) -> None:
    """Hallazgo MEDIUM de `security-reviewer`: el embedding se calcula sobre el mismo texto
    truncado que termina persistido, no sobre el `content` sin capear — así no hay desajuste
    semántico entre lo que representa el embedding y lo que `RecallMemoryTool` puede recuperar."""
    db_path = tmp_path / "jarvis.db"
    client = _FakeEmbeddingClient(embedding=[0.1, 0.2])
    tool = RememberTool(db_path=db_path, embedding_client=client)  # type: ignore[arg-type]
    long_content = "a" * (MAX_CONTENT_LENGTH + 300)

    asyncio.run(tool.execute(content=long_content))

    expected = long_content[:MAX_CONTENT_LENGTH]
    assert client.embeddings.received_inputs == [expected]
    assert list_facts(db_path=db_path) == [expected]
    assert list_facts_with_embeddings(db_path=db_path) == [(expected, [0.1, 0.2])]
