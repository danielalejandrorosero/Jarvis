"""Tests para `jarvis.memory.embeddings`.

Sin red real: el cliente `OpenAI` se reemplaza por un stub mínimo (`_FakeEmbeddingClient`) cuya
forma imita la respuesta real de `client.embeddings.create(...)` (`.data[0].embedding`) —
verificada contra el SDK instalado antes de escribir este test, no adivinada.
"""

from __future__ import annotations

import math

import pytest

from jarvis.memory.embeddings import EMBEDDING_MODEL, cosine_similarity, embed_text


class _FakeEmbeddingData:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embedding: list[float]) -> None:
        self.data = [_FakeEmbeddingData(embedding)]


class _FakeEmbeddingsResource:
    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding
        self.calls: list[dict[str, object]] = []

    def create(self, *, model: str, input: str) -> _FakeEmbeddingResponse:
        self.calls.append({"model": model, "input": input})
        return _FakeEmbeddingResponse(self._embedding)


class _FakeClient:
    def __init__(self, embedding: list[float]) -> None:
        self.embeddings = _FakeEmbeddingsResource(embedding)


# --- embed_text ----------------------------------------------------------------------------


def test_embed_text_returns_the_vector_from_the_response() -> None:
    client = _FakeClient([0.1, 0.2, 0.3])

    result = embed_text("mi color favorito es el azul", client=client)  # type: ignore[arg-type]

    assert result == [0.1, 0.2, 0.3]


def test_embed_text_passes_the_configured_model_and_the_exact_text() -> None:
    client = _FakeClient([0.0])

    embed_text("un texto cualquiera", client=client)  # type: ignore[arg-type]

    assert client.embeddings.calls == [
        {"model": EMBEDDING_MODEL, "input": "un texto cualquiera"}
    ]


# --- cosine_similarity ------------------------------------------------------------------------


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    vector = [0.5, -0.3, 0.8]

    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_similarity_of_scaled_vectors_ignores_magnitude() -> None:
    """Misma dirección, distinta magnitud — la similitud coseno mide ángulo, no longitud."""
    assert cosine_similarity([1.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)


def test_cosine_similarity_with_a_zero_vector_is_zero_not_a_division_error() -> None:
    """Caso límite: un vector de todos ceros no puede tener dirección — devuelve 0.0 en vez de
    lanzar `ZeroDivisionError`."""
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_matches_manual_calculation() -> None:
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    expected = sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )

    assert cosine_similarity(a, b) == pytest.approx(expected)
