"""Embeddings semánticos para la memoria de hechos (`jarvis.memory.store`, tabla `facts`).

Por qué la API de OpenAI y no un modelo local: JARVIS ya depende de red para el LLM (DeepSeek) y
el STT (`gpt-4o-transcribe`, OpenAI) — exigir red acá no es una restricción nueva. Un modelo de
embeddings local (`sentence-transformers`/`fastembed`) además da peor calidad semántica en
español que un servicio entrenado en la nube con muchos más recursos de los que esta máquina
tiene (pedido explícito del usuario: mejor usar embeddings de la nube ya entrenados y buenos que
forzar el hardware local con algo peor). Reusa el mismo cliente `OpenAI` que ya construye
`jarvis.audio.stt.load_stt_client` para transcribir — no es una dependencia nueva, `openai` ya
está en `pyproject.toml`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from openai import OpenAI

# text-embedding-3-small: 1536 dimensiones, USD 0.02 / 1M tokens, soporte multilingüe razonable
# (incluye español) — el modelo más barato de OpenAI alcanza de sobra para frases cortas de
# hechos guardados (no documentos largos), no hace falta la variante "large".
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text: str, *, client: OpenAI) -> list[float]:
    """Vector de embedding de `text` vía la API de OpenAI."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return list(response.data[0].embedding)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Similitud coseno entre dos vectores: 1.0 misma dirección, 0.0 sin relación/ortogonales,
    negativo = opuestos.

    Sin numpy: son vectores cortos (1536 floats, un hecho guardado a la vez) — un loop puro de
    Python alcanza y no justifica sumar una dependencia a este módulo solo para un producto
    punto (mismo criterio que ya aplica `resample.py` para no sumar `scipy`).
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
