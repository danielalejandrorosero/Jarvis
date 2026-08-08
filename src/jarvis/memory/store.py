"""Persistencia de memoria de JARVIS: hechos sueltos en texto plano sobre el usuario (ADR-0004,
"Persistencia: SQLite").

Una sola tabla, sin esquema clave-valor rígido: quien decide *qué* vale la pena recordar y con
qué frase es el LLM (`jarvis.tools.remember.RememberTool`), no este módulo — así que el store
solo necesita texto libre con marca de tiempo, no columnas tipadas por concepto. Nada de RAG ni
embeddings: el pedido es "acordate de lo que te dije", no búsqueda semántica sobre un corpus
grande — `sqlite3` de stdlib alcanza (ADR-0004 ya descartó Postgres: sin proceso de servidor para
una app local de un solo usuario).

Sin ORM ni dependencia nueva: `sqlite3` (stdlib) es suficiente para una tabla y dos queries.

Hallazgos LOW de `security-reviewer` (junto al hallazgo HIGH mitigado en
`jarvis.audio.pipeline._build_system_prompt`, ver docstring de ese módulo): sin tope de longitud,
un `content` arbitrariamente largo (ej. el LLM "recordando" un fragmento entero de una página web)
se guardaría y reinyectaría entero en cada prompt futuro, amplificando el riesgo de contenido no
confiable persistente — `MAX_CONTENT_LENGTH` lo trunca, mismo principio que
`CONTENT_SNIPPET_MAX_CHARS`/`ANSWER_MAX_CHARS` en `jarvis.tools.search`. `MAX_STORED_FACTS` es
higiene de almacenamiento, no mitigación de seguridad primaria: sin tope, la tabla crece sin
límite con el uso prolongado; `save_fact` poda las filas más viejas por encima de ese tope en cada
escritura.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

# Relativo al directorio desde el que se corre JARVIS (mismo criterio que
# `jarvis.config.load_dotenv`'s default `.env`), no un path absoluto hardcodeado — así se puede
# sobreescribir pasando `db_path` explícito (tests con `tmp_path`, u otro layout de despliegue)
# sin tocar este módulo.
DEFAULT_DB_PATH = Path("data/jarvis.db")

# Tope por defecto de `list_facts`: evita que el system prompt crezca sin límite turno a turno a
# medida que se acumulan hechos — un recorte razonable de contexto reciente, no el historial
# completo.
DEFAULT_LIST_LIMIT = 20

# Tope duro de longitud por hecho (hallazgo LOW #2 de `security-reviewer`): un hecho se trunca a
# esto antes de guardarse, nunca se persiste (ni se reinyecta) contenido de largo arbitrario.
MAX_CONTENT_LENGTH = 500

# Tope duro sobre el total histórico de filas en `facts` (hallazgo LOW #3, menor, no bloqueante):
# `save_fact` poda las más viejas por encima de este número en cada escritura, así la tabla nunca
# crece sin límite con el uso prolongado.
MAX_STORED_FACTS = 500

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Abrir una conexión a `db_path`, creando el directorio contenedor y la tabla `facts` si
    todavía no existen — cubre tanto el primer uso (archivo de DB inexistente) como usos
    posteriores (tabla ya creada, `CREATE TABLE IF NOT EXISTS` es un no-op)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _truncate(text: str, *, max_chars: int) -> str:
    """Capear `text` a `max_chars`, con un indicador de corte — mismo helper que
    `jarvis.tools.search._truncate` (duplicado a propósito, ver `_escape_untrusted` en
    `jarvis.audio.pipeline` para el mismo razonamiento: no es un contrato compartido entre
    módulos)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def save_fact(content: str, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Guardar `content` (un hecho en texto plano, ej. "el usuario prefiere respuestas cortas")
    con la marca de tiempo actual (UTC, ISO 8601), truncado a `MAX_CONTENT_LENGTH`.

    Lanza `ValueError` si `content` está vacío o es solo espacios — validación mínima acá; el
    mensaje amigable de "no se especificó qué recordar" es responsabilidad del tool
    (`jarvis.tools.remember.RememberTool`), no de este módulo de persistencia.

    Tras insertar, poda las filas más viejas por encima de `MAX_STORED_FACTS` (hallazgo LOW #3)
    — en la misma transacción, así una escritura nunca deja la tabla momentáneamente por encima
    del tope.
    """
    stripped = content.strip()
    if not stripped:
        raise ValueError("content no puede estar vacío")
    truncated = _truncate(stripped, max_chars=MAX_CONTENT_LENGTH)
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO facts (content, created_at) VALUES (?, ?)",
            (truncated, datetime.now(UTC).isoformat()),
        )
        conn.execute(
            "DELETE FROM facts WHERE id NOT IN "
            "(SELECT id FROM facts ORDER BY id DESC LIMIT ?)",
            (MAX_STORED_FACTS,),
        )
        conn.commit()
    finally:
        conn.close()


def list_facts(
    *, db_path: str | Path = DEFAULT_DB_PATH, limit: int = DEFAULT_LIST_LIMIT
) -> list[str]:
    """Devolver hasta `limit` hechos guardados, más reciente primero.

    Sobre un `db_path` que todavía no existe (nunca se llamó `save_fact` con este path), inicializa
    la DB vacía igual que `_connect` y devuelve `[]` — no es un error, es el estado inicial
    esperado antes de la primera memoria guardada.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT content FROM facts ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()
