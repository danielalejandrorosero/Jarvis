"""Tests para `jarvis.memory.store` (ADR-0004, "Persistencia: SQLite").

`sqlite3` es stdlib, no red ni hardware — a diferencia de otros tests del repo que stubean
dependencias externas, acá se usa un archivo SQLite real en `tmp_path` (misma filosofía que
testear una función de filesystem: rápido, determinístico, sin mocks de por medio).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.memory import store
from jarvis.memory.store import (
    list_facts,
    list_speech_samples,
    save_fact,
    save_speech_sample,
)


def test_list_facts_on_nonexistent_db_file_returns_empty_and_creates_it(
    tmp_path: Path,
) -> None:
    """Sin ningún `save_fact` previo, el archivo de DB no existe todavía — `list_facts` no
    lanza, devuelve `[]`, y deja la DB (con la tabla `facts`) inicializada para el próximo uso."""
    db_path = tmp_path / "jarvis.db"
    assert not db_path.exists()

    result = list_facts(db_path=db_path)

    assert result == []
    assert db_path.exists()


def test_save_fact_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    """`save_fact` crea el directorio contenedor si todavía no existe (caso `data/` recién
    clonado el repo, sin ese directorio creado)."""
    db_path = tmp_path / "nested" / "dir" / "jarvis.db"
    assert not db_path.parent.exists()

    save_fact("el usuario prefiere respuestas cortas", db_path=db_path)

    assert db_path.exists()
    assert list_facts(db_path=db_path) == ["el usuario prefiere respuestas cortas"]


def test_save_and_list_facts_round_trips_content(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    save_fact("le gusta el café sin azúcar", db_path=db_path)

    assert list_facts(db_path=db_path) == ["le gusta el café sin azúcar"]


def test_save_fact_strips_surrounding_whitespace(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    save_fact("  hecho con espacios alrededor  ", db_path=db_path)

    assert list_facts(db_path=db_path) == ["hecho con espacios alrededor"]


def test_save_fact_rejects_blank_content(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    try:
        save_fact("   ", db_path=db_path)
    except ValueError:
        pass
    else:
        raise AssertionError("se esperaba ValueError para contenido vacío")
    assert list_facts(db_path=db_path) == []


def test_list_facts_returns_most_recent_first(tmp_path: Path) -> None:
    """Orden documentado: más reciente primero, no orden de inserción."""
    db_path = tmp_path / "jarvis.db"
    save_fact("primero", db_path=db_path)
    save_fact("segundo", db_path=db_path)
    save_fact("tercero", db_path=db_path)

    assert list_facts(db_path=db_path) == ["tercero", "segundo", "primero"]


def test_list_facts_respects_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    for index in range(5):
        save_fact(f"hecho {index}", db_path=db_path)

    result = list_facts(db_path=db_path, limit=2)

    assert result == ["hecho 4", "hecho 3"]


def test_list_facts_default_limit_caps_at_default_list_limit(tmp_path: Path) -> None:
    """Sin pasar `limit`, no se devuelven más de `DEFAULT_LIST_LIMIT` hechos — evita que el
    contexto que se inyecta en el system prompt crezca sin tope turno a turno."""
    db_path = tmp_path / "jarvis.db"
    total = store.DEFAULT_LIST_LIMIT + 5
    for index in range(total):
        save_fact(f"hecho {index}", db_path=db_path)

    result = list_facts(db_path=db_path)

    assert len(result) == store.DEFAULT_LIST_LIMIT
    # El más reciente (el último guardado) sigue siendo el primero de la lista.
    assert result[0] == f"hecho {total - 1}"


# --- MAX_CONTENT_LENGTH / MAX_STORED_FACTS (hallazgos LOW #2 y #3 de `security-reviewer`) ------


def test_save_fact_truncates_content_longer_than_max_content_length(
    tmp_path: Path,
) -> None:
    """Un `content` más largo que `MAX_CONTENT_LENGTH` (ej. el LLM "recordando" un fragmento
    entero de una página web) se trunca antes de guardarse — nunca se persiste, ni se reinyecta
    en un prompt futuro, contenido de largo arbitrario (hallazgo LOW #2, amplifica el hallazgo
    HIGH #1)."""
    db_path = tmp_path / "jarvis.db"
    overlong = "x" * (store.MAX_CONTENT_LENGTH + 200)

    save_fact(overlong, db_path=db_path)

    (stored,) = list_facts(db_path=db_path)
    assert len(stored) <= store.MAX_CONTENT_LENGTH + 1  # +1: el indicador de corte "…"
    assert stored.endswith("…")
    assert stored != overlong


def test_save_fact_leaves_content_within_max_content_length_unchanged(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    short = "un hecho normal, bien corto"

    save_fact(short, db_path=db_path)

    assert list_facts(db_path=db_path) == [short]


def test_save_fact_prunes_oldest_rows_beyond_max_stored_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sin tope, `facts` crecería sin límite con el uso prolongado (hallazgo LOW #3) — cada
    escritura poda lo más viejo por encima de `MAX_STORED_FACTS`, conservando los más recientes."""
    db_path = tmp_path / "jarvis.db"
    monkeypatch.setattr(store, "MAX_STORED_FACTS", 3)

    for index in range(5):
        save_fact(f"hecho {index}", db_path=db_path)

    result = list_facts(db_path=db_path, limit=10)

    assert result == ["hecho 4", "hecho 3", "hecho 2"]


# --- speech_samples (log automático, sin curación del LLM — distinto de `facts`) ----------------


def test_list_speech_samples_on_nonexistent_db_file_returns_empty_and_creates_it(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    assert not db_path.exists()

    result = list_speech_samples(db_path=db_path)

    assert result == []
    assert db_path.exists()


def test_save_speech_sample_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "jarvis.db"
    assert not db_path.parent.exists()

    save_speech_sample("ey parce qué más", db_path=db_path)

    assert db_path.exists()
    assert list_speech_samples(db_path=db_path) == ["ey parce qué más"]


def test_save_and_list_speech_samples_round_trips_text(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    save_speech_sample("hágale pues, dale con eso", db_path=db_path)

    assert list_speech_samples(db_path=db_path) == ["hágale pues, dale con eso"]


def test_save_speech_sample_strips_surrounding_whitespace(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    save_speech_sample("  con espacios alrededor  ", db_path=db_path)

    assert list_speech_samples(db_path=db_path) == ["con espacios alrededor"]


def test_save_speech_sample_rejects_blank_text(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    try:
        save_speech_sample("   ", db_path=db_path)
    except ValueError:
        pass
    else:
        raise AssertionError("se esperaba ValueError para texto vacío")
    assert list_speech_samples(db_path=db_path) == []


def test_list_speech_samples_returns_most_recent_first(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    save_speech_sample("primero", db_path=db_path)
    save_speech_sample("segundo", db_path=db_path)
    save_speech_sample("tercero", db_path=db_path)

    assert list_speech_samples(db_path=db_path) == ["tercero", "segundo", "primero"]


def test_list_speech_samples_respects_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    for index in range(5):
        save_speech_sample(f"muestra {index}", db_path=db_path)

    result = list_speech_samples(db_path=db_path, limit=2)

    assert result == ["muestra 4", "muestra 3"]


def test_list_speech_samples_default_limit_caps_at_default_speech_sample_list_limit(
    tmp_path: Path,
) -> None:
    """Sin pasar `limit`, no se devuelven más de `DEFAULT_SPEECH_SAMPLE_LIST_LIMIT` muestras —
    deliberadamente chico frente a `DEFAULT_LIST_LIMIT`: unos pocos ejemplos de estilo recientes,
    no un transcript completo, ver docstring del módulo."""
    db_path = tmp_path / "jarvis.db"
    total = store.DEFAULT_SPEECH_SAMPLE_LIST_LIMIT + 5
    for index in range(total):
        save_speech_sample(f"muestra {index}", db_path=db_path)

    result = list_speech_samples(db_path=db_path)

    assert len(result) == store.DEFAULT_SPEECH_SAMPLE_LIST_LIMIT
    assert result[0] == f"muestra {total - 1}"


def test_save_speech_sample_truncates_text_longer_than_max_speech_sample_length(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    overlong = "x" * (store.MAX_SPEECH_SAMPLE_LENGTH + 200)

    save_speech_sample(overlong, db_path=db_path)

    (stored,) = list_speech_samples(db_path=db_path)
    assert (
        len(stored) <= store.MAX_SPEECH_SAMPLE_LENGTH + 1
    )  # +1: el indicador de corte "…"
    assert stored.endswith("…")
    assert stored != overlong


def test_save_speech_sample_leaves_text_within_max_length_unchanged(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    short = "una muestra normal, bien corta"

    save_speech_sample(short, db_path=db_path)

    assert list_speech_samples(db_path=db_path) == [short]


def test_save_speech_sample_prunes_oldest_rows_beyond_max_stored_speech_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "jarvis.db"
    monkeypatch.setattr(store, "MAX_STORED_SPEECH_SAMPLES", 3)

    for index in range(5):
        save_speech_sample(f"muestra {index}", db_path=db_path)

    result = list_speech_samples(db_path=db_path, limit=10)

    assert result == ["muestra 4", "muestra 3", "muestra 2"]


def test_speech_samples_and_facts_are_independent_tables(tmp_path: Path) -> None:
    """Regresión de layout: guardar en una tabla no afecta la otra, ambas conviven en el mismo
    archivo de DB (ADR-0004, mismo `db_path`)."""
    db_path = tmp_path / "jarvis.db"

    save_fact("un hecho", db_path=db_path)
    save_speech_sample("una muestra de habla", db_path=db_path)

    assert list_facts(db_path=db_path) == ["un hecho"]
    assert list_speech_samples(db_path=db_path) == ["una muestra de habla"]
