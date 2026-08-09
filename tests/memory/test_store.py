"""Tests para `jarvis.memory.store` (ADR-0004, "Persistencia: SQLite").

`sqlite3` es stdlib, no red ni hardware — a diferencia de otros tests del repo que stubean
dependencias externas, acá se usa un archivo SQLite real en `tmp_path` (misma filosofía que
testear una función de filesystem: rápido, determinístico, sin mocks de por medio).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jarvis.memory import store
from jarvis.memory.store import (
    ConversationTurn,
    Reminder,
    ToolCallLogEntry,
    delete_reminder,
    list_due_reminders,
    list_facts,
    list_most_recent_tool_call_per_tool,
    list_recent_conversation_turns,
    list_reminders,
    list_speech_samples,
    save_conversation_turn,
    save_fact,
    save_reminder,
    save_speech_sample,
    save_tool_call,
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


# --- reminders (persistidos, anunciados proactivamente por `TimerScheduler`) --------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_list_reminders_on_nonexistent_db_file_returns_empty_and_creates_it(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    assert not db_path.exists()

    result = list_reminders(db_path=db_path)

    assert result == []
    assert db_path.exists()


def test_save_reminder_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(minutes=10))
    assert not db_path.parent.exists()

    save_reminder("llamar a mamá", due_at, db_path=db_path)

    assert db_path.exists()
    (reminder,) = list_reminders(db_path=db_path)
    assert reminder.text == "llamar a mamá"
    assert reminder.due_at == due_at


def test_save_and_list_reminders_round_trips(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(hours=1))

    save_reminder("sacar la comida del horno", due_at, db_path=db_path)

    (reminder,) = list_reminders(db_path=db_path)
    assert isinstance(reminder, Reminder)
    assert reminder.text == "sacar la comida del horno"
    assert reminder.due_at == due_at


def test_save_reminder_strips_surrounding_whitespace(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(minutes=5))

    save_reminder("  con espacios alrededor  ", due_at, db_path=db_path)

    (reminder,) = list_reminders(db_path=db_path)
    assert reminder.text == "con espacios alrededor"


def test_save_reminder_rejects_blank_text(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(minutes=5))

    with pytest.raises(ValueError, match="text"):
        save_reminder("   ", due_at, db_path=db_path)
    assert list_reminders(db_path=db_path) == []


def test_save_reminder_rejects_unparseable_due_at(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    with pytest.raises(ValueError, match="due_at"):
        save_reminder("algo", "no es una fecha", db_path=db_path)
    assert list_reminders(db_path=db_path) == []


def test_save_reminder_rejects_naive_due_at_without_timezone(tmp_path: Path) -> None:
    """Un `due_at` sin offset de zona horaria no se puede comparar de forma inequívoca contra el
    `now` aware que usa `list_due_reminders` — se rechaza en la escritura, no falla en silencio
    más tarde dentro del poll loop de `TimerScheduler`."""
    db_path = tmp_path / "jarvis.db"
    naive_due_at = datetime.now().isoformat()  # noqa: DTZ005 — sin tzinfo a propósito, es el caso bajo test

    with pytest.raises(ValueError, match="zona horaria"):
        save_reminder("algo", naive_due_at, db_path=db_path)
    assert list_reminders(db_path=db_path) == []


def test_save_reminder_truncates_text_longer_than_max_reminder_text_length(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(minutes=5))
    overlong = "x" * (store.MAX_REMINDER_TEXT_LENGTH + 200)

    save_reminder(overlong, due_at, db_path=db_path)

    (reminder,) = list_reminders(db_path=db_path)
    assert len(reminder.text) <= store.MAX_REMINDER_TEXT_LENGTH + 1
    assert reminder.text.endswith("…")


def test_save_reminder_prunes_farthest_due_at_rows_beyond_max_stored_reminders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Poda por `due_at` (más lejano primero), no por orden de inserción — en este caso
    coinciden (índice más alto = insertado después = vence más tarde), así que sirve como caso
    base; `test_save_reminder_prunes_by_due_at_not_insertion_order` cubre el caso donde
    insertar-más-tarde y vencer-más-tarde DIVERGEN (el hallazgo LOW real)."""
    db_path = tmp_path / "jarvis.db"
    monkeypatch.setattr(store, "MAX_STORED_REMINDERS", 3)
    base = datetime.now(UTC)

    for index in range(5):
        save_reminder(
            f"recordatorio {index}",
            _iso(base + timedelta(minutes=index)),
            db_path=db_path,
        )

    result = list_reminders(db_path=db_path)

    assert [r.text for r in result] == [
        "recordatorio 0",
        "recordatorio 1",
        "recordatorio 2",
    ]


def test_save_reminder_prunes_by_due_at_not_insertion_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regresión del hallazgo LOW de `security-reviewer`: el recordatorio más viejo *insertado*
    puede tener el `due_at` más próximo. Podar por `id`/orden de inserción (comportamiento viejo)
    evictaría justo ese, el más urgente, en vez del que vence más lejos en el futuro."""
    db_path = tmp_path / "jarvis.db"
    monkeypatch.setattr(store, "MAX_STORED_REMINDERS", 3)
    base = datetime.now(UTC)

    # Insertados del due_at más lejano al más próximo — orden de inserción inverso al orden de
    # urgencia.
    save_reminder("el más lejano", _iso(base + timedelta(hours=5)), db_path=db_path)
    save_reminder(
        "el segundo más lejano", _iso(base + timedelta(hours=4)), db_path=db_path
    )
    save_reminder("el del medio", _iso(base + timedelta(hours=3)), db_path=db_path)
    save_reminder(
        "el segundo más próximo", _iso(base + timedelta(hours=2)), db_path=db_path
    )
    save_reminder("el más próximo", _iso(base + timedelta(hours=1)), db_path=db_path)

    result = list_reminders(db_path=db_path)

    assert [r.text for r in result] == [
        "el más próximo",
        "el segundo más próximo",
        "el del medio",
    ]


def test_list_reminders_orders_by_due_at_ascending_soonest_first(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    base = datetime.now(UTC)
    # Guardados fuera de orden a propósito: el orden devuelto tiene que ser por `due_at`, no por
    # orden de inserción.
    save_reminder("el más tarde", _iso(base + timedelta(hours=2)), db_path=db_path)
    save_reminder("el más pronto", _iso(base + timedelta(minutes=5)), db_path=db_path)
    save_reminder("el del medio", _iso(base + timedelta(hours=1)), db_path=db_path)

    result = list_reminders(db_path=db_path)

    assert [r.text for r in result] == ["el más pronto", "el del medio", "el más tarde"]


def test_list_due_reminders_returns_only_reminders_at_or_before_now(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    base = datetime.now(UTC)
    save_reminder("ya venció", _iso(base - timedelta(minutes=1)), db_path=db_path)
    save_reminder("vence justo ahora", _iso(base), db_path=db_path)
    save_reminder("todavía no", _iso(base + timedelta(minutes=10)), db_path=db_path)

    due = list_due_reminders(base, db_path=db_path)

    assert {r.text for r in due} == {"ya venció", "vence justo ahora"}


def test_list_due_reminders_returns_empty_when_nothing_is_due_yet(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    base = datetime.now(UTC)
    save_reminder("todavía no", _iso(base + timedelta(minutes=10)), db_path=db_path)

    due = list_due_reminders(base, db_path=db_path)

    assert due == []


def test_list_due_reminders_orders_by_due_at_ascending(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    base = datetime.now(UTC)
    save_reminder("segundo", _iso(base - timedelta(minutes=1)), db_path=db_path)
    save_reminder("primero", _iso(base - timedelta(minutes=5)), db_path=db_path)

    due = list_due_reminders(base, db_path=db_path)

    assert [r.text for r in due] == ["primero", "segundo"]


def test_delete_reminder_removes_it_so_it_is_never_listed_again(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    base = datetime.now(UTC)
    save_reminder(
        "recordatorio único", _iso(base - timedelta(minutes=1)), db_path=db_path
    )
    (reminder,) = list_due_reminders(base, db_path=db_path)

    delete_reminder(reminder.id, db_path=db_path)

    assert list_reminders(db_path=db_path) == []
    assert list_due_reminders(base, db_path=db_path) == []


def test_delete_reminder_on_nonexistent_id_is_idempotent_and_does_not_raise(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"

    delete_reminder(999, db_path=db_path)  # no debe lanzar, aunque la fila no exista

    assert list_reminders(db_path=db_path) == []


def test_reminders_are_independent_from_facts_and_speech_samples(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(minutes=5))

    save_fact("un hecho", db_path=db_path)
    save_speech_sample("una muestra de habla", db_path=db_path)
    save_reminder("un recordatorio", due_at, db_path=db_path)

    assert list_facts(db_path=db_path) == ["un hecho"]
    assert list_speech_samples(db_path=db_path) == ["una muestra de habla"]
    assert [r.text for r in list_reminders(db_path=db_path)] == ["un recordatorio"]


# --- conversation_turns (historial de turnos, pedido explícito: "que se acuerde que dije antes") -


def test_list_recent_conversation_turns_on_nonexistent_db_file_returns_empty_and_creates_it(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    assert not db_path.exists()

    result = list_recent_conversation_turns(db_path=db_path)

    assert result == []
    assert db_path.exists()


def test_save_conversation_turn_creates_parent_directory_if_missing(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "nested" / "dir" / "jarvis.db"
    assert not db_path.parent.exists()

    save_conversation_turn("abrí YouTube", "Listo.", db_path=db_path)

    assert db_path.exists()
    assert list_recent_conversation_turns(db_path=db_path) == [
        ConversationTurn(user_text="abrí YouTube", assistant_text="Listo.")
    ]


def test_save_and_list_conversation_turns_round_trips(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    save_conversation_turn(
        "qué clima hace en Madrid", "En Madrid está soleado.", db_path=db_path
    )

    assert list_recent_conversation_turns(db_path=db_path) == [
        ConversationTurn(
            user_text="qué clima hace en Madrid",
            assistant_text="En Madrid está soleado.",
        )
    ]


def test_save_conversation_turn_strips_surrounding_whitespace_on_both_fields(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"

    save_conversation_turn("  con espacios  ", "  también acá  ", db_path=db_path)

    (turn,) = list_recent_conversation_turns(db_path=db_path)
    assert turn.user_text == "con espacios"
    assert turn.assistant_text == "también acá"


def test_save_conversation_turn_rejects_blank_user_text(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    with pytest.raises(ValueError, match="user_text"):
        save_conversation_turn("   ", "algo", db_path=db_path)
    assert list_recent_conversation_turns(db_path=db_path) == []


def test_save_conversation_turn_allows_blank_assistant_text(tmp_path: Path) -> None:
    """A diferencia de `user_text` (y de `save_fact`/`save_speech_sample`), `assistant_text`
    vacío es un caso legítimo: `SYSTEM_PROMPT` (`jarvis.audio.pipeline`) instruye al LLM a
    responder con la cadena vacía cuando el turno es "reproducir algo", para no hablar encima de
    lo que empieza a sonar — ese silencio es información real del turno, no un dato inválido."""
    db_path = tmp_path / "jarvis.db"

    save_conversation_turn("reproducí tal canción", "", db_path=db_path)

    (turn,) = list_recent_conversation_turns(db_path=db_path)
    assert turn.user_text == "reproducí tal canción"
    assert turn.assistant_text == ""


def test_list_recent_conversation_turns_returns_most_recent_first(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    save_conversation_turn("primero", "respuesta 1", db_path=db_path)
    save_conversation_turn("segundo", "respuesta 2", db_path=db_path)
    save_conversation_turn("tercero", "respuesta 3", db_path=db_path)

    result = list_recent_conversation_turns(db_path=db_path)

    assert [t.user_text for t in result] == ["tercero", "segundo", "primero"]


def test_list_recent_conversation_turns_respects_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    for index in range(5):
        save_conversation_turn(f"turno {index}", f"respuesta {index}", db_path=db_path)

    result = list_recent_conversation_turns(db_path=db_path, limit=2)

    assert [t.user_text for t in result] == ["turno 4", "turno 3"]


def test_list_recent_conversation_turns_default_limit_caps_at_default(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    total = store.DEFAULT_CONVERSATION_TURN_LIST_LIMIT + 5
    for index in range(total):
        save_conversation_turn(f"turno {index}", f"respuesta {index}", db_path=db_path)

    result = list_recent_conversation_turns(db_path=db_path)

    assert len(result) == store.DEFAULT_CONVERSATION_TURN_LIST_LIMIT
    assert result[0].user_text == f"turno {total - 1}"


def test_save_conversation_turn_truncates_fields_longer_than_max_length(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    overlong_user = "u" * (store.MAX_CONVERSATION_TURN_TEXT_LENGTH + 200)
    overlong_assistant = "a" * (store.MAX_CONVERSATION_TURN_TEXT_LENGTH + 200)

    save_conversation_turn(overlong_user, overlong_assistant, db_path=db_path)

    (turn,) = list_recent_conversation_turns(db_path=db_path)
    assert len(turn.user_text) <= store.MAX_CONVERSATION_TURN_TEXT_LENGTH + 1
    assert turn.user_text.endswith("…")
    assert len(turn.assistant_text) <= store.MAX_CONVERSATION_TURN_TEXT_LENGTH + 1
    assert turn.assistant_text.endswith("…")


def test_save_conversation_turn_prunes_oldest_rows_beyond_max_stored_conversation_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "jarvis.db"
    monkeypatch.setattr(store, "MAX_STORED_CONVERSATION_TURNS", 3)

    for index in range(5):
        save_conversation_turn(f"turno {index}", f"respuesta {index}", db_path=db_path)

    result = list_recent_conversation_turns(db_path=db_path, limit=10)

    assert [t.user_text for t in result] == ["turno 4", "turno 3", "turno 2"]


def test_conversation_turns_survive_reconnecting_to_the_same_db_file(
    tmp_path: Path,
) -> None:
    """Prueba de supervivencia a un reinicio del proceso: guardar, cerrar toda conexión abierta
    (cada función de este módulo abre y cierra su propia conexión, no hay una compartida de larga
    vida) y volver a leer desde el mismo archivo en disco tiene que devolver lo mismo — es la
    garantía real detrás de "sobrevive un reinicio de JARVIS", sin necesidad de reiniciar el
    proceso de verdad."""
    db_path = tmp_path / "jarvis.db"

    save_conversation_turn("abrí YouTube", "Listo.", db_path=db_path)
    save_conversation_turn("qué hora es", "Son las 10.", db_path=db_path)

    # Nueva "sesión": ninguna conexión de la escritura anterior sigue abierta acá, se reabre el
    # mismo archivo desde cero, como haría un proceso de JARVIS recién arrancado.
    result = list_recent_conversation_turns(db_path=db_path)

    assert [t.user_text for t in result] == ["qué hora es", "abrí YouTube"]
    assert [t.assistant_text for t in result] == ["Son las 10.", "Listo."]


def test_conversation_turns_are_independent_from_other_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(minutes=5))

    save_fact("un hecho", db_path=db_path)
    save_speech_sample("una muestra de habla", db_path=db_path)
    save_reminder("un recordatorio", due_at, db_path=db_path)
    save_conversation_turn("un turno", "una respuesta", db_path=db_path)

    assert list_facts(db_path=db_path) == ["un hecho"]
    assert list_speech_samples(db_path=db_path) == ["una muestra de habla"]
    assert [r.text for r in list_reminders(db_path=db_path)] == ["un recordatorio"]
    assert list_recent_conversation_turns(db_path=db_path) == [
        ConversationTurn(user_text="un turno", assistant_text="una respuesta")
    ]


# --- tool_call_log (log automático de tool-calls que llegaron a ejecutarse; resumen "última fila
# por tool" resuelve la referencia "la última canción"/"el mismo modo de LoL", ver docstring del
# módulo) -------------------------------------------------------------------------------------


def test_list_most_recent_tool_call_per_tool_on_nonexistent_db_returns_empty_and_creates_it(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    assert not db_path.exists()

    result = list_most_recent_tool_call_per_tool(db_path=db_path)

    assert result == []
    assert db_path.exists()


def test_save_tool_call_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "jarvis.db"
    assert not db_path.parent.exists()

    save_tool_call(
        "open_url", {"url": "https://youtube.com/watch?v=abc"}, db_path=db_path
    )

    assert db_path.exists()
    (entry,) = list_most_recent_tool_call_per_tool(db_path=db_path)
    assert entry.tool_name == "open_url"


def test_save_and_list_tool_call_round_trips_arguments_as_json(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    save_tool_call("set_lol_lobby_queue", {"queue_type": "arena"}, db_path=db_path)

    (entry,) = list_most_recent_tool_call_per_tool(db_path=db_path)
    assert isinstance(entry, ToolCallLogEntry)
    assert entry.tool_name == "set_lol_lobby_queue"
    assert json.loads(entry.arguments_json) == {"queue_type": "arena"}


def test_save_tool_call_strips_surrounding_whitespace_on_tool_name(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"

    save_tool_call("  open_url  ", {"url": "https://example.com"}, db_path=db_path)

    (entry,) = list_most_recent_tool_call_per_tool(db_path=db_path)
    assert entry.tool_name == "open_url"


def test_save_tool_call_rejects_blank_tool_name(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"

    with pytest.raises(ValueError, match="tool_name"):
        save_tool_call("   ", {}, db_path=db_path)
    assert list_most_recent_tool_call_per_tool(db_path=db_path) == []


def test_save_tool_call_truncates_arguments_longer_than_max_length(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    overlong_value = "x" * (store.MAX_TOOL_CALL_ARGUMENTS_LENGTH + 200)

    save_tool_call("search_web", {"query": overlong_value}, db_path=db_path)

    (entry,) = list_most_recent_tool_call_per_tool(db_path=db_path)
    assert len(entry.arguments_json) <= store.MAX_TOOL_CALL_ARGUMENTS_LENGTH + 1
    assert entry.arguments_json.endswith("…")


def test_save_tool_call_prunes_oldest_rows_beyond_max_stored_tool_call_log_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "jarvis.db"
    monkeypatch.setattr(store, "MAX_STORED_TOOL_CALL_LOG_ROWS", 3)

    for index in range(5):
        save_tool_call(f"tool_{index}", {"n": index}, db_path=db_path)

    result = list_most_recent_tool_call_per_tool(db_path=db_path)

    assert {entry.tool_name for entry in result} == {"tool_2", "tool_3", "tool_4"}


def test_list_most_recent_tool_call_per_tool_keeps_only_latest_call_of_same_tool(
    tmp_path: Path,
) -> None:
    """El insight de diseño clave: varias llamadas al MISMO tool solo dejan UNA fila en el
    resumen — la más reciente, no un historial que crece con cada llamada."""
    db_path = tmp_path / "jarvis.db"

    save_tool_call("set_lol_lobby_queue", {"queue_type": "aram"}, db_path=db_path)
    save_tool_call(
        "set_lol_lobby_queue", {"queue_type": "ranked_solo_duo"}, db_path=db_path
    )
    save_tool_call("set_lol_lobby_queue", {"queue_type": "arena"}, db_path=db_path)

    result = list_most_recent_tool_call_per_tool(db_path=db_path)

    assert len(result) == 1
    (entry,) = result
    assert entry.tool_name == "set_lol_lobby_queue"
    assert json.loads(entry.arguments_json) == {"queue_type": "arena"}


def test_list_most_recent_tool_call_per_tool_gives_each_distinct_tool_its_own_row(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"

    save_tool_call("open_url", {"url": "https://a.example"}, db_path=db_path)
    save_tool_call("set_lol_lobby_queue", {"queue_type": "arena"}, db_path=db_path)
    save_tool_call("open_url", {"url": "https://b.example"}, db_path=db_path)
    save_tool_call("volume_control", {"action": "mute"}, db_path=db_path)

    result = list_most_recent_tool_call_per_tool(db_path=db_path)

    by_name = {entry.tool_name: json.loads(entry.arguments_json) for entry in result}
    assert by_name == {
        "open_url": {"url": "https://b.example"},
        "set_lol_lobby_queue": {"queue_type": "arena"},
        "volume_control": {"action": "mute"},
    }


def test_list_most_recent_tool_call_per_tool_summary_never_grows_beyond_distinct_tool_count(
    tmp_path: Path,
) -> None:
    """La garantía central de este resumen: sin importar cuántas veces se llame al MISMO puñado
    de tools, el bloque nunca crece más allá de un renglón por nombre distinto."""
    db_path = tmp_path / "jarvis.db"
    tool_names = ["open_url", "set_lol_lobby_queue", "volume_control"]

    for index in range(30):
        save_tool_call(
            tool_names[index % len(tool_names)], {"n": index}, db_path=db_path
        )

    result = list_most_recent_tool_call_per_tool(db_path=db_path)

    assert len(result) == len(tool_names)


def test_tool_call_log_is_independent_from_other_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    due_at = _iso(datetime.now(UTC) + timedelta(minutes=5))

    save_fact("un hecho", db_path=db_path)
    save_speech_sample("una muestra de habla", db_path=db_path)
    save_reminder("un recordatorio", due_at, db_path=db_path)
    save_conversation_turn("un turno", "una respuesta", db_path=db_path)
    save_tool_call("open_url", {"url": "https://example.com"}, db_path=db_path)

    assert list_facts(db_path=db_path) == ["un hecho"]
    assert list_speech_samples(db_path=db_path) == ["una muestra de habla"]
    assert [r.text for r in list_reminders(db_path=db_path)] == ["un recordatorio"]
    assert list_recent_conversation_turns(db_path=db_path) == [
        ConversationTurn(user_text="un turno", assistant_text="una respuesta")
    ]
    (entry,) = list_most_recent_tool_call_per_tool(db_path=db_path)
    assert entry.tool_name == "open_url"
