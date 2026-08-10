"""Tests para el contrato de estado compartido `jarvis.ui.status` (ADR-0008).

Cubre `write_status`/`read_status` (roundtrip, resiliencia ante archivo ausente/corrupto/con
campos faltantes, truncado de `last_text`, y que un fallo real de I/O nunca se propaga), `is_stale`
(función pura) y `StatusHeartbeat` (el hilo de fondo que mantiene el archivo fresco durante
esperas largas sin transición real — ver docstring de la clase para el hallazgo en vivo que la
motiva). Todo corre contra `tmp_path` — nunca toca `data/status.json` del repo real, mismo
criterio que el resto de `tests/` sobre paths de estado en runtime.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from jarvis.ui.status import (
    MAX_LAST_TEXT_LENGTH,
    StatusHeartbeat,
    StatusSnapshot,
    StatusState,
    is_stale,
    read_status,
    write_status,
)


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    write_status(StatusState.LISTENING, "abrí discord", path=status_path)

    snapshot = read_status(status_path)

    assert snapshot is not None
    assert snapshot.state is StatusState.LISTENING
    assert snapshot.last_text == "abrí discord"
    assert snapshot.timestamp > 0


def test_write_status_creates_parent_directory(tmp_path: Path) -> None:
    status_path = tmp_path / "nested" / "dir" / "status.json"

    write_status(StatusState.IDLE, "", path=status_path)

    assert status_path.exists()


def test_write_status_truncates_last_text(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    long_text = "x" * (MAX_LAST_TEXT_LENGTH * 2)

    write_status(StatusState.THINKING, long_text, path=status_path)

    snapshot = read_status(status_path)
    assert snapshot is not None
    assert len(snapshot.last_text) == MAX_LAST_TEXT_LENGTH


def test_write_status_never_raises_on_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frontera de recuperación (ver docstring del módulo): un fallo de disco real nunca debe
    propagar — se loguea y se descarta, el llamador (`run()`) nunca ve una excepción."""
    status_path = tmp_path / "status.json"

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disco lleno, simulado")

    monkeypatch.setattr(Path, "write_text", _boom)

    write_status(StatusState.SPEAKING, "no debería crashear", path=status_path)


def test_write_status_never_raises_on_invalid_state(tmp_path: Path) -> None:
    """Frontera de recuperación: un `state` que no es un `StatusState` (sin `.value`) dispara
    `AttributeError` al armar el payload — antes del fix, eso pasaba *fuera* del `try`, rompiendo
    el contrato de "nunca lanza" que documenta la función."""
    status_path = tmp_path / "status.json"

    class _NotAStatusState:
        pass

    write_status(_NotAStatusState(), "texto", path=status_path)  # type: ignore[arg-type]

    assert read_status(status_path) is None


def test_write_status_never_raises_on_non_str_last_text(tmp_path: Path) -> None:
    """Un `last_text` no-`str` (ej. `None`) dispara `TypeError` al intentar hacer slicing —
    también debe degradarse en silencio, no propagar."""
    status_path = tmp_path / "status.json"

    write_status(StatusState.IDLE, None, path=status_path)  # type: ignore[arg-type]

    assert read_status(status_path) is None


def test_write_status_concurrent_calls_same_path_do_not_collide(tmp_path: Path) -> None:
    """El escenario real que motiva el nombre de archivo temporal único: `StatusHeartbeat.update()`
    puede ser llamado desde el thread principal de `run()` y desde su propio thread de fondo,
    ambos escribiendo al mismo `path` casi al mismo tiempo. Con un nombre de `.tmp` fijo, esto
    podía colisionar en Windows (`PermissionError`/`WinError 32`, silenciosamente absorbido por el
    `except` de `write_status`) y perder una actualización de estado. Con nombres únicos, ninguna
    de las escrituras concurrentes debería perderse: el archivo final siempre queda con uno de los
    estados válidos escritos por los threads, nunca corrupto ni ausente."""
    status_path = tmp_path / "status.json"
    states = [
        StatusState.LISTENING,
        StatusState.THINKING,
        StatusState.SPEAKING,
        StatusState.IDLE,
    ]

    def _write_many(state: StatusState) -> None:
        for _ in range(25):
            write_status(state, state.value, path=status_path)

    threads = [threading.Thread(target=_write_many, args=(state,)) for state in states]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Ningún `.tmp` residual: cada escritura debió completar su propio `os.replace()` (o, en el
    # peor caso, limpiar su temporal tras un fallo real) sin dejar basura en el directorio.
    leftover_tmp_files = list(tmp_path.glob("*.tmp*"))
    assert leftover_tmp_files == []

    snapshot = read_status(status_path)
    assert snapshot is not None
    assert snapshot.state in states
    assert snapshot.last_text == snapshot.state.value


def test_read_status_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_status(tmp_path / "no-existe.json") is None


def test_read_status_invalid_json_returns_none(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text("esto no es json{", encoding="utf-8")

    assert read_status(status_path) is None


def test_read_status_missing_field_returns_none(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps({"state": "idle"}), encoding="utf-8")

    assert read_status(status_path) is None


def test_read_status_unknown_state_returns_none(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps({"state": "bailando", "last_text": "", "timestamp": 1.0}),
        encoding="utf-8",
    )

    assert read_status(status_path) is None


@pytest.mark.parametrize(
    ("timestamp", "now", "max_age", "expected"),
    [
        (100.0, 100.0, 5.0, False),  # recién escrito
        (100.0, 104.9, 5.0, False),  # justo debajo del umbral
        (100.0, 105.1, 5.0, True),  # justo encima del umbral
        (
            100.0,
            90.0,
            5.0,
            False,
        ),  # timestamp "en el futuro" respecto a now — nunca stale
    ],
)
def test_is_stale(timestamp: float, now: float, max_age: float, expected: bool) -> None:
    assert is_stale(timestamp, now=now, max_age_seconds=max_age) is expected


def test_status_snapshot_is_frozen() -> None:
    snapshot = StatusSnapshot(state=StatusState.IDLE, last_text="", timestamp=1.0)
    with pytest.raises(AttributeError):
        snapshot.last_text = "otra cosa"  # type: ignore[misc]


# Intervalo muy corto (no `HEARTBEAT_INTERVAL_SECONDS` real, 2.0s) para que estos tests corran
# rápido sin ser flaky — igual de válido para probar el comportamiento del hilo, que no depende
# del valor concreto del intervalo.
_TEST_HEARTBEAT_INTERVAL_SECONDS = 0.05


def test_status_heartbeat_update_writes_immediately(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    heartbeat = StatusHeartbeat(
        path=status_path, interval_seconds=_TEST_HEARTBEAT_INTERVAL_SECONDS
    )

    heartbeat.update(StatusState.LISTENING, "hola")

    snapshot = read_status(status_path)
    assert snapshot is not None
    assert snapshot.state is StatusState.LISTENING
    assert snapshot.last_text == "hola"


def test_status_heartbeat_keeps_rewriting_without_new_update(tmp_path: Path) -> None:
    """El hallazgo real que motiva la clase: sin ningún `update()` nuevo, el archivo se sigue
    reescribiendo solo (con el mismo estado), y su timestamp avanza — así un estado de larga
    duración (ej. `idle` esperando la wake word) nunca se ve stale solo por falta de escritura."""
    status_path = tmp_path / "status.json"
    heartbeat = StatusHeartbeat(
        path=status_path, interval_seconds=_TEST_HEARTBEAT_INTERVAL_SECONDS
    )
    heartbeat.update(StatusState.IDLE, "esperando")
    first_snapshot = read_status(status_path)
    assert first_snapshot is not None

    try:
        heartbeat.start()
        deadline = time.monotonic() + 2.0
        refreshed = False
        while time.monotonic() < deadline:
            time.sleep(_TEST_HEARTBEAT_INTERVAL_SECONDS)
            snapshot = read_status(status_path)
            if snapshot is not None and snapshot.timestamp > first_snapshot.timestamp:
                refreshed = True
                assert snapshot.state is StatusState.IDLE
                assert snapshot.last_text == "esperando"
                break
    finally:
        heartbeat.stop()

    assert refreshed, "el heartbeat nunca reescribió el archivo sin un update() nuevo"


def test_status_heartbeat_stop_before_start_is_safe(tmp_path: Path) -> None:
    heartbeat = StatusHeartbeat(path=tmp_path / "status.json")
    heartbeat.stop()  # nunca arrancó — no debe lanzar


def test_status_heartbeat_start_twice_is_idempotent(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    heartbeat = StatusHeartbeat(
        path=status_path, interval_seconds=_TEST_HEARTBEAT_INTERVAL_SECONDS
    )
    try:
        heartbeat.start()
        first_thread = heartbeat._thread
        heartbeat.start()
        assert heartbeat._thread is first_thread
    finally:
        heartbeat.stop()


def test_status_heartbeat_stop_is_idempotent(tmp_path: Path) -> None:
    heartbeat = StatusHeartbeat(
        path=tmp_path / "status.json",
        interval_seconds=_TEST_HEARTBEAT_INTERVAL_SECONDS,
    )
    heartbeat.start()
    heartbeat.stop()
    heartbeat.stop()  # segunda vez no debe lanzar ni colgarse
