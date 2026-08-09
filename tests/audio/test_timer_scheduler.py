"""Tests para `TimerScheduler` (`jarvis.audio.timer_scheduler`).

Todo acá corre contra `poll_once()` con un `now` inyectado y un `tts` espía (`_SpyTTSClient`) —
nunca `threading.Timer`, nunca `time.sleep`/`Event.wait` reales, nunca un thread de fondo real
corriendo contra el reloj de pared. Eso mantiene la suite rápida y determinística (mismo criterio
que el resto de `tests/audio/`: núcleo puro testeado sin las partes de I/O/tiempo real). El único
test que toca `start()`/`stop()` (`test_start_and_stop_are_idempotent_and_return_promptly`)
verifica el lifecycle del thread, no timing — `stop()` interrumpe `Event.wait()` al instante sin
importar el `poll_interval_seconds` configurado, así que ese test también corre rápido.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarvis.audio.timer_scheduler import MAX_PENDING_TIMERS, TimerScheduler
from jarvis.memory.store import list_reminders, save_reminder


class _SpyTTSClient:
    """Stub de `TTSClient`: registra los textos con los que se llamó `.speak()`, sin reproducir
    audio real."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def speak(self, text: str) -> None:
        self.calls.append(text)


class _RaisingTTSClient:
    """Stub de `TTSClient` cuyo `.speak()` siempre lanza — para el caso "anunciar falló, no
    descartar el item pendiente" (ver docstring de `_announce_due_timers`/`_announce_due_reminders`
    en `jarvis.audio.timer_scheduler`)."""

    def speak(self, text: str) -> None:
        raise RuntimeError("TTS falló")


# --- timers (en memoria) -----------------------------------------------------------------------


def test_poll_once_does_not_announce_a_timer_before_it_is_due() -> None:
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=600, now=now)

    scheduler.poll_once(now=now + timedelta(seconds=599))

    assert tts.calls == []
    assert scheduler.pending_timer_count == 1


def test_poll_once_announces_a_timer_exactly_when_due() -> None:
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=600, now=now)

    scheduler.poll_once(now=now + timedelta(seconds=600))

    assert tts.calls == ["¡Se cumplió tu timer!"]
    assert scheduler.pending_timer_count == 0


def test_poll_once_announces_a_timer_with_label() -> None:
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=60, label="la pasta", now=now)

    scheduler.poll_once(now=now + timedelta(minutes=1))

    assert tts.calls == ["¡Se cumplió tu timer de la pasta!"]


def test_poll_once_announces_an_overdue_timer_found_late() -> None:
    """El poll loop corre cada `poll_interval_seconds`, no exactamente al instante en que un
    timer vence — un timer detectado varios segundos después de vencido igual se anuncia
    (`fire_at <= now`, no `==`)."""
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=10, now=now)

    scheduler.poll_once(now=now + timedelta(seconds=30))

    assert tts.calls == ["¡Se cumplió tu timer!"]


def test_a_timer_only_announces_once_across_multiple_polls() -> None:
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=60, now=now)
    due_at = now + timedelta(seconds=60)

    scheduler.poll_once(now=due_at)
    scheduler.poll_once(now=due_at + timedelta(seconds=10))
    scheduler.poll_once(now=due_at + timedelta(seconds=20))

    assert tts.calls == ["¡Se cumplió tu timer!"]


def test_multiple_due_timers_all_announce_in_a_single_poll() -> None:
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=60, label="uno", now=now)
    scheduler.schedule_timer(seconds=120, label="dos", now=now)

    scheduler.poll_once(now=now + timedelta(minutes=5))

    assert set(tts.calls) == {
        "¡Se cumplió tu timer de uno!",
        "¡Se cumplió tu timer de dos!",
    }
    assert scheduler.pending_timer_count == 0


def test_a_timer_that_fails_to_announce_is_retried_on_the_next_poll() -> None:
    """Si `tts.speak()` lanzara (no debería en producción, ver docstring del módulo), el timer no
    se descarta — sigue pendiente para reintentarse en el próximo poll."""
    tts = _RaisingTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=60, now=now)
    due_at = now + timedelta(seconds=60)

    try:
        scheduler.poll_once(now=due_at)
    except RuntimeError:
        pass

    assert scheduler.pending_timer_count == 1


# --- MAX_PENDING_TIMERS (hallazgo MEDIUM de `security-reviewer`) -------------------------------


def test_schedule_timer_returns_an_id_while_under_the_cap() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)

    timer_id = scheduler.schedule_timer(seconds=60, now=now)

    assert timer_id is not None
    assert scheduler.pending_timer_count == 1


def test_schedule_timer_rejects_once_max_pending_timers_is_reached() -> None:
    """Sin tope, `_timers` crecería sin límite en una corrida larga si ningún timer llega a
    vencer — `schedule_timer` devuelve `None` en vez de seguir agregando por encima del tope."""
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(MAX_PENDING_TIMERS):
        assert scheduler.schedule_timer(seconds=60, now=now) is not None

    rejected_id = scheduler.schedule_timer(seconds=60, now=now)

    assert rejected_id is None
    assert scheduler.pending_timer_count == MAX_PENDING_TIMERS


def test_schedule_timer_accepts_again_once_pending_timers_drop_below_the_cap() -> None:
    """El tope es sobre timers *pendientes*, no histórico — una vez que uno vence y se anuncia
    (liberando un lugar), un `schedule_timer` nuevo vuelve a aceptarse."""
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for _ in range(MAX_PENDING_TIMERS):
        scheduler.schedule_timer(seconds=60, now=now)
    assert scheduler.schedule_timer(seconds=60, now=now) is None

    scheduler.poll_once(now=now + timedelta(seconds=60))  # todos vencen y se anuncian
    assert scheduler.pending_timer_count == 0

    timer_id = scheduler.schedule_timer(seconds=60, now=now)

    assert timer_id is not None


# --- recordatorios (persistidos en SQLite) -----------------------------------------------------


def test_poll_once_does_not_announce_a_reminder_before_it_is_due(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts, db_path=db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    save_reminder(
        "llamar a mamá", (now + timedelta(minutes=10)).isoformat(), db_path=db_path
    )

    scheduler.poll_once(now=now)

    assert tts.calls == []
    assert len(list_reminders(db_path=db_path)) == 1


def test_poll_once_announces_a_due_reminder_and_deletes_it(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts, db_path=db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    save_reminder(
        "llamar a mamá", (now - timedelta(minutes=1)).isoformat(), db_path=db_path
    )

    scheduler.poll_once(now=now)

    assert tts.calls == ["Recordatorio: llamar a mamá"]
    assert list_reminders(db_path=db_path) == []


def test_a_reminder_only_announces_once_across_multiple_polls(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts, db_path=db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    save_reminder(
        "llamar a mamá", (now - timedelta(minutes=1)).isoformat(), db_path=db_path
    )

    scheduler.poll_once(now=now)
    scheduler.poll_once(now=now + timedelta(minutes=5))

    assert tts.calls == ["Recordatorio: llamar a mamá"]


def test_multiple_due_reminders_all_announce_in_a_single_poll(tmp_path: Path) -> None:
    db_path = tmp_path / "jarvis.db"
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts, db_path=db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    save_reminder("uno", (now - timedelta(minutes=2)).isoformat(), db_path=db_path)
    save_reminder("dos", (now - timedelta(minutes=1)).isoformat(), db_path=db_path)

    scheduler.poll_once(now=now)

    assert set(tts.calls) == {"Recordatorio: uno", "Recordatorio: dos"}
    assert list_reminders(db_path=db_path) == []


def test_poll_once_announces_both_a_due_timer_and_a_due_reminder(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jarvis.db"
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts, db_path=db_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=60, now=now)
    save_reminder(
        "llamar a mamá", (now - timedelta(minutes=1)).isoformat(), db_path=db_path
    )

    scheduler.poll_once(now=now + timedelta(minutes=1))

    assert set(tts.calls) == {"¡Se cumplió tu timer!", "Recordatorio: llamar a mamá"}


# --- cancel_timer() / list_pending_timers() (cancelación de timers, jarvis.tools.cancel_timer) --


def test_cancel_timer_removes_a_pending_timer_and_returns_true() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    timer_id = scheduler.schedule_timer(seconds=600, label="la pasta", now=now)
    assert timer_id is not None

    cancelled = scheduler.cancel_timer(timer_id)

    assert cancelled is True
    assert scheduler.pending_timer_count == 0


def test_cancel_timer_does_not_announce_the_cancelled_timer() -> None:
    tts = _SpyTTSClient()
    scheduler = TimerScheduler(tts=tts)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    timer_id = scheduler.schedule_timer(seconds=60, now=now)
    assert timer_id is not None

    scheduler.cancel_timer(timer_id)
    scheduler.poll_once(now=now + timedelta(seconds=60))

    assert tts.calls == []


def test_cancel_timer_returns_false_for_an_unknown_id() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scheduler.schedule_timer(seconds=60, now=now)

    cancelled = scheduler.cancel_timer(999)

    assert cancelled is False
    assert scheduler.pending_timer_count == 1


def test_cancel_timer_is_idempotent_on_the_same_id() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    timer_id = scheduler.schedule_timer(seconds=60, now=now)
    assert timer_id is not None
    assert scheduler.cancel_timer(timer_id) is True

    assert scheduler.cancel_timer(timer_id) is False


def test_cancel_timer_only_removes_the_targeted_timer() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first_id = scheduler.schedule_timer(seconds=60, label="uno", now=now)
    second_id = scheduler.schedule_timer(seconds=120, label="dos", now=now)
    assert first_id is not None and second_id is not None

    scheduler.cancel_timer(first_id)

    remaining = scheduler.list_pending_timers()
    assert [timer.id for timer in remaining] == [second_id]


def test_list_pending_timers_returns_a_snapshot_of_pending_timers() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first_id = scheduler.schedule_timer(seconds=60, label="uno", now=now)
    second_id = scheduler.schedule_timer(seconds=120, label="dos", now=now)

    pending = scheduler.list_pending_timers()

    assert {timer.id for timer in pending} == {first_id, second_id}
    assert {timer.label for timer in pending} == {"uno", "dos"}


def test_list_pending_timers_is_empty_when_nothing_is_pending() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())

    assert scheduler.list_pending_timers() == []


# --- lifecycle start()/stop() (idempotencia, no timing) -----------------------------------------


def test_start_and_stop_are_idempotent_and_return_promptly() -> None:
    """`stop()` interrumpe `Event.wait()` al instante — no hace falta esperar
    `poll_interval_seconds` para que el thread termine, sin importar qué tan grande sea."""
    scheduler = TimerScheduler(tts=_SpyTTSClient(), poll_interval_seconds=999.0)

    scheduler.start()
    scheduler.start()  # idempotente, no arranca un segundo thread
    scheduler.stop()
    scheduler.stop()  # idempotente, no lanza al llamarse dos veces


def test_stop_without_start_is_a_no_op() -> None:
    scheduler = TimerScheduler(tts=_SpyTTSClient())

    scheduler.stop()  # no debe lanzar aunque nunca se haya llamado start()
