"""Tests para `CancelTimerTool`/`CancelAllTimersTool` (`jarvis.tools.cancel_timer`).

Usa un `TimerScheduler` real (sin `.start()`, sin thread de fondo — mismo enfoque que
`tests/tools/test_timer.py`) para poder registrar timers reales con `schedule_timer` y verificar
que `cancel_timer`/`list_pending_timers` reflejan lo esperado, sin necesitar un mock.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from jarvis.audio.timer_scheduler import TimerScheduler
from jarvis.tools.base import RiskLevel
from jarvis.tools.cancel_timer import CancelAllTimersTool, CancelTimerTool

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_cancel_timer_tool_declares_safe_risk() -> None:
    """Cancelar un timer es menos invasivo que ponerlo (ya SAFE) — ver docstring del módulo."""
    assert CancelTimerTool.risk is RiskLevel.SAFE


def test_execute_cancels_by_exact_label_match() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="la pasta", now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="la pasta"))

    assert "cancelé el timer de la pasta" in result
    assert scheduler.pending_timer_count == 0


def test_execute_cancels_by_fuzzy_label_match() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="el lavarropas", now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="lavarropas"))

    assert "cancelé el timer de el lavarropas" in result
    assert scheduler.pending_timer_count == 0


def test_execute_cancels_most_recent_when_target_is_ultimo() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="uno", now=_NOW)
    scheduler.schedule_timer(seconds=300, label="dos", now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="último"))

    assert "cancelé el timer de dos" in result
    remaining = scheduler.list_pending_timers()
    assert [timer.label for timer in remaining] == ["uno"]


def test_execute_cancels_most_recent_without_accent() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="mi ultimo timer"))

    assert "cancelé el timer" in result
    assert scheduler.pending_timer_count == 0


def test_execute_does_not_special_case_todos_anymore() -> None:
    """ADR-0006 (aplicado hoy, ver docstring del módulo): `CancelTimerTool` perdió la rama
    "todos" — ahora "todos" se resuelve como cualquier otro texto libre contra las etiquetas
    pendientes (fuzzy-match), sin ningún efecto especial de vaciar todo lo pendiente. Cancelar
    TODO de una vez ahora es exclusivamente `CancelAllTimersTool` (CONFIRM)."""
    # Labels que no colisionan por substring con "todos" (a diferencia de, ej., "dos") — el
    # objetivo de este test es que "todos" no dispare ningún manejo especial, no ejercitar el
    # fuzzy-match en sí (ya cubierto por `test_execute_cancels_by_fuzzy_label_match`).
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="la pasta", now=_NOW)
    scheduler.schedule_timer(seconds=300, label="el lavarropas", now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="todos"))

    assert "No encontré ningún timer pendiente" in result
    assert scheduler.pending_timer_count == 2


def test_execute_reports_a_clear_message_when_nothing_is_pending() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="la pasta"))

    assert "No hay ningún timer pendiente" in result


def test_execute_reports_a_clear_message_when_the_label_does_not_match_anything() -> (
    None
):
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="la pasta", now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="algo completamente distinto"))

    assert "No encontré ningún timer pendiente" in result
    assert scheduler.pending_timer_count == 1


def test_execute_does_not_match_unlabeled_timers_by_text() -> None:
    """Un timer sin `label` no puede resolverse por nombre — solo por 'último'."""
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="la pasta"))

    assert "No encontré ningún timer pendiente" in result
    assert scheduler.pending_timer_count == 1


def test_execute_rejects_missing_target_without_cancelling_anything() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="la pasta", now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute())

    assert "No se especificó qué timer cancelar" in result
    assert scheduler.pending_timer_count == 1


def test_execute_rejects_blank_target_without_cancelling_anything() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="la pasta", now=_NOW)
    tool = CancelTimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(target="   "))

    assert "No se especificó qué timer cancelar" in result
    assert scheduler.pending_timer_count == 1


def test_cancel_all_timers_tool_declares_confirm_risk() -> None:
    """ADR-0006: borrar TODO lo pendiente de una sola frase, sin oportunidad de decir que no,
    cruza a CONFIRM — ver docstring del módulo."""
    assert CancelAllTimersTool.risk is RiskLevel.CONFIRM


def test_cancel_all_timers_execute_cancels_every_pending_timer() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="uno", now=_NOW)
    scheduler.schedule_timer(seconds=300, label="dos", now=_NOW)
    scheduler.schedule_timer(seconds=120, now=_NOW)
    tool = CancelAllTimersTool(scheduler=scheduler)

    result = asyncio.run(tool.execute())

    assert "cancelé 3 timers" in result
    assert scheduler.pending_timer_count == 0


def test_cancel_all_timers_execute_uses_singular_phrasing_for_one_timer() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, now=_NOW)
    tool = CancelAllTimersTool(scheduler=scheduler)

    result = asyncio.run(tool.execute())

    assert "cancelé 1 timer" in result
    assert scheduler.pending_timer_count == 0


def test_cancel_all_timers_execute_reports_a_clear_message_when_nothing_is_pending() -> (
    None
):
    scheduler = TimerScheduler(tts=_SpyTTS())
    tool = CancelAllTimersTool(scheduler=scheduler)

    result = asyncio.run(tool.execute())

    assert "No hay ningún timer pendiente" in result


def test_cancel_all_timers_describe_names_the_pending_count() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, label="uno", now=_NOW)
    scheduler.schedule_timer(seconds=300, label="dos", now=_NOW)
    tool = CancelAllTimersTool(scheduler=scheduler)

    description = tool.describe({})

    assert "TODOS los 2 timers pendientes" in description


def test_cancel_all_timers_describe_uses_singular_phrasing_for_one_timer() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    scheduler.schedule_timer(seconds=600, now=_NOW)
    tool = CancelAllTimersTool(scheduler=scheduler)

    description = tool.describe({})

    assert "TODOS los 1 timer pendientes" in description


def test_cancel_all_timers_describe_reports_zero_pending() -> None:
    scheduler = TimerScheduler(tts=_SpyTTS())
    tool = CancelAllTimersTool(scheduler=scheduler)

    description = tool.describe({})

    assert "no hay ninguno pendiente ahora mismo" in description


class _SpyTTS:
    """Stub mínimo de `TTSClient` — ningún test de este archivo depende de anuncios reales."""

    def speak(self, text: str) -> None:
        pass
