"""Tests para `TimerTool` (`jarvis.tools.timer`).

`TimerTool` no toca la DB ni la red — solo registra un timer en memoria contra un
`TimerScheduler` inyectado por constructor. Un `TimerScheduler` real (sin `.start()`, sin thread
de fondo corriendo) alcanza acá: lo único que se verifica es que `execute()` llama
`schedule_timer` con los argumentos correctos y confirma con una frase natural — el resto del
comportamiento del scheduler (poll loop, anuncio) se testea en
`tests/audio/test_timer_scheduler.py`.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from jarvis.audio.timer_scheduler import TimerScheduler
from jarvis.tools.base import RiskLevel
from jarvis.tools.timer import MAX_TIMER_SECONDS, TimerTool


def test_timer_tool_declares_safe_risk() -> None:
    """Registrar una notificación futura en memoria no muta nada visible del sistema — ver el
    comentario de `risk` en `jarvis.tools.timer` para el razonamiento completo."""
    assert TimerTool.risk is RiskLevel.SAFE


def test_execute_schedules_timer_and_confirms_in_minutes() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=600))

    scheduler.schedule_timer.assert_called_once_with(seconds=600.0, label=None)
    assert result == "Listo, timer de 10 minutos puesto."


def test_execute_confirms_in_seconds_when_not_a_multiple_of_a_minute() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=90))

    assert result == "Listo, timer de 90 segundos puesto."


def test_execute_includes_label_in_scheduling_and_confirmation() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=300, label="la pasta"))

    scheduler.schedule_timer.assert_called_once_with(seconds=300.0, label="la pasta")
    assert result == "Listo, timer de 5 minutos puesto para la pasta."


def test_execute_strips_and_caps_label_length() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    asyncio.run(tool.execute(seconds=60, label="   la pasta   "))

    scheduler.schedule_timer.assert_called_once_with(seconds=60.0, label="la pasta")


def test_execute_treats_blank_label_as_no_label() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=60, label="   "))

    scheduler.schedule_timer.assert_called_once_with(seconds=60.0, label=None)
    assert result == "Listo, timer de 1 minuto puesto."


# --- casos rechazados: nunca llegan a `schedule_timer` --------------------------------------


def test_execute_rejects_missing_seconds_without_scheduling_anything() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute())

    assert "No se especificó una duración válida" in result
    scheduler.schedule_timer.assert_not_called()


def test_execute_rejects_zero_seconds() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=0))

    assert "No se especificó una duración válida" in result
    scheduler.schedule_timer.assert_not_called()


def test_execute_rejects_negative_seconds() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=-10))

    assert "No se especificó una duración válida" in result
    scheduler.schedule_timer.assert_not_called()


def test_execute_rejects_non_numeric_seconds() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds="diez minutos"))

    assert "No se especificó una duración válida" in result
    scheduler.schedule_timer.assert_not_called()


def test_execute_rejects_boolean_seconds_despite_being_an_int_subclass() -> None:
    """`bool` es subclase de `int` en Python — sin chequeo explícito, `seconds=True` pasaría
    como un timer de 1 segundo."""
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=True))

    assert "No se especificó una duración válida" in result
    scheduler.schedule_timer.assert_not_called()


def test_execute_rejects_duration_beyond_max_timer_seconds() -> None:
    scheduler = MagicMock(spec=TimerScheduler)
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=MAX_TIMER_SECONDS + 1))

    assert "duración máxima" in result
    scheduler.schedule_timer.assert_not_called()


def test_execute_reports_a_clear_message_when_the_scheduler_is_at_capacity() -> None:
    """`schedule_timer` devuelve `None` una vez alcanzado `MAX_PENDING_TIMERS` en
    `TimerScheduler` (hallazgo MEDIUM de `security-reviewer`) — `execute()` tiene que reportarle
    eso al usuario, nunca confirmar un timer que en realidad no quedó registrado."""
    scheduler = MagicMock(spec=TimerScheduler)
    scheduler.schedule_timer.return_value = None
    tool = TimerTool(scheduler=scheduler)

    result = asyncio.run(tool.execute(seconds=600))

    scheduler.schedule_timer.assert_called_once_with(seconds=600.0, label=None)
    assert "No pude poner el timer" in result
    assert "Listo" not in result
