"""Tests para el núcleo puro de `jarvis.ui.overlay` — traducción de estado, geometría del dial
(marcas/arco animado) y aritmética de arrastre/posicionamiento.

No abre ninguna ventana de Tk real: igual que el resto de este repo no testea hardware de audio
real (`record_command`/`run()`, ver `tests/audio/test_pipeline.py`), una ventana Tkinter real no
es razonablemente testeable de forma aislada/CI-safe — toda la lógica de "qué mostrar/dónde dado
este estado o este click" está separada a propósito en funciones puras para que sí se pueda cubrir
sin abrir ningún widget."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.ui.overlay import (
    _EMPTY_TEXT_PLACEHOLDER,
    _STATE_COLORS,
    _STATE_LABELS,
    DETAIL_TEXT_MAX_LENGTH,
    DISCONNECTED_COLOR,
    DISCONNECTED_LABEL,
    DISCONNECTED_TEXT,
    MAX_DISPLAY_TEXT_LENGTH,
    OverlayPosition,
    clamp_position,
    compute_arc,
    compute_drag_position,
    default_position,
    load_position,
    resolve_display,
    resolve_initial_position,
    save_position,
    state_color,
    state_label,
    tick_marks,
    truncate_display_text,
)
from jarvis.ui.status import StatusSnapshot, StatusState

# ---------------------------------------------------------------------------
# resolve_display / truncate_display_text
# ---------------------------------------------------------------------------


def test_resolve_display_none_snapshot_is_disconnected() -> None:
    state, text = resolve_display(None, now=100.0)

    assert state is None
    assert text == DISCONNECTED_TEXT


def test_resolve_display_stale_snapshot_is_disconnected() -> None:
    snapshot = StatusSnapshot(
        state=StatusState.LISTENING, last_text="abrí discord", timestamp=100.0
    )

    state, text = resolve_display(snapshot, now=200.0, stale_after_seconds=5.0)

    assert state is None
    assert text == DISCONNECTED_TEXT


def test_resolve_display_fresh_snapshot_is_not_disconnected() -> None:
    snapshot = StatusSnapshot(
        state=StatusState.LISTENING, last_text="abrí discord", timestamp=100.0
    )

    state, text = resolve_display(snapshot, now=100.5, stale_after_seconds=5.0)

    assert state is StatusState.LISTENING
    assert text == "abrí discord"


@pytest.mark.parametrize("state", list(StatusState))
def test_resolve_display_returns_the_real_state_for_every_state(
    state: StatusState,
) -> None:
    snapshot = StatusSnapshot(state=state, last_text="algo", timestamp=100.0)

    resolved_state, _text = resolve_display(snapshot, now=100.0)

    assert resolved_state is state


def test_resolve_display_empty_text_shows_placeholder() -> None:
    snapshot = StatusSnapshot(state=StatusState.IDLE, last_text="   ", timestamp=100.0)

    _state, text = resolve_display(snapshot, now=100.0)

    assert text == _EMPTY_TEXT_PLACEHOLDER


def test_truncate_display_text_short_text_unchanged() -> None:
    assert truncate_display_text("hola") == "hola"


def test_truncate_display_text_strips_whitespace() -> None:
    assert truncate_display_text("  hola  ") == "hola"


def test_truncate_display_text_truncates_long_text() -> None:
    long_text = "a" * (MAX_DISPLAY_TEXT_LENGTH + 20)

    truncated = truncate_display_text(long_text)

    assert len(truncated) == MAX_DISPLAY_TEXT_LENGTH
    assert truncated.endswith("…")


def test_truncate_display_text_exact_length_unchanged() -> None:
    exact_text = "a" * MAX_DISPLAY_TEXT_LENGTH

    assert truncate_display_text(exact_text) == exact_text


def test_truncate_display_text_respects_custom_max_length() -> None:
    long_text = "a" * (DETAIL_TEXT_MAX_LENGTH + 10)

    truncated = truncate_display_text(long_text, max_length=DETAIL_TEXT_MAX_LENGTH)

    assert len(truncated) == DETAIL_TEXT_MAX_LENGTH


# ---------------------------------------------------------------------------
# state_label / state_color
# ---------------------------------------------------------------------------


def test_state_label_and_color_disconnected() -> None:
    assert state_label(None) == DISCONNECTED_LABEL
    assert state_color(None) == DISCONNECTED_COLOR


@pytest.mark.parametrize("state", list(StatusState))
def test_state_label_and_color_map_every_real_state(state: StatusState) -> None:
    assert state_label(state) == _STATE_LABELS[state]
    assert state_color(state) == _STATE_COLORS[state]


def test_every_status_state_has_a_label_and_a_color() -> None:
    """Si se agrega un quinto `StatusState` sin actualizar `_STATE_LABELS`/`_STATE_COLORS`, esto
    lo hace explícito (`KeyError` en el test, no un HUD mostrando texto/color vacío en vivo)."""
    assert set(StatusState) == set(_STATE_LABELS) == set(_STATE_COLORS)


# ---------------------------------------------------------------------------
# compute_arc
# ---------------------------------------------------------------------------


def test_compute_arc_disconnected_is_static_regardless_of_tick() -> None:
    first = compute_arc(None, tick=0)
    later = compute_arc(None, tick=500)

    assert first == later


def test_compute_arc_thinking_start_angle_rotates_with_tick() -> None:
    start_a, extent_a = compute_arc(StatusState.THINKING, tick=0)
    start_b, extent_b = compute_arc(StatusState.THINKING, tick=10)

    assert start_a != start_b
    # El extent (ancho del barrido) es fijo para THINKING — solo rota, no pulsa.
    assert extent_a == extent_b


@pytest.mark.parametrize(
    "state", [StatusState.IDLE, StatusState.LISTENING, StatusState.SPEAKING]
)
def test_compute_arc_pulsing_states_keep_start_angle_fixed(state: StatusState) -> None:
    start_a, _extent_a = compute_arc(state, tick=0)
    start_b, _extent_b = compute_arc(state, tick=7)

    assert start_a == start_b == 90


def test_compute_arc_speaking_has_the_widest_average_extent() -> None:
    """HABLANDO debe ser visualmente el arco más grande en promedio — la señal más notoria
    durante una partida, distinguible sin leer texto."""
    ticks = range(60)
    speaking_avg = sum(compute_arc(StatusState.SPEAKING, t)[1] for t in ticks) / len(
        ticks
    )
    idle_avg = sum(compute_arc(StatusState.IDLE, t)[1] for t in ticks) / len(ticks)

    assert speaking_avg > idle_avg


def test_compute_arc_extent_never_collapses_to_zero_or_negative() -> None:
    for state in StatusState:
        for tick in range(40):
            _start, extent = compute_arc(state, tick)
            assert extent > 0


# ---------------------------------------------------------------------------
# tick_marks
# ---------------------------------------------------------------------------


def test_tick_marks_returns_requested_count() -> None:
    marks = tick_marks(
        center=100.0,
        outer_radius=90.0,
        minor_inner_radius=80.0,
        major_inner_radius=70.0,
        count=12,
        major_every=4,
    )

    assert len(marks) == 12


def test_tick_marks_flags_major_ticks_every_n() -> None:
    marks = tick_marks(
        center=100.0,
        outer_radius=90.0,
        minor_inner_radius=80.0,
        major_inner_radius=70.0,
        count=8,
        major_every=4,
    )

    major_flags = [is_major for *_rest, is_major in marks]
    assert major_flags == [True, False, False, False, True, False, False, False]


def test_tick_marks_first_mark_points_along_positive_x_axis() -> None:
    """Ángulo 0 (índice 0) apunta hacia +x desde el centro — geometría estándar
    (`cos(0)=1, sin(0)=0`), así que el primer segmento va directo a la derecha del centro."""
    marks = tick_marks(
        center=100.0,
        outer_radius=90.0,
        minor_inner_radius=80.0,
        major_inner_radius=70.0,
        count=4,
        major_every=4,
    )

    x1, y1, x2, y2, is_major = marks[0]
    assert x1 == pytest.approx(190.0)
    assert y1 == pytest.approx(100.0)
    assert x2 == pytest.approx(170.0)  # marca mayor -> usa major_inner_radius
    assert y2 == pytest.approx(100.0)
    assert is_major is True


# ---------------------------------------------------------------------------
# posicionamiento: default / clamp / resolución inicial / arrastre
# ---------------------------------------------------------------------------


def test_default_position_is_bottom_right_with_margin() -> None:
    x, y = default_position(
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
        margin=16,
    )

    assert x == 1920 - 220 - 16
    assert y == 1080 - 220 - 16


def test_clamp_position_leaves_in_bounds_position_unchanged() -> None:
    x, y = clamp_position(
        500,
        400,
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
    )

    assert (x, y) == (500, 400)


def test_clamp_position_pulls_back_a_position_off_the_left_edge() -> None:
    x, y = clamp_position(
        -500,
        400,
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
    )

    # No queda en -500: se recorta para que al menos `min_visible_px` quede en pantalla.
    assert x == -220 + 24
    assert y == 400


def test_clamp_position_pulls_back_a_position_off_the_bottom_right() -> None:
    x, y = clamp_position(
        5000,
        5000,
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
    )

    assert x == 1920 - 24
    assert y == 1080 - 24


def test_resolve_initial_position_no_saved_position_uses_default() -> None:
    x, y = resolve_initial_position(
        None,
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
        margin=16,
    )

    assert (x, y) == default_position(
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
        margin=16,
    )


def test_resolve_initial_position_saved_in_bounds_is_used_as_is() -> None:
    saved = OverlayPosition(x=300, y=250)

    x, y = resolve_initial_position(
        saved,
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
        margin=16,
    )

    assert (x, y) == (300, 250)


def test_resolve_initial_position_saved_out_of_bounds_is_clamped() -> None:
    saved = OverlayPosition(x=-9000, y=-9000)

    x, y = resolve_initial_position(
        saved,
        window_width=220,
        window_height=220,
        screen_width=1920,
        screen_height=1080,
        margin=16,
    )

    assert x == -220 + 24
    assert y == -220 + 24


def test_compute_drag_position_basic_offset() -> None:
    x, y = compute_drag_position(500, 400, 20, 30)

    assert (x, y) == (480, 370)


def test_compute_drag_position_zero_offset_matches_pointer() -> None:
    assert compute_drag_position(100, 200, 0, 0) == (100, 200)


# ---------------------------------------------------------------------------
# load_position / save_position
# ---------------------------------------------------------------------------


def test_save_then_load_position_roundtrip(tmp_path: Path) -> None:
    position_path = tmp_path / "overlay_position.json"

    save_position(123, 456, path=position_path)
    loaded = load_position(position_path)

    assert loaded == OverlayPosition(x=123, y=456)


def test_save_position_creates_parent_directory(tmp_path: Path) -> None:
    position_path = tmp_path / "nested" / "dir" / "overlay_position.json"

    save_position(1, 2, path=position_path)

    assert position_path.exists()


def test_load_position_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_position(tmp_path / "no-existe.json") is None


def test_load_position_invalid_json_returns_none(tmp_path: Path) -> None:
    position_path = tmp_path / "overlay_position.json"
    position_path.write_text("esto no es json{", encoding="utf-8")

    assert load_position(position_path) is None


def test_load_position_missing_field_returns_none(tmp_path: Path) -> None:
    position_path = tmp_path / "overlay_position.json"
    position_path.write_text(json.dumps({"x": 1}), encoding="utf-8")

    assert load_position(position_path) is None


def test_save_position_never_raises_when_parent_cannot_be_created(
    tmp_path: Path,
) -> None:
    """Frontera de recuperación (mismo contrato que `jarvis.ui.status.write_status`): un fallo
    real de I/O nunca debe propagar — perder una posición arrastrada no es motivo para crashear el
    proceso del overlay. Se fuerza un fallo real de `mkdir` (no un mock) haciendo que un segmento
    del path ya exista como archivo, no como directorio."""
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("soy un archivo, no un directorio", encoding="utf-8")
    position_path = blocking_file / "overlay_position.json"

    save_position(1, 2, path=position_path)  # no debería lanzar

    assert load_position(position_path) is None


def test_overlay_position_is_frozen() -> None:
    position = OverlayPosition(x=1, y=2)
    with pytest.raises(AttributeError):
        position.x = 5  # type: ignore[misc]
