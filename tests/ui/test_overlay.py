"""Tests para el núcleo puro de `jarvis.ui.overlay` — `resolve_display`/`truncate_display_text`.

No abre ninguna ventana de Tk real: igual que el resto de este repo no testea hardware de audio
real (`record_command`/`run()`, ver `tests/audio/test_pipeline.py`), una ventana Tkinter real no
es razonablemente testeable de forma aislada/CI-safe — la lógica de "qué mostrar dado este
estado" está separada a propósito en funciones puras (`jarvis.ui.overlay.resolve_display`) para
que sí se pueda cubrir sin abrir ningún widget.
"""

from __future__ import annotations

import pytest

from jarvis.ui.overlay import (
    _EMPTY_TEXT_PLACEHOLDER,
    _STATE_COLORS,
    _STATE_LABELS,
    DISCONNECTED_COLOR,
    DISCONNECTED_LABEL,
    DISCONNECTED_TEXT,
    MAX_DISPLAY_TEXT_LENGTH,
    resolve_display,
    truncate_display_text,
)
from jarvis.ui.status import StatusSnapshot, StatusState


def test_resolve_display_none_snapshot_is_disconnected() -> None:
    label, color, text = resolve_display(None, now=100.0)

    assert label == DISCONNECTED_LABEL
    assert color == DISCONNECTED_COLOR
    assert text == DISCONNECTED_TEXT


def test_resolve_display_stale_snapshot_is_disconnected() -> None:
    snapshot = StatusSnapshot(
        state=StatusState.LISTENING, last_text="abrí discord", timestamp=100.0
    )

    label, color, text = resolve_display(snapshot, now=200.0, stale_after_seconds=5.0)

    assert label == DISCONNECTED_LABEL
    assert color == DISCONNECTED_COLOR
    assert text == DISCONNECTED_TEXT


def test_resolve_display_fresh_snapshot_is_not_disconnected() -> None:
    snapshot = StatusSnapshot(
        state=StatusState.LISTENING, last_text="abrí discord", timestamp=100.0
    )

    label, _color, text = resolve_display(snapshot, now=100.5, stale_after_seconds=5.0)

    assert label != DISCONNECTED_LABEL
    assert text == "abrí discord"


@pytest.mark.parametrize("state", list(StatusState))
def test_resolve_display_maps_every_state_to_label_and_color(
    state: StatusState,
) -> None:
    snapshot = StatusSnapshot(state=state, last_text="algo", timestamp=100.0)

    label, color, _text = resolve_display(snapshot, now=100.0)

    assert label == _STATE_LABELS[state]
    assert color == _STATE_COLORS[state]


def test_resolve_display_empty_text_shows_placeholder() -> None:
    snapshot = StatusSnapshot(state=StatusState.IDLE, last_text="   ", timestamp=100.0)

    _label, _color, text = resolve_display(snapshot, now=100.0)

    assert text == _EMPTY_TEXT_PLACEHOLDER


def test_resolve_display_never_disconnected_state_value_leaks() -> None:
    """`StatusState` no tiene ningún valor "desconectado" — ver docstring de
    `jarvis.ui.status.StatusState`. Esto solo confirma que el enum sigue teniendo exactamente los
    cuatro estados que `resolve_display`/`_STATE_LABELS`/`_STATE_COLORS` esperan mapear; si se
    agrega un quinto estado sin actualizar estos diccionarios, este test lo va a hacer explícito
    (KeyError) en vez de fallar en silencio con una ventana mostrando texto vacío."""
    assert set(StatusState) == set(_STATE_LABELS) == set(_STATE_COLORS)


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
