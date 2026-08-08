"""Tests para `jarvis.audio.device.is_combined_headset` — heurística por nombre para decidir si
el gate de audio fuerte del sistema (`jarvis.audio.loopback.SystemAudioMonitor`) tiene sentido o
no (ver docstring de `is_combined_headset` para el bug en vivo que motivó esto: el gate bloqueaba
toda detección de wake word con un headset puesto, porque asumía filtración acústica que no
existe entre el audio de salida y el mic de un mismo headset).

No abre ningún stream real: `sd.query_devices` se reemplaza por un stub controlado por el test.
"""

from __future__ import annotations

import pytest

from jarvis.audio import device


def _patch_query_devices(
    monkeypatch: pytest.MonkeyPatch, names: dict[int, str]
) -> None:
    monkeypatch.setattr(device.sd, "query_devices", lambda idx: {"name": names[idx]})


def test_is_combined_headset_true_when_both_names_mention_headset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_query_devices(
        monkeypatch,
        {
            15: "Auriculares con micrófono (HONOR CHOICE Headphones)",
            12: "Altavoz de PC (Realtek HD Audio 2nd output with HAP)",
        },
    )
    # El nombre de salida no menciona auriculares en este caso puntual (parlante real) —
    # confirma que no alcanza con que el MIC sea un headset, ambos lados tienen que serlo.
    assert device.is_combined_headset(15, 12) is False


def test_is_combined_headset_true_when_mic_and_output_are_the_same_headset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_query_devices(
        monkeypatch,
        {
            15: "Auriculares con micrófono (HONOR CHOICE Headphones)",
            3: "Auriculares (HONOR CHOICE Headphones)",
        },
    )
    assert device.is_combined_headset(15, 3) is True


def test_is_combined_headset_false_for_pc_mic_and_pc_speaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_query_devices(
        monkeypatch,
        {
            24: "Micrófono (Realtek HD Audio Mic input)",
            20: "Altavoz de PC (Realtek HD Audio 2nd output with HAP)",
        },
    )
    assert device.is_combined_headset(24, 20) is False


def test_is_combined_headset_matches_english_headphone_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_query_devices(
        monkeypatch,
        {
            5: "Headphones Microphone (Generic USB)",
            6: "Headphones (Generic USB)",
        },
    )
    assert device.is_combined_headset(5, 6) is True
