"""Selección del dispositivo de entrada de audio y su sample rate nativo.

Confirmado en vivo en esta máquina: el mismo micrófono físico expuesto por MME (lo que
`sounddevice.default.device` devolvía) capturaba señal casi muda — RMS ~0.5 hablando fuerte,
mientras el mismo micrófono por WASAPI daba RMS ~1670. Windows aplica el AGC y las mejoras de
audio del sistema sobre el endpoint WASAPI, no sobre MME. Por eso resolvemos explícitamente el
device WASAPI del micrófono default en vez de confiar en el default crudo de sounddevice.
"""

from __future__ import annotations

import sounddevice as sd


def resolve_input_device(device: int | None) -> int:
    """Resolver el índice de device de entrada a usar.

    Si se pide uno explícito, se respeta tal cual. Si no, se busca la variante WASAPI del
    micrófono default del sistema (ver docstring del módulo) en vez de usar
    `sounddevice.default.device` directo.
    """
    if device is not None:
        return device
    default_index = sd.default.device[0]
    if default_index is None:
        raise RuntimeError(
            "No hay dispositivo de entrada de audio default configurado en Windows."
        )
    default_name = sd.query_devices(default_index)["name"]
    for i, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] == 0:
            continue
        # MME trunca los nombres de device a 31 caracteres (confirmado en vivo: "Microphone
        # Array (AMD Audio Dev" sin cerrar paréntesis) — comparamos por prefijo en ambas
        # direcciones en vez de igualdad exacta, o nunca matchea contra el nombre completo de
        # WASAPI.
        name = info["name"]
        if not (name.startswith(default_name) or default_name.startswith(name)):
            continue
        if sd.query_hostapis(info["hostapi"])["name"] == "Windows WASAPI":
            return i
    return int(default_index)


def input_sample_rate(device: int) -> int:
    """Sample rate nativo de un device de entrada ya resuelto."""
    return int(sd.query_devices(device, kind="input")["default_samplerate"])


def resolve_output_device(device: int | None) -> int:
    """Resolver el índice de device de salida a usar (para loopback WASAPI, ver
    `jarvis.audio.loopback`). Mismo enfoque que `resolve_input_device`: si se pide uno explícito
    se respeta tal cual; si no, se busca la variante WASAPI del device de salida default en vez
    de confiar en el default crudo de sounddevice — loopback WASAPI necesita abrirse contra ese
    endpoint específico, no contra cualquier alias del mismo hardware.
    """
    if device is not None:
        return device
    default_index = sd.default.device[1]
    if default_index is None:
        raise RuntimeError(
            "No hay dispositivo de salida de audio default configurado en Windows."
        )
    default_name = sd.query_devices(default_index)["name"]
    for i, info in enumerate(sd.query_devices()):
        if info["max_output_channels"] == 0:
            continue
        # Mismo truncado de nombres por MME que `resolve_input_device` — ver su comentario.
        name = info["name"]
        if not (name.startswith(default_name) or default_name.startswith(name)):
            continue
        if sd.query_hostapis(info["hostapi"])["name"] == "Windows WASAPI":
            return i
    return int(default_index)


def output_sample_rate(device: int) -> int:
    """Sample rate nativo de un device de salida ya resuelto."""
    return int(sd.query_devices(device, kind="output")["default_samplerate"])
