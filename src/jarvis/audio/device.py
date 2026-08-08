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


# Palabras que Windows usa en los nombres de dispositivos de auriculares/headsets combinados
# (entrada+salida en el mismo hardware físico) — en español y en inglés, porque el nombre exacto
# depende del driver/fabricante, no solo del idioma de Windows.
_HEADSET_NAME_KEYWORDS = ("auricular", "audífono", "headphone", "headset")


def is_combined_headset(mic_device: int, output_device: int) -> bool:
    """`True` si el mic y el device de salida resueltos parecen ser el mismo headset físico
    (entrada y salida combinadas), a partir de que ambos nombres contienen una palabra de
    auriculares/headset.

    Por qué importa: `SystemAudioMonitor` (`jarvis.audio.loopback`) gatea la detección de wake
    word contra "¿el sistema está sonando fuerte ahora mismo?" para evitar que audio de
    juego/música que se *filtra al mic por el aire* (parlantes, un mic de PC abierto) dispare
    falsos positivos o alucinaciones del STT — confirmado en vivo esta noche como el bug real.
    Ese razonamiento no aplica a un headset: el audio de los auriculares suena adentro del oído,
    no hay filtración acústica real hacia el mic del mismo headset (a diferencia de un parlante
    de PC + mic abierto). Con el gate igual activo en ese caso, el umbral calibrado contra
    audio de juego/música (~30-100 RMS) termina bloqueando también la voz real del usuario
    mientras juega con auriculares puestos — confirmado en vivo esta misma noche: cero
    detecciones de wake word mientras sonaba League of Legends por auriculares, con el usuario
    diciendo "Alexa" activamente. Heurística por nombre, no por device ID: Windows expone el
    mismo headset físico como device de entrada y de salida separados, con nombres relacionados
    pero no necesariamente idénticos carácter a carácter (mismo problema de truncado MME que
    `resolve_input_device`/`resolve_output_device` ya documentan) — comparar por palabra clave de
    "auriculares/headset" presente en ambos es más robusto que comparar el nombre completo.
    """
    mic_name = str(sd.query_devices(mic_device)["name"]).lower()
    output_name = str(sd.query_devices(output_device)["name"]).lower()
    mic_is_headset = any(kw in mic_name for kw in _HEADSET_NAME_KEYWORDS)
    output_is_headset = any(kw in output_name for kw in _HEADSET_NAME_KEYWORDS)
    return mic_is_headset and output_is_headset
