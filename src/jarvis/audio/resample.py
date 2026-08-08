"""Resampleo de audio mono int16 entre sample rates.

Necesario porque el endpoint WASAPI del micrófono (ver `device.py`) no acepta pedir 16kHz
directamente — hay que grabar a la tasa nativa del dispositivo y resamplear en software antes
de pasarle el audio a openWakeWord o Whisper, que sí exigen 16kHz.
"""

from __future__ import annotations

import numpy as np


def resample(audio: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamplear audio mono int16 de `orig_sr` a `target_sr` por interpolación lineal.

    Interpolación lineal alcanza para voz — no es una tarea de fidelidad hi-fi — y evita sumar
    una dependencia nueva (scipy) solo para esto.
    """
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = round(duration * target_sr)
    orig_times = np.linspace(0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0, duration, num=target_len, endpoint=False)
    resampled = np.interp(target_times, orig_times, audio.astype(np.float64))
    return resampled.astype(np.int16)


def fit_frame(frame: np.ndarray, frame_samples: int) -> np.ndarray:
    """Ajustar `frame` a exactamente `frame_samples` (padding o truncado).

    El resampleo por duración puede quedar un sample corto o largo por redondeo; openWakeWord
    exige un tamaño de frame exacto.
    """
    if len(frame) == frame_samples:
        return frame
    if len(frame) > frame_samples:
        return frame[:frame_samples]
    return np.pad(frame, (0, frame_samples - len(frame)))
