"""Diagnóstico temporal: imprime el score máximo visto en una ventana, no solo si cruza umbral."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from jarvis.audio.wake_word import (  # noqa: E402
    WAKEWORD_NAMES,
    iter_microphone_frames,
    load_model,
)

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
DEVICE = int(sys.argv[2]) if len(sys.argv) > 2 else None
# Wakeword a diagnosticar: la primaria por default, o la que se pase como 3er argumento
# (útil para calibrar "alexa"/"hey_mycroft" individualmente más adelante).
TARGET_WAKEWORD = sys.argv[3] if len(sys.argv) > 3 else WAKEWORD_NAMES[0]

model = load_model()
frames = iter_microphone_frames(device=DEVICE, duration=DURATION)
max_score = 0.0
scores_over_time = []
print(
    f"Escuchando {DURATION}s (device={DEVICE})... decí '{TARGET_WAKEWORD}'",
    file=sys.stderr,
)
for frame in frames:
    predictions = model.predict(frame)
    score = predictions.get(TARGET_WAKEWORD, 0.0)
    scores_over_time.append(score)
    if score > max_score:
        max_score = score

print(f"Score maximo observado: {max_score:.4f}")
print(f"Frames con score > 0.05: {sum(1 for s in scores_over_time if s > 0.05)} de {len(scores_over_time)}")
print(f"Frames con score > 0.2: {sum(1 for s in scores_over_time if s > 0.2)}")
top5 = sorted(scores_over_time, reverse=True)[:5]
print(f"Top 5 scores: {[round(s, 4) for s in top5]}")
