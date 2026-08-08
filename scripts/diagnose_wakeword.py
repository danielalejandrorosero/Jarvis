"""Diagnóstico temporal: imprime el score máximo visto en una ventana, no solo si cruza umbral."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from jarvis.audio.wake_word import WAKEWORD_NAME, iter_microphone_frames, load_model  # noqa: E402

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
DEVICE = int(sys.argv[2]) if len(sys.argv) > 2 else None

model = load_model()
frames = iter_microphone_frames(device=DEVICE, duration=DURATION)
max_score = 0.0
scores_over_time = []
print(f"Escuchando {DURATION}s (device={DEVICE})... decí 'Hey Jarvis'", file=sys.stderr)
for frame in frames:
    predictions = model.predict(frame)
    score = predictions.get(WAKEWORD_NAME, 0.0)
    scores_over_time.append(score)
    if score > max_score:
        max_score = score

print(f"Score maximo observado: {max_score:.4f}")
print(f"Frames con score > 0.05: {sum(1 for s in scores_over_time if s > 0.05)} de {len(scores_over_time)}")
print(f"Frames con score > 0.2: {sum(1 for s in scores_over_time if s > 0.2)}")
top5 = sorted(scores_over_time, reverse=True)[:5]
print(f"Top 5 scores: {[round(s, 4) for s in top5]}")
