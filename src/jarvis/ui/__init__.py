"""Capa de presentación de JARVIS (ADR-0008): la primera UI que tuvo este proyecto.

Hasta esta fase, JARVIS era puramente background — sin ventana, sin feedback visual, solo voz y
`data/jarvis.log`. `jarvis.ui` agrega un overlay flotante de solo lectura (`jarvis.ui.overlay`)
que muestra el estado en vivo del pipeline de voz (`jarvis.audio.pipeline.run()`): esperando wake
word, escuchando, pensando, hablando, o desconectado.

Corre como **proceso separado** de `run()` (nunca importado ni lanzado dentro del mismo proceso —
ver docstring de `jarvis.ui.overlay` y ADR-0008 para el porqué), comunicado por un archivo JSON
simple (`jarvis.ui.status`) que `run()` escribe en cada transición de estado y que el overlay
sondea. Ningún módulo de `jarvis.ui` tiene autoridad sobre el pipeline real: es observabilidad,
no control — si el overlay nunca arranca, crashea, o queda desactualizado, el asistente de voz
sigue funcionando exactamente igual (`.claude/rules/security.md`/`.claude/rules/python.md`: capa
de recuperación explícita, nunca una dependencia dura).
"""

from __future__ import annotations
