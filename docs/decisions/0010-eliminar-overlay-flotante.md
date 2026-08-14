# 0010 — Eliminar el overlay flotante por completo

## Contexto

ADR-0008 introdujo la primera UI del proyecto (overlay flotante de solo lectura). ADR-0009 la
extendió con un panel bidireccional (chat tipeado + medidor de mic). Contando iteraciones previas
descartadas dentro de la misma sesión antes de llegar a ADR-0008/0009 (un toast simple, un dial
circular tipo "Iron Man HUD"), y un rediseño posterior explícitamente pedido como "más
tecnológico" (HUD circular arrastrable con glow, commit `96e4980`), el usuario rechazó **cuatro**
versiones visuales distintas en la misma sesión de trabajo. Ninguna llegó a convencer, incluida la
última, pedida con una dirección de diseño concreta. El usuario decidió explícitamente abandonar
la feature entera en vez de seguir iterando: "elimina la ventana, no sabes hacer ventanas".

## Decisión

Eliminar el overlay flotante por completo, no solo cerrar el proceso en ejecución:

- `src/jarvis/ui/` (paquete entero: `overlay.py`, `status.py`, `chat_inbox.py`, `__init__.py`) y
  `tests/ui/` (entero) se borran.
- La integración correspondiente en `src/jarvis/audio/pipeline.py` (`StatusHeartbeat`, el sondeo
  de `chat_inbox.json` en el loop de espera de la wake word, el parámetro `status_heartbeat` de
  `_process_command_text`, `last_status_text` como variable de puro reporte de estado) se retira
  quirúrgicamente, conservando intacta la lógica real de audio/VAD que compartía el mismo loop.
- `src/jarvis/audio/realtime_stt.py` pierde el parámetro `on_chunk_rms` (existía únicamente para
  alimentar el medidor de mic del overlay); el cálculo de `rms` que sí usa el VAD real no se toca.
- `scripts/start_jarvis.ps1` deja de lanzar un segundo proceso (`jarvis.ui.overlay`).
- ADR-0008 y ADR-0009 no se borran ni se reescriben — quedan como registro histórico de una
  decisión tomada y luego revertida (este mismo ADR es la reversión formal).

## Consecuencias

- JARVIS vuelve a ser puramente por voz, sin ninguna superficie visual: `data/status.json`,
  `data/chat_inbox.json` y `data/overlay_position.json` dejan de existir/generarse.
- Se pierde el chat tipeado (ADR-0009) como canal de entrada alternativo a la voz — no hay
  reemplazo previsto; si se pide de nuevo, es una feature nueva a diseñar, no un "revivir" de este
  código.
- Si en el futuro se retoma una UI, no partir de las iteraciones ya descartadas: el patrón de
  rechazo en esta sesión (cuatro direcciones visuales distintas, ninguna aprobada, incluida una
  pedida con dirección concreta) indica que el problema no era de implementación sino de
  expectativas de diseño no acordadas de antemano — la próxima vez conviene una discusión explícita
  de qué se espera ver, con ejemplos o referencias concretas, antes de escribir código.
