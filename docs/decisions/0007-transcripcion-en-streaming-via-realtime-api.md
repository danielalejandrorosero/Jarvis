# 0007 — Captura de comandos migra a transcripción en streaming (Realtime API); confirmaciones se quedan en batch

## Contexto

El pedido explícito del usuario, tras identificar "el reconocimiento (STT) tarda o falla en
entender" como el techo más limitante de JARVIS: `gpt-4o-transcribe` (el modelo en uso hasta esta
sesión, vía `client.audio.transcriptions.create()`) es un modelo batch — JARVIS graba el comando
completo (esperando silencio sostenido, VAD local) y recién ahí manda el clip entero por HTTP,
esperando el round-trip completo antes de tener texto. Toda la latencia de red se paga después de
que el usuario ya terminó de hablar.

OpenAI lanzó el 28-29 de julio de 2026 `gpt-live-transcribe`, un modelo pensado específicamente
para esto: corre sobre la Realtime API (WebSocket, no REST), devuelve texto parcial (`delta`)
mientras el audio todavía está llegando, y un texto final (`completed`) al cerrar el turno.
Confirmado contra la documentación real de OpenAI (no un resumen de tercero) y contra el SDK
instalado (`openai==2.53.0`, que ya expone `client.realtime.connect(...)` de forma nativa — sin
necesidad de hablar WebSocket a mano) antes de escribir código.

## Decisión

1. **La captura de comandos generales (`jarvis.audio.pipeline.run()`) migra a streaming**, vía el
   nuevo módulo `jarvis.audio.realtime_stt` (`stream_transcribe_command`). El corte de grabación
   (cuándo el usuario terminó de hablar) lo sigue decidiendo exclusivamente el VAD local ya
   calibrado en vivo esta sesión (`measure_noise_floor`/`calibrate_thresholds`, más
   `chunk_rms`/`is_speech_chunk`/`should_stop_recording`, extraídas a `jarvis.audio.vad` para
   romper el ciclo de import) — `turn_detection` se desactiva explícitamente del lado del
   servidor (`turn_detection: null`) para que nunca compitan dos VAD independientes por la misma
   decisión.
2. **`VoiceConfirmationChannel` (el sí/no hablado que gatea CONFIRM/DANGEROUS, ADR-0004/0005) se
   queda en el camino batch** (`jarvis.audio.stt.transcribe`, `gpt-4o-transcribe`), deliberadamente
   sin migrar. `gpt-live-transcribe` no expone `logprobs`/score de confianza; el camino batch sí
   (`temperature=0.0`, logging de confianza agregado esta misma sesión). En el único punto del
   sistema donde una transcripción mal interpretada tiene consecuencia de seguridad real, la
   fiabilidad probada pesa más que la latencia.
3. **`reduce_background_noise`/`normalize_gain` (resta espectral + AGC, agregadas esta misma
   sesión) no aplican al camino streaming** — ambas operan sobre el clip completo ya grabado
   (perfil espectral/pico de toda la muestra); aplicarlas por chunk exigiría acumular todo el
   audio antes de mandar nada (perdiendo la ventaja de latencia) o producir artefactos de borde
   por ventana aislada. Siguen protegiendo el camino batch (`record_command`,
   `VoiceConfirmationChannel`) sin cambios.
4. Nueva dependencia dura: `websockets>=13,<16` — el propio SDK de `openai` la importa de forma
   perezosa recién al conectar (`client.realtime.connect()`) y falla sin ella; no es opcional en
   la práctica para este módulo.

## Por qué

- Partir el módulo VAD (`jarvis.audio.vad`) en vez de duplicar su lógica es la solución correcta
  al ciclo de import (`pipeline.py` necesita invocar `realtime_stt.py`; `realtime_stt.py` necesita
  la misma lógica de corte que ya vivía en `pipeline.py`) — funciones puras, sin I/O, sin cambio
  de comportamiento ni de contrato público (`pipeline.py` re-exporta los mismos símbolos).
- Dos VAD compitiendo (local + servidor) es una superficie de bugs nueva e innecesaria — con
  `turn_detection: null`, la responsabilidad de "¿el usuario terminó de hablar?" tiene un solo
  dueño, igual que en el camino batch.
- Separar el camino de confirmaciones del camino de captura general no es indecisión: es
  reconocer que ambos tienen requisitos distintos (latencia vs. fiabilidad verificable) y que
  `Tool.risk`/`PolicyEngine` (ADR-0005) ya trazan exactamente esa misma línea — el punto con
  consecuencia de seguridad real no debería heredar automáticamente un cambio pensado para el
  caso general.

## Consecuencias

- **Dos caminos STT productivos coexisten** (`jarvis.audio.stt.transcribe` batch,
  `jarvis.audio.realtime_stt.stream_transcribe_command` streaming) — un cambio futuro al VAD, al
  manejo de `pre_roll`, o a la detección de alucinaciones (`_contains_non_latin_script`, hoy
  aplicado solo en `run()`, no en `VoiceConfirmationChannel.ask()`) tiene que evaluarse
  explícitamente en los dos lugares, no asumirse compartido. Mitigado en el caso de
  `_contains_non_latin_script`/confirmaciones por el propio contrato estricto de
  `_is_affirmative` (deniega por defecto ante cualquier texto no reconocido), pero por una razón
  distinta a la que protege `run()` — vale la pena revisarlo si `VoiceConfirmationChannel` alguna
  vez migra.
- **Costo de latencia no medido, pendiente de verificar en uso real**: `stream_transcribe_command`
  abre una sesión WebSocket nueva (`client.realtime.connect()`) en cada turno, incluida cada
  iteración de la ventana de seguimiento sin wake word (`FOLLOW_UP_WINDOW_SECONDS`) — el handshake
  por turno no es gratis y puede estar compitiendo con la ventaja de latencia que motiva esta
  migración entera. A diferencia de casi todo el resto de esta sesión (decisiones respaldadas por
  evidencia medida en vivo), esto todavía no lo está. Si en uso real el handshake por turno anula
  la ganancia, revisitar hacia una sesión WS de larga duración reusada entre turnos en vez de una
  por turno — no descartado de antemano, solo no implementado todavía (complejidad adicional de
  lifecycle no justificada sin evidencia de que hace falta).
- Sin score de confianza en el camino streaming — pérdida aceptada: ese logging nunca estuvo
  conectado a ningún filtro activo, es observabilidad perdida, no una protección real perdida.
- `STREAM_CHUNK_SECONDS` (`realtime_stt.py`) duplica el valor de `CHUNK_SECONDS` (`pipeline.py`,
  0.2s) como constante standalone para evitar el mismo ciclo de import — dos fuentes de verdad
  para el mismo número. Deuda de bajo costo a resolver moviendo ambas constantes a `vad.py` la
  próxima vez que se toque cualquiera de los dos módulos.
