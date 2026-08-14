# 0012 — Migrar el STT en streaming de la Realtime API de OpenAI a Speechmatics

## Contexto

ADR-0007 estableció que la captura de comandos generales (`jarvis.audio.pipeline.run()`) migra a
transcripción en streaming vía la Realtime API de OpenAI (`gpt-live-transcribe`), dejando las
confirmaciones habladas CONFIRM en el camino batch (`gpt-4o-transcribe`) por fiabilidad
verificable. Ese ADR no reabre "por qué streaming" — sigue vigente. Este ADR cambia el
**proveedor** dentro de esa misma arquitectura streaming, motivado por un problema distinto: el
usuario señaló el STT en sí (no solo la latencia) como el cuello de botella real de la experiencia
de voz — transcripciones mal entendidas con más frecuencia de la deseable en español
latinoamericano/acento colombiano.

Investigación con fuentes, no un cambio de gusto: un benchmark independiente sobre español
latinoamericano (dominio médico conversacional) midió Word Error Rate (WER) de:

| Proveedor | WER |
|---|---|
| **Speechmatics** | **7.3%** |
| GPT-4o (transcripción) | 9.5% |
| Whisper | 9.7% |
| Deepgram | 9.8% |

Speechmatics midió el mejor resultado con margen real (2.2-2.5 puntos porcentuales por debajo del
resto) — la mejor evidencia encontrada para este idioma/variante específica, aunque el corpus del
benchmark sea de dominio médico (una característica del corpus de evaluación, no algo que JARVIS
deba replicar en su propia configuración — ver más abajo).

Investigación del protocolo real de Speechmatics, mismo estándar de rigor que ya aplicó ADR-0007
con la Realtime API de OpenAI (confirmado contra la documentación oficial y el código fuente del
SDK, no adivinado — ver docstring de `jarvis.audio.realtime_stt` para el detalle completo con
citas):

- WebSocket (`wss://global.rt.speechmatics.com/v2`), autenticación `Authorization: Bearer
  <clave>` directa (sin flujo de JWT temporal — ese es para exponer la clave en un browser, no
  aplica a un proceso nativo server-side).
- A diferencia de `gpt-live-transcribe` (que rechaza turn detection de servidor por completo para
  este modo), Speechmatics **sí tiene** un mecanismo de servidor equivalente
  (`conversation_config.end_of_utterance_silence_trigger`) que hay que desactivar explícitamente
  — su default documentado es `0` ("desactiva la funcionalidad"), así que no enviar
  `conversation_config` en absoluto lo deja apagado sin que este módulo tenga que declarar nada.
- Speechmatics transmite `AddTranscript` progresivamente durante el turno (no un único evento con
  el texto completo al final, como sí hacía `gpt-live-transcribe`) — hay que acumular los tramos
  hasta `EndOfTranscript`.
- Acepta cualquier `sample_rate` entero (a diferencia de OpenAI, fijo a 24kHz) — permite reusar
  16kHz, el mismo dominio que ya usa el VAD local, sin un segundo resample por chunk.

## Decisión

1. **`jarvis.audio.realtime_stt` se reescribe por completo** para hablar el protocolo WebSocket
   documentado de Speechmatics directamente (`websockets`, ya dependencia dura del repo desde
   ADR-0007) en vez del SDK oficial `speechmatics-rt` — evaluado y descartado: su modelo de
   eventos no expone públicamente la tarea de recepción que este módulo necesita para el backstop
   de "conexión caída sin avisar" (ver punto 3). Mismo contrato de firma que antes
   (`stream_transcribe_command(...) -> tuple[str, bool]`), mismo llamador (`pipeline.run()`), para
   minimizar el cambio en `pipeline.py` (import + cómo se arma el cliente, no una reescritura de
   `run()`).
2. **El corte de turno sigue siendo 100% el VAD local** (`jarvis.audio.vad`,
   `jarvis.audio.speech_detector`) — Speechmatics nunca decide esto, aunque a diferencia de OpenAI
   sí tiene un mecanismo de servidor que hay que dejar explícitamente desactivado (ver Contexto).
   `TRAILING_SILENCE_SECONDS` (`jarvis.audio.vad`) baja de 1.2s a 0.7s — recomendación explícita
   de Speechmatics para voice AI conversacional (0.5-0.8s), pedido del usuario tras señalar
   también el tiempo de corte de turno como cuello de botella percibido.
3. **Mismo backstop dual contra "conexión caída sin avisar"** (hallazgo de seguridad HIGH de la
   sesión de ADR-0007): `TRANSCRIPTION_RESULT_TIMEOUT_SECONDS` como backstop duro +
   `add_done_callback` sobre la tarea de recepción para resolución inmediata. Se repite el mismo
   patrón exacto contra el protocolo de Speechmatics.
4. **`VoiceConfirmationChannel` (confirmaciones CONFIRM) NO migra** — se queda en OpenAI,
   `gpt-4o-transcribe`, batch. Mismo motivo que ADR-0007: fiabilidad probada + observabilidad de
   confianza (`logprobs`), que Speechmatics tampoco expone en este modo. Alcance de este ADR:
   solo el camino streaming de comandos generales.
5. **Sin `additional_vocab`** (el equivalente de `keywords`, que sí se usaba con OpenAI) —
   regresión aceptada explícitamente: la documentación de Speechmatics advierte un costo de hasta
   15 segundos de latencia adicional al iniciar la sesión al usarlo, y este módulo abre una sesión
   nueva por cada comando (no una conexión persistente) — pagar ese costo en cada "Alexa, ..."
   anularía el objetivo de baja latencia que motiva streaming en primer lugar.
6. **Sin `domain="medical"`** — el benchmark que motiva este ADR midió WER sobre un corpus médico,
   pero eso es una característica del corpus de evaluación, no algo que JARVIS (asistente de
   propósito general) deba pedirle a la API.
7. Sin dependencia nueva: se descartó `speechmatics-rt` (el SDK oficial) a favor de seguir usando
   `websockets` directo — ver punto 1 y el docstring del módulo para el razonamiento completo.

## Por qué

- La evidencia (benchmark independiente, español latinoamericano, mejor WER con margen real) es
  más fuerte que cualquier ajuste incremental disponible sobre el proveedor anterior — no había
  ninguna perilla de OpenAI pendiente de probar que prometiera una mejora comparable.
- Cambiar el proveedor sin reabrir "por qué streaming" (ADR-0007) mantiene la superficie del
  cambio acotada: mismo contrato para el llamador, misma arquitectura de capas
  (`pipeline.py → realtime_stt.py → VAD local`), solo cambia el transporte/protocolo.
- Hablar el protocolo documentado directamente, en vez de sumar el SDK oficial, preserva una
  garantía de seguridad concreta (el backstop dual contra conexión caída) sin depender de
  internals no públicos de una librería de terceros — la misma clase de decisión que ya tomó
  ADR-0007 al no reinventar `client.realtime.connect()` de OpenAI (ahí el SDK oficial sí exponía
  lo necesario; acá no, así que la decisión correcta es la opuesta, no una preferencia fija por
  "SDK vs. protocolo crudo" en abstracto).

## Consecuencias

- **Confiado a benchmark de terceros, no medido en producción todavía**: el 7.3% WER es de un
  corpus médico conversacional, no del uso real de JARVIS (comandos de voz cortos, ruido de
  fondo de juego/música). Si en uso real la mejora no se siente, o aparecen alucinaciones nuevas
  no vistas con OpenAI, revisar — no hay logging de confianza en ninguno de los dos proveedores
  streaming que permita cuantificarlo automáticamente todavía.
- **`additional_vocab` no se usa** — regresión aceptada (ver Decisión #5): nombres propios/jerga
  específica de JARVIS (apps, juegos, el propio nombre de activación) pierden el hint que sí tenía
  el camino OpenAI (`TRANSCRIPTION_KEYWORDS`). Si en el futuro `pipeline.py` pasara a reusar una
  conexión de larga duración en vez de una por comando, reevaluar — el costo de latencia
  documentado deja de aplicar una vez por turno.
- **Dos caminos STT productivos siguen coexistiendo** (batch/OpenAI para confirmaciones,
  streaming/Speechmatics para comandos generales) — mismo riesgo ya documentado en ADR-0007: un
  cambio futuro al VAD o a la detección de alucinaciones tiene que evaluarse explícitamente en los
  dos lugares.
- `TRAILING_SILENCE_SECONDS` más corto (0.7s, antes 1.2s) es una recomendación de proveedor, no
  verificada en vivo contra el ambiente real de este usuario todavía — si aparecen cortes a mitad
  de frase con más frecuencia, subir de nuevo hacia 1.0-1.2s (ver comentario en `vad.py`).
