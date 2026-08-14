# 0009 — Panel bidireccional: chat tipeado y medidor de mic, reabriendo ADR-0008

## Contexto

El usuario rechazó explícitamente la segunda versión visual del overlay (el dial circular tipo
"Iron Man HUD" de ADR-0008 — "ese reloj") y pidió algo distinto: un panel rectangular con reloj,
un indicador de estado, una caja de texto para poder **tipear** comandos (no solo hablarlos), un
medidor de nivel de mic en vivo (esta sesión estuvo, repetidamente, plagada de bugs de calibración
de umbral de voz — ver `MIN_SILENCE_RMS_THRESHOLD`/`NOISE_FLOOR_*` en `jarvis.audio.pipeline` — y
pidió esto explícitamente para poder depurar esa clase de problema mirando el nivel real en vez de
solo los logs), y un botón real para cerrar la ventana (el dial frameless no tenía ninguna forma de
cerrarlo).

ADR-0008 estableció, como parte de su decisión, que el overlay es **de solo lectura**: `run()`
escribe `data/status.json`, el overlay solo lo lee, nunca al revés. El chat tipeado rompe esa
premisa a propósito — es la primera vez que el overlay necesita mandarle algo a `run()`, no solo
mostrar lo que `run()` ya hizo. Esto no es una corrección de ADR-0008 (la razón de ser de ese canal
de solo lectura sigue vigente para `status.json`), es una extensión: un canal nuevo, bidireccional,
que convive con el existente sin reemplazarlo.

## Decisión

1. **Canal de entrada nuevo, `data/chat_inbox.json` (`jarvis.ui.chat_inbox`), con semántica de
   CONSUMO ÚNICO** — deliberadamente distinta de `status.json` (que se sobreescribe/relee sin
   noción de "ya visto"). `run()` sondea este archivo, con throttle, en el mismo loop donde ya
   espera la wake word; al encontrar un mensaje, lo lee Y LO BORRA en el mismo paso
   (`consume_pending_message`), antes de intentar despacharlo — así un comando tipeado nunca se
   redespacha en la siguiente vuelta del loop, se haya podido procesar con éxito o no. Mismo
   patrón de escritura atómica (`tempfile.mkstemp` + `os.replace`) que `status.json`/
   `overlay_position.json`, y mismo contrato "ausencia = nada pendiente, no error" que
   `read_status`.
2. **Invariante duro: un comando tipeado pasa por EXACTAMENTE el mismo camino de autorización que
   uno hablado** — `dispatch_turn`/`PolicyEngine.authorize_and_execute`, sin ninguna versión
   relajada según el canal de entrada (`.claude/rules/security.md`: la clasificación SAFE/CONFIRM/
   DANGEROUS de un `Tool` no depende de cómo llegó el texto). Se extrajo `_process_command_text`
   (`jarvis.audio.pipeline`) como el núcleo compartido de "qué hacer con un comando ya resuelto a
   texto: dormir/despertar o despachar", con dos call sites (voz, tipeado) — no dos
   implementaciones que puedan divergir con el tiempo. Lo único que NO se comparte, porque es
   específico de STT y no tiene sentido para texto que el usuario tipeó él mismo: el filtro de
   alucinaciones de transcripción (`_contains_non_latin_script`) y el guardado de muestra de
   estilo de HABLA (`save_speech_sample`).
3. **CONFIRM sigue confirmándose por VOZ, incluso para un comando tipeado** — no se construyó un
   segundo `ConfirmationChannel` basado en texto. `PolicyEngine`/`VoiceConfirmationChannel` ya son
   agnósticos al canal que originó el comando (solo reciben un `describe()` y preguntan); un
   comando tipeado que dispare una acción CONFIRM (hoy, únicamente `SystemPowerTool` — todo lo
   demás se bajó a SAFE esta misma noche) sigue preguntando y esperando una respuesta HABLADA, no
   una segunda línea de texto. Evita duplicar la lógica de "silencio/timeout ⇒ denegar por
   defecto" (ADR-0004) en un canal nuevo sin necesidad real.
4. **Medidor de mic: `mic_rms`, un campo más en el MISMO `data/status.json`**, no un tercer
   archivo/canal — mismo razonamiento que ADR-0008 ya aplicó para descartar sockets/memoria
   compartida: un campo más en un archivo que ya se sondea/escribe con esta cadencia no es una
   decisión nueva. Se escribe desde el mismo punto de `run()` donde ya se calcula RMS por chunk
   (`chunk_rms`, ver `jarvis.audio.vad`) — durante la espera de la wake word (nuevo, vía el mismo
   callback que revisa `chat_inbox.json`) y durante la grabación de un comando
   (`jarvis.audio.realtime_stt._capture_and_stream`, que ya lo calculaba para su propio VAD) —
   throttleado a ~200ms, no en cada chunk crudo. Deliberadamente NO tejido dentro del mecanismo
   memoizado de `StatusHeartbeat` (que reescribe el ÚLTIMO estado conocido sin cambios cada 2s
   mientras no haya una transición real): si el nivel de mic viajara ahí, un silencio real después
   de un pico se seguiría mostrando como ese pico viejo hasta la próxima transición de `state` —
   el medidor mentiría exactamente en el escenario que existe para diagnosticar.
   `StatusHeartbeat.update_mic_rms()` es un método aparte, con su propia cadencia, que preserva
   `state`/`last_text` en la misma escritura (mismo archivo, nunca dos escrituras parciales que
   puedan pisarse entre sí).
5. **Historial de chat sin campo nuevo**: se reutilizan `state`/`last_text` que ya existían —
   `jarvis.ui.overlay.extract_new_reply` detecta una respuesta nueva de Alexa mirando transiciones
   a `state == SPEAKING` con un `last_text` distinto del último ya mostrado. No distingue si esa
   respuesta vino de un turno hablado o tipeado (la señal es la misma para ambos); lo que el
   usuario tipeó sí se distingue con un eco local inmediato del lado del overlay, antes de mandarlo
   a `chat_inbox.json`.
6. **Rediseño visual**: panel rectangular (360×400, dentro del rango pedido 320-420px de ancho),
   mismo fondo oscuro/acentos cian-azul que las dos versiones anteriores. Se elimina
   `-transparentcolor` (ya no hay esquinas "de sobra" que recortar — es un rectángulo relleno,
   pensado para interactuar con él). Se agrega un botón "×" real (`root.destroy()`) — la primera
   forma de cerrar la ventana sin matar el proceso a mano. Arrastre acotado a la franja superior
   (reloj/identidad/cerrar), no a todo el panel, para no interferir con clicks en el campo de
   texto/historial.

## Por qué

- **Consumo único para `chat_inbox.json`, sobreescribir-y-releer para `status.json`**: son
  contratos con forma distinta a propósito. `status.json` es un ESTADO ("¿qué está pasando ahora?"
  — releerlo cien veces sin cambios es correcto). Un comando tipeado es un EVENTO ("hacé esto UNA
  vez") — si se leyera sin consumir, el mismo comando se redespacharía en cada vuelta del loop
  mientras el archivo siga ahí, lo cual es al menos un bug molesto y, para un tool con efectos
  reales, algo peor.
- **`_process_command_text` compartido, no dos implementaciones paralelas**: la alternativa (una
  ruta de dispatch propia para texto tipeado) es exactamente el tipo de deuda que rompe un
  invariante de seguridad con el tiempo — alguien cambia la ruta de voz (agrega un chequeo, ajusta
  un caso límite) y se olvida de la ruta tipeada, o viceversa. Una sola función con dos call sites
  hace que "mismo camino de autorización" sea cierto por construcción, no por disciplina.
- **Sin un segundo `ConfirmationChannel` de texto**: `PolicyEngine` ya es agnóstico al canal
  (`ConfirmationChannel.ask(prompt) -> bool`, sin ninguna referencia a voz en su `Protocol`) —
  reusar `VoiceConfirmationChannel` sin condición es la opción de menor superficie nueva, y evita
  reintroducir en un canal nuevo una decisión de seguridad (silencio/timeout ⇒ denegar) que ADR-
  0004 ya resolvió una vez.
- **`mic_rms` en el mismo archivo, no un canal nuevo**: mismo argumento que ADR-0008 ya usó contra
  sockets/memoria compartida — el costo de un canal nuevo (lifecycle, otro punto de fallo, otro
  archivo a limpiar) no se justifica para unos pocos bytes más en un archivo que ya se escribe/lee
  con la cadencia que hace falta.
- **`update_mic_rms()` separado del mecanismo memoizado de `StatusHeartbeat`**: es la única forma
  de que el medidor sea honesto. Mezclar ambos mecanismos (mismo objeto, misma cadencia) haría que
  el nivel de mic se "congele" en el último pico hasta la próxima transición de estado real —
  exactamente el tipo de mentira sutil que un medidor de debugging no se puede permitir sin dejar
  de cumplir su propósito.

## Consecuencias

- **El overlay deja de ser puramente de solo lectura** — `jarvis.ui.overlay` ahora importa
  `jarvis.ui.chat_inbox` además de `jarvis.ui.status`, y escribe un archivo. Sigue sin importar
  nada de `jarvis.audio`/`jarvis.llm`/`jarvis.security`, y sigue sin tener autoridad para ejecutar
  nada por su cuenta — solo deja un mensaje pendiente, `run()` (el otro proceso) es quien decide
  si se autoriza.
- **Nueva superficie de ataque hacia el loop de dispatch**: un archivo en `data/` que, si alguien
  (o algo) más además del overlay pudiera escribirlo, terminaría como texto pasado a
  `dispatch_turn` con la misma autoridad que un comando de voz. Mitigado por pasar por la MISMA
  `PolicyEngine`/clasificación de riesgo que ya existía — no es una vía nueva de bypass de
  seguridad, es un canal nuevo hacia la MISMA puerta de autorización. Riesgo abierto explícito para
  `security-reviewer`: revisar permisos de archivo/ubicación de `data/chat_inbox.json` y el punto
  de interrupción del loop de espera de la wake word (`iter_with_periodic_check`).
- **Un archivo más en `data/`** (`chat_inbox.json`, gitignored igual que el resto de `data/`) —
  igual que `status.json`, efímero y sin impacto fuera de este canal.
- **Clic-through eliminado**: a diferencia del dial (que usaba `-transparentcolor` para que solo la
  forma visible interceptara clicks), el panel rectangular intercepta mouse en toda su área
  mientras esté sobre la pantalla — trade-off aceptado a cambio de que el panel sea real y
  usable (tipear, arrastrar, cerrar). Si esto molesta en uso real durante una partida, la
  extensión natural es un modo "colapsado" que vuelva a ser solo lectura/mínimo, no revertir esta
  decisión.
- **`MIC_LEVEL_MIN_RMS`/`MIC_LEVEL_MAX_RMS` (0–6000) son un rango razonado a partir de valores
  logueados esta sesión, no medidos exhaustivamente en producción** — igual que
  `HEARTBEAT_INTERVAL_SECONDS`/`DEFAULT_STALE_AFTER_SECONDS` en ADR-0008, sujeto a ajuste si en
  uso real el rango resulta mal calibrado para el hardware de mic real del usuario.
