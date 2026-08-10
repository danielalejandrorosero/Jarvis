# 0008 — Overlay flotante de estado: primera UI de JARVIS, proceso separado, comunicado por archivo JSON

## Contexto

Pedido explícito del usuario: una ventanita flotante, siempre visible en una esquina de la
pantalla, que muestre si Alexa está escuchando/pensando/hablando y lo último que se dijo — "como
el círculo de Siri, pero para Windows". Requisito no negociable, dado que el usuario juega League
of Legends con JARVIS corriendo de fondo todo el tiempo: liviana, y que no le robe el foco a la
ventana activa mientras juega.

Hasta esta fase, JARVIS no tenía ninguna interfaz visual — el pipeline de voz (`jarvis.audio.
pipeline.run()`) corre sin consola (`pythonw.exe`, arranque automático vía Tarea Programada, ver
ADR-0004) y el único rastro de lo que hace es `data/jarvis.log`/`data/jarvis-error.log`. Esta es,
literalmente, la primera capa de presentación del proyecto.

`run()` es un loop `asyncio` bespoke (ADR-0005) con puntos de transición de estado ya claros y
nombrados en el propio código: espera de wake word, grabación/transcripción de un comando,
despacho a LLM/tools (`dispatch_turn`), síntesis de voz (`tts.speak`). Tkinter (stdlib, ya
disponible en el Python 3.12 en uso) necesita su propio loop de eventos bloqueante
(`Tk.mainloop()`) para dibujar y reaccionar a interacción — dos loops de eventos incompatibles
(`asyncio` de `run()` y el de Tk) no pueden convivir de forma segura en el mismo hilo/proceso sin
una integración no trivial que no se justifica para una ventana de solo lectura.

## Decisión

1. **Nuevo paquete `src/jarvis/ui/`** (primera vez que el repo tiene una capa de presentación):
   - `jarvis.ui.status`: contrato de datos puro (`StatusState`, `StatusSnapshot`,
     `write_status`/`read_status`/`is_stale`, y `StatusHeartbeat`) — sin `tkinter`, sin nada de
     `jarvis.audio`/`jarvis.llm`. Es lo único que `jarvis.audio.pipeline` importa de `jarvis.ui`.
   - `jarvis.ui.overlay`: la ventana Tkinter (`Overlay`) más las dos funciones puras que deciden
     qué mostrar (`resolve_display`/`truncate_display_text`), separadas a propósito de la
     construcción de widgets para que sean testeables sin abrir una ventana real. Runnable
     standalone: `python -m jarvis.ui.overlay`.
2. **Proceso separado, siempre** — `jarvis.ui.overlay` nunca se importa desde `jarvis.audio.
   pipeline` ni comparte el mismo intérprete/loop de eventos que `run()`. `scripts/
   start_jarvis.ps1` lanza los dos como `Start-Process` independientes (mismo `pythonw.exe`/
   `PYTHONPATH`, logs separados: `jarvis-overlay.log`/`jarvis-overlay-error.log`) — matar o
   reiniciar uno no requiere tocar el otro, y ninguno depende de que el otro exista para
   funcionar.
3. **Comunicación por un archivo JSON plano** (`data/status.json`, `jarvis.ui.status.
   DEFAULT_STATUS_PATH`): tres campos — `state`, `last_text`, `timestamp` — sin versión de
   esquema. `run()` escribe en cada transición real (wake word detectada → `listening`; comando
   transcripto → `thinking`; respuesta lista → `speaking`; vuelta a esperar wake word → `idle`),
   sin restructurar el loop existente — cada escritura se agregó en un punto de transición que ya
   existía. `write_status()` escribe atómicamente (archivo temporal + `os.replace()`) y nunca
   lanza: un fallo de I/O acá jamás puede tumbar un turno de voz real (misma frontera de
   recuperación que ya usa el resto de `run()`, ver `.claude/rules/python.md`). El overlay
   (`jarvis.ui.overlay`) sondea ese archivo cada ~200ms vía `Tk.after()` (el propio scheduler de
   Tk, no un thread aparte) y deriva de `is_stale()` si `run()` sigue vivo.
4. **`StatusHeartbeat`** (`jarvis.ui.status`): un hilo de fondo, mismo patrón `start()`/`stop()`
   idempotente que los otros tres servicios de fondo de `run()` (`SystemAudioMonitor`,
   `LCUAutoAcceptMonitor`, `TimerScheduler`), que reescribe el último estado conocido cada
   `HEARTBEAT_INTERVAL_SECONDS` (2s) aunque no haya ninguna transición nueva. Se agregó tras
   confirmar en vivo, contra el proceso real, que sin esto el estado `idle` (con diferencia el más
   duradero de los cuatro — `run()` puede pasar minutos bloqueado en `next(detect(...))` esperando
   la wake word) dejaba de escribirse por completo mientras esperaba, y el overlay lo mostraba
   como "desconectado" pasados `DEFAULT_STALE_AFTER_SECONDS` (5s) aunque `run()` seguía
   perfectamente vivo — exactamente el falso negativo que el requisito de degradación (punto 4 de
   abajo) quería evitar en la otra dirección.
5. **Degradación explícita**: `resolve_display()` (función pura, `jarvis.ui.overlay`) muestra un
   estado "sin conexión" distinto, con su propio color/texto, cuando `read_status()` devuelve
   `None` (archivo ausente — `run()` nunca arrancó, o el overlay arrancó primero) o cuando el
   snapshot leído es más viejo que `DEFAULT_STALE_AFTER_SECONDS` — nunca se muestra un estado
   viejo como si fuera en vivo. Del otro lado, si el overlay nunca arranca, crashea, o el archivo
   no se puede escribir, `run()` no lo nota: ninguna llamada a `status_heartbeat.update()` puede
   lanzar ni bloquear un turno.
6. **Ventana**: `overrideredirect(True)` (sin bordes/título) + `-topmost` (siempre visible) +
   `-alpha` (semi-transparente) + `-toolwindow` (oculta de la barra de tareas/Alt-Tab en Windows),
   los cuatro atributos nativos de Tk/Win32, sin dependencia nueva. Posicionada en la esquina
   inferior derecha por default. Nunca llama a `focus_force()`/`grab_set()` — no hay ningún código
   que le robe foco a la ventana activa.

## Por qué

- **`tkinter`, no una librería de GUI nueva**: viene con el Python 3.12 ya instalado, cero
  dependencias nuevas (`.claude/rules/python.md`: "sin dependencias nuevas sin justificar la
  necesidad") — cubre exactamente lo que hace falta (ventana frameless, semi-transparente,
  always-on-top) sin arrastrar un framework de UI completo (Qt/wx/Electron) para una ventana de
  cuatro estados y una línea de texto.
- **Proceso separado, no un thread dentro de `run()`**: mezclar el loop `asyncio` de `run()` con
  el `mainloop()` bloqueante de Tk en el mismo proceso exige o correr Tk en un thread aparte
  (Tkinter no es thread-safe para mutar widgets fuera de su propio loop — fuente clásica de bugs
  intermitentes) o integrar ambos loops de eventos a mano — complejidad real sin necesidad: el
  overlay es puramente de lectura, nunca necesita llamar de vuelta a `run()`. Dos procesos
  independientes, con un contrato de datos mínimo entre ellos, es la solución más simple que
  cumple el requisito.
- **Archivo JSON sondeado, no un socket local ni memoria compartida** — alternativas evaluadas y
  descartadas:
  - *Socket local (TCP/named pipe)*: exige que ambos procesos coordinen quién escucha y quién
    conecta, con lógica de reconexión si cualquiera de los dos se reinicia — justo el escenario
    que el punto 5 (degradación) tiene que cubrir de todos modos. Un archivo no tiene "conexión"
    que mantener: cada lectura es independiente, y un archivo ausente/viejo ya es exactamente la
    señal que el overlay necesita para mostrarse como desconectado, sin protocolo de
    reconexión aparte.
  - *Memoria compartida*: no aporta nada frente a un archivo para este volumen de datos (unos
    pocos bytes, actualizados unas pocas veces por segundo como máximo) y agrega gestión de
    lifecycle (crear/adjuntar/liberar el segmento) sin beneficio de latencia perceptible — un
    archivo sondeado cada ~200ms es indistinguible en UX de cualquier alternativa más rápida para
    una ventana que un humano lee a ojo.
  - Ambas alternativas además atan la disponibilidad de un proceso a que el otro esté vivo en el
    momento exacto de la comunicación; el archivo desacopla eso por completo (ver punto 5).
- **`StatusHeartbeat` como hilo de fondo con el mismo patrón que los servicios ya existentes**: no
  es una abstracción nueva — es el mismo `start()`/`stop()` idempotente que `SystemAudioMonitor`/
  `LCUAutoAcceptMonitor`/`TimerScheduler` ya usan, aplicado a un problema real y confirmado en
  vivo (no anticipado en el diseño original: la primera versión de esta feature no lo tenía, y se
  agregó recién al confirmar el falso "desconectado" contra el proceso real corriendo).

## Consecuencias

- **`jarvis.audio.pipeline` gana una dependencia de import hacia `jarvis.ui.status`** (no hacia
  `jarvis.ui.overlay`, que nunca se importa desde ahí) — acoplamiento mínimo y unidireccional,
  limitado a un módulo de datos puro sin `tkinter`. Si `jarvis.ui.status` alguna vez necesitara
  algo de `tkinter`, habría que revisar esta decisión.
- **Un archivo más en `data/`** (`status.json`, gitignored igual que `jarvis.db`/los logs) — sin
  impacto en el resto del sistema; nadie más lo lee ni escribe.
- **Sin verificación en vivo contra un juego en fullscreen exclusivo real** (a diferencia de
  fullscreen sin bordes/ventana): en teoría, un juego en modo exclusivo podría renderizar por
  encima del overlay pese a `-topmost`, dependiendo de cómo Windows maneje esa combinación
  específica — no es algo que Tkinter controle desde este lado, y no se descarta como riesgo
  abierto para `security-reviewer`/verificación futura del usuario en su sesión real de juego.
- **Click-through no implementado en esta primera versión** (pedido como "nice-to-have", no
  requisito) — la ventana puede recibir clicks/foco si el usuario interactúa con ella
  directamente; no se llama a ningún método que le robe foco activamente, pero tampoco hay
  `WS_EX_TRANSPARENT`/`WS_EX_NOACTIVATE` vía `ctypes` para que sea imposible de clickear. Si en
  uso real esto molesta durante el juego, es la extensión natural a evaluar después.
- **`HEARTBEAT_INTERVAL_SECONDS` (2s) y `DEFAULT_STALE_AFTER_SECONDS` (5s) son valores razonados,
  no medidos exhaustivamente en producción** — el margen (más del doble) absorbe una demora
  puntual del hilo de heartbeat sin falsos "desconectado", pero no se verificó bajo carga real
  sostenida (ej. el proceso con CPU muy ocupada por varios tool-calls encadenados en el mismo
  turno) si ese margen sigue siendo suficiente.
