# 0004 — Stack funcional de JARVIS (voz, LLM, memoria, arranque)

## Contexto

Cerrado el entorno de desarrollo (fase 0), se definió el producto: un asistente ambiental de voz
para Windows que arranca solo al iniciar sesión, sin ningún click. Estas decisiones vienen de la
conversación de diseño previa, no de esta fase de hardening — se registran acá porque no había
ADR que las capturara todavía.

## Decisión

- **LLM**: DeepSeek V4-Flash, vía SDK `openai` (compatible con su formato) apuntando al
  `base_url` de DeepSeek. El cliente LLM vive detrás de una interfaz swappable (un protocolo/ABC
  que expone `complete(...)`, no una llamada directa al SDK esparcida por el código) — no porque
  se anticipe cambiar de proveedor pronto, sino porque DeepSeek anunció una subida de precios
  significativa sin fecha ni cifra confirmada, y ese es un riesgo real, no hipotético.
- **Loop de agente**: bespoke, diseñado por `agent-engineer`. Sin LangChain ni framework externo.
- **Persistencia**: SQLite.
- **Interfaz**: voz completa, sin CLI/GUI como interfaz principal.
  - Wake word: `openWakeWord`, modelo preentrenado `hey_jarvis` (licencia **CC BY-NC** — uso
    personal está cubierto; si JARVIS se volviera comercial algún día, este modelo específico no
    sirve y hay que revisar la licencia primero).
  - STT: `faster-whisper`, local. El audio no sale de la máquina.
  - TTS: **detrás de interfaz swappable**, no una llamada directa a un proveedor.
    - Primario: `edge-tts` (endpoint no oficial de Microsoft — funcional y gratis, pero frágil:
      puede romperse o bloquearse sin aviso porque no es una API soportada).
    - Fallback local obligatorio: Piper o SAPI. Si `edge-tts` falla, JARVIS degrada a voz local
      en vez de quedar mudo. Esto no es opcional — es parte del contrato de la interfaz TTS.
- **Arranque**: Tarea Programada de Windows con trigger "al iniciar sesión". No Windows Service
  (evita la complejidad de instalación/permisos de admin que no se justifica para un asistente
  personal de una sola máquina).
- **Contrato de confirmación por voz**: toda operación clasificada CONFIRM (`.claude/rules/security.md`)
  exige una confirmación verbal explícita del usuario antes de ejecutarse. Silencio o timeout ⇒
  **denegar por defecto**, nunca ejecutar. Este contrato vive en el policy engine de JARVIS (la
  capa SECURITY/POLICY de `.claude/rules/architecture.md`), no en el prompt del LLM — el LLM no
  tiene autoridad para decidir que un silencio cuenta como aprobación.

## Por qué

- DeepSeek es, hoy, la opción más barata con capacidad real de tool-calling (ver conversación de
  pricing) — pero el riesgo de subida de precio ya está anunciado, de ahí la interfaz swappable.
- `edge-tts` no oficial es la única opción gratuita con calidad neural aceptable; asumir que puede
  romperse y diseñar el fallback desde el día uno es más barato que descubrirlo en producción con
  JARVIS mudo.
- Confirmación verbal con "silencio = denegar" es la única postura consistente con
  `.claude/rules/security.md` — un asistente que interpreta silencio como "sí" convierte cada
  CONFIRM en un SAFE de facto.

## Consecuencias

- El primer código real de `agent-engineer` debe incluir la interfaz swappable de LLM y de TTS
  desde el arranque, no como refactor posterior — es más barato definir el contrato ahora que
  reescribir todas las llamadas directas más adelante.
- El modelo `hey_jarvis` (CC BY-NC) ata a JARVIS a uso no comercial mientras se use ese modelo
  específico — registrado acá para no perder este detalle en tres meses.
- Nada de esto está implementado todavía. Es diseño, no código.
