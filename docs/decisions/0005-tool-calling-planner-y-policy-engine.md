# 0005 — Tool-calling, planner y policy engine en runtime (fase 5)

## Contexto

Fases 0-4 dejaron el loop de voz completo (`src/jarvis/audio/pipeline.py`) pero el LLM solo
conversa — sin tools, sin ejecución sobre el sistema. ADR-0002 ya dejó registrado que la
seguridad de Claude Code (hooks, `permissions.deny`) protege el entorno de desarrollo, no el
runtime de JARVIS, y que esa misma decisión debía reimplementarse en la propia capa
security/policy de JARVIS "cuando JARVIS exista como sistema propio". Fase 5 es ese momento:
el primer tool real (consulta de clima) obliga a construir el planner y el policy engine que
`.claude/rules/architecture.md` exige, no solo a agregar una función.

## Decisión

- **Planner**: function-calling nativo de DeepSeek (formato compatible OpenAI) + un loop de
  dispatch bespoke (sin framework, consistente con ADR-0004) que orquesta el ciclo: LLM pide
  tool_call → `PolicyEngine` autoriza o pide confirmación → tool ejecuta o se deniega →
  resultado vuelve al LLM como mensaje `role: tool` → el LLM produce la respuesta final que se
  habla por TTS. El "bespoke" de ADR-0004 es este loop de orquestación, no una reimplementación
  de parsing de intención — eso ya lo resuelve el `tool_calls` estructurado del provider.
- **`LLMClient` (interfaz existente, ADR-0004) se extiende**: de `complete(prompt, system=...)`
  a un método que acepta una lista de mensajes y un schema de tools opcional, y devuelve texto
  final o una solicitud de tool-call estructurada. Cambio de contrato sobre la única
  implementación actual (`DeepSeekClient`).
- **`Tool`**: clase base abstracta (deliberadamente `ABC`, no `Protocol` como `LLMClient`/
  `TTSClient` — acá el patrón es un registro heterogéneo de N tools que comparten generación de
  schema, no un backend único intercambiable). Contrato: `name`, `description`, `parameters`
  (JSON schema para el LLM), `risk: RiskLevel` (SAFE/CONFIRM/DANGEROUS, atributo de clase
  obligatorio — sin default a SAFE), `async def execute(...)`.
- **`PolicyEngine`** (`src/jarvis/security/policy.py`): único punto de paso entre dispatch y
  `Tool.execute`. La clasificación vive en el `Tool`; el comportamiento por nivel vive
  centralizado acá — SAFE ejecuta, CONFIRM pide confirmación verbal vía `ConfirmationChannel`
  (Protocol implementado por `pipeline.py` reusando `tts.speak`/`record_command`/`transcribe`,
  sin que la policy dependa de audio directamente), DANGEROUS nunca se ejecuta desde el loop
  automatizado, con o sin confirmación. Contrato de ADR-0004 ("silencio o timeout ⇒ denegar")
  se implementa acá: `ConfirmationChannel.ask()` devuelve `False` en timeout o transcripción
  vacía, nunca `True` por defecto.
- **Primer tool: clima**, vía una API acotada sin key (open-meteo). "Qué hora es" no es un
  tool — se resuelve localmente con `zoneinfo`, sin red. Búsqueda web general queda
  explícitamente fuera de esta fase.

## Por qué

- Function-calling nativo evita reimplementar lo que el provider ya resuelve mejor y sin costo
  adicional — un planner bespoke que reparse intención en texto libre sería una abstracción
  prematura y peor que el JSON-schema constrainer del proveedor.
- Clasificación estática en el `Tool` en vez de una tabla central evita que un tool nuevo quede
  sin clasificar por omisión — la lección concreta de ADR-0002 (`Read` sin cubrir sobre `.env`)
  aplicada al propio runtime de JARVIS.
- `PolicyEngine` como único gate hace cumplible por construcción la regla dura de
  `architecture.md` ("el planner nunca llama a Windows directamente; siempre pasa por
  tools → security") — no depende de que el dispatch loop "se acuerde" de chequear.
- Clima en vez de búsqueda web general mantiene el primer tool en el riesgo mínimo real
  (SAFE, sin contenido no confiable devuelto al contexto del LLM) y pospone explícitamente una
  pregunta de seguridad distinta (inyección de contenido web) a su propia decisión.

## Consecuencias

- `DeepSeekClient` y cualquier test double existente de `LLMClient` deben actualizarse al nuevo
  contrato de mensajes/tools — es una ruptura de interfaz, no aditiva.
- `pipeline.py` deja de ser un loop lineal wake→record→transcribe→llm→speak: el sub-loop de
  confirmación CONFIRM agrega un segundo turno de escucha antes de ejecutar o denegar.
- Ningún tool DANGEROUS puede añadirse asumiendo que "ya hay confirmación" — `PolicyEngine` no
  tiene código que lo permita, por diseño; ejecución manual fuera del agente sigue siendo la
  única vía, como especifica `.claude/rules/security.md`.
- Búsqueda web general queda pendiente de su propio ADR, que deberá resolver el riesgo de
  contenido inyectado al contexto del LLM antes de implementarse.
