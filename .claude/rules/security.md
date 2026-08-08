# Security

Responsabilidad de esta regla: la taxonomía de riesgo de JARVIS y quién tiene autoridad para
decidir. Es la fuente canónica — no la redefinas en otra rule ni en un agente.

## Principio

La IA propone. La capa de seguridad decide. La herramienta ejecuta. Ningún tool de JARVIS debe
ejecutar una acción solo porque el modelo la generó o el usuario la pidió en texto libre — toda
acción pasa por clasificación antes de ejecutarse.

## Niveles

**SAFE** — se ejecuta sin confirmación.
Lectura, consultas, listar procesos/archivos, abrir aplicaciones, abrir URLs, información del
sistema. No muta estado persistente.

**CONFIRM** — requiere confirmación explícita del usuario antes de ejecutar.
Modificar archivos, instalar software, modificar configuración, terminar procesos, operaciones
administrativas. Reversible o de impacto acotado, pero muta estado.

**DANGEROUS** — nunca se ejecuta de forma automática, ni con confirmación de un solo paso.
Eliminación masiva, modificación crítica del registro, comandos irreversibles, operaciones
destructivas, cambios administrativos de alto impacto. Requiere ejecución manual del usuario
fuera del agente, o un flujo de confirmación reforzado explícitamente diseñado para ese caso.

## Dónde vive el enforcement

- **Hoy (entorno de Claude Code)**, dos capas sin solapamiento (ver ADR-0002):
  - Hooks (`guard-dangerous-commands.ps1`, `protect-sensitive-files.ps1`) — patrones de comando
    DANGEROUS y accesos a archivos sensibles *vía shell*, por regex, en `PreToolUse`.
  - `permissions.deny` en `.claude/settings.json` — `Read`/`Edit` nativos sobre `.env` y
    variantes, credenciales y claves, con precedencia absoluta sobre cualquier `allow`.
  `permissions.allow` solo pre-aprueba operaciones SAFE de lectura para reducir fricción.
- **Mañana (JARVIS como sistema)**: cada tool debe declarar su propio nivel de riesgo y la capa
  de policy debe evaluarlo antes de invocar la capa Windows — ver `.claude/rules/architecture.md`.
  Esto es diseño para la próxima fase, no algo a implementar ahora.

## Higiene de secretos

Nunca imprimir valores de `.env` (o de cualquier archivo sensible) en outputs, documentación,
código, mensajes de commit, o en el chat. Para verificar `.env`, comprobar que existe y qué
nombres de variable tiene — nunca sus valores. Antes de cualquier `git add`/commit, correr
`git status` y revisar que nada sensible quedó en staging.

Motivo concreto, no hipotético: durante el hardening de esta fase, un intento de leer `.env` con
la herramienta `Read` no fue bloqueado por `permissions.deny` (ver ADR-0002) y el valor de la API
key terminó impreso en la conversación. Esta regla existe para que ese tipo de descuido no se
repita aunque una capa de enforcement falle — es la última línea, no la única.

## Cuándo delegar en `security-reviewer`

- Cualquier tool nuevo que toque filesystem, procesos, red o registro.
- Cualquier cambio a la clasificación SAFE/CONFIRM/DANGEROUS o a los hooks de guard.
- Cualquier cambio a `.claude/settings.json` (permisos, hooks).

`security-reviewer` revisa, no implementa. Si encuentra un problema, lo reporta al orchestrator;
no lo arregla directamente (preserva la independencia del review).
