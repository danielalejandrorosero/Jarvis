# 0002 — La seguridad se aplica en dos capas complementarias, no una sola

## Contexto

JARVIS eventualmente controlará Windows. Un LLM puede ser convencido, confundido o simplemente
equivocarse al decidir si un comando es seguro. Si la única barrera es "instruir al modelo para
que no haga cosas peligrosas", esa barrera es tan fuerte como el prompt — es decir, no es una
barrera real.

La versión original de este ADR optaba por un único mecanismo (hooks) para evitar que dos
mecanismos se contradijeran. En el uso real apareció un hueco concreto: `protect-sensitive-files.ps1`
solo intercepta las herramientas `Edit`/`Write` — nada impedía que la herramienta `Read` cargara
el contenido de `.env` al contexto de la sesión. Un hook adicional podía cerrar ese hueco, pero
`Read` de un archivo específico es exactamente el caso para el que existe el mecanismo nativo de
`permissions.deny` de Claude Code — reservarlo solo para eso, sin que compita con los hooks, es
más robusto que replicar la misma lógica en PowerShell.

## Decisión

Dos capas, cada una responsable de una cosa distinta, sin solapamiento:

1. **Hooks (`guard-dangerous-commands.ps1`, `protect-sensitive-files.ps1`)** — única fuente de
   verdad para **patrones de comando shell**: comandos DANGEROUS (`Remove-Item -Recurse -Force`,
   `diskpart`, etc.) y, ahora también, intentos de leer/escribir archivos sensibles *vía shell*
   (`Get-Content .env`, `Set-Content .env`, `type .env`, `echo x > .env`) — porque esto ocurre
   dentro de la herramienta `PowerShell`, fuera del alcance de `permissions.deny`.
2. **`permissions.deny` en `settings.json`** — única fuente de verdad para **primitivas nativas de
   archivo**: `Read`, `Edit` (que también cubre `Write`, ver nota) sobre `.env` y variantes,
   credenciales, secrets, `*.pem`, `*.key`, `*.pfx`, `id_rsa*`, `id_ed25519*`. Este es el mecanismo
   correcto porque el cliente de Claude Code lo evalúa con precedencia absoluta (`deny` siempre
   gana sobre `allow`, sin importar especificidad — `code.claude.com/docs/en/permissions.md`),
   antes de que la herramienta se ejecute.

Nada se duplica entre las dos capas: los hooks no intentan bloquear `Read`/`Edit`/`Write` directos
(eso ya lo hace `settings.json` de forma nativa y más confiable), y `settings.json` no intenta
expresar patrones de comando de shell (para eso el regex de un hook es muchísimo más preciso que
el matching por prefijo de `permissions`).

`.env.example`, `.env.sample` y `.env.template` quedan explícitamente fuera de ambas capas — son
plantillas sin secretos, pensadas para commitear.

## Por qué

- Exit code 2 en un hook `PreToolUse` bloquea la llamada a la herramienta antes de que se
  ejecute, sin depender de que el modelo "decida bien".
- `permissions.deny` no soporta negación (no hay forma de decir "todo `.env.*` excepto
  `.env.example`" en un solo patrón — verificado contra la documentación oficial). Por eso la
  lista de `deny` enumera variantes concretas (`.env`, `.env.local`, `.env.production`,
  `.env.development`) en vez de un wildcard que arrastraría a los templates. Consecuencia
  aceptada: una variante nueva no listada (`.env.staging`, por ejemplo) no queda cubierta por
  `settings.json` — pero sí por el hook, cuyo regex (`\.env(\.|$)`) no tiene ese límite. Esa es
  precisamente la razón de tener dos capas en vez de una.
- Nota de sintaxis verificada: una regla `Write(...)` en `deny` no se aplica de forma confiable;
  la forma documentada de bloquear `Write` es una regla `Edit(...)` sobre la misma ruta (un
  `Read` deny cubre `Read`+`Edit` pero no `Write`). Por eso cada ruta sensible tiene entradas
  `Read(...)` **y** `Edit(...)`, no solo una.

## Alternativas consideradas

- **Solo instrucciones en CLAUDE.md/rules**: rechazado — no es enforcement, es una sugerencia que
  un prompt adversarial o un malentendido puede saltarse.
- **Solo hooks (versión original de este ADR)**: rechazado tras encontrar el hueco de `Read` —
  un hook no puede interceptar la herramienta `Read` nativa de Claude Code, solo `permissions.deny`
  puede.
- **Solo `settings.json` deny list**: rechazado como mecanismo único — no cubre comandos de shell
  arbitrarios que logran el mismo resultado (`Get-Content`, redirecciones), y no tiene negación
  para separar `.env` real de `.env.example`.

## Consecuencias

- Los hooks son fail-open ante errores de parseo del payload (ver comentario en cada script): si
  el hook se rompe, no bloquea todo el trabajo, pero tampoco protege — esto es una compensación
  deliberada entre disponibilidad y seguridad, documentada explícitamente, no un descuido.
  `permissions.deny` no tiene ese modo de falla — es evaluado por el cliente, no por un script.
- La lista de `deny` en `settings.json` requiere mantenimiento manual si aparece una nueva
  variante de archivo sensible que se quiera cubrir de forma nativa (el hook la cubre igual, pero
  sin la garantía de precedencia absoluta de `permissions.deny`).
- Cuando JARVIS exista como sistema propio (fuera de Claude Code), esta misma decisión debe
  reimplementarse en su propia capa security/policy — ni los hooks ni `permissions.deny` de
  Claude Code protegen a JARVIS en producción, solo protegen el entorno de desarrollo.
