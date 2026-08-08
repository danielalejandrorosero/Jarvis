# 0003 — Skills en vez de `.claude/commands/`

## Contexto

Claude Code soporta tanto `.claude/commands/` (slash commands legacy) como Skills. El repo no
tenía ninguno de los dos.

## Decisión

Todos los workflows repetibles se implementan como Skills (`.claude/skills/`). No se crea
`.claude/commands/`.

## Por qué

Skills es el mecanismo recomendado actualmente para workflows reutilizables: soporta
descripciones ricas con criterios de cuándo usar/no usar, y evita mantener dos formatos
paralelos para el mismo tipo de cosa. No hay ningún workflow en este proyecto que necesite algo
que solo `.claude/commands/` ofrezca.

## Consecuencias

Si en el futuro aparece una razón concreta para un command (ej. algo que Skills no puede
expresar), esa razón se documenta aquí como actualización de esta ADR antes de crear
`.claude/commands/` — no se crea "por si acaso".
