# Git

Responsabilidad de esta regla: convenciones de control de versiones. No decide cuándo delegar
(`.claude/rules/agents.md`) ni qué es seguro ejecutar (`.claude/rules/security.md`).

## Reglas duras

- Nunca commitear automáticamente. El usuario decide cuándo.
- Nunca `git push --force`, `git reset --hard`, ni reescribir historia ya publicada sin
  confirmación explícita del usuario en ese momento concreto.
- `git status` antes de cualquier operación que pueda descartar trabajo no commiteado
  (`checkout`, `restore`, `reset`, `clean`).
- Antes de un `git add` amplio, revisar qué quedó staged (`git status`) y el contenido de
  cualquier archivo que pudiera contener secretos, aunque el nombre parezca inocuo.

## Convenciones

- Mensajes de commit: línea de resumen en imperativo, cuerpo opcional explicando el *por qué* si
  no es obvio del diff.
- Un commit por unidad lógica de cambio — no mezclar refactor no relacionado con la tarea.
- Ramas: `main` como rama estable. Ramas de trabajo con nombre descriptivo cuando el cambio sea
  grande o experimental; para cambios pequeños, commitear directo es aceptable.
