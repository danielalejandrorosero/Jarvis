---
name: verify
description: Run the full "Done" verification gate from .claude/rules/testing.md over a set of changes — tests, lint/type-check, and independent review appropriate to what changed. Use before considering any non-trivial task finished. Do not use for a single trivial edit (a typo fix, a comment) where the gate is obviously overkill.
allowed-tools: Read, Grep, Glob, PowerShell, Agent
disallowed-tools: Edit, Write
---

# /verify

## Propósito

Operacionalizar la definición de "Done" del proyecto: implementado + verificado + revisado + sin
regresiones conocidas (`.claude/rules/testing.md`). Evita cerrar una tarea solo porque el código
fue escrito.

Esta skill solo verifica — `disallowed-tools` le quita Edit/Write mientras corre. Si encuentra
algo roto, lo reporta; arreglarlo es una tarea aparte, delegada al especialista correspondiente.

## Cuándo usarla

Después de cualquier cambio no trivial: nuevo tool, cambio a la capa de seguridad, cambio a
`.claude/settings.json` o hooks, o cualquier feature con lógica real. No para un typo o un cambio
de una línea sin riesgo.

## Cuándo NO usarla

Cuando el cambio es puramente cosmético o el usuario ya pidió explícitamente saltarse la
verificación para un experimento desechable.

## Procedimiento

1. **Clasifica el cambio**: código Python / configuración / arquitectura / seguridad / tool
   nuevo. Un cambio puede caer en varias categorías.
2. **Código** → correr `pytest`, `ruff check`, y (si aplica) `mypy`. Reportar resultado, no el
   log completo.
3. **Configuración** (`.claude/settings.json`, hooks) → validar sintaxis JSON/PowerShell, y
   probar el hook con un caso que debe pasar y uno que debe bloquear.
4. **Arquitectura** → si el cambio introdujo o movió un límite entre componentes, delega en
   `architect` para revisión independiente.
5. **Seguridad** → si el cambio toca filesystem, procesos, red, registro, permisos, o la propia
   capa de seguridad, delega en `security-reviewer`. Nunca te saltas este paso para un tool
   nuevo.
6. **Tool nuevo** → confirma que existe al menos un test de caso SAFE ejecutado y un caso
   CONFIRM/DANGEROUS rechazado. Si falta, es un gap, no un "pendiente".
7. Reporta un veredicto único: **Done** o **No done**, con la lista concreta de lo que falta si
   es lo segundo.

## Criterio de verificación

El propio resultado de este skill: cada categoría aplicable tiene un check explícito en verde, no
una suposición de que "probablemente está bien".

## Resultado esperado

Un veredicto claro (Done / No done) más la lista de lo que falta, nunca solo "listo".
