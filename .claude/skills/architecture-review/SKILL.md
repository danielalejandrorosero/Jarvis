---
name: architecture-review
description: Get an independent architectural review from the architect agent before committing to a non-trivial design decision (new component boundary, new layer, or a trade-off with no repo precedent), and record it as an ADR if it's significant enough to matter later. Do not use for implementation-level choices inside an already-defined layer.
allowed-tools: Read, Grep, Glob, Agent, Write, Edit
---

# /architecture-review

## Propósito

Forzar una revisión independiente antes de comprometerse con una decisión estructural, en vez de
que el orchestrator decida solo y en caliente. Cierra el ciclo con un registro (ADR) cuando la
decisión tiene trade-offs que alguien futuro necesitará entender.

## Cuándo usarla

Antes de: introducir un nuevo límite entre componentes, añadir una capa, elegir entre dos
enfoques razonables sin precedente en el repo, o cualquier cambio que toque más de una capa de
`USER → ORCHESTRATOR → PLANNER → TOOLS → SECURITY/POLICY → WINDOWS`.

## Cuándo NO usarla

Decisiones dentro de un módulo ya definido (eso lo resuelve el especialista directamente:
`python-engineer` o `windows-engineer`), o cuando ya existe una ADR que cubre exactamente este
caso — en ese caso, síguela en vez de reabrir la discusión.

## Procedimiento

1. Formula la pregunta como una decisión concreta con alternativas, no como "¿qué opinas de X?".
2. Delega en el agente `architect` con el contexto mínimo necesario (qué se está decidiendo, qué
   alternativas hay, qué restricciones existen).
3. Si `architect` señala que la decisión reabre una ADR existente, resuélvelo explícitamente
   (superar la ADR anterior con una nueva, o descartar el cambio) — nunca lo dejes ambiguo.
4. Si `architect` indica que amerita ADR nueva: créala en `docs/decisions/NNNN-titulo.md` con
   contexto, decisión, y consecuencias (trade-offs aceptados).
5. Si no amerita ADR (decisión menor pero aun así no trivial), registra la decisión en el resumen
   de la tarea, no en un documento nuevo.

## Criterio de verificación

La decisión quedó por escrito en un lugar que alguien puede encontrar en 3 meses sin haber
estado en esta conversación: o la ADR, o el commit que la implementa con mensaje explicativo.

## Resultado esperado

Una decisión tomada (no una lista de opciones abiertas) y, si aplica, una ADR nueva o actualizada.
