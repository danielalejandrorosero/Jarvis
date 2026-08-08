# Agents (delegation policy)

Responsabilidad de esta regla: cuándo el orchestrator delega y cómo evita trabajo duplicado. No
repite la responsabilidad de cada agente — eso vive en `.claude/agents/*.md`.

## Cuándo delegar

Delegar aporta valor cuando da: contexto aislado, experiencia especializada, revisión
independiente, análisis paralelo, reducción de complejidad, o una garantía que el orchestrator no
puede dar por sí mismo (p.ej. independencia de revisión).

No delegar tareas triviales, ni para "parecer más agente". Una pregunta de una línea o un cambio
de una sola línea no necesita un subagente.

## Cómo delegar

1. Divide el problema en partes que puedan analizarse independientemente.
2. Lanza en paralelo solo las partes que no dependen entre sí.
3. Mantén el scope de cada agente pequeño y explícito en el prompt (qué debe mirar, qué no).
4. El agente devuelve: resumen, hallazgos, decisiones, archivos afectados, riesgos,
   recomendaciones — no una narración larga.
5. El orchestrator contrasta resultados contradictorios y decide. La decisión final nunca se
   delega.
6. Verifica el resultado antes de aceptarlo (`.claude/rules/testing.md`).

## Evitar trabajo duplicado

Antes de lanzar un agente, comprueba si Claude Code ya resuelve la tarea de forma nativa
(búsqueda de código, edición directa, el skill `/code-review` para revisión de diffs, el skill
`/security-review` para vulnerabilidades OWASP en el diff actual). Los agentes de este repo
existen para lo que esos mecanismos nativos no cubren: roles con memoria de dominio específica de
JARVIS (arquitectura del sistema, modelo de seguridad propio, Windows, Python) que se re-invocan
consistentemente a lo largo de meses de trabajo.
