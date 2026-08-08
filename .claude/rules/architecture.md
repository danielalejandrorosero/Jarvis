# Architecture

Responsabilidad de esta regla: los límites entre capas de JARVIS. No repite convenciones de
Python, Windows o seguridad — eso vive en sus propias rules.

## Capas (nunca se mezclan)

```
USER → ORCHESTRATOR → PLANNER → TOOLS → SECURITY/POLICY → WINDOWS
```

- **Orchestrator**: entiende la petición, decide si delega, integra resultados. No ejecuta nada
  directamente sobre Windows.
- **Planner**: descompone una intención en pasos de tool-calling concretos. No decide permisos.
- **Tools**: implementan una capacidad concreta (leer archivo, listar procesos, abrir app). No
  deciden si están autorizadas a correr — eso lo hace la capa de seguridad, siempre antes de
  ejecutar.
- **Security/Policy**: clasifica cada acción como SAFE / CONFIRM / DANGEROUS
  (`.claude/rules/security.md`) y decide si se ejecuta, se pide confirmación o se rechaza. Esta
  capa es la única con autoridad para bloquear.
- **Windows**: la ejecución real (proceso, filesystem, registro, API de Windows).

Regla dura: ninguna capa superior puede saltarse una capa inferior obligatoria. El planner nunca
llama a Windows directamente; siempre pasa por tools → security.

## Cuándo consultar al agente `architect`

- Antes de introducir un nuevo límite entre componentes o una nueva capa.
- Cuando una decisión tiene trade-offs no triviales (rendimiento vs seguridad, simplicidad vs
  extensibilidad).
- Cuando dos formas razonables de resolver algo compiten y no hay un precedente en el repo.

No lo consultes para decisiones de implementación dentro de una capa ya definida — eso es trabajo
del especialista correspondiente (`python-engineer`, `windows-engineer`).

## Decisiones ya tomadas (no las reabras sin una ADR nueva)

Ver `docs/decisions/`. En particular: la seguridad se aplica en hooks deterministas, no en
instrucciones al LLM (ADR-0002).
