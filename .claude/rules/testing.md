# Testing

Responsabilidad de esta regla: qué significa "verificado" en este repo. Define el criterio; no
redefine convenciones de Python (`.claude/rules/python.md`) ni de seguridad
(`.claude/rules/security.md`).

## Definición de "Done"

Ninguna tarea se considera terminada solo porque el código fue escrito. Done =
**implementado + verificado + revisado + sin regresiones conocidas**.

- Código → tests (`pytest`) + lint/type-check (`ruff`, `mypy`) en verde.
- Configuración (`.claude/settings.json`, hooks) → sintaxis validada, hook probado con un caso que
  debe pasar y uno que debe bloquear.
- Arquitectura → revisión independiente (`architect`).
- Seguridad → revisión independiente (`security-reviewer`).
- Un tool nuevo → al menos un caso SAFE que se ejecuta y un caso CONFIRM/DANGEROUS que se
  rechaza o pide confirmación. Un tool sin el caso rechazado no está probado.

## Estrategia

- Unit tests junto al módulo que cubren (`tests/` espejando `src/`).
- Integration tests para flujos que cruzan capas (tool → security → windows), con la capa Windows
  real mockeada explícitamente, no ausente.
- Ningún test de un tool con efectos en el sistema real corre contra el sistema del usuario sin
  aislamiento (sandbox, directorio temporal, proceso dummy).

## Cuándo delegar en `test-engineer`

Diseño de estrategia de test para un módulo nuevo, casos límite de un tool, o cuando una
regresión no tiene un test que la hubiera atrapado (en ese caso: el test que falta se escribe
antes de cerrar el bug).
