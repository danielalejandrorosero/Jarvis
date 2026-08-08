---
paths:
  - "**/*.py"
---

# Python

Responsabilidad de esta regla: convenciones de código Python para JARVIS. Aplica desde el momento
en que exista el primer archivo `.py` — no hay código todavía, esto es la convención a seguir
cuando lo haya.

## Versión y herramientas

- Python 3.12 (ya instalado en el entorno de desarrollo).
- Formato y lint: `ruff` (`ruff format`, `ruff check`). Aplicado automáticamente al guardar por
  `.claude/hooks/python-format-on-save.ps1` — no lo ejecutes manualmente salvo para verificar.
- Testing: `pytest`. Ver `.claude/rules/testing.md` para estrategia.
- Tipado: `mypy` en modo estricto sobre `src/`. Todo símbolo público lleva type hints.

## Convenciones

- Layout `src/`: el paquete instalable vive en `src/<paquete>/`, no en la raíz del repo.
- Async: cualquier código que interactúe con el LLM, red o procesos de larga duración usa
  `asyncio`. No mezclar código bloqueante síncrono dentro de rutas async sin `asyncio.to_thread`.
- Sin dependencias nuevas sin justificar la necesidad — preferir stdlib cuando cubra el caso.
- Errores: excepciones específicas, nunca `except Exception` silencioso salvo en la frontera de
  un hook o de una capa de recuperación explícitamente documentada.
- Nada de `print()` para diagnóstico en código de librería — logging estructurado.

## Cuándo delegar en `python-engineer`

Implementación, refactor con justificación concreta, y cualquier decisión de diseño interno a un
módulo Python (no de límites entre componentes — eso es `architect`).
