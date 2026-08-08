# JARVIS

## Identidad y objetivo

JARVIS es un sistema de herramientas controlado por un LLM para automatizar tareas en Windows
(archivos, procesos, aplicaciones, información del sistema). **No es un chatbot con acceso a
shell**: es un pipeline `usuario → orchestrator → planner → tools → security/policy → Windows`
donde cada capa tiene una única responsabilidad y la capa de seguridad decide, no el modelo.

Estado actual: fase 0, entorno de desarrollo. Sin código de JARVIS todavía.

## Stack

- **Python 3.12** — núcleo de JARVIS (orquestación, tools, seguridad). Ver `.claude/rules/python.md`.
- **PowerShell / Windows APIs** — capa de ejecución del lado del sistema operativo. Ver `.claude/rules/windows.md`.
- **Git** — control de versiones, sin commits automáticos.

## Arquitectura de alto nivel

```
USER → ORCHESTRATOR → PLANNER → TOOLS → SECURITY/POLICY → WINDOWS
```

Nunca se mezclan razonamiento, permisos, ejecución, interfaz y memoria en el mismo componente.
Detalle completo en `.claude/rules/architecture.md`.

## Estructura del repositorio

```
jarvis/
├── CLAUDE.md              este archivo
├── .gitignore
├── .claude/
│   ├── settings.json      permisos, hooks
│   ├── rules/              conocimiento especializado por dominio
│   ├── agents/             subagentes especializados
│   └── skills/             workflows repetibles
└── docs/decisions/         ADRs (decisiones arquitectónicas importantes)
```

El código fuente de JARVIS (`src/`, `tests/`, `pyproject.toml`) todavía no existe — se crea en la
siguiente fase, no en esta.

## Comandos esenciales

- `git status` / `git log --oneline` — estado del repo.
- `python -m pytest` — suite de tests (cuando exista código).
- `python -m ruff check .` / `ruff format .` — lint y formato (aplicado automáticamente por hook al editar `.py`).

## Convenciones globales

- Sin abstracciones prematuras, sin dependencias sin justificar, sin refactors fuera de alcance.
- Toda tarea significativa termina en: **implementado + verificado + revisado + sin regresiones conocidas**.
- Commits: nunca automáticos. El usuario decide cuándo commitear.

## Restricciones críticas

- El LLM nunca ejecuta un comando solo porque el usuario lo pidió: toda acción se clasifica como
  SAFE / CONFIRM / DANGEROUS antes de ejecutarse (`.claude/rules/security.md`).
- Nada de secretos en el repo. `.env`, credenciales y claves están en `.gitignore` y protegidos por hook.
- No se crean `.claude/commands/`: los workflows repetibles son Skills (ver `.claude/skills/`).

## Estrategia de delegación

El agente principal (orchestrator) conserva la decisión final. Delega a subagentes especializados
solo cuando aporta contexto aislado, revisión independiente o análisis paralelo — nunca para
tareas triviales. Detalle de cuándo usar cada agente en `.claude/rules/agents.md` y en la
descripción de cada agente bajo `.claude/agents/`.
