# 0001 — Python 3.12 como stack núcleo de JARVIS

## Contexto

JARVIS necesita un lenguaje para su capa de orquestación LLM/tools y su capa de automatización de
Windows. El repo estaba vacío; no había runtime instalado (ni Git ni Python).

## Decisión

Python 3.12 como lenguaje núcleo. Se instaló junto con Git vía `winget` con confirmación explícita
del usuario (instalar software es CONFIRM, no una decisión que la IA toma sola).

## Por qué

- Ecosistema maduro para LLM/tool-calling (SDKs de Anthropic/OpenAI, frameworks de agentes).
- Acceso directo a Windows vía `pywin32`/`wmi`/`subprocess` sin capa intermedia.
- Tipado gradual (`mypy`) suficiente para un proyecto de meses sin la sobrecarga de un lenguaje
  estáticamente tipado desde el día uno.

## Alternativas consideradas

- **Node/TypeScript**: buen ecosistema de agentes también, pero peor integración nativa con APIs
  de Windows a bajo nivel sin depender de bindings menos maduros.
- **C#/.NET**: integración de Windows superior, pero ecosistema de tooling LLM/agentes menos
  maduro y mayor fricción para iterar rápido en la fase de diseño de agentes.

## Consecuencias

- `.claude/rules/python.md` y `.claude/agents/python-engineer.md` asumen este stack.
- El hook `python-format-on-save.ps1` asume `ruff` como formateador/linter — se activa cuando
  `ruff` se instale como dependencia de desarrollo, no antes.
- No se crea `pyproject.toml`/`src/` en esta fase — eso es implementación de JARVIS, fuera de
  scope de esta sesión (entorno de desarrollo, no producto).
