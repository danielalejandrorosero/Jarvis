---
paths:
  - "src/jarvis/tools/windows/**"
---

# Windows

Responsabilidad de esta regla: cómo el código de JARVIS (no el tooling de Claude Code) debe
interactuar con Windows. Complementa, no repite, las notas de sintaxis de PowerShell ya presentes
en el entorno de la herramienta de shell.

> El glob de `paths` asume el paquete `src/jarvis/` con una capa `tools/windows/` — todavía no
> existe ese código. Ajustar este glob en cuanto se defina el layout real del paquete (puede
> no llamarse exactamente así). Hasta entonces, esta rule solo se carga si ya hay archivos bajo
> esa ruta — es decir, no carga en ninguna sesión hoy.

## Principios para la capa Windows de JARVIS

- Toda interacción con el sistema (filesystem, procesos, registro, APIs de Windows) pasa por la
  capa security/policy antes de ejecutarse — nunca se invoca directamente desde el planner o el
  orchestrator. Ver `.claude/rules/architecture.md`.
- Preferir APIs de Windows tipadas (`pywin32`, `wmi`, o `subprocess` con argumentos explícitos en
  lista) sobre construir comandos de shell como strings — reduce superficie de inyección.
- Cualquier ruta de filesystem que provenga de input del usuario o del LLM se normaliza y valida
  contra path traversal antes de usarse.
- Operaciones sobre el registro de Windows son CONFIRM como mínimo; las que tocan `HKLM` o claves
  de arranque/servicios son DANGEROUS por defecto (`.claude/rules/security.md`).

## Cuándo delegar en `windows-engineer`

Cualquier implementación que toque procesos, filesystem a bajo nivel, registro, permisos de
Windows o automatización del sistema operativo. `windows-engineer` implementa;
`security-reviewer` revisa el resultado antes de considerarlo terminado.
