# JARVIS

Asistente personal de voz para Windows, controlado por un LLM. Se llama **Alexa** cuando habla
(solo por voz — el proyecto y el código siguen llamándose JARVIS internamente).

No es un chatbot con acceso a shell. Es un pipeline con capas separadas donde el modelo propone
acciones pero nunca decide por sí mismo si se ejecutan:

```
USUARIO → ORCHESTRATOR → PLANNER → TOOLS → SECURITY/POLICY → WINDOWS
```

Cada acción que JARVIS puede tomar se clasifica en uno de tres niveles antes de ejecutarse:

- **SAFE** — se ejecuta directo (leer, consultar, abrir apps/URLs, info del sistema).
- **CONFIRM** — pide confirmación hablada antes de ejecutar (cerrar procesos, lockear un pick de
  campeón en League, etc.).
- **DANGEROUS** — nunca se ejecuta de forma automática, ni con confirmación de un solo paso.

Ver `.claude/rules/security.md` y `.claude/rules/architecture.md` para el detalle completo del
modelo de seguridad y de capas.

## Qué puede hacer hoy

- **Activación por voz**: responde a "Alexa", "Hey Jarvis" o "Hey Mycroft". Entiende comandos de
  seguimiento sin repetir la wake word ("Follow-Up Mode"), y se puede poner a dormir/despertar
  por voz ("Alexa, descansá" / "Alexa, volvé").
- **Clima, búsqueda web, recordar datos, abrir apps/URLs, cerrar procesos**.
- **Timers y recordatorios** con anuncio proactivo por voz, persistidos en SQLite.
- **Control de medios y volumen** (universal, cualquier reproductor), **info del sistema en vivo**
  (CPU/RAM/GPU), **capturas de pantalla** por voz.
- **Memoria**: hechos que el usuario pide recordar explícitamente, muestras de estilo de habla
  para imitar su forma de hablar, e historial de conversación reciente — todo en SQLite local,
  todo sobrevive un reinicio.
- **League of Legends** (vía la LCU API local del propio cliente): aceptar partidas automáticamente,
  buscar/cancelar cola, elegir campeón (preview sin confirmación, lock con confirmación hablada),
  configurar runas y hechizos de invocador — con reglas específicas por modo de juego (Arena,
  ARAM) donde runas/hechizos no siempre aplican.

## Stack

- **Python 3.12** — núcleo (`src/jarvis/`), tipado estricto con `mypy`, formateado con `ruff`.
- **OpenAI** — transcripción (`gpt-4o-transcribe`) y voz (`gpt-4o-mini-tts`, acento
  latinoamericano/colombiano). **DeepSeek** — el LLM que razona y decide qué tools llamar.
- **SQLite** — memoria persistente (hechos, muestras de habla, recordatorios, historial de
  conversación).
- **Windows** — Scheduled Task para arranque automático (`scripts/start_jarvis.ps1`), APIs
  nativas (`ctypes`) para medios/volumen, WASAPI loopback para detectar audio del sistema.

## Estructura

```
jarvis/
├── src/jarvis/
│   ├── audio/       pipeline de voz: wake word, STT, TTS, loop principal, timers
│   ├── tools/        cada capacidad invocable por voz (un archivo por tool)
│   ├── security/      PolicyEngine — único punto de paso antes de ejecutar un tool
│   ├── memory/        persistencia SQLite
│   ├── league/        integración con la LCU API de League of Legends
│   └── llm/            cliente del LLM (DeepSeek, function-calling)
├── tests/              espeja la estructura de src/
├── docs/decisions/     ADRs — por qué se decidió cada cosa así
├── .claude/rules/       convenciones del repo (Python, seguridad, arquitectura, git, testing)
└── scripts/            arranque automático (Scheduled Task de Windows)
```

## Cómo correrlo

```powershell
python -m pytest              # suite de tests
python -m ruff check .        # lint
python -m ruff format .       # formato
```

El arranque automático al iniciar sesión de Windows está configurado como Scheduled Task —
ver `scripts/start_jarvis.ps1`. Requiere un `.env` con las credenciales de API (ver
`.env.example`, nunca commiteado).

## Decisiones de diseño

Cada decisión no trivial queda registrada como ADR en `docs/decisions/` — arquitectura de
tool-calling, por qué la seguridad se aplica en la capa de policy y no en el LLM, por qué un
parámetro que cambia el radio de impacto de una acción se resuelve partiendo el tool en dos en
vez de evaluar riesgo por invocación, etc. Si una decisión de este README no tiene ADR asociado,
probablemente todavía no se justificó formalmente — preguntar antes de asumir el motivo.
