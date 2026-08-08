# 0006 — Un parámetro que cambia el radio de impacto se resuelve partiendo el tool, no evaluando riesgo por invocación

## Contexto

`PickChampionTool` (tools de League of Legends por voz, fase 5 extendida) recibe `champion_name`
y un booleano `lock`. Con `lock=false` hace *hover* de un campeón en selección — trivialmente
reversible, sin consecuencia fuera del propio cliente. Con `lock=true` confirma el pick
(`completed=true` en el PATCH a la LCU API) — compromete la elección para los 4 compañeros de una
partida real, con "deshacer" no garantizado (depende de que un compañero acepte un trade, no
disponible en toda cola/fase). Combinado con matching de nombre por voz deliberadamente permisivo
(`cutoff=0.45`, necesario para tolerar transcripciones garbladas) y sin paso de confirmación
propio, un nombre mal reconocido podía lockear un campeón no deseado sin red de seguridad.

Un `security-reviewer` marcó esto como hallazgo y señaló, correctamente, que no lo podía resolver
unilateralmente: es exactamente el tipo de trade-off de límite entre capas que
`.claude/rules/architecture.md` reserva para `architect`. La pregunta no era solo "¿`lock=true`
amerita CONFIRM?" sino, dado que sí lo amerita, **cómo** expresarlo — hoy `Tool.risk` es un
atributo de clase fijo (ADR-0005), declarado una vez, no evaluado por invocación.

## Decisión

Cuando un parámetro de un tool cambia el radio de impacto real de la acción lo suficiente como
para cruzar de SAFE a CONFIRM (o de CONFIRM a DANGEROUS), **se parte el tool en dos, cada uno con
su propio `RiskLevel` fijo** — no se extiende `PolicyEngine`/`Tool` para evaluar riesgo
condicional por parámetro.

Aplicado acá: `PickChampionTool` se separa en `PreviewChampionTool` (SAFE, solo hover) y
`LockChampionTool` (CONFIRM, lockea el pick). Lógica de matching de nombre y resolución del
`actionId` activo se comparten a nivel de módulo — la duplicación aceptable es la clase wrapper,
no la lógica, mismo patrón ya usado entre `open_app.py`/`close_app.py`.

## Por qué

- `Tool.risk` es estático a propósito (ADR-0005, "Por qué": evita que un tool quede sin
  clasificar por omisión — la lección concreta de ADR-0002 aplicada al runtime de JARVIS).
  Convertir `PolicyEngine.authorize_and_execute` a evaluar riesgo por invocación toca la ruta
  crítica de seguridad completa (el dispatch de *todos* los tools) para resolver un caso que hoy
  es único en el repo — abstracción prematura que `CLAUDE.md` pide evitar explícitamente.
- Partir en dos tools no toca `Tool`/`PolicyEngine` en absoluto: cada uno declara su `risk` fijo,
  tal como exige el contrato ya existente. Cero cambio a la ruta de seguridad.
- El eje relevante de SAFE vs. CONFIRM es mutación + alcance fuera de la propia máquina, no la
  duración del estado — un lock que solo dura una partida igual compromete a 4 personas reales
  sin posibilidad garantizada de deshacerlo, que es exactamente el escenario que CONFIRM existe
  para cubrir.

## Consecuencias

- Precedente para cualquier tool futuro con una bandera booleana que cambie el radio de impacto
  real (ejemplos concretos a vigilar: `force=true` al cerrar un proceso, `recursive=true` al
  borrar algo) — se resuelve partiendo el tool, no con un parámetro `risk_override` ni con
  evaluación condicional en `PolicyEngine`.
- `PolicyEngine.authorize_and_execute` sigue dispatchando puramente por `type(tool).risk` — si en
  el futuro aparece un caso que genuinamente no se puede resolver partiendo tools (no identificado
  todavía), amerita su propio ADR, no una excepción ad hoc a esta regla.
