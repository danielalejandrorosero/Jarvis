"""Persistencia de memoria de JARVIS: hechos sueltos en texto plano sobre el usuario (ADR-0004,
"Persistencia: SQLite").

Una sola tabla, sin esquema clave-valor rígido: quien decide *qué* vale la pena recordar y con
qué frase es el LLM (`jarvis.tools.remember.RememberTool`), no este módulo — así que el store
solo necesita texto libre con marca de tiempo, no columnas tipadas por concepto. Nada de RAG ni
embeddings: el pedido es "acordate de lo que te dije", no búsqueda semántica sobre un corpus
grande — `sqlite3` de stdlib alcanza (ADR-0004 ya descartó Postgres: sin proceso de servidor para
una app local de un solo usuario).

Sin ORM ni dependencia nueva: `sqlite3` (stdlib) es suficiente para una tabla y dos queries.

Hallazgos LOW de `security-reviewer` (junto al hallazgo HIGH mitigado en
`jarvis.audio.pipeline._build_system_prompt`, ver docstring de ese módulo): sin tope de longitud,
un `content` arbitrariamente largo (ej. el LLM "recordando" un fragmento entero de una página web)
se guardaría y reinyectaría entero en cada prompt futuro, amplificando el riesgo de contenido no
confiable persistente — `MAX_CONTENT_LENGTH` lo trunca, mismo principio que
`CONTENT_SNIPPET_MAX_CHARS`/`ANSWER_MAX_CHARS` en `jarvis.tools.search`. `MAX_STORED_FACTS` es
higiene de almacenamiento, no mitigación de seguridad primaria: sin tope, la tabla crece sin
límite con el uso prolongado; `save_fact` poda las filas más viejas por encima de ese tope en cada
escritura.

Segunda tabla, mismo archivo de DB: `speech_samples`. Distinta de `facts` en una dimensión clave —
quién decide qué se guarda. `facts` es contenido *curado*: el LLM juzga qué vale la pena recordar
y con qué frase (`RememberTool`). `speech_samples` es un log automático, sin juicio del LLM de por
medio: cada transcripción no vacía que produce el pipeline de voz (`jarvis.audio.pipeline.run`) se
guarda tal cual, verbatim, sin que el modelo decida si "vale la pena" — el propósito no es recordar
*qué* dijo el usuario sino *cómo* habla (registro, modismos, informalidad), para que
`_build_system_prompt` pueda mostrarle al LLM ejemplos de estilo a imitar en sus propias
respuestas. Como no hay gatekeeping del LLM, las filas se acumulan mucho más rápido que `facts` —
`MAX_STORED_SPEECH_SAMPLES` y sobre todo `DEFAULT_SPEECH_SAMPLE_LIST_LIMIT` están elegidos para
que el prompt solo vea un puñado de ejemplos recientes representativos, no un log de conversación
completo (ver constantes abajo).

Tercera tabla: `reminders` (`jarvis.tools.reminder.ReminderTool`, `jarvis.audio.timer_scheduler.
TimerScheduler`). A diferencia de `facts`/`speech_samples` (recall pasivo, reinyectado en el
prompt), `reminders` alimenta un mecanismo *proactivo*: un recordatorio tiene que sobrevivir hasta
su `due_at` y anunciarse solo, sin que el usuario pregunte nada — de ahí que necesite persistencia
real (un recordatorio con "en 20 minutos" tiene que sobrevivir aunque JARVIS se reinicie en el
medio) y una forma de consultar "¿qué está vencido ahora?" (`list_due_reminders`), no solo "leer
todo lo guardado" como `list_facts`/`list_speech_samples`.

Decisión de diseño deliberada: los *timers* ("Alexa, poné un timer de 10 minutos") NO usan esta
tabla — viven puramente en memoria, dentro de `TimerScheduler` (ver ese módulo). Un timer es
efímero por naturaleza (minutos, no días) y perderlo si JARVIS se reinicia a mitad de un timer de
cocina es una degradación aceptable y rara, mientras que pagar un round-trip a SQLite por algo tan
de corta vida (crear, y casi inmediatamente después borrar la fila) no compra nada. Un recordatorio
("recordame llamar a X mañana a la mañana") sí puede tener que sobrevivir horas o días — ahí la
persistencia si importa. Ambos casos comparten el mismo mecanismo de *anuncio* (el poll loop de
`TimerScheduler`), pero solo uno de los dos comparte almacenamiento con `facts`/`speech_samples`.

Cuarta tabla: `conversation_turns` (`jarvis.audio.pipeline.dispatch_turn`/`_build_system_prompt`).
Pedido explícito del usuario: "quiero que el agente tenga memoria que se acuerde que dije antes",
distinto tanto de `facts` (memoria *curada*, solo lo que el usuario pide explícitamente recordar)
como de `speech_samples` (registro/estilo de habla, no el contenido de la conversación) — acá se
guarda el flujo real de turnos (qué pidió el usuario, qué respondió JARVIS) para que el LLM tenga
continuidad de contexto de corto plazo entre turnos ("y ese", "hacé lo mismo de antes") incluso
después de un reinicio del proceso, no solo dentro de la lista de `messages` en memoria de un
`dispatch_turn` (que se descarta al terminar el turno). Igual que `speech_samples`, se guarda
automáticamente sin juicio del LLM — cada turno completo que pasa por `dispatch_turn` se persiste
tal cual. A diferencia de `speech_samples`, cada fila tiene dos campos de texto (lo que dijo el
usuario y lo que respondió JARVIS), no uno solo. Alcance deliberadamente chico: "últimos N turnos,
el más viejo se poda primero" alcanza para continuidad conversacional — no es un archivo de
transcript completo, ni tiene búsqueda ni resumen (ver CLAUDE.md: sin abstracciones prematuras)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Relativo al directorio desde el que se corre JARVIS (mismo criterio que
# `jarvis.config.load_dotenv`'s default `.env`), no un path absoluto hardcodeado — así se puede
# sobreescribir pasando `db_path` explícito (tests con `tmp_path`, u otro layout de despliegue)
# sin tocar este módulo.
DEFAULT_DB_PATH = Path("data/jarvis.db")

# Tope por defecto de `list_facts`: evita que el system prompt crezca sin límite turno a turno a
# medida que se acumulan hechos — un recorte razonable de contexto reciente, no el historial
# completo.
DEFAULT_LIST_LIMIT = 20

# Tope duro de longitud por hecho (hallazgo LOW #2 de `security-reviewer`): un hecho se trunca a
# esto antes de guardarse, nunca se persiste (ni se reinyecta) contenido de largo arbitrario.
MAX_CONTENT_LENGTH = 500

# Tope duro sobre el total histórico de filas en `facts` (hallazgo LOW #3, menor, no bloqueante):
# `save_fact` poda las más viejas por encima de este número en cada escritura, así la tabla nunca
# crece sin límite con el uso prolongado.
MAX_STORED_FACTS = 500

# Tope por defecto de `list_speech_samples`: deliberadamente chico frente a `DEFAULT_LIST_LIMIT`
# (20) — el objetivo es un puñado de ejemplos de estilo recientes y representativos para el
# system prompt, no un transcript. Cuantos más ejemplos, más footprint de contenido sin curar
# (aunque sea del propio usuario, no de terceros) reinyectado en cada turno.
DEFAULT_SPEECH_SAMPLE_LIST_LIMIT = 8

# Tope duro de longitud por muestra de habla: mismo orden de magnitud que `MAX_CONTENT_LENGTH`
# — un comando de voz normal nunca se acerca a esto; existe para el caso raro de una
# transcripción anormalmente larga (ruido, Whisper alucinando sobre audio largo).
MAX_SPEECH_SAMPLE_LENGTH = 500

# Tope duro sobre el total histórico de filas en `speech_samples`. Se guarda automáticamente sin
# juicio del LLM (cada transcripción no vacía), así que la tabla crece mucho más rápido que
# `facts` con el uso normal — el tope evita que el archivo de DB crezca sin límite indefinidamente
# con meses de uso diario. No es la mitigación principal de footprint en el prompt (esa es
# `DEFAULT_SPEECH_SAMPLE_LIST_LIMIT`, que solo importa cuántas de estas filas se leen por turno),
# es higiene de almacenamiento, igual que `MAX_STORED_FACTS`.
MAX_STORED_SPEECH_SAMPLES = 300

# Tope duro de longitud por texto de recordatorio — mismo orden de magnitud que
# `MAX_CONTENT_LENGTH`/`MAX_SPEECH_SAMPLE_LENGTH`, mismo motivo: un recordatorio hablado nunca se
# acerca a esto, existe para el caso raro de una transcripción/alucinación anormalmente larga.
MAX_REMINDER_TEXT_LENGTH = 300

# Tope duro sobre el total de recordatorios *pendientes* a la vez (no histórico: las filas se
# borran apenas se anuncian, ver `delete_reminder`) — higiene ante el caso de que el LLM programe
# muchos recordatorios sin que ninguno llegue a vencer nunca; no un límite pensado para uso normal.
MAX_STORED_REMINDERS = 100

# Tope por defecto de `list_recent_conversation_turns`: cuántos turnos pasados se inyectan en el
# system prompt de cada turno nuevo (`jarvis.audio.pipeline._build_system_prompt`). Más chico que
# `DEFAULT_SPEECH_SAMPLE_LIST_LIMIT` (8) a propósito: cada turno acá pesa el doble en el prompt
# (texto del usuario Y de JARVIS, no una sola línea), y el propósito es continuidad de corto plazo
# ("y ese", "hacé lo mismo de antes"), no un historial largo — un puñado de turnos recientes
# alcanza.
DEFAULT_CONVERSATION_TURN_LIST_LIMIT = 6

# Tope duro de longitud por campo (`user_text`/`assistant_text`) — mismo orden de magnitud que
# `MAX_CONTENT_LENGTH`/`MAX_SPEECH_SAMPLE_LENGTH`/`MAX_REMINDER_TEXT_LENGTH`: un turno de voz
# normal nunca se acerca a esto, existe para el caso raro de una transcripción/respuesta
# anormalmente larga.
MAX_CONVERSATION_TURN_TEXT_LENGTH = 500

# Tope duro sobre el total histórico de filas en `conversation_turns`. Igual que
# `speech_samples`, se guarda automáticamente sin juicio del LLM (un turno por cada
# `dispatch_turn` completo), así que crece con el uso normal — mismo orden de magnitud que
# `MAX_STORED_SPEECH_SAMPLES` (300), higiene de almacenamiento, no la mitigación de footprint del
# prompt (esa es `DEFAULT_CONVERSATION_TURN_LIST_LIMIT`).
MAX_STORED_CONVERSATION_TURNS = 300

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_CREATE_SPEECH_SAMPLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS speech_samples (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_CREATE_REMINDERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    due_at TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_CREATE_CONVERSATION_TURNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class ConversationTurn:
    """Una fila de `conversation_turns`: lo que dijo el usuario y la respuesta final de JARVIS en
    ese turno. Sin `id`/`created_at` — quien consume esto (`_build_system_prompt`) solo necesita
    los dos textos, no metadata de la fila (mismo criterio que `list_facts`/`list_speech_samples`
    devolviendo `list[str]`, acá con dos campos hace falta un dataclass en vez de un str)."""

    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class Reminder:
    """Una fila de `reminders`: `due_at`/`created_at` son ISO 8601 (UTC, aware) en texto — mismo
    formato que `facts.created_at`/`speech_samples.created_at`, sin un tipo de columna `DATETIME`
    dedicado (`sqlite3` no tiene uno nativo; comparar como texto ISO en Python, no en SQL, es más
    simple que confiar en que `sqlite3` ordene/compare fechas como texto correctamente — ver
    `list_due_reminders`)."""

    id: int
    text: str
    due_at: str


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Abrir una conexión a `db_path`, creando el directorio contenedor y las tablas `facts`,
    `speech_samples`, `reminders` y `conversation_turns` si todavía no existen — cubre tanto el
    primer uso (archivo de DB inexistente) como usos posteriores (tablas ya creadas,
    `CREATE TABLE IF NOT EXISTS` es un no-op)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(_CREATE_SPEECH_SAMPLES_TABLE_SQL)
    conn.execute(_CREATE_REMINDERS_TABLE_SQL)
    conn.execute(_CREATE_CONVERSATION_TURNS_TABLE_SQL)
    conn.commit()
    return conn


def _prune_reminders_beyond_cap(conn: sqlite3.Connection) -> None:
    """Podar los recordatorios pendientes por encima de `MAX_STORED_REMINDERS`, evictando primero
    el de `due_at` MÁS LEJANO en el futuro — no por orden de inserción/`id` (hallazgo LOW de
    `security-reviewer`: podar por `id` evictaría el recordatorio más viejo *creado*, aunque su
    `due_at` fuera más próximo e importante que el de uno recién creado con `due_at` lejano;
    `ReminderTool.execute()` tampoco avisa nunca de esta poda, así que el que se pierde tiene que
    ser, como mínimo, el menos urgente).

    Ordena en Python, no con `ORDER BY due_at` en SQL — mismo motivo que documenta
    `list_due_reminders`: `datetime.isoformat()` omite los microsegundos cuando son exactamente
    cero, lo que puede dejar timestamps de distinta longitud de string y una comparación
    lexicográfica incorrecta. Con pocas filas a la vez (`MAX_STORED_REMINDERS`), traer todas y
    ordenar en Python es simple y suficientemente rápido, igual que ahí.
    """
    rows = conn.execute("SELECT id, due_at FROM reminders").fetchall()
    if len(rows) <= MAX_STORED_REMINDERS:
        return
    parsed: list[tuple[int, datetime]] = []
    for row_id, row_due_at in rows:
        try:
            parsed.append((row_id, datetime.fromisoformat(row_due_at)))
        except ValueError:
            # Fila con due_at malformado — no debería poder existir (save_reminder valida antes
            # de insertar), pero si aparece de todos modos, se la trata como la más "lejana"
            # (datetime.max) para que sea la primera candidata a evictarse, no una real.
            parsed.append((row_id, datetime.max.replace(tzinfo=UTC)))
    # Más próximo primero; lo que sobra por encima del tope son los de due_at más lejano.
    parsed.sort(key=lambda item: item[1])
    ids_to_evict = [row_id for row_id, _ in parsed[MAX_STORED_REMINDERS:]]
    conn.executemany(
        "DELETE FROM reminders WHERE id = ?", [(row_id,) for row_id in ids_to_evict]
    )


def _truncate(text: str, *, max_chars: int) -> str:
    """Capear `text` a `max_chars`, con un indicador de corte — mismo helper que
    `jarvis.tools.search._truncate` (duplicado a propósito, ver `_escape_untrusted` en
    `jarvis.audio.pipeline` para el mismo razonamiento: no es un contrato compartido entre
    módulos)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def save_fact(content: str, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Guardar `content` (un hecho en texto plano, ej. "el usuario prefiere respuestas cortas")
    con la marca de tiempo actual (UTC, ISO 8601), truncado a `MAX_CONTENT_LENGTH`.

    Lanza `ValueError` si `content` está vacío o es solo espacios — validación mínima acá; el
    mensaje amigable de "no se especificó qué recordar" es responsabilidad del tool
    (`jarvis.tools.remember.RememberTool`), no de este módulo de persistencia.

    Tras insertar, poda las filas más viejas por encima de `MAX_STORED_FACTS` (hallazgo LOW #3)
    — en la misma transacción, así una escritura nunca deja la tabla momentáneamente por encima
    del tope.
    """
    stripped = content.strip()
    if not stripped:
        raise ValueError("content no puede estar vacío")
    truncated = _truncate(stripped, max_chars=MAX_CONTENT_LENGTH)
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO facts (content, created_at) VALUES (?, ?)",
            (truncated, datetime.now(UTC).isoformat()),
        )
        conn.execute(
            "DELETE FROM facts WHERE id NOT IN "
            "(SELECT id FROM facts ORDER BY id DESC LIMIT ?)",
            (MAX_STORED_FACTS,),
        )
        conn.commit()
    finally:
        conn.close()


def list_facts(
    *, db_path: str | Path = DEFAULT_DB_PATH, limit: int = DEFAULT_LIST_LIMIT
) -> list[str]:
    """Devolver hasta `limit` hechos guardados, más reciente primero.

    Sobre un `db_path` que todavía no existe (nunca se llamó `save_fact` con este path), inicializa
    la DB vacía igual que `_connect` y devuelve `[]` — no es un error, es el estado inicial
    esperado antes de la primera memoria guardada.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT content FROM facts ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def save_speech_sample(text: str, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Guardar `text` (una transcripción real del usuario, verbatim) con la marca de tiempo
    actual (UTC, ISO 8601), truncada a `MAX_SPEECH_SAMPLE_LENGTH`.

    A diferencia de `save_fact`, no hay decisión del LLM de por medio — quien llama a esto
    (`jarvis.audio.pipeline.run`) lo hace incondicionalmente sobre cada transcripción no vacía; la
    única validación acá es rechazar texto vacío/solo-espacios (`ValueError`), igual que
    `save_fact`, tanto por higiene de datos como por si esta función se llama alguna vez desde
    otro punto que no sea `run()` y no filtre eso de antemano.

    Poda las filas más viejas por encima de `MAX_STORED_SPEECH_SAMPLES` tras insertar, en la misma
    transacción — mismo patrón que `save_fact`/`MAX_STORED_FACTS`.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("text no puede estar vacío")
    truncated = _truncate(stripped, max_chars=MAX_SPEECH_SAMPLE_LENGTH)
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO speech_samples (text, created_at) VALUES (?, ?)",
            (truncated, datetime.now(UTC).isoformat()),
        )
        conn.execute(
            "DELETE FROM speech_samples WHERE id NOT IN "
            "(SELECT id FROM speech_samples ORDER BY id DESC LIMIT ?)",
            (MAX_STORED_SPEECH_SAMPLES,),
        )
        conn.commit()
    finally:
        conn.close()


def list_speech_samples(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = DEFAULT_SPEECH_SAMPLE_LIST_LIMIT,
) -> list[str]:
    """Devolver hasta `limit` muestras de habla guardadas, más reciente primero.

    Mismo comportamiento que `list_facts` sobre un `db_path` que todavía no existe: inicializa la
    DB vacía y devuelve `[]`, no es un error.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT text FROM speech_samples ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def save_reminder(
    text: str, due_at: str, *, db_path: str | Path = DEFAULT_DB_PATH
) -> None:
    """Guardar un recordatorio: `text` (qué recordarle al usuario) y `due_at` (cuándo, ISO 8601
    *aware* — con offset de zona horaria explícito, ej. lo que devuelve `datetime.isoformat()`
    sobre un `datetime` con `tzinfo`) truncado/validado antes de persistir.

    Lanza `ValueError` si `text` está vacío/solo espacios (mismo criterio que `save_fact`), o si
    `due_at` no parsea como un timestamp ISO 8601 *con* zona horaria — un `due_at` naive (sin
    offset) no se puede comparar de forma inequívoca contra el `now` aware que usa
    `list_due_reminders` (`TypeError` de Python al comparar aware con naive), así que se rechaza
    acá, en el punto de escritura, en vez de fallar silenciosamente más tarde dentro del poll loop
    de `TimerScheduler` (`jarvis.audio.timer_scheduler`) — un recordatorio que nunca se anuncia
    por un timestamp mal formado es peor que uno que nunca se guardó, porque el usuario cree que
    quedó programado.

    Poda por encima de `MAX_STORED_REMINDERS` tras insertar, en la misma transacción — a
    diferencia de `save_fact`/`save_speech_sample` (que podan por orden de inserción, lo más
    viejo primero), acá se evicta por `due_at` — lo más lejano en el futuro primero, ver
    `_prune_reminders_beyond_cap` — porque un recordatorio no pierde relevancia por haber sido
    *creado* hace más tiempo, sino por estar programado más lejos en el futuro. En la práctica
    esto casi nunca dispara: los recordatorios se borran solos apenas se anuncian
    (`delete_reminder`), así que no se acumulan como `facts`/`speech_samples`; el tope es una red
    de seguridad, no la mitigación principal.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("text no puede estar vacío")
    truncated = _truncate(stripped, max_chars=MAX_REMINDER_TEXT_LENGTH)
    try:
        parsed_due_at = datetime.fromisoformat(due_at)
    except ValueError as exc:
        raise ValueError(
            f"due_at no es un timestamp ISO 8601 válido: {due_at!r}"
        ) from exc
    if parsed_due_at.tzinfo is None:
        raise ValueError(
            "due_at debe incluir información de zona horaria (datetime aware), "
            f"se recibió: {due_at!r}"
        )
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
            (truncated, due_at, datetime.now(UTC).isoformat()),
        )
        _prune_reminders_beyond_cap(conn)
        conn.commit()
    finally:
        conn.close()


def list_reminders(*, db_path: str | Path = DEFAULT_DB_PATH) -> list[Reminder]:
    """Devolver todos los recordatorios pendientes, ordenados por `due_at` ascendente (el próximo
    a vencer primero) — para un futuro tool de consulta ("qué recordatorios tengo"), no usado
    todavía por `TimerScheduler` (que usa `list_due_reminders`).

    Sobre un `db_path` que todavía no existe, inicializa la DB vacía y devuelve `[]`, igual que
    `list_facts`/`list_speech_samples`."""
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT id, text, due_at FROM reminders ORDER BY due_at ASC"
        )
        return [
            Reminder(id=row[0], text=row[1], due_at=row[2]) for row in cursor.fetchall()
        ]
    finally:
        conn.close()


def list_due_reminders(
    now: datetime, *, db_path: str | Path = DEFAULT_DB_PATH
) -> list[Reminder]:
    """Devolver los recordatorios cuyo `due_at` ya pasó (`due_at <= now`), ordenados por `due_at`
    ascendente — lo que `TimerScheduler` sondea periódicamente para saber qué anunciar.

    `now` tiene que ser un `datetime` *aware* (mismo motivo que `save_reminder` rechaza `due_at`
    naive: comparar aware con naive lanza `TypeError`). El filtrado se hace en Python, parseando
    cada `due_at` de vuelta a `datetime` y comparando objetos reales — no una comparación de texto
    ISO en SQL: `datetime.isoformat()` omite los microsegundos cuando son exactamente cero, lo que
    puede dejar dos timestamps del mismo formato con longitudes de string distintas y una
    comparación lexicográfica incorrecta (`"...:56"` compara como *menor* que `"...:56.500000"`
    sin importar el valor real). Con pocas filas pendientes a la vez (`MAX_STORED_REMINDERS`),
    traer todas y filtrar en Python es simple y suficientemente rápido — no hace falta un índice
    ni una cláusula `WHERE` con conversión de tipos.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT id, text, due_at FROM reminders ORDER BY due_at ASC"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()
    due: list[Reminder] = []
    for row_id, row_text, row_due_at in rows:
        try:
            parsed_due_at = datetime.fromisoformat(row_due_at)
        except ValueError:
            # Fila con un due_at malformado — no debería poder existir (save_reminder valida
            # antes de insertar), pero se salta en vez de lanzar: un dato corrupto en una fila no
            # tiene que tumbar el poll loop entero de TimerScheduler para las demás.
            continue
        if parsed_due_at <= now:
            due.append(Reminder(id=row_id, text=row_text, due_at=row_due_at))
    return due


def delete_reminder(reminder_id: int, *, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Borrar el recordatorio `reminder_id` — se llama una vez que `TimerScheduler` ya lo anunció,
    para que no vuelva a anunciarse en el próximo poll. Idempotente: si el id no existe (ya
    borrado, o nunca existió), no hace nada ni lanza."""
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        conn.commit()
    finally:
        conn.close()


def save_conversation_turn(
    user_text: str, assistant_text: str, *, db_path: str | Path = DEFAULT_DB_PATH
) -> None:
    """Guardar un turno completo de conversación: lo que dijo el usuario (`user_text`) y la
    respuesta final de JARVIS en ese turno (`assistant_text`), con la marca de tiempo actual
    (UTC, ISO 8601), cada campo truncado a `MAX_CONVERSATION_TURN_TEXT_LENGTH`.

    A diferencia de `save_fact`/`save_speech_sample`/`save_reminder`, solo `user_text` tiene que
    ser no vacío (`ValueError` si está vacío/solo espacios, mismo criterio que esas funciones
    aplican a su campo principal) — `assistant_text` SÍ puede ser la cadena vacía de forma
    legítima: `SYSTEM_PROMPT` (`jarvis.audio.pipeline`) instruye al LLM a responder con la cadena
    vacía cuando el turno es "reproducir algo" (para no hablar por encima de lo que empieza a
    sonar), y ese silencio es en sí mismo información real sobre qué pasó en el turno, no un dato
    inválido a rechazar acá. `dispatch_turn` (`jarvis.audio.pipeline`) siempre llama a esto con un
    `user_text` no vacío (el pipeline de voz ya filtra transcripciones vacías antes de invocarlo),
    así que el `ValueError` acá es una red de seguridad, no el camino esperado.

    Poda las filas más viejas por encima de `MAX_STORED_CONVERSATION_TURNS` tras insertar, en la
    misma transacción — mismo patrón que `save_fact`/`save_speech_sample`.
    """
    stripped_user_text = user_text.strip()
    if not stripped_user_text:
        raise ValueError("user_text no puede estar vacío")
    truncated_user_text = _truncate(
        stripped_user_text, max_chars=MAX_CONVERSATION_TURN_TEXT_LENGTH
    )
    truncated_assistant_text = _truncate(
        assistant_text.strip(), max_chars=MAX_CONVERSATION_TURN_TEXT_LENGTH
    )
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO conversation_turns (user_text, assistant_text, created_at) "
            "VALUES (?, ?, ?)",
            (
                truncated_user_text,
                truncated_assistant_text,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.execute(
            "DELETE FROM conversation_turns WHERE id NOT IN "
            "(SELECT id FROM conversation_turns ORDER BY id DESC LIMIT ?)",
            (MAX_STORED_CONVERSATION_TURNS,),
        )
        conn.commit()
    finally:
        conn.close()


def list_recent_conversation_turns(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = DEFAULT_CONVERSATION_TURN_LIST_LIMIT,
) -> list[ConversationTurn]:
    """Devolver hasta `limit` turnos de conversación guardados, más reciente primero — mismo
    contrato de orden que `list_facts`/`list_speech_samples`. Quien arma el bloque del system
    prompt (`jarvis.audio.pipeline._build_system_prompt`) es responsable de invertir el orden si
    quiere presentarlos cronológicamente (más viejo primero, para que se lean como una
    conversación real) — no este módulo, que mantiene el mismo contrato de orden que el resto de
    los `list_*` de acá.

    Sobre un `db_path` que todavía no existe, inicializa la DB vacía y devuelve `[]`, igual que
    `list_facts`/`list_speech_samples`/`list_reminders`.
    """
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "SELECT user_text, assistant_text FROM conversation_turns "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [
            ConversationTurn(user_text=row[0], assistant_text=row[1])
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()
