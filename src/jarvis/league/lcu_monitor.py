"""Auto-accept de colas de matchmaking de League of Legends vía la LCU API (League Client Update).

Arquitectura: igual que `jarvis.audio.loopback.SystemAudioMonitor`, esto es un servicio de fondo
proactivo (thread daemon, lifecycle `start()`/`stop()`), NO un `Tool` invocado por el LLM — no
pasa por `PolicyEngine` (`jarvis.security.policy`) ni por la clasificación SAFE/CONFIRM/DANGEROUS
que aplica a *acciones que el usuario pide por voz* (`.claude/rules/architecture.md`). No hay una
decisión del LLM de por medio acá, ni un usuario al que pedirle confirmación en el momento (un
ready-check dura ~10-20s; no aceptarlo automáticamente simplemente significa que el usuario lo
hubiera aceptado él mismo con un click). Clasificación de riesgo, a efectos de
`.claude/rules/security.md` y de la revisión de `security-reviewer`: **SAFE** — no muta estado de
Windows, no toca otros procesos de forma destructiva, y solo interactúa con una API local que el
propio cliente del juego expone activamente para este caso de uso (integración pre-partida
explícitamente permitida por la política de terceros de Riot Games — igual que Blitz.gg o
Porofessor; distinto de botear dentro de la partida, que no está permitido y este módulo no hace).

Qué es la LCU API (documentación no oficial pero ampliamente usada: hextechdocs.dev; referencias
de implementación: `moleicafe/lol-auto-accept`, `r48n34/lol-accepter`): mientras el League Client
está corriendo, escribe un "lockfile" en su directorio de instalación (`<install_dir>/lockfile`,
comúnmente `C:\\Riot Games\\League of Legends\\lockfile`, pero el directorio de instalación puede
variar) con 5 campos separados por `:`: `nombre_proceso:pid:puerto:password:protocolo`. Ese
`password` es un token de auth HTTP Basic (usuario fijo `riot`) contra
`https://127.0.0.1:<puerto>`, servido con un certificado TLS autofirmado — cualquier integración
de la LCU (Blitz, Porofessor, este monitor) tiene que aceptar ese certificado para conectarse.

Descubrimiento del lockfile (`_find_lockfile_path`): primero se prueban directorios de instalación
conocidos/default (`DEFAULT_INSTALL_DIRS` — pasable por constructor para tests, mismo espíritu que
`START_MENU_DIRECTORIES` en `jarvis.tools.open_app`, pero como parámetro de instancia en vez de
variable de módulo porque acá sí hay una clase con constructor). Si ninguno tiene lockfile, se
intenta descubrir el directorio real preguntándole a Windows dónde vive el ejecutable de
`LeagueClientUx.exe` si está corriendo (`_discover_install_dir_from_running_process`, vía
PowerShell `Get-CimInstance` — no WMIC, deprecado en builds recientes de Windows) y usar su
carpeta contenedora (el lockfile vive junto al ejecutable). Cubre instalaciones en una unidad o
ruta no estándar sin tener que enumerar cada variante posible.

Conexión y certificado autofirmado (`_build_client`): el host de conexión está **hardcodeado a
`127.0.0.1`**, nunca tomado del lockfile ni de ninguna otra fuente — solo protocolo y puerto vienen
de ahí. `verify=False` es necesario porque el League Client sirve un certificado autofirmado (así
funciona la LCU API para cualquier integración de terceros, no una debilidad introducida acá) y
está acotado a este `httpx.Client` puntual — nunca se deshabilita la verificación de certificados
a nivel de proceso/librería ni para ningún otro cliente HTTP de JARVIS (`jarvis.tools.search`,
`jarvis.tools.weather` siguen verificando normalmente).

Resiliencia (mismo estándar que `SystemAudioMonitor`, ver `jarvis.audio.loopback`): "League no
está corriendo" es un estado normal y esperado (el usuario no siempre está jugando), no un error —
se loguea una sola vez a nivel INFO por transición de estado, no en cada poll. Si el cliente se
cierra a mitad de sesión (lockfile desaparece, la conexión empieza a fallar), el monitor vuelve al
estado "esperando a que arranque League" sin tumbar el thread; un error inesperado cualquiera en
el ciclo de polling se atrapa en el borde exterior (`_run`) por el mismo motivo documentado en
`jarvis.audio.pipeline.run()` — un fallo puntual no puede tumbar el thread de fondo entero.

`connect_to_lcu` (más abajo): único símbolo público agregado a este módulo pensado para un
consumidor distinto de `LCUAutoAcceptMonitor` — los tools de League invocados por voz
(`jarvis.tools.lol_runes.SetRunesTool`, `jarvis.tools.lol_summoner_spells.SetSummonerSpellsTool`,
`jarvis.tools.lol_champion_select.PickChampionTool`). Esos tools SÍ pasan por `PolicyEngine`
(`jarvis.security.policy`) y por la clasificación SAFE/CONFIRM/DANGEROUS — a diferencia de
`LCUAutoAcceptMonitor`, ahí sí hay una decisión de tool-calling del LLM de por medio. Comparten con
el monitor únicamente la mecánica de conexión (lockfile, auth, certificado autofirmado), no la
clasificación de riesgo ni el lifecycle.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

LCU_USERNAME = "riot"
READY_CHECK_PHASE = "ReadyCheck"
GAMEFLOW_PHASE_ENDPOINT = "/lol-gameflow/v1/gameflow-phase"
READY_CHECK_ACCEPT_ENDPOINT = "/lol-matchmaking/v1/ready-check/accept"

REQUEST_TIMEOUT_SECONDS = 5.0
# Poll rápido mientras hay cliente conectado: una API local es barata, y un ready-check dura solo
# ~10-20s antes de expirar — un intervalo lento arriesgaría perderlo. Poll lento mientras se espera
# a que League arranque: no hay apuro, y evita gastar ciclos (y en el caso de fallback, invocar
# PowerShell) a repetición sin necesidad.
GAMEFLOW_POLL_SECONDS = 1.5
WAITING_FOR_CLIENT_POLL_SECONDS = 5.0
PROCESS_LOOKUP_TIMEOUT_SECONDS = 5.0

LEAGUE_CLIENT_PROCESS_NAME = "LeagueClientUx.exe"
LOCKFILE_NAME = "lockfile"

# Únicos dos valores que el League Client real escribe en el campo `protocolo` del lockfile.
# `_parse_lockfile` valida contra esta allow-list (en vez de aceptar cualquier string) porque ese
# campo se interpola directo en la URL base de `_build_client`
# (`f"{credentials.protocol}://127.0.0.1:{credentials.port}"`) — un lockfile con, por ejemplo,
# `protocolo="https://evil.com"` produciría una URL cuyo netloc real (para cualquier parser de
# URL estándar, incluido el que usa `httpx`) es `evil.com`, no `127.0.0.1`, contradiciendo la
# garantía documentada más arriba de que el host de conexión está hardcodeado. El lockfile lo
# escribe el propio proceso de League, así que en el caso normal esto nunca dispara — pero
# tratarlo explícitamente como parseo inválido (igual que puerto no numérico o password vacío)
# es la misma postura defensiva que ya aplica el resto de esta función ante un lockfile con
# forma inesperada.
_VALID_LOCKFILE_PROTOCOLS = frozenset({"http", "https"})


def _default_install_dirs() -> list[Path]:
    """Ubicación(es) default conocida(s) del directorio de instalación de League of Legends.
    Función, no valor calculado a import-time (mismo motivo que
    `jarvis.tools.open_app._default_start_menu_directories`) — se evalúa una vez al definir
    `DEFAULT_INSTALL_DIRS`, el default que usa `LCUAutoAcceptMonitor` cuando no se le pasa
    `install_dirs` explícito (los tests sí lo pasan, apuntando a un `tmp_path`).
    """
    return [Path(r"C:\Riot Games\League of Legends")]


DEFAULT_INSTALL_DIRS: list[Path] = _default_install_dirs()


@dataclass(frozen=True)
class LCUCredentials:
    """Credenciales parseadas de un lockfile — ver `_parse_lockfile`."""

    port: int
    password: str
    protocol: str


def _parse_lockfile(content: str) -> LCUCredentials | None:
    """Parsear el contenido de un lockfile
    (`nombre_proceso:pid:puerto:password:protocolo`).

    Devuelve `None` ante cualquier formato inesperado — nunca lanza. Un lockfile a medio escribir
    (el cliente recién arrancando, o cerrándose) es un estado transitorio normal de por sí, no un
    error a propagar. `protocolo` se valida contra `_VALID_LOCKFILE_PROTOCOLS` (nunca aceptado
    tal cual) — ver comentario junto a esa constante para el motivo concreto.
    """
    parts = content.strip().split(":")
    if len(parts) != 5:
        return None
    _process_name, _pid, port_str, password, protocol = parts
    if not port_str.isdigit() or not password:
        return None
    if protocol not in _VALID_LOCKFILE_PROTOCOLS:
        return None
    return LCUCredentials(port=int(port_str), password=password, protocol=protocol)


def _read_lockfile(path: Path) -> LCUCredentials | None:
    """Leer y parsear el lockfile en `path`. `None` tanto si no se puede leer (carrera esperable:
    el archivo pudo borrarse entre encontrarlo y leerlo, el cliente cerrándose justo en ese
    instante) como si el contenido no tiene el formato esperado — ambos se tratan igual por el
    llamador (`_run`): volver a esperar, no un error fatal.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_lockfile(content)


def _discover_install_dir_from_running_process() -> Path | None:
    """Último recurso cuando ningún directorio de `DEFAULT_INSTALL_DIRS` (ni el `install_dirs`
    pasado al constructor) tiene lockfile: preguntarle a Windows dónde vive el ejecutable de
    `LeagueClientUx.exe` si está corriendo, y usar su carpeta contenedora (el lockfile vive junto
    al ejecutable). Cubre instalaciones en una unidad o ruta no estándar sin enumerar cada
    variante posible.

    Vía PowerShell (`Get-CimInstance`, no WMIC — deprecado en builds recientes de Windows), mismo
    patrón de "subprocess con lista de args, sin shell" que `jarvis.tools.close_app`/
    `jarvis.tools.open_app`. Nunca lanza: cualquier fallo (PowerShell no disponible, timeout,
    proceso no encontrado) devuelve `None`, tratado por el llamador igual que "League no está
    corriendo".

    Bug real, en vivo: `LCUAutoAcceptMonitor._run` llama a esto cada `WAITING_FOR_CLIENT_POLL_
    SECONDS` (5s) mientras League no esté corriendo — sin `creationflags=CREATE_NO_WINDOW`,
    cada uno de esos `powershell.exe` abría su propia ventana de consola visible (JARVIS corre
    vía `pythonw.exe`, sin consola propia a la que Windows pueda adjuntar el subproceso, así que
    crea una nueva) — el usuario lo vio como "se abren terminales de la nada" cada pocos
    segundos, jugando, sin haber pedido nada.
    """
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "(Get-CimInstance Win32_Process -Filter "
                    f"\"Name='{LEAGUE_CLIENT_PROCESS_NAME}'\" -ErrorAction "
                    "SilentlyContinue | Select-Object -First 1).ExecutablePath"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=PROCESS_LOOKUP_TIMEOUT_SECONDS,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    if not output:
        return None
    exe_path = Path(output.splitlines()[0].strip())
    if not exe_path.name:
        return None
    return exe_path.parent


def _find_lockfile_path(install_dirs: list[Path]) -> Path | None:
    """Buscar el lockfile en `install_dirs`, en orden; si ninguno lo tiene, intentar descubrir el
    directorio real vía el proceso en ejecución (`_discover_install_dir_from_running_process`).
    Devuelve `None` si no se encuentra en ningún lado — el estado normal y esperado cuando League
    no está corriendo, no un error.
    """
    for install_dir in install_dirs:
        candidate = install_dir / LOCKFILE_NAME
        if candidate.is_file():
            return candidate
    discovered_dir = _discover_install_dir_from_running_process()
    if discovered_dir is not None:
        candidate = discovered_dir / LOCKFILE_NAME
        if candidate.is_file():
            return candidate
    return None


def _build_client(credentials: LCUCredentials) -> httpx.Client:
    """Cliente HTTP contra la LCU API de este proceso puntual de League — ver docstring del
    módulo para por qué el host está hardcodeado a `127.0.0.1` y por qué `verify=False` es seguro
    acotado a este cliente.
    """
    return httpx.Client(
        base_url=f"{credentials.protocol}://127.0.0.1:{credentials.port}",
        auth=(LCU_USERNAME, credentials.password),
        verify=False,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def connect_to_lcu(*, install_dirs: list[Path] | None = None) -> httpx.Client | None:
    """Conectar a la LCU API del League Client actualmente en ejecución, si hay uno — punto de
    entrada público que reutilizan los tools de League invocados por voz
    (`jarvis.tools.lol_runes`, `jarvis.tools.lol_summoner_spells`,
    `jarvis.tools.lol_champion_select`), que a diferencia de `LCUAutoAcceptMonitor` no son un
    servicio de fondo sino un `Tool` (`jarvis.tools.base`) invocado puntualmente cuando el usuario
    lo pide por voz — no tiene sentido que reimplementen el descubrimiento de lockfile ni la
    construcción del cliente HTTP, así que llaman a esta función en vez de duplicar
    `_find_lockfile_path`/`_read_lockfile`/`_build_client`.

    Devuelve `None` (nunca lanza) si League no está corriendo — sin lockfile en ningún directorio
    de `install_dirs` (default `DEFAULT_INSTALL_DIRS`, mismo fallback vía proceso en ejecución que
    `_find_lockfile_path`) — o si el lockfile encontrado no se pudo leer/parsear (carrera con el
    cliente cerrándose, igual que interpreta `LCUAutoAcceptMonitor._run`). Cada tool llamador
    traduce ese `None` a su propio mensaje ("League no está corriendo").

    El caller es responsable de cerrar el cliente devuelto (`with connect_to_lcu() as client:` o
    `client.close()` en un `finally`) — mismo contrato que `_build_client`.
    """
    resolved_install_dirs = (
        install_dirs if install_dirs is not None else DEFAULT_INSTALL_DIRS
    )
    lockfile_path = _find_lockfile_path(resolved_install_dirs)
    if lockfile_path is None:
        return None
    credentials = _read_lockfile(lockfile_path)
    if credentials is None:
        return None
    return _build_client(credentials)


class LCUAutoAcceptMonitor:
    """Acepta automáticamente las colas de matchmaking (ready-check) de League of Legends,
    corriendo en un thread de fondo mientras JARVIS está activo — mismo lifecycle que
    `SystemAudioMonitor` (`start()`/`stop()`, thread daemon, idempotente, ver
    `jarvis.audio.loopback`). Se arranca una sola vez por corrida de `jarvis.audio.pipeline.run()`
    y corre continuamente sin necesidad de ningún comando de voz — no hay nada que "activarlo",
    igual que un companion app tipo Blitz.
    """

    def __init__(
        self,
        *,
        install_dirs: list[Path] | None = None,
        gameflow_poll_seconds: float = GAMEFLOW_POLL_SECONDS,
        waiting_poll_seconds: float = WAITING_FOR_CLIENT_POLL_SECONDS,
    ) -> None:
        self._install_dirs = (
            install_dirs if install_dirs is not None else DEFAULT_INSTALL_DIRS
        )
        self._gameflow_poll_seconds = gameflow_poll_seconds
        self._waiting_poll_seconds = waiting_poll_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Arrancar el thread de fondo. Idempotente: no hace nada si ya está corriendo."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Señalizar el fin del thread de fondo y esperar a que cierre. Idempotente."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        logged_waiting = False
        while not self._stop_event.is_set():
            try:
                lockfile_path = _find_lockfile_path(self._install_dirs)
                if lockfile_path is None:
                    if not logged_waiting:
                        logger.info(
                            "League Client no está corriendo (lockfile no encontrado) — "
                            "auto-accept en espera."
                        )
                        logged_waiting = True
                    self._stop_event.wait(self._waiting_poll_seconds)
                    continue

                credentials = _read_lockfile(lockfile_path)
                if credentials is None:
                    self._stop_event.wait(self._waiting_poll_seconds)
                    continue

                logged_waiting = False
                logger.info("League Client detectado — auto-accept activo.")
                with _build_client(credentials) as client:
                    self._poll_gameflow(client)
                logger.info(
                    "Se perdió la conexión con League Client — volviendo a esperar."
                )
            except Exception as exc:  # noqa: BLE001 — última línea de defensa del thread: un
                # fallo inesperado en un ciclo puntual (parseo, red, lo que sea no previsto por
                # las excepciones específicas de más arriba) no puede tumbar el thread de fondo
                # entero. Mismo criterio que el `except Exception` del loop principal de turnos en
                # `jarvis.audio.pipeline.run()`.
                logger.warning(
                    "Error inesperado en el monitor de auto-accept de League: %r", exc
                )
            self._stop_event.wait(self._waiting_poll_seconds)

    def _poll_gameflow(self, client: httpx.Client) -> None:
        """Poll de `gameflow-phase` mientras la conexión siga viva; acepta un ready-check una sola
        vez por aparición — `ready_check_handled` se rearma apenas la fase deja de ser
        `ReadyCheck`, para poder aceptar el próximo ready-check más adelante en la misma partida
        (recola, remake, etc.) sin volver a intentarlo en cada poll mientras el actual sigue
        activo. Vuelve (sin lanzar) apenas la conexión falla — el llamador (`_run`) interpreta eso
        como "el cliente se cerró/dejó de responder" y vuelve al estado de espera.
        """
        ready_check_handled = False
        while not self._stop_event.is_set():
            try:
                response = client.get(GAMEFLOW_PHASE_ENDPOINT)
            except httpx.HTTPError as exc:
                logger.info("Conexión con League Client interrumpida (%r).", exc)
                return
            if response.status_code != httpx.codes.OK:
                self._stop_event.wait(self._gameflow_poll_seconds)
                continue
            try:
                phase = response.json()
            except ValueError:
                self._stop_event.wait(self._gameflow_poll_seconds)
                continue

            if phase == READY_CHECK_PHASE:
                if not ready_check_handled:
                    ready_check_handled = True
                    self._accept_ready_check(client)
            else:
                ready_check_handled = False
            self._stop_event.wait(self._gameflow_poll_seconds)

    def _accept_ready_check(self, client: httpx.Client) -> None:
        try:
            response = client.post(READY_CHECK_ACCEPT_ENDPOINT)
        except httpx.HTTPError as exc:
            logger.warning("No pude aceptar la cola automáticamente (%r).", exc)
            return
        if response.status_code >= httpx.codes.BAD_REQUEST:
            logger.warning(
                "El accept de ready-check devolvió %s.", response.status_code
            )
            return
        logger.info("Cola de League of Legends aceptada automáticamente.")
