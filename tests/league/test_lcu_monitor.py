"""Tests para `jarvis.league.lcu_monitor.LCUAutoAcceptMonitor`.

No tocan un League Client real ni la red real: `_discover_install_dir_from_running_process`
(único punto que invocaría un subprocess/PowerShell real) se monkeypatchea siempre a `None`,
`websockets.connect` se reemplaza por un fake que entrega mensajes WAMP pre-armados (opcode 8,
`OnJsonApiEvent`, ver `_extract_gameflow_phase`), y `_build_async_client` se reemplaza por un fake
`httpx.AsyncClient` controlado por el test — mismo enfoque que `tests/audio/test_loopback.py` usa
para `sd.InputStream` (stub in-proceso, sin I/O real, CI-safe).

Detección de ready-check vía eventos push (no polling REST) desde la reescritura que corrige el
bug real reportado por el usuario (segunda cola tras un rechazo no se aceptaba) — ver docstring de
`jarvis.league.lcu_monitor` para el detalle completo. `test_ready_check_can_be_accepted_again_
after_phase_changes_away_and_back` es el test que cubre exactamente ese escenario.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import Any, Self

import pytest

from jarvis.league import lcu_monitor
from jarvis.league.lcu_monitor import LCUAutoAcceptMonitor, LCUCredentials

POLL_TIMEOUT_SECONDS = 2.0
POLL_INTERVAL_SECONDS = 0.01

_FAST_WAITING_POLL = 0.01


def _poll_until(predicate: object, *, timeout: float = POLL_TIMEOUT_SECONDS) -> bool:
    """Esperar hasta que `predicate()` sea verdadero o pasen `timeout` segundos — evita un
    `sleep` fijo (flaky por definición) para sincronizar con el thread de fondo."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return bool(predicate())  # type: ignore[operator]


def _write_lockfile(
    directory: Path, *, port: int = 12345, password: str = "secretpw"
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lockfile = directory / "lockfile"
    lockfile.write_text(f"LeagueClientUx:1234:{port}:{password}:https")
    return lockfile


def _phase_event(phase: str) -> str:
    """Mensaje WAMP real (opcode 8, `OnJsonApiEvent`) que emitiría la LCU para una transición de
    `gameflow-phase` — mismo shape confirmado en el docstring del módulo contra la documentación
    real (`[8,"OnJsonApiEvent",{"data":...,"eventType":"Update","uri":"..."}]`)."""
    return (
        '[8,"OnJsonApiEvent",{"data":"'
        + phase
        + '","eventType":"Update","uri":"'
        + lcu_monitor.GAMEFLOW_PHASE_ENDPOINT
        + '"}]'
    )


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_body: Any = None) -> None:
        self.status_code = status_code
        self._json_body = json_body

    def json(self) -> Any:
        return self._json_body


class _FakeAsyncLCUClient:
    """Stub de `httpx.AsyncClient`: solo necesita responder al `POST` de accept y contar
    llamadas — a diferencia del cliente REST viejo, ya no sirve fases de gameflow (eso ahora
    llega por websocket, ver `_FakeWebSocket`)."""

    def __init__(self) -> None:
        self.accept_calls = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def post(self, _path: str) -> _FakeResponse:
        self.accept_calls += 1
        return _FakeResponse(status_code=200, json_body=None)


class _FakeWebSocket:
    """Stub del websocket de eventos de la LCU: entrega `messages` en orden vía `recv()`, y una
    vez agotados simula que la conexión se cerró (`ConnectionError`, subclase de `OSError` —
    mismo tipo que `_watch_gameflow_events` atrapa en el borde exterior). Registra lo enviado
    (`sent`, para poder confirmar la suscripción WAMP) y si el `async with` ya se cerró
    (`closed`)."""

    def __init__(self, messages: list[str]) -> None:
        self._messages = list(messages)
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        self.closed = True
        return False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        raise ConnectionError("fake: conexión de websocket cerrada")


class _FakeHangingWebSocket(_FakeWebSocket):
    """Variante que, agotados los mensajes pre-armados, se queda "colgada" esperando el próximo
    evento en vez de reventar (`asyncio.Event().wait()`, nunca se completa) — igual que un
    websocket real todavía abierto en medio de `ChampSelect`/`InProgress`, donde pueden pasar
    varios minutos sin ningún evento de `gameflow-phase`. Ejercita que `stop()` puede interrumpir
    un `ws.recv()` en curso (`_watch_gameflow_events` corre `stop_future` en carrera contra cada
    `recv()`) en vez de quedar bloqueado hasta que llegue un evento que puede no llegar nunca."""

    async def recv(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


def _install_fake_websocket(
    monkeypatch: pytest.MonkeyPatch, *fakes: _FakeWebSocket | Exception
) -> None:
    """Reemplazar `websockets.connect` por un fake que, en cada llamada (cada intento de
    conexión/reconexión de `_watch_gameflow_events`), consume el próximo elemento de `fakes` — un
    `_FakeWebSocket` se devuelve como el objeto de conexión, una `Exception` se lanza directo
    (simula que la conexión en sí falló, antes de llegar a `async with`). Si `fakes` se agota,
    sigue devolviendo el último elemento indefinidamente (mismo criterio que `_FakeLCUClient`
    servía la última fase repetida) — evita que un test necesite anticipar cuántas reconexiones
    exactas va a intentar el thread de fondo antes de que el test llame a `stop()`.
    """
    remaining = list(fakes)

    def _fake_connect(*_args: Any, **_kwargs: Any) -> _FakeWebSocket:
        next_fake = remaining.pop(0) if remaining else fakes[-1]
        if isinstance(next_fake, Exception):
            raise next_fake
        return next_fake

    monkeypatch.setattr(lcu_monitor.websockets, "connect", _fake_connect)


def _install_fake_async_client(
    monkeypatch: pytest.MonkeyPatch, fake_client: _FakeAsyncLCUClient
) -> None:
    monkeypatch.setattr(
        lcu_monitor, "_build_async_client", lambda _credentials: fake_client
    )


def _disable_process_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nunca invocar PowerShell real desde un test — ver docstring del módulo."""
    monkeypatch.setattr(
        lcu_monitor, "_discover_install_dir_from_running_process", lambda: None
    )


# --- parseo de lockfile (lógica pura, sin thread) ----------------------------------------------


def test_parse_lockfile_extracts_port_password_protocol() -> None:
    credentials = lcu_monitor._parse_lockfile(
        "LeagueClientUx:12345:54321:abc123XYZ:https"
    )
    assert credentials == LCUCredentials(
        port=54321, password="abc123XYZ", protocol="https"
    )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not enough fields",
        "a:b:c:d",  # solo 4 campos
        "a:b:notaport:pw:https",  # puerto no numérico
        "a:b:54321::https",  # password vacío
        "a:b:54321:pw:",  # protocolo vacío
        "a:b:54321:pw:ftp",  # protocolo fuera de la allow-list
        # Protocolo malicioso/malformado: si se interpolara sin validar en
        # `_build_client` (`f"{protocol}://127.0.0.1:{port}"`), un parser de URL estándar
        # interpretaría el netloc resultante como `evil.com`, no `127.0.0.1` — ver
        # `_VALID_LOCKFILE_PROTOCOLS`. Tiene que rechazarse en el parseo, no llegar a
        # construirse un cliente HTTP con esto.
        "a:b:54321:pw:https://evil.com",
    ],
)
def test_parse_lockfile_returns_none_for_malformed_content(content: str) -> None:
    assert lcu_monitor._parse_lockfile(content) is None


def test_read_lockfile_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert lcu_monitor._read_lockfile(tmp_path / "does-not-exist") is None


# --- parseo de eventos del websocket (lógica pura, sin thread) ----------------------------------


def test_extract_gameflow_phase_parses_a_real_wamp_event() -> None:
    assert (
        lcu_monitor._extract_gameflow_phase(_phase_event("ReadyCheck")) == "ReadyCheck"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "[]",
        "[8]",
        '[8,"OnJsonApiEvent","not-a-dict"]',
        # Opcode que no es de evento (5 = subscribe, eco del propio mensaje enviado).
        '[5,"OnJsonApiEvent"]',
        # URI de otro evento — no es la fase de gameflow, se ignora.
        '[8,"OnJsonApiEvent",{"data":"x","eventType":"Update","uri":"/lol-ranked/v1/notifications"}]',
        # `data` no es un string.
        '[8,"OnJsonApiEvent",{"data":123,"eventType":"Update","uri":"/lol-gameflow/v1/gameflow-phase"}]',
    ],
)
def test_extract_gameflow_phase_returns_none_for_anything_else(raw: str) -> None:
    assert lcu_monitor._extract_gameflow_phase(raw) is None


def test_extract_gameflow_phase_returns_none_for_binary_frames() -> None:
    assert lcu_monitor._extract_gameflow_phase(b"\x00\x01") is None


# --- ventana de consola oculta (bug real, en vivo) ----------------------------------------------
# `_discover_install_dir_from_running_process` es el único punto de este módulo que de verdad
# invoca un subprocess (PowerShell) — llamado cada `WAITING_FOR_CLIENT_POLL_SECONDS` mientras
# League no esté corriendo, así que sin `creationflags=CREATE_NO_WINDOW` cada llamada abría una
# ventana de consola visible (JARVIS corre vía `pythonw.exe`, sin consola propia): el usuario lo
# vio en vivo como "se abren terminales de la nada" cada pocos segundos, jugando.


def test_discover_install_dir_from_running_process_hides_console_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _FakeCompletedProcess:
        returncode = 1  # no importa el resultado acá, solo los kwargs de la llamada
        stdout = ""

    def _fake_run(_args: list[str], **kwargs: Any) -> _FakeCompletedProcess:
        captured_kwargs.update(kwargs)
        return _FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    lcu_monitor._discover_install_dir_from_running_process()

    assert captured_kwargs.get("creationflags") == subprocess.CREATE_NO_WINDOW


# --- descubrimiento del lockfile -----------------------------------------------------------------


def test_find_lockfile_path_returns_none_when_no_install_dir_has_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_process_fallback(monkeypatch)
    assert lcu_monitor._find_lockfile_path([tmp_path / "nope"]) is None


def test_find_lockfile_path_finds_it_in_a_known_install_dir(tmp_path: Path) -> None:
    lockfile = _write_lockfile(tmp_path)
    assert lcu_monitor._find_lockfile_path([tmp_path]) == lockfile


def test_find_lockfile_path_falls_back_to_discovered_process_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ningún directorio conocido tiene lockfile, pero el proceso en ejecución revela uno
    distinto — el fallback lo encuentra ahí."""
    discovered_dir = tmp_path / "discovered"
    lockfile = _write_lockfile(discovered_dir)
    monkeypatch.setattr(
        lcu_monitor,
        "_discover_install_dir_from_running_process",
        lambda: discovered_dir,
    )

    result = lcu_monitor._find_lockfile_path([tmp_path / "not-here"])

    assert result == lockfile


# --- lifecycle: lockfile no encontrado -> queda idle sin crashear ------------------------------


def test_monitor_stays_idle_without_crashing_when_lockfile_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_process_fallback(monkeypatch)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path / "no-league-here"],
        waiting_poll_seconds=_FAST_WAITING_POLL,
    )

    monitor.start()
    try:
        # Le da tiempo a correr un par de ciclos de "esperando" sin encontrar nada.
        time.sleep(0.1)
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
    finally:
        monitor.stop()

    assert monitor._thread is None


# --- lockfile encontrado, sin ready-check activo -> ninguna acción -----------------------------


def test_no_action_taken_when_no_ready_check_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    fake_ws = _FakeWebSocket(
        messages=[
            _phase_event("None"),
            _phase_event("Lobby"),
            _phase_event("Matchmaking"),
        ]
    )
    _install_fake_websocket(monkeypatch, fake_ws)
    fake_client = _FakeAsyncLCUClient()
    _install_fake_async_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    try:
        assert _poll_until(lambda: not fake_ws._messages)
        time.sleep(0.05)
    finally:
        monitor.stop()

    assert fake_client.accept_calls == 0
    # Confirma que sí se suscribió a eventos (opcode 5, `OnJsonApiEvent`) — sin esto, "no se
    # aceptó nada" sería un falso positivo (el monitor podría no estar escuchando en absoluto).
    # Puede haber más de un envío: agotados los mensajes, `recv()` fuerza una reconexión (mismo
    # fake reutilizado, ver `_install_fake_websocket`), y cada reconexión vuelve a suscribirse.
    assert fake_ws.sent
    assert all(sent == '[5, "OnJsonApiEvent"]' for sent in fake_ws.sent)


# --- ready-check detectado -> accept llamado exactamente una vez -------------------------------


def test_ready_check_detected_triggers_accept_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    # Dos eventos seguidos de "ReadyCheck" sin ninguna fase distinta entre medio — sin la guarda
    # de `ready_check_handled`, el segundo evento dispararía un accept nuevo.
    fake_ws = _FakeWebSocket(
        messages=[
            _phase_event("Lobby"),
            _phase_event("ReadyCheck"),
            _phase_event("ReadyCheck"),
        ]
    )
    _install_fake_websocket(monkeypatch, fake_ws)
    fake_client = _FakeAsyncLCUClient()
    _install_fake_async_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    try:
        assert _poll_until(lambda: fake_client.accept_calls >= 1)
        time.sleep(0.05)
    finally:
        monitor.stop()

    assert fake_client.accept_calls == 1


def test_ready_check_can_be_accepted_again_after_phase_changes_away_and_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El escenario exacto reportado por el usuario: acepta la primera cola, alguien rechaza, y
    una segunda cola (nuevo ready-check) también tiene que aceptarse — la guarda se rearma
    apenas llega un evento de una fase distinta de ReadyCheck, sin depender de ningún timing real
    (acá el evento intermedio siempre está presente, a diferencia del polling viejo que podía
    perdérselo — ver docstring del módulo)."""
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    fake_ws = _FakeWebSocket(
        messages=[
            _phase_event("ReadyCheck"),
            _phase_event("Matchmaking"),  # alguien rechazó, vuelve a buscar partida
            _phase_event("ReadyCheck"),  # segunda cola: tiene que volver a aceptarse
        ]
    )
    _install_fake_websocket(monkeypatch, fake_ws)
    fake_client = _FakeAsyncLCUClient()
    _install_fake_async_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    try:
        assert _poll_until(lambda: fake_client.accept_calls >= 2)
        time.sleep(0.05)
    finally:
        monitor.stop()

    assert fake_client.accept_calls == 2


# --- error de conexión -> degrada, no crashea, se recupera --------------------------------------


def test_connection_error_mid_poll_degrades_gracefully_and_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El primer intento de conexión al websocket falla de entrada (conexión perdida); el thread
    vuelve al estado de espera y, en la próxima vuelta, la conexión sí prospera — el thread nunca
    muere y termina aceptando el ready-check una vez reconectado."""
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    healthy_ws = _FakeWebSocket(messages=[_phase_event("ReadyCheck")])
    _install_fake_websocket(
        monkeypatch, ConnectionError("conexión perdida"), healthy_ws
    )
    fake_client = _FakeAsyncLCUClient()
    _install_fake_async_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    try:
        assert _poll_until(lambda: fake_client.accept_calls >= 1)
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
    finally:
        monitor.stop()

    assert monitor._thread is None
    assert fake_client.accept_calls == 1


def test_lockfile_disappearing_mid_session_goes_back_to_waiting_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El cliente se cierra: el lockfile deja de existir y la conexión de websocket activa
    empieza a fallar — el monitor vuelve a buscar el lockfile (estado "esperando") en vez de
    crashear."""
    _disable_process_fallback(monkeypatch)
    install_dir = tmp_path / "league"
    _write_lockfile(install_dir)
    failing_ws = _FakeWebSocket(messages=[])  # falla en el primer recv()
    _install_fake_websocket(monkeypatch, failing_ws)
    fake_client = _FakeAsyncLCUClient()
    _install_fake_async_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[install_dir], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    try:
        assert _poll_until(lambda: failing_ws.closed)
        (install_dir / "lockfile").unlink()
        # Sigue vivo esperando a que el lockfile reaparezca, sin excepciones no manejadas.
        time.sleep(0.1)
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
    finally:
        monitor.stop()

    assert monitor._thread is None


# --- stop() responsivo contra un websocket colgado -----------------------------------------------


def test_stop_interrupts_a_hanging_websocket_wait_promptly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`stop()` tiene que devolver rápido aunque el websocket esté "colgado" esperando el próximo
    evento (el cliente puede pasar varios minutos en `ChampSelect`/`InProgress` sin emitir ningún
    evento de `gameflow-phase`) — ejercita que `_watch_gameflow_events` corre `stop_future` en
    carrera contra `ws.recv()`, en vez de que `stop()` quede bloqueado hasta que llegue un evento
    que puede no llegar nunca."""
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    hanging_ws = _FakeHangingWebSocket(messages=[])
    _install_fake_websocket(monkeypatch, hanging_ws)
    _install_fake_async_client(monkeypatch, _FakeAsyncLCUClient())
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    assert _poll_until(lambda: bool(hanging_ws.sent))  # ya conectado y suscripto

    stop_started = time.monotonic()
    monitor.stop()
    stop_elapsed = time.monotonic() - stop_started

    assert monitor._thread is None
    assert (
        stop_elapsed < 2.0
    )  # el timeout de `join()` en `stop()` — no debería ni acercarse


# --- lifecycle genérico (idempotencia, mismo contrato que SystemAudioMonitor) -------------------


def test_start_is_idempotent_does_not_spawn_a_second_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_process_fallback(monkeypatch)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    first_thread = monitor._thread
    monitor.start()

    assert monitor._thread is first_thread
    monitor.stop()


def test_stop_is_idempotent_and_joins_the_background_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_process_fallback(monkeypatch)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path], waiting_poll_seconds=_FAST_WAITING_POLL
    )

    monitor.start()
    monitor.stop()
    monitor.stop()  # segunda llamada no debe lanzar ni bloquear

    assert monitor._thread is None
