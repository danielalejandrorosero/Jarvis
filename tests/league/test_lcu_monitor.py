"""Tests para `jarvis.league.lcu_monitor.LCUAutoAcceptMonitor`.

No tocan un League Client real ni la red real: `_discover_install_dir_from_running_process`
(único punto que invocaría un subprocess/PowerShell real) se monkeypatchea siempre a `None`, y
`_build_client` se reemplaza por un fake `httpx.Client` controlado por el test — mismo enfoque que
`tests/audio/test_loopback.py` usa para `sd.InputStream` (stub in-proceso, sin I/O real, CI-safe).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Self

import httpx
import pytest

from jarvis.league import lcu_monitor
from jarvis.league.lcu_monitor import LCUAutoAcceptMonitor, LCUCredentials

POLL_TIMEOUT_SECONDS = 2.0
POLL_INTERVAL_SECONDS = 0.01

_FAST_GAMEFLOW_POLL = 0.01
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


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_body: Any = None) -> None:
        self.status_code = status_code
        self._json_body = json_body

    def json(self) -> Any:
        return self._json_body


class _FakeLCUClient:
    """Stub de `httpx.Client`: sirve fases de gameflow de una lista, en orden (repite la última
    indefinidamente), y registra cuántas veces se llamó a accept."""

    def __init__(
        self, phases: list[str], *, get_error: Exception | None = None
    ) -> None:
        self._phases = phases
        self._index = 0
        self._get_error = get_error
        self.accept_calls = 0
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.closed = True
        return False

    def get(self, _path: str) -> _FakeResponse:
        if self._get_error is not None:
            raise self._get_error
        if self._index < len(self._phases):
            phase = self._phases[self._index]
            self._index += 1
        elif self._phases:
            phase = self._phases[-1]
        else:
            phase = "None"
        return _FakeResponse(status_code=200, json_body=phase)

    def post(self, _path: str) -> _FakeResponse:
        self.accept_calls += 1
        return _FakeResponse(status_code=200, json_body=None)


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, fake_client: _FakeLCUClient
) -> None:
    monkeypatch.setattr(lcu_monitor, "_build_client", lambda _credentials: fake_client)


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
    fake_client = _FakeLCUClient(phases=["None", "Lobby", "Matchmaking"])
    _install_fake_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path],
        gameflow_poll_seconds=_FAST_GAMEFLOW_POLL,
        waiting_poll_seconds=_FAST_WAITING_POLL,
    )

    monitor.start()
    try:
        assert _poll_until(lambda: fake_client._index >= 3)
    finally:
        monitor.stop()

    assert fake_client.accept_calls == 0


# --- ready-check detectado -> accept llamado exactamente una vez -------------------------------


def test_ready_check_detected_triggers_accept_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    # La fase se mantiene en "ReadyCheck" indefinidamente (repite el último elemento) — sin la
    # guarda de `ready_check_handled`, cada poll dispararía un accept nuevo.
    fake_client = _FakeLCUClient(phases=["Lobby", "ReadyCheck"])
    _install_fake_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path],
        gameflow_poll_seconds=_FAST_GAMEFLOW_POLL,
        waiting_poll_seconds=_FAST_WAITING_POLL,
    )

    monitor.start()
    try:
        assert _poll_until(lambda: fake_client.accept_calls >= 1)
        # Varios polls más, la fase sigue siendo "ReadyCheck" (repetida) — el accept no debe
        # volver a dispararse.
        time.sleep(0.1)
    finally:
        monitor.stop()

    assert fake_client.accept_calls == 1


def test_ready_check_can_be_accepted_again_after_phase_changes_away_and_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """La guarda se rearma cuando la fase deja de ser ReadyCheck — un segundo ready-check
    (después de un remake, por ejemplo) sí se acepta."""
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    fake_client = _FakeLCUClient(
        phases=["ReadyCheck", "Matchmaking", "ReadyCheck", "ReadyCheck"]
    )
    _install_fake_client(monkeypatch, fake_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path],
        gameflow_poll_seconds=_FAST_GAMEFLOW_POLL,
        waiting_poll_seconds=_FAST_WAITING_POLL,
    )

    monitor.start()
    try:
        assert _poll_until(lambda: fake_client.accept_calls >= 2)
        time.sleep(0.1)
    finally:
        monitor.stop()

    assert fake_client.accept_calls == 2


# --- error de conexión mid-poll -> degrada, no crashea, se recupera ----------------------------


def test_connection_error_mid_poll_degrades_gracefully_and_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El primer cliente construido falla de entrada (conexión perdida); `_run` vuelve al estado
    de espera y, en la próxima vuelta, `_build_client` devuelve un cliente sano — el thread nunca
    muere y termina aceptando el ready-check una vez reconectado."""
    _disable_process_fallback(monkeypatch)
    _write_lockfile(tmp_path)
    failing_client = _FakeLCUClient(
        phases=[], get_error=httpx.ConnectError("conexión perdida")
    )
    healthy_client = _FakeLCUClient(phases=["ReadyCheck"])
    clients = iter([failing_client, healthy_client])
    monkeypatch.setattr(
        lcu_monitor, "_build_client", lambda _credentials: next(clients)
    )
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[tmp_path],
        gameflow_poll_seconds=_FAST_GAMEFLOW_POLL,
        waiting_poll_seconds=_FAST_WAITING_POLL,
    )

    monitor.start()
    try:
        assert _poll_until(lambda: healthy_client.accept_calls >= 1)
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
    finally:
        monitor.stop()

    assert monitor._thread is None
    assert healthy_client.accept_calls == 1


def test_lockfile_disappearing_mid_session_goes_back_to_waiting_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """El cliente se cierra: el lockfile deja de existir y la conexión activa empieza a fallar —
    el monitor vuelve a buscar el lockfile (estado "esperando") en vez de crashear."""
    _disable_process_fallback(monkeypatch)
    install_dir = tmp_path / "league"
    _write_lockfile(install_dir)
    failing_client = _FakeLCUClient(
        phases=[], get_error=httpx.ConnectError("cliente cerrado")
    )
    _install_fake_client(monkeypatch, failing_client)
    monitor = LCUAutoAcceptMonitor(
        install_dirs=[install_dir],
        gameflow_poll_seconds=_FAST_GAMEFLOW_POLL,
        waiting_poll_seconds=_FAST_WAITING_POLL,
    )

    monitor.start()
    try:
        assert _poll_until(lambda: failing_client.closed)
        (install_dir / "lockfile").unlink()
        # Sigue vivo esperando a que el lockfile reaparezca, sin excepciones no manejadas.
        time.sleep(0.1)
        assert monitor._thread is not None
        assert monitor._thread.is_alive()
    finally:
        monitor.stop()

    assert monitor._thread is None


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
