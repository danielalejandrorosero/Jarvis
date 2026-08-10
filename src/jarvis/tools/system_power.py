"""Tool para apagar, reiniciar o suspender la computadora completa (ADR-0005).

Un solo tool con un parámetro `action` (`'shutdown'` / `'restart'` / `'sleep'`), no tres tools
separados — mismo patrón que `MediaControlTool` (`media_control.py`: una acción entre varias
opciones excluyentes de la misma familia, un solo nombre que el LLM tiene que aprender a invocar).

Mecanismo, por acción:

- `shutdown`/`restart`: `subprocess.run(["shutdown", "/s"|"/r", "/t", "15"], ...)` — el binario
  `shutdown.exe` que ya viene con Windows, sin dependencia nueva. `/t 15` (no `0`, ver
  `_SHUTDOWN_DELAY_SECONDS` para el porqué de ese número concreto): da margen para que TTS termine
  de decir la frase de confirmación y, sobre todo, deja una ventana real antes de que el apagado
  ocurra de verdad — pedido explícito del usuario de tener "mucho cuidado" con este tool.
  `shutdown.exe` en sí ya soporta cancelar un apagado programado dentro de ese margen
  (`shutdown /a`); `jarvis.tools.cancel_system_power.CancelSystemPowerTool` expone justamente ese
  cancel por voz.
- `sleep`: `ctypes.windll.powrprof.SetSuspendState(False, True, False)` — la función de bajo nivel
  de `PowrProf.dll` que expone Windows para pedir una transición de energía, vía `ctypes` (stdlib,
  sin dependencia nueva). Se prefirió sobre lanzar `rundll32 powrprof.dll,SetSuspendState` como
  subproceso porque es exactamente la misma llamada subyacente sin pagar el costo ni la superficie
  extra (ventana de consola, código de salida a interpretar) de lanzar un proceso nuevo solo para
  invocar una única función de una DLL que `ctypes` ya puede llamar directamente en el propio
  proceso — mismo criterio que `media_control.py` aplicó al elegir `keybd_event` por sobre
  alternativas más pesadas para un caso simple. Ver `_suspend_system` para el detalle de cada
  argumento posicional de `SetSuspendState`.

Cualquier `subprocess.run` de este módulo lleva `creationflags=subprocess.CREATE_NO_WINDOW`: sin
eso, Windows abre una ventana de consola visible para `shutdown.exe` cada vez — JARVIS corre vía
`pythonw.exe`, sin consola propia a la que adjuntarlo (mismo bug real encontrado en vivo, misma
sesión, en `jarvis.league.lcu_monitor`, `jarvis.tools.close_app` y `jarvis.tools.system_info`).

Ambos mecanismos viven en funciones de nivel de módulo, cada una monkeypatcheable por separado
(`run_power_command` para shutdown/restart/cancel, `_suspend_system` para sleep) — mismo patrón que
`_press_media_key` en `media_control.py` o `_terminate_process` en `close_app.py`: la suite de
tests nunca apaga, reinicia ni suspende la máquina real donde corre.

## Clasificación de riesgo: CONFIRM (no SAFE, no DANGEROUS)

`.claude/rules/security.md` da "terminar procesos" como ejemplo explícito de CONFIRM
("reversible o de impacto acotado, pero muta estado") — apagar/reiniciar/suspender la propia
computadora es, en ese mismo espíritu, una versión de alcance más amplio de esa misma familia de
acción (en vez de un proceso puntual, la sesión completa), no una categoría nueva a inventar:

- **Reversible a nivel de sistema, no a nivel de "trabajo no guardado"**: la máquina en sí se
  recupera prendiéndola de nuevo (botón de power) o despertándola (tecla/mouse), sin corrupción de
  disco ni de ningún estado persistente de Windows — pero cualquier cambio sin guardar en una
  aplicación abierta al momento de apagar/reiniciar SÍ se pierde (por eso la `description` del
  tool, que el usuario escucha antes de confirmar, lo advierte explícitamente: la confirmación
  informada es lo que hace aceptable este riesgo, no la ausencia del riesgo en sí). Reiniciar es,
  por diseño, un ciclo completo de apagado+encendido: el mismo argumento aplica. Mismo trade-off,
  mismo alcance, que ya acepta `close_app` bajo CONFIRM (cerrar una app también puede perder
  trabajo no guardado en ella).
- **Impacto acotado**: afecta únicamente esta máquina local, nunca infraestructura de terceros ni
  datos de otros usuarios — comparable en alcance a `close_app` (que ya afecta "todas las ventanas
  abiertas de una app", CONFIRM), solo que acá el universo afectado es "todo el sistema" en vez de
  "un proceso", pero sigue siendo una sola máquina bajo control directo del usuario que la pidió.
- **No cumple la vara de DANGEROUS**: `.claude/rules/security.md` reserva DANGEROUS para
  "eliminación masiva, modificación crítica del registro, comandos irreversibles, operaciones
  destructivas, cambios administrativos de alto impacto" — nada de esto aplica acá: no se borra
  nada del disco, no se corrompe el registro, no hay ningún efecto que sobreviva a que la máquina
  vuelva a prenderse.
- **DANGEROUS dejaría este comando inutilizable por voz, sin ningún beneficio de seguridad real**:
  `PolicyEngine.authorize_and_execute` (`jarvis.security.policy`) no tiene ningún código path que
  llame a `tool.execute()` para un tool DANGEROUS — ver su docstring: "no hay ningún código path
  que lo permita, por diseño". Clasificar este tool como DANGEROUS no lo haría "más seguro con más
  fricción", lo haría simplemente inalcanzable por voz para siempre, contradiciendo el pedido
  explícito del usuario de poder decir "Alexa, apagá la PC" y que, tras confirmar, efectivamente
  pase. CONFIRM ya exige una confirmación hablada explícita antes de ejecutar (mismo mecanismo que
  protege `close_app`/`lock_lol_champion`/`cancel_all_timers`) — la barrera correcta para una acción
  reversible y de alcance acotado, ni ausente (SAFE) ni prohibitiva sin motivo (DANGEROUS).
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import subprocess
from typing import Any, ClassVar

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

# Delay antes de que `shutdown.exe` ejecute de verdad, en segundos — no 0, y no 5 (valor original,
# revisado por `security-reviewer`: 5s no alcanzaba en la práctica). El margen tiene que sobrevivir
# al camino completo que un "cancelá" real recorre antes de que `cancel_system_power` llegue a
# correr, no solo a que termine el audio de confirmación:
#   - TTS reproduciendo "Listo, apagando la computadora." antes de que el usuario pueda reaccionar:
#     ~1.5-2.5s.
#   - El usuario reaccionando y diciendo "cancelá" en la ventana de seguimiento sin wake word
#     (`FOLLOW_UP_WINDOW_SECONDS` en `jarvis.audio.pipeline`) más el silencio sostenido que corta
#     esa grabación (`TRAILING_SILENCE_SECONDS` en `jarvis.audio.vad`, 1.2s): otro ~1.5-2s entre
#     empezar a hablar y que la grabación se dé por terminada.
#   - El round-trip real a `dispatch_turn` (`jarvis.audio.pipeline`) para que el LLM resuelva la
#     palabra dicha como el tool-call `cancel_system_power` y `PolicyEngine` lo autorice (SAFE, sin
#     una segunda confirmación) antes de que `run_power_command` corra `shutdown /a`: ~2-3s típico
#     de latencia de red al LLM.
# Sumado, el camino completo hasta que `shutdown /a` efectivamente corre ronda los 5-8s en el caso
# feliz — con 5s de margen, ese camino casi nunca cierra a tiempo (el "safety net" documentado
# antes era, en la práctica, ilusorio). 15s deja margen real incluso si alguna de esas etapas tarda
# más de lo típico, sin que el apagado se sienta artificialmente demorado para el caso normal en el
# que nadie cancela nada.
_SHUTDOWN_DELAY_SECONDS = "15"

_ACTION_TO_COMMAND: dict[str, list[str]] = {
    "shutdown": ["shutdown", "/s", "/t", _SHUTDOWN_DELAY_SECONDS],
    "restart": ["shutdown", "/r", "/t", _SHUTDOWN_DELAY_SECONDS],
}

# Frase a usar en el mensaje de fracaso ("No pude <frase>: <detalle>.") y también la fuente de
# verdad de qué acciones son válidas (las tres claves cubren 'shutdown'/'restart'/'sleep').
_FAILURE_LABELS: dict[str, str] = {
    "shutdown": "apagar la computadora",
    "restart": "reiniciar la computadora",
    "sleep": "suspender la computadora",
}

_SUCCESS_LABELS: dict[str, str] = {
    "shutdown": "apagando la computadora",
    "restart": "reiniciando la computadora",
    "sleep": "suspendiendo la computadora",
}

# Frase natural para el prompt de confirmación por voz de `PolicyEngine` (ver `Tool.describe()`
# en `base.py`: los tools CONFIRM/DANGEROUS deberían sobreescribir el default genérico).
_DESCRIBE_PHRASES: dict[str, str] = {
    "shutdown": "apagar la computadora",
    "restart": "reiniciar la computadora",
    "sleep": "suspender la computadora",
}

_INVALID_ACTION_MESSAGE = "No se especificó una acción de energía válida (esperaba 'shutdown', 'restart' o 'sleep')."


def run_power_command(command: list[str]) -> tuple[bool, str]:
    """Ejecuta `command` (`shutdown.exe /s`, `/r` o `/a`) y devuelve `(éxito, detalle)` según el
    código de salida real, nunca una asunción optimista — mismo patrón que `_terminate_process` en
    `close_app.py`. Nivel de módulo, monkeypatcheable (ver docstring del módulo): ningún test de
    este tool invoca `shutdown.exe` de verdad.

    Sin prefijo `_` a propósito, a diferencia del resto de los helpers de este módulo: es un
    contrato compartido real con `jarvis.tools.cancel_system_power.CancelSystemPowerTool` (que la
    importa para su `shutdown /a`), no un detalle interno de un solo tool — este repo normalmente
    prefiere *duplicar* funciones privadas de una línea entre módulos en vez de importarlas (ver
    `_escape_untrusted` en `jarvis.tools.search`/`jarvis.audio.pipeline`), pero acá el cuerpo tiene
    lógica real de manejo de errores (excepción vs. código de salida vs. stderr/stdout) que
    duplicar dejaría con riesgo real de divergir entre los dos tools de apagado con el tiempo.

    `detalle` es la cadena vacía en éxito (no hay nada que reportar); en fracaso, es el mensaje de
    error (excepción, o stderr/stdout de `shutdown.exe`) para que `execute()` pueda informarlo.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            # Ver docstring del módulo — evita la ventana de consola visible bajo `pythonw.exe`.
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError as exc:
        return False, exc.__class__.__name__

    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip() or result.stdout.strip() or "error desconocido"


def _suspend_system() -> bool:
    """Suspende (sleep) el sistema vía `ctypes.windll.powrprof.SetSuspendState` — nivel de módulo,
    monkeypatcheable (ver docstring del módulo), así ningún test suspende la máquina real.

    Firma Win32 `SetSuspendState(bHibernate, bForce, bWakeupEventsDisabled)`:

    - `bHibernate=False`: pide suspensión estándar a RAM (retoma casi instantáneo), no hibernación
      a disco (mucho más lenta, y deshabilitada por default en muchas instalaciones de escritorio/
      laptop de Windows 11) — "suspender"/"dormir" en el pedido del usuario se corresponde con
      sleep, no con hibernate.
    - `bForce=True`: no depende de que cada aplicación abierta apruebe la transición antes de
      suspender. Análogo solo parcial a `taskkill /F` en `close_app.py` (`_terminate_process`), no
      equivalente: `taskkill /F` fuerza el cierre de UN proceso que el usuario nombró explícitamente,
      mientras que `bForce=True` acá pasa por encima del veto de TODAS las aplicaciones abiertas del
      sistema a la vez, incluidas las que el usuario nunca mencionó y que podrían estar a mitad de
      una operación sensible (grabando, haciendo backup, quemando un disco). Se acepta igual porque,
      a diferencia de cerrar un proceso, suspender no destruye nada: las apps no se cierran, siguen
      corriendo en RAM y retoman exactamente donde estaban al despertar la máquina — el usuario ya
      confirmó por voz que quiere suspender, y la operación en sí es reversible sin pérdida de
      trabajo (a diferencia de `shutdown`/`restart`), no que cada app individual tenga la misma
      oportunidad de objetar que tendría si el usuario suspendiera la máquina manualmente.
    - `bWakeupEventsDisabled=False`: no deshabilita los eventos de wake normales (mover el mouse,
      tocar una tecla) — suspender no debería dejar la máquina en un estado donde despertarla
      requiera el botón de power físico.

    Esta llamada BLOQUEA hasta que el sistema efectivamente se suspende y después se despierta de
    nuevo (no retorna "ya programé la suspensión", retorna "ya volví de suspenderme") — por eso
    `execute()` la corre vía `asyncio.to_thread`, igual que cualquier otra llamada bloqueante de
    este repo (`.claude/rules/python.md`), y no de forma directa en el loop de asyncio.

    Devuelve el resultado de la llamada como `bool` — Win32 devuelve un `BOOL` (entero) no-cero en
    éxito, cero en fracaso.
    """
    result: int = ctypes.windll.powrprof.SetSuspendState(False, True, False)
    return bool(result)


class SystemPowerTool(Tool):
    """Apaga, reinicia o suspende la computadora. CONFIRM: requiere confirmación explícita por voz
    antes de ejecutarse (ver docstring del módulo, sección de clasificación de riesgo)."""

    name = "system_power"
    description = (
        "Apaga, reinicia o suspende (duerme) la computadora completa. Usar cuando el usuario "
        "pide apagar, reiniciar, o suspender/dormir la PC. ADVERTENCIA: apagar y reiniciar "
        "cierran TODAS las aplicaciones abiertas de inmediato, sin oportunidad de guardar "
        "cambios pendientes — reiniciar además interrumpe cualquier sesión de Windows en curso. "
        "Suspender no cierra nada (la máquina retoma exactamente donde estaba al despertarla), "
        "pero la deja sin responder hasta que alguien la despierte. Requiere confirmación "
        "explícita del usuario antes de ejecutarse: no es sorpresa, es el comportamiento "
        "esperado. El apagado/reinicio real ocurre unos segundos después de confirmar, no "
        "instantáneo — si el usuario se arrepiente en ese margen, usar cancel_system_power."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["shutdown", "restart", "sleep"],
                "description": (
                    "Acción de energía a realizar: 'shutdown' (apagar la computadora), "
                    "'restart' (reiniciarla) o 'sleep' (suspenderla/dormirla)."
                ),
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    }
    risk = RiskLevel.CONFIRM

    def describe(self, kwargs: dict[str, Any]) -> str:
        action = kwargs.get("action")
        if isinstance(action, str) and action in _DESCRIBE_PHRASES:
            return _DESCRIBE_PHRASES[action]
        return super().describe(kwargs)

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action")
        if not isinstance(action, str) or action not in _FAILURE_LABELS:
            return _INVALID_ACTION_MESSAGE

        if action == "sleep":
            try:
                success = await asyncio.to_thread(_suspend_system)
            except OSError as exc:
                return f"No pude {_FAILURE_LABELS[action]} ({exc.__class__.__name__})."
            if not success:
                return f"No pude {_FAILURE_LABELS[action]}."
            return f"Listo, {_SUCCESS_LABELS[action]}."

        success, detail = await asyncio.to_thread(
            run_power_command, _ACTION_TO_COMMAND[action]
        )
        if not success:
            return f"No pude {_FAILURE_LABELS[action]}: {detail}."
        return f"Listo, {_SUCCESS_LABELS[action]}."
