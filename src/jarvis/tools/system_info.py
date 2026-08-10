"""Tool de solo lectura que reporta el estado actual del sistema: uso de CPU, RAM, y — si hay una
GPU NVIDIA presente — uso/memoria/temperatura de GPU (ADR-0005).

Dependencia nueva: `psutil`. Stdlib no tiene una forma limpia y estable de leer CPU%/RAM en
Windows sin recurrir a WMI (`wmi`/`win32com`), que además de agregar su propia dependencia es
notoriamente verboso e inestable entre versiones de Windows para esta clase de consulta. `psutil`
es la librería estándar, madura y ampliamente usada para exactamente esto — mismo criterio y
mismo nivel de justificación que ya se aplicó a `httpx` en este repo (`.claude/rules/python.md`:
"sin dependencias nuevas sin justificar la necesidad — preferir stdlib cuando cubra el caso"; acá
stdlib no cubre el caso de forma razonable).

GPU: sin dependencia nueva. Se invoca `nvidia-smi` (`subprocess.run`, lista de args, sin
`shell=True`) — el binario que instala el propio driver de NVIDIA, ya presente en esta máquina
(RTX 3050). Si `nvidia-smi` no existe (máquina sin GPU NVIDIA, o no está en el PATH), el tool
degrada limpiamente a reportar solo CPU/RAM en vez de fallar — la sección de GPU es información
best-effort, no un requisito del tool.

SAFE bajo `.claude/rules/security.md`: puramente de lectura, no muta ningún estado del sistema —
"información del sistema" está en la lista explícita de ejemplos SAFE.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any, ClassVar

import psutil

from jarvis.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1024**3
_MIB_PER_GB = 1024

# Intervalo de muestreo de `psutil.cpu_percent` — sin esto (`interval=None`), la primera lectura
# de un proceso siempre devuelve 0.0 (no hay una medición previa contra la cual comparar). Medio
# segundo es un compromiso razonable entre una lectura representativa y no demorar la respuesta.
_CPU_SAMPLE_INTERVAL_SECONDS = 0.3

_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
_NVIDIA_SMI_COMMAND = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
]


def _cpu_usage_percent() -> float:
    """% de uso de CPU total del sistema, nivel de módulo y monkeypatcheable (mismo patrón que
    `_list_running_process_names` en `close_app.py`) para que los tests nunca midan CPU real."""
    return psutil.cpu_percent(interval=_CPU_SAMPLE_INTERVAL_SECONDS)


def _memory_stats() -> tuple[float, float, float]:
    """`(usado_gb, total_gb, porcentaje)` de RAM del sistema."""
    virtual_memory = psutil.virtual_memory()
    return (
        virtual_memory.used / _BYTES_PER_GB,
        virtual_memory.total / _BYTES_PER_GB,
        virtual_memory.percent,
    )


def _raw_gpu_stats() -> str | None:
    """Línea CSV cruda de `nvidia-smi` (`utilization.gpu, memory.used, memory.total,
    temperature.gpu`), o `None` si `nvidia-smi` no está disponible, tarda demasiado, o devuelve
    un código de error — nunca lanza. Nivel de módulo, monkeypatcheable, mismo motivo que
    `_cpu_usage_percent`."""
    try:
        result = subprocess.run(
            _NVIDIA_SMI_COMMAND,
            capture_output=True,
            text=True,
            check=False,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            # Sin esto, Windows abre una ventana de consola visible para `nvidia-smi.exe` cada
            # vez — mismo bug real encontrado en vivo en `jarvis.league.lcu_monitor`/
            # `jarvis.tools.close_app` (JARVIS corre vía `pythonw.exe`, sin consola propia).
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        # OSError cubre el caso común: `nvidia-smi` no existe en el PATH (máquina sin GPU NVIDIA,
        # o driver no instalado) — degradación esperada, no un error a reportar.
        logger.info("nvidia-smi no disponible; se omite la sección de GPU.")
        return None
    if result.returncode != 0:
        logger.info(
            "nvidia-smi devolvió código %s; se omite la sección de GPU.",
            result.returncode,
        )
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return line or None


def _parse_gpu_line(line: str) -> tuple[float, float, float, float] | None:
    """Parsea una línea CSV de `nvidia-smi` a `(uso%, mem_usada_mib, mem_total_mib, temp_c)`, o
    `None` si el formato no es el esperado — degradación, no excepción, mismo criterio que
    `_raw_gpu_stats`."""
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 4:
        return None
    try:
        utilization, memory_used, memory_total, temperature = (
            float(field) for field in fields
        )
    except ValueError:
        return None
    return utilization, memory_used, memory_total, temperature


class SystemInfoTool(Tool):
    """Reporta un resumen del estado actual del sistema: uso de CPU, uso de RAM, y (si hay una
    GPU NVIDIA disponible) uso/memoria/temperatura de GPU. Solo lectura."""

    name = "system_info"
    description = (
        "Reporta el estado actual de la computadora: porcentaje de uso de CPU, uso de memoria "
        "RAM (usada de total, y porcentaje), y si hay una GPU NVIDIA disponible también su uso, "
        "memoria y temperatura. Usar cuando el usuario pregunta cómo está de recursos la "
        "computadora, si está lenta, cuánta RAM/CPU/GPU está usando, o similar."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    async def execute(self, **kwargs: Any) -> str:
        try:
            cpu_percent = await asyncio.to_thread(_cpu_usage_percent)
            used_gb, total_gb, memory_percent = await asyncio.to_thread(_memory_stats)
        except OSError as exc:
            return (
                f"No pude obtener información del sistema ({exc.__class__.__name__})."
            )

        parts = [
            f"CPU al {cpu_percent:.0f} por ciento",
            f"RAM: {used_gb:.1f} de {total_gb:.1f} GB ({memory_percent:.0f} por ciento)",
        ]

        gpu_line = await asyncio.to_thread(_raw_gpu_stats)
        if gpu_line is not None:
            parsed = _parse_gpu_line(gpu_line)
            if parsed is not None:
                gpu_percent, memory_used_mib, memory_total_mib, temperature_c = parsed
                parts.append(
                    f"GPU al {gpu_percent:.0f} por ciento, memoria "
                    f"{memory_used_mib / _MIB_PER_GB:.1f} de "
                    f"{memory_total_mib / _MIB_PER_GB:.1f} GB, {temperature_c:.0f} grados"
                )

        return ", ".join(parts) + "."
