"""jarvis.ui.overlay — ventana flotante mostrando en vivo el estado de
`jarvis.audio.pipeline.run()` (ADR-0008): un círculo tipo Siri, pero para Windows.

Proceso separado, siempre — nunca se importa desde `jarvis.audio.pipeline` ni comparte proceso
con `run()`. `run()` ya corre su propio loop de eventos (`asyncio`, wake word/STT/LLM/TTS
bloqueantes despachados con `asyncio.to_thread`); Tkinter necesita el suyo propio
(`Tk.mainloop()`), y mezclar dos loops de eventos distintos en el mismo proceso es una fuente de
bugs de sincronización que no se justifica para una ventana de solo lectura (ver ADR-0008 para la
comparación completa de alternativas). Este módulo, a propósito, no importa nada de
`jarvis.audio`/`jarvis.llm`/`jarvis.security` — su única dependencia dentro del repo es
`jarvis.ui.status` (el contrato de datos puro, sin tkinter).

Runnable standalone: `python -m jarvis.ui.overlay`. En producción se lanza como
`pythonw.exe -m jarvis.ui.overlay` (sin consola), igual que `jarvis.audio.pipeline` — ver
`scripts/start_jarvis.ps1`. Ambos procesos son independientes: matar o reiniciar uno no requiere
tocar el otro (el overlay simplemente vuelve a mostrar "sin conexión" hasta que `run()` vuelva a
escribir estado, y `run()` nunca depende de que el overlay exista para funcionar).

La lógica de "¿qué mostrar dado este snapshot?" (`resolve_display`/`truncate_display_text`) está
separada a propósito de la construcción de widgets (`Overlay`) — Tkinter real no es
razonablemente testeable en este entorno (mismo criterio que el repo ya aplica a
`record_command`/audio de hardware real: sin mocks de bajo nivel que terminen probando el mock en
vez del comportamiento), pero la traducción snapshot → (label, color, texto) es una función pura
sin ningún widget de por medio, y esa sí se cubre con tests (`tests/ui/test_overlay.py`).
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path

from jarvis.ui.status import (
    DEFAULT_STALE_AFTER_SECONDS,
    DEFAULT_STATUS_PATH,
    StatusSnapshot,
    StatusState,
    is_stale,
    read_status,
)

# Pedido explícito: sondeo cada "~150-250ms" — 200ms es el punto medio de ese rango. Vía
# `Tk.after()` (el propio scheduler del loop de eventos de Tkinter), no un thread aparte: un
# thread de fondo tocando widgets de Tk desde fuera del loop principal es una fuente clásica de
# bugs de Tkinter (no es thread-safe para mutación de widgets fuera de su propio loop).
POLL_INTERVAL_MS = 200

WINDOW_WIDTH = 260
WINDOW_HEIGHT = 84
SCREEN_MARGIN = (
    16  # separación del borde de pantalla, esquina inferior derecha por default
)
WINDOW_ALPHA = 0.88  # semi-transparente — soportado por `-alpha` en Tk/Windows

# Ventana "de un vistazo" (pedido explícito: "no es un chat log completo") — bastante más corto
# que `jarvis.ui.status.MAX_LAST_TEXT_LENGTH` (200, el tope de lo que se persiste en el archivo);
# acá se trunca otra vez, más agresivo, solo para lo que entra legible en una ventana chica.
MAX_DISPLAY_TEXT_LENGTH = 70

DISCONNECTED_LABEL = "JARVIS — sin conexión"
DISCONNECTED_COLOR = "#5c1a1a"
DISCONNECTED_TEXT = "Esperando a que arranque el asistente de voz..."
_EMPTY_TEXT_PLACEHOLDER = "(sin actividad todavía)"

# Un color/etiqueta por estado — deliberadamente simple (texto + color de fondo, sin iconos
# custom ni animaciones): "simple y legible de un vistazo durante una partida" es un requisito
# explícito, no una limitación de esta primera versión a resolver después.
_STATE_LABELS: dict[StatusState, str] = {
    StatusState.IDLE: "JARVIS — esperando",
    StatusState.LISTENING: "JARVIS — escuchando",
    StatusState.THINKING: "JARVIS — pensando",
    StatusState.SPEAKING: "JARVIS — hablando",
}
_STATE_COLORS: dict[StatusState, str] = {
    StatusState.IDLE: "#2b2b2b",
    StatusState.LISTENING: "#1b5e20",
    StatusState.THINKING: "#8a6100",
    StatusState.SPEAKING: "#0d47a1",
}


def truncate_display_text(
    text: str, *, max_length: int = MAX_DISPLAY_TEXT_LENGTH
) -> str:
    """Recortar `text` a `max_length` caracteres, con un `…` final si se cortó algo — función
    pura, sin ningún widget de por medio."""
    stripped = text.strip()
    if len(stripped) <= max_length:
        return stripped
    return stripped[: max_length - 1].rstrip() + "…"


def resolve_display(
    snapshot: StatusSnapshot | None,
    *,
    now: float,
    stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
) -> tuple[str, str, str]:
    """Traducir un snapshot leído (o su ausencia) a `(label, color, texto)` — función pura, cubre
    la regla de degradación (punto 4 del pedido original): sin snapshot, o con uno más viejo que
    `stale_after_seconds`, se muestra como desconectado en vez de como el último estado real
    conocido (nunca "estado stale mostrado como si fuera en vivo").

    `now` siempre inyectado explícitamente (nunca `time.time()` adentro) — testeable sin depender
    del reloj real, mismo criterio que `jarvis.ui.status.is_stale`/
    `jarvis.audio.pipeline._current_time_line`.
    """
    if snapshot is None or is_stale(
        snapshot.timestamp, now=now, max_age_seconds=stale_after_seconds
    ):
        return DISCONNECTED_LABEL, DISCONNECTED_COLOR, DISCONNECTED_TEXT
    label = _STATE_LABELS[snapshot.state]
    color = _STATE_COLORS[snapshot.state]
    text = (
        truncate_display_text(snapshot.last_text)
        if snapshot.last_text.strip()
        else _EMPTY_TEXT_PLACEHOLDER
    )
    return label, color, text


class Overlay:
    """Ventana flotante, sin bordes, siempre encima, en una esquina de la pantalla.

    `overrideredirect(True)` (sin bordes/título) + `-topmost` (siempre visible) + `-alpha`
    (semi-transparente) + `-toolwindow` (oculta de la barra de tareas/Alt-Tab en Windows) — las
    cuatro son atributos nativos de Tk/Win32, sin dependencia nueva. Nunca llama a
    `focus_force()`/`grab_set()` ni nada que le robe foco a la ventana activa (pedido explícito:
    "no debe robar foco mientras juega") — riesgo residual documentado en ADR-0008: no hay
    verificación en vivo contra un cliente de League corriendo en fullscreen exclusivo real (a
    diferencia de fullscreen sin bordes), que en teoría podría renderizar por encima o por debajo
    del overlay según cómo maneje el compositor de Windows esa combinación — no es algo que
    Tkinter pueda controlar desde este lado.

    `path`/`poll_interval_ms`/`stale_after_seconds` son parámetros explícitos (no constantes de
    módulo hardcodeadas adentro) para dejar el constructor testeable en teoría — en la práctica
    esta clase no se instancia en tests (ver docstring del módulo), pero la forma es la misma que
    ya usa el resto del repo para funciones con I/O real (ej. `ScreenshotTool`,
    `jarvis.ui.status.write_status`).
    """

    def __init__(
        self,
        *,
        status_path: str | Path = DEFAULT_STATUS_PATH,
        poll_interval_ms: int = POLL_INTERVAL_MS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self._status_path = status_path
        self._poll_interval_ms = poll_interval_ms
        self._stale_after_seconds = stale_after_seconds
        self._root = tk.Tk()
        self._configure_window()
        self._label_var = tk.StringVar(value=DISCONNECTED_LABEL)
        self._text_var = tk.StringVar(value=DISCONNECTED_TEXT)
        self._frame, self._label_widget, self._text_widget = self._build_widgets()

    def _configure_window(self) -> None:
        root = self._root
        root.title("JARVIS")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", WINDOW_ALPHA)
        except tk.TclError:
            # Build de Tk sin soporte de `-alpha` (poco común en Windows, pero no imposible) —
            # degrada a ventana opaca en vez de crashear, mismo espíritu de resiliencia que el
            # resto del repo (`.claude/rules/python.md`).
            pass
        try:
            root.attributes("-toolwindow", True)
        except tk.TclError:
            pass
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = screen_w - WINDOW_WIDTH - SCREEN_MARGIN
        y = screen_h - WINDOW_HEIGHT - SCREEN_MARGIN
        root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")
        root.configure(bg=DISCONNECTED_COLOR)

    def _build_widgets(self) -> tuple[tk.Frame, tk.Label, tk.Label]:
        frame = tk.Frame(self._root, bg=DISCONNECTED_COLOR)
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        label_widget = tk.Label(
            frame,
            textvariable=self._label_var,
            fg="white",
            bg=DISCONNECTED_COLOR,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        label_widget.pack(fill="x")
        text_widget = tk.Label(
            frame,
            textvariable=self._text_var,
            fg="#e0e0e0",
            bg=DISCONNECTED_COLOR,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=WINDOW_WIDTH - 24,
        )
        text_widget.pack(fill="x", pady=(4, 0))
        return frame, label_widget, text_widget

    def _poll(self) -> None:
        snapshot = read_status(self._status_path)
        label, color, text = resolve_display(
            snapshot, now=time.time(), stale_after_seconds=self._stale_after_seconds
        )
        self._label_var.set(label)
        self._text_var.set(text)
        self._root.configure(bg=color)
        self._frame.configure(bg=color)
        self._label_widget.configure(bg=color)
        self._text_widget.configure(bg=color)
        # Reprogramarse a sí mismo — el propio scheduler de Tk, no un `while True` ni un thread
        # aparte (ver comentario de `POLL_INTERVAL_MS`).
        self._root.after(self._poll_interval_ms, self._poll)

    def run(self) -> None:
        """Arrancar el sondeo y ceder el control al loop de eventos de Tk. No retorna hasta que
        se cierra la ventana."""
        self._poll()
        self._root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay flotante de estado en vivo de JARVIS (ADR-0008)"
    )
    parser.add_argument(
        "--status-path",
        type=str,
        default=str(DEFAULT_STATUS_PATH),
        help="Ruta del archivo de estado escrito por jarvis.audio.pipeline.run()",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=POLL_INTERVAL_MS,
        help="Cada cuántos milisegundos releer el archivo de estado",
    )
    args = parser.parse_args()
    Overlay(status_path=args.status_path, poll_interval_ms=args.poll_interval_ms).run()


if __name__ == "__main__":
    main()
