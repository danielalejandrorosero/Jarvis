"""jarvis.ui.overlay — HUD circular flotante mostrando en vivo el estado de
`jarvis.audio.pipeline.run()`: un dial tipo "Iron Man HUD", arrastrable, con posición persistida.

Segunda versión visual de esta feature (la primera, un rectángulo tipo toast con texto, fue
rechazada explícitamente por el usuario: "toast notification garbage"). El mecanismo de
comunicación entre procesos (`jarvis.ui.status`: `data/status.json`, sondeado cada ~200ms) no
cambió — sigue siendo el contrato correcto (ver ADR-0008), lo único que cambió acá es cómo se
dibuja y cómo se posiciona esa información.

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

Igual que en la primera versión, toda la lógica de "¿qué mostrar/dónde, dado este snapshot o este
click del mouse?" vive en funciones puras separadas de la construcción de widgets (`Overlay`) —
Tkinter real no es razonablemente testeable en este entorno (mismo criterio que el repo ya aplica
a `record_command`/audio de hardware real), pero la traducción snapshot → estado visible, la
geometría del dial (marcas, arco) y la aritmética de arrastre/posicionamiento sí son funciones
puras sin ningún widget de por medio, y esas se cubren con tests (`tests/ui/test_overlay.py`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

from jarvis.ui.status import (
    DEFAULT_STALE_AFTER_SECONDS,
    DEFAULT_STATUS_PATH,
    StatusSnapshot,
    StatusState,
    is_stale,
    read_status,
)

logger = logging.getLogger(__name__)

# Pedido explícito: sondeo cada "~150-250ms" — 200ms es el punto medio de ese rango. Vía
# `Tk.after()` (el propio scheduler del loop de eventos de Tkinter), no un thread aparte: un
# thread de fondo tocando widgets de Tk desde fuera del loop principal es una fuente clásica de
# bugs de Tkinter (no es thread-safe para mutación de widgets fuera de su propio loop). El mismo
# tick que dispara la relectura de `data/status.json` también avanza el contador de animación
# (`Overlay._tick`) — no hay un timer de animación separado.
POLL_INTERVAL_MS = 200

# Diámetro aproximado del dial — dentro del rango pedido (180-260px), pensado para sentarse sin
# molestar durante una partida (mismo espíritu que el rectángulo 260x84 de la primera versión,
# ahora cuadrado porque un dial circular no cabe bien en un rectángulo ancho y bajo).
WINDOW_SIZE = 220
SCREEN_MARGIN = 16  # separación del borde de pantalla para la posición default (esquina inferior derecha)

# Geometría del dial, todo relativo al centro de la ventana (`WINDOW_SIZE / 2`).
CIRCLE_MARGIN = 6
OUTER_RING_RADIUS = WINDOW_SIZE / 2 - CIRCLE_MARGIN
TICK_OUTER_RADIUS = OUTER_RING_RADIUS - 2
TICK_MINOR_INNER_RADIUS = OUTER_RING_RADIUS - 10
TICK_MAJOR_INNER_RADIUS = OUTER_RING_RADIUS - 17
STATE_RING_RADIUS = OUTER_RING_RADIUS - 32
TICK_COUNT = 36
TICK_MAJOR_EVERY = 6

# Paleta — cian/azul en tonos oscuros sobre fondo casi negro, aproximación estilizada del HUD de
# referencia (Tkinter no puede hacer bloom/glow real, así que el "glow" es aproximado con anillos
# finos de distinto brillo, no con blending real).
DISC_COLOR = "#050a10"
OUTER_RING_COLOR = "#123244"
TICK_MINOR_COLOR = "#173f4f"
TICK_MAJOR_COLOR = "#2fa0c2"
IDENTITY_COLOR = "#eafcff"
IDENTITY_DISCONNECTED_COLOR = "#8a8a8a"
DETAIL_TEXT_COLOR = "#7c98a3"

# Color-clave para el "recorte" de la ventana (`-transparentcolor`, atributo nativo de Tk en
# Windows): todo pixel exactamente de este color se vuelve invisible *y* transparente al click —
# así las esquinas del cuadrado por fuera del dial ni se ven ni interceptan el mouse (el juego de
# fondo las recibe), y arrastrar solo funciona clickeando el dial visible en sí (pedido explícito:
# "dragging happens directly on the visible HUD surface"). Magenta bien saturado, deliberadamente
# fuera de la paleta cian/azul/gris de arriba, para que ningún elemento real del dial coincida por
# accidente con la clave y se vuelva invisible sin querer.
TRANSPARENT_KEY = "#ff00fe"
# Si el build de Tk no soporta `-transparentcolor` (poco común en Windows, pero no imposible — ver
# `_configure_window`), se usa este color de fondo opaco normal en su lugar: sin esquinas magenta
# ni click-through, pero la ventana sigue siendo la misma, sin crashear.
FALLBACK_BACKGROUND_COLOR = "#0a0c0e"

# El texto identitario visible en el HUD. Nunca "JARVIS": el propio `SYSTEM_PROMPT` le prohíbe al
# asistente presentarse como JARVIS en voz o en UI — "Alexa en voz, JARVIS en código" (ver
# README). El usuario ve/oye "Alexa"; el paquete y el código internamente siguen llamándose
# `jarvis`, pero eso nunca debe filtrarse a algo que el usuario mira en pantalla.
IDENTITY_TEXT = "ALEXA"

DISCONNECTED_LABEL = "SIN CONEXIÓN"
DISCONNECTED_TEXT = "Esperando a que arranque el asistente de voz..."
_EMPTY_TEXT_PLACEHOLDER = "(sin actividad todavía)"

# Ventana "de un vistazo" (pedido explícito original: "no es un chat log completo") — el texto de
# detalle del dial es más chico que el rectángulo de la primera versión (bastante menos lugar en
# un círculo de 220px que en una franja de 260x84), así que se trunca más agresivo acá, aparte del
# truncado ya aplicado en `jarvis.ui.status.MAX_LAST_TEXT_LENGTH` (200, lo que se persiste) y en
# `MAX_DISPLAY_TEXT_LENGTH` (70, lo que entraría en una ventana más ancha).
MAX_DISPLAY_TEXT_LENGTH = 70
DETAIL_TEXT_MAX_LENGTH = 42

# Un color por estado — el mismo tono maneja el anillo estructural y el arco animado de ese
# estado, para que "un color, un estado" sea consistente en todo el dial de un vistazo.
_STATE_COLORS: dict[StatusState, str] = {
    StatusState.IDLE: "#1c4b5e",
    StatusState.LISTENING: "#00e5ff",
    StatusState.THINKING: "#ffb300",
    StatusState.SPEAKING: "#2979ff",
}
DISCONNECTED_COLOR = "#7a2323"

_STATE_LABELS: dict[StatusState, str] = {
    StatusState.IDLE: "EN ESPERA",
    StatusState.LISTENING: "ESCUCHANDO",
    StatusState.THINKING: "PENSANDO",
    StatusState.SPEAKING: "HABLANDO",
}

# Extent base (grados) del arco animado por estado — ver `compute_arc` para cómo cada uno anima
# alrededor de este valor. Elegidos para que los cuatro estados sean distinguibles a ojo sin leer
# texto: HABLANDO es el arco más grande y el que pulsa más rápido, PENSANDO es el único que rota
# en vez de pulsar (barrido tipo "escaneando"), ESCUCHANDO pulsa medio, EN ESPERA apenas respira.
_ARC_BASE_EXTENT: dict[StatusState, int] = {
    StatusState.IDLE: 40,
    StatusState.LISTENING: 100,
    StatusState.THINKING: 70,
    StatusState.SPEAKING: 130,
}
_DISCONNECTED_ARC_START = 90
_DISCONNECTED_ARC_EXTENT = 20

DEFAULT_POSITION_PATH = Path("data/overlay_position.json")
# Cuántos píxeles del dial se garantizan visibles/agarrables dentro de la pantalla como mínimo —
# una posición guardada de un segundo monitor ya desconectado, o de una resolución vieja, nunca
# debe dejar el HUD completamente fuera de pantalla e inalcanzable para volver a arrastrarlo.
MIN_VISIBLE_MARGIN_PX = 24


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
) -> tuple[StatusState | None, str]:
    """Traducir un snapshot leído (o su ausencia) a `(estado, texto)` — función pura, cubre la
    regla de degradación: sin snapshot, o con uno más viejo que `stale_after_seconds`, el estado
    resuelto es `None` ("desconectado") en vez del último estado real conocido (nunca un estado
    stale mostrado como si fuera en vivo). `None` nunca es un `StatusState` real — ver docstring de
    `jarvis.ui.status.StatusState` — es exclusivamente la conclusión que este overlay deriva.

    `now` siempre inyectado explícitamente (nunca `time.time()` adentro) — testeable sin depender
    del reloj real, mismo criterio que `jarvis.ui.status.is_stale`/
    `jarvis.audio.pipeline._current_time_line`.
    """
    if snapshot is None or is_stale(
        snapshot.timestamp, now=now, max_age_seconds=stale_after_seconds
    ):
        return None, DISCONNECTED_TEXT
    text = (
        truncate_display_text(snapshot.last_text)
        if snapshot.last_text.strip()
        else _EMPTY_TEXT_PLACEHOLDER
    )
    return snapshot.state, text


def state_label(state: StatusState | None) -> str:
    """Etiqueta corta en español para el estado resuelto — `None` es "desconectado"."""
    return DISCONNECTED_LABEL if state is None else _STATE_LABELS[state]


def state_color(state: StatusState | None) -> str:
    """Un único color por estado — maneja tanto el anillo estructural como el arco animado de ese
    estado en el dial (ver comentario de `_STATE_COLORS`)."""
    return DISCONNECTED_COLOR if state is None else _STATE_COLORS[state]


def compute_arc(state: StatusState | None, tick: int) -> tuple[int, int]:
    """Mapeo puro `(estado, tick de animación) -> (ángulo_inicio, extensión)` en grados, listo
    para `tkinter.Canvas.create_arc`/`itemconfigure(start=..., extent=...)`. `tick` incrementa una
    vez por sondeo (~200ms) — el mismo reloj que ya maneja `Overlay._poll`, sin timer de animación
    aparte.

    `state=None` (desconectado) es un arco chico y fijo, sin animación: no hay ninguna señal en
    vivo que representar mientras `run()` no está reportando estado.

    Cada estado real anima distinto a propósito, para que los cuatro sean distinguibles a un
    vistazo durante una partida sin tener que leer la etiqueta de texto:
    - PENSANDO es el único que rota (`start` avanza con `tick`, `extent` fijo) — un barrido tipo
      "escaneando/procesando", no un pulso.
    - EN ESPERA, ESCUCHANDO y HABLANDO mantienen `start` fijo (arco centrado arriba) y hacen
      "respirar" el `extent` con una onda seno de período creciente en velocidad e intensidad:
      EN ESPERA es el más lento y sutil, HABLANDO el más rápido y el de mayor swing.
    """
    if state is None:
        return _DISCONNECTED_ARC_START, _DISCONNECTED_ARC_EXTENT

    base_extent = _ARC_BASE_EXTENT[state]

    if state is StatusState.THINKING:
        start = (tick * 15) % 360
        return start, base_extent

    period, swing = {
        StatusState.IDLE: (30, 8),
        StatusState.LISTENING: (10, 30),
        StatusState.SPEAKING: (6, 35),
    }[state]
    wave = math.sin(2 * math.pi * (tick % period) / period)
    extent = base_extent + round(wave * swing)
    return 90, max(extent, 10)


def tick_marks(
    *,
    center: float,
    outer_radius: float,
    minor_inner_radius: float,
    major_inner_radius: float,
    count: int,
    major_every: int,
) -> list[tuple[float, float, float, float, bool]]:
    """Geometría pura de las marcas radiales alrededor del dial: `count` segmentos
    `(x1, y1, x2, y2, es_mayor)` equiespaciados en un círculo de `outer_radius`, cada uno llegando
    hacia adentro hasta `major_inner_radius` (marcas mayores, cada `major_every`) o
    `minor_inner_radius` (el resto) — sin ningún `Canvas` de por medio, así el patrón de "gauge
    radial" es testeable sin abrir una ventana real."""
    marks: list[tuple[float, float, float, float, bool]] = []
    for i in range(count):
        angle = 2 * math.pi * i / count
        is_major = (i % major_every) == 0
        inner_radius = major_inner_radius if is_major else minor_inner_radius
        x1 = center + outer_radius * math.cos(angle)
        y1 = center + outer_radius * math.sin(angle)
        x2 = center + inner_radius * math.cos(angle)
        y2 = center + inner_radius * math.sin(angle)
        marks.append((x1, y1, x2, y2, is_major))
    return marks


@dataclass(frozen=True)
class OverlayPosition:
    """Posición guardada del HUD (esquina superior izquierda de la ventana) — inmutable, sin
    lógica propia, mismo espíritu que `jarvis.ui.status.StatusSnapshot`."""

    x: int
    y: int


def load_position(
    path: str | Path = DEFAULT_POSITION_PATH,
) -> OverlayPosition | None:
    """Leer y parsear `path`. Devuelve `None` si el archivo no existe, no es JSON válido, o le
    falta algún campo esperado — nunca lanza, mismo contrato que
    `jarvis.ui.status.read_status`. `None` es el resultado esperado en el primer arranque de
    siempre (todavía no se arrastró nunca el HUD) — el llamador (`Overlay`) lo trata como "usar la
    posición default", no como un error."""
    resolved_path = Path(path)
    try:
        raw = resolved_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return OverlayPosition(x=int(data["x"]), y=int(data["y"]))
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def save_position(x: int, y: int, *, path: str | Path = DEFAULT_POSITION_PATH) -> None:
    """Guardar la posición actual del HUD a `path`, en un formato JSON plano, atómicamente
    (archivo temporal + `os.replace()`) — mismo patrón que `jarvis.ui.status.write_status`, mismo
    motivo: que nunca se vea un JSON a medio escribir.

    Nunca lanza (mismo contrato que `write_status`): perder una posición arrastrada por un fallo
    real de disco (lleno, permisos) no es motivo para crashear el proceso del overlay — a lo sumo,
    la próxima vez que arranque vuelve a la posición default o a la última que sí se pudo guardar.
    Se loguea a WARNING y se descarta, nunca se re-lanza.
    """
    resolved_path = Path(path)
    tmp_path: Path | None = None
    try:
        payload = {"x": int(x), "y": int(y)}
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=resolved_path.parent,
            prefix=resolved_path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(json.dumps(payload))
        os.replace(tmp_path, resolved_path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("No se pudo guardar la posición del overlay (%s): %r", path, exc)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def clamp_position(
    x: int,
    y: int,
    *,
    window_width: int,
    window_height: int,
    screen_width: int,
    screen_height: int,
    min_visible_px: int = MIN_VISIBLE_MARGIN_PX,
) -> tuple[int, int]:
    """Ajustar `(x, y)` para que al menos `min_visible_px` del HUD quede dentro de los límites de
    pantalla — nunca lo deja completamente fuera de vista (y por lo tanto inalcanzable para volver
    a arrastrarlo), sea por una posición guardada de un monitor que ya no está conectado, un
    cambio de resolución, o un arrastre que se pasó del borde durante la propia sesión."""
    min_x = -window_width + min_visible_px
    max_x = screen_width - min_visible_px
    min_y = -window_height + min_visible_px
    max_y = screen_height - min_visible_px
    clamped_x = max(min_x, min(x, max_x))
    clamped_y = max(min_y, min(y, max_y))
    return clamped_x, clamped_y


def default_position(
    *,
    window_width: int,
    window_height: int,
    screen_width: int,
    screen_height: int,
    margin: int,
) -> tuple[int, int]:
    """Posición default: esquina inferior derecha, mismo criterio que la primera versión — solo
    se usa cuando no hay ninguna posición guardada todavía (primer arranque de siempre)."""
    return screen_width - window_width - margin, screen_height - window_height - margin


def resolve_initial_position(
    saved: OverlayPosition | None,
    *,
    window_width: int,
    window_height: int,
    screen_width: int,
    screen_height: int,
    margin: int,
) -> tuple[int, int]:
    """Posición con la que abrir la ventana: la guardada (recortada a pantalla, por si cambió de
    resolución/monitor desde que se guardó) si existe, o la default (esquina inferior derecha) en
    el primer arranque de siempre."""
    if saved is None:
        return default_position(
            window_width=window_width,
            window_height=window_height,
            screen_width=screen_width,
            screen_height=screen_height,
            margin=margin,
        )
    return clamp_position(
        saved.x,
        saved.y,
        window_width=window_width,
        window_height=window_height,
        screen_width=screen_width,
        screen_height=screen_height,
    )


def compute_drag_position(
    pointer_root_x: int, pointer_root_y: int, offset_x: int, offset_y: int
) -> tuple[int, int]:
    """Nueva esquina superior izquierda de la ventana dado el puntero en coordenadas absolutas de
    pantalla (`event.x_root`/`event.y_root`) y el offset del click dentro de la ventana
    (`event.x`/`event.y` capturados en `<ButtonPress-1>`) — patrón estándar de Tkinter para
    arrastrar una ventana sin bordes, extraído a función pura para poder testear la aritmética sin
    abrir un `Canvas` real."""
    return pointer_root_x - offset_x, pointer_root_y - offset_y


class Overlay:
    """HUD circular, sin bordes, siempre encima, arrastrable con el mouse a cualquier posición de
    pantalla (posición persistida en `DEFAULT_POSITION_PATH`).

    `overrideredirect(True)` (sin bordes/título) + `-topmost` (siempre visible) +
    `-transparentcolor` (recorta la ventana a la forma del dial, ver `TRANSPARENT_KEY` — con
    fallback a un fondo opaco normal si el build de Tk no lo soporta) + `-toolwindow` (oculta de
    la barra de tareas/Alt-Tab en Windows) — las cuatro son atributos nativos de Tk/Win32, sin
    dependencia nueva. Nunca llama a `focus_force()`/`grab_set()` ni nada que le robe foco a la
    ventana activa (pedido explícito: "no debe robar foco mientras juega", y el usuario aclaró que
    "modal" para él significa "una ventana real con la que puedo interactuar y moverla", no un
    diálogo bloqueante — nada acá bloquea ni roba foco) — riesgo residual documentado en ADR-0008:
    no hay verificación en vivo contra un cliente de League corriendo en fullscreen exclusivo real
    (a diferencia de fullscreen sin bordes), que en teoría podría renderizar por encima o por
    debajo del overlay según cómo maneje el compositor de Windows esa combinación — no es algo que
    Tkinter pueda controlar desde este lado.

    `status_path`/`position_path`/`poll_interval_ms`/`stale_after_seconds` son parámetros
    explícitos (no constantes de módulo hardcodeadas adentro) para dejar el constructor testeable
    en teoría — en la práctica esta clase no se instancia en tests (ver docstring del módulo), pero
    la forma es la misma que ya usa el resto del repo para funciones/clases con I/O real (ej.
    `ScreenshotTool`, `jarvis.ui.status.StatusHeartbeat`).
    """

    def __init__(
        self,
        *,
        status_path: str | Path = DEFAULT_STATUS_PATH,
        position_path: str | Path = DEFAULT_POSITION_PATH,
        poll_interval_ms: int = POLL_INTERVAL_MS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self._status_path = status_path
        self._position_path = position_path
        self._poll_interval_ms = poll_interval_ms
        self._stale_after_seconds = stale_after_seconds
        self._tick = 0
        self._drag_offset: tuple[int, int] | None = None
        self._current_position: tuple[int, int] | None = None

        self._root = tk.Tk()
        self._transparent_supported = self._configure_window()
        background = (
            TRANSPARENT_KEY
            if self._transparent_supported
            else FALLBACK_BACKGROUND_COLOR
        )
        self._canvas = self._build_canvas(background)
        (
            self._state_ring_id,
            self._arc_id,
            self._identity_text_id,
            self._state_label_id,
            self._detail_text_id,
        ) = self._build_dial_items()
        self._bind_drag_events()

    def _configure_window(self) -> bool:
        """Aplicar los atributos de ventana nativos. Devuelve si `-transparentcolor` quedó activo
        (determina qué color de fondo usa el canvas, ver `__init__`)."""
        root = self._root
        root.title("Alexa")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-toolwindow", True)
        except tk.TclError:
            pass

        transparent_supported = True
        try:
            root.attributes("-transparentcolor", TRANSPARENT_KEY)
        except tk.TclError:
            # Build de Tk sin soporte de `-transparentcolor` (poco común en Windows, pero no
            # imposible) — degrada a una ventana cuadrada opaca semi-transparente (el
            # comportamiento de la primera versión) en vez de crashear, mismo espíritu de
            # resiliencia que el resto del repo (`.claude/rules/python.md`).
            transparent_supported = False
            try:
                root.attributes("-alpha", 0.92)
            except tk.TclError:
                pass

        saved = load_position(self._position_path)
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x, y = resolve_initial_position(
            saved,
            window_width=WINDOW_SIZE,
            window_height=WINDOW_SIZE,
            screen_width=screen_w,
            screen_height=screen_h,
            margin=SCREEN_MARGIN,
        )
        self._current_position = (x, y)
        root.geometry(f"{WINDOW_SIZE}x{WINDOW_SIZE}+{x}+{y}")
        return transparent_supported

    def _build_canvas(self, background: str) -> tk.Canvas:
        canvas = tk.Canvas(
            self._root,
            width=WINDOW_SIZE,
            height=WINDOW_SIZE,
            bg=background,
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(fill="both", expand=True)
        self._root.configure(bg=background)

        center = WINDOW_SIZE / 2
        # Cara del dial: disco oscuro casi negro, estático, dibujado una sola vez.
        canvas.create_oval(
            center - OUTER_RING_RADIUS,
            center - OUTER_RING_RADIUS,
            center + OUTER_RING_RADIUS,
            center + OUTER_RING_RADIUS,
            fill=DISC_COLOR,
            outline=OUTER_RING_COLOR,
            width=2,
        )
        # Marcas radiales tipo gauge, estáticas — el "detalle técnico" del HUD de referencia.
        for x1, y1, x2, y2, is_major in tick_marks(
            center=center,
            outer_radius=TICK_OUTER_RADIUS,
            minor_inner_radius=TICK_MINOR_INNER_RADIUS,
            major_inner_radius=TICK_MAJOR_INNER_RADIUS,
            count=TICK_COUNT,
            major_every=TICK_MAJOR_EVERY,
        ):
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=TICK_MAJOR_COLOR if is_major else TICK_MINOR_COLOR,
                width=2 if is_major else 1,
            )
        return canvas

    def _build_dial_items(self) -> tuple[int, int, int, int, int]:
        canvas = self._canvas
        center = WINDOW_SIZE / 2
        bbox = (
            center - STATE_RING_RADIUS,
            center - STATE_RING_RADIUS,
            center + STATE_RING_RADIUS,
            center + STATE_RING_RADIUS,
        )
        # Anillo estructural del estado (fino, 360° completos) + arco animado encima (más grueso,
        # parcial) — mismo radio, el arco se dibuja después así queda arriba en el stacking order
        # del canvas. Ambos arrancan en el color/ángulo de "desconectado"; `_render` los actualiza
        # en el primer `_poll`.
        state_ring_id = canvas.create_oval(*bbox, outline=DISCONNECTED_COLOR, width=2)
        arc_id = canvas.create_arc(
            *bbox,
            start=_DISCONNECTED_ARC_START,
            extent=_DISCONNECTED_ARC_EXTENT,
            style=tk.ARC,
            outline=DISCONNECTED_COLOR,
            width=6,
        )
        identity_text_id = canvas.create_text(
            center,
            center - 10,
            text=IDENTITY_TEXT,
            fill=IDENTITY_DISCONNECTED_COLOR,
            font=("Segoe UI", 13, "bold"),
        )
        state_label_id = canvas.create_text(
            center,
            center + 10,
            text=DISCONNECTED_LABEL,
            fill=DISCONNECTED_COLOR,
            font=("Segoe UI", 8, "bold"),
        )
        detail_text_id = canvas.create_text(
            center,
            center + 32,
            text=DISCONNECTED_TEXT,
            fill=DETAIL_TEXT_COLOR,
            font=("Segoe UI", 7),
            width=int(STATE_RING_RADIUS * 2.1),
            justify="center",
        )
        return state_ring_id, arc_id, identity_text_id, state_label_id, detail_text_id

    def _bind_drag_events(self) -> None:
        # Bindeado directo sobre el canvas (la superficie visible del HUD), no sobre la ventana en
        # general — pedido explícito: "dragging happens directly on the visible HUD surface". Con
        # `-transparentcolor` activo, además, los píxeles fuera del dial ni siquiera reciben estos
        # eventos (son transparentes al click), así que en la práctica ya queda acotado al dial
        # visible incluso sin este bind siendo más selectivo por ítem.
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

    def _on_press(self, event: tk.Event) -> None:
        self._drag_offset = (event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_offset is None:
            return
        offset_x, offset_y = self._drag_offset
        raw_x, raw_y = compute_drag_position(
            event.x_root, event.y_root, offset_x, offset_y
        )
        x, y = clamp_position(
            raw_x,
            raw_y,
            window_width=WINDOW_SIZE,
            window_height=WINDOW_SIZE,
            screen_width=self._root.winfo_screenwidth(),
            screen_height=self._root.winfo_screenheight(),
        )
        self._root.geometry(f"+{x}+{y}")
        self._current_position = (x, y)

    def _on_release(self, _event: tk.Event) -> None:
        self._drag_offset = None
        if self._current_position is not None:
            x, y = self._current_position
            save_position(x, y, path=self._position_path)

    def _render(self, state: StatusState | None, text: str) -> None:
        color = state_color(state)
        start, extent = compute_arc(state, self._tick)
        self._canvas.itemconfigure(self._state_ring_id, outline=color)
        self._canvas.itemconfigure(
            self._arc_id, outline=color, start=start, extent=extent
        )
        self._canvas.itemconfigure(
            self._identity_text_id,
            fill=IDENTITY_COLOR if state is not None else IDENTITY_DISCONNECTED_COLOR,
        )
        self._canvas.itemconfigure(
            self._state_label_id, text=state_label(state), fill=color
        )
        self._canvas.itemconfigure(
            self._detail_text_id,
            text=truncate_display_text(text, max_length=DETAIL_TEXT_MAX_LENGTH),
        )

    def _poll(self) -> None:
        snapshot = read_status(self._status_path)
        state, text = resolve_display(
            snapshot, now=time.time(), stale_after_seconds=self._stale_after_seconds
        )
        self._tick += 1
        self._render(state, text)
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
        description="HUD circular flotante de estado en vivo de Alexa/JARVIS (ADR-0008)"
    )
    parser.add_argument(
        "--status-path",
        type=str,
        default=str(DEFAULT_STATUS_PATH),
        help="Ruta del archivo de estado escrito por jarvis.audio.pipeline.run()",
    )
    parser.add_argument(
        "--position-path",
        type=str,
        default=str(DEFAULT_POSITION_PATH),
        help="Ruta donde se persiste la última posición arrastrada del HUD",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=POLL_INTERVAL_MS,
        help="Cada cuántos milisegundos releer el archivo de estado",
    )
    args = parser.parse_args()
    Overlay(
        status_path=args.status_path,
        position_path=args.position_path,
        poll_interval_ms=args.poll_interval_ms,
    ).run()


if __name__ == "__main__":
    main()
