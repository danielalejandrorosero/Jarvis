# Lanzador para la Tarea Programada de Windows (arranque automático, ADR-0004).
#
# No se puede setear PYTHONPATH directamente en la acción de una Tarea Programada, por eso
# existe este wrapper: setea el env var y lanza pythonw.exe (sin ventana de consola) con la
# salida redirigida a data/jarvis.log, para poder revisar qué pasó si algo falla en silencio.
#
# `-u` (unbuffered) es necesario, no cosmético: sin consola adjunta, Python bufferea stdout por
# bloque en vez de por línea — "Dijiste: ..."/"JARVIS: ..." quedaban en un buffer en memoria y
# nunca se escribían a jarvis.log hasta que el proceso terminara, dejando el log vacío mientras
# corría (confirmado en vivo: el log estaba vacío pese a horas de uso real).
#
# Segundo proceso, `jarvis.ui.overlay` (ADR-0008): la ventana flotante de estado en vivo. Se
# lanza como un `Start-Process` completamente separado del pipeline de voz, a propósito — mismo
# `pythonw.exe`/`PYTHONPATH`/carpeta de logs, pero procesos independientes de principio a fin
# (dos PID distintos, dos pares de log distintos). Matar o reiniciar uno no afecta al otro: si el
# overlay crashea o nunca llega a arrancar, `jarvis.audio.pipeline` seguí funcionando exactamente
# igual (el overlay no tiene autoridad sobre el pipeline, ver `jarvis.ui`); si el overlay se
# reinicia, no hace falta reiniciar el reconocimiento de voz.

$ProjectRoot = "C:\Users\Hewlett-Packard\Desktop\jarvis"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$DataDir = Join-Path $ProjectRoot "data"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

# Sin `--device` explícito: se probó apuntar a un índice fijo del mic integrado (para evitar que
# Windows forzara el perfil Hands-Free del headset Bluetooth) como hipótesis para un audio "bajo
# el agua" reportado en vivo por el usuario, pero se descartó en la misma sesión — el problema
# persistía incluso con JARVIS completamente apagado, así que no era la causa. Revertido a
# `resolve_input_device(None)` (default del sistema, ver `device.py`) para no perder captación de
# voz sin ningún beneficio real; además los índices de `sounddevice` no son estables entre
# reconexiones de Bluetooth (confirmado en vivo: un índice fijo apuntó a un device de salida
# después de reconectar el headset, y tumbó el arranque).
$PythonwExe = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$OutLog = Join-Path $DataDir "jarvis.log"
$ErrLog = Join-Path $DataDir "jarvis-error.log"
$OverlayOutLog = Join-Path $DataDir "jarvis-overlay.log"
$OverlayErrLog = Join-Path $DataDir "jarvis-overlay-error.log"

Start-Process -FilePath $PythonwExe `
    -ArgumentList "-u", "-m", "jarvis.audio.pipeline" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden

# Overlay flotante de estado (ADR-0008) — proceso separado, ver comentario de arriba. Si esto
# falla en arrancar (ej. una máquina sin sesión gráfica interactiva todavía disponible en el
# momento exacto del login), el pipeline de voz de arriba ya arrancó y sigue funcionando: no hay
# dependencia entre los dos `Start-Process`.
Start-Process -FilePath $PythonwExe `
    -ArgumentList "-u", "-m", "jarvis.ui.overlay" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $OverlayOutLog `
    -RedirectStandardError $OverlayErrLog `
    -WindowStyle Hidden
