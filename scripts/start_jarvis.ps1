# Lanzador para la Tarea Programada de Windows (arranque automático, ADR-0004).
#
# No se puede setear PYTHONPATH directamente en la acción de una Tarea Programada, por eso
# existe este wrapper: setea el env var y lanza pythonw.exe (sin ventana de consola) con la
# salida redirigida a un archivo de log, para poder revisar qué pasó si algo falla en silencio.
#
# `-u` (unbuffered) es necesario, no cosmético: sin consola adjunta, Python bufferea stdout por
# bloque en vez de por línea — "Dijiste: ..."/"JARVIS: ..." quedaban en un buffer en memoria y
# nunca se escribían al log hasta que el proceso terminara, dejando el log vacío mientras corría
# (confirmado en vivo: el log estaba vacío pese a horas de uso real).
#
# Un solo proceso, solo voz (ADR-0010): hubo un segundo proceso (`jarvis.ui.overlay`, ventana
# flotante de estado) lanzado acá mismo, eliminado tras varias iteraciones visuales rechazadas —
# ver ADR-0010 para el porqué.
#
# Dos archivos FIJOS, `jarvis.log`/`jarvis-error.log` (pedido explícito del usuario: nombres fijos,
# nunca inflar archivos nuevos en cada reinicio), en modo APPEND (pedido explícito, aparte: "que
# no se borren al reiniciar" — las dos cosas juntas). `Start-Process -RedirectStandardOutput`/
# `-RedirectStandardError` de PowerShell NO soportan append, solo truncan el archivo en cada
# arranque — por eso, en vez de esos parámetros, se lanza `pythonw.exe` a través de
# `cmd.exe /c ... >> jarvis.log 2>> jarvis-error.log`, redirección nativa de shell que sí soporta
# append (`>>`) para cada stream por separado. Se probaron y descartaron dos variantes antes de
# esta (ambas en vivo, con el usuario viendo el resultado): un archivo por corrida con timestamp
# (mucha proliferación de archivos) y un solo archivo fusionando ambos streams (perdía la
# separación stdout/stderr que el usuario sí quería conservar). Sin rotación/límite de antigüedad
# todavía: son logs de texto de un asistente personal, no tráfico de producción — si el volumen se
# vuelve un problema real algún día, ahí se agrega una política de retención, no antes.

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

# El comando entero va envuelto en un par EXTRA de comillas ("" al principio y al final) —
# truco necesario de `cmd.exe /c`: cuando el argumento tiene más de un segmento entre comillas
# (acá, la ruta de pythonw.exe y las de los dos logs), el "quote stripping" especial de `/c`
# necesita ese par externo para no confundirse con los internos — confirmado en vivo: sin el par
# extra, cmd.exe arrancaba y moría al instante sin crear ningún log ni ningún error visible
# (silencioso, el peor tipo de falla — costó varias corridas de prueba encontrarlo).
$CmdArgs = "/c `"`"$PythonwExe`" -u -m jarvis.audio.pipeline >> `"$OutLog`" 2>> `"$ErrLog`"`""

Start-Process -FilePath "cmd.exe" `
    -ArgumentList $CmdArgs `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden
