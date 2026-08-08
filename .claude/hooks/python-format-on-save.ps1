# PostToolUse hook — matcher: Edit|Write
#
# Garantia determinista de formato/lint consistente en codigo Python, sin depender de que el
# modelo se acuerde de formatear. Alcance: solo el archivo que se acaba de tocar (barato, no
# corre sobre el repo completo). No-op silencioso si ruff no esta instalado todavia (no hay
# codigo Python en el repo en esta fase) o si el archivo no es .py.

try {
    $raw = [Console]::In.ReadToEnd()
    $payload = $raw | ConvertFrom-Json -ErrorAction Stop
    $path = $payload.tool_input.file_path
    if ([string]::IsNullOrWhiteSpace($path) -or ($path -notmatch '\.py$')) { exit 0 }
    if (-not (Get-Command ruff -ErrorAction SilentlyContinue)) { exit 0 }
    if (-not (Test-Path $path)) { exit 0 }

    & ruff format -- "$path" 2>&1 | Out-Null
    & ruff check --fix -- "$path" 2>&1 | Out-Null
    exit 0
} catch {
    exit 0
}
