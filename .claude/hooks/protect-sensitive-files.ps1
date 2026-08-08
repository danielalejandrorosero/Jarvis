# PreToolUse hook — matcher: Edit|Write
#
# Bloquea escritura/edicion sobre archivos que probablemente contienen secretos, sin importar
# que herramienta los invoque. Complementa (no reemplaza) el .gitignore: el .gitignore evita que
# un secreto ya escrito se commitee; este hook evita que el agente lo escriba o lo toque en
# primer lugar.

try {
    $raw = [Console]::In.ReadToEnd()
    $payload = $raw | ConvertFrom-Json -ErrorAction Stop
    $path = $payload.tool_input.file_path
    if ([string]::IsNullOrWhiteSpace($path)) { exit 0 }

    $safeExceptions = @(
        '\.env\.(example|sample|template)$'
    )
    foreach ($exception in $safeExceptions) {
        if ($path -match $exception) { exit 0 }
    }

    $sensitivePatterns = @(
        '\.env(\.|$)',
        '\.credentials\.json$',
        'secrets?\.\w+$',
        '\.pem$',
        '\.key$',
        '\.pfx$',
        'id_rsa(\.pub)?$',
        'id_ed25519(\.pub)?$'
    )

    foreach ($pattern in $sensitivePatterns) {
        if ($path -match $pattern) {
            [Console]::Error.WriteLine(
                "BLOQUEADO por protect-sensitive-files: '$path' coincide con un patron de " +
                "archivo sensible ('$pattern'). Si esto es intencional, edita el archivo " +
                "manualmente fuera de Claude Code."
            )
            exit 2
        }
    }
    exit 0
} catch {
    [Console]::Error.WriteLine("protect-sensitive-files: no se pudo evaluar el payload, se permite por fail-open. Revisar hook.")
    exit 1
}
