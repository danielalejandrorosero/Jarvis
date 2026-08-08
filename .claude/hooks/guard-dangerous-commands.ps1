# PreToolUse hook — matcher: Bash|PowerShell
#
# Bloquea de forma determinista comandos que caen en la categoria DANGEROUS del modelo de
# seguridad (.claude/rules/security.md), independientemente de lo que el LLM haya decidido o de
# como el usuario haya formulado la peticion. Esta es la capa de politica: decide, no propone.
#
# Fail-open deliberado: si el hook no puede parsear el payload (p.ej. el esquema de hooks cambia
# en una version futura de Claude Code), no bloquea todo el trabajo del usuario. El guard es
# defensa en profundidad, no la unica capa de seguridad — los prompts de permiso normales siguen
# activos. Ver docs/decisions/0002-security-enforcement-in-hooks.md.

try {
    $raw = [Console]::In.ReadToEnd()
    $payload = $raw | ConvertFrom-Json -ErrorAction Stop
    $command = $payload.tool_input.command
    if ([string]::IsNullOrWhiteSpace($command)) { exit 0 }

    $dangerousPatterns = @(
        'Remove-Item\s+.*-Recurse.*-Force.*[A-Za-z]:\\\s*$',
        'Remove-Item\s+.*-Recurse.*-Force.*\\Windows(\\|\s|$)',
        'rm\s+-rf\s+/',
        'Format-Volume',
        '\bdiskpart\b',
        'reg(\.exe)?\s+delete\s+HKLM',
        'vssadmin\s+delete',
        'cipher\s+/w',
        'Stop-Computer',
        'Restart-Computer',
        'shutdown(\.exe)?\s+/[sr]\b',
        'bcdedit',
        'Clear-Disk',
        'Initialize-Disk'
    )

    foreach ($pattern in $dangerousPatterns) {
        if ($command -match $pattern) {
            [Console]::Error.WriteLine(
                "BLOQUEADO por guard-dangerous-commands: el comando coincide con un patron " +
                "DANGEROUS ('$pattern'). Esta operacion requiere ejecucion manual y explicita " +
                "del usuario, no automatizacion por el agente. Si es intencional, pide al " +
                "usuario que lo ejecute el mismo."
            )
            exit 2
        }
    }

    # --- Cierre del bypass de shell sobre archivos sensibles ---
    # protect-sensitive-files.ps1 solo intercepta las herramientas Edit/Write/Read. Nada impedia
    # antes que el mismo resultado se lograra invocando PowerShell directamente (Set-Content,
    # Get-Content, redirecciones). Este bloque replica esa proteccion a nivel de comando de shell.
    # Las excepciones example/sample/template se evaluan primero para no bloquear plantillas.
    $safeFileExceptions = @(
        '\.env\.(example|sample|template)\b'
    )
    $touchesSafeException = $false
    foreach ($exception in $safeFileExceptions) {
        if ($command -match $exception) { $touchesSafeException = $true; break }
    }

    if (-not $touchesSafeException) {
        $sensitiveFileFragment = '(\.env(\.\w+)?\b|\.credentials\.json\b|secrets?\.\w+\b|\.pem\b|\.key\b|\.pfx\b|id_rsa(\.pub)?\b|id_ed25519(\.pub)?\b)'
        $writeVerbPattern = '(Set-Content|Add-Content|Out-File|>{1,2})'
        $readVerbPattern = '(Get-Content|(?<![\w-])type(?![\w-])|(?<![\w-])cat(?![\w-]))'

        $touchesSensitiveFile = $command -match $sensitiveFileFragment
        $isWriteOp = $command -match $writeVerbPattern
        $isReadOp = $command -match $readVerbPattern

        if ($touchesSensitiveFile -and ($isWriteOp -or $isReadOp)) {
            [Console]::Error.WriteLine(
                "BLOQUEADO por guard-dangerous-commands: el comando parece leer o escribir un " +
                "archivo sensible via shell (Set-Content/Add-Content/Out-File/redireccion, o " +
                "Get-Content/type/cat). Usa el archivo .env solo por fuera de Claude Code, o pide " +
                "al usuario que lo haga el."
            )
            exit 2
        }
    }

    exit 0
} catch {
    [Console]::Error.WriteLine("guard-dangerous-commands: no se pudo evaluar el payload, se permite por fail-open. Revisar hook.")
    exit 1
}
