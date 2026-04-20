param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"

function Get-CgcCommand {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\Scripts\cgc.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\Scripts\cgc.exe"),
        "cgc"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -eq "cgc") {
            return $candidate
        }
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Could not find cgc.exe. Install CodeGraphContext first."
}

function Get-CodeGraphMcpProcesses {
    $processes = Get-CimInstance Win32_Process -Filter "Name = 'cgc.exe'"
    return @(
        $processes | Where-Object {
            $cmd = $_.CommandLine
            $null -ne $cmd -and
            $cmd -match '(?i)cgc\.exe' -and
            $cmd -match '(?i)\bmcp\s+start\b'
        }
    )
}

function Stop-CodeGraphMcpProcesses {
    $targets = Get-CodeGraphMcpProcesses
    if (-not $targets) {
        Write-Host "No running CodeGraph MCP server processes were found."
        return
    }

    foreach ($target in $targets) {
        Write-Host ("Stopping CodeGraph MCP PID {0}" -f $target.ProcessId)
        Stop-Process -Id $target.ProcessId -Force
    }

    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Start-Sleep -Milliseconds 500
        if (-not (Get-CodeGraphMcpProcesses)) {
            return
        }
    }

    throw "Some CodeGraph MCP processes are still alive after stop attempts."
}

function Test-CodeGraphDb {
    param([string]$CgcCommand)

    Write-Host "Verifying that CodeGraph can read the local graph database..."
    & $CgcCommand list
    if ($LASTEXITCODE -ne 0) {
        throw "CodeGraph DB check failed."
    }
}

$cgc = Get-CgcCommand

if (-not $CheckOnly) {
    Stop-CodeGraphMcpProcesses
}

Test-CodeGraphDb -CgcCommand $cgc
Write-Host "CodeGraph reset/check completed. Retry the MCP request from a fresh chat or rerun the tool action."
