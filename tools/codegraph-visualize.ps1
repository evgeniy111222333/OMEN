param(
    [string]$Context = "omen-visual",
    [int]$Port = 8000,
    [switch]$Refresh,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
$repoRoot = Split-Path -Parent $PSScriptRoot

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

$cgc = Get-CgcCommand

if ($Refresh) {
    $refreshArgs = @("index", $repoRoot, "--context", $Context)
    if ($Force) {
        $refreshArgs += "--force"
    }

    Write-Host "Refreshing visual context '$Context'..."
    & $cgc @refreshArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Starting CodeGraph visual UI on http://localhost:$Port"
Write-Host "Context: $Context"
Write-Host "Repository: $repoRoot"
& $cgc visualize --repo $repoRoot --port $Port --context $Context
