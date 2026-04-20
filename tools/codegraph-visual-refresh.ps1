param(
    [string]$Context = "omen-visual",
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
$indexArgs = @("index", $repoRoot, "--context", $Context)
if ($Force) {
    $indexArgs += "--force"
}

Write-Host "Refreshing visual context '$Context' for $repoRoot..."
& $cgc @indexArgs
exit $LASTEXITCODE
