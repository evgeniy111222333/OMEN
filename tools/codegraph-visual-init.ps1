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

Write-Host "Creating visual context '$Context' if needed..."
& $cgc context create $Context --database kuzudb
if ($LASTEXITCODE -ne 0) {
    Write-Host "Context creation returned a non-zero exit code. Continuing with indexing in case it already exists."
}

$indexArgs = @("index", $repoRoot, "--context", $Context)
if ($Force) {
    $indexArgs += "--force"
}

Write-Host "Indexing $repoRoot into visual context '$Context'..."
& $cgc @indexArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Done. You can now run:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\tools\codegraph-visualize.ps1"
