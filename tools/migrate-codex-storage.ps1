[CmdletBinding()]
param(
    [ValidateSet('StageAndQueue', 'Finalize')]
    [string]$Mode = 'StageAndQueue',

    [string]$UserRoot = 'C:\Users\HP',
    [string]$TargetRoot = 'E:\Users\HP',
    [string]$LogPath = 'E:\Users\HP\migration\codex-storage-migration.log'
)

$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message)

    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$timestamp] $Message"
    Write-Host $line

    $logDir = Split-Path -Parent $LogPath
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }

    Add-Content -LiteralPath $LogPath -Value $line
}

function Get-MigrationItems {
    @(
        @{
            Name = '.codex'
            Source = Join-Path $UserRoot '.codex'
            Target = Join-Path $TargetRoot '.codex'
        },
        @{
            Name = '.codegraphcontext'
            Source = Join-Path $UserRoot '.codegraphcontext'
            Target = Join-Path $TargetRoot '.codegraphcontext'
        }
    )
}

function Ensure-Directory {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Invoke-Robocopy {
    param(
        [string]$Source,
        [string]$Target
    )

    Ensure-Directory -Path $Target

    $arguments = @(
        $Source,
        $Target,
        '/MIR',
        '/COPY:DAT',
        '/DCOPY:DAT',
        '/R:2',
        '/W:1',
        '/XJ',
        '/FFT',
        '/NP',
        '/NFL',
        '/NDL'
    )

    Write-Log "Sync start: $Source -> $Target"
    & robocopy @arguments | Out-Null
    $exitCode = $LASTEXITCODE

    if ($exitCode -gt 7) {
        throw "Robocopy failed for $Source -> $Target with exit code $exitCode"
    }

    Write-Log "Sync complete: $Source -> $Target (robocopy exit $exitCode)"
}

function Wait-ForProcessExit {
    param([string[]]$Names)

    Write-Log "Waiting for processes to exit: $($Names -join ', ')"

    while ($true) {
        $running = Get-Process -ErrorAction SilentlyContinue | Where-Object {
            $Names -contains $_.ProcessName
        }

        if (-not $running) {
            break
        }

        $summary = $running | Sort-Object ProcessName, Id | ForEach-Object {
            "$($_.ProcessName)#$($_.Id)"
        }

        Write-Log "Still running: $($summary -join ', ')"
        Start-Sleep -Seconds 5
    }

    Write-Log 'Blocking processes are closed.'
}

function Test-ExpectedJunction {
    param(
        [string]$Path,
        [string]$ExpectedTarget
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $item = Get-Item -LiteralPath $Path -Force
    if ($item.LinkType -ne 'Junction') {
        return $false
    }

    $target = @($item.Target | ForEach-Object { $_.ToString() })
    return $target -contains $ExpectedTarget
}

function Convert-ToJunction {
    param(
        [string]$Source,
        [string]$Target
    )

    if (Test-ExpectedJunction -Path $Source -ExpectedTarget $Target) {
        Write-Log "Already migrated: $Source -> $Target"
        return
    }

    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Source path not found: $Source"
    }

    if (-not (Test-Path -LiteralPath $Target)) {
        throw "Target path not found: $Target"
    }

    $backup = '{0}.bak-{1}' -f $Source, (Get-Date -Format 'yyyyMMddHHmmss')
    Write-Log "Renaming source to backup: $Source -> $backup"
    Rename-Item -LiteralPath $Source -NewName (Split-Path -Leaf $backup)

    try {
        Write-Log "Creating junction: $Source -> $Target"
        New-Item -ItemType Junction -Path $Source -Target $Target -Force | Out-Null

        if (-not (Test-ExpectedJunction -Path $Source -ExpectedTarget $Target)) {
            throw "Verification failed for junction $Source -> $Target"
        }

        Write-Log "Removing backup after verified cutover: $backup"
        Remove-Item -LiteralPath $backup -Recurse -Force
    }
    catch {
        Write-Log "Cutover failed, restoring original directory for $Source"

        if (Test-Path -LiteralPath $Source) {
            $sourceItem = Get-Item -LiteralPath $Source -Force
            if ($sourceItem.LinkType -eq 'Junction') {
                Remove-Item -LiteralPath $Source -Force
            }
            else {
                Remove-Item -LiteralPath $Source -Recurse -Force
            }
        }

        Rename-Item -LiteralPath $backup -NewName (Split-Path -Leaf $Source)
        throw
    }
}

function Start-BackgroundFinalizer {
    param([string]$ScriptPath)

    Write-Log "Spawning background finalizer from $ScriptPath"

    $args = @(
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', $ScriptPath,
        '-Mode', 'Finalize',
        '-UserRoot', $UserRoot,
        '-TargetRoot', $TargetRoot,
        '-LogPath', $LogPath
    )

    $process = Start-Process -FilePath 'powershell.exe' -ArgumentList $args -WindowStyle Hidden -PassThru
    Write-Log "Background finalizer PID: $($process.Id)"
}

$items = Get-MigrationItems
Ensure-Directory -Path $TargetRoot

switch ($Mode) {
    'StageAndQueue' {
        foreach ($item in $items) {
            try {
                Invoke-Robocopy -Source $item.Source -Target $item.Target
            }
            catch {
                Write-Log "Stage sync incomplete for $($item.Source): $($_.Exception.Message)"
                Write-Log 'Deferring remaining copy work to the finalizer after processes close.'
            }
        }

        Start-BackgroundFinalizer -ScriptPath $PSCommandPath
        Write-Log 'Stage complete. Finalizer is queued and will cut over after Codex/cgc processes close.'
    }

    'Finalize' {
        Wait-ForProcessExit -Names @('Codex', 'codex', 'cgc')

        foreach ($item in $items) {
            Invoke-Robocopy -Source $item.Source -Target $item.Target
            Convert-ToJunction -Source $item.Source -Target $item.Target
        }

        Write-Log 'Finalize complete. Both storage paths now point to E: through junctions.'
    }
}
