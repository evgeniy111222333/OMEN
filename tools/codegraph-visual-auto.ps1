param(
    [int]$Port = 8010,
    [int]$PollSeconds = 5,
    [int]$DebounceSeconds = 3,
    [switch]$ForceInitialRebuild
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"

$repoRoot = Split-Path -Parent $PSScriptRoot
$stateDir = Join-Path $repoRoot ".codegraph-visual-state"
$statePath = Join-Path $stateDir "controller.json"
$url = "http://localhost:$Port/index.html"

function Write-Status {
    param([string]$Message)
    Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $Message)
}

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

function Get-RgCommand {
    $rg = Get-Command rg -ErrorAction SilentlyContinue
    if ($null -ne $rg) {
        return $rg.Source
    }
    return $null
}

function Ensure-StateDir {
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
}

function Read-State {
    if (-not (Test-Path $statePath)) {
        return $null
    }

    try {
        return Get-Content $statePath -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Write-State {
    param(
        [string]$ActiveContext,
        [int]$LaunchPid,
        [int]$ServerPid,
        [string]$Fingerprint,
        [string]$Status
    )

    $payload = [ordered]@{
        controller_pid = $PID
        active_context = $ActiveContext
        launch_pid = $LaunchPid
        server_pid = $ServerPid
        port = $Port
        url = $url
        fingerprint = $Fingerprint
        status = $Status
        updated_at = (Get-Date).ToString("o")
    }

    $payload | ConvertTo-Json | Set-Content -Path $statePath -Encoding UTF8
}

function Remove-State {
    if (Test-Path $statePath) {
        Remove-Item $statePath -Force
    }
}

function Test-ProcessAlive {
    param([int]$ProcessId)

    if ($ProcessId -le 0) {
        return $false
    }

    try {
        Get-Process -Id $ProcessId -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Get-ListeningPid {
    param([int]$ListenPort)

    $match = netstat -ano | Select-String -Pattern "^\s*TCP\s+\S+:$ListenPort\s+\S+\s+LISTENING\s+(\d+)\s*$" | Select-Object -First 1
    if ($null -eq $match) {
        return $null
    }

    return [int]$match.Matches[0].Groups[1].Value
}

function Stop-KnownProcess {
    param([int]$ProcessId)

    if (-not (Test-ProcessAlive -ProcessId $ProcessId)) {
        return
    }

    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    }
    catch {
        Write-Status "Could not stop process $ProcessId automatically."
    }
}

function Cleanup-StaleRun {
    $state = Read-State
    if ($null -eq $state) {
        return
    }

    $controllerAlive = Test-ProcessAlive -ProcessId ([int]$state.controller_pid)
    if ($controllerAlive) {
        Write-Status "A CodeGraph visual controller is already running on $($state.url). Opening browser."
        Start-Process $state.url | Out-Null
        exit 0
    }

    if ($state.launch_pid) {
        Stop-KnownProcess -ProcessId ([int]$state.launch_pid)
    }

    if ($state.server_pid) {
        Stop-KnownProcess -ProcessId ([int]$state.server_pid)
    }

    Remove-State
}

function Ensure-Context {
    param([string]$ContextName)

    $args = @("context", "create", $ContextName, "--database", "kuzudb")
    try {
        & $script:cgc @args *> $null
    }
    catch {
        # Ignore errors here. The context may already exist, and CodeGraphContext
        # on Windows sometimes raises formatting/encoding errors after success.
    }
}

function Invoke-Index {
    param(
        [string]$ContextName,
        [switch]$Force
    )

    $args = @("index", $repoRoot, "--context", $ContextName)
    if ($Force) {
        $args += "--force"
    }

    Write-Status "Refreshing context '$ContextName'..."
    & $script:cgc @args
    if ($LASTEXITCODE -ne 0) {
        throw "Indexing failed for context '$ContextName'."
    }
}

function Wait-ForUrl {
    param(
        [string]$TargetUrl,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $TargetUrl -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                return $true
            }
        }
        catch {
        }

        Start-Sleep -Seconds 1
    }

    return $false
}

function Start-Visualizer {
    param([string]$ContextName)

    $visualScript = Join-Path $PSScriptRoot "codegraph-visualize.ps1"
    Write-Status "Starting visual UI on $url using context '$ContextName'..."
    $process = Start-Process -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $visualScript, "-Context", $ContextName, "-Port", "$Port") `
        -WorkingDirectory $repoRoot `
        -WindowStyle Minimized `
        -PassThru

    $ready = Wait-ForUrl -TargetUrl $url -TimeoutSeconds 60
    $listenPid = Get-ListeningPid -ListenPort $Port
    if (-not $ready -and $null -eq $listenPid) {
        Stop-KnownProcess -ProcessId $process.Id
        throw "Visualizer did not become ready on $url."
    }

    if (-not $ready -and $null -ne $listenPid) {
        Write-Status "Visualizer listener is up on port $Port. Continuing even though the HTTP probe did not return 200 within the initial wait window."
    }

    if ($null -eq $listenPid) {
        $listenPid = $process.Id
    }

    Write-Status "Visualizer is ready."
    return [pscustomobject]@{
        LaunchPid = $process.Id
        ListenPid = $listenPid
    }
}

function Stop-Visualizer {
    param($Process)

    if ($null -eq $Process) {
        return
    }

    if ($Process.ListenPid) {
        Stop-KnownProcess -ProcessId ([int]$Process.ListenPid)
    }

    if ($Process.LaunchPid -and $Process.LaunchPid -ne $Process.ListenPid) {
        Stop-KnownProcess -ProcessId ([int]$Process.LaunchPid)
    }
}

function Get-TrackedRelativePaths {
    $rg = Get-RgCommand
    if ($null -ne $rg) {
        Push-Location $repoRoot
        try {
            $paths = & $rg --files --hidden --glob "!.git" --glob "!.codegraph-visual-state" --glob "!tools/.codegraph-visual-state"
        }
        finally {
            Pop-Location
        }

        $extras = @(".cgcignore", ".gitignore", "AGENTS.md")
        foreach ($extra in $extras) {
            if (Test-Path (Join-Path $repoRoot $extra)) {
                $paths += $extra
            }
        }

        return $paths | Where-Object { $_ } | Sort-Object -Unique
    }

    $files = Get-ChildItem -Path $repoRoot -Recurse -File |
        Where-Object {
            $_.FullName -notlike "*\.git\*" -and
            $_.FullName -notlike "*\.codegraph-visual-state\*"
        }

    return $files | ForEach-Object {
        [System.IO.Path]::GetRelativePath($repoRoot, $_.FullName)
    } | Sort-Object -Unique
}

function Get-RepoFingerprint {
    $paths = Get-TrackedRelativePaths
    $count = 0
    $size = [int64]0
    $latest = [int64]0

    foreach ($relativePath in $paths) {
        $fullPath = Join-Path $repoRoot $relativePath
        if (-not (Test-Path $fullPath)) {
            continue
        }

        $item = Get-Item $fullPath
        $count += 1
        $size += [int64]$item.Length
        if ($item.LastWriteTimeUtc.Ticks -gt $latest) {
            $latest = $item.LastWriteTimeUtc.Ticks
        }
    }

    return "$count|$size|$latest"
}

$script:cgc = Get-CgcCommand
Ensure-StateDir
Cleanup-StaleRun

$sessionId = "{0:yyyyMMddHHmmss}-{1}" -f (Get-Date), $PID
$contexts = @(
    "omen-visual-$sessionId-a",
    "omen-visual-$sessionId-b"
)

$busyPid = Get-ListeningPid -ListenPort $Port
if ($null -ne $busyPid) {
    throw "Port $Port is already in use by PID $busyPid. Free that port or rerun with -Port."
}

$serverProcess = $null
$activeContext = $contexts[0]
$standbyContext = $contexts[1]

try {
    Write-Status "Preparing CodeGraph visual contexts..."
    foreach ($contextName in $contexts) {
        Ensure-Context -ContextName $contextName
    }

    Invoke-Index -ContextName $activeContext -Force:$ForceInitialRebuild
    $baselineFingerprint = Get-RepoFingerprint

    $serverProcess = Start-Visualizer -ContextName $activeContext
    Write-State -ActiveContext $activeContext -LaunchPid $serverProcess.LaunchPid -ServerPid $serverProcess.ListenPid -Fingerprint $baselineFingerprint -Status "running"

    Write-Status "Opening browser..."
    Start-Process $url | Out-Null

    Write-Status "Warming standby context '$standbyContext'..."
    Invoke-Index -ContextName $standbyContext -Force:$ForceInitialRebuild

    $pendingFingerprint = $null
    $pendingSince = $null

    Write-Status "Auto-refresh controller is active. Leave this window open while editing."
    Write-Status "Changes are picked up with debounce, refreshed incrementally, and then the UI is restarted safely."

    while ($true) {
        Start-Sleep -Seconds $PollSeconds

        $observedFingerprint = Get-RepoFingerprint
        if ($observedFingerprint -eq $baselineFingerprint) {
            $pendingFingerprint = $null
            $pendingSince = $null
            continue
        }

        if ($pendingFingerprint -ne $observedFingerprint) {
            $pendingFingerprint = $observedFingerprint
            $pendingSince = Get-Date
            Write-Status "Repository change detected. Waiting for edits to settle..."
            continue
        }

        if (((Get-Date) - $pendingSince).TotalSeconds -lt $DebounceSeconds) {
            continue
        }

        $nextContext = if ($activeContext -eq $contexts[0]) { $contexts[1] } else { $contexts[0] }
        Invoke-Index -ContextName $nextContext

        Stop-Visualizer -Process $serverProcess
        $serverProcess = Start-Visualizer -ContextName $nextContext

        $activeContext = $nextContext
        $baselineFingerprint = $pendingFingerprint
        $pendingFingerprint = $null
        $pendingSince = $null

        Write-State -ActiveContext $activeContext -LaunchPid $serverProcess.LaunchPid -ServerPid $serverProcess.ListenPid -Fingerprint $baselineFingerprint -Status "running"
        Write-Status "Visualization refreshed. Reload the browser if the page does not update automatically."
    }
}
finally {
    Stop-Visualizer -Process $serverProcess
    Remove-State
}
