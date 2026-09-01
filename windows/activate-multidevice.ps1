$ErrorActionPreference = 'Stop'
$taskName = 'DeepLiveCamNetwork'
$root = 'C:\Deep-Live-Cam'
$stage = 'C:\Deep-Live-Cam\runtime\multidevice-stage'
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$backup = Join-Path "$root\runtime" ('backup-before-multidevice-' + $stamp)
$files = @(
    'run_network.py',
    'run-network-service.cmd',
    'run.py',
    'modules\device_slots.py',
    'modules\network_router.py',
    'modules\live_stream.py',
    'modules\live_processor.py',
    'modules\remote_control.py',
    'modules\globals.py',
    'modules\face_analyser.py',
    'modules\face_tracking.py',
    'modules\quality_pipeline.py',
    'modules\pipeline_benchmark.py',
    'modules\swapper_contract.py',
    'modules\instyle256_swapper.py',
    'modules\simswap512_swapper.py',
    'modules\processors\frame\face_swapper.py',
    'modules\processors\frame\frequency_repair.py',
    'modules\processors\frame\boundary_repair.py',
    'tools\stability_report.py'
)
New-Item -ItemType Directory -Force -Path $backup | Out-Null

foreach ($relative in $files) {
    $source = Join-Path $root $relative
    if (Test-Path $source) {
        $destination = Join-Path $backup $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
        Copy-Item $source $destination
    }
}

Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddSeconds(12)
do {
    Start-Sleep -Milliseconds 250
    $old = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'run_network.py|run-network-service.cmd'
    }
} while ($old -and (Get-Date) -lt $deadline)

if ($old) {
    # Task Scheduler can leave Python grandchildren orphaned. The task is
    # already stopped; these matches are restricted to this camera service.
    $old | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'ffmpeg.exe' -and
        $_.CommandLine -match 'srt://0\.0\.0\.0:1000[02468]|srt://192\.168\.1\.(11|12):1000[13579]|(?:srt://0\.0\.0\.0|239\.255\.77\.77):10010'
    } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

foreach ($relative in $files) {
    $destination = Join-Path $root $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    Copy-Item -Force (Join-Path $stage $relative) $destination
}

if (-not (Get-NetFirewallRule -DisplayName 'Deep-Live-Cam device slots' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule `
        -DisplayName 'Deep-Live-Cam device slots' `
        -Direction Inbound `
        -Action Allow `
        -Protocol UDP `
        -LocalPort 10000-10009 `
        -Profile Private | Out-Null
}

if (-not (Get-NetFirewallRule -DisplayName 'Deep-Live-Cam selected stream' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule `
        -DisplayName 'Deep-Live-Cam selected stream' `
        -Direction Inbound `
        -Action Allow `
        -Protocol UDP `
        -LocalPort 10010 `
        -Profile Private | Out-Null
}

if (-not (Get-NetFirewallRule -DisplayName 'Deep-Live-Cam native control' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule `
        -DisplayName 'Deep-Live-Cam native control' `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 8090 `
        -Profile Private | Out-Null
}

Start-ScheduledTask -TaskName $taskName
$deadline = (Get-Date).AddSeconds(75)
do {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/healthz' -TimeoutSec 2
    } catch {
        $health = $null
    }
} while (($null -eq $health -or $null -eq $health.selected_device_id) -and (Get-Date) -lt $deadline)

if ($null -eq $health -or $null -eq $health.selected_device_id) {
    throw 'five-slot health did not become ready'
}

# Activation establishes Android slot 0 as the requested live route.  The
# selection is persisted by the service, but remains freely changeable later
# through the manager/API without restarting any camera endpoint.
$androidSelection = @{
    device_id = 'android-phone'
} | ConvertTo-Json
Invoke-RestMethod `
    -Method Post `
    -Uri 'http://127.0.0.1:8090/api/devices/select' `
    -ContentType 'application/json' `
    -Body $androidSelection | Out-Null

# Use the balanced native-256 generator only when its exact offline asset is
# installed. A partial/corrupt provision leaves the proven 128px path active.
$instyleModel = Join-Path $root 'models\InStyleSwapper256_Version_B.fp16.onnx'
$instyleReady = (
    (Test-Path $instyleModel) -and
    ((Get-Item $instyleModel).Length -eq 277295431) -and
    ((Get-FileHash -Algorithm SHA256 $instyleModel).Hash.ToLower() -eq '0870b6c75eaea239bdd72b6c6d0910cb285310736e356c17a2cd67a961738116')
)
$selectedSwapper = if ($instyleReady) { 'instyle-256' } else { 'inswapper-128' }

# Model activation must not overwrite the user's current repair, tracking, or
# output settings. ControlState persists and restores those independently.
$modelConfig = @{
    swapper_model = $selectedSwapper
} | ConvertTo-Json
Invoke-RestMethod `
    -Method Post `
    -Uri 'http://127.0.0.1:8090/api/config' `
    -ContentType 'application/json' `
    -Body $modelConfig | Out-Null
Start-Sleep -Milliseconds 500
$health = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/healthz' -TimeoutSec 3

[pscustomobject]@{
    backup = $backup
    task = (Get-ScheduledTask -TaskName $taskName).State
    health = $health
} | ConvertTo-Json -Depth 8 -Compress
