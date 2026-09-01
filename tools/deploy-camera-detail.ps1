$ErrorActionPreference = 'Stop'

$root = 'C:\Deep-Live-Cam'
$stage = Join-Path $root 'runtime\detector-detail-stage'
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$backup = Join-Path $root ("runtime\backup-before-camera-detail-$stamp")
$taskName = 'DeepLiveCamNetwork'
$relativeFiles = @(
    'modules\globals.py',
    'modules\live_processor.py',
    'modules\remote_control.py',
    'modules\processors\frame\face_swapper.py',
    'modules\processors\frame\frequency_repair.py'
)

New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($relative in $relativeFiles) {
    $source = Join-Path $root $relative
    $staged = Join-Path $stage $relative
    if (-not (Test-Path $staged -PathType Leaf)) {
        throw "staged file is missing: $staged"
    }
    $saved = Join-Path $backup $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $saved) | Out-Null
    Copy-Item -Path $source -Destination $saved
}

Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddSeconds(15)
do {
    Start-Sleep -Milliseconds 250
    $old = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'run_network.py|run-network-service.cmd'
    }
} while ($old -and (Get-Date) -lt $deadline)
if ($old) {
    $old | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

foreach ($relative in $relativeFiles) {
    Copy-Item -Force -Path (Join-Path $stage $relative) -Destination (Join-Path $root $relative)
}
Start-ScheduledTask -TaskName $taskName

$deadline = (Get-Date).AddSeconds(90)
$health = $null
do {
    Start-Sleep -Seconds 1
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/healthz' -TimeoutSec 2
    } catch {
        $health = $null
    }
} while (($null -eq $health -or $health.state -notmatch 'streaming') -and (Get-Date) -lt $deadline)
if ($null -eq $health -or $health.state -notmatch 'streaming') {
    throw 'Windows processing service did not return to streaming state'
}

$config = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/api/config' -TimeoutSec 3
$devices = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/api/devices' -TimeoutSec 3
if ($devices.selected_device_id -ne 'arch-webcam') {
    throw "route changed unexpectedly to $($devices.selected_device_id)"
}
if ($null -eq $config.repair_camera_detail) {
    throw 'deployed API does not expose repair_camera_detail'
}
if ($null -eq $config.repair_boundary_strength) {
    throw 'deployed API does not expose repair_boundary_strength'
}

[pscustomobject]@{
    backup = $backup
    task = (Get-ScheduledTask -TaskName $taskName).State
    state = $health.state
    selected = $devices.selected_device_id
    repair_camera_detail = $config.repair_camera_detail
    repair_boundary_strength = $config.repair_boundary_strength
} | ConvertTo-Json -Compress
