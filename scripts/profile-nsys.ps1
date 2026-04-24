param(
    [string]$Scene = "scenes/customscene2.txt",
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [ValidateSet("full", "hotkey")]
    [string]$Mode = "hotkey",
    [string]$OutputName = "",
    [switch]$OpenReport
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$nsysExe = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe"
$nsysUiExe = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\host-windows-x64\nsys-ui.exe"
$appExe = Join-Path $repoRoot "build\bin\$Config\cis565_path_tracer.exe"
$scenePath = Join-Path $repoRoot $Scene
$profilesDir = Join-Path $repoRoot "profiles"

if (-not (Test-Path $nsysExe)) {
    throw "Nsight Systems CLI not found at '$nsysExe'."
}

if (-not (Test-Path $appExe)) {
    throw "Application executable not found at '$appExe'. Build the $Config configuration first."
}

if (-not (Test-Path $scenePath)) {
    throw "Scene file not found at '$scenePath'."
}

New-Item -ItemType Directory -Force -Path $profilesDir | Out-Null

if ([string]::IsNullOrWhiteSpace($OutputName)) {
    $sceneBase = [System.IO.Path]::GetFileNameWithoutExtension($scenePath)
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputName = "nsys-$sceneBase-$Config-$Mode-$timestamp"
}

$outputPath = Join-Path $profilesDir $OutputName

$nsysArgs = @(
    "profile"
    "--force-overwrite=true"
    "--trace=cuda,nvtx,opengl"
    "--sample=none"
    "--cpuctxsw=none"
    "--cuda-memory-usage=false"
    "--stats=true"
    "--output=$outputPath"
)

if ($Mode -eq "hotkey") {
    $nsysArgs += @(
        "--capture-range=hotkey"
        "--capture-range-end=stop"
        "--hotkey-capture=F12"
        "--kill=false"
    )
}
else {
    $nsysArgs += @(
        "--kill=false"
    )
}

$nsysArgs += @(
    $appExe
    $scenePath
)

Push-Location $repoRoot
try {
    Write-Host "Launching Nsight Systems profile..."
    Write-Host "  Scene : $scenePath"
    Write-Host "  Config: $Config"
    Write-Host "  Mode  : $Mode"
    Write-Host "  Output: $outputPath.nsys-rep"
    if ($Mode -eq "hotkey") {
        Write-Host "Press F12 inside the app window to start capture, then F12 again to stop."
        Write-Host "Close the app window after the report finishes writing."
    }
    else {
        Write-Host "Capture starts immediately and runs until you close the app window."
    }

    & $nsysExe @nsysArgs

    if ($OpenReport) {
        $reportPath = "$outputPath.nsys-rep"
        if (Test-Path $reportPath -and (Test-Path $nsysUiExe)) {
            Start-Process -FilePath $nsysUiExe -ArgumentList $reportPath | Out-Null
        }
    }
}
finally {
    Pop-Location
}
