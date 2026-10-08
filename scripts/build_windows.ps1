[CmdletBinding()]
param(
    [string]$MainPython = ".\.venv\Scripts\python.exe",
    [string]$SttPython = ".\data\stt-prototype\runtime\Scripts\python.exe",
    [string]$TtsPython = ".\data\tts-prototype\runtime\Scripts\python.exe",
    [string]$Tag = "",
    [switch]$SkipWorkers
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot
$packageRoot = Join-Path $repoRoot "build\package"
$metadataRoot = Join-Path $packageRoot "metadata"
$distRoot = Join-Path $packageRoot "dist"
$workRoot = Join-Path $packageRoot "work"
$releaseRoot = Join-Path $repoRoot "dist"

$requiredPythons = @($MainPython)
if (-not $SkipWorkers) {
    $requiredPythons += @($SttPython, $TtsPython)
}
foreach ($python in $requiredPythons) {
    $isPath = $python.Contains("\") -or $python.Contains("/")
    if (($isPath -and -not (Test-Path -LiteralPath $python -PathType Leaf)) -or
        (-not $isPath -and -not (Get-Command $python -ErrorAction SilentlyContinue))) {
        throw "Required build Python is missing: $python"
    }
}

$prepare = @("scripts\prepare_release.py", "--output", $metadataRoot)
if ($Tag) {
    $prepare += @("--tag", $Tag)
}
& $MainPython @prepare
if ($LASTEXITCODE -ne 0) { throw "Could not prepare build metadata" }
$env:PITWARD_VERSION_FILE = Join-Path $metadataRoot "windows-version.txt"
$version = & $MainPython -c "from race_engineer.version import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0) { throw "Could not read Pitward version" }

& $MainPython -m PyInstaller --noconfirm --clean --distpath $distRoot `
    --workpath (Join-Path $workRoot "app") packaging\Pitward.spec
if ($LASTEXITCODE -ne 0) { throw "Pitward application build failed" }

& $MainPython -m PyInstaller --noconfirm --clean --distpath $distRoot `
    --workpath (Join-Path $workRoot "components") packaging\PitwardComponents.spec
if ($LASTEXITCODE -ne 0) { throw "Pitward component installer build failed" }

$appRoot = Join-Path $distRoot "Pitward"
Copy-Item -LiteralPath (Join-Path $distRoot "PitwardComponents.exe") -Destination $appRoot
Copy-Item -LiteralPath (Join-Path $metadataRoot "build-info.json") -Destination $appRoot
Copy-Item -LiteralPath "LICENSE" -Destination $appRoot
Copy-Item -LiteralPath "distribution\README.txt" -Destination $appRoot
Copy-Item -LiteralPath "distribution\THIRD_PARTY_NOTICES.md" -Destination $appRoot
Copy-Item -LiteralPath "src\race_engineer\assets\components.v1.json" -Destination $appRoot

if (-not $SkipWorkers) {
    & $SttPython -m PyInstaller --noconfirm --clean --distpath $distRoot `
        --workpath (Join-Path $workRoot "stt") packaging\PitwardSTTWorker.spec
    if ($LASTEXITCODE -ne 0) { throw "Pitward STT worker build failed" }
    & $TtsPython -m PyInstaller --noconfirm --clean --distpath $distRoot `
        --workpath (Join-Path $workRoot "tts") packaging\PitwardTTSWorker.spec
    if ($LASTEXITCODE -ne 0) { throw "Pitward TTS worker build failed" }
    $workers = Join-Path $appRoot "workers"
    New-Item -ItemType Directory -Path $workers -Force | Out-Null
    Move-Item -LiteralPath (Join-Path $distRoot "PitwardSTTWorker") -Destination $workers
    Move-Item -LiteralPath (Join-Path $distRoot "PitwardTTSWorker") -Destination $workers
}

New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null
$archive = Join-Path $releaseRoot "Pitward-$version-windows-x64.zip"
Compress-Archive -LiteralPath $appRoot -DestinationPath $archive -CompressionLevel Optimal -Force
$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $(Split-Path -Leaf $archive)" | Set-Content `
    -LiteralPath (Join-Path $releaseRoot "SHA256SUMS.txt") -Encoding utf8
Write-Output "Built $archive"
