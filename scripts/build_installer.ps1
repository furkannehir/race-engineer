[CmdletBinding()]
param(
    [string]$MainPython = ".\.venv\Scripts\python.exe",
    [string]$Compiler = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

if (-not $Compiler) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    $Compiler = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (-not $Compiler -or -not (Test-Path -LiteralPath $Compiler -PathType Leaf)) {
    throw "Inno Setup 6 compiler was not found"
}

$isPythonPath = $MainPython.Contains("\") -or $MainPython.Contains("/")
if (($isPythonPath -and -not (Test-Path -LiteralPath $MainPython -PathType Leaf)) -or
    (-not $isPythonPath -and -not (Get-Command $MainPython -ErrorAction SilentlyContinue))) {
    throw "Required build Python is missing: $MainPython"
}

$version = & $MainPython -c "from race_engineer.version import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0) { throw "Could not read Pitward version" }
& $Compiler "/DAppVersion=$version" packaging\Pitward.iss
if ($LASTEXITCODE -ne 0) { throw "Installer build failed" }

$checksumPath = Join-Path $repoRoot "dist\SHA256SUMS.txt"
$sums = @()
if (Test-Path -LiteralPath $checksumPath) {
    $sums = @(Get-Content -LiteralPath $checksumPath | Where-Object { $_ -notmatch "Pitward-Setup-" })
}
Get-ChildItem -LiteralPath (Join-Path $repoRoot "dist") -Filter "Pitward-Setup-*.exe" |
    ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $sums += "$hash  $($_.Name)"
    }
$sums | Set-Content -LiteralPath $checksumPath -Encoding utf8
Write-Output "Built Pitward $version installer."
