param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path "weights_v2\scut\best.pth")) {
    throw "weights_v2 not found. Download weights_v2-v2.1.0.zip from GitHub Releases and extract it into the project root."
}

& $Python -m PyInstaller --noconfirm --clean "颜值打分系统_v2.spec"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

Write-Host "Build complete: dist\颜值打分系统_v2\颜值打分系统_v2.exe"

