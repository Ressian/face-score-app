param(
    [string]$Version = "2.1.0"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if (-not (Test-Path "dist\颜值打分系统_v2\颜值打分系统_v2.exe")) {
    throw "Build output not found. Run scripts\build_windows.ps1 first."
}

$Out = Join-Path $Root "release-assets"
New-Item -ItemType Directory -Path $Out -Force | Out-Null

$AppZip = Join-Path $Out "beauty-score-app-v$Version-windows-x64.zip"
$WeightsZip = Join-Path $Out "weights_v2-v$Version.zip"

if (Test-Path $AppZip) { Remove-Item -LiteralPath $AppZip -Force }
if (Test-Path $WeightsZip) { Remove-Item -LiteralPath $WeightsZip -Force }

Compress-Archive -Path "dist\颜值打分系统_v2" -DestinationPath $AppZip -CompressionLevel Optimal
Compress-Archive -Path "weights_v2" -DestinationPath $WeightsZip -CompressionLevel Optimal

$Hashes = @()
foreach ($File in @($AppZip, $WeightsZip)) {
    $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $File).Hash
    $Hashes += "$Hash  $([System.IO.Path]::GetFileName($File))"
}
$Hashes | Set-Content -Encoding ASCII (Join-Path $Out "SHA256SUMS.txt")

Write-Host "Release assets written to $Out"

