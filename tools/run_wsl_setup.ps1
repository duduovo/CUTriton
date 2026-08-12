[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$installedOutput = & wsl.exe --list --quiet 2>$null
$installed = @($installedOutput | ForEach-Object { ($_ -replace "`0", "").Trim() })
if ($installed -notcontains $Distribution) {
    throw "$Distribution is not installed. Run tools\install_wsl.ps1 as Administrator, then restart."
}

$escapedRepository = $repository.Replace('\', '\\')
$wslPathOutput = & wsl.exe --distribution $Distribution -- wslpath -a $escapedRepository
if ($LASTEXITCODE -ne 0 -or $null -eq $wslPathOutput) {
    throw "Could not convert the repository path. Launch $Distribution once and finish user setup first."
}
$wslRepository = ([string]$wslPathOutput).Trim()
if ([string]::IsNullOrWhiteSpace($wslRepository)) {
    throw "WSL returned an empty repository path."
}

Write-Host "Configuring CUTriton in $Distribution at $wslRepository"
& wsl.exe --distribution $Distribution -- bash "$wslRepository/tools/setup_wsl_cuda.sh"
if ($LASTEXITCODE -ne 0) {
    throw "WSL environment setup failed with exit code $LASTEXITCODE."
}
