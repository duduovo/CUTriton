[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu",
    [string]$StorageRoot = "G:\Ubuntu_"
)

$ErrorActionPreference = "Stop"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    throw "Run PowerShell as Administrator, then execute this script again."
}

$os = Get-CimInstance Win32_OperatingSystem
$build = [int]$os.BuildNumber
if ($build -lt 19041) {
    throw "WSL2 requires Windows 10 build 19041 or newer. Current build: $build."
}

Write-Host "Windows build: $build"
Write-Host "Target distribution: $Distribution"

$StorageRoot = [IO.Path]::GetFullPath($StorageRoot).TrimEnd('\')
$InstallLocation = Join-Path $StorageRoot "Distro"
if (-not (Test-Path -LiteralPath $StorageRoot)) {
    New-Item -ItemType Directory -Path $StorageRoot | Out-Null
}

# WSL keeps its global config in the Windows profile, but the large swap VHD
# is redirected to the requested data drive.
$swapPath = (Join-Path $StorageRoot "wsl-swap.vhdx").Replace('\', '\\')
$wslConfigPath = Join-Path $env:USERPROFILE ".wslconfig"
if (Test-Path -LiteralPath $wslConfigPath) {
    $wslConfig = Get-Content -Raw -LiteralPath $wslConfigPath
    if ($wslConfig -match '(?im)^\s*swapFile\s*=.*$') {
        $wslConfig = $wslConfig -replace '(?im)^\s*swapFile\s*=.*$', "swapFile=$swapPath"
    } elseif ($wslConfig -match '(?im)^\s*\[wsl2\]\s*$') {
        $wslConfig = $wslConfig -replace '(?im)^(\s*\[wsl2\]\s*)$', "`$1`r`nswapFile=$swapPath"
    } else {
        $wslConfig = $wslConfig.TrimEnd() + "`r`n`r`n[wsl2]`r`nswapFile=$swapPath`r`n"
    }
} else {
    $wslConfig = "[wsl2]`r`nswapFile=$swapPath`r`n"
}
Set-Content -LiteralPath $wslConfigPath -Value $wslConfig -Encoding ASCII

$installedOutput = & wsl.exe --list --quiet 2>$null
$installed = @($installedOutput | ForEach-Object { ($_ -replace "`0", "").Trim() })

if ($installed -contains $Distribution) {
    Write-Host "$Distribution is already installed."
} else {
    Write-Host "Installing WSL2 and $Distribution at $InstallLocation ..."
    & wsl.exe --install --web-download --distribution $Distribution `
        --location $InstallLocation --no-launch
    if ($LASTEXITCODE -ne 0) {
        throw "wsl --install failed with exit code $LASTEXITCODE. Make sure Windows Update is current."
    }
}

& wsl.exe --set-default-version 2
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Could not set the default WSL version yet. Run this script again after reboot."
}

& wsl.exe --update --web-download
if ($LASTEXITCODE -ne 0) {
    Write-Warning "WSL update did not complete. Run 'wsl --update' after reboot."
}

Write-Host ""
Write-Host "WSL installation stage completed. Next steps:"
Write-Host "1. Restart Windows."
Write-Host "2. Open $Distribution from the Start menu and create a Linux user."
Write-Host "3. Run tools\run_wsl_setup.ps1 in a regular PowerShell window."
