<#
Create a reproducible conda environment for arm_bci_system.py.

Default behavior:
  - creates/updates a conda env
  - installs signal processing, machine learning, GUI, and 3D physics dependencies
  - detects the NVIDIA driver and installs a matching GPU PyTorch build unless -SkipTorch is used
  - verifies imports, the MI8 paradigm, and arm_bci_system.py compilation

Usage:
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1

Optional:
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1 -Force
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1 -TorchBackend cu128
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1 -TorchBackend cpu
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1 -SkipTorch
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1 -PipIndexUrl https://pypi.org/simple
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1 -UseClassicSolver
  powershell -ExecutionPolicy Bypass -File .\setup_conda_env.ps1 -EnvName bciwork_1 -CondaChannel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

After setup:
  conda activate bciwork_1
  python .\arm_bci_system.py
#>

param(
    [string]$EnvName = "bciwork_1",
    [string]$PythonVersion = "3.10",
    [string]$CondaChannel = "conda-forge",
    [ValidateSet("auto", "cu128", "cu126", "cu118", "cpu")]
    [string]$TorchBackend = "auto",
    [string]$PipIndexUrl = "https://pypi.org/simple",
    [switch]$SkipTorch,
    [switch]$Force,
    [switch]$UseClassicSolver
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Command
}

function Compare-VersionAtLeast {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VersionText,
        [Parameter(Mandatory = $true)]
        [string]$MinimumText
    )

    try {
        return ([version]$VersionText -ge [version]$MinimumText)
    }
    catch {
        return $false
    }
}

function Get-NvidiaSmiPath {
    $nvidiaSmiCommand = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmiCommand) {
        return $nvidiaSmiCommand.Source
    }

    $candidates = @(
        "$env:SystemRoot\System32\nvidia-smi.exe",
        "C:\Windows\System32\nvidia-smi.exe",
        "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }

    return ($candidates | Select-Object -First 1)
}

function Get-NvidiaDriverVersion {
    $nvidiaSmi = Get-NvidiaSmiPath
    if (-not $nvidiaSmi) {
        return $null
    }

    $driverVersionText = & $nvidiaSmi --query-gpu=driver_version --format=csv,noheader 2>$null |
        Select-Object -First 1

    if (-not $driverVersionText) {
        return $null
    }

    $driverVersion = $driverVersionText.Trim()

    if ($driverVersion) {
        Write-Host "Detected NVIDIA driver: $driverVersion" -ForegroundColor Cyan
    }

    return $driverVersion
}

function Get-MinimumDriverForTorchBackend {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Backend
    )

    switch ($Backend) {
        "cu128" { return "570.65" }
        "cu126" { return "560.76" }
        "cu118" { return "522.06" }
        default { return $null }
    }
}

function Resolve-TorchBackend {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RequestedBackend
    )

    if ($RequestedBackend -ne "auto") {
        $minimumDriver = Get-MinimumDriverForTorchBackend -Backend $RequestedBackend
        if ($minimumDriver) {
            $driverVersion = Get-NvidiaDriverVersion
            if (-not $driverVersion) {
                throw "TorchBackend $RequestedBackend was requested, but nvidia-smi was not found. Install an NVIDIA driver or run with -TorchBackend cpu."
            }
            if (-not (Compare-VersionAtLeast $driverVersion $minimumDriver)) {
                throw "TorchBackend $RequestedBackend requires NVIDIA driver $minimumDriver or newer, but detected $driverVersion."
            }
        }

        return $RequestedBackend
    }

    $driverVersion = Get-NvidiaDriverVersion
    if (-not $driverVersion) {
        Write-Host "No NVIDIA driver was detected. Falling back to CPU PyTorch." -ForegroundColor Yellow
        return "cpu"
    }

    if (Compare-VersionAtLeast $driverVersion "570.65") {
        return "cu128"
    }
    if (Compare-VersionAtLeast $driverVersion "560.76") {
        return "cu126"
    }
    if (Compare-VersionAtLeast $driverVersion "522.06") {
        return "cu118"
    }

    Write-Host "NVIDIA driver $driverVersion is too old for the supported GPU PyTorch wheels. Falling back to CPU PyTorch." -ForegroundColor Yellow
    return "cpu"
}

$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCommand) {
    $CondaExe = $condaCommand.Source
}
else {
    $condaCandidates = @(
        $env:CONDA_EXE,
        "F:\anaconda\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    ) | Where-Object { $_ -and (Test-Path $_) }

    $CondaExe = $condaCandidates | Select-Object -First 1
}

if (-not $CondaExe) {
    throw "conda was not found. Please install Miniconda or Anaconda, then reopen Anaconda Prompt or PowerShell."
}

$condaSolverArgs = @()
if (-not $UseClassicSolver) {
    $condaInstallHelp = & $CondaExe install --help 2>$null | Out-String
    if ($condaInstallHelp -match "--solver") {
        $condaSolverArgs = @("--solver", "libmamba")
    }
}

$condaChannelArgs = @("--override-channels", "-c", $CondaChannel)
$resolvedTorchBackend = "skipped"
$torchIndexUrl = $null
if (-not $SkipTorch) {
    $resolvedTorchBackend = Resolve-TorchBackend -RequestedBackend $TorchBackend
    $torchIndexUrls = @{
        "cu128" = "https://download.pytorch.org/whl/cu128"
        "cu126" = "https://download.pytorch.org/whl/cu126"
        "cu118" = "https://download.pytorch.org/whl/cu118"
        "cpu" = "https://download.pytorch.org/whl/cpu"
    }
    $torchIndexUrl = $torchIndexUrls[$resolvedTorchBackend]
}
$pipIndexArgs = @()
if ($PipIndexUrl) {
    $pipIndexArgs = @("-i", $PipIndexUrl)
}

if ($SkipTorch) {
    Write-Host "PyTorch installation: skipped (-SkipTorch). arm_bci_system.py does not require Torch." -ForegroundColor Cyan
}
else {
    Write-Host "PyTorch backend: $resolvedTorchBackend" -ForegroundColor Cyan
}

$condaEnvList = & $CondaExe env list
$envExists = $condaEnvList | Select-String -Pattern "^\s*$([regex]::Escape($EnvName))\s+"

if ($envExists -and $Force) {
    Invoke-Step "Removing existing conda environment: $EnvName" {
        & $CondaExe env remove -n $EnvName -y
    }
    $envExists = $false
}

if (-not $envExists) {
    Invoke-Step "Creating conda environment: $EnvName" {
        & $CondaExe create @condaSolverArgs @condaChannelArgs -n $EnvName "python=$PythonVersion" pip -y
    }
}
else {
    Write-Host "Conda environment '$EnvName' already exists. Use -Force to recreate it." -ForegroundColor Yellow
}

Invoke-Step "Upgrading pip build tools" {
    & $CondaExe run -n $EnvName python -m pip install --upgrade pip setuptools wheel
}

Invoke-Step "Installing PyBullet physics engine" {
    & $CondaExe install @condaSolverArgs @condaChannelArgs -n $EnvName -y pybullet=3.25
}

Invoke-Step "Installing Python runtime packages" {
    & $CondaExe run -n $EnvName python -m pip install @pipIndexArgs `
        numpy `
        scipy `
        scikit-learn `
        mne `
        joblib `
        PyQt6 `
        pyqtgraph `
        PyOpenGL
}

if (-not $SkipTorch) {
    Invoke-Step "Installing PyTorch ($resolvedTorchBackend)" {
        & $CondaExe run -n $EnvName python -m pip uninstall -y torch torchvision torchaudio
        & $CondaExe run -n $EnvName python -m pip install torch torchvision torchaudio --index-url $torchIndexUrl
    }
}
else {
    Write-Host ""
    Write-Host "==> Skipping PyTorch installation" -ForegroundColor Cyan
}

$verifyScript = @'
import importlib
import os
import py_compile
from pathlib import Path

required_modules = [
    "numpy",
    "scipy",
    "sklearn",
    "mne",
    "joblib",
    "PyQt6",
    "pybullet",
    "pyqtgraph",
    "OpenGL",
]

if os.environ.get("EXPECT_TORCH") == "1":
    required_modules.append("torch")

for module_name in required_modules:
    importlib.import_module(module_name)
    print(f"OK: {module_name}")

if os.environ.get("EXPECT_TORCH") == "1":
    import torch
    print(f"OK: torch version = {torch.__version__}")
    print(f"OK: torch CUDA runtime = {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"OK: torch.cuda.is_available() = {cuda_available}")
    if cuda_available:
        print(f"OK: CUDA GPU = {torch.cuda.get_device_name(0)}")
    elif os.environ.get("EXPECT_CUDA") == "1":
        raise SystemExit("ERROR: GPU PyTorch was installed, but torch cannot see a CUDA GPU.")
else:
    print("OK: torch verification skipped")

system_file = Path("arm_bci_system.py")
if not system_file.exists():
    raise SystemExit("ERROR: arm_bci_system.py was not found beside setup_conda_env.ps1.")

py_compile.compile(str(system_file), doraise=True)
print("OK: arm_bci_system.py compiles")

from arm_bci_system import GRIP_TOGGLE_LABEL, MI8_PARADIGM, SignalConfig, HybridBCIModel

movement_labels = [label for label in MI8_PARADIGM if label not in {"rest", GRIP_TOGGLE_LABEL}]
if len(movement_labels) != 6:
    raise SystemExit(f"ERROR: expected 6 active masseter+MI movement labels, got {len(movement_labels)}.")
if GRIP_TOGGLE_LABEL not in MI8_PARADIGM:
    raise SystemExit("ERROR: expected triple-blink grip-toggle paradigm entry.")

config = SignalConfig()
if abs(config.calibration_trial_seconds - 4.0) > 1e-9:
    raise SystemExit("ERROR: expected default collection/execution window to be 4.0 seconds.")

print(f"OK: movement labels = {', '.join(movement_labels)}")
print(f"OK: grip toggle label = {GRIP_TOGGLE_LABEL}")
print("OK: default collection/execution window = 4.0s")

print("Environment verification finished.")
'@

Invoke-Step "Verifying robotic arm BCI imports" {
    $env:EXPECT_TORCH = if ($SkipTorch) { "0" } else { "1" }
    $env:EXPECT_CUDA = if ($SkipTorch -or $resolvedTorchBackend -eq "cpu") { "0" } else { "1" }
    try {
        Push-Location $ProjectRoot
        $verifyScript | & $CondaExe run -n $EnvName python -
    }
    finally {
        Pop-Location
        Remove-Item Env:\EXPECT_TORCH -ErrorAction SilentlyContinue
        Remove-Item Env:\EXPECT_CUDA -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
if ($SkipTorch) {
    Write-Host "PyTorch backend: skipped" -ForegroundColor Green
}
else {
    Write-Host "PyTorch backend: $resolvedTorchBackend" -ForegroundColor Green
}
Write-Host "Activate the environment with:"
Write-Host "  conda activate $EnvName" -ForegroundColor Green
Write-Host "Run the robotic arm BCI app with:"
Write-Host "  python .\arm_bci_system.py" -ForegroundColor Green
Write-Host "Or open the GUI with:"
Write-Host "  python .\arm_bci_system.py gui" -ForegroundColor Green
