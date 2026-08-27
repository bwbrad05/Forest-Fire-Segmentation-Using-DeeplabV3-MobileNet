<#
.SYNOPSIS
    Create the CUDA conda environment for this project and verify it.

.DESCRIPTION
    Order matters here, and getting it wrong is the failure this project has
    already hit twice (see the notes at the top of requirements-gpu.txt):

      1. conda provides Python and nothing else. Conda-installed scientific
         packages pull in a second OpenMP runtime next to torch's, and the
         process then aborts with "OMP: Error #15". Keep the stack pure pip.
      2. torch/torchvision are installed FIRST, from the CUDA index. If
         requirements-gpu.txt is installed first, pip resolves a CPU-only wheel
         from PyPI and the GPU sits idle with no error message.
      3. rasterio is deliberately absent. Its pip wheel bundles its own GDAL,
         which corrupts the heap next to conda's copy (silent 0xC0000374 exit).
         The pipeline reads pixel arrays only and falls back to tifffile, which
         returns bit-identical data.

    This environment adds tensorboard on top of the old one, so training curves
    can be watched live rather than only plotted after the fact.

.PARAMETER EnvName
    Conda environment name (default firenet-gpu).

.PARAMETER Cuda
    CUDA wheel tag matching your driver (default cu124). See https://pytorch.org.

.PARAMETER Python
    Python version (default 3.11).

.PARAMETER Force
    Remove an existing environment of the same name first.

.EXAMPLE
    .\scripts\setup_gpu.ps1
    Create firenet-gpu with cu124 wheels.

.EXAMPLE
    .\scripts\setup_gpu.ps1 -Cuda cu121 -Force
    Recreate it against CUDA 12.1.
#>

[CmdletBinding()]
param(
    [string]$EnvName = "firenet-gpu",
    [string]$Cuda = "cu124",
    [string]$Python = "3.11",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda not found on PATH. Open an 'Anaconda PowerShell Prompt', or run `conda init powershell` once and reopen the shell."
}

# --------------------------------------------------------------------------- #
# 1. Environment
# --------------------------------------------------------------------------- #
$existing = conda env list | Select-String -Pattern "^\s*$([regex]::Escape($EnvName))\s"
if ($existing) {
    if ($Force) {
        Write-Host "Removing existing environment '$EnvName'..." -ForegroundColor Yellow
        conda env remove -n $EnvName -y
    } else {
        throw "Environment '$EnvName' already exists. Re-run with -Force to recreate it, or just 'conda activate $EnvName'."
    }
}

Write-Host "`n[1/4] Creating conda environment '$EnvName' (python $Python)..." -ForegroundColor Cyan
conda create -n $EnvName "python=$Python" -y
if ($LASTEXITCODE -ne 0) { throw "conda create failed." }

# Resolve the env's python.exe directly. Calling it by absolute path avoids
# depending on 'conda activate' working inside a non-interactive script.
$envPath = (conda env list | Select-String -Pattern "^\s*$([regex]::Escape($EnvName))\s+\*?\s*(.+)$").Matches.Groups[1].Value.Trim()
$envPython = Join-Path $envPath "python.exe"
if (-not (Test-Path $envPython)) { throw "Could not locate python.exe in $envPath" }
Write-Host "  python: $envPython" -ForegroundColor DarkGray

# --------------------------------------------------------------------------- #
# 2. torch FIRST, from the CUDA index
# --------------------------------------------------------------------------- #
Write-Host "`n[2/4] Installing torch + torchvision ($Cuda) -- this is the ~2.5 GB download..." -ForegroundColor Cyan
& $envPython -m pip install --upgrade pip
& $envPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) {
    throw "torch install failed. Check that '$Cuda' matches your driver -- see https://pytorch.org."
}

# --------------------------------------------------------------------------- #
# 3. Everything else, from PyPI
# --------------------------------------------------------------------------- #
Write-Host "`n[3/4] Installing requirements-gpu.txt..." -ForegroundColor Cyan
& $envPython -m pip install -r requirements-gpu.txt
if ($LASTEXITCODE -ne 0) { throw "requirements-gpu.txt install failed." }

# --------------------------------------------------------------------------- #
# 4. Verify — all four of these must pass before starting a multi-hour run
# --------------------------------------------------------------------------- #
Write-Host "`n[4/4] Verifying..." -ForegroundColor Cyan

$verify = @'
import sys
ok = True

import torch
print(f"  python      : {sys.executable}")
print(f"  torch       : {torch.__version__}")
print(f"  torch.cuda  : {torch.version.cuda}")
print(f"  available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device      : {torch.cuda.get_device_name(0)}")
else:
    print("  ERROR: CUDA not available -- this is a CPU wheel. Reinstall from the CUDA index.")
    ok = False

import pytorch_lightning, timm, segmentation_models_pytorch, tifffile, polars
print(f"  lightning   : {pytorch_lightning.__version__}")
print(f"  timm        : {timm.__version__}")
print(f"  smp         : {segmentation_models_pytorch.__version__}")

try:
    import tensorboard
    print(f"  tensorboard : {tensorboard.__version__}")
except ImportError:
    print("  WARNING: tensorboard missing -- runs will log to CSV only.")

# The modules added in Part F must import and build.
from neural_net import DeepLabV3PlusMobileViT
m = DeepLabV3PlusMobileViT(backbone="mobilevit_xxs", in_channels=8, n_classes=2,
                           encoder_weights=None, strip_pooling=True, attention="ca")
n = sum(p.numel() for p in m.parameters())
print(f"  SP+CA model : builds, {n/1e6:.3f} M parameters")

sys.exit(0 if ok else 1)
'@

$verifyFile = Join-Path $env:TEMP "firenet_verify.py"
Set-Content -Path $verifyFile -Value $verify -Encoding utf8
& $envPython $verifyFile
$verifyExit = $LASTEXITCODE
Remove-Item $verifyFile -ErrorAction SilentlyContinue

if ($verifyExit -ne 0) {
    throw "Verification failed -- see above. Do not start a training run until CUDA is available."
}

Write-Host @"

Environment '$EnvName' is ready.

Next:
  conda activate $EnvName
  python scripts/sanity_and_splits.py --root data/indonesia     # confirm the split
  .\scripts\run_ablation.ps1 -Only abl_baseline,abl_sp_ca       # the claim + its reference

Watch it live, from a second terminal:
  conda activate $EnvName
  tensorboard --logdir lightning_logs
"@ -ForegroundColor Green
