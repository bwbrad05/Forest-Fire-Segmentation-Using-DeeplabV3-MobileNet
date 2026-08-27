<#
.SYNOPSIS
    Run the Enhanced-DeepLabV3+ ablation sweep (Part F of CHANGES_SINCE_MEETING.md).

.DESCRIPTION
    Trains one variant per row of the ablation table, each into its own
    lightning_logs/<run_name>/ subtree, then builds the comparison table and
    figure. `mode=train` already runs the test phase against the best
    checkpoint, so every variant produces test_iou / test_f1 without a second
    command.

    Variants are ordered by what they buy you, not by module count:
      abl_baseline      the reference every delta is measured against
      abl_sp_ca         the paper's claim (+6.28 mIoU) -- run this second
      abl_ca            explains the claim: CA alone
      abl_sp            explains the claim: SP alone (expected flat or DOWN --
                        the paper's own Table 4 shows SP alone hurts)
      abl_cbam          CBAM vs CA head to head (Part E.3 vs Part F.4)
      abl_sp_ca_decca   + the decoder's dynamic weight allocation
      abl_warmup        their LR schedule, orthogonal to the modules

    Already-finished runs are skipped, so the sweep is safe to interrupt and
    restart. Delete a run's directory to force it to re-run.

.PARAMETER Only
    Run just these variants, e.g. -Only abl_baseline,abl_sp_ca

.PARAMETER Epochs
    Epoch budget per run (default 100, matching Part D).

.PARAMETER Fold
    Held-out test fold (default 0, matching Part D).

.PARAMETER Model
    Model config name (default deeplabv3plus_mobilevit_xxs).

.PARAMETER DryRun
    Print the commands without running them.

.EXAMPLE
    .\scripts\run_ablation.ps1
    Full sweep, 7 runs.

.EXAMPLE
    .\scripts\run_ablation.ps1 -Only abl_baseline,abl_sp_ca
    Just the claim and its reference -- start here if GPU time is short.

.EXAMPLE
    .\scripts\run_ablation.ps1 -Epochs 30 -DryRun
    Check what would run.
#>

[CmdletBinding()]
param(
    [string[]]$Only,
    [int]$Epochs = 100,
    [int]$Fold = 0,
    [string]$Model = "deeplabv3plus_mobilevit_xxs",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Repo root is this script's parent directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# --------------------------------------------------------------------------- #
# The ablation table. Add a row here to add an experiment.
# --------------------------------------------------------------------------- #
$Variants = @(
    @{ Name = "abl_baseline";    Overrides = @() ;
       Desc = "current best configuration, no new modules" },

    @{ Name = "abl_sp_ca";       Overrides = @("model.strip_pooling=true", "model.attention=ca") ;
       Desc = "THE CLAIM: strip pooling + coordinate attention (+6.28 mIoU in the paper)" },

    @{ Name = "abl_ca";          Overrides = @("model.attention=ca") ;
       Desc = "coordinate attention alone" },

    @{ Name = "abl_sp";          Overrides = @("model.strip_pooling=true") ;
       Desc = "strip pooling alone -- expected flat or down, see Part F.4" },

    @{ Name = "abl_cbam";        Overrides = @("model.attention=cbam") ;
       Desc = "CBAM alone -- head to head against CA" },

    @{ Name = "abl_sp_ca_decca"; Overrides = @("model.strip_pooling=true", "model.attention=ca", "model.decoder_attention=ca") ;
       Desc = "full structural package incl. decoder dynamic weighting" },

    @{ Name = "abl_warmup";      Overrides = @("model.scheduler=warmup_cosine") ;
       Desc = "warm-up + cosine LR (their eq. 21), no architecture change" }
)

if ($Only) {
    $Variants = $Variants | Where-Object { $Only -contains $_.Name }
    if (-not $Variants) {
        throw "No variant matched -Only. Valid names: abl_baseline, abl_sp_ca, abl_ca, abl_sp, abl_cbam, abl_sp_ca_decca, abl_warmup"
    }
}

# --------------------------------------------------------------------------- #
# Preflight: fail now, not three hours in.
# --------------------------------------------------------------------------- #
Write-Host "`n=== Preflight ===" -ForegroundColor Cyan

# Skipped under -DryRun so the plan can be reviewed on a laptop with no GPU.
if (-not $DryRun) {
    $gpuCheck = python -c "import torch, sys; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  torch reports: $gpuCheck" -ForegroundColor Red
        throw "CUDA is not available. Activate the GPU env (conda activate firenet-gpu) or reinstall torch from the CUDA index -- see scripts/setup_gpu.ps1."
    }
    Write-Host "  torch / cuda / available : $gpuCheck" -ForegroundColor Green
} else {
    Write-Host "  GPU check                : skipped (-DryRun)" -ForegroundColor DarkGray
}

$splits = Join-Path $RepoRoot "data\indonesia\splits.parquet"
if (-not (Test-Path $splits)) {
    throw "Missing $splits. Create it first:`n  python scripts/sanity_and_splits.py --root data/indonesia --create"
}
Write-Host "  splits.parquet           : found" -ForegroundColor Green
Write-Host "  model / epochs / fold    : $Model / $Epochs / $Fold"
Write-Host "  variants to run          : $($Variants.Count)"

# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
$startedAll = Get-Date
$ran = @()
$skipped = @()

foreach ($v in $Variants) {
    $runDir = Join-Path $RepoRoot "lightning_logs\$($v.Name)"

    # A run counts as finished if any metrics.csv under it records a test_iou.
    $done = $false
    if ((Test-Path $runDir) -and (-not $Force)) {
        $csvs = Get-ChildItem -Path $runDir -Filter metrics.csv -Recurse -ErrorAction SilentlyContinue
        foreach ($c in $csvs) {
            if (Select-String -Path $c.FullName -Pattern "test_iou" -Quiet -ErrorAction SilentlyContinue) {
                $done = $true; break
            }
        }
    }
    if ($done) {
        Write-Host "`n--- $($v.Name): already complete, skipping (use -Force to redo)" -ForegroundColor DarkGray
        $skipped += $v.Name
        continue
    }

    $cmdArgs = @(
        "main.py",
        "mode=train",
        "model=$Model",
        "dataset=indonesia",
        "dataset.test_fold=$Fold",
        "trainer.max_epochs=$Epochs",
        "run_name=$($v.Name)",
        "~logger"                 # local TensorBoard + CSV instead of Comet
    ) + $v.Overrides

    Write-Host "`n=== $($v.Name) ===" -ForegroundColor Cyan
    Write-Host "  $($v.Desc)" -ForegroundColor Gray
    Write-Host "  python $($cmdArgs -join ' ')" -ForegroundColor DarkGray

    if ($DryRun) { continue }

    $started = Get-Date
    python @cmdArgs
    $exit = $LASTEXITCODE
    $elapsed = (Get-Date) - $started

    if ($exit -ne 0) {
        Write-Host "  FAILED after $($elapsed.ToString('hh\:mm\:ss')) (exit $exit)" -ForegroundColor Red
        Write-Host "  Continuing with the remaining variants; re-run this one later." -ForegroundColor Yellow
    } else {
        Write-Host "  done in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        $ran += $v.Name
    }
}

if ($DryRun) {
    Write-Host "`nDry run -- nothing executed.`n" -ForegroundColor Yellow
    exit 0
}

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
Write-Host "`n=== Comparison ===" -ForegroundColor Cyan
python scripts/compare_runs.py --runs "lightning_logs/abl_*" --baseline abl_baseline

$total = (Get-Date) - $startedAll
Write-Host "`nSweep finished in $($total.ToString('hh\:mm\:ss'))." -ForegroundColor Green
if ($skipped) { Write-Host "  skipped (already complete): $($skipped -join ', ')" -ForegroundColor DarkGray }
Write-Host @"

Look at the results:
  lightning_logs\comparison\ablation_summary.md    table, ready to paste
  lightning_logs\comparison\ablation_curves.png    val IoU / loss, all runs overlaid
  tensorboard --logdir lightning_logs              interactive, all runs

Per-run curves:
  python scripts/plot_curves.py --run lightning_logs\abl_sp_ca\version_0

Read abl_sp_ca against abl_baseline first -- that pair is the paper's claim.
abl_sp alone being flat or lower is the expected result, not a bug (Part F.4).
"@ -ForegroundColor Cyan
