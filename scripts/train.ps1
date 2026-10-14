param(
    [string]$Python = "python",
    [string]$Config = "configs/train_div2k.yaml",
    [string]$PcaConfig = "configs/pca_div2k.yaml",
    [string]$Basis = "basis/avgpool_x4_rank8.pt",
    [string]$ResumeFrom,
    [switch]$FitPca
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if ($FitPca) {
        & $Python fit_pca.py --config $PcaConfig --output $Basis
        if ($LASTEXITCODE -ne 0) { throw "PCA fitting failed with exit code $LASTEXITCODE" }
    }
    elseif (-not (Test-Path -LiteralPath $Basis -PathType Leaf)) {
        throw "PCA basis not found: $Basis. Run with -FitPca or fit it separately."
    }

    $TrainArgs = @("train.py", "--config", $Config, "--basis", $Basis)
    if ($ResumeFrom) { $TrainArgs += @("--resume", $ResumeFrom) }
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
