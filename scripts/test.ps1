param(
    [string]$Python = "python",
    [string]$Config = "configs/test_div2k.yaml",
    [string]$Checkpoint = "runs/nusesr_x4_k8/best.pt",
    [string]$HrDir,
    [string]$Output,
    [switch]$SaveImages
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Checkpoint not found: $Checkpoint"
    }

    $TestArgs = @("test.py", "--config", $Config, "--checkpoint", $Checkpoint)
    if ($HrDir) { $TestArgs += @("--hr-dir", $HrDir) }
    if ($Output) { $TestArgs += @("--output", $Output) }
    if ($SaveImages) { $TestArgs += "--save-images" }
    & $Python @TestArgs
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
