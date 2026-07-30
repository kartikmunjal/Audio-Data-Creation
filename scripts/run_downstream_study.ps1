param(
  [string]$WhisperRoot = (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "whisper-domain-adaptation")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $WhisperRoot ".venv\Scripts\python.exe"
$config = Join-Path $WhisperRoot "configs\financial_finetune.yaml"
$evalManifest = Join-Path $WhisperRoot "data\earnings21_eval\eval_manifest.parquet"
$vocab = Join-Path $WhisperRoot "configs\financial_terms.txt"
$validation = Join-Path $repoRoot "experiments\downstream_study\validation.parquet"

function Invoke-CheckedPython {
  & $python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE"
  }
}

$arms = @(
  "curation_control",
  "curation_policy",
  "augmentation_control_common",
  "augmentation_targeted_50pct"
)
foreach ($arm in $arms) {
  $train = Join-Path $repoRoot "experiments\downstream_study\$arm.parquet"
  foreach ($seed in @(11, 22, 33, 44, 55)) {
    $checkpoint = Join-Path $repoRoot "checkpoints\downstream_study\$arm\seed_$seed"
    $result = Join-Path $repoRoot "experiments\results\downstream_study\$arm\seed_$seed.json"
    if (-not (Test-Path "$checkpoint\adapter\adapter_config.json")) {
      Set-Location $WhisperRoot
      Invoke-CheckedPython scripts/run_finetune.py `
        --config $config `
        --train_manifest $train `
        --eval_manifest $validation `
        --output_dir $checkpoint `
        --seed $seed
    }
    if (-not (Test-Path $result)) {
      Set-Location $WhisperRoot
      Invoke-CheckedPython scripts/evaluate_longform.py `
        --adapter-path "$checkpoint\adapter" `
        --base-model openai/whisper-small `
        --eval-manifest $evalManifest `
        --domain-vocab $vocab `
        --seed $seed `
        --output $result
    }
  }
}

Set-Location $repoRoot
Invoke-CheckedPython scripts/summarize_downstream_study.py `
  --whisper-root $WhisperRoot
