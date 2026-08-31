param(
  [string]$WhisperRoot = "C:\Users\Kunal Munjal\Downloads\whisper-domain-adaptation"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $WhisperRoot ".venv\Scripts\python.exe"
$config = Join-Path $WhisperRoot "configs\financial_finetune.yaml"
$validation = Join-Path $WhisperRoot "data\financial_research\validation_manifest.parquet"
$earnings = Join-Path $WhisperRoot "data\earnings21_eval\eval_manifest.parquet"
$vocab = Join-Path $WhisperRoot "configs\financial_terms.txt"
$study = Join-Path $repoRoot "experiments\crawl_training_study"
$results = Join-Path $repoRoot "experiments\results\crawl_training_study"
$seeds = @(11, 22, 33, 44, 55)

function Invoke-CheckedPython { & $python @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }

foreach ($seed in $seeds) {
  $arms = @{
    "control" = Join-Path $WhisperRoot "checkpoints\financial_research\seed_$seed\adapter"
    "augmented" = Join-Path $repoRoot "checkpoints\crawl_training_study\augmented\seed_$seed\adapter"
  }
  $augmentedRoot = Split-Path -Parent $arms["augmented"]
  if (-not (Test-Path (Join-Path $arms["augmented"] "adapter_config.json"))) {
    Set-Location $WhisperRoot
    Invoke-CheckedPython scripts/run_finetune.py `
      --config $config `
      --train_manifest (Join-Path $study "augmented_50pct_crawler.parquet") `
      --eval_manifest $validation `
      --output_dir $augmentedRoot `
      --seed $seed
  }
  foreach ($arm in @("control", "augmented")) {
    foreach ($corpus in @("earnings21", "openslr31")) {
      $eval = if ($corpus -eq "earnings21") { $earnings } else { Join-Path $study "openslr31_speaker_disjoint_test.parquet" }
      $output = Join-Path $results "$arm\$corpus\seed_$seed.json"
      if (-not (Test-Path $output)) {
        Set-Location $WhisperRoot
        Invoke-CheckedPython scripts/evaluate_longform.py `
          --adapter-path $arms[$arm] `
          --base-model openai/whisper-small `
          --eval-manifest $eval `
          --domain-vocab $vocab `
          --seed $seed `
          --output $output
      }
    }
  }
}

Set-Location $repoRoot
Invoke-CheckedPython scripts/summarize_crawl_training_study.py
