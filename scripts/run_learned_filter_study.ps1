param([string]$WhisperRoot = "C:\Users\Kunal Munjal\Downloads\whisper-domain-adaptation")
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $WhisperRoot ".venv\Scripts\python.exe"
$config = Join-Path $WhisperRoot "configs\financial_finetune.yaml"
$validation = Join-Path $WhisperRoot "data\financial_research\validation_manifest.parquet"
$earnings = Join-Path $WhisperRoot "data\earnings21_eval\eval_manifest.parquet"
$slr = Join-Path $repoRoot "experiments\crawl_training_study\openslr31_speaker_disjoint_test.parquet"
$vocab = Join-Path $WhisperRoot "configs\financial_terms.txt"
$train = Join-Path $repoRoot "experiments\learned_filter_study\learned_filter_arm.parquet"
$results = Join-Path $repoRoot "experiments\results\learned_filter_study"
function Invoke-CheckedPython { & $python @args; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }
foreach ($seed in @(11, 22, 33, 44, 55)) {
  $checkpoint = Join-Path $repoRoot "checkpoints\learned_filter_study\seed_$seed"
  $adapter = Join-Path $checkpoint "adapter"
  if (-not (Test-Path (Join-Path $adapter "adapter_config.json"))) {
    Set-Location $WhisperRoot
    Invoke-CheckedPython scripts/run_finetune.py --config $config --train_manifest $train `
      --eval_manifest $validation --output_dir $checkpoint --seed $seed
  }
  foreach ($corpus in @("earnings21", "openslr31")) {
    $eval = if ($corpus -eq "earnings21") { $earnings } else { $slr }
    $output = Join-Path $results "learned\$corpus\seed_$seed.json"
    if (-not (Test-Path $output)) {
      Set-Location $WhisperRoot
      Invoke-CheckedPython scripts/evaluate_longform.py --adapter-path $adapter `
        --base-model openai/whisper-small --eval-manifest $eval --domain-vocab $vocab `
        --seed $seed --output $output
    }
  }
}
Set-Location $repoRoot
Invoke-CheckedPython scripts/summarize_learned_filter_study.py
