[![CI](https://github.com/kartikmunjal/Audio-Data-Creation/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmunjal/Audio-Data-Creation/actions/workflows/ci.yml)

# Audio Data Curation

A reproducible audio-corpus inspection pipeline covering signal-quality
diagnostics, exact and candidate near-duplicate discovery, and descriptive
diversity reporting.

## Research status

The software pipeline is implemented, but downstream ASR-benefit claims are not
currently verified. Previous example files reported ratio-specific WER and a
“50% synthetic optimum,” although the checked-in evaluator did not train a
model on each mixture. Those values have been withdrawn.

The locked validation protocol is in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).
Frozen-model WER and MFCC nearest-neighbor overlap are diagnostic measurements;
neither is evidence that training on a curation policy improves ASR.

## Pipeline

```text
input manifest
  -> audio loading and signal diagnostics
  -> configurable quality policy
  -> exact audio hashing
  -> MFCC-LSH candidate near-duplicate discovery
  -> descriptive coverage report
  -> filtered manifest plus provenance
```

MFCC-LSH output must be interpreted as candidate pairs, not verified semantic
duplicates, until precision and recall are measured on a labeled audit set.
Reference-free SNR is also a heuristic and is reported as an estimate.

## Quickstart

Use Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/download_sample.py \
  --n_samples 500 \
  --split validation \
  --output_dir data/raw

python scripts/run_pipeline.py \
  --manifest data/raw/manifest.parquet \
  --output_dir outputs
```

Primary outputs:

- `outputs/filtered_manifest.parquet`
- `outputs/curation_report.json`
- per-clip diagnostic fields used to audit each decision

## Research-grade acceptance boundary

No quality threshold, deduplication threshold, or synthetic-data ratio will be
called beneficial or optimal until:

- the quality policy is compared with manually labeled clips;
- deduplication precision and recall are measured on labeled pairs;
- speaker-disjoint train/validation/test partitions are fixed first;
- each curation policy trains the same downstream ASR recipe;
- five paired trials are complete; and
- WER differences carry `N_trials=5` and paired 95% confidence intervals.

Demographic metadata is reported with missingness. Metadata attached to a TTS
voice is not treated as observed human demographic identity.

## Tests

```bash
pytest -q
```

## License

Code is released under the MIT License. Input datasets retain their upstream
licenses and must be reviewed before redistribution.
