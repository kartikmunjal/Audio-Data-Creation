[![CI](https://github.com/kartikmunjal/Audio-Data-Creation/actions/workflows/ci.yml/badge.svg)](https://github.com/kartikmunjal/Audio-Data-Creation/actions/workflows/ci.yml)

# Audio Data Curation

A reproducible audio-corpus inspection pipeline covering signal-quality
diagnostics, exact and candidate near-duplicate discovery, and descriptive
diversity reporting.

## Research status

The software pipeline and the preregistered downstream ASR experiment are
complete. The generated
[five-seed report](experiments/results/downstream_study/summary.md) compares two
paired interventions on the same real Earnings-21 evaluation set:

- The named curation policy is neutral: overall ΔWER is +0.02 percentage
  points (95% trial-bootstrap CI -0.02 to +0.05; `N_trials=5`).
- Replacing half of a 24-clip training set with targeted financial utterances
  changes overall WER by -0.05 points (-0.09 to 0.00). The apparent change is
  confined to the eight-clip common-control slice; domain WER is unchanged.

These small, boundary-touching effects do not support a beneficial or optimal
policy claim. Previous example files that called a “50% synthetic optimum”
without training every mixture were withdrawn. The locked protocol and claim
rules are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md); all reported values regenerate
from primary per-seed predictions with 10,000 paired bootstrap resamples.

The controlled MFCC-LSH benchmark records 57% candidate recall (95% Wilson CI
47.2–66.3%) and 74.0% precision (63.3–82.5%) on identity-preserving
perturbations versus distinct-source negatives. It therefore supports the
current decision to keep automatic near-duplicate removal disabled. This is a
controlled identity benchmark, not human semantic-pair validation.

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

The downstream five-seed requirement is now satisfied, and its null/near-null
result is reported above. No quality threshold, deduplication threshold, or
synthetic-data ratio will be called beneficial or optimal unless:

- the quality policy is compared with manually labeled clips;
- deduplication precision and recall are measured on labeled pairs;
- speaker-disjoint train/validation/test partitions are fixed first;
- each curation policy trains the same downstream ASR recipe; and
- WER differences carry paired 95% confidence intervals.

The manual listening-quality ledger remains pending genuine human labels. The
repository does not infer those labels from signal heuristics or fabricate
them. Until it is completed, the downstream comparison evaluates a named
machine policy, not validated human-perceived audio quality.

Demographic metadata is reported with missingness. Metadata attached to a TTS
voice is not treated as observed human demographic identity.

## Tests

```bash
pytest -q
```

## License

Code is released under the MIT License. Input datasets retain their upstream
licenses and must be reviewed before redistribution.
