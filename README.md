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

The controlled [MFCC-LSH benchmark](experiments/results/dedup_validation/summary.md)
records 57% candidate recall (95% Wilson CI
47.2–66.3%) and 74.0% precision (63.3–82.5%) on identity-preserving
perturbations versus distinct-source negatives. It therefore supports the
current decision to keep automatic near-duplicate removal disabled. This is a
controlled identity benchmark, not human semantic-pair validation.

The committed evidence package contains 20 downstream ASR reports (four
conditions × five seeds), their run-provenance records, the fixed manifests,
and both machine-readable JSON and rendered Markdown summaries. Large model
weights and source audio remain excluded from Git.

The locked [open-corpus acquisition pilot](CRAWL_RESEARCH_PLAN.md) adds a
production-minded crawler stage using OpenSLR SLR31 Mini LibriSpeech. The
[generated data card](experiments/results/openslr31_pilot/DATA_CARD.md) records
the live robots policy, source-page and archive hashes, CC BY 4.0 attribution,
and the full funnel. From 1 crawled source page it found 2 unique archives,
acquired the preregistered 126 MB development archive, deterministically
inspected 250 of its 1,089 aligned clips, passed 203 through the unchanged
quality policy (81.2%), removed 0 exact duplicates, and retained 203. The 192
MFCC-LSH candidates were not removed because the repository's labeled benchmark
does not justify automatic near-duplicate deletion.

The paired [frozen-ASR audit](experiments/results/openslr31_pilot/ASR_SELECTION_AUDIT.md)
finds essentially neutral selection effects: retained-minus-all WER is +0.06
points (95% clip-bootstrap CI -0.20 to +0.30) for base Whisper-small and +0.12
points (-0.15 to +0.40) for the seed-11 financial adapter (`N_trials=1`). On
all 250 clips the adapter regresses by +0.73 points (+0.19 to +1.32) relative
to base. SLR31 contains none of the configured financial terms, so domain WER
is undefined and common WER equals overall WER. This is a descriptive
selection/robustness audit, not evidence about retraining on crawled data.

That open question is now addressed by a separate, prospectively locked
[five-seed training study](CRAWL_TRAINING_PLAN.md). The control retains all 294
financial examples; the matched-size intervention replaces half with 147
quality-retained SLR31 clips from four training speakers and evaluates on two
untouched SLR31 speakers plus real Earnings-21. The generated
[training report](experiments/results/crawl_training_study/REPORT.md) rejects
the beneficial claim: held-out SLR31 WER worsens from 4.43% to 5.60%, a paired
+1.17-point change (95% seed-bootstrap CI +0.88 to +1.46; `N_trials=5`). The
regression occurs in all five seeds. Earnings-21 moves from 11.47% to 10.83%
(-0.64 points, -1.43 to +0.21), which is inconclusive overall; its common slice
improves by -0.64 points (-0.95 to -0.32), while its domain interval crosses
zero. A plausible explanation is specialization to the four crawler training
speakers or read-speech styles rather than speaker-general robustness, but the
study did not preregister a mechanism test, so that remains an inference.

The experimental [learned quality filter](LEARNED_FILTER_PLAN.md) was then
tested as an alternative to, not a replacement for, the production heuristic.
It predicts the heuristic's weak pass/fail labels from lightweight audio
features; it is not trained on human quality or downstream-utility labels.
Leave-one-speaker-out development predictions agreed on 205/206 clips (99.51%;
balanced accuracy 98.84%), and the two untouched speakers agreed on all 44
clips. Silence fraction accounts for 95.45% of fitted feature importance. The
only disagreement remains explicitly pending human listening, so no human
preference claim is made.

After a prospective pre-ASR amendment removed an unrelated sampling confound,
the learned and heuristic crawler selections overlap on 146/147 clips. The
[five-seed downstream report](experiments/results/learned_filter_study/REPORT.md)
finds Earnings-21 overall WER of 11.07% versus 10.83% for the heuristic arm,
a paired +0.24-point change (95% seed-bootstrap CI -0.42 to +0.72;
`N_trials=5`). Held-out SLR31 WER is 5.38% versus 5.60%, a -0.21-point change
(-0.72 to +0.45). Both intervals cross zero, the locked replacement gate fails,
and the heuristic remains the default. This demonstrates recovery of the
existing decision boundary, not improved audio quality.

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

Run the bounded, license-gated acquisition pilot (static HTML only; no browser
automation) with:

```bash
python scripts/crawl_openslr.py
```

It refuses a source-page identity/license mismatch, checks `robots.txt` on
every host, honors the greater of the declared crawl delay or five seconds,
uses atomic bounded downloads with official-mirror failover, safely extracts
tar members, and emits the existing Parquet manifest contract. Raw audio and
archives are ignored by Git; compact reports and per-clip ASR predictions are
the reviewable evidence.

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

## Regenerating reported tables

No README result is hand-calculated. Regenerate the controlled deduplication
table directly from its committed JSON report:

```bash
python scripts/render_dedup_validation.py
```

Regenerate the downstream five-seed table from the committed per-seed
predictions, using the adjacent Whisper repository for the shared WER analyzer:

```bash
python -m pip install -e ../whisper-domain-adaptation
python scripts/summarize_downstream_study.py \
  --whisper-root ../whisper-domain-adaptation
```

Regenerate the crawler selection audit from its committed GPU predictions:

```bash
python scripts/summarize_crawl_asr.py \
  --base experiments/results/openslr31_pilot/base_raw.json \
  --adapted experiments/results/openslr31_pilot/financial_seed11_raw.json \
  --filtered-manifest data/openslr31_pilot/curated/filtered_manifest.parquet \
  --output experiments/results/openslr31_pilot/asr_selection_audit.json \
  --markdown experiments/results/openslr31_pilot/ASR_SELECTION_AUDIT.md
```

Regenerate the five-seed crawler-training table from all 20 committed
model/corpus prediction reports:

```bash
python scripts/summarize_crawl_training_study.py
```

Regenerate the learned-filter weak-label artifacts and its five-seed downstream
comparison with:

```bash
python scripts/run_learned_filter.py \
  --manifest data/openslr31_pilot/raw_manifest.parquet \
  --heuristic-arm experiments/crawl_training_study/augmented_50pct_crawler.parquet
python scripts/summarize_learned_filter_study.py
```

## License

Code is released under the MIT License. Input datasets retain their upstream
licenses and must be reviewed before redistribution.
