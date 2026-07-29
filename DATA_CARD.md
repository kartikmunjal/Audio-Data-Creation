# Data Card Template: Audio Curation Run

This file defines required disclosures. Run-specific values must be generated
from the input manifest and `curation_report.json`; hand-entered experimental
counts or performance values are prohibited.

## Required run identity

- Source dataset, release, locale, and upstream license
- Acquisition command and timestamp
- Input manifest SHA-256
- Git commit and dirty-worktree flag
- Random seed and full threshold configuration
- Raw, accepted, rejected, and failed-to-load counts

## Required partition audit

- Speaker-disjoint train/validation/test counts
- Speaker overlap checks
- Exact audio-hash overlap checks
- Normalized transcript overlap checks
- Missingness for speaker, age, gender, accent, and locale metadata

## Required quality validation

Reference-free SNR, silence, clipping, and duration are diagnostic estimates.
Any selected threshold must report precision, recall, and a 95% confidence
interval against a manually labeled audit sample.

## Required deduplication validation

Exact hashes identify byte-normalized duplicates. MFCC-LSH output is a candidate
set only. A run claiming near-duplicate removal must provide a labeled-pair
benchmark, LSH candidate recall, final precision/recall, threshold selection
procedure, and 95% confidence intervals.

## Required downstream validation

Frozen-ASR WER and acoustic nearest-neighbor overlap cannot measure training
benefit. A downstream claim requires matched ASR training under every policy,
five paired trials, an untouched real-speech test set, `N_trials=5`, and paired
95% confidence intervals.

## Demographic interpretation

Report observed metadata and missingness without inferring identity. TTS voice
catalog labels are not evidence of human demographic representation.
