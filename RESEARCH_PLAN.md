# Locked Research Plan

Status: locked on 2026-07-29. Changes require an explicit amendment documenting
the reason, date, and whether any results had already been observed.

## Research question

Can an audio-curation policy improve downstream ASR performance, and what is
the effect of targeted synthetic augmentation after controlling total training
size?

## Design

- Dataset versions, acquisition parameters, manifest hashes, and source
  licenses are recorded before analysis.
- Speaker-disjoint train, validation, and test partitions are created before
  quality thresholds are evaluated.
- Test data is immutable and is never filtered using an outcome observed on the
  test set.
- Curation policies are fitted or selected using training and validation only.
- Downstream effect is measured by training the same ASR recipe under each
  policy. Frozen-model WER and MFCC nearest-neighbor overlap are diagnostic
  measurements, not estimates of training benefit.
- Comparisons keep model initialization, optimization budget, training-set
  size, evaluation set, and seed paired where applicable.
- Trials use seeds 11, 22, 33, 44, and 55.
- WER results use 10,000 paired utterance- or speaker-cluster bootstrap
  resamples and report `N_trials=5` with 95% confidence intervals.

## Required validation

- Quality thresholds are evaluated against a manually labeled audit sample.
- Deduplication threshold and LSH recall are evaluated against labeled
  duplicate/nonduplicate pairs.
- Demographic findings report sample counts, missingness, and uncertainty.
- TTS voice metadata is never treated as observed human demographic identity.
- Synthetic augmentation is evaluated on untouched real speech.

## Reproducibility

Every result records the Git commit, dirty-worktree flag, exact arguments,
configuration hash, dataset hashes, dependency versions, seed, hardware,
predictions, and metrics. Documentation tables are generated from committed
artifacts by named scripts; hand-entered experimental numbers are prohibited.

## Acceptance criteria

No curation or augmentation policy is called beneficial or optimal unless its
paired 95% CI on the prespecified downstream metric excludes zero across five
trials. Proxy measurements may motivate hypotheses but cannot support causal
or downstream-performance claims.
