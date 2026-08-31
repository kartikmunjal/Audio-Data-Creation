# Locked Crawler-Augmentation Training Study

Protocol locked 2026-08-31 before observing any augmented-model result.

## Question

Does replacing half of a fixed-size financial fine-tuning set with licensed,
quality-retained OpenSLR speech improve out-of-domain read-speech WER without
degrading real financial-call WER?

## Arms and fixed training recipe

- Control: the unchanged financial-research training manifest.
- Augmented: the same total number of rows, comprising 50% financial rows and
  50% OpenSLR rows. Rows are selected once with seed 20260831 and held fixed
  across training trials. If an arm lacks enough rows, preparation fails rather
  than sampling with replacement.
- Both arms use `configs/financial_finetune.yaml`, the same financial validation
  manifest for checkpoint selection, five epochs, and seeds 11, 22, 33, 44, 55.
- The already-completed control checkpoints may be reused only when their
  manifest/config provenance matches the locked financial study.

Equal arm size holds optimizer updates approximately fixed and tests a data-
composition intervention rather than confounding composition with more compute.

## Leakage controls and evaluation

The 203 retained SLR31 clips are partitioned by speaker before training:

- augmentation speakers: 1272, 1462, 174, 1988;
- untouched read-speech test speakers: 1993, 2035.

No test-speaker clip can enter either training arm. Checkpoint selection uses
only the pre-existing financial validation set. Each seed is evaluated on:

1. the untouched real Earnings-21 manifest for overall/domain/common WER; and
2. the 40-clip speaker-disjoint SLR31 test manifest for overall/common WER.

SLR31 domain WER is expected to be undefined if no configured financial term
occurs; it will be reported as missing, never imputed.

## Estimand and claims

The primary estimands are augmented-minus-control WER for Earnings-21 overall
and SLR31 overall, paired by seed. Secondary estimands are Earnings-21 domain
and common WER. Every estimate reports `N_trials=5` and a 10,000-resample paired
seed-bootstrap 95% CI.

The intervention is called beneficial only if the SLR31 overall CI is entirely
below zero and the Earnings-21 overall CI does not exceed +1.0 percentage point
at its upper bound. Otherwise the result is neutral, mixed, or adverse. No
mixture ratio, speaker split, threshold, epoch count, or seed is changed after
results are observed.
