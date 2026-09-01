# Locked Learned-Filter Extension

Protocol locked before fitting the classifier or observing learned-filter results.

## Claim boundary

The target is the existing `QualityFilter` pass/fail decision. These are weak
labels, not human judgments and not downstream-utility labels. The experiment
asks whether a small interpretable model can recover the heuristic boundary and
whether its disagreements matter downstream. The heuristic remains the default.

## Data and leakage controls

- Corpus: the same 250 deterministic SLR31 pilot clips.
- Classifier-development speakers: 1272, 1462, 174, 1988.
- Untouched classifier/ASR test speakers: 1993, 2035.
- Development predictions used for ASR selection are leave-one-speaker-out:
  no clip is classified by a model trained on its speaker.
- The full development-speaker model is evaluated once on the two untouched
  speakers. Those speakers never enter either ASR training arm.

## Features and fixed model

Features are duration, reference-free SNR, silence fraction, clipping flag,
RMS, MFCC mean/std statistics, spectral-flatness mean/std, and zero-crossing
mean/std. The locked classifier is scikit-learn `GradientBoostingClassifier`
with 100 estimators, learning rate 0.05, depth 2, seed 20260831, balanced sample
weights, and a fixed 0.5 probability threshold. No hyperparameter search occurs.

Feature importance, confusion matrices, accuracy, balanced accuracy, precision,
recall, F1, and agreement are reported. A direct-feature model is expected to
approximate the hand-coded conjunction; it is not evidence of superior quality.

## Human disagreement ledger

Up to 30 disagreements are selected deterministically, prioritizing untouched
speakers. `human_acceptable` and notes remain blank until a person listens.
No model-generated or heuristic-generated value may be presented as a human
label. The incomplete ledger does not block the weak-label or downstream study.

## Downstream comparison

The prior heuristic arm is reused exactly: 147 fixed financial rows plus 147
heuristic-pass crawler rows. The learned arm uses the identical 147 financial
rows and 147 leave-one-speaker-out learned-pass crawler rows, with identical
total size, validation data, Whisper-small LoRA recipe, and seeds 11, 22, 33,
44, 55. If fewer than 147 learned-pass rows exist, the study fails rather than
sampling with replacement or changing the mixture.

Both arms are evaluated on real Earnings-21 and the same 40 untouched SLR31
clips. Primary estimands are learned-minus-heuristic overall WER on each corpus,
with 10,000-resample paired seed-bootstrap 95% CIs and `N_trials=5`.

The learned filter earns replacement consideration only if its SLR31 overall CI
is entirely below zero and its Earnings-21 upper bound is at most +1.0 point.
Otherwise the heuristic remains the production default. No threshold, feature,
speaker split, mixture, seed, or claim gate changes after results are observed.

## Prospective execution amendment

Recorded after classifier evaluation but before any learned-filter ASR training
or result. The classifier produced one disagreement, but independently sampling
147 pass clips caused a 32-clip arm difference unrelated to that disagreement.
To isolate the filter decision, the learned arm now starts from the fixed
heuristic arm and applies only decision-required swaps: each learned-only clip
is inserted and an equal number of heuristic-selected clips is removed by
descending SHA-256 of clip ID; any heuristic-only selected clip is removed and
replaced by the lowest-SHA-256 shared learned-pass clip not already selected.
The financial rows, arm size, seeds, recipe, evaluation, and claim gate are
unchanged. The amendment was motivated by causal identifiability, not downstream
performance, which remained unseen.
