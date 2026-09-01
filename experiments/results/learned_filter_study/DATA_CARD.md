# Experimental learned-filter data card

## Intended use

This experimental classifier approximates the repository's existing heuristic
quality-filter decision. It does not predict human preference, perceptual
quality, or whether a clip will improve downstream ASR. The heuristic remains
the production default.

## Data and labels

The study uses the deterministic 250-clip OpenSLR SLR31 pilot (CC BY 4.0).
Labels are the unchanged heuristic conjunction over duration, estimated SNR,
silence fraction, and clipping. Speakers 1272, 1462, 174, and 1988 form the
development set (206 clips); speakers 1993 and 2035 are untouched classifier
and downstream-ASR test speakers (44 clips). Development predictions are
leave-one-speaker-out to prevent a clip from being scored by a model trained on
its speaker.

## Model and features

The fixed model is a scikit-learn gradient-boosted tree with 100 estimators,
learning rate 0.05, depth 2, seed 20260831, balanced sample weights, and a 0.5
threshold. Inputs are duration, reference-free SNR, silence fraction, clipping,
RMS, MFCC mean/std statistics, spectral flatness, and zero-crossing statistics.
There was no hyperparameter search.

## Classifier results

Leave-one-speaker-out development agreement is 99.51% (205/206), with 98.84%
balanced accuracy. Untouched-speaker agreement and balanced accuracy are both
100% (44/44). The model has one disagreement, clip `1988-147956-0022`, which it
passes while the heuristic rejects. Its manual-listening field remains blank;
because only one disagreement exists, a 20--30 disagreement audit cannot be
performed without fabricating cases. Silence fraction contributes 95.45% of
fitted feature importance, so the result primarily shows that the model can
recover the direct heuristic feature boundary.

## Downstream results

The learned and heuristic training arms share all 147 financial clips and
146/147 crawler clips. Five Whisper-small LoRA trials (seeds 11, 22, 33, 44,
55) use identical validation and evaluation data. Learned-minus-heuristic
Earnings-21 overall WER is +0.24 percentage points (95% paired seed-bootstrap
CI -0.42 to +0.72; `N_trials=5`). On 40 clips from two untouched SLR31
speakers, the difference is -0.21 points (-0.72 to +0.45). Both intervals cross
zero, so the preregistered replacement gate is not met.

## Artifacts and limitations

`classifier_report.json`, `scored_manifest.parquet`,
`disagreement_ledger.csv`, `learned_filter_arm.parquet`, the ten per-seed ASR
reports, and `summary.json` are the primary artifacts. The classifier learns
weak labels from direct inputs to the heuristic, the corpus is small and
read-speech-only, the sole disagreement is not human adjudicated, and the
five-seed downstream study cannot establish equivalence from nonsignificance.
