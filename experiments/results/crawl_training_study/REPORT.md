# Crawler-augmentation training study

Generated from five paired training trials by `scripts/summarize_crawl_training_study.py`. Differences are augmented minus control; intervals are 10,000-resample paired seed-bootstrap 95% CIs.

| Corpus | WER slice | Control mean | Augmented mean | Paired difference |
|---|---|---:|---:|---:|
| earnings21 | overall | 11.47% | 10.83% | -0.64 pp (-1.43, +0.21) |
| earnings21 | domain_terms | 10.64% | 10.00% | -0.64 pp (-1.49, +0.28) |
| earnings21 | common_terms | 22.34% | 21.70% | -0.64 pp (-0.95, -0.32) |
| openslr31 | overall | 4.43% | 5.60% | +1.17 pp (+0.88, +1.46) |
| openslr31 | domain_terms | undefined | undefined | undefined |
| openslr31 | common_terms | 4.43% | 5.60% | +1.17 pp (+0.88, +1.46) |

Locked beneficial claim gate: **not passed**. `N_trials=5`.
