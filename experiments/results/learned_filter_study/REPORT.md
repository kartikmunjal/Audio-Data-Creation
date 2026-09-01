# Learned-filter downstream study

Generated from five paired trials. Differences are learned minus heuristic; intervals are 10,000-resample paired seed-bootstrap 95% CIs.

| Corpus | WER slice | Heuristic mean | Learned mean | Paired difference |
|---|---|---:|---:|---:|
| earnings21 | overall | 10.83% | 11.07% | +0.24 pp (-0.42, +0.72) |
| earnings21 | domain_terms | 10.00% | 10.28% | +0.28 pp (-0.41, +0.79) |
| earnings21 | common_terms | 21.70% | 21.38% | -0.32 pp (-0.75, +0.00) |
| openslr31 | overall | 5.60% | 5.38% | -0.21 pp (-0.72, +0.45) |
| openslr31 | domain_terms | undefined | undefined | undefined |
| openslr31 | common_terms | 5.60% | 5.38% | -0.21 pp (-0.72, +0.45) |

Locked replacement gate: **not passed**. `N_trials=5`.
