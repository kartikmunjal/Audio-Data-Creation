#!/usr/bin/env python3
"""Summarize frozen-ASR behavior before/after crawler quality selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd


WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(), jiwer.ReduceToListOfListOfWords(),
])


def edit_arrays(predictions: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    errors, words = [], []
    for row in predictions:
        result = jiwer.process_words(
            row["reference"], row["hypothesis"],
            reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM,
        )
        errors.append(result.substitutions + result.deletions + result.insertions)
        words.append(result.hits + result.substitutions + result.deletions)
    return np.asarray(errors), np.asarray(words)


def interval(errors: np.ndarray, words: np.ndarray, *, seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n)
    for i in range(n):
        index = rng.integers(0, len(errors), len(errors))
        estimates[i] = errors[index].sum() / words[index].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": float(errors.sum() / words.sum()), "ci_low": float(low),
            "ci_high": float(high), "n_resamples": n}


def paired_delta(left: np.ndarray, right: np.ndarray, words: np.ndarray, *, seed: int, n: int) -> dict:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n)
    for i in range(n):
        index = rng.integers(0, len(words), len(words))
        estimates[i] = (right[index].sum() - left[index].sum()) / words[index].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": float((right.sum() - left.sum()) / words.sum()),
            "ci_low": float(low), "ci_high": float(high), "n_resamples": n}


def selection_delta(errors: np.ndarray, words: np.ndarray, kept: np.ndarray, *, seed: int, n: int) -> dict:
    """Retained-minus-all corpus WER under shared full-corpus bootstrap draws."""
    rng = np.random.default_rng(seed)
    estimates = np.empty(n)
    for i in range(n):
        index = rng.integers(0, len(words), len(words))
        chosen = kept[index]
        estimates[i] = (
            errors[index][chosen].sum() / words[index][chosen].sum()
            - errors[index].sum() / words[index].sum()
        )
    point = errors[kept].sum() / words[kept].sum() - errors.sum() / words.sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": float(point), "ci_low": float(low), "ci_high": float(high),
            "n_resamples": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--filtered-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text())
    adapted = json.loads(Path(args.adapted).read_text())
    if [x["id"] for x in base["predictions"]] != [x["id"] for x in adapted["predictions"]]:
        raise ValueError("model prediction IDs are not paired")
    ids = np.asarray([x["id"] for x in base["predictions"]])
    retained_ids = set(pd.read_parquet(args.filtered_manifest)["id"].astype(str))
    kept = np.asarray([value in retained_ids for value in ids])
    if kept.sum() != len(retained_ids):
        raise ValueError("filtered manifest IDs do not match prediction IDs")
    base_errors, words = edit_arrays(base["predictions"])
    adapted_errors, adapted_words = edit_arrays(adapted["predictions"])
    if not np.array_equal(words, adapted_words):
        raise ValueError("reference word counts differ between models")
    n = args.bootstrap_resamples
    result = {
        "schema_version": 1, "seed": args.seed, "n_trials": 1,
        "n_all": len(ids), "n_retained": int(kept.sum()),
        "scope": "descriptive frozen-model selection audit; not a training-effect estimate",
        "domain_slice": {"n": 0, "reason": "no financial vocabulary terms occur in SLR31 pilot references"},
        "base": {
            "all": interval(base_errors, words, seed=args.seed, n=n),
            "retained": interval(base_errors[kept], words[kept], seed=args.seed, n=n),
            "retained_minus_all": selection_delta(base_errors, words, kept, seed=args.seed, n=n),
        },
        "financial_adapter_seed11": {
            "all": interval(adapted_errors, words, seed=args.seed, n=n),
            "retained": interval(adapted_errors[kept], words[kept], seed=args.seed, n=n),
            "retained_minus_all": selection_delta(adapted_errors, words, kept, seed=args.seed, n=n),
        },
        "adapted_minus_base": {
            "all": paired_delta(base_errors, adapted_errors, words, seed=args.seed, n=n),
            "retained": paired_delta(base_errors[kept], adapted_errors[kept], words[kept], seed=args.seed, n=n),
        },
        "inputs": {"base": args.base, "adapted": args.adapted,
                   "filtered_manifest": args.filtered_manifest},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pct = lambda x: f"{100*x:.2f}%"
    ci = lambda x: f"{pct(x['estimate'])} ({pct(x['ci_low'])}, {pct(x['ci_high'])})"
    markdown = f"""# Frozen-ASR audit of the crawler pilot

Generated by `scripts/summarize_crawl_asr.py` from committed per-clip predictions.
Intervals are 10,000-resample clip-bootstrap 95% CIs; `N_trials=1`.

| Frozen model | All 250 clips | Quality-retained 203 clips | Retained − all |
|---|---:|---:|---:|
| Base Whisper-small | {ci(result['base']['all'])} | {ci(result['base']['retained'])} | {ci(result['base']['retained_minus_all'])} |
| Financial adapter (seed 11) | {ci(result['financial_adapter_seed11']['all'])} | {ci(result['financial_adapter_seed11']['retained'])} | {ci(result['financial_adapter_seed11']['retained_minus_all'])} |

The paired adapter-minus-base difference is {ci(result['adapted_minus_base']['all'])}
on all clips and {ci(result['adapted_minus_base']['retained'])} on retained clips.

This is a descriptive selection audit, not a training-data intervention. The corpus
contains zero configured financial-domain terms, so domain WER is undefined and common
WER equals overall WER. The result measures how the fixed quality policy changes the
aggregate evaluation population and whether the financial adapter regresses on open
read speech; it does not show what retraining on SLR31 would do.
"""
    Path(args.markdown).write_text(markdown)


if __name__ == "__main__":
    main()
