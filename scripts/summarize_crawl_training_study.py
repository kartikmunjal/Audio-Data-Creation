#!/usr/bin/env python3
"""Aggregate the locked five-seed crawler augmentation study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = [11, 22, 33, 44, 55]
METRICS = ["overall", "domain_terms", "common_terms"]


def paired_ci(values: list[float], *, seed: int, n: int = 10_000) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_trials": 0, "n_resamples": n}
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(n, len(array)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(array.mean()), "ci_low": float(low), "ci_high": float(high),
            "n_trials": len(array), "n_resamples": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/results/crawl_training_study")
    parser.add_argument("--output", default="experiments/results/crawl_training_study/summary.json")
    parser.add_argument("--markdown", default="experiments/results/crawl_training_study/REPORT.md")
    args = parser.parse_args()
    root = Path(args.results_dir)
    loaded = {}
    for arm in ("control", "augmented"):
        for corpus in ("earnings21", "openslr31"):
            reports = [json.loads((root / arm / corpus / f"seed_{seed}.json").read_text()) for seed in SEEDS]
            if [r["provenance"]["seed"] for r in reports] != SEEDS:
                raise RuntimeError(f"seed provenance mismatch: {arm}/{corpus}")
            loaded[(arm, corpus)] = reports
    result = {"schema_version": 1, "seeds": SEEDS, "n_trials": 5, "comparisons": {}}
    for corpus in ("earnings21", "openslr31"):
        result["comparisons"][corpus] = {}
        for metric in METRICS:
            control = np.asarray([r["wer"][metric] for r in loaded[("control", corpus)]], dtype=float)
            augmented = np.asarray([r["wer"][metric] for r in loaded[("augmented", corpus)]], dtype=float)
            result["comparisons"][corpus][metric] = {
                "control_mean": float(np.nanmean(control)) if np.isfinite(control).any() else None,
                "augmented_mean": float(np.nanmean(augmented)) if np.isfinite(augmented).any() else None,
                "augmented_minus_control": paired_ci((augmented - control).tolist(), seed=20260831),
            }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    def cell(item):
        if item["estimate"] is None: return "undefined"
        return f"{100*item['estimate']:+.2f} pp ({100*item['ci_low']:+.2f}, {100*item['ci_high']:+.2f})"
    rows = []
    for corpus in ("earnings21", "openslr31"):
        for metric in METRICS:
            item = result["comparisons"][corpus][metric]
            rows.append(f"| {corpus} | {metric} | {100*item['control_mean']:.2f}%" if item["control_mean"] is not None else f"| {corpus} | {metric} | undefined")
            rows[-1] += f" | {100*item['augmented_mean']:.2f}%" if item["augmented_mean"] is not None else " | undefined"
            rows[-1] += f" | {cell(item['augmented_minus_control'])} |"
    slr = result["comparisons"]["openslr31"]["overall"]["augmented_minus_control"]
    earn = result["comparisons"]["earnings21"]["overall"]["augmented_minus_control"]
    beneficial = slr["ci_high"] is not None and slr["ci_high"] < 0 and earn["ci_high"] <= 0.01
    text = """# Crawler-augmentation training study\n\nGenerated from five paired training trials by `scripts/summarize_crawl_training_study.py`. Differences are augmented minus control; intervals are 10,000-resample paired seed-bootstrap 95% CIs.\n\n| Corpus | WER slice | Control mean | Augmented mean | Paired difference |\n|---|---|---:|---:|---:|\n""" + "\n".join(rows) + f"\n\nLocked beneficial claim gate: **{'passed' if beneficial else 'not passed'}**. `N_trials=5`.\n"
    Path(args.markdown).write_text(text)
    print(text)


if __name__ == "__main__": main()
