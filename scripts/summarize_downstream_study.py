#!/usr/bin/env python3
"""Summarize paired five-seed curation and augmentation ASR contrasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SEEDS = (11, 22, 33, 44, 55)
METRICS = ("overall", "domain_terms", "common_terms")


def summarize_contrast(
    root: Path,
    control: str,
    treatment: str,
    analyzer,
    paired_bootstrap_difference_ci,
    n_resamples: int,
) -> dict:
    control_reports = [
        json.loads((root / control / f"seed_{seed}.json").read_text())
        for seed in SEEDS
    ]
    treatment_reports = [
        json.loads((root / treatment / f"seed_{seed}.json").read_text())
        for seed in SEEDS
    ]
    rng = np.random.default_rng(20260729)
    metrics = {}
    for metric in METRICS:
        control_values, treatment_values, deltas, clip_cis = [], [], [], []
        for seed, left, right in zip(SEEDS, control_reports, treatment_reports):
            left_predictions, right_predictions = left["predictions"], right["predictions"]
            if [x["id"] for x in left_predictions] != [x["id"] for x in right_predictions]:
                raise RuntimeError(f"Unpaired evaluation IDs for seed {seed}")
            references = [x["reference"] for x in left_predictions]
            mask = [analyzer._contains_domain_term(ref) for ref in references]
            keep = (
                [True] * len(mask) if metric == "overall"
                else mask if metric == "domain_terms"
                else [not value for value in mask]
            )
            refs = [value for value, flag in zip(references, keep) if flag]
            left_hyps = [
                value["hypothesis"] for value, flag in zip(left_predictions, keep) if flag
            ]
            right_hyps = [
                value["hypothesis"] for value, flag in zip(right_predictions, keep) if flag
            ]
            paired = paired_bootstrap_difference_ci(
                refs, left_hyps, right_hyps, n_resamples=n_resamples, seed=seed
            )
            control_values.append(float(left["wer"][metric]))
            treatment_values.append(float(right["wer"][metric]))
            deltas.append(float(paired["estimate"]))
            clip_cis.append(paired)
        control_array = np.asarray(control_values)
        treatment_array = np.asarray(treatment_values)
        delta_array = np.asarray(deltas)
        control_means = np.asarray([
            rng.choice(control_array, len(control_array), replace=True).mean()
            for _ in range(n_resamples)
        ])
        treatment_means = np.asarray([
            rng.choice(treatment_array, len(treatment_array), replace=True).mean()
            for _ in range(n_resamples)
        ])
        delta_means = np.asarray([
            rng.choice(delta_array, len(delta_array), replace=True).mean()
            for _ in range(n_resamples)
        ])
        metrics[metric] = {
            "control_mean_wer": float(control_array.mean()),
            "control_trial_values": control_values,
            "control_trial_bootstrap_95_ci": np.quantile(
                control_means, [0.025, 0.975]
            ).tolist(),
            "treatment_mean_wer": float(treatment_array.mean()),
            "treatment_trial_values": treatment_values,
            "treatment_trial_bootstrap_95_ci": np.quantile(
                treatment_means, [0.025, 0.975]
            ).tolist(),
            "mean_paired_delta_treatment_minus_control": float(delta_array.mean()),
            "paired_delta_trial_values": deltas,
            "paired_delta_trial_bootstrap_95_ci": np.quantile(
                delta_means, [0.025, 0.975]
            ).tolist(),
            "per_trial_paired_clip_bootstrap": clip_cis,
        }
    return {
        "control": control,
        "treatment": treatment,
        "n_trials": 5,
        "seeds": list(SEEDS),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-root", required=True)
    parser.add_argument(
        "--results-dir", default="experiments/results/downstream_study"
    )
    parser.add_argument(
        "--output", default="experiments/results/downstream_study/summary.json"
    )
    parser.add_argument(
        "--markdown-output",
        default="experiments/results/downstream_study/summary.md",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()
    whisper_root = Path(args.whisper_root).resolve()
    sys.path.insert(0, str(whisper_root / "src"))
    from whisper_adapt.evaluation.wer import (  # noqa: PLC0415
        DomainWERAnalyzer,
        load_domain_vocab,
        paired_bootstrap_difference_ci,
    )
    analyzer = DomainWERAnalyzer(
        load_domain_vocab(whisper_root / "configs" / "financial_terms.txt")
    )
    results = Path(args.results_dir)
    summary = {
        "schema_version": 1,
        "contrasts": {
            "curation": summarize_contrast(
                results, "curation_control", "curation_policy", analyzer,
                paired_bootstrap_difference_ci, args.bootstrap_resamples,
            ),
            "targeted_augmentation": summarize_contrast(
                results, "augmentation_control_common",
                "augmentation_targeted_50pct", analyzer,
                paired_bootstrap_difference_ci, args.bootstrap_resamples,
            ),
        },
        "claim_boundary": (
            "The quality audit ledger remains pending; curation results compare "
            "a named policy but do not validate human-perceived quality labels."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    labels = {
        "overall": "Overall",
        "domain_terms": "Domain",
        "common_terms": "Common",
    }
    contrast_labels = {
        "curation": "Curation policy",
        "targeted_augmentation": "Targeted augmentation",
    }
    lines = [
        "# Downstream ASR study",
        "",
        "| Contrast | Split | Control WER (95% trial CI) | Treatment WER (95% trial CI) | Paired ΔWER (95% trial CI) | N_trials |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for contrast_name, contrast in summary["contrasts"].items():
        for metric in METRICS:
            item = contrast["metrics"][metric]
            control_ci = item["control_trial_bootstrap_95_ci"]
            treatment_ci = item["treatment_trial_bootstrap_95_ci"]
            delta_ci = item["paired_delta_trial_bootstrap_95_ci"]
            lines.append(
                f"| {contrast_labels[contrast_name]} | {labels[metric]} | "
                f"{item['control_mean_wer'] * 100:.2f}% "
                f"({control_ci[0] * 100:.2f}–{control_ci[1] * 100:.2f}%) | "
                f"{item['treatment_mean_wer'] * 100:.2f}% "
                f"({treatment_ci[0] * 100:.2f}–{treatment_ci[1] * 100:.2f}%) | "
                f"{item['mean_paired_delta_treatment_minus_control'] * 100:+.2f} pp "
                f"({delta_ci[0] * 100:+.2f}–{delta_ci[1] * 100:+.2f} pp) | 5 |"
            )
    lines.extend([
        "",
        f"> {summary['claim_boundary']}",
        "",
        "Generated by `scripts/summarize_downstream_study.py`.",
        "",
    ])
    markdown = Path(args.markdown_output)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
