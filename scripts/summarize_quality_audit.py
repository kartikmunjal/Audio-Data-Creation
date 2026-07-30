#!/usr/bin/env python3
"""Score the manually labeled quality-policy audit with binomial intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_bool(value: object, column: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unrecognized {column} label: {value!r}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def rate(successes: int, total: int) -> dict:
    return {
        "estimate": successes / total if total else None,
        "numerator": successes,
        "denominator": total,
        "wilson_95_ci": wilson_interval(successes, total) if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        default="experiments/downstream_study/quality_audit_ledger.csv",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/downstream_study/quality_audit_summary.json",
    )
    args = parser.parse_args()
    ledger = Path(args.ledger)
    frame = pd.read_csv(ledger)
    required = {"id", "policy_passes", "manual_acceptable"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ledger is missing columns: {sorted(missing)}")
    if frame["manual_acceptable"].isna().any():
        count = int(frame["manual_acceptable"].isna().sum())
        raise RuntimeError(
            f"Manual audit is incomplete: {count} rows lack manual_acceptable labels"
        )
    predicted = [parse_bool(x, "policy_passes") for x in frame["policy_passes"]]
    observed = [
        parse_bool(x, "manual_acceptable") for x in frame["manual_acceptable"]
    ]
    tp = sum(p and y for p, y in zip(predicted, observed))
    fp = sum(p and not y for p, y in zip(predicted, observed))
    fn = sum(not p and y for p, y in zip(predicted, observed))
    tn = sum(not p and not y for p, y in zip(predicted, observed))
    summary = {
        "schema_version": 1,
        "ledger": str(ledger),
        "n_labeled": len(frame),
        "confusion_matrix": {
            "policy_pass_acceptable": tp,
            "policy_pass_unacceptable": fp,
            "policy_reject_acceptable": fn,
            "policy_reject_unacceptable": tn,
        },
        "acceptable_precision_among_policy_passes": rate(tp, tp + fp),
        "acceptable_recall": rate(tp, tp + fn),
        "unacceptable_recall": rate(tn, tn + fp),
        "accuracy": rate(tp + tn, len(frame)),
        "interval_method": "Wilson score, two-sided 95%",
        "generator": "scripts/summarize_quality_audit.py",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
