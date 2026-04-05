#!/usr/bin/env python3
"""
Generate visualizations from a saved curation report.

Usage
-----
    python scripts/plot_report.py --report outputs/curation_report.json
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="outputs/curation_report.json")
    p.add_argument("--output_dir", default="outputs/plots")
    return p.parse_args()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.report) as f:
        report = json.load(f)

    quality = report.get("quality", {})
    div_before = report.get("diversity_before", {})
    div_after = report.get("diversity_after", {})
    dedup = report.get("deduplication", {})

    # ---- Figure 1: Pipeline funnel ----
    fig, ax = plt.subplots(figsize=(8, 4))
    stages = ["Input", "Quality filter", "Dedup"]
    counts = [
        report.get("n_input", 0),
        quality.get("passed", 0),
        report.get("n_output", 0),
    ]
    bars = ax.barh(stages, counts, color=["#4C72B0", "#55A868", "#C44E52"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=10)
    ax.set_xlabel("Number of samples")
    ax.set_title("Curation pipeline: sample counts per stage")
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(output_dir / "pipeline_funnel.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: Quality failure reasons ----
    fail_reasons = quality.get("fail_reason_counts", {})
    if fail_reasons:
        fig, ax = plt.subplots(figsize=(8, 4))
        reasons = list(fail_reasons.keys())
        values = list(fail_reasons.values())
        ax.barh(reasons, values, color="#4C72B0")
        ax.set_xlabel("Number of clips")
        ax.set_title("Quality filter — failure reason breakdown")
        fig.tight_layout()
        fig.savefig(output_dir / "quality_failures.png", dpi=150)
        plt.close(fig)

    # ---- Figure 3: Gender distribution before/after ----
    def extract_gender(div):
        gd = div.get("gender_distribution", {})
        counts_dict = gd.get("count", {})
        return counts_dict

    g_before = extract_gender(div_before)
    g_after = extract_gender(div_after)
    if g_before and g_after:
        labels = sorted(set(g_before) | set(g_after))
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width / 2, [g_before.get(l, 0) for l in labels], width, label="Before", color="#4C72B0")
        ax.bar(x + width / 2, [g_after.get(l, 0) for l in labels], width, label="After", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count")
        ax.set_title("Gender distribution: before vs. after curation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "gender_distribution.png", dpi=150)
        plt.close(fig)

    # ---- Figure 4: Diversity scores ----
    div_scores_before = div_before.get("diversity_scores", {})
    div_scores_after = div_after.get("diversity_scores", {})
    common_keys = [k for k in div_scores_after if k in div_scores_before and k != "overall"]
    if common_keys:
        x = np.arange(len(common_keys))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width / 2, [div_scores_before[k] for k in common_keys], width, label="Before", color="#4C72B0")
        ax.bar(x + width / 2, [div_scores_after[k] for k in common_keys], width, label="After", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(common_keys)
        ax.set_ylabel("Normalized entropy [0, 1]")
        ax.set_title("Diversity scores: before vs. after curation")
        ax.set_ylim(0, 1.1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "diversity_scores.png", dpi=150)
        plt.close(fig)

    print(f"Plots written to {output_dir}/")


if __name__ == "__main__":
    main()
