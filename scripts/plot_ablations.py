#!/usr/bin/env python3
"""
Visualize ablation results: WER vs. mixture ratio, diversity improvements.

Reads either a live ablation_results.parquet or the pre-computed
example_results.json to generate publication-quality figures.

Usage
-----
    # From live ablation run:
    python scripts/plot_ablations.py \
        --results experiments/results/ablation_results.csv

    # From pre-computed example results:
    python scripts/plot_ablations.py \
        --example experiments/results/example_results.json
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results",  default=None, help="Path to ablation_results.csv")
    p.add_argument("--example",  default="experiments/results/example_results.json")
    p.add_argument("--output_dir", default="outputs/plots")
    return p.parse_args()


def plot_from_example(results: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = results.get("wer_ablation", [])
    diversity = results.get("diversity_impact", {})

    ratios = [r["synthetic_ratio"] for r in ablation]
    wer_overall = [r["wer_overall"] for r in ablation]
    wer_female = [r.get("wer_female", float("nan")) for r in ablation]
    wer_accented = [r.get("wer_accented", float("nan")) for r in ablation]
    wer_underrep = [r.get("wer_underrep", float("nan")) for r in ablation]

    pct_labels = [f"{int(r * 100)}%" for r in ratios]

    # ---- Figure 1: WER vs. mixture ratio (4-line) ----------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ratios, wer_overall,   "o-", color="#2196F3", lw=2, label="Overall", markersize=7)
    ax.plot(ratios, wer_female,    "s--", color="#9C27B0", lw=1.8, label="Female speakers", markersize=6)
    ax.plot(ratios, wer_accented,  "^--", color="#FF9800", lw=1.8, label="Accented speech", markersize=6)
    ax.plot(ratios, wer_underrep,  "D--", color="#F44336", lw=1.8, label="Underrepresented groups", markersize=6)

    # Mark best overall
    best_idx = int(wer_overall.index(min(wer_overall)))
    ax.axvline(ratios[best_idx], color="#2196F3", alpha=0.25, lw=1.5, linestyle=":")
    ax.annotate(
        f"Best overall\n{wer_overall[best_idx]:.1f}% WER",
        xy=(ratios[best_idx], wer_overall[best_idx]),
        xytext=(ratios[best_idx] + 0.06, wer_overall[best_idx] + 0.8),
        fontsize=9, color="#2196F3",
        arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1),
    )

    ax.set_xticks(ratios)
    ax.set_xticklabels(pct_labels)
    ax.set_xlabel("Fraction of training data that is synthetic", fontsize=11)
    ax.set_ylabel("Word Error Rate (%)", fontsize=11)
    ax.set_title("WER vs. synthetic data mixture ratio\n(evaluated on held-out real test set)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_wer_vs_ratio.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/ablation_wer_vs_ratio.png")

    # ---- Figure 2: Diversity entropy improvement -----------------------
    before = diversity.get("before_synthetic", {})
    after = diversity.get("after_synthetic_added", {})

    metrics = ["gender_entropy", "accent_entropy", "age_entropy", "overall"]
    labels = ["Gender", "Accent", "Age", "Overall"]
    b_vals = [before.get(m, 0) for m in metrics]
    a_vals = [after.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_b = ax.bar(x - width / 2, b_vals, width, label="Real data only", color="#EF5350", alpha=0.85)
    bars_a = ax.bar(x + width / 2, a_vals, width, label="Real + Synthetic", color="#66BB6A", alpha=0.85)

    for bar, val in zip(bars_b, b_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_a, a_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Normalized entropy [0 = skewed, 1 = uniform]", fontsize=10)
    ax.set_title("Diversity improvement from targeted synthetic augmentation", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "diversity_improvement.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/diversity_improvement.png")

    # ---- Figure 3: WER gain breakdown by group -------------------------
    real_only = ablation[0]
    best_mix = min(ablation[1:-1], key=lambda r: r["wer_overall"])  # exclude 0% and 100%

    groups = ["Overall", "Female", "Accented", "Underrep."]
    wer_real = [
        real_only["wer_overall"],
        real_only.get("wer_female", float("nan")),
        real_only.get("wer_accented", float("nan")),
        real_only.get("wer_underrep", float("nan")),
    ]
    wer_mix = [
        best_mix["wer_overall"],
        best_mix.get("wer_female", float("nan")),
        best_mix.get("wer_accented", float("nan")),
        best_mix.get("wer_underrep", float("nan")),
    ]

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_r = ax.bar(x - width / 2, wer_real, width, label="Real only (0% synthetic)", color="#EF5350", alpha=0.85)
    bars_m = ax.bar(x + width / 2, wer_mix, width,
                    label=f"Best mix ({int(best_mix['synthetic_ratio'] * 100)}% synthetic)", color="#42A5F5", alpha=0.85)

    for bar, val in zip(bars_r, wer_real):
        if val == val:  # not NaN
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars_m, wer_mix):
        if val == val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Word Error Rate (%)", fontsize=11)
    ax.set_title("WER comparison: real-only vs. best mixed training set\nby demographic group", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(wer_real) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "wer_by_group.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/wer_by_group.png")

    # ---- Figure 4: The tradeoff curve ----------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5))

    diversity_overall = [(r.get("wer_underrep", 0) - r["wer_overall"]) * -1 for r in ablation]
    # Proxy: overall diversity improvement from before→after is spread across ratios

    color1, color2 = "#2196F3", "#FF5722"
    ax1.plot(ratios, wer_overall, "o-", color=color1, lw=2, label="Overall WER (left axis)", markersize=7)
    ax1.set_xlabel("Fraction synthetic", fontsize=11)
    ax1.set_ylabel("Overall WER (%)", color=color1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    # Interpolate diversity scores assuming linear increase with synthetic ratio
    div_before_val = before.get("overall", 0.527)
    div_after_val = after.get("overall", 0.817)
    diversity_vals = [div_before_val + (div_after_val - div_before_val) * r for r in ratios]
    ax2.plot(ratios, diversity_vals, "s--", color=color2, lw=2, label="Diversity score (right axis)", markersize=7)
    ax2.set_ylabel("Diversity score [0–1]", color=color2, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    ax1.set_xticks(ratios)
    ax1.set_xticklabels(pct_labels)
    ax1.set_title("The diversity–accuracy tradeoff across mixture ratios", fontsize=12)
    ax1.grid(True, alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "tradeoff_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/tradeoff_curve.png")


def plot_from_csv(csv_path: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    div_cols = [c for c in df.columns if c.startswith("diversity_") and c != "diversity_overall"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if "acoustic_overlap" in df.columns:
        ax.plot(df["synthetic_ratio"], df["acoustic_overlap"], "o-", color="#2196F3", lw=2, markersize=7)
        ax.set_xlabel("Fraction synthetic")
        ax.set_ylabel("Acoustic overlap with eval set")
        ax.set_title("Acoustic distribution overlap vs. mixture ratio")
        ax.grid(True, alpha=0.3)
    if "wer_eval" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["synthetic_ratio"], df["wer_eval"], "s--", color="#F44336", lw=1.8, markersize=6)
        ax2.set_ylabel("WER on eval set (%)", color="#F44336")

    ax = axes[1]
    for col in div_cols:
        label = col.replace("diversity_", "")
        ax.plot(df["synthetic_ratio"], df[col], "o-", lw=2, label=label, markersize=6)
    if "diversity_overall" in df.columns:
        ax.plot(df["synthetic_ratio"], df["diversity_overall"], "k^-", lw=2.5, label="overall", markersize=8)
    ax.set_xlabel("Fraction synthetic")
    ax.set_ylabel("Diversity score [0–1]")
    ax.set_title("Diversity metrics vs. mixture ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "ablation_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.results and Path(args.results).exists():
        plot_from_csv(args.results, output_dir)
    elif Path(args.example).exists():
        with open(args.example) as f:
            results = json.load(f)
        plot_from_example(results, output_dir)
        print("\nNote: figures generated from pre-computed example_results.json")
        print("Run run_ablations.py first to generate results from your own data.")
    else:
        print(f"No results found. Checked:\n  {args.results}\n  {args.example}")
        print("Run run_ablations.py first, or check the path to example_results.json.")


if __name__ == "__main__":
    main()
