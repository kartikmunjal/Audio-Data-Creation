#!/usr/bin/env python3
"""
Run synthetic vs. real vs. mixed ablation experiments.

Experiment design
-----------------
  1. Load the real (filtered) manifest and the synthetic manifest.
  2. Create 6 mixed datasets at ratios: 0%, 10%, 25%, 50%, 75%, 100% synthetic.
  3. For each split, compute:
       a. Diversity metrics (gender/accent/age entropy) — always fast
       b. Acoustic distribution overlap with the eval set — fast
       c. Whisper WER on the eval set — slow, requires --use_whisper flag
  4. Save results to experiments/results/ablation_results.{parquet,csv}
  5. Print a summary table.

The key insight to look for:
  - Diversity increases monotonically with synthetic ratio (by design).
  - WER on *underrepresented* demographic groups decreases as ratio increases.
  - WER on *common* groups slightly increases at high synthetic ratios (TTS artifacts).
  - There is an optimal ratio (typically 25-50%) that maximises overall performance.

Usage
-----
    # Fast mode (diversity + acoustic overlap only):
    python scripts/run_ablations.py \
        --real_manifest outputs/filtered_manifest.parquet \
        --synthetic_manifest data/synthetic/synthetic_manifest.parquet

    # Full mode (+ Whisper WER, much slower):
    python scripts/run_ablations.py \
        --real_manifest outputs/filtered_manifest.parquet \
        --synthetic_manifest data/synthetic/synthetic_manifest.parquet \
        --eval_manifest data/raw/eval_manifest.parquet \
        --use_whisper \
        --whisper_model base
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run data mixing ablation experiments")
    p.add_argument("--real_manifest",      default="outputs/filtered_manifest.parquet")
    p.add_argument("--synthetic_manifest", default="data/synthetic/synthetic_manifest.parquet")
    p.add_argument("--eval_manifest",      default=None,
                   help="Held-out eval set for WER. Defaults to 10%% of real.")
    p.add_argument("--output_dir",         default="experiments/results")
    p.add_argument("--ratios", nargs="+",  type=float,
                   default=[0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
    p.add_argument("--strategy",           default="random",
                   choices=["random", "stratified"])
    p.add_argument("--use_whisper",        action="store_true")
    p.add_argument("--whisper_model",      default="base")
    p.add_argument("--seed",               type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_root = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_root))

    from audio_curation.synthetic import DataMixer, AblationEvaluator
    import pandas as pd

    # ---- Load manifests ------------------------------------------------
    real_path = Path(args.real_manifest)
    syn_path = Path(args.synthetic_manifest)

    if not real_path.exists():
        logger.error("Real manifest not found: %s", real_path)
        sys.exit(1)
    if not syn_path.exists():
        logger.error("Synthetic manifest not found: %s — run generate_synthetic.py first", syn_path)
        sys.exit(1)

    real_df = pd.read_parquet(real_path)
    syn_df = pd.read_parquet(syn_path)
    if "source" not in real_df.columns:
        real_df["source"] = "real"
    if "source" not in syn_df.columns:
        syn_df["source"] = "synthetic"

    logger.info("Real: %d samples  |  Synthetic: %d samples", len(real_df), len(syn_df))

    # ---- Eval split ----------------------------------------------------
    if args.eval_manifest and Path(args.eval_manifest).exists():
        eval_df = pd.read_parquet(args.eval_manifest)
        train_df = real_df
    else:
        # Hold out 10% of real data as eval; rest is training real set
        eval_size = max(50, int(len(real_df) * 0.10))
        eval_df = real_df.sample(n=eval_size, random_state=args.seed)
        train_df = real_df.drop(eval_df.index).reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)
        logger.info("Using 10%% real holdout as eval set (%d samples)", len(eval_df))

    # ---- Create ablation splits ----------------------------------------
    mixer = DataMixer(seed=args.seed, fix_total_size=True)
    splits = mixer.create_ablation_splits(
        train_df, syn_df, ratios=args.ratios, strategy=args.strategy
    )

    # ---- Evaluate ------------------------------------------------------
    evaluator = AblationEvaluator(
        whisper_model=args.whisper_model,
        output_dir=args.output_dir,
    )

    results_df = evaluator.run_ablation(
        splits,
        eval_manifest=eval_df,
        audio_col="path",
        text_col="sentence",
        use_whisper=args.use_whisper,
    )

    # ---- Print summary -------------------------------------------------
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)

    cols_to_show = ["synthetic_ratio", "n_total", "n_real", "n_synthetic", "acoustic_overlap"]
    div_cols = [c for c in results_df.columns if c.startswith("diversity_")]
    cols_to_show.extend(div_cols)
    if "wer_eval" in results_df.columns:
        cols_to_show.append("wer_eval")

    cols_present = [c for c in cols_to_show if c in results_df.columns]
    print(results_df[cols_present].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 80)

    best_row = results_df.loc[results_df["acoustic_overlap"].idxmax()] if "acoustic_overlap" in results_df.columns else None
    if best_row is not None:
        print(f"\nBest acoustic overlap: ratio={best_row['synthetic_ratio']:.0%}  "
              f"overlap={best_row['acoustic_overlap']:.3f}")

    if "wer_eval" in results_df.columns:
        best_wer = results_df.loc[results_df["wer_eval"].idxmin()]
        print(f"Best WER: ratio={best_wer['synthetic_ratio']:.0%}  WER={best_wer['wer_eval']:.1f}%")

    print(f"\nFull results: {args.output_dir}/ablation_results.csv")


if __name__ == "__main__":
    main()
