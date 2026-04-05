#!/usr/bin/env python3
"""
Generate synthetic speech samples to fill demographic gaps in a curated dataset.

This script picks up where run_pipeline.py leaves off: it reads the filtered
manifest, identifies which demographic groups are underrepresented, generates
targeted TTS samples using edge-tts, then re-runs diversity analysis to
show how the gaps closed.

Usage
-----
    python scripts/generate_synthetic.py \
        --manifest outputs/filtered_manifest.parquet \
        --output_dir data/synthetic \
        --max_synthetic_ratio 0.5 \
        --min_snr_db 20

Prerequisites
-------------
    pip install edge-tts
    python scripts/download_sample.py  # to get a real manifest first
    python scripts/run_pipeline.py     # to get the filtered manifest
"""
import argparse
import json
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
    p = argparse.ArgumentParser(description="Generate targeted synthetic audio samples")
    p.add_argument("--manifest",             default="outputs/filtered_manifest.parquet")
    p.add_argument("--output_dir",           default="data/synthetic")
    p.add_argument("--combined_output",      default="outputs/augmented_manifest.parquet")
    p.add_argument("--max_synthetic_ratio",  type=float, default=0.5)
    p.add_argument("--min_gap",              type=float, default=0.03,
                   help="Min fractional gap before we generate for a group")
    p.add_argument("--min_snr_db",           type=float, default=20.0)
    p.add_argument("--rate_limit_delay",     type=float, default=0.3,
                   help="Seconds between TTS requests")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the generation plan without actually synthesizing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_root = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_root))

    from audio_curation.synthetic import GapAnalyzer, TTSGenerator, VOICE_CATALOG
    from audio_curation.diversity import DiversityAnalyzer
    import pandas as pd

    # ---- Load filtered manifest -----------------------------------------
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s — run run_pipeline.py first", manifest_path)
        sys.exit(1)

    real_df = pd.read_parquet(manifest_path)
    if "source" not in real_df.columns:
        real_df["source"] = "real"
    logger.info("Loaded real manifest: %d samples", len(real_df))

    # ---- Diversity analysis BEFORE --------------------------------------
    analyzer_before = DiversityAnalyzer(real_df)
    div_before = analyzer_before.diversity_score()
    print("\n[Before synthetic augmentation]")
    print(f"  Gender entropy : {div_before.get('gender', float('nan')):.3f}")
    print(f"  Accent entropy : {div_before.get('accent', float('nan')):.3f}")
    print(f"  Overall        : {div_before.get('overall', float('nan')):.3f}")
    print()

    # ---- Gap analysis ---------------------------------------------------
    analyzer = GapAnalyzer(
        max_synthetic_ratio=args.max_synthetic_ratio,
        min_gap_for_generation=args.min_gap,
    )
    targets = analyzer.analyze(real_df, VOICE_CATALOG)

    print(f"[Synthesis plan] {len(targets)} targets, "
          f"{sum(t.n_samples for t in targets)} samples requested")
    print(f"  {'Gender':<8} {'Accent':<15} {'Voice':<35} {'N':>5} {'Priority':>8}")
    print("  " + "-" * 75)
    for t in targets[:15]:  # show top 15
        print(f"  {t.gender:<8} {t.accent:<15} {t.voice_id:<35} {t.n_samples:>5} {t.priority:>8.3f}")
    if len(targets) > 15:
        print(f"  ... and {len(targets) - 15} more targets")

    if args.dry_run:
        logger.info("Dry run — exiting without generating audio.")
        return

    # ---- Generate -------------------------------------------------------
    generator = TTSGenerator(
        output_dir=args.output_dir,
        rate_limit_delay=args.rate_limit_delay,
        apply_quality_filter=True,
    )
    synthetic_df = generator.generate_batch(targets, show_progress=True)

    if synthetic_df.empty:
        logger.error("No synthetic samples generated. Check edge-tts installation.")
        sys.exit(1)

    logger.info("Generated %d synthetic samples", len(synthetic_df))

    # ---- Combine and analyze AFTER --------------------------------------
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    analyzer_after = DiversityAnalyzer(combined_df)
    div_after = analyzer_after.diversity_score()

    print("\n[After synthetic augmentation]")
    print(f"  Gender entropy : {div_after.get('gender', float('nan')):.3f}  "
          f"(Δ {div_after.get('gender', 0) - div_before.get('gender', 0):+.3f})")
    print(f"  Accent entropy : {div_after.get('accent', float('nan')):.3f}  "
          f"(Δ {div_after.get('accent', 0) - div_before.get('accent', 0):+.3f})")
    print(f"  Overall        : {div_after.get('overall', float('nan')):.3f}  "
          f"(Δ {div_after.get('overall', 0) - div_before.get('overall', 0):+.3f})")

    # ---- Save combined manifest ----------------------------------------
    combined_output = Path(args.combined_output)
    combined_output.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(combined_output, index=False)
    logger.info("Augmented manifest written to %s (%d rows)", combined_output, len(combined_df))

    # ---- Save diversity comparison report -------------------------------
    div_report = {
        "before": div_before,
        "after": div_after,
        "delta": {k: round(div_after.get(k, 0) - div_before.get(k, 0), 4) for k in div_before},
        "n_real": len(real_df),
        "n_synthetic": len(synthetic_df),
        "n_combined": len(combined_df),
    }
    report_path = Path(args.combined_output).parent / "diversity_augmentation_report.json"
    with open(report_path, "w") as f:
        json.dump(div_report, f, indent=2)
    logger.info("Diversity report written to %s", report_path)


if __name__ == "__main__":
    main()
