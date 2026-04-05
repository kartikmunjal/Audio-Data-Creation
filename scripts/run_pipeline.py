#!/usr/bin/env python3
"""
Run the full curation pipeline on a downloaded manifest.

Usage
-----
    python scripts/run_pipeline.py \
        --manifest data/raw/manifest.parquet \
        --output_dir outputs \
        --min_snr 15 \
        --min_duration 0.5 \
        --max_duration 20 \
        --dedup_threshold 0.97

Prerequisites: run download_sample.py first, or supply your own manifest.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/raw/manifest.parquet")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--min_snr", type=float, default=15.0)
    p.add_argument("--max_snr", type=float, default=80.0)
    p.add_argument("--min_duration", type=float, default=0.5)
    p.add_argument("--max_duration", type=float, default=20.0)
    p.add_argument("--max_silence_ratio", type=float, default=0.4)
    p.add_argument("--dedup_threshold", type=float, default=0.97)
    p.add_argument("--target_sr", type=int, default=16_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from audio_curation import CurationPipeline
    from audio_curation.quality import QualityThresholds

    import pandas as pd

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s — run download_sample.py first", manifest_path)
        sys.exit(1)

    manifest = pd.read_parquet(manifest_path)
    logger.info("Loaded manifest: %d samples", len(manifest))

    thresholds = QualityThresholds(
        min_duration_sec=args.min_duration,
        max_duration_sec=args.max_duration,
        min_snr_db=args.min_snr,
        max_silence_ratio=args.max_silence_ratio,
    )

    pipeline = CurationPipeline(
        quality_thresholds=thresholds,
        dedup_threshold=args.dedup_threshold,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
    )

    filtered, report = pipeline.run(manifest, audio_col="path", id_col="id")

    print("\n" + "=" * 60)
    print("CURATION SUMMARY")
    print("=" * 60)
    print(f"Input samples         : {report['n_input']}")
    print(f"After quality filter  : {report['quality']['passed']}")
    print(f"After deduplication   : {report['n_output']}")
    print(f"Overall retention     : {report['overall_retention_rate']:.1%}")
    print(f"Quality pass rate     : {report['quality']['pass_rate']:.1%}")
    print(f"Mean SNR              : {report['quality']['mean_snr_db']:.1f} dB")
    print(f"Near-dups removed     : {report['deduplication']['near_duplicates_removed']}")
    print(f"Diversity (after)     : {report['diversity_after']['diversity_scores'].get('overall', 'N/A'):.3f}")
    print(f"Total audio kept      : {report['diversity_after']['duration_stats'].get('total_hours', 0):.2f} hours")
    print(f"Elapsed               : {report['elapsed_sec']:.1f}s")
    print("=" * 60)
    print(f"\nFull report: {args.output_dir}/curation_report.json")
    print(f"Filtered manifest: {args.output_dir}/filtered_manifest.parquet")


if __name__ == "__main__":
    main()
