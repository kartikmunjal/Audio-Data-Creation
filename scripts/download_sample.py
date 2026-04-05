#!/usr/bin/env python3
"""
Download a sample of the Mozilla Common Voice dataset via HuggingFace Datasets.

Usage
-----
    python scripts/download_sample.py --n_samples 500 --split validation --output_dir data/raw

Notes
-----
Common Voice clips are short (1-10s), speaker-labeled, and include age/gender/accent
metadata, making it a good testbed for the full curation pipeline. The validation
split is used by default since it requires no separate login.
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Common Voice sample")
    p.add_argument("--n_samples", type=int, default=500, help="Number of clips to download")
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--locale", default="en", help="Language locale (e.g. en, de, fr)")
    p.add_argument("--output_dir", default="data/raw", help="Directory to save audio + manifest")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed. Run: pip install datasets")
        sys.exit(1)

    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile not installed. Run: pip install soundfile")
        sys.exit(1)

    import pandas as pd
    import numpy as np

    logger.info("Loading Common Voice %s split for locale '%s'...", args.split, args.locale)

    ds = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        args.locale,
        split=args.split,
        trust_remote_code=True,
    )

    # Subsample
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(ds))
    indices = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
    ds = ds.select(indices)

    logger.info("Writing %d clips to %s...", n, output_dir)

    records = []
    for i, sample in enumerate(ds):
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        clip_id = f"cv_{args.locale}_{args.split}_{i:06d}"
        wav_path = output_dir / f"{clip_id}.wav"

        sf.write(str(wav_path), audio_array, sr)

        records.append({
            "id": clip_id,
            "path": str(wav_path),
            "speaker_id": sample.get("client_id", ""),
            "sentence": sample.get("sentence", ""),
            "age": sample.get("age", ""),
            "gender": sample.get("gender", ""),
            "accent": sample.get("accent", ""),
            "locale": args.locale,
            "upvotes": sample.get("up_votes", 0),
            "downvotes": sample.get("down_votes", 0),
            "source_split": args.split,
        })

        if (i + 1) % 100 == 0:
            logger.info("  %d / %d", i + 1, n)

    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info("Manifest written to %s (%d rows)", manifest_path, len(manifest))


if __name__ == "__main__":
    main()
