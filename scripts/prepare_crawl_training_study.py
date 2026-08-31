#!/usr/bin/env python3
"""Prepare the locked matched-size crawler augmentation training manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

SELECTION_SEED = 20260831
TRAIN_SPEAKERS = {"1272", "1462", "174", "1988"}
TEST_SPEAKERS = {"1993", "2035"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    result["path"] = [str(PurePosixPath(prefix) / Path(value).name) for value in result.path]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-train", required=True)
    parser.add_argument("--crawler-manifest", required=True)
    parser.add_argument("--output-dir", default="experiments/crawl_training_study")
    parser.add_argument("--crawler-audio-prefix", default="data/openslr31_pilot/audio")
    args = parser.parse_args()
    financial_path = Path(args.financial_train)
    crawler_path = Path(args.crawler_manifest)
    financial = pd.read_parquet(financial_path)
    crawler = pd.read_parquet(crawler_path)
    required = {"id", "path", "sentence"}
    for name, frame in (("financial", financial), ("crawler", crawler)):
        if missing := required - set(frame.columns):
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError(f"{name} IDs must be non-null and unique")
    crawler["speaker_id"] = crawler.speaker_id.astype(str)
    observed = set(crawler.speaker_id)
    if observed != TRAIN_SPEAKERS | TEST_SPEAKERS:
        raise ValueError(f"locked speaker set mismatch: {sorted(observed)}")
    train_crawler = crawler[crawler.speaker_id.isin(TRAIN_SPEAKERS)].copy()
    test_crawler = crawler[crawler.speaker_id.isin(TEST_SPEAKERS)].copy()
    if set(train_crawler.speaker_id) & set(test_crawler.speaker_id):
        raise RuntimeError("speaker leakage")

    n_total = len(financial)
    n_crawler = n_total // 2
    n_financial = n_total - n_crawler
    if len(train_crawler) < n_crawler or len(financial) < n_financial:
        raise RuntimeError("insufficient rows for locked no-replacement mixture")
    rng = np.random.default_rng(SELECTION_SEED)
    selected_financial = financial.iloc[np.sort(rng.choice(len(financial), n_financial, replace=False))]
    selected_crawler = train_crawler.iloc[np.sort(rng.choice(len(train_crawler), n_crawler, replace=False))]
    selected_crawler = portable(selected_crawler, args.crawler_audio_prefix)
    selected_crawler["is_domain"] = False
    selected_crawler["source_arm"] = "openslr31"
    selected_financial = selected_financial.copy()
    selected_financial["source_arm"] = "financial"
    augmented = pd.concat([selected_financial, selected_crawler], ignore_index=True, sort=False)
    augmented = augmented.sample(frac=1, random_state=SELECTION_SEED).reset_index(drop=True)
    test_crawler = portable(test_crawler, args.crawler_audio_prefix)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    control_path = output / "control_financial.parquet"
    augmented_path = output / "augmented_50pct_crawler.parquet"
    test_path = output / "openslr31_speaker_disjoint_test.parquet"
    financial.to_parquet(control_path, index=False)
    augmented.to_parquet(augmented_path, index=False)
    test_crawler.to_parquet(test_path, index=False)
    report = {
        "schema_version": 1, "selection_seed": SELECTION_SEED,
        "train_speakers": sorted(TRAIN_SPEAKERS), "test_speakers": sorted(TEST_SPEAKERS),
        "speaker_disjoint": True, "control_rows": len(financial),
        "augmented_rows": len(augmented), "augmented_financial_rows": n_financial,
        "augmented_crawler_rows": n_crawler, "crawler_test_rows": len(test_crawler),
        "matched_size": len(financial) == len(augmented),
        "source_hashes": {"financial_train": sha256(financial_path),
                          "crawler_manifest": sha256(crawler_path)},
        "output_hashes": {p.name: sha256(p) for p in (control_path, augmented_path, test_path)},
    }
    (output / "study_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
