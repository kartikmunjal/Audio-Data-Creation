#!/usr/bin/env python3
"""Run the locked controlled MFCC-LSH duplicate-pair benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audio_curation.deduplication import (  # noqa: E402
    RandomProjectionLSH,
    compute_mfcc_embedding,
    cosine_similarity,
)

SEED = 20260729
LSH_SEED = 42
N_BITS = 18
THRESHOLD = 0.97


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if not total:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - half, center + half]


def metric(successes: int, total: int) -> dict:
    return {
        "estimate": successes / total if total else None,
        "numerator": successes,
        "denominator": total,
        "wilson_95_ci": wilson(successes, total) if total else None,
    }


def evaluate_pair(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    lsh: RandomProjectionLSH,
) -> tuple[bool, bool, float]:
    left_embedding = compute_mfcc_embedding(left, sample_rate)
    right_embedding = compute_mfcc_embedding(right, sample_rate)
    candidate = lsh.hash(left_embedding) == lsh.hash(right_embedding)
    similarity = cosine_similarity(left_embedding, right_embedding)
    accepted = candidate and similarity >= THRESHOLD
    return candidate, accepted, similarity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--manifest", default="data/financial_research/train_manifest.parquet"
    )
    parser.add_argument(
        "--output",
        default="experiments/results/dedup_validation/summary.json",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = Path(args.source_root).resolve()
    manifest = source / args.manifest
    frame = pd.read_parquet(manifest)
    if len(frame) < 100:
        raise RuntimeError("Controlled benchmark requires at least 100 source clips")
    rng = np.random.default_rng(SEED)
    positive_indices = np.sort(rng.choice(len(frame), 50, replace=False))
    negative_left = rng.choice(len(frame), 100, replace=True)
    negative_right = rng.choice(len(frame), 100, replace=True)
    for index in range(100):
        while negative_right[index] == negative_left[index]:
            negative_right[index] = rng.integers(0, len(frame))

    cache: dict[int, tuple[np.ndarray, int]] = {}

    def load(index: int) -> tuple[np.ndarray, int]:
        if index not in cache:
            value = Path(frame.iloc[index].path)
            path = value if value.is_absolute() else source / value
            audio, sr = librosa.load(path, sr=16_000, mono=True)
            cache[index] = (audio.astype(np.float32), sr)
        return cache[index]

    lsh = RandomProjectionLSH(dim=40, n_bits=N_BITS, seed=LSH_SEED)
    rows = []
    for index in positive_indices:
        audio, sr = load(int(index))
        gain_pad = np.pad(audio * 0.8, (int(0.08 * sr), 0))
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=14_400)
        resampled = librosa.resample(resampled, orig_sr=14_400, target_sr=sr)
        for transform, transformed in (
            ("gain_0.8_plus_80ms_leading_pad", gain_pad),
            ("resample_16k_14.4k_16k", resampled),
        ):
            candidate, accepted, similarity = evaluate_pair(
                audio, transformed, sr, lsh
            )
            rows.append({
                "label_duplicate": True,
                "transform": transform,
                "candidate": candidate,
                "accepted": accepted,
                "cosine_similarity": similarity,
            })
    for left_index, right_index in zip(negative_left, negative_right):
        left, sr = load(int(left_index))
        right, right_sr = load(int(right_index))
        if right_sr != sr:
            raise RuntimeError("Unexpected sample-rate mismatch")
        candidate, accepted, similarity = evaluate_pair(left, right, sr, lsh)
        rows.append({
            "label_duplicate": False,
            "transform": "distinct_source_ids",
            "candidate": candidate,
            "accepted": accepted,
            "cosine_similarity": similarity,
        })

    positives = [row for row in rows if row["label_duplicate"]]
    negatives = [row for row in rows if not row["label_duplicate"]]
    candidate_tp = sum(row["candidate"] for row in positives)
    candidate_fp = sum(row["candidate"] for row in negatives)
    final_tp = sum(row["accepted"] for row in positives)
    final_fp = sum(row["accepted"] for row in negatives)
    final_tn = len(negatives) - final_fp
    summary = {
        "schema_version": 1,
        "benchmark_scope": "known recording identity under two fixed perturbations",
        "n_positive_pairs": len(positives),
        "n_negative_pairs": len(negatives),
        "configuration": {
            "seed": SEED,
            "lsh_seed": LSH_SEED,
            "n_lsh_bits": N_BITS,
            "cosine_threshold": THRESHOLD,
        },
        "candidate_recall": metric(candidate_tp, len(positives)),
        "candidate_precision": metric(candidate_tp, candidate_tp + candidate_fp),
        "final_recall": metric(final_tp, len(positives)),
        "final_precision": metric(final_tp, final_tp + final_fp),
        "final_specificity": metric(final_tn, len(negatives)),
        "pair_results": rows,
        "source_manifest": args.manifest,
        "source_manifest_sha256": sha256(manifest),
        "git_commit": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "git_dirty": bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()),
        "claim_boundary": (
            "Controlled identity perturbations do not validate semantic or "
            "human-perceived near-duplicate judgments."
        ),
        "generator": "scripts/run_dedup_validation.py",
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "pair_results"}, indent=2))


if __name__ == "__main__":
    main()
