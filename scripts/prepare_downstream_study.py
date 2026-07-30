#!/usr/bin/env python3
"""Prepare locked matched-size curation and targeted-augmentation ASR arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audio_curation.pipeline import CurationPipeline

SEED = 20260729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolve_audio(frame: pd.DataFrame, source_root: Path) -> pd.DataFrame:
    result = frame.copy()
    result["path"] = [
        str(path if (path := Path(value)).is_absolute() else source_root / path)
        for value in result.path
    ]
    return result


def make_paths_portable(frame: pd.DataFrame, source_root: Path) -> pd.DataFrame:
    """Store paths relative to the paired Whisper repository."""
    result = frame.copy()
    portable = []
    for value in result.path:
        path = Path(value).resolve()
        try:
            portable.append(path.relative_to(source_root).as_posix())
        except ValueError as error:
            raise ValueError(
                f"Audio path is outside --source-root and cannot be serialized: {path}"
            ) from error
    result["path"] = portable
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", default="experiments/downstream_study")
    parser.add_argument("--audit-size", type=int, default=100)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = Path(args.source_root).resolve()
    output = (repo / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = {
        split: source / "data" / "financial_research" / f"{split}_manifest.parquet"
        for split in ("train", "validation", "test")
    }
    frames = {
        split: resolve_audio(pd.read_parquet(path), source)
        for split, path in manifests.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(frames[left].voice) & set(frames[right].voice)
        if overlap:
            raise RuntimeError(f"Voice leakage between {left} and {right}: {sorted(overlap)}")

    run_metadata = {
        "generator": "scripts/prepare_downstream_study.py",
        "seed": SEED,
        "source_root": str(source),
        "source_manifest_sha256": {
            split: sha256(path) for split, path in manifests.items()
        },
        "git_commit": subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "git_dirty": bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()),
    }
    pipeline = CurationPipeline(
        output_dir=output / "curation",
        remove_near_duplicates=False,
    )
    curated, curation_report = pipeline.run(
        frames["train"], run_metadata=run_metadata
    )
    curation_report_path = output / "curation" / "curation_report.json"
    # Canonicalize generated text before hashing so artifacts are byte-stable
    # across Windows and POSIX hosts.
    write_text_lf(
        curation_report_path,
        json.dumps(
            json_safe(json.loads(curation_report_path.read_text())),
            indent=2,
            allow_nan=False,
        ),
    )
    if not len(curated):
        raise RuntimeError("Quality policy retained zero training rows")
    rng = np.random.default_rng(SEED)
    control = frames["train"].iloc[
        np.sort(rng.choice(len(frames["train"]), len(curated), replace=False))
    ].reset_index(drop=True)
    if len(control) != len(curated):
        raise RuntimeError("Matched-size curation invariant failed")

    common = frames["train"][~frames["train"].is_domain].reset_index(drop=True)
    domain = frames["train"][frames["train"].is_domain].reset_index(drop=True)
    target_size = len(common)
    domain_size = target_size // 2
    common_size = target_size - domain_size
    if len(domain) < domain_size:
        raise RuntimeError("Insufficient domain rows for fixed-size augmentation arm")
    targeted = pd.concat([
        common.iloc[np.sort(rng.choice(len(common), common_size, replace=False))],
        domain.iloc[np.sort(rng.choice(len(domain), domain_size, replace=False))],
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    arms = {
        "curation_control": control,
        "curation_policy": curated,
        "augmentation_control_common": common,
        "augmentation_targeted_50pct": targeted,
        "validation": frames["validation"],
    }
    portable_arms = {
        name: make_paths_portable(frame, source) for name, frame in arms.items()
    }
    for name, frame in portable_arms.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    make_paths_portable(curated, source).to_parquet(
        output / "curation" / "filtered_manifest.parquet", index=False
    )

    inspected = pd.read_parquet(output / "curation" / "filtered_manifest.parquet")
    scored = frames["train"].merge(
        inspected[["id", "qf_passes"]], on="id", how="left"
    )
    # qf_passes is only retained in the filtered output, so recover failures
    # deterministically from membership rather than inventing labels.
    scored["policy_passes"] = scored.id.isin(set(curated.id))
    pass_rows = scored[scored.policy_passes]
    fail_rows = scored[~scored.policy_passes]
    n_fail = min(len(fail_rows), args.audit_size // 2)
    n_pass = min(len(pass_rows), args.audit_size - n_fail)
    audit = pd.concat([
        fail_rows.sample(n=n_fail, random_state=SEED) if n_fail else fail_rows,
        pass_rows.sample(n=n_pass, random_state=SEED) if n_pass else pass_rows,
    ]).sample(frac=1, random_state=SEED)
    audit = audit[["id", "path", "sentence", "voice", "policy_passes"]].copy()
    audit = make_paths_portable(audit, source)
    audit["manual_acceptable"] = pd.NA
    audit["manual_notes"] = ""
    write_text_lf(
        output / "quality_audit_ledger.csv",
        audit.to_csv(index=False, lineterminator="\n"),
    )

    report = {
        "schema_version": 1,
        **run_metadata,
        "arm_counts": {name: len(frame) for name, frame in arms.items()},
        "curation_matched_size": len(control) == len(curated),
        "augmentation_matched_size": len(common) == len(targeted),
        "augmentation_target_domain_fraction": float(targeted.is_domain.mean()),
        "voice_disjoint": True,
        "test_set_used_for_selection": False,
        "manifest_path_base": "paired_whisper_repository_root",
        "quality_audit": {
            "ledger": (
                output / "quality_audit_ledger.csv"
            ).relative_to(repo).as_posix(),
            "n_rows": len(audit),
            "status": "pending_manual_labels",
        },
        "curation_report_sha256": sha256(curation_report_path),
        "arm_manifest_sha256": {
            name: sha256(output / f"{name}.parquet") for name in arms
        },
    }
    write_text_lf(
        output / "study_manifest.json",
        json.dumps(report, indent=2, allow_nan=False),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
