#!/usr/bin/env python3
"""Rebase a curated manifest for the shared Whisper `id/path/sentence` contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-prefix", required=True,
                        help="Path relative to the Whisper repository root")
    args = parser.parse_args()
    source = Path(args.input)
    frame = pd.read_parquet(source)
    required = {"id", "path", "sentence"}
    if missing := required - set(frame.columns):
        raise ValueError(f"manifest missing shared columns: {sorted(missing)}")
    if frame["id"].duplicated().any() or frame["id"].isna().any():
        raise ValueError("id must be non-null and unique")
    exported = frame.copy()
    exported["path"] = [
        str(PurePosixPath(args.audio_prefix) / Path(value).name)
        for value in frame["path"]
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exported.to_parquet(output, index=False)
    provenance = {
        "schema_version": 1,
        "generator": "scripts/export_whisper_manifest.py",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "audio_prefix": args.audio_prefix,
        "n_samples": len(exported),
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
