#!/usr/bin/env python3
"""Fit and audit the locked weak-label audio-quality classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from audio_curation.quality import QualityFilter

SEED = 20260831
TRAIN_SPEAKERS = ["1272", "1462", "174", "1988"]
TEST_SPEAKERS = ["1993", "2035"]
DIRECT = ["duration_sec", "snr_db", "silence_ratio", "is_clipped", "rms_db"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path: str, target_sr: int = 16_000) -> tuple[dict, bool]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    report = QualityFilter().inspect(audio, sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=512, hop_length=160)
    flatness = librosa.feature.spectral_flatness(y=audio)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    features = {
        "duration_sec": report.duration_sec, "snr_db": report.snr_db,
        "silence_ratio": report.silence_ratio, "is_clipped": float(report.is_clipped),
        "rms_db": report.rms_db, "spectral_flatness_mean": float(flatness.mean()),
        "spectral_flatness_std": float(flatness.std()), "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
    }
    for index in range(20):
        features[f"mfcc_{index:02d}_mean"] = float(mfcc[index].mean())
        features[f"mfcc_{index:02d}_std"] = float(mfcc[index].std())
    return features, report.passes


def model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=2, random_state=SEED
    )


def fit_predict(train_x, train_y, test_x) -> tuple[np.ndarray, GradientBoostingClassifier]:
    estimator = model()
    estimator.fit(train_x, train_y, sample_weight=compute_sample_weight("balanced", train_y))
    return estimator.predict_proba(test_x)[:, 1], estimator


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    return {
        "n": len(labels), "n_pass": int(labels.sum()),
        "agreement": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(labels, predictions, labels=[False, True]).ravel().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heuristic-arm", required=True)
    parser.add_argument("--output-dir", default="experiments/learned_filter_study")
    parser.add_argument("--result-dir", default="experiments/results/learned_filter_study")
    parser.add_argument("--audio-prefix", default="data/openslr31_pilot/audio")
    args = parser.parse_args()
    source = Path(args.manifest)
    frame = pd.read_parquet(source).copy()
    frame["speaker_id"] = frame.speaker_id.astype(str)
    if set(frame.speaker_id) != set(TRAIN_SPEAKERS + TEST_SPEAKERS):
        raise ValueError("locked speaker set mismatch")
    rows, labels = [], []
    for path in tqdm(frame.path, desc="Extracting learned-filter features"):
        values, label = extract(path)
        rows.append(values); labels.append(label)
    features = pd.DataFrame(rows)
    feature_names = features.columns.tolist()
    y = np.asarray(labels, dtype=bool)
    probabilities = np.full(len(frame), np.nan)
    fold_reports = {}
    for speaker in TRAIN_SPEAKERS:
        train = frame.speaker_id.isin(set(TRAIN_SPEAKERS) - {speaker}).to_numpy()
        held = (frame.speaker_id == speaker).to_numpy()
        probabilities[held], _ = fit_predict(features.loc[train], y[train], features.loc[held])
        fold_reports[speaker] = metrics(y[held], probabilities[held])
    development = frame.speaker_id.isin(TRAIN_SPEAKERS).to_numpy()
    test = frame.speaker_id.isin(TEST_SPEAKERS).to_numpy()
    probabilities[test], final_model = fit_predict(features.loc[development], y[development], features.loc[test])
    if np.isnan(probabilities).any():
        raise RuntimeError("missing classifier predictions")
    learned = probabilities >= 0.5
    scored = frame.copy()
    scored["heuristic_pass"] = y
    scored["learned_pass_probability"] = probabilities
    scored["learned_pass"] = learned
    scored["classifier_split"] = np.where(test, "untouched_test", "development_oof")
    scored["path"] = [str(PurePosixPath(args.audio_prefix) / Path(value).name) for value in frame.path]

    heuristic_arm = pd.read_parquet(args.heuristic_arm)
    financial = heuristic_arm[heuristic_arm.source_arm == "financial"].copy()
    candidates = scored[development & learned].copy()
    if len(financial) != 147 or len(candidates) < 147:
        raise RuntimeError(f"locked arm unavailable: financial={len(financial)}, learned_pass={len(candidates)}")
    selected = candidates.sample(n=147, random_state=SEED).copy()
    selected["source_arm"] = "openslr31_learned_filter"
    learned_arm = pd.concat([financial, selected], ignore_index=True, sort=False)
    learned_arm = learned_arm.sample(frac=1, random_state=SEED).reset_index(drop=True)

    output = Path(args.output_dir); result_dir = Path(args.result_dir)
    output.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output / "scored_manifest.parquet"
    arm_path = output / "learned_filter_arm.parquet"
    scored.to_parquet(scored_path, index=False); learned_arm.to_parquet(arm_path, index=False)
    disagreements = scored[scored.heuristic_pass != scored.learned_pass].copy()
    disagreements["priority"] = np.where(disagreements.classifier_split == "untouched_test", 0, 1)
    ledger = disagreements.sort_values(["priority", "id"]).head(30)[
        ["id", "path", "speaker_id", "classifier_split", "heuristic_pass",
         "learned_pass", "learned_pass_probability"]
    ].copy()
    ledger["human_acceptable"] = pd.NA; ledger["human_notes"] = ""
    ledger.to_csv(output / "disagreement_ledger.csv", index=False)
    importance = sorted(zip(feature_names, final_model.feature_importances_), key=lambda item: item[1], reverse=True)
    report = {
        "schema_version": 1, "target": "heuristic_quality_filter_pass",
        "weak_labels_not_human_quality": True, "threshold": 0.5,
        "model": {"type": "GradientBoostingClassifier", "n_estimators": 100,
                  "learning_rate": 0.05, "max_depth": 2, "seed": SEED},
        "development_oof": metrics(y[development], probabilities[development]),
        "untouched_speakers": metrics(y[test], probabilities[test]),
        "per_development_speaker": fold_reports,
        "n_disagreements": len(disagreements), "ledger_rows": len(ledger),
        "ledger_human_status": "pending",
        "top_feature_importances": [{"feature": key, "importance": float(value)} for key, value in importance[:15]],
        "learned_pass_development": int((development & learned).sum()),
        "downstream_arm_rows": len(learned_arm),
        "input_hashes": {"manifest": sha256(source), "heuristic_arm": sha256(Path(args.heuristic_arm))},
        "output_hashes": {"scored_manifest": sha256(scored_path), "learned_filter_arm": sha256(arm_path)},
    }
    (result_dir / "classifier_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
