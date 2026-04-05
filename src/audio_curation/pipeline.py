"""
End-to-end curation pipeline.

Orchestrates quality filtering → deduplication → diversity analysis
and writes a summary report alongside the filtered manifest.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .deduplication import DeduplicationEngine
from .diversity import DiversityAnalyzer
from .quality import QualityFilter, QualityThresholds

logger = logging.getLogger(__name__)


class CurationPipeline:
    """
    Full curation run: quality → dedup → diversity.

    Parameters
    ----------
    quality_thresholds : QualityThresholds, optional
        Custom quality thresholds. Defaults are conservative for speech TTS/ASR.
    dedup_threshold : float
        Cosine similarity threshold for near-duplicate detection.
    output_dir : str | Path
        Where to write filtered manifests and reports.
    target_sr : int
        Resample all audio to this rate before analysis (16 kHz recommended).
    """

    def __init__(
        self,
        quality_thresholds: Optional[QualityThresholds] = None,
        dedup_threshold: float = 0.97,
        output_dir: str | Path = "outputs",
        target_sr: int = 16_000,
    ) -> None:
        self.quality_filter = QualityFilter(quality_thresholds)
        self.dedup_engine = DeduplicationEngine(similarity_threshold=dedup_threshold)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_audio(self, path: str | Path) -> tuple[np.ndarray, int]:
        import librosa
        audio, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
        return audio, sr

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        manifest: pd.DataFrame,
        audio_col: str = "path",
        id_col: str = "id",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Curate a dataset given a manifest DataFrame.

        Parameters
        ----------
        manifest : pd.DataFrame
            Must contain `audio_col` (file paths) and `id_col` (unique identifiers).
            Any additional columns (speaker_id, age, gender, accent, sentence) are
            used for diversity analysis if present.
        audio_col : str
            Column containing paths to audio files.
        id_col : str
            Column containing unique sample identifiers.

        Returns
        -------
        filtered_manifest : pd.DataFrame
            Subset of `manifest` after quality + dedup filtering.
        report : dict
            Full curation report with per-stage statistics.
        """
        start = time.time()
        logger.info("Starting curation pipeline on %d samples", len(manifest))
        report: dict = {"n_input": len(manifest)}

        # ---- Stage 1: Load audio ----------------------------------------
        logger.info("Loading audio files...")
        audio_list: list[np.ndarray] = []
        sr_list: list[int] = []
        load_errors: list[str] = []

        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Loading audio"):
            try:
                audio, sr = self._load_audio(row[audio_col])
                audio_list.append(audio)
                sr_list.append(sr)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", row[audio_col], exc)
                audio_list.append(np.zeros(self.target_sr, dtype=np.float32))
                sr_list.append(self.target_sr)
                load_errors.append(str(row[id_col]))

        report["load_errors"] = len(load_errors)

        # ---- Stage 2: Quality filtering ---------------------------------
        logger.info("Running quality filters...")
        keep_mask, quality_reports = self.quality_filter.filter_batch(
            audio_list, sr_list, show_progress=True
        )
        quality_summary = self.quality_filter.summarize(quality_reports)
        report["quality"] = quality_summary

        # Attach quality metrics to manifest
        manifest = manifest.copy()
        manifest["qf_snr_db"] = [r.snr_db for r in quality_reports]
        manifest["qf_duration_sec"] = [r.duration_sec for r in quality_reports]
        manifest["qf_silence_ratio"] = [r.silence_ratio for r in quality_reports]
        manifest["qf_passes"] = keep_mask

        # Filter
        quality_mask = np.array(keep_mask)
        manifest_q = manifest[quality_mask].reset_index(drop=True)
        audio_q = [a for a, k in zip(audio_list, keep_mask) if k]
        sr_q = [s for s, k in zip(sr_list, keep_mask) if k]
        ids_q = manifest_q[id_col].tolist()

        logger.info("After quality filter: %d samples remain", len(manifest_q))

        # ---- Stage 3: Deduplication -------------------------------------
        logger.info("Running deduplication...")
        kept_ids, dedup_report = self.dedup_engine.deduplicate(ids_q, audio_q, sr_q)
        report["deduplication"] = dedup_report

        kept_id_set = set(kept_ids)
        manifest_dedup = manifest_q[manifest_q[id_col].isin(kept_id_set)].reset_index(drop=True)
        logger.info("After dedup: %d samples remain", len(manifest_dedup))

        # ---- Stage 4: Diversity analysis --------------------------------
        logger.info("Running diversity analysis...")
        analyzer_before = DiversityAnalyzer(manifest)
        analyzer_after = DiversityAnalyzer(manifest_dedup)

        report["diversity_before"] = analyzer_before.report()
        report["diversity_after"] = analyzer_after.report()

        # ---- Stage 5: Write outputs -------------------------------------
        elapsed = time.time() - start
        report["elapsed_sec"] = round(elapsed, 1)
        report["n_output"] = len(manifest_dedup)
        report["overall_retention_rate"] = len(manifest_dedup) / max(len(manifest), 1)

        manifest_dedup.to_parquet(self.output_dir / "filtered_manifest.parquet", index=False)

        report_path = self.output_dir / "curation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(
            "Pipeline complete in %.1fs. Input: %d  Output: %d  Retention: %.1f%%",
            elapsed,
            len(manifest),
            len(manifest_dedup),
            100 * report["overall_retention_rate"],
        )

        return manifest_dedup, report
