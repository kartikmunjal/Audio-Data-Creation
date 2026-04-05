"""
Diversity analysis for audio datasets.

Audio diversity has dimensions that text doesn't:
  - Speaker demographics  : age, gender, dialect/accent
  - Acoustic environment  : indoor/outdoor, microphone type, background noise type
  - Linguistic coverage   : vocabulary, sentence length, domain (read vs. spontaneous)
  - Temporal distribution : recording year/session spread (recording equipment evolves)

We quantify each dimension and flag when a training split would be unbalanced.
An unbalanced split produces models that work well for one group and poorly for others —
a pattern that's hard to catch without explicit measurement.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of a count distribution; 0 for empty."""
    counts = counts[counts > 0].astype(float)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def _normalized_entropy(counts: np.ndarray) -> float:
    """Entropy normalized to [0, 1] by log(n_categories)."""
    n = (counts > 0).sum()
    if n <= 1:
        return 0.0
    return _safe_entropy(counts) / np.log(n)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class DiversityAnalyzer:
    """
    Computes diversity metrics from a metadata DataFrame.

    Expected columns (all optional — metrics are skipped if absent):
        speaker_id, age, gender, accent, locale, sentence, duration_sec, split
    """

    KNOWN_AGE_BINS = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties", "nineties"]
    KNOWN_GENDERS = ["male", "female", "other"]

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self._normalize_columns()

    def _normalize_columns(self) -> None:
        """Lowercase string columns for consistent grouping."""
        for col in ["gender", "age", "accent", "locale"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.lower().str.strip().replace("", np.nan)

    # ------------------------------------------------------------------
    # Speaker demographics
    # ------------------------------------------------------------------

    def speaker_counts(self) -> pd.Series:
        """Number of clips per unique speaker."""
        if "speaker_id" not in self.df.columns:
            return pd.Series(dtype=int)
        return self.df.groupby("speaker_id").size().sort_values(ascending=False)

    def gender_distribution(self) -> pd.DataFrame:
        """Count and percentage per gender label."""
        if "gender" not in self.df.columns:
            return pd.DataFrame()
        counts = self.df["gender"].value_counts(dropna=False).rename("count")
        pct = (counts / counts.sum() * 100).round(2).rename("pct")
        return pd.concat([counts, pct], axis=1)

    def age_distribution(self) -> pd.DataFrame:
        """Count and percentage per age bucket."""
        if "age" not in self.df.columns:
            return pd.DataFrame()
        counts = self.df["age"].value_counts(dropna=False).rename("count")
        pct = (counts / counts.sum() * 100).round(2).rename("pct")
        return pd.concat([counts, pct], axis=1)

    def speaker_imbalance_ratio(self) -> float:
        """
        Max clips / median clips across speakers.

        A ratio > 50 means a handful of speakers dominate the dataset —
        the model may overfit to those speakers' acoustic style.
        """
        counts = self.speaker_counts()
        if counts.empty:
            return float("nan")
        return float(counts.max() / (counts.median() + 1e-6))

    # ------------------------------------------------------------------
    # Accent / locale
    # ------------------------------------------------------------------

    def accent_distribution(self) -> pd.DataFrame:
        """Distribution over accent labels (Common Voice uses locale codes)."""
        col = "accent" if "accent" in self.df.columns else "locale" if "locale" in self.df.columns else None
        if col is None:
            return pd.DataFrame()
        counts = self.df[col].value_counts(dropna=False).rename("count")
        pct = (counts / counts.sum() * 100).round(2).rename("pct")
        return pd.concat([counts, pct], axis=1)

    def accent_entropy(self) -> float:
        """Normalized Shannon entropy of accent distribution (higher = more diverse)."""
        col = "accent" if "accent" in self.df.columns else "locale" if "locale" in self.df.columns else None
        if col is None:
            return float("nan")
        counts = self.df[col].value_counts().values
        return _normalized_entropy(counts)

    # ------------------------------------------------------------------
    # Linguistic / domain
    # ------------------------------------------------------------------

    def sentence_length_stats(self) -> dict:
        """Word-count statistics over the sentence column."""
        if "sentence" not in self.df.columns:
            return {}
        lengths = self.df["sentence"].dropna().str.split().str.len()
        return {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "p5": float(lengths.quantile(0.05)),
            "p95": float(lengths.quantile(0.95)),
            "std": float(lengths.std()),
        }

    def vocabulary_size(self) -> int:
        """Unique word count across all sentences (rough domain breadth proxy)."""
        if "sentence" not in self.df.columns:
            return 0
        words = self.df["sentence"].dropna().str.lower().str.split().explode()
        return int(words.nunique())

    def domain_coverage(self) -> dict:
        """
        Rough domain bucketing by sentence length.

        Short (1-5 words)  → commands / labels
        Medium (6-15 words) → phrases / news headlines
        Long (16+ words)   → read speech / conversational
        """
        if "sentence" not in self.df.columns:
            return {}
        lengths = self.df["sentence"].dropna().str.split().str.len()
        return {
            "commands_pct": float((lengths <= 5).mean() * 100),
            "phrases_pct": float(((lengths > 5) & (lengths <= 15)).mean() * 100),
            "read_speech_pct": float((lengths > 15).mean() * 100),
        }

    # ------------------------------------------------------------------
    # Duration stats
    # ------------------------------------------------------------------

    def duration_stats(self) -> dict:
        """Summary statistics over clip duration."""
        if "duration_sec" not in self.df.columns:
            return {}
        d = self.df["duration_sec"].dropna()
        return {
            "total_hours": float(d.sum() / 3600),
            "mean_sec": float(d.mean()),
            "median_sec": float(d.median()),
            "p5_sec": float(d.quantile(0.05)),
            "p95_sec": float(d.quantile(0.95)),
            "std_sec": float(d.std()),
        }

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    def diversity_score(self) -> dict:
        """
        Composite diversity scores in [0, 1] across demographic and linguistic axes.

        These are diagnostic, not absolute ground truth. Use them to compare
        two candidate splits or to track how curation choices affect balance.
        """
        scores: dict[str, float] = {}

        if "gender" in self.df.columns:
            counts = self.df["gender"].value_counts().values
            scores["gender"] = _normalized_entropy(counts)

        if "age" in self.df.columns:
            counts = self.df["age"].value_counts().values
            scores["age"] = _normalized_entropy(counts)

        accent_col = "accent" if "accent" in self.df.columns else "locale" if "locale" in self.df.columns else None
        if accent_col:
            counts = self.df[accent_col].value_counts().values
            scores["accent"] = _normalized_entropy(counts)

        if "speaker_id" in self.df.columns:
            spk = self.speaker_counts()
            # Normalize: perfectly uniform = 1, monopoly = 0
            probs = spk / spk.sum()
            scores["speaker_balance"] = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(len(spk) + 1e-12))

        if scores:
            scores["overall"] = float(np.mean(list(scores.values())))

        return scores

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """Aggregate all metrics into a single dictionary."""
        return {
            "n_samples": len(self.df),
            "n_speakers": self.df["speaker_id"].nunique() if "speaker_id" in self.df.columns else None,
            "gender_distribution": self.gender_distribution().to_dict() if not self.gender_distribution().empty else {},
            "age_distribution": self.age_distribution().to_dict() if not self.age_distribution().empty else {},
            "accent_distribution": self.accent_distribution().head(20).to_dict() if not self.accent_distribution().empty else {},
            "accent_entropy": self.accent_entropy(),
            "speaker_imbalance_ratio": self.speaker_imbalance_ratio(),
            "sentence_length_stats": self.sentence_length_stats(),
            "vocabulary_size": self.vocabulary_size(),
            "domain_coverage": self.domain_coverage(),
            "duration_stats": self.duration_stats(),
            "diversity_scores": self.diversity_score(),
        }
