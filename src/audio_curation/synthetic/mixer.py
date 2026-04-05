"""
Dataset mixing: constructs combined real+synthetic manifests at specified ratios.

The key design question is *how* to sample at a given ratio. Two strategies:
  1. Random sampling — simple, but at low synthetic ratios it may completely miss
     rare accent groups if the synthetic set is small.
  2. Stratified sampling — ensures each (gender, accent) combination is represented
     proportionally in both the real and synthetic components.

We implement both and default to stratified, since that's what matters for the
demographic gap experiments. The ablation runner uses random sampling to isolate
the pure quantity effect from the stratification effect.

Mixture ratio semantics
-----------------------
  ratio = 0.0 → 100% real, 0% synthetic  (real-only baseline)
  ratio = 0.5 → equal parts real and synthetic (N_real == N_synthetic)
  ratio = 1.0 → 100% synthetic, 0% real  (synthetic-only baseline)

To compare fairly across ratios, we fix the *total* dataset size to the smaller
of (real size, synthetic size) × 2. This avoids conflating size effects with
distribution effects.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataMixer:
    """
    Constructs mixed datasets for ablation experiments.

    Parameters
    ----------
    seed : int
        Random seed for reproducible subsampling.
    fix_total_size : bool
        If True, hold total dataset size constant across ratios (recommended
        for ablations that isolate distribution vs. quantity effects).
    """

    def __init__(self, seed: int = 42, fix_total_size: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        self.fix_total_size = fix_total_size

    # ------------------------------------------------------------------
    # Core mixing
    # ------------------------------------------------------------------

    def create_mix(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        synthetic_ratio: float,
        strategy: str = "random",
    ) -> pd.DataFrame:
        """
        Create a single mixed manifest.

        Parameters
        ----------
        real_df : pd.DataFrame
        synthetic_df : pd.DataFrame
        synthetic_ratio : float
            Fraction of the output that should be synthetic. 0 = real only, 1 = synthetic only.
        strategy : str
            "random" | "stratified". Stratified samples by (gender, accent) proportionally.
        """
        if not 0.0 <= synthetic_ratio <= 1.0:
            raise ValueError(f"synthetic_ratio must be in [0, 1], got {synthetic_ratio}")

        n_real = len(real_df)
        n_syn = len(synthetic_df)

        if self.fix_total_size:
            # Fix total to min(n_real, n_syn) * 2 so all ratios see the same volume
            total = min(n_real, n_syn) * 2 if n_syn > 0 else n_real
        else:
            total = n_real + n_syn

        n_synthetic_in_mix = int(round(total * synthetic_ratio))
        n_real_in_mix = total - n_synthetic_in_mix

        if strategy == "stratified":
            real_sample = self._stratified_sample(real_df, n_real_in_mix)
            syn_sample = self._stratified_sample(synthetic_df, n_synthetic_in_mix)
        else:
            real_sample = self._random_sample(real_df, n_real_in_mix)
            syn_sample = self._random_sample(synthetic_df, n_synthetic_in_mix)

        mixed = pd.concat([real_sample, syn_sample], ignore_index=True)
        mixed["mix_synthetic_ratio"] = synthetic_ratio
        return mixed.sample(frac=1, random_state=int(self.rng.integers(1_000_000))).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Ablation suite
    # ------------------------------------------------------------------

    def create_ablation_splits(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        ratios: Optional[list[float]] = None,
        strategy: str = "random",
    ) -> dict[float, pd.DataFrame]:
        """
        Create one mixed manifest per ratio for the ablation experiment.

        Returns a dict mapping ratio → manifest DataFrame.
        """
        if ratios is None:
            ratios = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

        splits: dict[float, pd.DataFrame] = {}
        for ratio in ratios:
            splits[ratio] = self.create_mix(real_df, synthetic_df, ratio, strategy=strategy)
            logger.info(
                "Mix ratio=%.2f: %d real + %d synthetic = %d total",
                ratio,
                (splits[ratio]["source"] == "real").sum() if "source" in splits[ratio].columns else "?",
                (splits[ratio]["source"] == "synthetic").sum() if "source" in splits[ratio].columns else "?",
                len(splits[ratio]),
            )
        return splits

    # ------------------------------------------------------------------
    # Demographic balance helpers
    # ------------------------------------------------------------------

    def balance_by_gender(
        self,
        df: pd.DataFrame,
        target_dist: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Downsample majority gender groups to approach target_dist.

        Parameters
        ----------
        target_dist : dict | None
            e.g. {"male": 0.5, "female": 0.5}. Defaults to uniform.
        """
        if "gender" not in df.columns:
            return df

        groups = df["gender"].dropna().unique()
        if target_dist is None:
            target_dist = {g: 1.0 / len(groups) for g in groups}

        counts = df["gender"].value_counts()
        # Target is proportional to the smallest group's actual size
        min_group_size = min(
            int(counts.get(g, 0)) for g in target_dist if counts.get(g, 0) > 0
        )
        target_n = {g: int(min_group_size / (target_dist.get(g, 1e-6) / min(target_dist.values()))) for g in groups}

        balanced_parts = []
        for g in groups:
            subset = df[df["gender"] == g]
            n = min(target_n.get(g, len(subset)), len(subset))
            balanced_parts.append(subset.sample(n=n, random_state=42))

        return pd.concat(balanced_parts, ignore_index=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or df.empty:
            return df.iloc[:0]
        n = min(n, len(df))
        return df.sample(n=n, random_state=int(self.rng.integers(1_000_000)), replace=False)

    def _stratified_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Sample proportionally from each (gender, accent) stratum."""
        if n <= 0 or df.empty:
            return df.iloc[:0]
        n = min(n, len(df))

        strat_col = None
        for col in ["accent", "locale", "gender"]:
            if col in df.columns:
                strat_col = col
                break

        if strat_col is None:
            return self._random_sample(df, n)

        groups = df.groupby(strat_col, dropna=False)
        total = len(df)
        parts = []
        allocated = 0

        for name, group in groups:
            group_frac = len(group) / total
            n_group = max(1, int(round(n * group_frac)))
            n_group = min(n_group, len(group))
            parts.append(group.sample(n=n_group, random_state=int(self.rng.integers(1_000_000))))
            allocated += n_group

        result = pd.concat(parts, ignore_index=True)
        # Trim or top-up to exactly n
        if len(result) > n:
            result = result.sample(n=n, random_state=0)
        elif len(result) < n:
            shortfall = n - len(result)
            extra = df[~df.index.isin(result.index)].sample(
                n=min(shortfall, len(df) - len(result)), random_state=1
            )
            result = pd.concat([result, extra], ignore_index=True)

        return result.reset_index(drop=True)
