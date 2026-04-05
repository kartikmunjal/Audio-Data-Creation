"""
Unit tests for the synthetic data module.

These tests do NOT call edge-tts or any external service. They verify the
gap analysis logic, mixing math, and evaluation utilities using synthetic
DataFrames and mocked audio.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from audio_curation.synthetic.gap_analyzer import GapAnalyzer, SYNTHESIS_SENTENCES
from audio_curation.synthetic.mixer import DataMixer
from audio_curation.synthetic.evaluator import _safe_wer, _batch_wer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_real_manifest(n=200, gender_skew=0.70, seed=42) -> pd.DataFrame:
    """Simulate a male-skewed real manifest like Common Voice English."""
    rng = np.random.default_rng(seed)
    n_male = int(n * gender_skew)
    genders = ["male"] * n_male + ["female"] * (n - n_male)
    rng.shuffle(genders)

    accents = rng.choice(
        ["american", "british", "australian", "indian", "canadian"],
        size=n,
        p=[0.55, 0.25, 0.10, 0.06, 0.04],
    )
    return pd.DataFrame({
        "id": [f"real_{i:05d}" for i in range(n)],
        "path": [f"/data/real_{i:05d}.wav" for i in range(n)],
        "speaker_id": [f"spk_{i % 30:03d}" for i in range(n)],
        "sentence": [SYNTHESIS_SENTENCES[i % len(SYNTHESIS_SENTENCES)] for i in range(n)],
        "gender": genders,
        "accent": accents,
        "age": rng.choice(["twenties", "thirties", "forties", "fifties"], size=n),
        "duration_sec": rng.uniform(1.5, 8.0, size=n),
        "source": "real",
    })


def make_synthetic_manifest(n=80, seed=99) -> pd.DataFrame:
    """Simulate a synthetic manifest with good gender/accent coverage."""
    rng = np.random.default_rng(seed)
    genders = rng.choice(["male", "female"], size=n)
    accents = rng.choice(
        ["american", "british", "australian", "indian", "canadian", "irish", "south_african"],
        size=n,
    )
    return pd.DataFrame({
        "id": [f"syn_{i:05d}" for i in range(n)],
        "path": [f"/data/syn_{i:05d}.wav" for i in range(n)],
        "speaker_id": [f"tts_voice_{i % 14:02d}" for i in range(n)],
        "sentence": [SYNTHESIS_SENTENCES[i % len(SYNTHESIS_SENTENCES)] for i in range(n)],
        "gender": genders,
        "accent": accents,
        "age": "unknown",
        "duration_sec": rng.uniform(2.0, 6.0, size=n),
        "source": "synthetic",
    })


VOICE_CATALOG_STUB = {
    ("male",   "american"):   ["en-US-GuyNeural"],
    ("female", "american"):   ["en-US-JennyNeural"],
    ("male",   "british"):    ["en-GB-RyanNeural"],
    ("female", "british"):    ["en-GB-SoniaNeural"],
    ("male",   "indian"):     ["en-IN-PrabhatNeural"],
    ("female", "indian"):     ["en-IN-NeerjaNeural"],
    ("female", "irish"):      ["en-IE-EmilyNeural"],
    ("female", "australian"): ["en-AU-NatashaNeural"],
}


# ---------------------------------------------------------------------------
# GapAnalyzer tests
# ---------------------------------------------------------------------------

class TestGapAnalyzer:
    def setup_method(self):
        self.analyzer = GapAnalyzer(max_synthetic_ratio=0.5, min_gap_for_generation=0.02)
        self.real_df = make_real_manifest(n=200, gender_skew=0.70)

    def test_returns_synthesis_targets(self):
        targets = self.analyzer.analyze(self.real_df, VOICE_CATALOG_STUB)
        assert len(targets) > 0, "Expected at least one synthesis target for skewed manifest"

    def test_female_targeted_when_underrepresented(self):
        targets = self.analyzer.analyze(self.real_df, VOICE_CATALOG_STUB)
        female_targets = [t for t in targets if t.gender == "female"]
        assert len(female_targets) > 0, "Female underrepresented — should be targeted"

    def test_texts_assigned(self):
        targets = self.analyzer.analyze(self.real_df, VOICE_CATALOG_STUB)
        for t in targets:
            assert len(t.texts) == t.n_samples, \
                f"Text count {len(t.texts)} != n_samples {t.n_samples} for {t.voice_id}"

    def test_max_synthetic_ratio_respected(self):
        targets = self.analyzer.analyze(self.real_df, VOICE_CATALOG_STUB)
        total_requested = sum(t.n_samples for t in targets)
        max_allowed = int(len(self.real_df) * self.analyzer.max_synthetic_ratio)
        assert total_requested <= max_allowed + 5, \
            f"Requested {total_requested} exceeds budget {max_allowed}"

    def test_no_gender_targets_when_gender_balanced(self):
        # When gender is perfectly balanced AND accent is covered, no gender-driven targets.
        # Use a catalog that only has american voices so no accent gaps exist either.
        american_only_catalog = {
            ("male",   "american"): ["en-US-GuyNeural"],
            ("female", "american"): ["en-US-JennyNeural"],
        }
        balanced = self.real_df.copy()
        n = len(balanced)
        balanced["gender"] = ["male" if i < n // 2 else "female" for i in range(n)]
        balanced["accent"] = "american"
        analyzer = GapAnalyzer(min_gap_for_generation=0.10)
        targets = analyzer.analyze(balanced, american_only_catalog)
        # Gender gap is 0 → no gender targets. Accent gap is also 0 → no accent targets.
        assert len(targets) == 0, f"Expected 0 targets for balanced manifest, got {len(targets)}"

    def test_summary_returns_dict(self):
        summary = self.analyzer.summary(self.real_df)
        assert "n_samples" in summary
        assert summary["n_samples"] == len(self.real_df)

    def test_sorted_by_priority(self):
        targets = self.analyzer.analyze(self.real_df, VOICE_CATALOG_STUB)
        priorities = [t.priority for t in targets]
        assert priorities == sorted(priorities, reverse=True), "Targets should be sorted by priority descending"

    def test_synthesis_sentences_nonempty(self):
        assert len(SYNTHESIS_SENTENCES) >= 30


# ---------------------------------------------------------------------------
# DataMixer tests
# ---------------------------------------------------------------------------

class TestDataMixer:
    def setup_method(self):
        self.real_df = make_real_manifest(n=200)
        self.syn_df = make_synthetic_manifest(n=80)
        self.mixer = DataMixer(seed=42, fix_total_size=True)

    def test_ratio_zero_is_real_only(self):
        mixed = self.mixer.create_mix(self.real_df, self.syn_df, synthetic_ratio=0.0)
        assert "source" in mixed.columns
        assert (mixed["source"] == "synthetic").sum() == 0

    def test_ratio_one_is_synthetic_only(self):
        mixed = self.mixer.create_mix(self.real_df, self.syn_df, synthetic_ratio=1.0)
        assert (mixed["source"] == "real").sum() == 0

    def test_ratio_half_balanced(self):
        mixed = self.mixer.create_mix(self.real_df, self.syn_df, synthetic_ratio=0.5)
        n_syn = (mixed["source"] == "synthetic").sum()
        n_real = (mixed["source"] == "real").sum()
        # With fix_total_size, total = min(200, 80) * 2 = 160; each half = 80
        assert abs(n_syn - n_real) <= 2, f"Expected balanced: n_real={n_real} n_syn={n_syn}"

    def test_total_size_fixed_across_ratios(self):
        """
        When fix_total_size=True, total = min(n_real, n_syn) * 2 = 160.
        This holds as long as neither pool is exhausted by the requested ratio.
        With 200 real and 80 synthetic: ratios ≤ 0.5 are within capacity.
        At ratio=0.75 we'd need 120 synthetic but only have 80, so the mixer
        caps and returns fewer samples — that is correct, documented behavior.
        """
        # Ratios where neither pool is exhausted at size 160
        totals = []
        for ratio in [0.0, 0.25, 0.5]:
            mixed = self.mixer.create_mix(self.real_df, self.syn_df, ratio)
            totals.append(len(mixed))
        assert len(set(totals)) == 1, f"Inconsistent totals within capacity: {totals}"

        # At ratio=0.75, synthetic pool is too small to hit 120 → total < 160
        high_mix = self.mixer.create_mix(self.real_df, self.syn_df, 0.75)
        assert len(high_mix) < 160, "High synthetic ratio should produce fewer samples when pool is exhausted"

    def test_ablation_splits_keys(self):
        ratios = [0.0, 0.25, 0.5, 1.0]
        splits = self.mixer.create_ablation_splits(self.real_df, self.syn_df, ratios=ratios)
        assert set(splits.keys()) == set(ratios)

    def test_stratified_vs_random_same_total(self):
        n_rand = len(self.mixer.create_mix(self.real_df, self.syn_df, 0.5, strategy="random"))
        n_strat = len(self.mixer.create_mix(self.real_df, self.syn_df, 0.5, strategy="stratified"))
        assert abs(n_rand - n_strat) <= 5

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            self.mixer.create_mix(self.real_df, self.syn_df, synthetic_ratio=1.5)

    def test_empty_synthetic_still_works(self):
        empty_syn = self.syn_df.iloc[:0]
        mixed = self.mixer.create_mix(self.real_df, empty_syn, synthetic_ratio=0.0)
        assert len(mixed) > 0

    def test_gender_balance(self):
        before_male_frac = (self.real_df["gender"] == "male").mean()
        assert before_male_frac > 0.60  # fixture is 70% male

        balanced = self.mixer.balance_by_gender(self.real_df, {"male": 0.5, "female": 0.5})
        after_male_frac = (balanced["gender"] == "male").mean()
        assert abs(after_male_frac - 0.5) < 0.10, f"Expected ~50% male, got {after_male_frac:.1%}"


# ---------------------------------------------------------------------------
# Evaluator / WER tests
# ---------------------------------------------------------------------------

class TestWER:
    def test_perfect_match(self):
        assert _safe_wer("hello world", "hello world") == 0.0

    def test_complete_mismatch(self):
        wer = _safe_wer("one two three", "four five six")
        assert wer == 1.0

    def test_one_substitution(self):
        wer = _safe_wer("the cat sat here", "the cat sat there")
        assert abs(wer - 0.25) < 0.01  # 1/4 words wrong

    def test_empty_reference(self):
        wer = _safe_wer("", "hello")
        assert wer == 1.0

    def test_both_empty(self):
        assert _safe_wer("", "") == 0.0

    def test_batch_wer_aggregation(self):
        refs = ["hello world", "the cat sat"]
        hyps = ["hello world", "the dog sat"]
        wer = _batch_wer(refs, hyps)
        # 0 errors + 1 error / 5 total words = 0.2
        assert abs(wer - 0.2) < 0.05

    def test_batch_wer_all_correct(self):
        refs = ["a b c", "d e f"]
        assert _batch_wer(refs, refs) == 0.0


# ---------------------------------------------------------------------------
# Integration: mixing improves diversity
# ---------------------------------------------------------------------------

class TestMixingImproveDiversity:
    def test_mixing_increases_accent_entropy(self):
        from audio_curation.diversity import DiversityAnalyzer

        real_df = make_real_manifest(n=200, gender_skew=0.70)
        syn_df = make_synthetic_manifest(n=80)

        mixer = DataMixer(seed=0, fix_total_size=False)
        mixed = mixer.create_mix(real_df, syn_df, synthetic_ratio=0.5, strategy="random")

        div_real = DiversityAnalyzer(real_df).diversity_score()
        div_mixed = DiversityAnalyzer(mixed).diversity_score()

        assert div_mixed.get("accent", 0) >= div_real.get("accent", 0), \
            "Mixed data should have >= accent entropy than real-only"

    def test_mixing_increases_gender_entropy(self):
        from audio_curation.diversity import DiversityAnalyzer

        real_df = make_real_manifest(n=200, gender_skew=0.80)  # very skewed
        syn_df = make_synthetic_manifest(n=100)

        mixer = DataMixer(seed=0, fix_total_size=False)
        mixed = mixer.create_mix(real_df, syn_df, synthetic_ratio=0.5)

        div_real = DiversityAnalyzer(real_df).diversity_score()
        div_mixed = DiversityAnalyzer(mixed).diversity_score()

        assert div_mixed.get("gender", 0) >= div_real.get("gender", 0), \
            "Mixed data should have >= gender entropy than real-only"
