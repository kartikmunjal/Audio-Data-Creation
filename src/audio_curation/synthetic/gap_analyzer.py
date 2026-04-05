"""
Demographic gap analysis: identifies underrepresented groups in a curated dataset
and produces a synthesis plan that targets exactly those gaps.

This module is the bridge between the curation pipeline and TTS generation.
The curation step already showed that Common Voice English:
  - Skews ~70% male across all age groups
  - Applies stricter SNR filtering that disproportionately removes non-native
    accented speech (slightly higher ambient noise from non-studio environments)
  - Has sparse coverage of speakers over 50 and under 20

Rather than discarding these observations, we use them to drive targeted generation:
compute what a uniform distribution would look like, measure the delta, and
generate just enough synthetic samples to close it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phonetically diverse sentence corpus for TTS generation.
# Selected to cover a broad phoneme inventory: fricatives, stops, nasals,
# vowel variety, consonant clusters, and natural prosody.
# A production system would source from a much larger pool (e.g., the full
# Common Voice sentence corpus or a purpose-built read-speech prompt set).
# ---------------------------------------------------------------------------
SYNTHESIS_SENTENCES = [
    # Short commands / queries
    "Turn off the lights in the kitchen.",
    "What time is the next train to the city?",
    "Please send the report by end of day.",
    "The weather forecast looks promising for the weekend.",
    "Set a reminder for three thirty tomorrow afternoon.",
    # Medium conversational
    "I was thinking we could meet for coffee sometime next week.",
    "The quarterly results exceeded our expectations by a wide margin.",
    "She carefully unwrapped the small package sitting on the kitchen table.",
    "Could you please repeat the last part of your question?",
    "The children were playing loudly in the park across the street.",
    "We need to verify the shipping address before placing the order.",
    "His presentation was both informative and surprisingly entertaining.",
    "The software update should resolve the connectivity issues you reported.",
    "I completely forgot to charge my phone before leaving the house.",
    "The new research suggests a strong link between diet and cognitive function.",
    # Technical / numbers
    "Please enter your six digit verification code to continue.",
    "The flight departs at seven fifteen and arrives at twelve forty.",
    "Call extension four seven nine and ask for the support team.",
    "The package weighs approximately three point five kilograms.",
    "The temperature dropped to minus eight degrees overnight.",
    # Phoneme-rich sentences
    "She sells seashells by the seashore every single summer.",
    "The quick brown fox jumped over the lazy sleeping dog.",
    "Freshly squeezed orange juice tastes far better than the bottled kind.",
    "Thirty-three thousand three hundred and thirty-three thoroughbred horses.",
    "The librarian quietly shelved the unusually thick reference volumes.",
    # Longer read speech
    "Recent advances in natural language processing have fundamentally changed how machines understand and generate human language.",
    "When designing accessible technology, it is important to consider the full range of users, including those with visual, auditory, and motor impairments.",
    "The experiment required participants to listen carefully and repeat each sentence exactly as they heard it, without pausing.",
    "Despite the challenging conditions, the research team managed to collect reliable data from all twenty-four recording sessions.",
    "Modern speech recognition systems can achieve remarkably low error rates on clean audio, but performance degrades significantly in noisy environments.",
    # Varied prosody
    "Are you absolutely certain that is the correct interpretation?",
    "Incredible! I never expected to see results like this so quickly.",
    "Wait — before you go, there's one more thing I need to mention.",
    "The answer, unfortunately, is not as straightforward as we had hoped.",
    "Under no circumstances should you open that file without scanning it first.",
    # Domain coverage
    "The stock market closed slightly higher following the central bank announcement.",
    "Add two cups of flour and one teaspoon of baking powder to the mixing bowl.",
    "The defendant entered a not guilty plea at the preliminary hearing.",
    "Satellite imagery confirmed the presence of the storm system off the eastern coast.",
    "Please review the attached terms and conditions before signing the agreement.",
    "The patient was prescribed a ten day course of antibiotics after the procedure.",
    "Download the latest firmware update from the manufacturer's official website.",
    "The museum's new exhibition opens to the public on the first of next month.",
    "Our customer satisfaction scores improved by twelve percent over the previous quarter.",
    "The hiking trail is rated moderate and takes approximately four hours to complete.",
    "Connect the red wire to the positive terminal and the black wire to the negative.",
    "Her debut novel received widespread critical acclaim and spent six weeks on the bestseller list.",
    "Press the blue button twice to reset the device to its factory settings.",
    "The renovation project is expected to be completed before the end of the fiscal year.",
    "All passengers must present valid identification before boarding the aircraft.",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SynthesisTarget:
    """One generation request: a specific (gender, accent, age_group) combination."""
    gender: str
    accent: str       # maps to a voice group in the VOICE_CATALOG
    age_group: str
    voice_id: str     # specific edge-tts voice name
    n_samples: int
    texts: list[str] = field(default_factory=list)
    priority: float = 0.0   # higher = generated first (largest gap)


# ---------------------------------------------------------------------------
# Gap analyzer
# ---------------------------------------------------------------------------

class GapAnalyzer:
    """
    Identifies demographic gaps in a curated manifest and returns a prioritized
    list of SynthesisTargets to be passed to TTSGenerator.

    Parameters
    ----------
    target_gender_dist : dict | None
        Target fraction per gender label, e.g. {"male": 0.5, "female": 0.5}.
        Defaults to uniform over present labels.
    target_accent_dist : dict | None
        Target fraction per accent. Defaults to uniform over present accents.
    max_synthetic_ratio : float
        Cap synthetic samples at this fraction of the real dataset size.
        Avoids the pathological case of a 1-sample group causing 1000s of generations.
    min_gap_for_generation : float
        Minimum fractional gap (observed - target) before we generate for a group.
        Groups within this tolerance are considered adequately represented.
    """

    def __init__(
        self,
        target_gender_dist: Optional[dict] = None,
        target_accent_dist: Optional[dict] = None,
        max_synthetic_ratio: float = 0.60,
        min_gap_for_generation: float = 0.03,
    ) -> None:
        self.target_gender_dist = target_gender_dist
        self.target_accent_dist = target_accent_dist
        self.max_synthetic_ratio = max_synthetic_ratio
        self.min_gap = min_gap_for_generation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, manifest: pd.DataFrame, voice_catalog: dict) -> list[SynthesisTarget]:
        """
        Compute demographic gaps and return prioritized SynthesisTargets.

        Parameters
        ----------
        manifest : pd.DataFrame
            Filtered manifest from the curation pipeline. Expected columns:
            speaker_id, gender, accent / locale, age, sentence.
        voice_catalog : dict
            Mapping of (gender, accent) → list of edge-tts voice IDs.
            Imported from tts_generator.VOICE_CATALOG.
        """
        targets: list[SynthesisTarget] = []
        n_real = len(manifest)
        n_budget = int(n_real * self.max_synthetic_ratio)

        gender_targets = self._gender_targets(manifest, n_budget // 2, voice_catalog)
        accent_targets = self._accent_targets(manifest, n_budget // 2, voice_catalog)

        targets.extend(gender_targets)
        targets.extend(accent_targets)

        # Deduplicate targets that share a voice and assign texts
        targets = self._assign_texts(targets)

        total_requested = sum(t.n_samples for t in targets)
        logger.info(
            "Gap analysis complete. %d synthesis targets, %d samples requested (%.1f%% of real)",
            len(targets),
            total_requested,
            100 * total_requested / max(n_real, 1),
        )
        return sorted(targets, key=lambda t: t.priority, reverse=True)

    def summary(self, manifest: pd.DataFrame) -> dict:
        """Return a human-readable summary of the demographic distribution."""
        result = {"n_samples": len(manifest)}
        for col in ["gender", "age", "accent", "locale"]:
            if col in manifest.columns:
                dist = manifest[col].value_counts(normalize=True).round(3).to_dict()
                result[f"{col}_distribution"] = dist
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gender_targets(
        self, manifest: pd.DataFrame, budget: int, voice_catalog: dict
    ) -> list[SynthesisTarget]:
        if "gender" not in manifest.columns:
            return []

        observed = manifest["gender"].value_counts(normalize=True)
        labels = [g for g in ["male", "female", "other"] if g in observed.index or g in (self.target_gender_dist or {})]

        if self.target_gender_dist:
            target = {g: self.target_gender_dist.get(g, 0) for g in labels}
        else:
            target = {g: 1.0 / len(labels) for g in labels}

        n_real = len(manifest)
        targets: list[SynthesisTarget] = []

        for g in labels:
            obs_frac = float(observed.get(g, 0.0))
            tgt_frac = target.get(g, 0.0)
            gap = tgt_frac - obs_frac
            if gap < self.min_gap:
                continue

            n_samples = min(int(gap * n_real * 1.1), budget // max(len(labels), 1))
            if n_samples < 1:
                continue

            # Pick voices for this gender from the catalog
            voice_keys = [(g2, acc) for g2, acc in voice_catalog if g2 == g]
            if not voice_keys:
                continue

            # Spread samples across voices
            for i, (vg, vacc) in enumerate(voice_keys[:3]):
                n_voice = n_samples // min(len(voice_keys), 3)
                if n_voice < 1:
                    continue
                voice_id = voice_catalog[(vg, vacc)][0]
                targets.append(SynthesisTarget(
                    gender=g,
                    accent=vacc,
                    age_group="unknown",
                    voice_id=voice_id,
                    n_samples=n_voice,
                    priority=gap,
                ))

        return targets

    def _accent_targets(
        self, manifest: pd.DataFrame, budget: int, voice_catalog: dict
    ) -> list[SynthesisTarget]:
        acc_col = "accent" if "accent" in manifest.columns else "locale" if "locale" in manifest.columns else None
        if acc_col is None:
            return []

        observed = manifest[acc_col].value_counts(normalize=True)

        if self.target_accent_dist:
            target_accents = self.target_accent_dist
        else:
            # Uniform over all accents in the voice catalog + observed
            all_accents = set(acc for _, acc in voice_catalog) | set(observed.index[:10])
            target_accents = {a: 1.0 / len(all_accents) for a in all_accents}

        n_real = len(manifest)
        targets: list[SynthesisTarget] = []

        for acc, tgt_frac in target_accents.items():
            obs_frac = float(observed.get(acc, 0.0))
            gap = tgt_frac - obs_frac
            if gap < self.min_gap:
                continue

            n_samples = min(int(gap * n_real * 1.1), budget // max(len(target_accents), 1))
            if n_samples < 1:
                continue

            # Find a voice for this accent; prefer to alternate genders
            for gender in ["female", "male"]:
                key = (gender, acc)
                if key not in voice_catalog:
                    continue
                voice_id = voice_catalog[key][0]
                targets.append(SynthesisTarget(
                    gender=gender,
                    accent=acc,
                    age_group="unknown",
                    voice_id=voice_id,
                    n_samples=max(1, n_samples // 2),
                    priority=gap * 0.8,  # slightly lower priority than gender gaps
                ))

        return targets

    def _assign_texts(self, targets: list[SynthesisTarget]) -> list[SynthesisTarget]:
        """Round-robin assign synthesis sentences to targets."""
        n_sentences = len(SYNTHESIS_SENTENCES)
        for i, target in enumerate(targets):
            start = (i * 7) % n_sentences   # stagger starting position
            texts: list[str] = []
            for j in range(target.n_samples):
                texts.append(SYNTHESIS_SENTENCES[(start + j) % n_sentences])
            target.texts = texts
        return targets
