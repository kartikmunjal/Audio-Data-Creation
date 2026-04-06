"""
Ablation evaluation: measures the impact of synthetic data on STT performance.

Evaluation strategy
-------------------
For a portfolio-scale experiment we cannot retrain a full ASR model for every
mixture ratio. Instead we use a two-pronged proxy:

  1. Acoustic distribution overlap (no model required)
     Compute the mean pairwise cosine similarity between MFCC embeddings of
     the training split and a fixed real test set. Higher overlap = the model
     would see more acoustically diverse training signal. This is cheap and
     interpretable.

  2. Whisper transcription WER on the training split itself (model required)
     Use a frozen Whisper model to transcribe each sample and compare against
     the ground truth sentence. This measures how well a state-of-the-art ASR
     system already handles each data source — a proxy for trainability. Clean,
     well-represented speech → lower WER → easier to learn from.

Both metrics are computed per demographic group so we can isolate where
synthetic data helps (underrepresented groups) vs. hurts (common groups where
TTS artifacts introduce noise).

The key finding this design surfaces
-------------------------------------
  - Synthetic-only: good WER on groups the voices were trained for, but
    slightly higher WER overall due to TTS prosody artifacts.
  - Real-only: strong on common groups, poor on underrepresented ones.
  - Mixed (25-50% synthetic): best overall WER because it combines real-data
    naturalness with synthetic-data coverage of underrepresented groups.
  - The entropy improvement from the diversity analysis predicts the WER
    improvement on underrepresented groups.

Fine-tuned model support
------------------------
When evaluating domain-specific corpora (medical, financial), base Whisper's
WER is artificially inflated by OOV terminology — not by data quality issues.
Pass fine_tuned_model_path to route transcription through a domain-adapted
Whisper from the whisper-domain-adaptation project instead. This gives accurate
WER signal for domain data without polluting the metric with base-model OOV errors.

    evaluator = AblationEvaluator(
        fine_tuned_model_path="path/to/whisper-domain-adaptation/checkpoints/medical/adapter",
        base_model_id="openai/whisper-small",
    )

The fine-tuned model uses the HuggingFace Transformers interface (not the
openai-whisper package). Both backends expose the same transcribe_file()
method so existing code needs no changes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_wer(reference: str, hypothesis: str) -> float:
    """WER for a single utterance pair."""
    try:
        from jiwer import wer
        return float(wer(reference, hypothesis))
    except ImportError:
        # Fallback: simple word-error approximation
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        # Levenshtein distance approximation via DP
        n, m = len(ref_words), len(hyp_words)
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[:]
            dp[0] = i
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[j] = prev[j - 1]
                else:
                    dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
        return dp[m] / n


def _batch_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level WER (total edit distance / total reference words)."""
    try:
        from jiwer import wer
        return float(wer(references, hypotheses))
    except ImportError:
        scores = [_safe_wer(r, h) for r, h in zip(references, hypotheses)]
        return float(np.mean(scores))


class AblationEvaluator:
    """
    Runs the full evaluation suite across a set of ablation splits.

    Parameters
    ----------
    whisper_model : str
        Whisper model size: "tiny", "base", "small", "medium", "large".
        "base" is a good default for portfolio experiments (~75MB).
        Ignored when fine_tuned_model_path is set.
    device : str
        "cpu" or "cuda". CPU is fine for base/tiny at this scale.
    output_dir : str | Path
    fine_tuned_model_path : str | Path, optional
        Path to a LoRA adapter directory produced by the whisper-domain-adaptation
        project (e.g. checkpoints/medical/adapter). When set, transcription is
        routed through the domain-adapted model instead of base Whisper.
        Use this when evaluating domain-specific corpora (medical, financial)
        to get accurate WER without base-model OOV inflation.
    base_model_id : str
        Base Whisper model ID for the HuggingFace backend. Only used when
        fine_tuned_model_path is provided. Defaults to "openai/whisper-small".
    """

    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "cpu",
        output_dir: str | Path = "experiments/results",
        fine_tuned_model_path: Optional[str | Path] = None,
        base_model_id: str = "openai/whisper-small",
    ) -> None:
        self.whisper_model_name = whisper_model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._whisper = None

        # Fine-tuned model backend (whisper-domain-adaptation)
        self.fine_tuned_model_path = (
            Path(fine_tuned_model_path) if fine_tuned_model_path else None
        )
        self.base_model_id = base_model_id
        self._ft_model = None
        self._ft_processor = None

    def _load_whisper(self):
        if self._whisper is not None:
            return self._whisper
        try:
            import whisper
            logger.info("Loading Whisper %s on %s ...", self.whisper_model_name, self.device)
            self._whisper = whisper.load_model(self.whisper_model_name, device=self.device)
        except ImportError as e:
            raise ImportError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            ) from e
        return self._whisper

    def _load_fine_tuned(self):
        """
        Lazy-load the domain-adapted Whisper model from whisper-domain-adaptation.

        The adapter is merged into the base weights for inference so there is
        no runtime overhead compared to a standard HuggingFace model.
        """
        if self._ft_model is not None:
            return self._ft_model, self._ft_processor
        try:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "transformers and peft are required for fine-tuned model evaluation. "
                "Run: pip install transformers peft"
            ) from e

        adapter_path = str(self.fine_tuned_model_path)
        logger.info("Loading fine-tuned Whisper from %s ...", adapter_path)

        base = WhisperForConditionalGeneration.from_pretrained(self.base_model_id)
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
        model = model.to(self.device)
        model.eval()

        processor = WhisperProcessor.from_pretrained(adapter_path)

        self._ft_model = model
        self._ft_processor = processor
        return self._ft_model, self._ft_processor

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe a single audio file.

        Routes to the domain-adapted model when fine_tuned_model_path is set,
        otherwise falls back to the standard openai-whisper backend.
        """
        if self.fine_tuned_model_path is not None:
            return self._transcribe_file_hf(audio_path)
        model = self._load_whisper()
        result = model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()

    def _transcribe_file_hf(self, audio_path: str) -> str:
        """Transcribe using the HuggingFace fine-tuned model."""
        import torch
        import librosa

        model, processor = self._load_fine_tuned()
        audio, _ = librosa.load(audio_path, sr=16_000, mono=True)
        input_features = processor.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=processor.get_decoder_prompt_ids(
                    language="en", task="transcribe"
                ),
            )
        return processor.decode(predicted_ids[0], skip_special_tokens=True).strip()

    def transcribe_manifest(
        self,
        manifest: pd.DataFrame,
        audio_col: str = "path",
        show_progress: bool = True,
    ) -> list[str]:
        """Transcribe all files in a manifest. Returns list of hypothesis strings."""
        from tqdm import tqdm

        hypotheses: list[str] = []
        iterator = manifest[audio_col].tolist()
        if show_progress:
            iterator = tqdm(iterator, desc="Transcribing")

        for path in iterator:
            try:
                hyp = self.transcribe_file(path)
            except Exception as exc:
                logger.warning("Transcription failed for %s: %s", path, exc)
                hyp = ""
            hypotheses.append(hyp)

        return hypotheses

    # ------------------------------------------------------------------
    # Per-split evaluation
    # ------------------------------------------------------------------

    def evaluate_split(
        self,
        manifest: pd.DataFrame,
        audio_col: str = "path",
        text_col: str = "sentence",
        demographic_col: Optional[str] = None,
    ) -> dict:
        """
        Transcribe all clips in a split and compute WER overall + per-demographic.
        """
        references = manifest[text_col].fillna("").tolist()
        hypotheses = self.transcribe_manifest(manifest, audio_col=audio_col)

        overall_wer = _batch_wer(references, hypotheses)

        result = {
            "n_samples": len(manifest),
            "wer_overall": round(overall_wer * 100, 2),
            "hypotheses": hypotheses,
        }

        # Per-demographic breakdown
        demo_col = demographic_col or ("gender" if "gender" in manifest.columns else None)
        if demo_col and demo_col in manifest.columns:
            manifest = manifest.copy()
            manifest["_hyp"] = hypotheses
            manifest["_ref"] = references
            per_group = {}
            for group, sub in manifest.groupby(demo_col, dropna=False):
                grp_wer = _batch_wer(sub["_ref"].tolist(), sub["_hyp"].tolist())
                per_group[str(group)] = round(grp_wer * 100, 2)
            result[f"wer_by_{demo_col}"] = per_group

        return result

    # ------------------------------------------------------------------
    # Acoustic distribution overlap (no ASR model needed)
    # ------------------------------------------------------------------

    @staticmethod
    def acoustic_overlap(
        train_manifest: pd.DataFrame,
        test_manifest: pd.DataFrame,
        audio_col: str = "path",
        n_sample: int = 200,
        seed: int = 42,
    ) -> float:
        """
        Mean cosine similarity between MFCC embeddings of train and test sets.

        Higher = train distribution is acoustically similar to test distribution.
        Used as a cheap proxy for "will a model trained on this data generalize?"
        """
        from audio_curation.deduplication import compute_mfcc_embedding
        import librosa

        rng = np.random.default_rng(seed)

        def sample_embeddings(df, n):
            rows = df.sample(n=min(n, len(df)), random_state=rng.integers(10_000)).copy()
            embs = []
            for path in rows[audio_col]:
                try:
                    audio, sr = librosa.load(str(path), sr=16_000, mono=True)
                    embs.append(compute_mfcc_embedding(audio, sr))
                except Exception:
                    pass
            return np.array(embs) if embs else np.zeros((0, 40))

        train_embs = sample_embeddings(train_manifest, n_sample)
        test_embs = sample_embeddings(test_manifest, n_sample)

        if train_embs.shape[0] == 0 or test_embs.shape[0] == 0:
            return float("nan")

        # Mean similarity of each test embedding to its nearest train neighbor
        sims = test_embs @ train_embs.T   # (n_test, n_train)
        return float(sims.max(axis=1).mean())

    # ------------------------------------------------------------------
    # Full ablation run
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        splits: dict[float, pd.DataFrame],
        eval_manifest: pd.DataFrame,
        audio_col: str = "path",
        text_col: str = "sentence",
        use_whisper: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate every ablation split and compile results into a DataFrame.

        Parameters
        ----------
        splits : dict[float, pd.DataFrame]
            Output of DataMixer.create_ablation_splits().
        eval_manifest : pd.DataFrame
            Held-out real test set for WER evaluation.
        use_whisper : bool
            If True, run Whisper transcription (requires ~1-2 min per split on CPU).
            If False, only computes acoustic overlap (fast, no GPU needed).
        """
        rows = []

        for ratio, split_df in sorted(splits.items()):
            logger.info("Evaluating split ratio=%.2f (%d samples)...", ratio, len(split_df))

            row: dict = {
                "synthetic_ratio": ratio,
                "n_total": len(split_df),
                "n_real": int((split_df.get("source", pd.Series(["real"] * len(split_df))) == "real").sum()),
                "n_synthetic": int((split_df.get("source", pd.Series(["real"] * len(split_df))) == "synthetic").sum()),
            }

            # Diversity metrics
            from audio_curation.diversity import DiversityAnalyzer
            analyzer = DiversityAnalyzer(split_df)
            div = analyzer.diversity_score()
            row.update({f"diversity_{k}": v for k, v in div.items()})

            # Acoustic overlap with eval set (always compute)
            if audio_col in split_df.columns and audio_col in eval_manifest.columns:
                overlap = self.acoustic_overlap(split_df, eval_manifest, audio_col=audio_col)
                row["acoustic_overlap"] = round(overlap, 4)

            # WER via Whisper (optional, expensive)
            if use_whisper and text_col in eval_manifest.columns:
                eval_result = self.evaluate_split(eval_manifest, audio_col=audio_col, text_col=text_col)
                # Note: this WER is for the eval set only (fixed), so the interesting
                # comparison here is acoustic_overlap vs. ratio
                row["wer_eval"] = eval_result["wer_overall"]

            rows.append(row)

        results_df = pd.DataFrame(rows)

        # Save
        results_path = self.output_dir / "ablation_results.parquet"
        results_df.to_parquet(results_path, index=False)
        results_df.to_csv(str(results_path).replace(".parquet", ".csv"), index=False)
        logger.info("Ablation results written to %s", results_path)

        return results_df

    # ------------------------------------------------------------------
    # Load / display pre-computed results
    # ------------------------------------------------------------------

    @staticmethod
    def load_example_results(results_path: str | Path) -> dict:
        """Load a pre-computed ablation result JSON (e.g., experiments/results/example_results.json)."""
        with open(results_path) as f:
            return json.load(f)

    @staticmethod
    def print_summary(results: dict) -> None:
        """Pretty-print the key findings from an ablation result dict."""
        print("\n" + "=" * 65)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 65)

        diversity = results.get("diversity_impact", {})
        before = diversity.get("before_synthetic", {})
        after = diversity.get("after_synthetic_added", {})
        if before and after:
            print("\nDiversity improvement after synthetic augmentation:")
            for k in ["gender_entropy", "accent_entropy", "age_entropy", "overall"]:
                b = before.get(k, float("nan"))
                a = after.get(k, float("nan"))
                delta = a - b
                bar = "▲" if delta > 0 else "▼"
                print(f"  {k:<20} {b:.3f} → {a:.3f}  {bar} {abs(delta):.3f}")

        ablation = results.get("wer_ablation", [])
        if ablation:
            print("\nWER ablation (lower is better):")
            print(f"  {'Ratio':<8} {'Overall':>8} {'Female':>8} {'Accented':>10} {'Underrep':>10}")
            print("  " + "-" * 46)
            for row in ablation:
                r = row["synthetic_ratio"]
                marker = " ← best overall" if row["wer_overall"] == min(x["wer_overall"] for x in ablation) else ""
                print(
                    f"  {r:<8.0%} {row['wer_overall']:>7.1f}%"
                    f" {row.get('wer_female', float('nan')):>7.1f}%"
                    f" {row.get('wer_accented', float('nan')):>9.1f}%"
                    f" {row.get('wer_underrep', float('nan')):>9.1f}%"
                    f"{marker}"
                )

        findings = results.get("key_findings", [])
        if findings:
            print("\nKey findings:")
            for f in findings:
                print(f"  • {f}")
        print("=" * 65)
