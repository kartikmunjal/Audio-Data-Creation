"""
Evaluate a curated manifest using a domain-adapted Whisper model.

This closes the feedback loop between Audio-Data-Creation and
whisper-domain-adaptation:

    [Audio-Data-Creation] curate corpus → filtered_manifest.parquet
        ↓
    [whisper-domain-adaptation] fine-tune Whisper → checkpoints/medical/adapter
        ↓  (this script)
    [Audio-Data-Creation] evaluate new curation runs with domain-aware WER

Why this matters
----------------
Base Whisper reports inflated WER on medical/financial corpora because it's
penalised for OOV terms it was never trained on — not because the audio is
low quality. Using a fine-tuned model here gives a cleaner signal: WER
differences between ablation splits reflect data quality and diversity,
not base-model vocabulary gaps.

As the fine-tuned model improves (more curation iterations), the WER numbers
become more precise, which in turn guides better curation thresholds.

Usage
-----
# Medical corpus evaluation
python scripts/evaluate_with_domain_model.py \
    --manifest outputs/filtered_manifest.parquet \
    --model_path ../whisper-domain-adaptation/checkpoints/medical/adapter \
    --base_model openai/whisper-small \
    --output experiments/results/domain_eval.json

# With demographic breakdown
python scripts/evaluate_with_domain_model.py \
    --manifest outputs/filtered_manifest.parquet \
    --model_path ../whisper-domain-adaptation/checkpoints/medical/adapter \
    --demographic_col gender \
    --output experiments/results/domain_eval_by_gender.json

# Compare base vs fine-tuned side by side
python scripts/evaluate_with_domain_model.py \
    --manifest outputs/filtered_manifest.parquet \
    --model_path ../whisper-domain-adaptation/checkpoints/medical/adapter \
    --compare_base \
    --output experiments/results/domain_eval_comparison.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_curation.synthetic.evaluator import AblationEvaluator

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate curation manifest with a domain-adapted Whisper model"
    )
    p.add_argument("--manifest", required=True,
                   help="Parquet manifest from run_pipeline.py (filtered_manifest.parquet)")
    p.add_argument("--model_path", required=True,
                   help="Path to LoRA adapter from whisper-domain-adaptation "
                        "(e.g. ../whisper-domain-adaptation/checkpoints/medical/adapter)")
    p.add_argument("--base_model", default="openai/whisper-small",
                   help="Base Whisper model ID (must match what was fine-tuned)")
    p.add_argument("--demographic_col", default=None,
                   help="Column to use for per-group WER breakdown (e.g. 'gender', 'accent')")
    p.add_argument("--text_col", default="sentence")
    p.add_argument("--audio_col", default="path")
    p.add_argument("--compare_base", action="store_true",
                   help="Also run base Whisper evaluation for comparison")
    p.add_argument("--output", default="experiments/results/domain_eval.json")
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps (auto-detected if not specified)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap number of samples to evaluate (for quick checks)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main() -> None:
    args = parse_args()
    device = args.device or auto_device()
    logger.info("Using device: %s", device)

    df = pd.read_parquet(args.manifest)
    logger.info("Loaded manifest: %d samples", len(df))

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        logger.info("Subsampled to %d samples", len(df))

    result: dict = {
        "manifest": args.manifest,
        "n_samples": len(df),
        "model_path": args.model_path,
        "base_model": args.base_model,
        "device": device,
    }

    # ── Fine-tuned model evaluation ───────────────────────────────────────────
    logger.info("Evaluating with domain-adapted model...")
    ft_evaluator = AblationEvaluator(
        fine_tuned_model_path=args.model_path,
        base_model_id=args.base_model,
        device=device,
        output_dir=str(Path(args.output).parent),
    )

    ft_result = ft_evaluator.evaluate_split(
        manifest=df,
        audio_col=args.audio_col,
        text_col=args.text_col,
        demographic_col=args.demographic_col,
    )
    result["fine_tuned"] = {k: v for k, v in ft_result.items() if k != "hypotheses"}

    logger.info("Fine-tuned WER: %.2f%%", ft_result["wer_overall"])
    if args.demographic_col:
        for group, wer in ft_result.get(f"wer_by_{args.demographic_col}", {}).items():
            logger.info("  %-20s  %.2f%%", group, wer)

    # ── Optional base Whisper comparison ─────────────────────────────────────
    if args.compare_base:
        logger.info("Running base Whisper for comparison...")
        base_evaluator = AblationEvaluator(
            whisper_model="small",
            device=device,
            output_dir=str(Path(args.output).parent),
        )
        base_result = base_evaluator.evaluate_split(
            manifest=df,
            audio_col=args.audio_col,
            text_col=args.text_col,
            demographic_col=args.demographic_col,
        )
        result["base_whisper"] = {k: v for k, v in base_result.items() if k != "hypotheses"}

        base_wer = base_result["wer_overall"]
        ft_wer = ft_result["wer_overall"]
        delta = ft_wer - base_wer
        result["comparison"] = {
            "base_wer": base_wer,
            "fine_tuned_wer": ft_wer,
            "absolute_delta": round(delta, 2),
            "relative_improvement": f"{abs(delta / base_wer) * 100:.1f}%" if base_wer > 0 else "N/A",
            "direction": "improvement" if delta < 0 else "regression",
        }
        logger.info("Base WER: %.2f%%  →  Fine-tuned WER: %.2f%%  (Δ %.2fpp)",
                    base_wer, ft_wer, delta)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
