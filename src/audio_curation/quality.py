"""
Audio quality filtering: SNR estimation, silence detection, duration gating, clipping.

Why these matter for training data:
  - Low-SNR clips teach the model noise patterns rather than clean speech.
  - Silent or heavily padded clips waste training compute and distort duration stats.
  - Clipped audio introduces harmonic distortion that doesn't reflect real speech.
  - Very short clips (<0.5s) rarely contain a full phoneme inventory; >30s clips are
    usually concatenated recordings with inconsistent recording conditions.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0
    min_snr_db: float = 15.0
    max_silence_ratio: float = 0.4
    clipping_threshold: float = 0.99
    min_rms_db: float = -40.0   # floor for extremely quiet recordings


@dataclass
class QualityReport:
    duration_sec: float
    snr_db: float
    silence_ratio: float
    is_clipped: bool
    rms_db: float
    passes: bool
    fail_reasons: list[str] = field(default_factory=list)


def estimate_snr(audio: np.ndarray, sr: int, frame_duration_ms: float = 25.0) -> float:
    """
    Estimate signal-to-noise ratio in dB.

    Strategy: treat the lowest-energy decile of frames as the noise floor and
    the 90th-percentile frame as the signal peak. Works without a separate noise
    reference, which we rarely have for in-the-wild recordings.

    Returns np.inf for silence (no detectable noise floor).
    """
    frame_len = int(frame_duration_ms / 1000 * sr)
    hop_len = frame_len // 2

    rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    rms = rms[rms > 0]

    if len(rms) == 0:
        return 0.0

    noise_floor = np.percentile(rms, 10)
    signal_peak = np.percentile(rms, 90)

    if noise_floor < 1e-10:
        return 60.0  # essentially no background noise
    if signal_peak < 1e-10:
        return 0.0

    snr_db = 20.0 * np.log10(signal_peak / (noise_floor + 1e-10))
    return float(np.clip(snr_db, 0.0, 80.0))


def detect_silence_ratio(
    audio: np.ndarray,
    sr: int,
    top_db: float = 30.0,
    frame_duration_ms: float = 25.0,
) -> float:
    """
    Return the fraction of frames classified as silence.

    Uses librosa.effects.split under the hood; frames whose RMS is more than
    `top_db` below the peak are considered silent.
    """
    frame_len = int(frame_duration_ms / 1000 * sr)
    hop_len = frame_len // 2

    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=frame_len, hop_length=hop_len)

    if len(intervals) == 0:
        return 1.0  # entirely silent

    speech_samples = sum(end - start for start, end in intervals)
    silence_ratio = 1.0 - speech_samples / len(audio)
    return float(np.clip(silence_ratio, 0.0, 1.0))


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    """
    Detect hard clipping: more than 0.1% of samples saturated near ±1.

    Clipped recordings exhibit severe harmonic distortion. Even a handful of
    saturated samples per utterance is a red flag in a 16-bit recording.
    """
    clipped_frac = np.mean(np.abs(audio) >= threshold)
    return bool(clipped_frac > 0.001)


def compute_rms_db(audio: np.ndarray) -> float:
    """Return overall RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms < 1e-10:
        return -120.0
    return float(20.0 * np.log10(rms))


class QualityFilter:
    """
    Stateless filter that scores individual audio samples and batch-filters datasets.

    Usage
    -----
    >>> qf = QualityFilter(thresholds=QualityThresholds(min_snr_db=20))
    >>> report = qf.inspect(audio_array, sample_rate=16_000)
    >>> print(report.passes, report.snr_db)
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def inspect(self, audio: np.ndarray, sr: int) -> QualityReport:
        """Compute all quality metrics and return a QualityReport."""
        t = self.thresholds

        # --- basic measurements ---
        duration = len(audio) / sr
        snr_db = estimate_snr(audio, sr)
        silence_ratio = detect_silence_ratio(audio, sr)
        is_clipped = detect_clipping(audio, t.clipping_threshold)
        rms_db = compute_rms_db(audio)

        # --- gate checks ---
        fail_reasons: list[str] = []
        if duration < t.min_duration_sec:
            fail_reasons.append(f"too_short ({duration:.2f}s < {t.min_duration_sec}s)")
        if duration > t.max_duration_sec:
            fail_reasons.append(f"too_long ({duration:.2f}s > {t.max_duration_sec}s)")
        if snr_db < t.min_snr_db:
            fail_reasons.append(f"low_snr ({snr_db:.1f} dB < {t.min_snr_db} dB)")
        if silence_ratio > t.max_silence_ratio:
            fail_reasons.append(f"too_silent ({silence_ratio:.1%} > {t.max_silence_ratio:.1%})")
        if is_clipped:
            fail_reasons.append("clipped")
        if rms_db < t.min_rms_db:
            fail_reasons.append(f"too_quiet ({rms_db:.1f} dBFS < {t.min_rms_db} dBFS)")

        return QualityReport(
            duration_sec=duration,
            snr_db=snr_db,
            silence_ratio=silence_ratio,
            is_clipped=is_clipped,
            rms_db=rms_db,
            passes=len(fail_reasons) == 0,
            fail_reasons=fail_reasons,
        )

    def filter_batch(
        self,
        audio_list: list[np.ndarray],
        sr_list: list[int],
        show_progress: bool = True,
    ) -> tuple[list[bool], list[QualityReport]]:
        """
        Filter a list of audio arrays. Returns (keep_mask, reports).
        """
        from tqdm import tqdm

        keep_mask: list[bool] = []
        reports: list[QualityReport] = []

        iterator = zip(audio_list, sr_list)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Quality filtering", unit="clip")

        for audio, sr in iterator:
            try:
                report = self.inspect(audio, sr)
            except Exception as exc:
                logger.warning("Quality check failed: %s", exc)
                report = QualityReport(
                    duration_sec=0.0,
                    snr_db=0.0,
                    silence_ratio=1.0,
                    is_clipped=False,
                    rms_db=-120.0,
                    passes=False,
                    fail_reasons=[f"exception: {exc}"],
                )
            keep_mask.append(report.passes)
            reports.append(report)

        n_kept = sum(keep_mask)
        logger.info("Quality filter: kept %d / %d (%.1f%%)", n_kept, len(keep_mask), 100 * n_kept / max(len(keep_mask), 1))
        return keep_mask, reports

    @staticmethod
    def summarize(reports: list[QualityReport]) -> dict:
        """Aggregate quality statistics across a batch."""
        from collections import Counter

        all_reasons = []
        for r in reports:
            all_reasons.extend(r.fail_reasons)

        return {
            "total": len(reports),
            "passed": sum(r.passes for r in reports),
            "failed": sum(not r.passes for r in reports),
            "pass_rate": sum(r.passes for r in reports) / max(len(reports), 1),
            "mean_snr_db": float(np.mean([r.snr_db for r in reports])),
            "mean_duration_sec": float(np.mean([r.duration_sec for r in reports])),
            "mean_silence_ratio": float(np.mean([r.silence_ratio for r in reports])),
            "clipping_rate": float(np.mean([r.is_clipped for r in reports])),
            "fail_reason_counts": dict(Counter(all_reasons)),
        }
