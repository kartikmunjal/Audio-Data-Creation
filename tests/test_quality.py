"""Unit tests for quality filtering."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_curation.quality import (
    QualityFilter,
    QualityThresholds,
    estimate_snr,
    detect_silence_ratio,
    detect_clipping,
    compute_rms_db,
)

SR = 16_000


def make_tone(freq=440, duration=2.0, sr=SR, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_noise(duration=2.0, sr=SR, amplitude=0.1):
    rng = np.random.default_rng(0)
    return (amplitude * rng.standard_normal(int(sr * duration))).astype(np.float32)


def make_silence(duration=2.0, sr=SR):
    return np.zeros(int(sr * duration), dtype=np.float32)


# ---------------------------------------------------------------------------
# SNR
# ---------------------------------------------------------------------------

class TestSNR:
    def test_clean_tone_high_snr(self):
        audio = make_tone(amplitude=0.8) + make_noise(amplitude=0.001)
        snr = estimate_snr(audio, SR)
        assert snr > 20, f"Expected high SNR for clean tone, got {snr:.1f}"

    def test_noisy_signal_low_snr(self):
        audio = make_tone(amplitude=0.1) + make_noise(amplitude=0.15)
        snr = estimate_snr(audio, SR)
        assert snr < 15, f"Expected low SNR for noisy signal, got {snr:.1f}"

    def test_silence_returns_nonzero(self):
        # Silence shouldn't crash; returns 60 dB (effectively silent background)
        snr = estimate_snr(make_silence(), SR)
        assert isinstance(snr, float)


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------

class TestSilenceDetection:
    def test_pure_silence_ratio_one(self):
        ratio = detect_silence_ratio(make_silence(), SR)
        assert ratio == 1.0

    def test_pure_tone_low_silence_ratio(self):
        audio = make_tone(amplitude=0.5)
        ratio = detect_silence_ratio(audio, SR)
        assert ratio < 0.3, f"Expected low silence ratio for tone, got {ratio:.2f}"

    def test_half_silence(self):
        sr = SR
        silence = make_silence(duration=1.0, sr=sr)
        tone = make_tone(duration=1.0, sr=sr, amplitude=0.5)
        audio = np.concatenate([silence, tone])
        ratio = detect_silence_ratio(audio, sr)
        # Should be roughly 50% ± some tolerance
        assert 0.3 < ratio < 0.7, f"Unexpected ratio {ratio:.2f} for half-silent audio"


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

class TestClipping:
    def test_clipped_audio(self):
        audio = np.ones(SR * 2, dtype=np.float32)  # fully saturated
        assert detect_clipping(audio, threshold=0.99)

    def test_normal_audio_not_clipped(self):
        audio = make_tone(amplitude=0.5)
        assert not detect_clipping(audio, threshold=0.99)


# ---------------------------------------------------------------------------
# RMS
# ---------------------------------------------------------------------------

class TestRMS:
    def test_silence_very_low_db(self):
        db = compute_rms_db(make_silence())
        assert db < -100

    def test_full_scale_near_zero_db(self):
        audio = np.ones(SR, dtype=np.float32)
        db = compute_rms_db(audio)
        assert -2 < db < 2


# ---------------------------------------------------------------------------
# QualityFilter integration
# ---------------------------------------------------------------------------

class TestQualityFilter:
    def setup_method(self):
        self.qf = QualityFilter(QualityThresholds(
            min_duration_sec=0.5,
            max_duration_sec=10.0,
            min_snr_db=10.0,
            max_silence_ratio=0.5,
        ))

    def test_clean_tone_passes(self):
        audio = make_tone(duration=2.0, amplitude=0.5) + make_noise(duration=2.0, amplitude=0.001)
        report = self.qf.inspect(audio, SR)
        assert report.passes, f"Expected pass, got: {report.fail_reasons}"

    def test_too_short_fails(self):
        audio = make_tone(duration=0.1)
        report = self.qf.inspect(audio, SR)
        assert not report.passes
        assert any("too_short" in r for r in report.fail_reasons)

    def test_clipped_fails(self):
        audio = np.ones(SR * 2, dtype=np.float32)
        report = self.qf.inspect(audio, SR)
        assert not report.passes
        assert any("clipped" in r for r in report.fail_reasons)

    def test_mostly_silent_fails(self):
        audio = np.concatenate([make_silence(duration=1.9), make_tone(duration=0.05, amplitude=0.5)])
        report = self.qf.inspect(audio, SR)
        assert not report.passes
        assert any("too_silent" in r for r in report.fail_reasons)

    def test_summarize_smoke(self):
        audios = [
            make_tone(duration=2.0) + make_noise(duration=2.0, amplitude=0.001),
            make_silence(duration=2.0),
            np.ones(SR * 2, dtype=np.float32),
        ]
        _, reports = self.qf.filter_batch(audios, [SR] * 3, show_progress=False)
        summary = QualityFilter.summarize(reports)
        assert summary["total"] == 3
        assert summary["passed"] >= 1
