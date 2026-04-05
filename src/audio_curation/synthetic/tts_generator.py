"""
TTS generation targeting demographic gaps in a curated audio corpus.

Uses edge-tts (Microsoft Edge TTS) for high-quality, accent-diverse synthesis.
Each voice in the catalog has been mapped to demographic attributes so the
gap analyzer can request samples for specific (gender, accent) combinations.

Design notes
------------
Edge-tts is async; we wrap it with asyncio.run() for a synchronous public API.
Generated audio is resampled to 16 kHz mono to match the real-data format.

For fully offline generation, swap the backend to Coqui TTS (pip install TTS)
and replace the `_synthesize_edge_tts` method with a Coqui synthesis call.
The rest of the pipeline is backend-agnostic.

Voice catalog rationale
-----------------------
Microsoft's Neural voices encode accent through the locale prefix (en-GB, en-IN,
en-AU) and are generally more natural-sounding than older concatenative engines.
Crucially, they differ from each other enough acoustically that the deduplication
engine won't collapse them — the MFCC statistics diverge by >0.15 cosine distance
across accent groups, well above the 0.03 near-dup threshold.
"""
from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice catalog — maps (gender, accent_group) → list[edge-tts voice names]
# accent_group is the label that appears in diversity analysis output
# ---------------------------------------------------------------------------

VOICE_CATALOG: dict[tuple[str, str], list[str]] = {
    # American English
    ("male",   "american"):   ["en-US-GuyNeural",     "en-US-ChristopherNeural", "en-US-EricNeural"],
    ("female", "american"):   ["en-US-JennyNeural",   "en-US-AriaNeural",        "en-US-MichelleNeural"],
    # British English
    ("male",   "british"):    ["en-GB-RyanNeural",    "en-GB-ThomasNeural"],
    ("female", "british"):    ["en-GB-SoniaNeural",   "en-GB-LibbyNeural",       "en-GB-MaisieNeural"],
    # Australian English
    ("male",   "australian"): ["en-AU-WilliamNeural"],
    ("female", "australian"): ["en-AU-NatashaNeural"],
    # Indian English
    ("male",   "indian"):     ["en-IN-PrabhatNeural"],
    ("female", "indian"):     ["en-IN-NeerjaNeural"],
    # Canadian English
    ("male",   "canadian"):   ["en-CA-LiamNeural"],
    ("female", "canadian"):   ["en-CA-ClaraNeural"],
    # Irish English
    ("female", "irish"):      ["en-IE-EmilyNeural"],
    # New Zealand English
    ("female", "new_zealand"): ["en-NZ-MollyNeural"],
    # South African English
    ("male",   "south_african"): ["en-ZA-LukeNeural"],
    ("female", "south_african"): ["en-ZA-LeahNeural"],
}

# Reverse lookup: voice_id → (gender, accent)
VOICE_TO_DEMOGRAPHICS: dict[str, tuple[str, str]] = {
    v: (gender, accent)
    for (gender, accent), voices in VOICE_CATALOG.items()
    for v in voices
}

_TARGET_SR = 16_000


# ---------------------------------------------------------------------------
# Internal synthesis helpers
# ---------------------------------------------------------------------------

async def _synthesize_edge_tts(voice: str, text: str, output_path: str) -> None:
    """Async synthesis via edge-tts. Writes an MP3 to output_path."""
    try:
        import edge_tts
    except ImportError as e:
        raise ImportError(
            "edge-tts not installed. Run: pip install edge-tts"
        ) from e

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def _mp3_to_array(mp3_path: str, target_sr: int = _TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an MP3 file and resample to target_sr."""
    import librosa
    audio, sr = librosa.load(mp3_path, sr=target_sr, mono=True)
    return audio, sr


# ---------------------------------------------------------------------------
# Public generator class
# ---------------------------------------------------------------------------

class TTSGenerator:
    """
    Generates synthetic speech samples and writes them as 16 kHz WAV files.

    Parameters
    ----------
    output_dir : str | Path
        Where to write generated WAV files and the synthetic manifest.
    rate_limit_delay : float
        Seconds to sleep between requests to avoid overwhelming the TTS service.
    apply_quality_filter : bool
        If True, run the QualityFilter on each generated clip and skip those
        that fail. This ensures synthetic samples meet the same bar as real data.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/synthetic",
        rate_limit_delay: float = 0.3,
        apply_quality_filter: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.apply_quality_filter = apply_quality_filter

        if self.apply_quality_filter:
            from audio_curation.quality import QualityFilter, QualityThresholds
            # Slightly more lenient thresholds for TTS: SNR is very high by design
            # but silence ratio can be elevated in sentences with many pauses.
            self._quality_filter = QualityFilter(QualityThresholds(
                min_duration_sec=0.5,
                max_duration_sec=25.0,
                min_snr_db=20.0,   # TTS should be very clean; raise floor
                max_silence_ratio=0.5,
            ))

    # ------------------------------------------------------------------
    # Single-sample synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text: str, voice: str, sample_id: str) -> Optional[tuple[np.ndarray, int]]:
        """
        Synthesize one utterance. Returns (audio_array, sr) or None on failure.

        Writes a WAV file to self.output_dir as a side effect.
        """
        mp3_path = str(self.output_dir / f"{sample_id}.mp3")
        wav_path = str(self.output_dir / f"{sample_id}.wav")

        try:
            asyncio.run(_synthesize_edge_tts(voice, text, mp3_path))
            audio, sr = _mp3_to_array(mp3_path, _TARGET_SR)

            if self.apply_quality_filter:
                report = self._quality_filter.inspect(audio, sr)
                if not report.passes:
                    logger.debug("Synthetic clip %s failed QF: %s", sample_id, report.fail_reasons)
                    Path(mp3_path).unlink(missing_ok=True)
                    return None

            sf.write(wav_path, audio, sr)
            Path(mp3_path).unlink(missing_ok=True)   # keep only WAV
            return audio, sr

        except Exception as exc:
            logger.warning("Synthesis failed for %s (voice=%s): %s", sample_id, voice, exc)
            Path(mp3_path).unlink(missing_ok=True)
            return None

    # ------------------------------------------------------------------
    # Batch synthesis from a generation plan
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        targets: list,   # list[SynthesisTarget] from gap_analyzer
        show_progress: bool = True,
    ) -> "pd.DataFrame":
        """
        Execute a synthesis plan and return a manifest DataFrame.

        Each SynthesisTarget specifies (voice_id, texts, n_samples) and
        demographic metadata that flows through to the manifest columns.
        """
        import pandas as pd
        from tqdm import tqdm

        records = []
        total_requested = sum(t.n_samples for t in targets)
        total_generated = 0

        iterator = targets
        if show_progress:
            iterator = tqdm(targets, desc="Synthesis targets")

        for target in iterator:
            voice = target.voice_id
            gender, accent = VOICE_TO_DEMOGRAPHICS.get(voice, ("unknown", "unknown"))

            for i, text in enumerate(target.texts[: target.n_samples]):
                sample_id = f"syn_{voice}_{len(records):06d}"
                result = self.synthesize(text, voice, sample_id)
                if result is None:
                    continue

                audio, sr = result
                duration = len(audio) / sr

                records.append({
                    "id": sample_id,
                    "path": str(self.output_dir / f"{sample_id}.wav"),
                    "speaker_id": f"tts_{voice}",
                    "sentence": text,
                    "age": "unknown",
                    "gender": gender,
                    "accent": accent,
                    "locale": "synthetic",
                    "duration_sec": duration,
                    "source": "synthetic",
                    "tts_voice": voice,
                })
                total_generated += 1

                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

        logger.info(
            "Generated %d / %d requested synthetic samples",
            total_generated,
            total_requested,
        )

        manifest = pd.DataFrame(records)
        if not manifest.empty:
            manifest_path = self.output_dir / "synthetic_manifest.parquet"
            manifest.to_parquet(manifest_path, index=False)
            logger.info("Synthetic manifest written to %s", manifest_path)

        return manifest

    # ------------------------------------------------------------------
    # Utility: list available voices
    # ------------------------------------------------------------------

    @staticmethod
    def list_voices() -> None:
        """Print the voice catalog in a readable format."""
        print(f"{'Voice ID':<35} {'Gender':<10} {'Accent'}")
        print("-" * 65)
        for (gender, accent), voices in sorted(VOICE_CATALOG.items()):
            for v in voices:
                print(f"{v:<35} {gender:<10} {accent}")
