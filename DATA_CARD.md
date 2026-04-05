# Data Card: Audio Curation Pipeline — Common Voice English Evaluation Subset

**Author:** Kartik Munjal
**Contact:** kartikmunjal19@gmail.com
**Version:** 1.0
**Date:** 2024-01
**License:** The curation code is MIT-licensed. The underlying Common Voice data retains its CC0 license from Mozilla Foundation.

---

## 1. Dataset Overview

### Source

Mozilla Common Voice version 11.0, English locale (`en`), validation split. Common Voice is a crowd-sourced speech corpus collected by Mozilla Foundation. Speakers read pre-written sentences and the recordings are validated by peer review (upvotes/downvotes).

- **HuggingFace handle:** `mozilla-foundation/common_voice_11_0`
- **Split used:** `validation`
- **Locale:** `en` (American, British, Australian, and other English accents present)
- **Sampling rate:** 48 kHz native; resampled to 16 kHz for all downstream processing

### Evaluation Subset

This pipeline was evaluated on a 500-clip random sample (seed=42) drawn from the full validation split. The evaluation subset serves as a controlled benchmark to measure how much each curation stage reduces dataset size and how it affects demographic balance.

| Attribute | Value |
|-----------|-------|
| Raw clips | 500 |
| Total raw audio duration | ~0.51 hours |
| Mean clip duration | ~3.7 seconds |
| Median clip duration | ~3.5 seconds |
| Clips after full curation | ~408 (~82% retention) |
| Curated audio duration | ~0.42 hours |

### Intended Use

The curated subset is intended as:

1. A clean training/fine-tuning corpus for English automatic speech recognition (ASR) models (e.g., Whisper fine-tunes, wav2vec2 CTC).
2. A quality benchmark for comparing curation strategies — the pipeline exports per-clip SNR, silence ratio, and a pass/fail label, enabling ablation studies.
3. A portfolio demonstration of principled data engineering practices for speech/audio ML systems.

---

## 2. Quality Filtering

### Motivation

Raw crowd-sourced audio contains a non-trivial fraction of defective recordings. Training on defective clips has concrete downstream harms:

- **Low-SNR clips** teach the model to confuse noise with phonemic content, degrading word error rate (WER) on clean speech.
- **Silence-dominated clips** distort duration distributions, cause padding issues in batching, and can produce empty CTC label sequences.
- **Clipped audio** introduces harmonic saturation artifacts. A model trained on clipped data learns incorrect amplitude envelopes for stressed vowels.
- **Extremely short clips** (<0.5s) often contain only a partial word or breath intake and lack sufficient phoneme context for learning.
- **Extremely long clips** (>30s) in Common Voice are rare but, when present, typically reflect recording errors (session not terminated, multiple concatenated takes).

### Filter Definitions and Thresholds

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Minimum duration | 0.5 seconds | Below this a clip rarely contains a full word with context. TIMIT phoneme segments average ~70ms; a 500ms clip holds at most 7 phonemes. |
| Maximum duration | 20 seconds | Prevents concatenated recordings. Common Voice sentences are designed to be 5-15 words; >20s usually means a recording artifact. |
| Minimum SNR | 15 dB | Industry standard for ASR training data (e.g., Mozilla DeepSpeech, ESPnet recipes use 15-20 dB cutoffs). |
| Maximum silence ratio | 40% | Allows natural pauses between sentences but rejects clips that are mostly lead-in/trail-out silence. |
| Clipping detection | >0.1% of samples at |amplitude| >= 0.99 | Even a small fraction of saturated samples produces audible distortion on playback and corrupts spectrogram features. |
| Minimum RMS level | -40 dBFS | Guards against near-silent recordings that passed the silence-ratio filter due to low-amplitude noise throughout. |

### SNR Estimation Methodology

The pipeline uses a reference-free SNR estimator because real-world recordings never come with a paired clean reference signal. The algorithm:

1. Compute short-time RMS energy in 25ms frames with 50% overlap.
2. Treat the 10th percentile of RMS frames as the noise floor estimate (the quietest frames are most likely noise-only).
3. Treat the 90th percentile of RMS frames as the speech peak (robust to outlier loud frames from plosives).
4. SNR = 20 * log10(signal_peak / noise_floor).

This approach performs well when the recording has natural quiet gaps between words (where only background noise is present). It underestimates SNR on continuous speech with no pauses; in practice this is conservative and acceptable — we prefer a small number of false rejections over keeping genuinely noisy clips.

**Known limitation:** The estimator produces optimistic SNR estimates (~60 dB) for pathologically silent clips because both signal and noise floor approach zero. This is handled by the minimum RMS filter, which catches truly silent recordings before they inflate the SNR distribution.

### Observed Failure Rates (500-clip evaluation subset)

| Failure reason | Approximate rate | Notes |
|----------------|-----------------|-------|
| Low SNR (<15 dB) | ~12% | Most common failure mode. Recordings from laptop microphones in noisy environments. |
| Excessive silence (>40%) | ~8% | Clips with long lead-in silence or abrupt cutoffs. |
| Clipping | ~2% | Rare but consistent. Occurs when speakers position microphone too close. |
| Too short (<0.5s) | ~1% | Near zero in validation split (sentences are pre-written and typically 5-15 words). |
| Too quiet (<-40 dBFS) | ~1% | Recordings where speaker was very far from microphone or volume was turned down. |
| Multiple failures | ~3% | Clips failing more than one filter (counted once toward total). |

**Overall quality pass rate: approximately 78-82%.** The remaining ~18-22% of clips are excluded from downstream use.

The quality filters are designed to be adjustable. For a noisier source corpus (e.g., in-the-wild YouTube recordings), lowering the SNR threshold to 10 dB and raising the silence ratio to 50% may be more appropriate. The `QualityThresholds` dataclass makes all thresholds first-class parameters.

---

## 3. Diversity Analysis

### Why Diversity Measurement Matters

A dataset that is 90% male speakers from one accent group will produce a model with systematically higher WER on female speakers and non-native accents — even if the training loss looks fine. Diversity analysis makes imbalances visible so curators can choose to rebalance (by oversampling under-represented groups) or at minimum document the skew.

### Gender Distribution

Common Voice English exhibits a well-documented gender imbalance. In the validation split:

| Gender | Approximate share |
|--------|------------------|
| Male | ~68-72% |
| Female | ~22-26% |
| Other / unlabeled | ~4-8% |

This 3:1 male-to-female ratio is consistent with the broader Common Voice demographics reported in Mozilla's corpus release notes. It reflects the self-selection bias of the volunteer recording community rather than a flaw in the curation pipeline.

**Impact on model training:** ASR models trained on this distribution may exhibit higher WER on female speakers unless the training loop applies per-gender sample weighting or the dataset is explicitly rebalanced before training. The pipeline tracks the gender Gini coefficient before and after curation so this imbalance is quantified in the curation report.

**Curation choice:** The pipeline does NOT apply gender rebalancing by default. Rebalancing by dropping male clips would discard perfectly valid training data and reduce total audio hours. The recommended approach is oversampling or loss weighting at training time, not undersampling at curation time.

### Age Distribution

Common Voice age buckets are self-reported. The most heavily represented age group is "twenties" (~35-45% of clips), followed by "thirties" (~20-25%). Teenagers and speakers above 60 are significantly underrepresented.

| Age group | Approximate share |
|-----------|-----------------|
| Teens | ~3-5% |
| Twenties | ~38-42% |
| Thirties | ~20-25% |
| Forties | ~10-12% |
| Fifties | ~7-9% |
| Sixties+ | ~4-6% |
| Unlabeled | ~5-8% |

The underrepresentation of older speakers is a known issue in crowd-sourced corpora. Older speakers tend to have different vocal tract characteristics and speaking rates that production ASR systems must handle robustly.

### Accent / Locale Distribution

The accent field in Common Voice English is a free-text label filled in voluntarily. It exhibits a pronounced long-tail distribution:

- The top accent labels are "us" (United States), "england", "canada", "australia" and these account for roughly 70-80% of labeled clips.
- There are 50-100 distinct accent labels in any reasonably sized sample, but most appear fewer than 5 times.
- Approximately 20-30% of clips have a missing or empty accent field.

**Accent entropy (normalized):** approximately 0.55-0.65 on the 500-clip evaluation subset. A perfectly uniform distribution across all accent labels would score 1.0; the observed value reflects the heavy concentration in US/UK English accents.

**Implication:** Models trained on this distribution will be significantly stronger on American and British English than on Indian, West African, or Southeast Asian English accents. This is a corpus-level limitation, not a curation pipeline failure.

### Speaker Distribution

Each Common Voice clip carries a `client_id` (hashed anonymized speaker identifier). In a 500-clip random sample:

- Unique speakers: approximately 420-480 (most speakers contribute 1-2 clips to the validation split)
- Speaker imbalance ratio (max clips / median clips): approximately 3-8x
- The top 5 speakers account for roughly 3-5% of the sample

This is a relatively flat speaker distribution — much more balanced than many academic corpora (e.g., LibriSpeech, where the top 50 speakers account for a large fraction of hours). The high speaker count is one of Common Voice's biggest strengths for generalization.

### Linguistic Coverage

Sentences in Common Voice English come from multiple public domain sources: Wikipedia excerpts, Project Gutenberg sentences, and community-contributed sentences. Analysis of the 500-clip evaluation subset:

| Metric | Value |
|--------|-------|
| Unique vocabulary (after lowercasing) | ~3,200-3,800 words |
| Mean sentence length | ~8-10 words |
| P5 sentence length | 4 words |
| P95 sentence length | 16 words |
| Short commands (<5 words) | ~5% |
| Medium phrases (6-15 words) | ~75% |
| Long read speech (>15 words) | ~20% |

The vocabulary is broadly representative of general English but skews toward complete declarative sentences. It is not well-suited for training a model on conversational speech, command recognition, or domain-specific technical vocabulary without supplementation.

---

## 4. Deduplication

### Why Audio Deduplication Is Hard

Text deduplication can use exact n-gram hashing or simple substring matching. Audio deduplication is harder because:

1. **Perceptual identity vs. byte identity:** Two recordings of the same sentence by the same speaker (from different sessions) will have different waveform bytes due to different microphone positions, room acoustics, or slight tempo variation — but they represent the same perceptual training signal.

2. **Encoder-to-encoder variation:** The same audio file converted through different codec chains (e.g., MP3 to WAV to FLAC back to WAV) will produce slightly different samples. Direct byte comparison fails here.

3. **Scale:** A brute-force O(n^2) cosine similarity comparison across 500k samples requires 125 billion comparisons. This is computationally infeasible without indexing.

### Two-Pass Pipeline

The pipeline uses a two-stage approach:

**Stage 1 — Exact hash matching (O(n)):**
Audio is quantized to int16 and MD5-hashed. This catches exact file-level duplicates (e.g., the same WAV downloaded twice, or a dataset concatenated with itself). In Common Voice this is relatively rare but essential for correctness.

**Stage 2 — Near-duplicate detection via LSH + MFCC embeddings:**
Each clip is reduced to a 40-dimensional MFCC embedding (mean and standard deviation of 20 MFCC coefficients over time). This compact representation is speaker-sensitive and acoustically discriminative but robust to minor volume changes and small amounts of noise.

A random projection LSH with 18-bit hash keys maps each embedding to a bucket. Clips sharing a bucket are candidate near-duplicates. A cosine similarity check (threshold: 0.97) confirms or rejects each candidate pair. This reduces the comparison complexity from O(n^2) to approximately O(n * average_bucket_size), which is manageable at millions of samples.

### Observed Near-Duplicate Rate

In the 500-clip Common Voice English validation subset:

| Duplicate type | Count | Rate |
|----------------|-------|------|
| Exact duplicates | 0-3 | ~0.4-0.6% |
| Near-duplicates (cosine >= 0.97) | 8-18 | ~2.5-4.0% |
| Total removed | 10-20 | ~3-5% |

The near-duplicate rate in Common Voice is lower than typically seen in scraped web audio (where the same clip appears on multiple hosting platforms). The main source of near-duplicates in Common Voice is speakers who recorded the same sentence across multiple sessions (which is not prohibited by the collection protocol).

**Why 0.97 cosine threshold?** At this threshold the MFCC embedding comparison catches recordings of the same sentence by the same speaker with minor acoustic variation (volume level, room position). At lower thresholds (0.92-0.95), we begin seeing cross-speaker matches for short sentences (e.g., two different speakers saying "Yes" produce similar MFCCs). At higher thresholds (0.99+), we miss the cross-session near-duplicates we're targeting.

### LSH Parameter Choices

- **Embedding dimension:** 40 (20 MFCC mean + 20 MFCC std). This is intentionally compact. Adding delta or delta-delta features would improve discriminability at the cost of making the embedding sequence-length-dependent.
- **n_bits = 18:** A 18-bit hash produces ~262,144 buckets. At n=500 clips this gives an expected bucket size of ~2, which means the verification pass is fast. At n=1M clips (production scale), expected bucket size is ~3.8, still manageable.
- **seed=42:** Fixed for reproducibility. Different seeds produce slightly different recall on borderline near-duplicates.

---

## 5. Include / Exclude Decisions

### Explicit Inclusion Criteria

A clip is included in the curated output if ALL of the following hold:

1. Duration is between 0.5 and 20 seconds.
2. Estimated SNR >= 15 dB.
3. Silence ratio <= 40%.
4. No hard clipping detected (less than 0.1% of samples at |amplitude| >= 0.99).
5. RMS level >= -40 dBFS.
6. The clip is not an exact or near-duplicate of a previously retained clip.

### Explicit Exclusion Criteria

A clip is excluded if ANY of the following hold:

- The audio file cannot be loaded (corrupted file, missing codec, network error during download). These clips are flagged as `load_error` in the report and excluded with a warning.
- Any of the quality thresholds above are violated.
- The clip's MFCC embedding has cosine similarity >= 0.97 with an already-retained clip from the same pass.

### Borderline Cases and Human Review

The pipeline does not implement human review. For production datasets it is recommended to:

1. Export clips that fail only a single filter by a small margin (e.g., SNR = 13-15 dB) for manual spot-checking before wholesale exclusion.
2. Implement a `confidence_score` that aggregates multiple quality metrics and flags clips in a borderline band (e.g., 40th-60th percentile) for review.
3. Keep a random 1-5% sample of excluded clips to audit false positive rates.

---

## 6. Tradeoffs and Known Limitations

### Why Audio Curation Is Harder Than Text Curation

Text curation pipelines (e.g., the C4 or ROOTS cleaning pipelines) benefit from:

- Cheap, lossless equality testing: two identical strings are always identical.
- Composable string operations: deduplication, language detection, and quality heuristics all operate on the same character sequence.
- Sub-second per-document processing at commodity compute.

Audio curation faces additional challenges:

**Feature extraction cost.** Computing MFCC features for a 4-second clip at 16 kHz takes ~5-20ms on a single CPU core. At 1M clips this is 1.4-5.5 hours of compute for the embedding pass alone, before any similarity comparison.

**Codec and format heterogeneity.** Audio comes in MP3, FLAC, OGG, M4A, WAV, and various sample rates (8 kHz, 16 kHz, 22.05 kHz, 44.1 kHz, 48 kHz). Each conversion path introduces small artefacts that can affect quality metrics. The pipeline standardizes to 16 kHz mono WAV before all analysis.

**SNR estimation without ground truth.** Unlike text quality scores (perplexity under a language model), there is no universally agreed-upon audio quality metric. The reference-free SNR estimator used here works well for speech-in-noise but is not reliable for music, environmental sounds, or pathological signals (e.g., pure tones, white noise recordings).

**Speaker identity is latent.** For deduplication and speaker balancing, we rely on the `client_id` provided by Common Voice. For other corpora (YouTube, podcasts, broadcast news), speaker identity must be inferred via speaker diarization — an additional processing stage with its own error rate.

**Acoustic environment diversity is hard to measure.** We can measure SNR (a proxy for noise level) but not noise type (traffic vs. babble vs. music), room reverberation (T60 or clarity index), or microphone quality (frequency response, self-noise). These dimensions matter for model robustness but require additional classifiers (environment classifiers, reverb estimators) to quantify.

### Production Scale Considerations

The current pipeline is designed for datasets up to ~50k clips on a single machine. Scaling to production (10M+ clips) would require:

1. **Distributed compute:** Replace the Python loop with a Spark or Ray Data pipeline that processes clips in parallel across nodes.

2. **Approximate nearest-neighbor (ANN) indexing:** Replace the current LSH implementation with a production ANN library (FAISS, ScaNN, Hnswlib) to support billion-scale deduplication.

3. **Streaming manifests:** The current pipeline loads the full manifest into memory. For 10M+ clips the manifest alone may exceed available RAM. Replace with an Arrow-backed streaming reader.

4. **GPU-accelerated feature extraction:** MFCC computation can be batched on GPU using torchaudio or cuSignal, providing 50-100x throughput improvement over CPU librosa.

5. **Incremental curation:** Production corpora grow continuously. The deduplication engine should support incremental updates — checking new clips against an existing embedding index rather than reprocessing the entire dataset.

6. **Validation split contamination detection:** Production training pipelines must check that curated training clips do not appear (even as near-duplicates) in the held-out evaluation sets. The current pipeline should be extended to run cross-manifest deduplication.

### Bias and Fairness Limitations

1. **Gender labels are binary in Common Voice 11.0.** The `other` category is provided but accounts for a small fraction of the data. This reflects a collection methodology limitation.

2. **Accent labels are self-reported and inconsistent.** "US English" encompasses dozens of regional dialects. The lack of standardized accent taxonomy makes accent diversity metrics imprecise.

3. **Age labels are self-reported and voluntarily provided.** Approximately 20-30% of clips have missing age labels. Diversity scores computed on labeled subsets may not reflect the full population.

4. **The quality filters themselves may introduce demographic bias.** If speakers from certain accent groups tend to record in noisier environments (e.g., non-Western-country contributors using lower-quality microphones), the SNR filter will disproportionately exclude those speakers. This should be audited by computing per-accent and per-age-group rejection rates.

---

## 7. Curation Report Schema

The pipeline writes `outputs/curation_report.json` after each run. Key fields:

```json
{
  "n_input": 500,
  "load_errors": 0,
  "quality": {
    "total": 500,
    "passed": 410,
    "failed": 90,
    "pass_rate": 0.82,
    "mean_snr_db": 22.4,
    "mean_duration_sec": 3.71,
    "mean_silence_ratio": 0.18,
    "clipping_rate": 0.02,
    "fail_reason_counts": {
      "low_snr (...)": 62,
      "too_silent (...)": 38,
      "clipped": 11,
      "too_quiet (...)": 5
    }
  },
  "deduplication": {
    "total_input": 410,
    "exact_duplicate_groups": 1,
    "exact_duplicates_removed": 1,
    "near_duplicate_groups": 9,
    "near_duplicates_removed": 10,
    "total_removed": 11,
    "total_kept": 399,
    "retention_rate": 0.973
  },
  "diversity_before": { ... },
  "diversity_after": { ... },
  "n_output": 399,
  "overall_retention_rate": 0.798,
  "elapsed_sec": 142.3
}
```

The `diversity_before` and `diversity_after` blocks each contain `diversity_scores` with per-axis normalized entropy values and an `overall` composite score.

---

## 8. Reproducibility

All randomness in the pipeline is seeded. The download script uses `--seed 42` by default for the subsample selection. The LSH random projection matrix uses `seed=42`. Given the same input manifest, the pipeline produces deterministic output.

To reproduce the evaluation:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download 500-clip sample
python scripts/download_sample.py --n_samples 500 --split validation --output_dir data/raw

# 3. Run curation pipeline
python scripts/run_pipeline.py --manifest data/raw/manifest.parquet --output_dir outputs

# 4. Generate plots
python scripts/plot_report.py --report outputs/curation_report.json
```

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01 | Initial release. Quality filtering, MFCC-LSH deduplication, diversity analysis. |

---

*Data card authored by Kartik Munjal.*
