# Audio Data Curation Pipeline

A production-grade pipeline for curating audio training data. Built on top of Mozilla Common Voice to demonstrate the full curation lifecycle: quality filtering → deduplication → diversity analysis → data card.

The core thesis: **audio curation is fundamentally harder than text curation** because you cannot read the data. Signal quality, speaker overlap, and acoustic condition diversity each require dedicated tooling — none of which translates from the text world.

---

## What This Does

```
Raw audio corpus
      │
      ▼
┌─────────────────────────────┐
│  Quality Filter             │  SNR, silence ratio, duration, clipping
└─────────────┬───────────────┘
              │ ~85% pass
              ▼
┌─────────────────────────────┐
│  Deduplication              │  Exact hash + MFCC near-dup (LSH)
└─────────────┬───────────────┘
              │ ~96% of quality-passed
              ▼
┌─────────────────────────────┐
│  Diversity Analysis         │  Gender, age, accent entropy, speaker balance
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Filtered Manifest          │  Parquet + JSON report + DATA_CARD.md
└─────────────────────────────┘
```

### On a 500-clip Common Voice sample:

| Stage | Clips remaining | Notes |
|-------|----------------|-------|
| Raw input | 500 | Common Voice validation split, English |
| After quality filter | ~430 | ~70 removed: low SNR, silence, too short |
| After deduplication | ~415 | ~15 near-duplicates removed |
| Diversity score (after) | ~0.62 / 1.0 | Gender skew persists — see DATA_CARD.md |

---

## Project Structure

```
audio-curation/
├── src/
│   └── audio_curation/
│       ├── quality.py          # SNR estimation, silence, clipping, duration
│       ├── diversity.py        # Speaker demographics, accent entropy, domain
│       ├── deduplication.py    # Exact hash + MFCC LSH near-duplicate detection
│       ├── pipeline.py         # Orchestrates all stages
│       └── synthetic/
│           ├── gap_analyzer.py   # Identify underrepresented demographic groups
│           ├── tts_generator.py  # Generate targeted samples via edge-tts voices
│           ├── mixer.py          # Blend synthetic with real audio at a target ratio
│           └── evaluator.py      # Ablation harness: WER before/after each split
├── scripts/
│   ├── download_sample.py      # Pull N clips from Common Voice via HuggingFace
│   ├── run_pipeline.py         # End-to-end curation pipeline CLI
│   ├── plot_report.py          # Generate visualisations from curation_report.json
│   ├── generate_synthetic.py   # Fill demographic gaps with TTS-generated samples
│   ├── run_ablations.py        # Measure quality impact of each filter in isolation
│   └── plot_ablations.py       # Visualise ablation results
├── tests/
│   ├── test_quality.py
│   ├── test_deduplication.py
│   └── test_synthetic.py
├── outputs/                    # Filtered manifests and reports land here
└── DATA_CARD.md                # Full curation findings, tradeoffs, and known biases
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download a sample of Common Voice English
python scripts/download_sample.py --n_samples 500 --split validation --output_dir data/raw

# 3. Run the full curation pipeline
python scripts/run_pipeline.py \
    --manifest data/raw/manifest.parquet \
    --output_dir outputs \
    --min_snr 15 \
    --min_duration 0.5 \
    --max_duration 20 \
    --dedup_threshold 0.97

# 4. Generate visualizations
python scripts/plot_report.py --report outputs/curation_report.json
```

Output files:
- `outputs/filtered_manifest.parquet` — curated dataset manifest
- `outputs/curation_report.json` — full per-stage statistics
- `outputs/plots/` — pipeline funnel, failure breakdown, diversity comparisons

---

## Quality Filtering

Three metrics cover the main failure modes in crowd-sourced audio:

### SNR Estimation (reference-free)

Without a clean reference signal we estimate SNR from the recording itself: the 10th-percentile energy frame is treated as the noise floor, the 90th-percentile as the signal peak.

```
SNR = 20 · log10(P90_rms / P10_rms)
```

This catches recordings made in noisy environments (cafes, open offices, street traffic) that would otherwise teach the model to produce or tolerate background noise. Threshold: **≥ 15 dB**.

### Silence Detection

Fraction of frames more than 30 dB below peak energy. High silence ratios indicate:
- Excessive leading/trailing padding from a bad VAD
- Dead microphone or recording error
- Speaker hesitation clips that don't contain useful phoneme coverage

Threshold: **≤ 40% silence**.

### Duration Gating

- **< 0.5s**: Too short to contain a complete word boundary in most languages.
- **> 20s**: Usually concatenated takes with inconsistent acoustic conditions.

### Clipping Detection

More than 0.1% of samples at ±1.0 (after normalization) indicates hard clipping. Even mild clipping introduces harmonic distortion at integer multiples of the fundamental — artifacts that don't appear in natural speech.

---

## Deduplication

Two-stage pipeline designed to scale past O(n²):

### Stage 1 — Exact Hash (O(n))

MD5 of int16-quantized audio bytes. Catches verbatim duplicates from dataset aggregation pipelines that accidentally include the same recording twice.

### Stage 2 — Near-Duplicate Detection (O(n) with LSH)

1. Compute a 40-dimensional MFCC embedding per clip (mean + std of 20 MFCCs over time).
2. Hash each embedding into an 18-bit bucket using random binary projections (Locality-Sensitive Hashing).
3. For each bucket with ≥ 2 candidates, compute cosine similarity and flag pairs above threshold 0.97.

Why this matters: the same speaker re-recording the same sentence across two sessions produces different bytes but nearly identical MFCC statistics. Training on both copies overweights that speaker/sentence pair without adding phonemic diversity.

LSH complexity: O(n) embedding + O(n) bucketing + O(k²) per bucket where k is bucket size. In practice k is small (2-5) so total cost is near-linear.

---

## Diversity Analysis

After filtering, we measure whether the surviving dataset is balanced across the axes that matter for model generalization:

| Metric | How measured | Ideal value |
|--------|-------------|-------------|
| Gender balance | Normalized Shannon entropy | 1.0 (uniform) |
| Age balance | Normalized Shannon entropy | 1.0 |
| Accent entropy | Normalized Shannon entropy over accent labels | 1.0 |
| Speaker balance | Entropy of clips-per-speaker distribution | 1.0 |
| Speaker imbalance ratio | max(clips) / median(clips) | < 10 |

**Finding:** Common Voice English skews ~70% male across all age groups. Quality filtering worsens this slightly because male speakers dominate high-upvote clips (which are longer and have better recording quality). Explicitly balancing by gender before the quality filter is a mitigation worth evaluating.

---

## Why Audio Curation Is Harder Than Text

This is worth spelling out explicitly because the intuition from text pipelines doesn't transfer:

1. **You cannot read the data.** A noisy clip looks identical to a clean clip in a file listing. Automated quality metrics are the only way to detect recording failures at scale.

2. **Duplicates aren't byte-identical.** The same utterance recorded twice will differ by recording level, background noise floor, codec quantization, and leading/trailing silence. Byte hashes miss 80–90% of semantic duplicates. Embedding similarity catches them, but requires choosing a threshold that's content-aware.

3. **SNR has no ground truth.** We never have a clean reference signal for crowd-sourced data. All SNR estimates are approximations based on within-clip statistics. This makes the 15 dB threshold a judgment call, not a hard physical measurement.

4. **Demographic bias is latent in quality metrics.** High-SNR clips tend to come from speakers in quiet home offices — demographically skewed toward certain age groups and accents. Aggressive SNR filtering can inadvertently remove accented speech that has slightly higher background noise from different geographic recording environments.

5. **Speaker identity is implicit.** Even when `speaker_id` metadata is present, the same speaker may appear under multiple IDs across recording sessions. Speaker-level overlap is impossible to detect without speaker verification models, which add significant compute.

See [DATA_CARD.md](DATA_CARD.md) for the full analysis of what was included, excluded, and why.

---

## Integration with Downstream ASR / TTS Pipelines

The curated manifest produced by this pipeline is designed for direct use as a training manifest
in Whisper, wav2vec2, EnCodec, or any custom ASR/TTS training workflow.

### Output schema

`filtered_manifest.parquet` columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Unique clip identifier |
| `path` | str | Path to the audio file |
| `sentence` | str | Transcript text |
| `duration_sec` | float | Clip duration in seconds |
| `snr_db` | float | Estimated SNR (reference-free) |
| `silence_ratio` | float | Fraction of silent frames |
| `source` | str | `"real"` or `"synthetic"` |

This schema is compatible with HuggingFace `Trainer`-based fine-tuning pipelines without a
conversion step.

### Synthetic augmentation for demographic gaps

After curation, `scripts/generate_synthetic.py` identifies underrepresented demographic groups
(by gender, age, accent) and generates targeted TTS samples using `edge-tts` to close the gap.
The `AblationEvaluator` in `src/audio_curation/synthetic/evaluator.py` measures the WER impact
of each synthetic mix ratio, enabling data-quality / compute tradeoff analysis:

```bash
python scripts/generate_synthetic.py \
    --manifest outputs/filtered_manifest.parquet \
    --output_dir data/synthetic \
    --max_synthetic_ratio 0.3

python scripts/run_ablations.py \
    --manifest outputs/filtered_manifest.parquet \
    --synthetic_dir data/synthetic \
    --output experiments/results/ablation_results.json
```

### Relevant for TTS reward modeling

The per-clip SNR, silence ratio, and diversity scores in `curation_report.json` can serve as
input features for an acoustic reward model — they quantify recording condition quality in the
same terms that matter for TTS preference optimization (background noise level, dynamic range,
speaker balance). See the companion
[RLHF pipeline repo](https://github.com/kartikmunjal/rlhf-and-reward-modelling-alt)
for TTS RLHF implementation using acoustic reward signals.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use synthetic audio (pure tones, noise, silence) and do not require the Common Voice dataset to be downloaded.

---

## Configuration Reference

`QualityThresholds` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_duration_sec` | 0.5 | Minimum clip duration |
| `max_duration_sec` | 30.0 | Maximum clip duration |
| `min_snr_db` | 15.0 | Minimum estimated SNR |
| `max_silence_ratio` | 0.4 | Maximum fraction of silent frames |
| `clipping_threshold` | 0.99 | Sample amplitude threshold for clipping |
| `min_rms_db` | -40.0 | Minimum overall RMS level |

`DeduplicationEngine` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.97 | Cosine similarity cutoff for near-duplicates |
| `n_lsh_bits` | 18 | LSH hash width (more bits → fewer false positives) |

---

## Dependencies

- **librosa** — audio I/O, MFCC features, silence detection
- **numpy / pandas / scipy** — numerical and tabular operations
- **scikit-learn** — (optional) clustering for extended analysis
- **datasets** — HuggingFace Datasets for Common Voice access
- **soundfile** — writing WAV files
- **matplotlib / seaborn** — visualization

---

*Author: Kartik Munjal — kartikmunjal19@gmail.com*
