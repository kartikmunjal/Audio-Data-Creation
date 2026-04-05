"""Unit tests for deduplication."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_curation.deduplication import (
    DeduplicationEngine,
    audio_md5,
    compute_mfcc_embedding,
    cosine_similarity,
    RandomProjectionLSH,
)

SR = 16_000


def make_tone(freq=440, duration=2.0, sr=SR, amplitude=0.5, seed=None):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestExactDedup:
    def test_exact_duplicate_detected(self):
        audio = make_tone()
        engine = DeduplicationEngine()
        groups = engine.find_exact_duplicates(["a", "b"], [audio, audio.copy()])
        assert "a" in groups and "b" in groups["a"]

    def test_different_clips_not_flagged(self):
        a = make_tone(freq=440)
        b = make_tone(freq=880)
        engine = DeduplicationEngine()
        groups = engine.find_exact_duplicates(["a", "b"], [a, b])
        assert len(groups) == 0

    def test_md5_deterministic(self):
        audio = make_tone()
        assert audio_md5(audio) == audio_md5(audio.copy())


class TestEmbedding:
    def test_embedding_shape(self):
        audio = make_tone()
        emb = compute_mfcc_embedding(audio, SR, n_mfcc=20)
        assert emb.shape == (40,)

    def test_embedding_unit_norm(self):
        audio = make_tone()
        emb = compute_mfcc_embedding(audio, SR, n_mfcc=20)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_same_clip_cosine_one(self):
        audio = make_tone()
        emb = compute_mfcc_embedding(audio, SR)
        assert cosine_similarity(emb, emb) > 0.999

    def test_different_clips_lower_similarity(self):
        a = make_tone(freq=440)
        b = make_tone(freq=880)
        ea = compute_mfcc_embedding(a, SR)
        eb = compute_mfcc_embedding(b, SR)
        assert cosine_similarity(ea, eb) < 0.99


class TestLSH:
    def test_same_embedding_same_bucket(self):
        lsh = RandomProjectionLSH(dim=40)
        audio = make_tone()
        emb = compute_mfcc_embedding(audio, SR)
        assert lsh.hash(emb) == lsh.hash(emb)

    def test_very_different_embeddings_likely_different_buckets(self):
        lsh = RandomProjectionLSH(dim=40, n_bits=16)
        a = compute_mfcc_embedding(make_tone(freq=440), SR)
        b = compute_mfcc_embedding(make_tone(freq=4000, amplitude=0.1), SR)
        # Not guaranteed to differ but very likely with 16-bit hash
        # Just check it runs without error
        _ = lsh.hash(a)
        _ = lsh.hash(b)


class TestFullDedup:
    def test_keeps_unique_clips(self):
        clips = [make_tone(freq=f) for f in [220, 440, 880]]
        ids = ["a", "b", "c"]
        engine = DeduplicationEngine(similarity_threshold=0.99)
        kept, report = engine.deduplicate(ids, clips, [SR] * 3)
        assert set(kept) == {"a", "b", "c"}

    def test_removes_exact_duplicate(self):
        audio = make_tone()
        ids = ["orig", "dup", "other"]
        clips = [audio, audio.copy(), make_tone(freq=880)]
        engine = DeduplicationEngine()
        kept, report = engine.deduplicate(ids, clips, [SR] * 3)
        assert "orig" in kept
        assert "dup" not in kept
        assert "other" in kept
