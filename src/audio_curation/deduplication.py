"""
Audio deduplication via acoustic fingerprinting and embedding similarity.

Why this is harder than text deduplication:
  - Two recordings of the exact same sentence by the same speaker may differ by
    a few milliseconds of silence padding, a slight volume change, or a resample
    artefact — none of which matter perceptually but all of which produce
    different byte hashes.
  - Embedding-based near-duplicate detection (MFCC cosine similarity) bridges
    this gap at the cost of more compute. We run it in two stages:
        1. Exact hash match  → O(n) pass, eliminates obvious dupes.
        2. MFCC LSH buckets  → O(n) pass, flags near-duplicates for review.
    The LSH approach avoids the O(n^2) brute-force similarity comparison that
    becomes prohibitive past ~100k samples.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Optional

import librosa
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fingerprinting primitives
# ---------------------------------------------------------------------------

def audio_md5(audio: np.ndarray) -> str:
    """Byte-level MD5 hash. Catches only perfect duplicates."""
    # Quantize to int16 to normalize float precision noise
    pcm = (audio * 32767).astype(np.int16)
    return hashlib.md5(pcm.tobytes()).hexdigest()


def compute_mfcc_embedding(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Compact MFCC embedding: mean + std of each coefficient over time.

    Shape: (2 * n_mfcc,) — intentionally small so we can compare millions
    of pairs in memory. Delta and delta-delta are excluded to keep the
    representation sequence-length-invariant.
    """
    if len(audio) == 0:
        return np.zeros(2 * n_mfcc)

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # (n_mfcc, T)

    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    embedding = np.concatenate([mean, std]).astype(np.float32)
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both unit vectors already


# ---------------------------------------------------------------------------
# LSH bucketing for scalable near-dup detection
# ---------------------------------------------------------------------------

class RandomProjectionLSH:
    """
    Locality-sensitive hashing via random binary projections.

    Maps each embedding to a `n_bits`-bit bucket key. Embeddings in the same
    bucket are candidates for full cosine-similarity comparison.

    Increasing `n_bits` reduces false positives at the cost of fewer collisions
    (potentially missing near-duplicates). 16-24 bits works well in practice.
    """

    def __init__(self, dim: int, n_bits: int = 16, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.planes = rng.standard_normal((n_bits, dim)).astype(np.float32)

    def hash(self, embedding: np.ndarray) -> int:
        projections = self.planes @ embedding  # (n_bits,)
        bits = (projections > 0).astype(np.uint32)
        return int(sum(b << i for i, b in enumerate(bits)))


# ---------------------------------------------------------------------------
# Main deduplication engine
# ---------------------------------------------------------------------------

class DeduplicationEngine:
    """
    Two-pass deduplication: exact hash → near-duplicate LSH.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity above which two clips are considered near-duplicates.
        0.98 is a good default; lower values (0.92) catch more aggressive paraphrases
        recorded by the same speaker across sessions.
    n_lsh_bits : int
        LSH hash width. More bits → fewer candidate pairs but higher miss rate.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.97,
        n_lsh_bits: int = 18,
        embedding_dim: int = 40,  # 2 * n_mfcc
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.lsh = RandomProjectionLSH(dim=embedding_dim, n_bits=n_lsh_bits)
        self._embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # Stage 1 — exact duplicates
    # ------------------------------------------------------------------

    def find_exact_duplicates(
        self, ids: list[str], audio_list: list[np.ndarray]
    ) -> dict[str, list[str]]:
        """
        Return a mapping from canonical id → [duplicate ids].
        Keeps the first occurrence as canonical.
        """
        seen: dict[str, str] = {}      # md5 → first id
        groups: dict[str, list[str]] = defaultdict(list)

        for uid, audio in tqdm(zip(ids, audio_list), total=len(ids), desc="Exact dedup"):
            h = audio_md5(audio)
            if h in seen:
                groups[seen[h]].append(uid)
            else:
                seen[h] = uid

        return dict(groups)

    # ------------------------------------------------------------------
    # Stage 2 — near-duplicates
    # ------------------------------------------------------------------

    def find_near_duplicates(
        self,
        ids: list[str],
        audio_list: list[np.ndarray],
        sr_list: list[int],
        exact_dup_ids: Optional[set[str]] = None,
    ) -> dict[str, list[str]]:
        """
        Return near-duplicate groups via LSH + cosine verification.

        Already-identified exact duplicates are excluded from this pass.
        """
        exact_dup_ids = exact_dup_ids or set()
        embeddings: dict[str, np.ndarray] = {}
        buckets: dict[int, list[str]] = defaultdict(list)

        # Embed and bucket
        for uid, audio, sr in tqdm(
            zip(ids, audio_list, sr_list),
            total=len(ids),
            desc="Computing embeddings",
        ):
            if uid in exact_dup_ids:
                continue
            emb = compute_mfcc_embedding(audio, sr)
            embeddings[uid] = emb
            bucket = self.lsh.hash(emb)
            buckets[bucket].append(uid)

        # Verify candidates within each bucket
        near_dup_groups: dict[str, list[str]] = defaultdict(list)
        assigned: set[str] = set()

        for bucket_ids in tqdm(buckets.values(), desc="Near-dup verification"):
            if len(bucket_ids) < 2:
                continue
            for i in range(len(bucket_ids)):
                for j in range(i + 1, len(bucket_ids)):
                    a, b = bucket_ids[i], bucket_ids[j]
                    if a in assigned or b in assigned:
                        continue
                    sim = cosine_similarity(embeddings[a], embeddings[b])
                    if sim >= self.similarity_threshold:
                        near_dup_groups[a].append(b)
                        assigned.add(b)

        return dict(near_dup_groups)

    # ------------------------------------------------------------------
    # Combined pipeline
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        ids: list[str],
        audio_list: list[np.ndarray],
        sr_list: list[int],
    ) -> tuple[list[str], dict]:
        """
        Run full two-pass deduplication. Returns (kept_ids, report).
        """
        exact_groups = self.find_exact_duplicates(ids, audio_list)
        exact_dup_ids: set[str] = set(uid for dups in exact_groups.values() for uid in dups)

        near_groups = self.find_near_duplicates(ids, audio_list, sr_list, exact_dup_ids=exact_dup_ids)
        near_dup_ids: set[str] = set(uid for dups in near_groups.values() for uid in dups)

        all_dup_ids = exact_dup_ids | near_dup_ids
        kept_ids = [uid for uid in ids if uid not in all_dup_ids]

        report = {
            "total_input": len(ids),
            "exact_duplicate_groups": len(exact_groups),
            "exact_duplicates_removed": len(exact_dup_ids),
            "near_duplicate_groups": len(near_groups),
            "near_duplicates_removed": len(near_dup_ids),
            "total_removed": len(all_dup_ids),
            "total_kept": len(kept_ids),
            "retention_rate": len(kept_ids) / max(len(ids), 1),
        }

        logger.info(
            "Dedup: removed %d exact + %d near-dups; kept %d / %d (%.1f%%)",
            len(exact_dup_ids),
            len(near_dup_ids),
            len(kept_ids),
            len(ids),
            100 * len(kept_ids) / max(len(ids), 1),
        )

        return kept_ids, report
