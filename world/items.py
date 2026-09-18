"""Item content authoring: turns a short text description into the small,
fixed-dimension attribute vector item instances are stamped with (see
wiki/decisions.md #22, part 4). Lives entirely in world/'s
content-authoring layer -- nothing here is ever exposed to a fly; only
the resulting numeric vector, via a Percept, crosses that boundary.

`ItemEncoder` is the swap point, same pattern as director/'s
WorldController: `NomicItemEncoder` is the real backend (a local,
frozen, Matryoshka-truncatable text-embedding model -- no training,
just encoding), `HashingItemEncoder` is a zero-dependency stub used to
test everything downstream (jitter, similarity blending, the KC/MBON
circuit once it exists) without network access -- huggingface.co is not
reachable in every environment this project runs in (this one included;
see wiki/decisions.md #24), the same situation ClaudeController is
already in with no API credentials.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class ItemEncoder(ABC):
    output_dim: int

    @abstractmethod
    def encode(self, description: str) -> np.ndarray:
        """A unit-norm vector of length `output_dim` for a short text
        description. Deterministic: same description -> same vector.
        """


class HashingItemEncoder(ItemEncoder):
    """Zero-dependency stub: a classic feature-hashing ("hashing trick")
    bag-of-character-trigrams encoder. Not a semantic model -- similarity
    here comes from shared substrings, not meaning ("smelly meat" and
    "smelly apple" land closer than "meat" and "spider" because they
    share the trigram "sme", not because anything understands smell).
    Good enough to exercise the pipeline (jitter, cosine blending,
    downstream circuits) end to end without a network dependency; not a
    substitute for `NomicItemEncoder`'s actual semantic structure.
    """

    def __init__(self, output_dim: int = 64) -> None:
        self.output_dim = output_dim

    def encode(self, description: str) -> np.ndarray:
        text = description.lower()
        vector = np.zeros(self.output_dim, dtype=np.float64)
        trigrams = [text[i : i + 3] for i in range(len(text) - 2)] or [text]
        for trigram in trigrams:
            digest = hashlib.sha256(trigram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "little") % self.output_dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


class NomicItemEncoder(ItemEncoder):
    """Real backend: `nomic-embed-text-v1.5` (local, frozen, no training),
    truncated to `output_dim` via its native Matryoshka Representation
    Learning support (see wiki/decisions.md #22 for why truncation over a
    separately-fit PCA step -- no fitted vocabulary to go stale as new
    items get added mid-game). Requires the `sentence-transformers`
    package and the model weights, downloaded from huggingface.co on
    first use.

    **Not exercised live in this repo's dev/CI sandbox** -- huggingface.co
    is policy-blocked there (wiki/decisions.md #24), the same situation
    ClaudeController is in with no API credentials (decisions.md #16).
    Test this against the real model on a machine that can reach it
    before relying on it.
    """

    def __init__(self, output_dim: int = 64, model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.output_dim = output_dim
        self._model = SentenceTransformer(model_name, trust_remote_code=True, truncate_dim=output_dim)

    def encode(self, description: str) -> np.ndarray:
        vector = self._model.encode(description, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float64)


def encode_with_jitter(
    encoder: ItemEncoder, description: str, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """The per-instance vector for a newly spawned item: the description's
    prototype vector plus Gaussian jitter, re-normalized to unit norm so
    every downstream consumer can keep using plain cosine similarity
    (dot product, since both operands stay unit-norm).
    """
    prototype = encoder.encode(description)
    noise = rng.normal(0.0, sigma, size=prototype.shape)
    return _normalize(prototype + noise)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector
