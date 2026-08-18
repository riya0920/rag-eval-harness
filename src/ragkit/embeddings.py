"""Embedders behind one interface: `encode(list[str]) -> np.ndarray`.

The default is deliberately local and free. A hosted embedding API would make the
eval harness cost money per run and stop it from being runnable in CI, which
defeats the point of an eval gate. The interface exists so that swapping in a
real encoder is one line -- and so the matrix can *measure* what that swap buys
rather than assuming it.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .retrieval import tokenize


class HashingSVDEmbedder:
    """Hashed bag-of-ngrams -> TF-IDF -> truncated SVD (i.e. LSA).

    Honest description of what this is: a *lexical* embedder with a learned
    low-rank projection. It captures co-occurrence structure, so it will match
    "credential" to a chunk about "service keys" if those words co-occur in the
    corpus. It will NOT capture semantics it never saw -- there is no world
    knowledge in it. That limitation is stated in the README rather than hidden,
    because a dense retriever that is secretly lexical makes a hybrid ablation
    look pointless for the wrong reason.
    """

    def __init__(self, dim: int = 128, n_buckets: int = 2 ** 15, use_bigrams: bool = True, seed: int = 0):
        self.dim = dim
        self.n_buckets = n_buckets
        self.use_bigrams = use_bigrams
        self.seed = seed
        self.components = None
        self.idf = None

    def _features(self, text: str) -> dict:
        toks = tokenize(text)
        grams = list(toks)
        if self.use_bigrams:
            grams += ["%s_%s" % (a, b) for a, b in zip(toks, toks[1:])]
        out = {}
        for g in grams:
            h = int(hashlib.blake2b(g.encode(), digest_size=8, key=str(self.seed).encode()).hexdigest(), 16)
            out[h % self.n_buckets] = out.get(h % self.n_buckets, 0) + 1
        return out

    def _to_matrix(self, texts) -> np.ndarray:
        rows = [self._features(t) for t in texts]
        mat = np.zeros((len(rows), self.n_buckets), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, v in row.items():
                mat[i, j] = 1.0 + np.log(v)  # sublinear tf
        return mat

    def fit(self, corpus_texts) -> "HashingSVDEmbedder":
        mat = self._to_matrix(corpus_texts)
        df = (mat > 0).sum(axis=0)
        n = mat.shape[0]
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        mat = mat * self.idf
        # Keep only buckets the corpus actually uses; the rest are structurally
        # zero and SVD on a 32K-wide mostly-empty matrix is pure waste.
        self.active = np.flatnonzero(mat.any(axis=0))
        dense = mat[:, self.active]
        k = min(self.dim, min(dense.shape) - 1)
        _u, _s, vt = np.linalg.svd(dense, full_matrices=False)
        self.components = vt[:k]
        return self

    def encode(self, texts) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("call fit() on the corpus before encode()")
        mat = self._to_matrix(texts) * self.idf
        return (mat[:, self.active] @ self.components.T).astype(np.float32)


def build_default_embedder(chunks, dim: int = 128) -> HashingSVDEmbedder:
    return HashingSVDEmbedder(dim=dim).fit([c.text for c in chunks])
