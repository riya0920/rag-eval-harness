"""Retrieval: BM25, dense, and rank fusion -- implemented here, not imported.

No framework. The retrieval logic is the part of a RAG system that actually
determines quality, and owning it is the difference between being able to explain
a result and being able to quote a library's defaults.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN = re.compile(r"[a-z0-9$%\.\-]+")

# Stopwords hurt BM25 more than they help here: this corpus is full of short
# functional questions ("how long is the retention period") where the content
# words are the rare ones anyway, and IDF already discounts the common ones.
# The list is deliberately tiny -- removing "not" or "no" would be a correctness
# bug for a corpus this full of negative facts ("never auto-revokes").
STOPWORDS = {"the", "a", "an", "of", "to", "is", "are", "and", "or", "in", "for", "it", "that", "this"}


def tokenize(text: str, drop_stopwords: bool = True) -> list:
    """Tokenise, keeping money and hyphenated identifiers intact.

    The character class deliberately includes `.` and `-` so that `$0.045` and
    `vcpu-hour` survive as single tokens. The consequence is that a
    sentence-final period attaches too, turning `hours.` into a token that never
    matches `hours` -- invisible in BM25, where both sides get the same
    treatment, but silently fatal for key-point matching against the golden set.
    Stripping LEADING and TRAILING punctuation keeps the useful cases and drops
    the artefact.
    """
    toks = []
    for raw in TOKEN.findall(text.lower()):
        tok = raw.strip(".-")
        if tok and not (drop_stopwords and tok in STOPWORDS):
            toks.append(tok)
    return toks


class BM25:
    """Okapi BM25. k1 and b exposed because they are swept in the eval matrix.

    b=0.75 partially normalises for document length; with chunks of roughly equal
    size that matters less than it does over whole documents, which is itself an
    argument for chunking before indexing rather than after.
    """

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(c.text) for c in self.chunks]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0

        self.tf = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for t in self.doc_tokens:
            df.update(set(t))
        n = len(self.chunks)
        # Robertson IDF with the +0.5 smoothing, floored at a small positive value.
        # Without the floor, a term appearing in more than half the corpus gets a
        # NEGATIVE weight and actively penalises documents that contain it.
        self.idf = {
            term: max(math.log((n - freq + 0.5) / (freq + 0.5) + 1.0), 1e-6) for term, freq in df.items()
        }
        self.postings = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.postings[term].append(i)

    def search(self, query: str, k: int = 10) -> list:
        q_terms = tokenize(query)
        scores = defaultdict(float)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(self.avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i].chunk_id, float(s)) for i, s in ranked]


class DenseRetriever:
    """Dense retrieval over an embedding function.

    The embedder is injected. `embeddings.HashingSVDEmbedder` is the default so
    the harness runs offline with zero API cost, and swapping in a real sentence
    encoder is a one-line change -- the eval matrix then measures what that swap
    actually buys, which is the only honest way to justify the dependency.
    """

    def __init__(self, chunks, embedder):
        self.chunks = list(chunks)
        self.embedder = embedder
        mat = embedder.encode([c.text for c in self.chunks])
        self.matrix = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    def search(self, query: str, k: int = 10) -> list:
        q = self.embedder.encode([query])[0]
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i].chunk_id, float(sims[i])) for i in top]


def reciprocal_rank_fusion(rankings, k: int = 10, rrf_k: int = 60) -> list:
    """RRF: score = sum over lists of 1 / (rrf_k + rank).

    Chosen over score-based fusion because BM25 scores and cosine similarities
    live on incomparable scales, and normalising them (min-max, z-score) makes the
    fusion depend on the score *distribution of the current query*, which is
    unstable. RRF only uses rank order, so it cannot be destabilised that way.

    What it gives up: it discards confidence entirely. A result ranked 1 with a
    runaway score and one ranked 1 by a hair contribute identically. When you have
    enough labelled data to fit a learned fusion, that thrown-away signal is
    exactly what buys the improvement -- see docs/ for the crossover argument.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (chunk_id, _score) in enumerate(ranking):
            fused[chunk_id] += 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])[:k]


class HybridRetriever:
    """BM25 + dense, fused with RRF. Mode is switchable so the matrix can ablate."""

    def __init__(self, chunks, embedder=None, mode: str = "hybrid", k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.mode = mode
        self.rrf_k = rrf_k
        self.bm25 = BM25(chunks, k1=k1, b=b) if mode in ("bm25", "hybrid") else None
        self.dense = DenseRetriever(chunks, embedder) if mode in ("dense", "hybrid") else None

    def search(self, query: str, k: int = 10) -> list:
        if self.mode == "bm25":
            return self.bm25.search(query, k)
        if self.mode == "dense":
            return self.dense.search(query, k)
        # Over-fetch before fusing: a document ranked 15th by one retriever and
        # 2nd by the other should still be fusable, and fetching only k would
        # have thrown it away before the fusion ever saw it.
        pool = max(k * 3, 30)
        return reciprocal_rank_fusion(
            [self.bm25.search(query, pool), self.dense.search(query, pool)], k=k, rrf_k=self.rrf_k
        )
