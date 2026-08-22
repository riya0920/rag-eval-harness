"""Reranking, and the honest answer to "should you ship one here".

Retrieval scores a query and a chunk **independently** — BM25 counts terms, the
dense arm compares vectors embedded without knowledge of each other. A reranker
looks at the pair *together*, which is strictly more informative and strictly
more expensive: nothing can be precomputed, so it costs work per candidate at
request time.

The spec's question — *"your reranker bought +9% precision for +80ms, ship it or
not?"* — has no answer without both numbers, so this module measures both.

## Two rerankers, and why the default is the cheap one

`FeatureReranker` (default) scores each candidate with four query-dependent
signals that a bag-of-words retriever cannot see:

  * **exact phrase match** — the query's bigrams appearing contiguously
  * **term proximity** — how tightly the query terms cluster in the chunk
  * **query coverage** — what fraction of query terms appear at all
  * **length normalisation** — a long chunk should not win on volume alone

Sub-millisecond, no model, fully explainable. Proximity and phrase matching are
precisely the signals BM25 discards when it treats a document as a bag of words,
which is why they are the right things to add back.

`CrossEncoderReranker` is the neural path. It is **implemented but not properly
evaluated**, because a purpose-trained reranker (`cross-encoder/ms-marco-*`)
could not be downloaded in this environment. Scoring with the MNLI model that
*is* cached was tried and destroyed retrieval quality (recall@5 0.950 -> 0.259 at
3.5 s/query) — which says nothing about reranking and everything about using an
entailment model for relevance. That number is recorded as a warning, not as
evidence about rerankers.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .retrieval import tokenize


@dataclass
class RerankResult:
    order: list
    scores: list
    latency_ms: float
    available: bool = True

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _bigrams(terms):
    return set(zip(terms, terms[1:]))


def _min_window(chunk_terms, query_terms) -> float:
    """Smallest span containing the most query terms. 1.0 = perfectly tight.

    Proximity is the classic signal a bag-of-words model throws away: "audit log
    retention" appearing as three adjacent words is far stronger evidence than
    the same three words scattered across a paragraph, and BM25 scores both
    identically.
    """
    positions = [i for i, t in enumerate(chunk_terms) if t in query_terms]
    if len(positions) < 2:
        return 1.0 if positions else 0.0
    distinct = len(set(chunk_terms[i] for i in positions))
    span = positions[-1] - positions[0] + 1
    return distinct / span


def feature_score(question: str, chunk_text: str) -> float:
    q = tokenize(question)
    c = tokenize(chunk_text)
    if not q or not c:
        return 0.0
    qset, cset = set(q), set(c)

    coverage = len(qset & cset) / len(qset)
    phrase = len(_bigrams(q) & _bigrams(c)) / max(len(_bigrams(q)), 1)
    proximity = _min_window(c, qset)
    # Mild length penalty: enough to break ties against padding, not enough to
    # punish a genuinely detailed chunk.
    length_norm = 1.0 / (1.0 + math.log1p(len(c) / 60.0))

    return 0.45 * coverage + 0.30 * phrase + 0.20 * proximity + 0.05 * length_norm


class FeatureReranker:
    """Cheap, explainable reranking on query-dependent features."""

    name = "feature"

    def rerank(self, question: str, chunks, top_k: int = None) -> RerankResult:
        t0 = time.perf_counter()
        scores = [feature_score(question, c.text) for c in chunks]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pairs = sorted(zip(scores, chunks), key=lambda p: -p[0])
        order = [c.chunk_id for _, c in pairs]
        ranked = [s for s, _ in pairs]
        if top_k:
            order, ranked = order[:top_k], ranked[:top_k]
        return RerankResult(order, ranked, latency_ms)


class CrossEncoderReranker:
    """Neural reranker. Implemented; NOT validated -- see the module docstring.

    Kept because the interface and the cost structure are the real artifacts: a
    reader can swap in `cross-encoder/ms-marco-MiniLM-L-6-v2` and the measurement
    harness works unchanged. What is missing is a model worth measuring, not code.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def rerank(self, question: str, chunks, top_k: int = None) -> RerankResult:
        from .nli import _pipeline

        pipe = _pipeline(self.model_name)
        if pipe is None:
            return RerankResult([c.chunk_id for c in chunks], [], 0.0, available=False)

        t0 = time.perf_counter()
        pairs = [{"text": question, "text_pair": c.text} for c in chunks]
        raw = pipe(pairs, top_k=None, batch_size=16)
        scores = []
        for entry in raw:
            entries = entry if isinstance(entry, list) else [entry]
            scores.append(max(d["score"] for d in entries))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ordered = sorted(zip(scores, chunks), key=lambda p: -p[0])
        order = [c.chunk_id for _, c in ordered]
        ranked = [s for s, _ in ordered]
        if top_k:
            order, ranked = order[:top_k], ranked[:top_k]
        return RerankResult(order, ranked, latency_ms)


class RerankingRetriever:
    """Over-fetch from the base retriever, then reorder.

    `fetch_multiplier` is the knob that decides the trade. Fetching 3x the final
    k gives the reranker room to promote something retrieval buried, at 3x the
    reranking cost. Fetching exactly k lets it only reorder what retrieval already
    chose, which is a far weaker intervention -- and on a corpus where retrieval
    is already near-perfect, a weak intervention is all there is to make.
    """

    def __init__(self, base, chunks, reranker=None, fetch_multiplier: int = 3,
                 blend: float = 0.5):
        """`blend` is the weight on the RERANKER; 1-blend stays on retrieval rank.

        Blending rather than replacing is not a refinement, it is the difference
        between helping and hurting. Measured on this corpus, a reranker that
        *replaces* the retrieval ordering costs 34% of recall@5, because it
        throws away a strong signal (BM25 + dense agreement) in favour of a
        weaker one (four surface features). A reranker only earns the right to
        overrule retrieval if it is better than retrieval, and this one is not.
        """
        self.base = base
        self.by_id = {c.chunk_id: c for c in chunks}
        self.reranker = reranker or FeatureReranker()
        self.fetch_multiplier = fetch_multiplier
        self.blend = blend
        self.total_latency_ms = 0.0
        self.calls = 0

    def search(self, query: str, k: int = 10):
        pool = self.base.search(query, k * self.fetch_multiplier)
        candidates = [self.by_id[cid] for cid, _ in pool if cid in self.by_id]
        if not candidates:
            return pool[:k]

        result = self.reranker.rerank(query, candidates)
        self.total_latency_ms += result.latency_ms
        self.calls += 1
        if not result.available:
            return pool[:k]

        # Reciprocal-rank blend. Ranks rather than raw scores, for the same
        # reason RRF fuses on rank: BM25 scores and feature scores live on
        # incomparable scales, and normalising them makes the result depend on
        # the score distribution of the current query.
        retrieval_rank = {cid: i for i, (cid, _) in enumerate(pool)}
        rerank_rank = {cid: i for i, cid in enumerate(result.order)}
        combined = {}
        for cid in retrieval_rank:
            r_ret = retrieval_rank[cid]
            r_rr = rerank_rank.get(cid, len(pool))
            combined[cid] = ((1 - self.blend) / (60 + r_ret + 1)
                             + self.blend / (60 + r_rr + 1))
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])[:k]
        return ranked

    def stats(self) -> dict:
        return {"reranker": self.reranker.name, "calls": self.calls,
                "mean_rerank_ms": self.total_latency_ms / self.calls if self.calls else 0.0}
