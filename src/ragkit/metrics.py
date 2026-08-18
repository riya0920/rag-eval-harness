"""Retrieval metrics. Deterministic, free, and fast enough to run on every PR.

Every metric here is computed at the DOCUMENT level, because the golden set
labels relevant documents. Chunk-level scoring would make the numbers depend on
the chunking config, and the whole point of the experiment matrix is to compare
chunking configs against each other on a fixed yardstick.
"""
from __future__ import annotations

import numpy as np


def chunks_to_docs(ranked_chunk_ids) -> list:
    """Collapse a chunk ranking to a document ranking, keeping first appearance.

    First-appearance is the right rule: if a document's best chunk is at rank 2,
    the document was retrieved at rank 2, not at the rank of its worst chunk.
    """
    seen, out = set(), []
    for cid in ranked_chunk_ids:
        doc = cid.split("#")[0]
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def precision_at_k(ranked_docs, relevant, k: int) -> float:
    if not relevant:
        return float("nan")  # undefined, not zero -- see the unanswerable note below
    top = ranked_docs[:k]
    return len(set(top) & set(relevant)) / max(len(top), 1)


def recall_at_k(ranked_docs, relevant, k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked_docs[:k]) & set(relevant)) / len(relevant)


def mrr(ranked_docs, relevant) -> float:
    if not relevant:
        return float("nan")
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_docs, relevant, k: int) -> float:
    if not relevant:
        return float("nan")
    gains = [1.0 if d in relevant else 0.0 for d in ranked_docs[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(retriever, examples, ks=(1, 3, 5, 10), fetch_k: int = 10) -> dict:
    """Run the whole golden set through a retriever.

    Examples with no relevant documents (the pure-unanswerable ones) are excluded
    from precision/recall/MRR and counted separately. Scoring them as zero would
    punish a retriever for correctly finding nothing, and scoring them as one
    would reward a retriever that returns garbage. Retrieval metrics simply do not
    define them -- refusal is a *generation* metric, and it lives there.
    """
    per_example = []
    by_cat = {}
    scored = 0
    for ex in examples:
        relevant = ex.get("relevant_docs") or []
        ranked_chunks = [cid for cid, _ in retriever.search(ex["question"], max(fetch_k, max(ks)))]
        ranked_docs = chunks_to_docs(ranked_chunks)
        row = {
            "id": ex["id"],
            "category": ex["category"],
            "has_labels": bool(relevant),
            "retrieved_docs": ranked_docs[: max(ks)],
        }
        if relevant:
            scored += 1
            for k in ks:
                row["precision@%d" % k] = precision_at_k(ranked_docs, relevant, k)
                row["recall@%d" % k] = recall_at_k(ranked_docs, relevant, k)
                row["ndcg@%d" % k] = ndcg_at_k(ranked_docs, relevant, k)
            row["mrr"] = mrr(ranked_docs, relevant)
            by_cat.setdefault(ex["category"], []).append(row)
        per_example.append(row)

    metric_names = ["mrr"] + ["%s@%d" % (m, k) for k in ks for m in ("precision", "recall", "ndcg")]
    labelled = [r for r in per_example if r["has_labels"]]
    overall = {m: float(np.mean([r[m] for r in labelled])) for m in metric_names} if labelled else {}
    per_category = {
        cat: {m: float(np.mean([r[m] for r in rows])) for m in metric_names} for cat, rows in by_cat.items()
    }
    return {
        "overall": overall,
        "per_category": per_category,
        "n_scored": scored,
        "n_total": len(examples),
        "n_unlabelled_excluded": len(examples) - scored,
        "per_example": per_example,
    }
