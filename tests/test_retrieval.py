"""Tests for retrieval logic and metric definitions.

The metric tests matter more than the retrieval tests. A retrieval bug makes the
numbers worse; a metric bug makes the numbers *wrong*, and wrong numbers are how
an eval harness stops being an eval harness.
"""
import math

import numpy as np
import pytest

from ragkit.chunking import Chunk, chunk_documents, parse_corpus
from ragkit.embeddings import HashingSVDEmbedder
from ragkit.experiment import CORPUS, load_golden
from ragkit.metrics import chunks_to_docs, evaluate_retrieval, mrr, ndcg_at_k, precision_at_k, recall_at_k
from ragkit.retrieval import BM25, HybridRetriever, reciprocal_rank_fusion, tokenize


@pytest.fixture(scope="module")
def corpus_text():
    with open(CORPUS, encoding="utf-8") as fh:
        return fh.read()


def test_corpus_parses_into_addressable_docs(corpus_text):
    docs = parse_corpus(corpus_text)
    assert len(docs) >= 20
    assert all(d.startswith("doc:") for d in docs)
    assert "doc:billing-tiers" in docs


def test_golden_set_references_only_real_docs(corpus_text):
    """A golden set that points at documents which do not exist is silently broken."""
    docs = set(parse_corpus(corpus_text))
    for ex in load_golden():
        for ref in ex.get("relevant_docs", []):
            assert ref in docs, "%s references missing doc %s" % (ex["id"], ref)


def test_golden_set_has_all_four_categories_including_unanswerable():
    cats = {ex["category"] for ex in load_golden()}
    assert {"factual", "multi-hop", "unanswerable", "ambiguous"} <= cats
    unanswerable = [e for e in load_golden() if e["category"] == "unanswerable"]
    assert len(unanswerable) >= 5


def test_golden_ids_are_unique():
    ids = [e["id"] for e in load_golden()]
    assert len(ids) == len(set(ids))


def test_chunking_never_splits_mid_sentence():
    docs = {"doc:x": "First fact here. Second fact here. Third fact here. Fourth fact here."}
    chunks = chunk_documents(docs, target_tokens=6, overlap_sentences=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.rstrip().endswith(".")


def test_chunking_overlap_repeats_the_boundary_sentence():
    docs = {"doc:x": "Alpha one two. Bravo three four. Charlie five six. Delta seven eight."}
    no_overlap = chunk_documents(docs, target_tokens=6, overlap_sentences=0)
    with_overlap = chunk_documents(docs, target_tokens=6, overlap_sentences=1)
    assert len(with_overlap) >= len(no_overlap)
    # the last sentence of chunk i must reappear at the start of chunk i+1
    assert with_overlap[0].text.split(".")[-2].strip() in with_overlap[1].text


def test_tokenizer_keeps_money_and_negations():
    toks = tokenize("Overage is billed at $0.045 per vCPU-hour and is not refundable")
    assert "$0.045" in toks
    assert "vcpu-hour" in toks
    assert "not" in toks  # dropping negations would invert facts in this corpus


def test_bm25_idf_is_never_negative():
    """Unfloored Robertson IDF goes negative for terms in >half the corpus,
    which makes containing the query term *lower* your score."""
    chunks = [Chunk("d%d#0" % i, "d%d" % i, "common term here", 0) for i in range(10)]
    chunks.append(Chunk("d9#1", "d9", "rare unicorn", 1))
    bm = BM25(chunks)
    assert all(v > 0 for v in bm.idf.values())


def test_bm25_ranks_the_document_containing_the_rare_term_first():
    chunks = [
        Chunk("a#0", "a", "billing invoices are issued monthly", 0),
        Chunk("b#0", "b", "service keys rotate every ninety days", 0),
        Chunk("c#0", "c", "regions include us-east-2 and eu-central-1", 0),
    ]
    hits = BM25(chunks).search("how often do service keys rotate", k=3)
    assert hits[0][0] == "b#0"


def test_rrf_prefers_the_item_ranked_well_by_both():
    a = [("x", 9.0), ("y", 8.0), ("z", 1.0)]
    b = [("y", 0.9), ("x", 0.2), ("w", 0.1)]
    fused = dict(reciprocal_rank_fusion([a, b], k=4))
    # y is (2nd, 1st); x is (1st, 2nd) -- symmetric, so they tie
    assert math.isclose(fused["x"], fused["y"], rel_tol=1e-9)
    # z appears once at rank 3, w once at rank 3: both below the two-list items
    assert fused["x"] > fused["z"]


def test_rrf_is_immune_to_score_scale():
    """The reason RRF was chosen over score-normalised fusion."""
    a = [("x", 1e6), ("y", 1.0)]
    b = [("y", 0.001), ("x", 0.0005)]
    scaled_a = [(i, s * 1e9) for i, s in a]
    assert reciprocal_rank_fusion([a, b], k=2) == reciprocal_rank_fusion([scaled_a, b], k=2)


def test_chunks_to_docs_keeps_best_rank_per_doc():
    ranked = ["doc:a#3", "doc:b#0", "doc:a#0", "doc:c#1"]
    assert chunks_to_docs(ranked) == ["doc:a", "doc:b", "doc:c"]


def test_metrics_are_undefined_not_zero_without_labels():
    """Scoring an unanswerable question as recall=0 would punish correct behaviour."""
    assert math.isnan(recall_at_k(["a"], [], 5))
    assert math.isnan(precision_at_k(["a"], [], 5))
    assert math.isnan(mrr(["a"], []))


def test_metric_arithmetic():
    assert precision_at_k(["a", "b", "c", "d"], ["a", "c"], 4) == 0.5
    assert recall_at_k(["a", "b"], ["a", "c"], 2) == 0.5
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert ndcg_at_k(["a", "x"], ["a"], 2) > ndcg_at_k(["x", "a"], ["a"], 2)
    assert ndcg_at_k(["a"], ["a"], 1) == pytest.approx(1.0)


def test_evaluate_excludes_unlabelled_from_the_average():
    class Fixed:
        def search(self, q, k):
            return [("doc:a#0", 1.0)]

    examples = [
        {"id": "1", "category": "factual", "question": "q", "relevant_docs": ["doc:a"]},
        {"id": "2", "category": "unanswerable", "question": "q", "relevant_docs": []},
    ]
    res = evaluate_retrieval(Fixed(), examples, ks=(1,))
    assert res["n_scored"] == 1
    assert res["n_unlabelled_excluded"] == 1
    assert res["overall"]["recall@1"] == 1.0  # not dragged to 0.5 by the unlabelled row


def test_embedder_is_deterministic_and_shaped():
    chunks = [Chunk("d%d#0" % i, "d%d" % i, "text about topic %d and other words" % i, 0) for i in range(12)]
    emb = HashingSVDEmbedder(dim=8).fit([c.text for c in chunks])
    a = emb.encode(["text about topic 3"])
    b = emb.encode(["text about topic 3"])
    assert a.shape[1] <= 8
    assert np.allclose(a, b)


def test_hybrid_matches_a_fact_the_lexical_query_words_do_not_contain():
    """End-to-end smoke over the real corpus, on a paraphrased query."""
    from ragkit.chunking import load_chunks
    from ragkit.embeddings import build_default_embedder

    chunks = load_chunks(CORPUS, target_tokens=80)
    r = HybridRetriever(chunks, embedder=build_default_embedder(chunks), mode="hybrid")
    docs = chunks_to_docs([cid for cid, _ in r.search("how long until a session credential stops working", 10)])
    assert "doc:auth-tokens" in docs[:3]
