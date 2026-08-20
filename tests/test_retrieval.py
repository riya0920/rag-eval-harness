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


# --------------------------------------------------------------------------
# generation, cost, and the judge
# --------------------------------------------------------------------------

def test_golden_set_meets_the_spec_floor():
    """~100 examples with all four categories represented."""
    from collections import Counter

    g = load_golden()
    assert len(g) >= 100, "spec asks for ~100 examples, have %d" % len(g)
    counts = Counter(e["category"] for e in g)
    for cat in ("factual", "multi-hop", "unanswerable", "ambiguous"):
        assert counts[cat] >= 10, "%s has only %d" % (cat, counts[cat])


def test_extractive_generator_refuses_when_context_is_irrelevant():
    from ragkit.chunking import Chunk
    from ragkit.generation import ExtractiveGenerator

    gen = ExtractiveGenerator(min_overlap=2)
    chunks = [Chunk("d#0", "d", "Bananas are yellow and grow in bunches.", 0)]
    answer = gen.generate("What is the audit log retention period?", chunks)
    assert answer.refused


def test_extractive_answers_are_faithful_by_construction():
    """The control property: a copy-paste generator must score ~1.0.

    If the faithfulness metric cannot score verbatim extraction highly, the
    metric is broken, not the generator.
    """
    from ragkit.chunking import load_chunks
    from ragkit.generation import ExtractiveGenerator
    from ragkit.judge import faithfulness

    chunks = load_chunks(CORPUS, target_tokens=80)
    ctx = [c for c in chunks if c.doc_id == "doc:audit-log"]
    gen = ExtractiveGenerator()
    ans = gen.generate("How long are audit log entries retained?", ctx)
    result = faithfulness(ans.text, [c.text for c in ctx])
    assert result.score == 1.0


def test_faithfulness_catches_an_injected_fabrication():
    from ragkit.judge import faithfulness

    context = ["Audit log entries are immutable and retained for 400 days."]
    honest = "Audit log entries are immutable and retained for 400 days."
    lying = honest + " Entries can be edited by an administrator within 24 hours."
    assert faithfulness(honest, context).score == 1.0
    assert faithfulness(lying, context).score < 1.0


def test_claim_must_be_supported_by_a_SINGLE_chunk():
    """A claim stitched from fragments of different chunks is a fabrication."""
    from ragkit.judge import claim_supported

    chunks = ["The Scale tier costs $2,500 per month.", "GPU workloads run in us-east-2."]
    stitched = "The Scale tier GPU workloads cost $2,500 in us-east-2 per month."
    assert not claim_supported(stitched, chunks, threshold=0.9)


def test_key_point_coverage_scores_missing_facts():
    from ragkit.judge import key_point_coverage

    result = key_point_coverage("Session tokens expire after 12 hours.",
                                ["12 hours", "cannot be renewed"])
    assert result["n_points"] == 2
    assert result["covered"] == 1
    assert "cannot be renewed" in result["missing"]


def test_refusal_correctness_by_category():
    from ragkit.judge import refusal_correct

    assert refusal_correct(True, "unanswerable")
    assert not refusal_correct(False, "unanswerable")
    assert refusal_correct(False, "factual")
    assert not refusal_correct(True, "factual")


def test_cohens_kappa_punishes_a_degenerate_judge():
    """A judge that always says one label agrees often but knows nothing."""
    from ragkit.judge import cohens_kappa

    human = ["faithful"] * 90 + ["unfaithful"] * 10
    always_faithful = ["faithful"] * 100
    assert cohens_kappa(human, always_faithful) == 0.0 or cohens_kappa(human, always_faithful) < 0.01

    perfect = list(human)
    assert cohens_kappa(human, perfect) == 1.0


def test_cache_key_includes_the_context_not_just_the_question():
    """Otherwise a corpus update silently serves a stale answer."""
    from ragkit.generation import ResponseCache

    a = ResponseCache.key("what is the limit?", ["doc:a#0"])
    b = ResponseCache.key("what is the limit?", ["doc:b#0"])
    assert a != b


def test_cost_tracker_does_not_bill_cache_hits():
    from ragkit.generation import CostTracker

    t = CostTracker()
    t.record(1000, 100, cached=False)
    first = t.cost_usd
    t.record(1000, 100, cached=True)
    assert t.cost_usd == first
    assert t.summary()["cache_hits"] == 1
    assert t.summary()["billed_calls"] == 1


def test_second_identical_call_is_served_from_cache():
    from ragkit.chunking import Chunk
    from ragkit.generation import CachedGenerator, CostTracker, ExtractiveGenerator, ResponseCache

    chunks = [Chunk("d#0", "d", "The retention period is 400 days for audit entries.", 0)]
    gen = CachedGenerator(ExtractiveGenerator(), ResponseCache(), CostTracker())
    first = gen.generate("What is the retention period?", chunks)
    second = gen.generate("What is the retention period?", chunks)
    assert not first.cached and second.cached
    assert second.text == first.text
    assert gen.tracker.summary()["billed_calls"] == 1


# --------------------------------------------------------------------------
# contradiction detection -- the blind spot the judge validation exposed
# --------------------------------------------------------------------------

def test_polarity_flip_is_detected():
    from ragkit.contradiction import detect

    ctx = ["SLA credits are applied to a future invoice; they are never paid in cash."]
    claim = "SLA credits are paid in cash to the customer."
    r = detect(claim, ctx)
    assert r.contradicts and r.rule == "polarity_flip"


def test_numeric_mismatch_is_detected():
    from ragkit.contradiction import detect

    ctx = ["Audit log entries are immutable and retained for 400 days."]
    r = detect("Audit log entries are immutable and retained for 40 days.", ctx)
    assert r.contradicts and r.rule == "numeric_mismatch"


def test_antonym_substitution_is_detected():
    from ragkit.contradiction import detect

    ctx = ["Enabling residency disables cross-region reads entirely."]
    r = detect("Enabling residency enables cross-region reads entirely.", ctx)
    assert r.contradicts and r.rule == "antonym_substitution"


def test_a_faithful_restatement_is_not_flagged():
    """The detector must not fire on agreement, or it destroys precision."""
    from ragkit.contradiction import detect

    ctx = ["Peering connections take up to 15 minutes to become active."]
    r = detect("Peering connections take up to 15 minutes to become active.", ctx)
    assert not r.contradicts


def test_off_topic_claim_returns_unknown_not_supported():
    """Degrading to 'I don't know' is safe; degrading to 'fine' is not."""
    from ragkit.contradiction import detect

    ctx = ["Peering connections take up to 15 minutes to become active."]
    r = detect("Bananas ripen faster in warm weather conditions.", ctx)
    assert not r.contradicts
    assert r.rule == "unknown"


def test_faithfulness_now_rejects_a_contradiction_that_token_overlap_accepts():
    """The exact f041 failure, pinned so it cannot regress."""
    from ragkit.judge import claim_supported, faithfulness

    ctx = ["SLA credits are 10% of the monthly bill, rising to 50% below 95.0%. "
           "Credits must be claimed within 30 days and are never paid in cash."]
    fabrication = "Credits below 99.0% uptime are paid in cash within 14 days."

    # Token overlap alone accepts it -- that is the documented blind spot.
    assert claim_supported(fabrication, ctx, threshold=0.5)
    # The full faithfulness check, with contradiction detection, rejects it.
    assert faithfulness(fabrication, ctx).score < 1.0
    # And disabling the second gate reproduces the old, wrong behaviour.
    assert faithfulness(fabrication, ctx, check_contradiction=False).score == 1.0


def test_contradiction_rule_is_reported_not_just_a_boolean():
    """Which gate rejected a claim matters: invented and reversed are different bugs."""
    from ragkit.judge import faithfulness

    ctx = ["Backups are retained for 14 days on the Team tier."]
    result = faithfulness("Backups are retained for 99 days on the Team tier.", ctx)
    assert result.contradictions
    assert result.contradictions[0]["rule"] == "numeric_mismatch"
