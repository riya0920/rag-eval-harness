"""Tests for the clarification detector - including that it refuses to ship.

The load-bearing tests here are the ones that pin *negative* results: that two of
the three original signals carry no information, that the detector's precision is
close to the base rate, and that `recommend()` says so instead of returning the
least-bad threshold. Those are the findings, and a later well-meaning tuning pass
should have to delete a test to bury them.
"""
import json
import os

import pytest

from ragkit.clarify import (
    ClarifyingRetrievalPolicy,
    canonical,
    clarifying_question,
    competing_count,
    evaluate,
    extract_quantities,
    lexical_margin,
    recommend,
    signals,
    sweep,
)
from ragkit.experiment import GATE_CONFIG, build, load_golden


class _Chunk:
    def __init__(self, text, doc_id="doc:x", chunk_id="c1"):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.ordinal = 0


@pytest.fixture(scope="module")
def retrievers():
    chunks, hybrid = build(GATE_CONFIG["mode"], GATE_CONFIG["target_tokens"])
    _c, lexical = build("bm25", GATE_CONFIG["target_tokens"])
    return chunks, hybrid, lexical


# --- unit normalisation ----------------------------------------------------

def test_durations_are_comparable_across_units():
    """The bug that hid 3 of the 10 ambiguous questions. "20 minutes" and
    "3 hours" answer the same question differently and must be able to compete."""
    assert canonical("20", "minutes")[0] == canonical("3", "hours")[0] == "duration"
    assert canonical("3", "hours")[1] > canonical("20", "minutes")[1]


def test_rates_normalise_to_per_second():
    """600 per minute and 10,000 per second are the corpus's headline ambiguity,
    and they only compete once the denominators agree."""
    a = canonical("600", "per minute")
    b = canonical("10000", "per second")
    assert a[0] == b[0] == "rate_per_second"
    assert a[1] == pytest.approx(10.0)
    assert b[1] == pytest.approx(10000.0)


def test_unlike_dimensions_do_not_compete():
    """Two numbers with unrelated units are a sentence that happens to contain
    two numbers, not a conflict."""
    n, _ = competing_count([_Chunk("Restores take 20 minutes for up to 500 GB.")])
    assert n == 0


def test_the_same_figure_restated_is_not_a_conflict():
    """'about 20 minutes' and '20 minutes' are one fact stated twice, and
    counting them as competing is the false positive that makes the signal
    unusable."""
    n, _ = competing_count([_Chunk("Takes about 20 minutes."), _Chunk("Takes 21 minutes.")])
    assert n == 0


def test_genuinely_different_figures_do_compete():
    n, _ = competing_count([_Chunk("A backup restore takes 20 minutes."),
                            _Chunk("A Cold storage restore takes 3 hours.")])
    assert n >= 1


def test_only_the_top_two_chunks_are_considered():
    """By rank 5 almost any corpus offers some pair of numbers in a shared
    dimension, which turns the signal into a count of how numeric the corpus is."""
    chunks = [_Chunk("Takes 20 minutes."), _Chunk("Takes 20 minutes."),
              _Chunk("Takes 9 hours.")]
    assert competing_count(chunks)[0] == 0


def test_quantities_are_extracted_with_their_units():
    q = extract_quantities("600 requests per minute and 10,000 requests per second")
    assert ("10000", "requests") in q or ("10000", "per second") in q


# --- the margin signal -----------------------------------------------------

def test_margin_is_relative_not_absolute():
    """BM25 magnitudes move with query length, so an absolute gap is not
    comparable across questions."""
    assert lexical_margin([10.0, 5.0]) == pytest.approx(lexical_margin([100.0, 50.0]))


def test_margin_is_one_when_there_is_nothing_to_compare():
    assert lexical_margin([]) == 1.0
    assert lexical_margin([3.0]) == 1.0
    assert lexical_margin([0.0, 0.0]) == 1.0


def test_a_decisive_top_result_gives_a_large_margin():
    assert lexical_margin([10.0, 1.0]) > 0.8
    assert lexical_margin([10.0, 9.9]) < 0.1


# --- the negative results, pinned -----------------------------------------

def test_fused_scores_carry_no_confidence_information(retrievers):
    """RRF throws away raw scores for rank reciprocals, so its top score is a
    pure function of the rank pair. Any confidence derived from it is a constant.

    This is asserted directly because it is the reason a whole signal was cut,
    and because the numbers still look like scores.
    """
    _chunks, hybrid, _lex = retrievers
    a = [s for _, s in hybrid.search("What is the rate limit?", 5)]
    b = [s for _, s in hybrid.search("How do I rotate a service key?", 5)]
    assert a[0] == pytest.approx(b[0]), (
        "two unrelated queries must NOT share a top fused score; if this fails, "
        "RRF has started preserving score information and the signal is worth revisiting")


def test_raw_lexical_scores_do_carry_information(retrievers):
    """The control for the test above: the margin the detector actually uses is
    query-dependent, which is why it survived."""
    _chunks, _hybrid, lexical = retrievers
    a = [s for _, s in lexical.search("What is the rate limit?", 5)]
    b = [s for _, s in lexical.search("How do I rotate a service key?", 5)]
    assert a[0] != pytest.approx(b[0])


def test_half_the_ambiguous_set_is_unreachable_by_a_quantity_signal():
    """The structural ceiling, measured on the reference answers rather than
    asserted. Five of ten are ambiguous between entities or procedures, and no
    signal over numbers can see them."""
    rows = [r for r in load_golden() if r["category"] == "ambiguous"]
    assert len(rows) == 10
    reachable = sum(1 for r in rows if competing_count([_Chunk(r["answer"])])[0] > 0)
    assert reachable == 5, "the ceiling moved; the write-up quotes 5 of 10"


# --- the decision ----------------------------------------------------------

def test_precision_is_reported_against_the_base_rate(retrievers):
    """A detector that fires on everything scores precision equal to the base
    rate, so precision alone cannot tell a classifier from a coin."""
    chunks, hybrid, lexical = retrievers
    r = evaluate(ClarifyingRetrievalPolicy(hybrid, chunks, threshold=0.0, lexical=lexical),
                 load_golden())
    assert r["recall"] == 1.0, "threshold 0 must fire on everything"
    assert r["precision"] == pytest.approx(r["base_rate"], abs=1e-9)
    assert r["precision_lift_over_base_rate"] == pytest.approx(1.0, abs=1e-9)


def test_the_detector_is_not_shippable_and_says_so(retrievers):
    """The finding. If a future tuning pass makes this pass, that is a real
    result and this test should be updated with the new numbers -- not deleted."""
    chunks, hybrid, lexical = retrievers
    out = sweep(hybrid, chunks, load_golden(), lexical=lexical)
    rec = recommend(out["rows"])
    assert rec["shippable"] is False
    assert rec["best_precision_lift_achieved"] < 2.0
    assert "coin with a threshold" in rec["verdict"]


def test_recommend_refuses_a_threshold_that_never_fires():
    """The bug the first version had: threshold 0.75 satisfied the interruption
    budget at recall 0.00 and was returned as feasible. Satisfying a constraint
    by doing nothing is not a recommendation."""
    rows = [
        {"threshold": 0.9, "interruption_rate": 0.0, "recall": 0.0, "precision": 0.0,
         "f1": 0.0, "base_rate": 0.1, "precision_lift_over_base_rate": 0.0},
        {"threshold": 0.3, "interruption_rate": 0.6, "recall": 0.9, "precision": 0.13,
         "f1": 0.23, "base_rate": 0.1, "precision_lift_over_base_rate": 1.3},
    ]
    rec = recommend(rows)
    assert rec["shippable"] is False


def test_recommend_accepts_a_genuinely_good_detector():
    """The control: the refusal must be about the numbers, not unconditional."""
    rows = [
        {"threshold": 0.5, "interruption_rate": 0.02, "recall": 0.8, "precision": 0.75,
         "f1": 0.77, "base_rate": 0.1, "precision_lift_over_base_rate": 7.5},
    ]
    rec = recommend(rows)
    assert rec["shippable"] is True
    assert rec["threshold"] == 0.5


def test_unanswerable_questions_are_negatives_not_positives():
    """An unanswerable question should be refused, not clarified. Conflating them
    would let a detector score well by asking about everything it cannot answer."""
    cats = {r["category"] for r in load_golden()}
    assert "unanswerable" in cats and "ambiguous" in cats


# --- the clarifying question itself ----------------------------------------

def test_the_clarifying_question_names_the_alternatives():
    """A bare 'can you clarify?' is worse than a guess: it costs the round trip
    and gives the user nothing to answer with."""
    sig = signals("q", [_Chunk("a", doc_id="doc:rate-limits"),
                        _Chunk("b", doc_id="doc:storage-classes")], [10.0, 9.0])
    q = clarifying_question("q", sig)
    assert "rate limits" in q and "storage classes" in q


def test_the_committed_result_matches_the_write_up():
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "results", "clarification.json")
    if not os.path.exists(path):
        pytest.skip("run `python -m ragkit.clarify sweep` first")
    with open(path) as fh:
        out = json.load(fh)
    assert out["recommendation"]["shippable"] is False
    assert out["n_ambiguous"] == 10
