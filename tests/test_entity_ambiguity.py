"""Tests for the entity-ambiguity signals.

Two of these matter more than the rest: the one asserting the improvement over
the quantity-only detector is real, and the one asserting the module still says
out loud that its recall figure is circular. A later tuning pass that quietly
drops the caveat while keeping the number would be the bad outcome.
"""
import json
import os

import pytest

from ragkit import entity_ambiguity as ea
from ragkit.experiment import GATE_CONFIG, build, load_golden


class _Chunk:
    def __init__(self, text, doc_id="doc:x", chunk_id="c1"):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.ordinal = 0


# --- dangling referents ----------------------------------------------------

@pytest.mark.parametrize("q", [
    "Can I change it after creation?",
    "Is it included?",
    "How many can I have?",
])
def test_underspecified_questions_are_flagged(q):
    """The five the quantity signal cannot reach. The missing information is not
    in the retrieved text -- it was never sent."""
    assert ea.dangling_referent(q)["score"] > 0.5, q


@pytest.mark.parametrize("q", [
    "How do I rotate a service key for the data plane?",
    "What is the rate limit for the control plane per organisation?",
    "Which storage class should I use for archival backups?",
])
def test_well_specified_questions_are_not_flagged(q):
    """The negative direction, and the one that decides whether this is usable:
    a detector that fires on ordinary questions is worse than none."""
    assert ea.dangling_referent(q)["score"] < 0.7, q


def test_a_pronoun_with_an_antecedent_in_the_question_is_not_dangling():
    """'Can I change the retention policy after it is created?' supplies its own
    referent; 'Can I change it?' does not."""
    bare = ea.dangling_referent("Can I change it?")
    supplied = ea.dangling_referent(
        "Can I change the retention policy after it has been created for a project?")
    assert bare["dangling_pronoun"]
    assert not supplied["dangling_pronoun"]


def test_the_pronoun_rule_reads_the_raw_question_not_tokenised_output():
    """The first version's total failure: `tokenize` strips stopwords and pronouns
    are stopwords, so the word the rule exists to find was deleted before it
    looked. The rule could never fire."""
    from ragkit.retrieval import tokenize

    assert "it" not in tokenize("Can I change it after creation?")
    assert ea.dangling_referent("Can I change it after creation?")["dangling_pronoun"]


def test_a_qualified_definite_phrase_is_specific():
    r"""The greedy-match bug: `the\s+((?:[a-z]+\s+){0,2}[a-z]+)` swallows the
    qualifier out of 'the rate limit for the data plane', leaving a tail with no
    qualifier in it, and reports the most specific question in the set as
    underspecified."""
    assert not ea.dangling_referent(
        "What is the rate limit for the data plane?")["unqualified_definite"]
    assert ea.dangling_referent("What happens when I exceed the limit?")["unqualified_definite"]


def test_a_definite_phrase_that_is_a_prepositional_object_is_not_the_head():
    """'a service key for **the data plane**' qualifies something else. Treating
    it as the head flags every question that ends in a prepositional phrase."""
    assert not ea.dangling_referent(
        "How do I rotate a service key for the data plane?")["unqualified_definite"]


def test_first_and_second_person_pronouns_are_not_dangling():
    """'I' and 'you' refer to the asker and the system, which are always
    available. Treating them as missing antecedents would flag every question."""
    assert not ea.dangling_referent("How do I rotate a service key?")["dangling_pronoun"]


# --- competing definitions -------------------------------------------------

def test_competition_is_counted_per_document_not_per_mention():
    """The same term stated three times in one document is one answer; stated
    once in three documents is three answers."""
    one_doc = [_Chunk("retention period is 30 days", "doc:a"),
               _Chunk("retention period is 30 days", "doc:a"),
               _Chunk("retention period again", "doc:a")]
    three_docs = [_Chunk("retention period is 30 days", "doc:a"),
                  _Chunk("retention period is 13 months", "doc:b"),
                  _Chunk("retention period is 7 years", "doc:c")]
    q = "What is the retention period?"
    assert ea.competing_definitions(q, one_doc)["score"] == 0.0
    assert ea.competing_definitions(q, three_docs)["score"] > 0.0


def test_a_document_sharing_one_common_word_does_not_count_as_defining():
    """Otherwise every long question matches every document and the signal
    degenerates into a count of how many chunks were retrieved."""
    chunks = [_Chunk("retention period for backups", "doc:a"),
              _Chunk("the period between maintenance windows", "doc:b")]
    out = ea.competing_definitions("What is the backup retention period?", chunks)
    assert "doc:b" not in out["docs"]


def test_a_question_with_no_content_words_scores_zero():
    assert ea.competing_definitions("Is it?", [_Chunk("anything")])["score"] == 0.0


# --- combination -----------------------------------------------------------

def test_the_two_kinds_combine_by_max_not_by_sum():
    """They are alternative kinds of ambiguity, not two pieces of evidence for
    one. Summing lets two weak signals of different kinds add to a confident
    wrong answer."""
    chunks = [_Chunk("the limit is 600 per minute", "doc:a"),
              _Chunk("the limit is 10000 per second", "doc:b")]
    out = ea.combined_score("What happens when I exceed the limit?", chunks, [10.0, 9.5])
    assert out["score"] == max(out["quantity_score"], out["entity_score"])
    assert out["score"] <= 1.0


# --- the result ------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep_result():
    return ea.run()


def test_entity_signals_improve_the_precision_lift(sweep_result):
    """The headline: 1.87x -> 2.44x over the base rate, and recall at the peak
    from 0.20 to 0.90.

    Not the 3.79x an earlier version reported. That came from a BUG in the
    definite-phrase rule which made it fire on questions it should not, and on
    103 examples the broken feature correlated with ambiguity better than the
    correct one. Fixing it dropped the number, and the number stayed dropped."""
    assert sweep_result["combined_peak_lift"] > sweep_result["quantity_only_peak_lift"]
    assert sweep_result["combined_peak_lift"] < 3.0, (
        "a lift near 3.8x means the definite-phrase rule is over-firing again")


def test_it_is_still_not_shippable_under_the_stated_constraint(sweep_result):
    """Better is not the same as good. 12.9% interruption still fails a 5%
    budget, and the recommender says so rather than relaxing the constraint to
    fit the result."""
    assert sweep_result["recommendation"]["shippable"] is False


def test_the_module_states_its_own_circularity(sweep_result):
    """A guard against a later pass keeping the improved number and dropping the
    caveat that the features were designed with the positives visible."""
    assert "VISIBLE" in sweep_result["circularity"]
    assert "upper bound" in sweep_result["circularity"]
    assert "negative class" in sweep_result["circularity"].lower()
    assert "circular" in ea.__doc__


def test_the_negative_class_is_large_enough_for_the_precision_claim():
    """Recall is circular; the interruption rate is not, because the 93
    non-ambiguous questions were not consulted while designing the features. That
    asymmetry is only worth anything if the negative class is actually large."""
    rows = load_golden()
    negatives = [r for r in rows if r["category"] != "ambiguous"]
    assert len(negatives) >= 90
