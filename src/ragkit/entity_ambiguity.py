"""The other half of ambiguity: questions that underspecify their own subject.

    python -m ragkit.entity_ambiguity sweep

`clarify.py` detects ambiguity between two **numbers** and reaches 5 of the 10
ambiguous questions. This module goes after the other 5, and they turn out to
share a shape the quantity signal cannot see because **it is not in the retrieved
text at all - it is in the question**:

    a005  "Can I change it after creation?"          <- "it" has no antecedent
    a008  "How many can I have?"                     <- how many *what*
    a009  "Is it included?"                          <- "it" again
    a010  "What happens when I exceed the limit?"    <- *which* limit
    a006  "How long is the minimum retention?"       <- minimum retention of what

Every one is a **dangling referent**: a pronoun or a bare quantifier whose
antecedent the asker had in mind and did not supply. No amount of looking at the
retrieved chunks recovers it, because the missing information was never sent.

## Two signals, and they are of different kinds

**Dangling referent (question-side).** A pronoun or quantifier with no noun
phrase anywhere in the question to bind it. This costs two regexes over the
question and no retrieval at all, which makes it the cheapest possible check - it can run *before* the retriever does, and on a question it flags there is no
point retrieving at all.

**Competing definitions (retrieval-side).** The entity analogue of competing
quantities: the question's head noun is defined in **several different documents**
in the top-k. "retention period" appearing in `observability-logs`, `audit-log`
*and* `backup-policy` is three different answers to one question; the same noun
appearing three times inside `backup-policy` is one answer stated thrice. The unit
of competition is the **document**, not the mention.

## Read this before believing the numbers

These features were designed **with the ten ambiguous questions visible**. That
makes the recall figure an upper bound, not an estimate of how it would do on
questions it has not seen, and no amount of cross-validation fixes it - the
leakage is in the feature design, not the threshold.

What is *not* circular is the negative class. The 93 factual, multi-hop and
unanswerable questions were not consulted while designing anything here, so the
**false-positive rate is an honest measurement** even though the recall is not.
That asymmetry is the most useful thing this module can report, and it is why the
headline below is precision rather than recall.

The clean fix is a larger positive class written by someone who has not seen the
detector. Authoring 40 more myself, from my own taxonomy of what "ambiguous"
means, would make the evaluation *more* circular while making the number look
better - which is the trade this module declines to take.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from .clarify import (
    ClarifyingRetrievalPolicy,
    competing_count,
    lexical_margin,
    recommend,
)
from .retrieval import tokenize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "eval", "results")

# Pronouns that need an antecedent. "you"/"I" are excluded: they refer to the
# asker and the system, which are always available, so they are never dangling.
PRONOUNS = {"it", "its", "they", "them", "their", "this", "that", "those", "these", "one"}

# Nouns whose referent is a *slot* rather than a thing -- "how many" is a
# question about a quantity of something unnamed.
BARE_QUANTIFIERS = {"many", "much", "long", "often"}

# A determiner followed by a word: the cheapest usable stand-in for "this
# question contains a noun phrase". Word-anchored on both sides, because without
# `\b` the "a" alternative matches inside other words and the pattern reports a
# noun phrase in almost any sentence.
NOUN_PHRASE = re.compile(r"\b(?:the|a|an|each|every|per|my|your)\s+[a-z]+\b", re.I)

# Words that QUALIFY a noun phrase -- "the rate limit **for** the data plane".
QUALIFIER = {"for", "of", "on", "in", "per", "when", "during", "with", "after", "before"}

# "the" plus up to three words. The body is truncated at a qualifier below,
# because a greedy match on "the rate limit for the data plane" swallows "for"
# into the phrase, leaves a tail containing no qualifier, and reports the most
# specific question in the set as underspecified.
DEFINITE = re.compile(r"\bthe\s+([a-z]+(?:\s+[a-z]+){0,2})\b", re.I)


def unqualified_definite_phrase(question: str) -> bool:
    """Is the question's FIRST definite phrase left without a qualifier?

    "the limit" with nothing after it is underspecified; "the rate limit for the
    data plane" is not. Only the first is considered -- it is the head of the
    question, and checking the last instead flags every question that happens to
    end in a prepositional phrase.
    """
    # Skip definite phrases that are themselves the OBJECT of a preposition --
    # "a service key for **the data plane**". Those qualify something else; they
    # are not the head the question is asking about, and treating them as one
    # flags every question ending in a prepositional phrase.
    m = None
    for cand in DEFINITE.finditer(question):
        before = question[:cand.start()].lower().split()
        if before and before[-1] in QUALIFIER:
            continue
        m = cand
        break
    if not m:
        return False
    phrase = m.group(1).lower().split()
    cut = len(phrase)
    for i, w in enumerate(phrase):
        if w in QUALIFIER:
            cut = i
            break
    consumed = m.start() + len("the ") + len(" ".join(phrase[:cut]))
    tail = question[consumed:].lower()
    return not any(re.search(r"\b%s\b" % w, tail) for w in QUALIFIER)



def dangling_referent(question: str) -> dict:
    """Does the question fail to name what it is asking about?

    Works on the RAW question, not on `tokenize()` output. That was the first
    version's bug and it was total: `tokenize` strips stopwords, and pronouns are
    stopwords, so "Can I change it after creation?" tokenises to
    `['can','i','change','after','creation']` -- the word the detector exists to
    find is removed before it ever looks. The pronoun rule could never fire.

    Two ways a question can withhold its subject:

      * **A pronoun with no noun phrase anywhere in the question.** "Can I change
        it?" supplies nothing for "it" to refer to. "Can I change the retention
        policy after it is created?" supplies "the retention policy", so the
        pronoun is bound and the question is specific. Approximating "noun
        phrase" as a determiner followed by a word is crude and is right on every
        question in this corpus.
      * **A quantifier with no noun.** "How many can I have?" -- many *what*.

      * **An unqualified definite phrase.** "the limit" rather than "the rate
        limit for the data plane". Weighted lower (0.6) because it is the least
        certain of the three.

    That third rule has a history worth keeping. It was cut on the argument that
    surface form cannot separate a vague definite ("the limit") from a specific
    one ("the data plane"), then restored when the golden set showed peak
    precision lift falling from **3.79x to 2.44x** without it.

    Then the bug it was cut for got fixed -- a greedy match swallowed the
    qualifier, so "the rate limit for the data plane" was scored as
    underspecified -- and the lift came back at **2.44x**. Exactly the same as
    having no rule at all.

    So the 3.79x was the bug. A broken feature that fires on questions it should
    not was, on 103 examples, *better correlated with ambiguity* than the correct
    version of the same feature. That is what overfitting looks like from the
    inside, and the number was large enough to be worth keeping if nobody had
    checked why.

    The rule stays, fixed, because it is principled and fires on genuinely
    underspecified heads. It is documented as contributing **no measurable lift**
    on this sample, which is the honest description of a feature kept on
    reasoning rather than evidence.
    """
    q = question.lower()
    words = re.findall(r"[a-z']+", q)
    wordset = set(words)

    # A determiner followed by a word: the closest cheap thing to "a noun phrase
    # is present".
    has_noun_phrase = bool(NOUN_PHRASE.search(q))

    pronouns = wordset & PRONOUNS
    dangling_pronoun = bool(pronouns) and not has_noun_phrase

    quantifiers = wordset & BARE_QUANTIFIERS
    bare_quantifier = bool(quantifiers) and not has_noun_phrase
    unqualified_definite = unqualified_definite_phrase(question)

    score = 0.0
    reasons = []
    if dangling_pronoun:
        score = max(score, 1.0)
        reasons.append("pronoun %s with nothing in the question to bind it" % sorted(pronouns))
    if bare_quantifier:
        score = max(score, 1.0)
        reasons.append("quantifier %s with no noun attached" % sorted(quantifiers))
    if unqualified_definite:
        score = max(score, 0.6)
        reasons.append("definite noun phrase left unqualified")
    return {"score": score, "reasons": reasons,
            "dangling_pronoun": dangling_pronoun,
            "bare_quantifier": bare_quantifier,
            "unqualified_definite": unqualified_definite,
            "has_noun_phrase": has_noun_phrase}


STOP = {"what", "which", "how", "does", "do", "can", "is", "are", "the", "a", "an",
        "my", "i", "you", "when", "there", "get", "have", "much", "many", "long",
        "before", "after", "with", "for", "of", "and", "to", "in", "on", "it", "its"}


def head_terms(question: str) -> list:
    """The content words a definition would be attached to."""
    return [t for t in tokenize(question) if t not in STOP and len(t) > 3]


def competing_definitions(question: str, chunks) -> dict:
    """How many DIFFERENT documents define the question's head terms.

    Counted per document, not per mention. "retention period" appearing three
    times inside `backup-policy` is one answer stated thrice; appearing once each
    in `observability-logs`, `audit-log` and `backup-policy` is three different
    answers to one question, which is the thing being detected.
    """
    terms = head_terms(question)
    if not terms:
        return {"score": 0.0, "docs": [], "terms": []}

    by_doc = {}
    for c in chunks:
        hits = sum(1 for t in terms if t in c.text.lower())
        if hits:
            by_doc[c.doc_id] = max(by_doc.get(c.doc_id, 0), hits)

    # Only documents that match a MAJORITY of the head terms count as defining
    # the same thing. A doc sharing one common word is a coincidence.
    need = max(1, (len(terms) + 1) // 2)
    defining = sorted(d for d, h in by_doc.items() if h >= need)
    return {"score": min(max(len(defining) - 1, 0) / 2.0, 1.0),
            "docs": defining, "terms": terms}


def combined_score(question: str, chunks, lexical_scores=None) -> dict:
    """Quantity ambiguity OR entity ambiguity. Deliberately a max, not a sum.

    They are alternative *kinds* of ambiguity rather than two pieces of evidence
    for the same one: a question ambiguous between two numbers is not made more
    ambiguous by also having a dangling pronoun. Summing would let two weak
    signals of different kinds add up to a confident wrong answer.
    """
    flat = 1.0 - lexical_margin(lexical_scores)
    competing, _q = competing_count(list(chunks)[:5])
    quantity = flat * min(competing / 2.0, 1.0)

    dangle = dangling_referent(question)
    defs = competing_definitions(question, list(chunks)[:5])
    entity = max(dangle["score"] * 0.8, defs["score"])

    return {
        "score": max(quantity, entity),
        "quantity_score": quantity,
        "entity_score": entity,
        "dangling": dangle,
        "definitions": defs,
    }


class CombinedPolicy(ClarifyingRetrievalPolicy):
    """`ClarifyingRetrievalPolicy` with the entity signals added."""

    def decide(self, question: str) -> dict:
        hits = self.retriever.search(question, self.k)
        cand = [self.by_id[cid] for cid, _ in hits if cid in self.by_id]
        lex = None
        if self.lexical is not None:
            lex = [sc for _, sc in self.lexical.search(question, self.k)]
        sig = combined_score(question, cand, lex)
        ask = sig["score"] >= self.threshold
        return {"question": question, "ask_for_clarification": ask,
                "score": sig["score"], "threshold": self.threshold, "signals": sig,
                "clarifying_question": _ask(question, sig) if ask else None}


def _ask(question: str, sig: dict) -> str:
    d = sig["dangling"]
    if d["dangling_pronoun"] or d["bare_quantifier"]:
        return ("I want to make sure I answer the right thing - what specifically "
                "are you asking about?")
    docs = sig["definitions"]["docs"]
    if len(docs) > 1:
        names = ", ".join(x.replace("doc:", "").replace("-", " ") for x in docs[:3])
        return "That is defined differently in %s. Which did you mean?" % names
    return "Could you narrow that down? Several parts of the documentation apply."


def run(thresholds=None) -> dict:
    from .clarify import evaluate, sweep
    from .experiment import GATE_CONFIG, build, load_golden

    chunks, retriever = build(GATE_CONFIG["mode"], GATE_CONFIG["target_tokens"])
    _c, lexical = build("bm25", GATE_CONFIG["target_tokens"])
    examples = load_golden()

    thresholds = thresholds or [round(0.05 * i, 2) for i in range(0, 17)]
    rows = []
    for t in thresholds:
        r = evaluate(CombinedPolicy(retriever, chunks, threshold=t, lexical=lexical), examples)
        r.pop("per_example", None)
        rows.append(r)
        print("  threshold %.2f  recall %.2f  precision %.2f  lift %.2fx  interrupts %.1f%%"
              % (t, r["recall"], r["precision"], r["precision_lift_over_base_rate"],
                 100 * r["interruption_rate"]))

    # Silenced: `sweep` prints its own rows, and letting it do so here
    # interleaves the quantity-only baseline with the combined sweep above --
    # which is how the first run appeared to report a threshold whose printed
    # row said the opposite.
    import contextlib, io

    with contextlib.redirect_stdout(io.StringIO()):
        quantity_only = sweep(retriever, chunks, examples, lexical=lexical)

    def best(rs):
        return max(rs, key=lambda r: r["precision_lift_over_base_rate"])

    b_new, b_old = best(rows), best(quantity_only["rows"])
    return {
        "rows": rows,
        "recommendation": {**recommend(rows),
                           "what_would_change_it": (
                               "Entity signals lifted peak precision from 1.87x to 2.44x over "
                               "the base rate and recall at that point from 0.20 to 0.90, so the "
                               "ceiling is no longer blindness to non-numeric ambiguity. What "
                               "caps it now is the interruption rate: 31% of unambiguous "
                               "questions get interrupted to reach that recall, against a 5% "
                               "budget. And n=10 means recall moves in steps of 0.1 while the "
                               "features were designed with those 10 visible. A larger positive "
                               "class written by someone who has not seen the detector is the "
                               "only thing that moves this further.")},
        "quantity_only_peak_lift": b_old["precision_lift_over_base_rate"],
        "combined_peak_lift": b_new["precision_lift_over_base_rate"],
        "quantity_only_peak_recall": max(r["recall"] for r in quantity_only["rows"]),
        "combined_peak_recall": max(r["recall"] for r in rows),
        "circularity": (
            "The entity features were designed with the ten ambiguous questions VISIBLE, so "
            "recall here is an upper bound rather than an estimate of generalisation, and no "
            "cross-validation repairs that -- the leakage is in the feature design. The negative "
            "class was NOT consulted, so the interruption rate and therefore the precision are "
            "honest. That asymmetry is why the headline is precision."),
        "why_not_more_examples": (
            "Authoring 40 more ambiguous questions from the same taxonomy that produced these "
            "features would make the evaluation more circular while making the number look "
            "better. The clean fix is a positive class written by someone who has not seen the "
            "detector."),
    }


def main() -> int:
    out = run()
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "entity_ambiguity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print()
    print("quantity only : recall %.2f, peak lift %.2fx"
          % (out["quantity_only_peak_recall"], out["quantity_only_peak_lift"]))
    print("with entities : recall %.2f, peak lift %.2fx"
          % (out["combined_peak_recall"], out["combined_peak_lift"]))
    print()
    print(json.dumps(out["recommendation"], indent=2))
    print("\nwritten:", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
