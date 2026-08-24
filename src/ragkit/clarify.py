"""Asking instead of answering — and why that is a precision problem.

    python -m ragkit.clarify sweep
    python -m ragkit.clarify demo --question "What is the rate limit?"

## The failure this addresses

*"What is the rate limit?"* has two correct answers in this corpus — 600 requests
per minute on the control plane, 10,000 per second on the data plane. A system
that retrieves well and generates faithfully will confidently return **one of
them**, and it will be right about the sentence and wrong about the question. No
retrieval metric catches it: the correct chunk *was* retrieved. No faithfulness
metric catches it: the answer *is* grounded. It fails a check nothing upstream is
looking for.

The fix is to detect the condition and ask, which introduces the opposite risk.

## Why the threshold is the whole design

Clarification is not free. Every question routed to a clarifying question costs a
round trip, and a system that asks too often is worse than one that guesses —
users stop reading the question and pick the first option, which is a guess with
extra latency. So this is a **precision/recall trade with an asymmetric cost**,
and the operating point is a product decision rather than a modelling one:

  * **Recall** — of the genuinely ambiguous questions, how many did we catch?
  * **Precision** — of the questions we asked about, how many deserved it?

A detector at 100% recall that asks on a third of factual questions is not
usable. The sweep below reports both across the threshold range so the choice is
visible, and `recommend()` picks a point under a stated constraint ("no more than
5% of unambiguous questions may be interrupted") rather than optimising F1 and
calling it done — F1 weights the two errors equally, and here they are not.

## The signals — including the two that turned out to be worthless

The first version used three signals, argued for on the shape ambiguity takes in
retrieval: the top results are individually strong and collectively disagree. Two
of the three carry **no information at all** on this corpus, and finding out why
was the useful part.

**Document spread — dead.** The idea was that a question with one answer clusters
its top-k in one document. Measured across categories, mean spread is 0.95–1.00
for *every* category including plain factual questions. With 80-token chunks over
a many-document corpus, the top 5 essentially always come from 5 different
documents. The signal is a property of the chunk size, not of the question.

**Fused score margin — dead, and mechanically so.** The idea was that a
decisively-best chunk shows a large gap to rank 2. But the retriever fuses with
**RRF**, which throws away raw scores in favour of `1/(60 + rank)` sums. The top
fused score is `0.03279` for *"What is the rate limit?"* and `0.03279` for *"How
do I rotate a service key?"* — identical, because it is a pure function of the
rank pair and completely independent of the query. Every margin computed from
fused scores measures the lattice, not the retrieval. Mean flatness came out
0.98–0.99 for all four categories, exactly as that predicts.

That one is worth stating generally: **any confidence signal derived from RRF
scores is measuring a constant.** RRF is a rank-fusion method and it is not
supposed to preserve score information; using its output as a confidence is a
category error, and an easy one to make because the numbers look like scores.

**What survives**, and the pair only works together:

  * **A flat margin on the raw lexical arm.** BM25 scores are query-dependent, so
    the rank-1-to-rank-2 margin means something there. Measured: 0.369 on
    ambiguous questions against 0.525 on factual — flatter, in the right
    direction.
  * **Competing quantities in the top 2.** Values in the same *dimension* that
    differ. "600 per minute" against "10,000 per second" is the signature; the
    same figure restated is not.

    Grouping on the literal unit string was a bug worth its own note. "20
    minutes" and "3 hours" are competing answers to *"how long does a restore
    take"*, and comparing only within an identical unit never puts them side by
    side. Normalising into canonical dimensions — durations to seconds, sizes to
    bytes, rates to per-second — took the number of ambiguous questions the
    signal can even reach from **2 of 10 to 5 of 10**, and peak precision lift
    from 1.29x to 1.87x. Values within 10% of each other are then treated as one
    figure, because "about 20 minutes" and "20 minutes" are one fact stated twice.

The pair matters because the flat margin **cannot tell ambiguous from
unanswerable** — unanswerable questions are the flattest of all, at 0.354, since
nothing matches well. What separates them is that an unanswerable question has no
competing specifics to disagree about (1.00 versus 1.40). Retrieval uncertainty
says "something is wrong"; competing quantities says *which* thing.

## The verdict: not shippable, and the reason is structural

Peak precision **0.18 against a base rate of 0.097** — a lift of 1.87x, and
reaching it costs interrupting 10% of unambiguous questions. That is a coin with
a threshold, and `recommend()` returns `shippable: false` rather than picking the
least-bad row.

The ceiling is not tuning. Of the 10 ambiguous questions, **5 are ambiguous
between numbers and 5 are not** — *"Can I change it after creation?"*, *"Is it
included?"*, *"How many can I have?"* are ambiguous between entities and
procedures, and no quantity signal reaches them at all. Half the positive class
is invisible by construction.

The base-rate comparison is the part that is usually missing. A detector that
fires on everything scores precision equal to the base rate, so a reported 0.18
*sounds* like a working classifier while being 1.9x better than a coin. Every row
of the sweep carries `precision_lift_over_base_rate` for that reason.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field

from .retrieval import tokenize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "eval", "results")

# Numbers with their units, which is what makes two of them *competing* rather
# than merely different. "20 minutes" and "3 hours" answer the same question
# differently; "20" and "3" on their own might be a version and a count.
QUANTITY = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?)\s*"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|"
    r"ms|gb|mb|tb|kb|%|percent|requests?|rps|qps|per\s+\w+)\b", re.I)


@dataclass
class AmbiguitySignals:
    lexical_flatness: float    # 1 - (raw BM25 margin between rank 1 and 2)
    competing_quantities: int  # distinct values sharing a unit, across the top 2
    score: float = 0.0
    top_docs: list = field(default_factory=list)
    quantities: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["quantities"] = [list(q) for q in self.quantities]
        return d


# Canonical dimensions. Two quantities only compete if they are the same KIND of
# thing, and "same unit string" is the wrong test for that: "20 minutes" and
# "3 hours" are competing answers to "how long does a restore take", and grouping
# on the literal unit never compares them. Measured on the golden set, that bug
# alone hid 3 of the 10 ambiguous questions.
_TIME = {"second": 1.0, "sec": 1.0, "ms": 0.001, "minute": 60.0, "min": 60.0,
         "hour": 3600.0, "hr": 3600.0, "day": 86400.0, "week": 604800.0,
         "month": 2592000.0}
_SIZE = {"kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12}


def canonical(value: str, unit: str):
    """Map a (value, unit) pair onto (dimension, magnitude in a base unit).

    Returns None for units with no shared dimension -- percentages, request
    counts, bare rates. Those still compare within their own unit, they just
    cannot be converted into anything else.
    """
    u = unit.lower().strip().rstrip("s")
    try:
        v = float(value.replace(",", ""))
    except ValueError:
        return None
    if u in _TIME:
        return ("duration", v * _TIME[u])
    if u in _SIZE:
        return ("size", v * _SIZE[u])
    if u.startswith("per "):
        # "per minute" and "per second" are rate denominators, and a rate is only
        # comparable to another rate over the same denominator once the numerator
        # is normalised too -- which needs the numerator, which is the value here.
        # Normalise to per-second so 600/min and 10,000/s are comparable.
        denom = u[4:].rstrip("s")
        if denom in _TIME and _TIME[denom]:
            return ("rate_per_second", v / _TIME[denom])
        return ("rate:" + denom, v)
    return ("unit:" + u, v)


def extract_quantities(text: str) -> set:
    """Distinct (value, unit) pairs as written, for reporting."""
    out = set()
    for value, unit in QUANTITY.findall(text):
        out.add((value.replace(",", ""), unit.lower().strip()))
    return out


def competing_count(chunks) -> tuple:
    """Values in the same DIMENSION that differ. Only those compete.

    Restricted to the top 2 rather than the top 5: by rank 5 almost any corpus
    offers some pair of numbers in a shared dimension, so a wider window turns
    the signal into a measure of how many numbers the corpus contains.

    Values within 10% of each other are treated as the same figure. "about 20
    minutes" and "20 minutes" are one fact stated twice, and counting them as a
    conflict is the false positive that makes this signal unusable.
    """
    quantities = set()
    for c in chunks[:2]:
        quantities |= extract_quantities(c.text)

    by_dim = {}
    for value, unit in quantities:
        c = canonical(value, unit)
        if c is None:
            continue
        dim, magnitude = c
        by_dim.setdefault(dim, []).append(magnitude)

    competing = 0
    for magnitudes in by_dim.values():
        distinct = []
        for m in sorted(magnitudes):
            if not distinct or m > distinct[-1] * 1.1:
                distinct.append(m)
        competing += max(len(distinct) - 1, 0)
    return competing, sorted(quantities)


def lexical_margin(scores) -> float:
    """Relative rank-1-to-rank-2 margin on RAW scores.

    Must not be fed fused scores -- see the module docstring. Relative rather
    than absolute because BM25 magnitudes move with query length, so an absolute
    gap is not comparable across questions.
    """
    if not scores or len(scores) < 2 or scores[0] <= 0:
        return 1.0
    return max(min((scores[0] - scores[1]) / scores[0], 1.0), 0.0)


def signals(question: str, chunks, lexical_scores=None, k: int = 5) -> AmbiguitySignals:
    top = list(chunks)[:k]
    if not top:
        return AmbiguitySignals(0.0, 0, 0.0)
    competing, quantities = competing_count(top)
    return AmbiguitySignals(
        lexical_flatness=1.0 - lexical_margin(lexical_scores),
        competing_quantities=competing,
        top_docs=sorted({c.doc_id for c in top}),
        quantities=quantities,
    )


# Two signals, and the conjunction is the design. Either alone misfires: a flat
# margin fires hardest on unanswerable questions, and competing quantities fire
# on any passage that lists two figures. Multiplying them means both have to be
# present, which is what "the candidates are individually plausible AND disagree
# on a value" actually says.
def ambiguity_score(sig: AmbiguitySignals) -> float:
    competing = min(sig.competing_quantities / 2.0, 1.0)
    return sig.lexical_flatness * competing


def clarifying_question(question: str, sig: AmbiguitySignals) -> str:
    """Name the alternatives. A bare "can you clarify?" is a worse answer than a
    guess: it costs the round trip and gives the user nothing to answer with."""
    if len(sig.top_docs) > 1:
        readings = ", ".join(d.replace("doc:", "").replace("-", " ") for d in sig.top_docs[:3])
        return ("That could mean a few different things here — %s. Which did you mean?"
                % readings)
    if sig.quantities:
        vals = ", ".join("%s %s" % (v, u) for v, u in sig.quantities[:3])
        return ("There is more than one applicable figure (%s). Which case are you asking about?"
                % vals)
    return "Could you narrow that down? Several parts of the documentation apply."


class ClarifyingRetrievalPolicy:
    """Decide: answer, or ask. One threshold, and it is a product input."""

    def __init__(self, retriever, chunks, threshold: float = 0.45, k: int = 5,
                 lexical=None):
        self.retriever = retriever
        self.by_id = {c.chunk_id: c for c in chunks}
        self.threshold = threshold
        self.k = k
        # The RAW lexical arm, kept separately. Passing the fused retriever here
        # would silently reintroduce the RRF bug -- the scores would still be
        # numbers, they would just stop meaning anything.
        self.lexical = lexical

    def decide(self, question: str) -> dict:
        hits = self.retriever.search(question, self.k)
        cand = [self.by_id[cid] for cid, _ in hits if cid in self.by_id]
        lex_scores = None
        if self.lexical is not None:
            lex_scores = [sc for _, sc in self.lexical.search(question, self.k)]
        sig = signals(question, cand, lex_scores, k=self.k)
        sig.score = ambiguity_score(sig)
        ask = sig.score >= self.threshold
        return {
            "question": question,
            "ask_for_clarification": ask,
            "score": sig.score,
            "threshold": self.threshold,
            "signals": sig.as_dict(),
            "clarifying_question": clarifying_question(question, sig) if ask else None,
        }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate(policy: ClarifyingRetrievalPolicy, examples) -> dict:
    """Score against the golden set's own categories.

    `ambiguous` is the positive class. Everything else is negative — including
    `unanswerable`, which is a *different* failure with a different response: an
    unanswerable question should be refused, not clarified, and conflating them
    would let a detector score well by asking about everything it cannot answer.
    """
    tp = fp = tn = fn = 0
    asked_on = {}
    per_example = []
    for ex in examples:
        out = policy.decide(ex["question"])
        truth = ex["category"] == "ambiguous"
        pred = out["ask_for_clarification"]
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
            asked_on[ex["category"]] = asked_on.get(ex["category"], 0) + 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
        per_example.append({"id": ex["id"], "category": ex["category"],
                            "score": round(out["score"], 4), "asked": pred})

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_neg = tn + fp
    # The base rate is the number precision has to be compared against, and it is
    # the comparison that is usually missing. Ambiguous questions are 10 of 103
    # here, so a detector that fires on EVERYTHING scores precision 0.097 -- and
    # a reported precision of 0.12 sounds like a working classifier while being
    # a 1.2x lift over answering at random.
    base_rate = (tp + fn) / (tp + fn + tn + fp) if (tp + fn + tn + fp) else 0.0
    return {
        "threshold": policy.threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "base_rate": base_rate,
        "precision_lift_over_base_rate": precision / base_rate if base_rate else 0.0,
        # The number the product actually cares about: how often an unambiguous
        # question gets interrupted.
        "interruption_rate": fp / n_neg if n_neg else 0.0,
        "false_positives_by_category": asked_on,
        "per_example": per_example,
    }


def sweep(retriever, chunks, examples, thresholds=None, lexical=None) -> dict:
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(0, 17)]
    rows = []
    for t in thresholds:
        r = evaluate(ClarifyingRetrievalPolicy(retriever, chunks, threshold=t,
                                               lexical=lexical), examples)
        r.pop("per_example", None)
        rows.append(r)
        print("  threshold %.2f  recall %.2f  precision %.2f  interrupts %.1f%% of unambiguous"
              % (t, r["recall"], r["precision"], 100 * r["interruption_rate"]))
    return {"rows": rows}


MAX_INTERRUPTION = 0.05


def recommend(rows, max_interruption: float = MAX_INTERRUPTION,
              min_lift: float = 1.5) -> dict:
    """Pick an operating point under stated constraints, or refuse.

    F1 weights a missed ambiguity and an unnecessary interruption equally. They
    are not equal: a missed ambiguity produces one confidently wrong answer, and
    an over-eager detector trains every user to click through the clarifying
    question without reading it -- which destroys the feature for the cases it
    exists to catch.

    Two constraints, and the second one is the one that matters here:

      * interruption rate within budget
      * **precision meaningfully above the base rate.** A detector that fires on
        everything achieves precision equal to the base rate by construction, so
        precision alone cannot tell a classifier from a coin. The first version
        of this function checked only the interruption budget and duly returned
        threshold 0.75 as "feasible" -- at recall 0.00, a detector that never
        fires and therefore never interrupts anyone. Satisfying a constraint by
        doing nothing is not a recommendation.
    """
    best_f1 = max(rows, key=lambda r: r["f1"])
    feasible = [r for r in rows
                if r["interruption_rate"] <= max_interruption
                and r["recall"] > 0
                and r["precision_lift_over_base_rate"] >= min_lift]
    peak_lift = max(rows, key=lambda r: r["precision_lift_over_base_rate"])

    if not feasible:
        return {
            "shippable": False,
            "constraints": ("interruption <= %.0f%%, recall > 0, precision >= %.1fx base rate"
                            % (100 * max_interruption, min_lift)),
            "base_rate": rows[0]["base_rate"] if rows else None,
            "best_precision_lift_achieved": peak_lift["precision_lift_over_base_rate"],
            "at_threshold": peak_lift["threshold"],
            "with_recall": peak_lift["recall"],
            "with_interruption_rate": peak_lift["interruption_rate"],
            "f1_optimum_would_pick": best_f1["threshold"],
            "verdict": (
                "NOT SHIPPABLE. Peak precision is %.2f against a base rate of %.2f -- a lift of "
                "%.2fx, and reaching even that costs interrupting %.0f%% of unambiguous "
                "questions. A detector this close to the base rate is a coin with a threshold."
                % (peak_lift["precision"], peak_lift["base_rate"],
                   peak_lift["precision_lift_over_base_rate"],
                   100 * peak_lift["interruption_rate"])),
            "what_would_change_it": (
                "the surviving signal fires only when the ambiguity is between two NUMBERS. "
                "Measured on the reference answers, 5 of the 10 ambiguous questions are "
                "number-ambiguities and 5 are not -- 'Can I change it after creation?', 'Is it "
                "included?', 'How many can I have?' are ambiguous between entities and "
                "procedures, and no quantity signal can reach them. So half the positive class "
                "is invisible by construction and recall is capped there. Closing it needs a "
                "signal over entities and section topics, and a positive class larger than 10 to "
                "measure it with -- at n=10, recall moves in steps of 0.1."),
        }

    best = max(feasible, key=lambda r: (r["recall"], r["precision"]))
    return {
        "shippable": True,
        "constraints": ("interruption <= %.0f%%, recall > 0, precision >= %.1fx base rate"
                        % (100 * max_interruption, min_lift)),
        "threshold": best["threshold"],
        "recall": best["recall"],
        "precision": best["precision"],
        "precision_lift_over_base_rate": best["precision_lift_over_base_rate"],
        "interruption_rate": best["interruption_rate"],
        "f1_optimum_would_pick": best_f1["threshold"],
        "differs_from_f1_optimum": best["threshold"] != best_f1["threshold"],
    }


def main() -> int:
    from .experiment import GATE_CONFIG, build, load_golden

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sweep", "demo"])
    ap.add_argument("--question", default="What is the rate limit?")
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    chunks, retriever = build(GATE_CONFIG["mode"], GATE_CONFIG["target_tokens"])
    # Built separately and deliberately: the confidence signal needs raw lexical
    # scores, and the fused retriever cannot supply them.
    _c, lexical = build("bm25", GATE_CONFIG["target_tokens"])

    if args.command == "demo":
        policy = ClarifyingRetrievalPolicy(retriever, chunks, threshold=args.threshold,
                                           lexical=lexical)
        print(json.dumps(policy.decide(args.question), indent=2))
        return 0

    examples = load_golden()
    out = sweep(retriever, chunks, examples, lexical=lexical)
    out["recommendation"] = recommend(out["rows"])
    out["n_ambiguous"] = sum(1 for e in examples if e["category"] == "ambiguous")
    out["n_examples"] = len(examples)
    out["caveat"] = ("10 ambiguous examples is a small positive class: recall moves in steps of "
                     "0.1 and a single example changes the recommended threshold. The shape of "
                     "the trade is the result; the exact operating point is not.")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "clarification.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print()
    print(json.dumps(out["recommendation"], indent=2))
    print("\nwritten:", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
