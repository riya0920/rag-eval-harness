"""Generation metrics, and a judge that is itself validated.

Three metrics:

* **Claim-level faithfulness** - is every claim in the answer supported by the
  retrieved context? Scored per claim, not per answer, because "mostly faithful"
  hides exactly the one fabricated sentence that matters.
* **Key-point coverage** - did the answer contain the facts the golden set says
  it must? An unfaithful answer and an incomplete answer fail differently and a
  single score conflates them.
* **Refusal accuracy** - on unanswerable questions, did it refuse?

**The judge is validated against human labels**, and this is the part the spec
cares about. A judge whose agreement with a human is unmeasured is a random
number generator with good manners. `validate_judge` reports Cohen's kappa
against `eval/human_labels.jsonl`, and the README reports where they disagree.

**No LLM is available in this environment**, so the judge here is a lexical
entailment heuristic rather than a model. That is a real limitation and it is
stated everywhere the numbers appear - but the *methodology* (hand-label a
sample, compute agreement, inspect the disagreements, report the bias) is the
transferable part, and it is identical whichever judge sits underneath.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from .contradiction import detect as detect_contradiction
from .nli import detect_composed
from .generation import split_sentences
from .retrieval import tokenize


@dataclass
class FaithfulnessResult:
    n_claims: int
    n_supported: int
    score: float
    unsupported_claims: list
    contradictions: list = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def extract_claims(answer_text: str) -> list:
    """One claim per sentence.

    Crude: a real claim extractor splits conjunctions and resolves pronouns. The
    consequence is that a sentence containing one true and one false statement is
    scored as a single claim, which makes this metric OPTIMISTIC. Stated here
    rather than discovered by a reader.
    """
    return [s for s in split_sentences(answer_text) if len(tokenize(s)) >= 3]


def claim_supported(claim: str, context_texts, threshold: float = 0.65) -> bool:
    """Is this claim's content present in the retrieved context?

    Token-overlap entailment: what fraction of the claim's content words appear
    in any single context chunk. Requiring one CHUNK to contain the claim rather
    than the union of all chunks is deliberate - a claim assembled from fragments
    of three different documents is exactly the kind of plausible fabrication
    this metric exists to catch.
    """
    claim_terms = set(tokenize(claim))
    if not claim_terms:
        return True
    best = 0.0
    for text in context_texts:
        ctx = set(tokenize(text))
        best = max(best, len(claim_terms & ctx) / len(claim_terms))
    return best >= threshold


def faithfulness(answer_text: str, context_texts, threshold: float = 0.65,
                 check_contradiction: bool = True, use_nli: bool = False) -> FaithfulnessResult:
    """A claim is faithful only if it is BOTH supported and non-contradictory.

    Support alone is not enough, and that gap is not hypothetical -- it is the
    documented failure the judge validation found. A claim assembled from the
    corpus's own vocabulary can score 1.0 on token overlap while asserting the
    opposite of the source, so `contradiction.detect` runs as a second, separate
    gate. Which gate rejected a claim is recorded, because "invented" and
    "reversed" are different bugs with different fixes.
    """
    claims = extract_claims(answer_text)
    if not claims:
        return FaithfulnessResult(0, 0, float("nan"), [])

    unsupported, contradictions = [], []
    for c in claims:
        if not claim_supported(c, context_texts, threshold):
            unsupported.append(c)
            continue
        if check_contradiction:
            # Rules first, NLI only for the residue they cannot express. See
            # nli.detect_composed for why the order is load-bearing.
            verdict = (detect_composed(c, context_texts, use_nli=True) if use_nli
                       else detect_contradiction(c, context_texts))
            if verdict.contradicts:
                unsupported.append(c)
                contradictions.append({"claim": c[:160], "rule": verdict.rule,
                                       "detail": verdict.detail})

    n_sup = len(claims) - len(unsupported)
    result = FaithfulnessResult(len(claims), n_sup, n_sup / len(claims), unsupported)
    result.contradictions = contradictions
    return result


def key_point_coverage(answer_text: str, key_points, threshold: float = 0.6) -> dict:
    """Fraction of the golden set's required facts present in the answer."""
    if not key_points:
        return {"n_points": 0, "covered": 0, "score": float("nan"), "missing": []}
    answer_terms = set(tokenize(answer_text))
    missing = []
    covered = 0
    for point in key_points:
        terms = set(tokenize(point))
        if not terms:
            continue
        if len(terms & answer_terms) / len(terms) >= threshold:
            covered += 1
        else:
            missing.append(point)
    n = len([p for p in key_points if tokenize(p)])
    return {"n_points": n, "covered": covered, "score": covered / n if n else float("nan"),
            "missing": missing}


def refusal_correct(refused: bool, category: str) -> bool:
    """Unanswerable questions must be refused; everything else must not be."""
    return refused if category == "unanswerable" else not refused


# ---------------------------------------------------------------------------
# judge validation
# ---------------------------------------------------------------------------

def cohens_kappa(a, b) -> float:
    """Agreement corrected for chance.

    Raw agreement is misleading when one label dominates: a judge that always
    says "supported" agrees with a human 90% of the time on a corpus that is 90%
    supported, while carrying no information at all. Kappa removes that floor.
    """
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def interpret_kappa(k: float) -> str:
    if math.isnan(k):
        return "undefined"
    if k < 0.0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def validate_judge(labels_path: str, retriever, generator, examples_by_id: dict,
                   fetch_k: int = 5, threshold: float = 0.65, use_nli: bool = False) -> dict:
    """Run the judge over hand-labelled examples and report agreement.

    `human_labels.jsonl` carries, per example, a human verdict on whether the
    generated answer was faithful. The judge is run on the same answers and the
    two are compared.
    """
    with open(labels_path, encoding="utf-8") as fh:
        labels = [json.loads(line) for line in fh if line.strip()]

    human, machine, disagreements = [], [], []
    for row in labels:
        ex = examples_by_id.get(row["id"])
        if ex is None:
            continue
        hits = retriever.search(ex["question"], fetch_k)
        chunks = [c for c in retriever_chunks(retriever, hits)]
        answer = generator.generate(ex["question"], chunks)

        # ADVERSARIAL labels carry a fabricated sentence to append. Without them
        # the label set has no unfaithful examples at all -- the extractive
        # generator copies verbatim, so it is faithful by construction -- and
        # kappa on a single-class set is meaningless. Injecting known
        # fabrications is what makes the agreement number informative.
        text = answer.text
        fabrication = row.get("inject_fabrication")
        if fabrication:
            text = (text + " " + fabrication).strip()

        result = faithfulness(text, [c.text for c in chunks], threshold, use_nli=use_nli)

        judged = "faithful" if (math_isnan_safe(result.score) or result.score >= 0.999) else "unfaithful"
        if answer.refused and not fabrication:
            judged = "refused"

        human.append(row["human_faithful"])
        machine.append(judged)
        if row["human_faithful"] != judged:
            disagreements.append({
                "id": row["id"],
                "question": ex["question"],
                "human": row["human_faithful"],
                "judge": judged,
                "answer": text[:200],
                "judge_score": result.score,
                "unsupported": result.unsupported_claims[:2],
                "human_note": row.get("note", ""),
            })

    k = cohens_kappa(human, machine)
    agreement = sum(1 for a, b in zip(human, machine) if a == b) / len(human) if human else float("nan")
    return {
        "n_labelled": len(human),
        "raw_agreement": round(agreement, 4),
        "cohens_kappa": round(k, 4) if not math.isnan(k) else None,
        "kappa_interpretation": interpret_kappa(k),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "label_distribution_human": {l: human.count(l) for l in set(human)},
        "label_distribution_judge": {l: machine.count(l) for l in set(machine)},
    }


def math_isnan_safe(x) -> bool:
    try:
        return math.isnan(x)
    except TypeError:
        return False


def retriever_chunks(retriever, hits):
    """Map (chunk_id, score) hits back to chunk objects."""
    by_id = {}
    for source in (getattr(retriever, "bm25", None), getattr(retriever, "dense", None)):
        if source is not None:
            for c in source.chunks:
                by_id[c.chunk_id] = c
    return [by_id[cid] for cid, _ in hits if cid in by_id]
