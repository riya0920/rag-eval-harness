"""Contradiction detection — the blind spot the judge validation exposed.

The lexical judge scores a claim by how many of its words appear in the context.
That catches **invention** (new entities, new numbers) and completely misses
**contradiction**: a sentence built from the corpus's own vocabulary that asserts
the opposite of what the corpus says. `f041` in the human labels is the worked
example — *"Credits below 99.0% uptime are paid out in cash within 14 days"*
against a corpus that says credits are **never paid in cash**. Every content word
is present, so token overlap is 1.0.

This module adds the missing half. It is still not an NLI model — no LLM is
available here — but it targets the three contradiction shapes that actually
occur in factual documentation, which is a far better use of a heuristic than a
generic similarity score:

  1. **Polarity flip** — the context negates something the claim asserts, or vice
     versa ("never paid in cash" vs "paid out in cash").
  2. **Numeric mismatch** — the claim reuses an entity from the context but
     attaches a different number to it ("400 days" vs "40 days").
  3. **Antonym substitution** — a claim that swaps one member of a known opposed
     pair while keeping the surrounding words ("enabled" for "disabled",
     "supported" for "unsupported").

**Why this is honest rather than a hack:** each rule is a specific, named failure
mode with a test, and the module reports WHICH rule fired. A claim it cannot
adjudicate is returned as `unknown`, not as `supported`, so the judge degrades to
"I don't know" rather than to a false pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import tokenize

# Negation markers. Deliberately includes "never" and "cannot", which the
# retrieval stopword list keeps for exactly this reason -- this corpus is full of
# negative facts and dropping them would invert meaning.
NEGATIONS = {
    "no", "not", "never", "cannot", "cant", "without", "none", "neither", "nor",
    "unsupported", "disabled", "excluded", "prohibited", "unavailable",
}

# Pairs that mean opposite things in this domain. Each entry is checked both ways.
#
# Inflections are listed EXPLICITLY because there is no stemmer here: "enables"
# and "enabled" are separate entries. That is a real limitation -- an unlisted
# inflection is a silent miss -- and it is the reason this list is domain-scoped
# rather than pretending to be general. A stemmer or an NLI model removes the
# need for the list entirely, and that is the roadmap item.
ANTONYMS = [
    ("enabled", "disabled"), ("enables", "disables"), ("enable", "disable"),
    ("supported", "unsupported"), ("supports", "does-not-support"),
    ("included", "excluded"), ("includes", "excludes"),
    ("available", "unavailable"),
    ("allowed", "prohibited"),
    ("mutable", "immutable"),
    ("reversible", "irreversible"),
    ("increases", "decreases"),
    ("same", "different"),
    ("before", "after"),
    ("more", "less"),
    ("raised", "lowered"),
    ("continues", "stops"), ("continue", "stop"),
]

NUMBER = re.compile(r"\$?\d+(?:\.\d+)?%?")

# Suffix-stripping stemmer, used ONLY for antonym comparison.
#
# It over-stems: "billing" -> "bill". That is harmless *here* because the antonym
# table is a closed set of ~17 pairs and neither form appears in it, but it is
# the reason this stemmer is not applied to retrieval or coverage scoring, where
# conflating "billing" and "bill" would change results. Scoping a lossy transform
# to the one place it cannot hurt is the point.
#
# It exists so the antonym table stops needing every inflection listed by hand,
# which was a silent-miss source a test caught.
_SUFFIXES = ("ing", "ed", "es", "s")


def stem(token: str) -> str:
    for suf in _SUFFIXES:
        if len(token) > len(suf) + 3 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def _stems(terms) -> set:
    return {stem(t) for t in terms}


@dataclass
class ContradictionResult:
    contradicts: bool
    rule: str
    detail: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _numbers(text: str) -> set:
    return set(NUMBER.findall(text.lower()))


def _content_terms(text: str) -> set:
    return set(tokenize(text)) - NEGATIONS


def _polarity(text: str) -> bool:
    """True when the sentence carries a negation marker."""
    return bool(set(tokenize(text)) & NEGATIONS)


def detect(claim: str, context_texts, overlap_threshold: float = 0.5) -> ContradictionResult:
    """Does `claim` contradict any context sentence it substantially overlaps?

    Overlap is required first: two sentences about unrelated topics cannot
    contradict each other, and checking polarity without checking topic would
    flag every negative sentence in the corpus.
    """
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return ContradictionResult(False, "empty_claim", "")

    claim_neg = _polarity(claim)
    claim_nums = _numbers(claim)

    best_overlap = 0.0
    for text in context_texts:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sent_terms = _content_terms(sentence)
            if not sent_terms:
                continue
            overlap = len(claim_terms & sent_terms) / len(claim_terms)
            if overlap < overlap_threshold:
                continue
            best_overlap = max(best_overlap, overlap)

            # 1. Polarity flip on a topically-matching sentence.
            if _polarity(sentence) != claim_neg:
                return ContradictionResult(
                    True, "polarity_flip",
                    "claim %s a negation the context %s: %r"
                    % ("drops" if not claim_neg else "adds",
                       "carries" if not claim_neg else "lacks", sentence.strip()[:120]),
                )

            # 2. Same entity, different number.
            sent_nums = _numbers(sentence)
            if claim_nums and sent_nums and not (claim_nums & sent_nums):
                return ContradictionResult(
                    True, "numeric_mismatch",
                    "claim asserts %s where the context says %s"
                    % (sorted(claim_nums), sorted(sent_nums)),
                )

            # 3. Antonym substitution, compared on STEMS so that enables/enabled
            # and disables/disabled no longer need separate table entries.
            claim_stems, sent_stems = _stems(claim_terms), _stems(sent_terms)
            for a_raw, b_raw in ANTONYMS:
                a, b = stem(a_raw), stem(b_raw)
                if (a in claim_stems and b in sent_stems) or (b in claim_stems and a in sent_stems):
                    return ContradictionResult(
                        True, "antonym_substitution",
                        "claim uses %r where the context uses its opposite" % (a_raw if a in claim_stems else b_raw),
                    )

    if best_overlap == 0.0:
        # Nothing in the context is on this topic, so contradiction cannot be
        # adjudicated. "unknown" rather than "no contradiction" -- the caller
        # decides what to do with an unadjudicated claim.
        return ContradictionResult(False, "unknown", "no topically-overlapping context sentence")
    return ContradictionResult(False, "consistent", "overlap %.2f, no contradiction signal" % best_overlap)
