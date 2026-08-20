# RAG with an Evaluation Harness That Can Fail a Pull Request

Hybrid retrieval (BM25 + dense, RRF fusion) written without a framework, a
hand-authored golden set with four question categories, deterministic retrieval
metrics, and a CI gate that blocks a regression — plus an inverted CI step that
fails the build if the gate itself stops catching a planted regression.

> **Status: ~95% built.** Retrieval, a **103-example** golden set, retrieval
> metrics, the experiment matrix, the CI gate, **generation with faithfulness /
> coverage / refusal metrics**, a **judge validated against 42 human labels**,
> **cost tracking**, and **contradiction detection** are implemented and measured.
> A real LLM generator and an NLI judge are not — see [Roadmap](#roadmap).

## Why this exists

Every 2025-26 resume has a RAG project; almost none have an eval harness, and the
ones that do rarely ask whether the harness itself is valid. The two artifacts
here that are hard to fake are the **planted-regression CI step** and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md), which states in writing that this
project's own headline comparison is inside the noise band.

## Run it

```bash
pip install -r requirements.txt        # numpy + scipy + pytest. That is the whole list.
make test                              # 35 unit tests
make matrix                            # sweep 3 retrieval modes x 3 chunk sizes
make gate                              # exit 0 -- no regression vs eval/baseline.json
make demo                              # exit 1 -- the gate catching a planted regression
make generation                        # faithfulness, coverage, refusal, cost
make judge                             # agreement + kappa vs eval/human_labels.jsonl
```

There are no API keys and no network calls. The eval runs in about a second,
which is the only reason it can plausibly sit on every PR.

## The judge is validated, and the validation found a blind spot

42 hand-labelled examples in `eval/human_labels.jsonl`:

| | before contradiction detection | after |
|---|---|---|
| labelled examples | 42 | **48** |
| raw agreement | 97.6% | **97.9%** |
| **Cohen's kappa** | 0.954 | **0.963** |
| disagreements | 1 (a contradiction it missed) | 1 (an *implication* it cannot reach) |

**Why kappa and not raw agreement:** a judge that always says "faithful" agrees
97% of the time on a corpus that is 97% faithful, while carrying no information.
Kappa removes that floor, and `test_cohens_kappa_punishes_a_degenerate_judge`
pins the property.

**Why 12 of the 42 labels are deliberately fabricated.** The extractive generator
copies sentences verbatim, so it is faithful *by construction* — a label set
drawn only from its real output contains no unfaithful examples at all, and kappa
on a single-class set is meaningless. Twelve labels carry an
`inject_fabrication` field: a false sentence appended to a real answer, labelled
unfaithful. Without them the "kappa" would have been a decoration.

### The blind spot it found, and the fix

The first validation run found the judge scoring this **1.0 — fully supported**:

> *"Credits below 99.0% uptime are paid out in cash within 14 days."*

against a corpus that says credits are **never paid in cash**. Token-overlap
entailment asks whether a claim's *words* appear in the context, and this
fabrication is built entirely from the corpus's own vocabulary. **The judge
detected invention and missed contradiction** — biased in favour of exactly the
failure mode that matters most in a documentation assistant.

`contradiction.py` closes it with three rules aimed at the shapes that actually
occur in factual docs, each reporting *which* rule fired because "invented" and
"reversed" are different bugs:

| rule | catches |
|---|---|
| polarity flip | context negates what the claim asserts (*never paid* vs *paid*) |
| numeric mismatch | same entity, different number (*400 days* vs *40 days*) |
| antonym substitution | one member of an opposed pair swapped (*enabled* / *disabled*) |

A claim it cannot adjudicate returns **`unknown`, not `supported`** — degrading
to "I don't know" rather than to a false pass.

`test_faithfulness_now_rejects_a_contradiction_that_token_overlap_accepts` pins
the exact `f041` failure, and asserts that disabling the second gate reproduces
the old wrong behaviour.

### Then I made the labels harder, because kappa 1.0 is a warning

With contradiction detection on, agreement hit **1.0 with zero disagreements** —
which does not mean the judge is perfect, it means **the label set stopped
challenging it**. So six harder adversarial cases were added, chosen to probe the
new detector's limits rather than confirm its strengths.

**Five of six were caught. One was not**, and it is the one predicted to fail:

> *"Starter tier customers also receive a named technical account manager."*

The corpus grants a TAM only on Scale above $250k spend. There is no negation to
flip, no antonym, and no conflicting number *in the claim itself* — the
contradiction is an **implication**, and detecting it needs entailment rather
than surface cues. That is the honest boundary of a rule-based judge and the
precise argument for an NLI model, stated as a measured limit rather than a
disclaimer.

**A second finding, about the method rather than the judge:** one label had to be
*relabelled* mid-project. `f007` was originally labelled "refused"; a tokenizer
fix changed retrieval and the system began answering correctly, making the label
stale. Human labels rot when the system under test changes. That is a real
ongoing cost of judge validation and the label file records it.

## Generation metrics

103 golden examples, hybrid retrieval at k=5, extractive generator:

| metric | value |
|---|---|
| faithfulness | 0.887 |
| coverage (answerable only) | 0.667 |
| refusal accuracy (overall) | 0.796 |
| p50 retrieve / generate | 0.92 ms / 0.97 ms |
| **cost per query** | **$0.001196** |

By category — and the split is where the honesty lives:

| category | n | refusal accuracy | faithfulness | coverage |
|---|---|---|---|---|
| factual | 56 | 0.946 | 0.946 | 0.804 |
| multi-hop | 22 | **1.000** | 1.000 | 0.572 |
| ambiguous | 10 | 0.300 | 0.500 | 0.108 |
| **unanswerable** | 15 | **0.267** | 0.800 | 0.211 |

**Refusal accuracy on unanswerable questions is 26.7%, and that is the headline
failure of this system.** The extractive generator answers when it should
decline: given a question the corpus cannot answer, it finds *some* sentence with
enough word overlap and returns it. The most instructive case is `u012` — asked
which Kubernetes version Meridian runs, it answered about refund policy. Every
claim in that answer is grounded in the corpus, so it scores as **faithful while
being completely non-responsive**.

That case is why faithfulness and coverage are reported separately and why
refusal is a first-class metric. A single "quality" score would have hidden it.

**Ambiguous questions score worst on coverage (0.108)** because the golden answers
require enumerating several referents and the generator returns one. Recognising
ambiguity and asking for clarification is not implemented at all.

## Cost tracking

Every generation call goes through a cost tracker and a content-addressed cache:

* the cache is keyed on **(question, retrieved chunk ids)**, not the question
  alone — a question-only cache serves a stale answer after the corpus changes,
  and that bug is invisible until it matters
* cache hits are **not billed**, and the summary separates `calls` from
  `billed_calls`
* **$0.001196 per query** at the configured rates

## The experiment matrix

103 golden examples, 88 of them labelled with relevant documents.

| run | chunks | recall@5 | mrr | ndcg@10 | r@5 factual | r@5 multi-hop | r@5 ambiguous | ms/query |
|---|---|---|---|---|---|---|---|---|
| bm25_chunk40 | 65 | 0.927 | 0.929 | 0.911 | 1.000 | 0.871 | 0.648 | 0.37 |
| bm25_chunk80 | 25 | 0.943 | 0.928 | 0.920 | 1.000 | 0.894 | 0.759 | 0.36 |
| bm25_chunk160 | 24 | **0.953** | 0.939 | 0.929 | 1.000 | 0.939 | 0.759 | 0.39 |
| dense_chunk40 | 65 | 0.927 | 0.907 | 0.894 | 1.000 | 0.871 | 0.648 | 1.36 |
| dense_chunk80 | 25 | 0.944 | 0.932 | 0.929 | 0.982 | 0.917 | **0.833** | 1.35 |
| dense_chunk160 | 24 | 0.927 | 0.923 | 0.916 | 0.982 | 0.871 | 0.759 | 0.84 |
| hybrid_chunk40 | 65 | 0.927 | 0.924 | 0.908 | 1.000 | 0.871 | 0.648 | 2.16 |
| hybrid_chunk80 | 25 | 0.950 | 0.932 | 0.928 | 1.000 | 0.894 | **0.833** | 1.19 |
| hybrid_chunk160 | 24 | 0.951 | 0.934 | 0.926 | 1.000 | 0.932 | 0.759 | 1.41 |

With 88 labelled examples the standard error near 0.95 is about 0.023, so the
0.003 spread across the top four rows is **still inside the noise** — the larger
golden set narrowed the band but did not make these configurations separable.
The factual column remains saturated at 1.000; ambiguous is where they differ.

The gate pins `hybrid @ 80 tokens` rather than the nominal winner: it is tied on
the noisy overall metric and best on the one category that separates
configurations.

**A tokenizer bug the generation work exposed:** the token pattern deliberately
keeps `.` so that `$0.045` survives, which also meant a sentence-final period
attached — `hours.` never matched `hours`. Invisible in BM25, where both sides
get identical treatment, but silently fatal for key-point matching. Fixing it
moved recall@5 from 0.917 to 0.953. **A bug that only one of two consumers can
see is the argument for having two consumers.**

## Retrieval is written, not imported

## The four design decisions

**1. Retrieval is written, not imported.** BM25 (with a floored Robertson IDF —
unfloored, a term appearing in over half the corpus gets a *negative* weight and
penalises documents that contain it), the dense retriever, and RRF are ~150 lines
in `src/ragkit/retrieval.py`. `test_bm25_idf_is_never_negative` exists because
that bug is silent.

**2. RRF over score normalisation.** BM25 scores and cosine similarities are on
incomparable scales, and min-max or z-score normalising makes fusion depend on
the score distribution *of the current query*. RRF uses rank order only, so it
cannot be destabilised that way — `test_rrf_is_immune_to_score_scale` asserts it.
What RRF gives up is confidence: a rank-1 runaway and a rank-1 photo-finish
contribute identically. That discarded signal is exactly what a learned fusion
would monetise, once there is enough labelled data to fit one.

**3. Unanswerable questions are excluded from retrieval metrics, not scored zero.**
Scoring them zero punishes a retriever for correctly finding nothing; scoring
them one rewards returning garbage. Retrieval metrics simply do not define them —
refusal is a *generation* metric and belongs there. The count of excluded
examples is reported with every run so the exclusion is visible, not silent.

**4. The gate is verified by an inverted CI step.** `make gate` failing on a real
regression is the easy half. The CI also runs `gate --plant-regression`, which
collapses the embedding to 2 dimensions, and **fails the build if that passes**.
A gate that can never fail is not a gate.

```
$ make demo
[planted regression] embed_dim 128 -> 2, dense-only
FAIL recall@5     baseline=0.9111 current=0.3833 delta=-0.5278
FAIL mrr          baseline=0.9144 current=0.2714 delta=-0.6431

RETRIEVAL REGRESSION on: recall@5, mrr (tolerance 0.020 absolute)
```

## The golden set

`eval/golden.jsonl`, 35 hand-written examples over a fictional corpus, committed
to git with verified document ids (`test_golden_set_references_only_real_docs`
fails the build on a dangling reference).

| category | n | what it tests |
|---|---|---|
| factual | 12 | single-document lookup |
| multi-hop | 10 | the answer requires joining two or three documents |
| unanswerable | 8 | the correct behaviour is refusal, not a plausible answer |
| ambiguous | 5 | the question has more than one referent and should be clarified |

**The corpus is fictional on purpose.** Every fact exists only in
`corpus/meridian_docs.md`, so a model cannot answer from pretraining and the
harness measures retrieval and grounding rather than memorisation. Six of the
eight unanswerable questions are the hard kind — they *sound* answerable and
reference real documents ("what is the SLA for the Starter tier?" — Starter has
no SLA) rather than being obviously off-topic.

## Roadmap (the remaining ~60%)

| Milestone | Status |
|---|---|
| Hybrid retrieval, no framework | done |
| Golden set, 4 categories, verified doc ids | done |
| Retrieval metrics (p@k, r@k, MRR, nDCG) | done |
| Experiment matrix, 9 tracked runs | done |
| CI eval gate + planted-regression proof | done |
| Written limitations / validity analysis | done |
| Golden set expanded to 103 examples (spec floor ~100) | done |
| Generation over retrieved context | done |
| Claim-level faithfulness + key-point coverage | done |
| Judge validated against 42 human labels (kappa 0.954) | done |
| Adversarial fabrications so kappa is not degenerate | done |
| Refusal accuracy as a first-class per-category metric | done |
| Cost tracking with a context-keyed cache | done |
| Contradiction detection: polarity, numeric, antonym | done |
| Harder adversarial labels probing the detector's limits | done |
| **NLI judge (rules cannot reach implication contradiction)** | not started |
| **Stemming (antonym inflections are listed by hand today)** | not started |
| **A real LLM generator (extractive stands in)** | not started |
| **Reranker arm + the latency price it charges** | not started |
| **Clarification behaviour on ambiguous questions** | not started |
| **A held-out slice to detect golden-set overfitting** | not started |

## Honesty notes

* **No LLM is called anywhere in this repo.** The generator is extractive and the
  judge is lexical. The *methodology* — hand-label a sample, compute kappa,
  inspect the disagreements, report the bias — is what transfers; the specific
  faithfulness number does not.
* **The judge now catches contradiction by rule, not by understanding.** It
  handles polarity, numbers and a hand-listed antonym table; it cannot reach
  implication contradiction, demonstrated by the one remaining miss. Faithfulness
  remains an upper bound, just a tighter one.
* **The antonym list has no stemmer**, so an unlisted inflection is a silent
  miss. `enables`/`disables` had to be added by hand after a test caught it.
* **Refusal accuracy on unanswerable questions is 26.7%.** That is a genuine
  weakness of the extractive generator, reported rather than buried in an
  average.
* The dense retriever is **lexical** (TF-IDF + SVD), not semantic. It is honest
  about this in `embeddings.py` and it means the hybrid-vs-BM25 comparison here
  understates what a real encoder would buy.
* Latency numbers are single-process retrieval time on 24–65 chunks. They are
  not a serving benchmark and are not presented as one.
