# RAG with an Evaluation Harness That Can Fail a Pull Request

Hybrid retrieval (BM25 + dense, RRF fusion) written without a framework, a
hand-authored golden set with four question categories, deterministic retrieval
metrics, and a CI gate that blocks a regression — plus an inverted CI step that
fails the build if the gate itself stops catching a planted regression.

> **Status: ~100% of the spec built.** Retrieval, a **103-example** golden set, retrieval
> metrics, the experiment matrix, the CI gate, **generation with faithfulness /
> coverage / refusal metrics**, a **judge validated against 42 human labels**,
> **cost tracking**, **contradiction detection**, an **NLI judge**, a **reranker
> arm** and a **held-out slice** are implemented and measured. Two of those were
> measured and *rejected* — see below. A hosted LLM generator is the one thing
> still missing, and it needs an API key this environment does not have.

## Why this exists

Every 2025-26 resume has a RAG project; almost none have an eval harness, and the
ones that do rarely ask whether the harness itself is valid. The two artifacts
here that are hard to fake are the **planted-regression CI step** and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md), which states in writing that this
project's own headline comparison is inside the noise band.

## Run it

```bash
pip install -r requirements.txt        # numpy + scipy + pytest. That is the whole list.
make test                              # 46 unit tests
make matrix                            # sweep 3 retrieval modes x 3 chunk sizes
make gate                              # exit 0 -- no regression vs eval/baseline.json
make demo                              # exit 1 -- the gate catching a planted regression
make generation                        # faithfulness, coverage, refusal, cost
make judge                             # agreement + kappa vs eval/human_labels.jsonl
```

There are no API keys and no network calls. The eval runs in about a second,
which is the only reason it can plausibly sit on every PR.

## Two upgrades I built, measured, and did not ship

The spec rewards judgement, and judgement shows up in what you *decline* to ship.
Both of these are the obvious next move, both were built properly, and both lost
on measurement.

### The NLI judge does not beat the rules

The rule-based detector's documented limit is *implication* contradiction. The
textbook fix is an NLI model, so `nli.py` runs one locally
(`distilbert-base-uncased-mnli`, CPU, no API key). Swept against the 48 human
labels:

| premise overlap | confidence | kappa | disagreements |
|---|---|---|---|
| 0.30 | 0.70 | 0.716 | 8 |
| 0.30 | 0.90 | 0.747 | 7 |
| 0.50 | 0.90 | 0.889 | 3 |
| 0.70 | 0.70 | 0.926 | 2 |
| 0.70 | 0.98 | **0.963** | 1 |
| *rules alone* | — | **0.963** | 1 |

**Its best configuration exactly ties the rules and never beats them** — and gets
there only by being tuned so tight it almost never fires. Worse, at *every*
setting including the best, it **still misses `f054`**, the implication case it
was added to catch. It is off by default, kept behind a flag with the numbers
attached, because deleting it would invite the next person to try it again.

**A real bug this found**, and the reason the first attempt scored κ=0.194: an
MNLI model asked to compare two confident assertions about *different* subjects
returns **CONTRADICTION, not NEUTRAL**. "Service keys are long-lived" against a
chunk about deploy rollbacks scores CONTRADICTION at 1.000. With five retrieved
chunks per claim at least one is always off-topic, so accepting a contradiction
from *any* chunk fired constantly and called 25 of 26 verbatim-copied answers
unfaithful. **The premise has to be selected before the model is asked.** The
rule-based path never had this bug because it required lexical overlap first.

### The reranker makes retrieval worse here

`rerank.py` adds the signals BM25 discards — exact phrase match, term proximity,
query coverage, length normalisation — then blends with the retrieval ranking by
reciprocal rank. Sweeping the blend weight:

| weight on reranker | recall@5 | mrr | ndcg@10 | vs retrieval |
|---|---|---|---|---|
| 0.00 (retrieval only) | **0.9497** | **0.9321** | **0.9279** | — |
| 0.20 | 0.9358 | 0.8939 | 0.9014 | −1.46% |
| 0.35 | 0.9184 | 0.8194 | 0.8453 | −3.29% |
| 0.50 | 0.8976 | 0.7489 | 0.7887 | −5.48% |
| 0.80 | 0.7118 | 0.6107 | 0.6757 | −25.05% |

**Monotone decreasing, and that monotonicity is the proof.** If the reranker
carried any signal retrieval lacked, *some* weight above zero would beat zero.
None does — its information is strictly redundant on this corpus.

So the answer to *"your reranker bought +9% precision for +80ms — ship it or
not?"* is, here: **no.** It costs ~1.6 ms/query and buys negative quality. The
reason is headroom, not rerankers: recall@5 is already 0.9497 and the factual
category is saturated at 1.000. There is nothing left to reorder.

**An earlier version replaced the retrieval order instead of blending with it and
cost 34% of recall@5.** A reranker only earns the right to overrule retrieval if
it is better than retrieval.

**What is NOT claimed:** none of this shows rerankers are useless. A
purpose-trained cross-encoder could not be downloaded here, and `CrossEncoderReranker`
is implemented but deliberately **unevaluated** — scoring with the cached MNLI
model destroyed quality (recall@5 0.950 → 0.259 at 3.5 s/query), which says
everything about using an entailment model for relevance and nothing about
reranking.

## The held-out slice

Every configuration decision here was made against the same 103 examples, which
eventually measures fit to the file rather than retrieval quality.

```
$ python -m ragkit.experiment heldout

n_dev              83
n_heldout          20        (stratified: 11 factual, 4 multi-hop, 3 ambiguous, 2 unanswerable)
dev  recall@5      0.9444
held recall@5      0.9722
gap               -0.0278    -> no evidence of overfitting
```

The split is **hashed on example id, not shuffled**, so an example stays on the
same side forever — a reshuffle between runs would leak the held-out slice into
tuning one example at a time. It is **stratified**, because an all-factual
held-out slice would be uninformative given that category is saturated.

**What a small gap does and does not prove:** it bounds overfitting to the
*split*. Both slices come from one corpus written by one person, so it says
nothing about overfitting to the corpus or to my question-writing style.

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

## Clarification: the third component this repo built and rejected

*"What is the rate limit?"* has two correct answers in this corpus — 600 requests
per minute on the control plane, 10,000 per second on the data plane. A system
that retrieves well and generates faithfully returns **one of them**, right about
the sentence and wrong about the question. Retrieval metrics do not catch it (the
correct chunk *was* retrieved); faithfulness does not catch it (the answer *is*
grounded). It fails a check nothing upstream is looking for.

`make clarify` builds a detector for it, and the detector does not work.

| threshold | recall | precision | lift over base rate | interrupts |
|---|---|---|---|---|
| 0.25 | 0.90 | 0.13 | 1.34x | 67% |
| 0.35 | 0.80 | 0.14 | 1.44x | 54% |
| 0.60 | 0.30 | 0.18 | 1.87x | 15% |
| **0.65** | 0.20 | **0.18** | **1.87x** | 10% |
| 0.75 | 0.00 | 0.00 | — | 7% |

**Peak precision 0.18 against a base rate of 0.097.** `recommend()` returns
`shippable: false` rather than picking the least-bad row.

The base-rate column is the one usually missing. A detector that fires on
everything scores precision *equal to* the base rate by construction, so a
reported 0.18 sounds like a working classifier while being 1.9x better than a
coin. Every row carries the lift for that reason.

### Two of three signals carried no information, and one was a category error

The design started from the shape ambiguity takes in retrieval: top results
individually strong, collectively disagreeing. Three signals followed from that,
and measuring them killed two.

**Document spread — dead.** Mean spread is 0.95–1.00 for *every* category
including plain factual questions. With 80-token chunks over a many-document
corpus, the top 5 essentially always come from 5 different documents. The signal
measures the chunk size, not the question.

**Fused score margin — dead, mechanically.** The retriever fuses with **RRF**,
which discards raw scores for `1/(60 + rank)` sums. The top fused score is
`0.03279` for *"What is the rate limit?"* and `0.03279` for *"How do I rotate a
service key?"* — identical, because it is a pure function of the rank pair and
independent of the query. Mean flatness came out 0.98–0.99 across all four
categories, exactly as that predicts.

Worth stating generally: **any confidence signal derived from RRF scores is
measuring a constant.** RRF is rank fusion and is not supposed to preserve score
information; using its output as a confidence is a category error, and an easy
one because the numbers still look like scores. A test asserts the two queries
share a top score, so if that ever stops being true the signal is worth
revisiting.

The surviving pair is a **flat margin on the raw BM25 arm** (0.369 on ambiguous
against 0.525 on factual) **and competing quantities**. They only work together:
the flat margin fires hardest on *unanswerable* questions (0.354 — the flattest
of all, since nothing matches), and what separates those from ambiguous ones is
that an unanswerable question has no competing specifics to disagree about.

### A unit bug that hid three of the ten

Grouping quantities by their literal unit string never compares "20 minutes" with
"3 hours" — which are competing answers to *"how long does a restore take"*.
Normalising into canonical dimensions (durations to seconds, sizes to bytes,
rates to per-second) moved the questions the signal can even reach from **2 of 10
to 5 of 10**, and peak lift from 1.29x to 1.87x.

### Why it still fails, and it is not tuning

Of the 10 ambiguous questions, **5 are ambiguous between numbers and 5 are not**.
*"Can I change it after creation?"*, *"Is it included?"*, *"How many can I have?"*
are ambiguous between entities and procedures, and no quantity signal reaches
them. Half the positive class is invisible by construction. Closing that needs a
signal over entities and section topics — and a positive class larger than 10 to
measure it with, since at n=10 recall moves in steps of 0.1.

### One bug in the recommender itself

The first version checked only the interruption budget and returned threshold
0.75 as feasible — at **recall 0.00**, a detector that never fires and therefore
never interrupts anyone. Satisfying a constraint by doing nothing is not a
recommendation, and `recommend()` now requires recall above zero and precision
meaningfully above the base rate before it will name a threshold.

### Entity ambiguity: the other half of the positive class

The quantity detector reaches 5 of the 10 ambiguous questions. The other 5 share a
shape it cannot see, because **the missing information is not in the retrieved
text — it is absent from the question**:

```
a005  "Can I change it after creation?"        <- "it" has no antecedent
a008  "How many can I have?"                   <- how many *what*
a009  "Is it included?"                        <- "it" again
a010  "What happens when I exceed the limit?"  <- *which* limit
```

Two new signals. **Dangling referent** — a pronoun or bare quantifier with no
noun phrase anywhere in the question to bind it — costs two regexes and no
retrieval, so it can run *before* the retriever does. **Competing definitions** —
the question's head terms defined across several different *documents* in the
top-k, counted per document rather than per mention, because the same term
restated three times in one doc is one answer.

| | peak lift over base rate | recall at peak |
|---|---|---|
| quantity only | 1.87x | 0.20 |
| **+ entity signals** | **2.44x** | **0.90** |

Still `shippable: false` — reaching that recall interrupts **31%** of unambiguous
questions against a 5% budget. Better is not the same as good.

### A feature that scored better because it was broken

An earlier run of this reported **3.79x**, and keeping it would have been the
easy call. It came from a bug.

The definite-phrase rule ("the limit" is underspecified, "the rate limit for the
data plane" is not) used a greedy match that **swallowed the qualifier**, so the
most specific question in the set was scored as vague. Fixing that dropped the
lift from 3.79x to 2.44x — which is *exactly* the score with the rule removed
entirely.

So on 103 examples, a broken feature firing on questions it should not was better
correlated with ambiguity than the correct version of the same feature. That is
overfitting seen from the inside, and the number was large enough to be worth
keeping if nobody had asked why. The rule stays, fixed, documented as
contributing **no measurable lift** — a feature kept on reasoning rather than
evidence, labelled as such.

The route there is worth recording too. The rule was cut on the strength of three
counterexamples written by hand, restored when the golden set contradicted them
(3.79x vs 2.44x), and only then did fixing the bug show that the golden set had
been agreeing with a defect. **The hand-written probes were right and the
aggregate was wrong**, which is the opposite of the usual lesson and the reason
to keep both.

### Two bugs in the detector itself

* **The pronoun rule could never fire.** It read `tokenize()` output, and
  `tokenize` strips stopwords — pronouns are stopwords. "Can I change it after
  creation?" tokenises to `['can','i','change','after','creation']`: the one word
  the rule exists to find is deleted before it looks. It reads the raw question
  now.
* **A definite phrase that is a prepositional object is not the head.** "a service
  key for **the data plane**" qualifies something else, and treating it as the
  subject flags every question ending in a prepositional phrase.

### What is circular here, and what is not

The entity features were designed **with the ten ambiguous questions visible**, so
recall is an upper bound rather than an estimate of generalisation — and no
cross-validation repairs that, because the leakage is in the feature design.

The 93 negatives were **not** consulted while designing anything, so the
interruption rate, and therefore precision, is an honest measurement. That
asymmetry is why the headline is precision.

The clean fix is a larger positive class written by someone who has not seen the
detector. Authoring 40 more from the same taxonomy that produced these features
would make the evaluation *more* circular while making the number look better,
which is the trade this declines to take.

## Roadmap

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
| NLI judge, swept and measured — **rejected, does not beat rules** | done |
| Reranker arm with blend sweep — **rejected, monotone worse here** | done |
| Stemming, scoped to antonym comparison only | done |
| Held-out slice, stratified and hash-stable | done |
| **A hosted LLM generator (extractive stands in)** | blocked: no API key here |
| **A purpose-trained cross-encoder reranker** | blocked: model download refused |
| Clarification detector, swept and measured — **rejected, 1.9x over base rate** | done |
| Entity ambiguity: dangling referents + competing definitions (1.87x -> 2.44x) | done |
| **A positive class written by someone who has not seen the detector** | not available: authoring it here would be circular |

## Honesty notes

* **No LLM is called anywhere in this repo.** The generator is extractive and the
  judge is lexical. The *methodology* — hand-label a sample, compute kappa,
  inspect the disagreements, report the bias — is what transfers; the specific
  faithfulness number does not.
* **Two features were built and rejected on measurement** (NLI judge, reranker).
  They ship behind flags, off by default, with the numbers that justify the
  decision — not deleted, because the measurement is the artifact.
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
