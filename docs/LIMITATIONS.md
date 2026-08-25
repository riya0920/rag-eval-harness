# Limitations of this evaluation

Written before anyone asks, because an eval harness whose validity is unexamined
is just a number generator.

## 1. The corpus is small enough that factual retrieval is saturated

24 documents. Every configuration in the matrix scores **recall@5 = 1.000 on the
factual category**. That is not evidence that retrieval is solved; it is evidence
that the factual questions are too easy for a corpus this size, where the answer
document is often the only one containing the key term.

The categories that still discriminate are **multi-hop** (0.867-1.000) and
**ambiguous** (0.625-0.708). Those are the numbers worth reading, and the
per-category columns exist in the table so nobody quotes the saturated average.

**What this means for the winner.** `bm25_chunk160` "wins" overall by 0.017
recall@5 over `hybrid_chunk80`. With 27 labelled examples, the standard error on
a proportion near 0.9 is roughly `sqrt(0.9*0.1/27) ≈ 0.058`. **A 0.017 gap is
inside the noise.** The table reports a winner because the code has to pick one;
the honest reading is that BM25, dense, and hybrid are indistinguishable on this
corpus, and I would not ship a retrieval decision on this evidence.

## 2. The dense retriever is lexical, not semantic

`HashingSVDEmbedder` is TF-IDF over hashed uni/bigrams with a truncated SVD - i.e. LSA. It captures corpus co-occurrence, so it will connect "credential" to
"service key" if those words co-occur *in this corpus*. It has no world
knowledge. So:

* The **hybrid-vs-BM25 comparison here understates hybrid retrieval**, because
  both arms are lexical and fusing two correlated rankings adds little. With a
  real sentence encoder the dense arm would be genuinely independent and the
  fusion would have more to work with.
* Any conclusion of the form "hybrid isn't worth it" **does not transfer** to a
  system with a real encoder. The embedder is injected precisely so that swap can
  be measured rather than argued about.

## 3. No generation metrics yet

Faithfulness, key-point coverage, refusal accuracy and the judge-validation
number are **not implemented** (see the README roadmap). The `answer` and
`key_points` fields in the golden set exist and are hand-written, so the harness
is ready for them, but no generation number is reported anywhere in this repo
because none has been measured.

Specifically, **refusal accuracy on the 8 unanswerable questions is unmeasured.**
Retrieval metrics deliberately exclude those examples - a retriever cannot be
scored for correctly finding nothing - so the entire unanswerable category is
currently carried by the golden set and not by any reported metric.

## 4. The golden set was written by one person, who also wrote the corpus

There is no inter-annotator agreement number because there is one annotator. Two
consequences:

* **Question-writing bias.** I wrote questions knowing what the corpus contained,
  which biases toward questions the corpus answers cleanly. Real user questions
  are messier, worse-formed, and more often unanswerable than 8-in-35.
* **The relevant-doc labels encode my judgement** of what "relevant" means. For
  the ambiguous category especially, another annotator would plausibly label a
  different document set, which would move that category's numbers more than any
  config change in the matrix does.

## 5. Golden-set overfitting is a real risk and is not currently detected

Every iteration on the retriever is scored against the same 35 examples. After
enough iterations the harness measures "fit to this golden set", not retrieval
quality. Nothing in this repo currently detects that. The mitigation - a
held-out slice never used for tuning, plus periodic regeneration of the golden
set - is a roadmap item, not a thing that exists.

## 6. Single language, single domain, no distribution shift

English, one fictional product's documentation, no seasonality, no query
distribution drift. The gate would not catch a change that hurts a query type
absent from the golden set, which is the most common way an eval passes while
production gets worse.
