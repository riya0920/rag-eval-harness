# RAG with an Evaluation Harness That Can Fail a Pull Request

Hybrid retrieval (BM25 + dense, RRF fusion) written without a framework, a
hand-authored golden set with four question categories, deterministic retrieval
metrics, and a CI gate that blocks a regression — plus an inverted CI step that
fails the build if the gate itself stops catching a planted regression.

> **Status: ~40% built.** Retrieval, the golden set, retrieval metrics, the
> experiment matrix and the CI gate are implemented and tested. Generation
> metrics, the LLM judge and its human-agreement validation, and cost tracking
> are **not** — see [Roadmap](#roadmap). No generation number is reported
> anywhere in this repo, because none has been measured.

## Why this exists

Every 2025-26 resume has a RAG project; almost none have an eval harness, and the
ones that do rarely ask whether the harness itself is valid. The two artifacts
here that are hard to fake are the **planted-regression CI step** and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md), which states in writing that this
project's own headline comparison is inside the noise band.

## Run it

```bash
pip install -r requirements.txt        # numpy + pytest. That is the whole dependency list.
make test                              # 17 unit tests
make matrix                            # sweep 3 retrieval modes x 3 chunk sizes
make gate                              # exit 0 -- no regression vs eval/baseline.json
make demo                              # exit 1 -- the gate catching a planted regression
```

There are no API keys and no network calls. The eval runs in about a second,
which is the only reason it can plausibly sit on every PR.

## The experiment matrix

35 golden examples, 27 of them labelled with relevant documents.

| run | chunks | recall@5 | mrr | ndcg@10 | r@5 factual | r@5 multi-hop | r@5 ambiguous | ms/query |
|---|---|---|---|---|---|---|---|---|
| bm25_chunk40 | 65 | 0.872 | 0.917 | 0.880 | 1.000 | 0.867 | 0.625 | 0.11 |
| bm25_chunk80 | 25 | 0.883 | 0.900 | 0.886 | 1.000 | 0.900 | 0.625 | 0.08 |
| bm25_chunk160 | 24 | 0.917 | 0.900 | 0.888 | 1.000 | 1.000 | 0.625 | 0.08 |
| dense_chunk40 | 65 | 0.872 | 0.871 | 0.857 | 1.000 | 0.867 | 0.625 | 0.36 |
| dense_chunk80 | 25 | 0.911 | 0.914 | 0.910 | 1.000 | 0.950 | 0.708 | 0.23 |
| dense_chunk160 | 24 | 0.900 | 0.908 | 0.902 | 1.000 | 0.950 | 0.625 | 0.36 |
| hybrid_chunk40 | 65 | 0.872 | 0.921 | 0.888 | 1.000 | 0.867 | 0.625 | 0.49 |
| hybrid_chunk80 | 25 | 0.911 | 0.914 | 0.905 | 1.000 | 0.950 | **0.708** | 0.35 |
| hybrid_chunk160 | 24 | 0.900 | 0.908 | 0.902 | 1.000 | 0.950 | 0.625 | 0.29 |

**How to read this table, stated by the person who produced it:** the factual
column is saturated at 1.000 for every configuration, so the overall average is
mostly measuring an easy category. `bm25_chunk160` tops the overall column by
0.017 recall@5, which on 27 labelled examples is **well inside the ±0.058
standard error** — it is not a real difference and I would not ship a retrieval
decision on it. The columns that actually discriminate are multi-hop and
ambiguous. Full argument in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

The gate therefore pins `hybrid @ 80 tokens` rather than the nominal "winner":
it is tied on the noisy overall metric and best on the one category that
separates configurations.

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
| **Generation: answer synthesis over retrieved context** | not started |
| **Claim-level faithfulness + key-point coverage via LLM judge** | not started |
| **Judge validation against human labels (report kappa)** | not started |
| **Refusal accuracy on the 8 unanswerable questions** | not started |
| **Cost tracking: per-call cache, tokens counted, cost/query** | not started |
| **Reranker arm + the latency price it charges** | not started |
| **A held-out slice to detect golden-set overfitting** | not started |

## Honesty notes

* No LLM is called anywhere in this repo yet. There is no judge, so there is no
  judge-agreement number, and the README does not report one.
* The dense retriever is **lexical** (TF-IDF + SVD), not semantic. It is honest
  about this in `embeddings.py` and it means the hybrid-vs-BM25 comparison here
  understates what a real encoder would buy.
* Latency numbers are single-process retrieval time on 24–65 chunks. They are
  not a serving benchmark and are not presented as one.
