"""The experiment matrix + the CI gate.

    python -m ragkit.experiment matrix     # sweep configs, write a comparison table
    python -m ragkit.experiment gate       # fail non-zero if the baseline regressed

The gate is the point. An eval you run when you remember to is a notebook; an
eval that fails a PR is a test.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

from .chunking import load_chunks
from .embeddings import build_default_embedder
from .metrics import evaluate_retrieval
from .retrieval import HybridRetriever

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORPUS = os.path.join(ROOT, "corpus", "meridian_docs.md")
GOLDEN = os.path.join(ROOT, "eval", "golden.jsonl")
RESULTS = os.path.join(ROOT, "eval", "results")


def load_golden(path: str = GOLDEN) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build(mode: str, target_tokens: int, k1: float = 1.5, b: float = 0.75, dim: int = 128):
    chunks = load_chunks(CORPUS, target_tokens=target_tokens)
    embedder = build_default_embedder(chunks, dim=dim) if mode in ("dense", "hybrid") else None
    return chunks, HybridRetriever(chunks, embedder=embedder, mode=mode, k1=k1, b=b)


def run_matrix(out_dir: str = RESULTS) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    examples = load_golden()

    modes = ["bm25", "dense", "hybrid"]
    # 40/80/160 rather than 120/200: the corpus documents are short enough that
    # 120 and 200 produce an identical chunking, and a sweep whose arms are the
    # same configuration is not a sweep.
    chunk_sizes = [40, 80, 160]
    runs = []
    for mode, target in itertools.product(modes, chunk_sizes):
        t0 = time.perf_counter()
        chunks, retriever = build(mode, target)
        build_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        res = evaluate_retrieval(retriever, examples)
        query_s = time.perf_counter() - t1

        runs.append(
            {
                "run_id": "%s_chunk%d" % (mode, target),
                "config": {"mode": mode, "target_tokens": target, "k1": 1.5, "b": 0.75, "embed_dim": 128},
                "n_chunks": len(chunks),
                "build_s": round(build_s, 3),
                "ms_per_query": round(1000 * query_s / max(len(examples), 1), 2),
                "overall": res["overall"],
                "per_category": res["per_category"],
                "n_scored": res["n_scored"],
            }
        )
        print(
            "%-16s chunks=%3d  recall@5=%.3f  mrr=%.3f  %.1f ms/query"
            % (runs[-1]["run_id"], len(chunks), res["overall"]["recall@5"], res["overall"]["mrr"], runs[-1]["ms_per_query"])
        )

    best = max(runs, key=lambda r: r["overall"]["recall@5"])
    report = {"runs": runs, "winner": best["run_id"], "n_examples": len(examples)}
    with open(os.path.join(out_dir, "matrix.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(out_dir, "matrix.md"), "w") as fh:
        fh.write(to_markdown(report) + "\n")
    return report


def to_markdown(report: dict) -> str:
    cats = ["factual", "multi-hop", "ambiguous"]
    lines = [
        "| run | chunks | recall@5 | mrr | ndcg@10 | " + " | ".join("r@5 %s" % c for c in cats) + " | ms/query |",
        "|---|---|---|---|---|" + "---|" * len(cats) + "---|",
    ]
    for r in report["runs"]:
        o = r["overall"]
        per = ["%.3f" % r["per_category"].get(c, {}).get("recall@5", float("nan")) for c in cats]
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f | %s | %.2f |"
            % (r["run_id"], r["n_chunks"], o["recall@5"], o["mrr"], o["ndcg@10"], " | ".join(per), r["ms_per_query"])
        )
    lines.append("")
    lines.append("Winner by recall@5: **%s**" % report["winner"])
    return "\n".join(lines)


BASELINE = os.path.join(ROOT, "eval", "baseline.json")
GATE_METRICS = ("recall@5", "mrr")
TOLERANCE = 0.02  # absolute; a 2-point drop on 32 labelled examples is real, not noise


GATE_CONFIG = {"mode": "hybrid", "target_tokens": 80}


def run_gate(update: bool = False, plant_regression: bool = False) -> int:
    examples = load_golden()
    if plant_regression:
        # A deliberately crippled retriever, to prove the gate has teeth. This is
        # the demo for "would your CI actually catch a bad change?" -- collapsing
        # the embedding to 2 dimensions destroys the dense arm while leaving every
        # import, signature and test passing. Exactly the kind of change that
        # slips through code review.
        print("[planted regression] embed_dim 128 -> 2, dense-only")
        _chunks, retriever = build("dense", GATE_CONFIG["target_tokens"], dim=2)
    else:
        _chunks, retriever = build(GATE_CONFIG["mode"], GATE_CONFIG["target_tokens"])
    res = evaluate_retrieval(retriever, examples)
    current = {m: res["overall"][m] for m in GATE_METRICS}

    if update or not os.path.exists(BASELINE):
        with open(BASELINE, "w") as fh:
            json.dump({"metrics": current, "config": GATE_CONFIG}, fh, indent=2)
        print("baseline written:", json.dumps(current, indent=2))
        return 0

    with open(BASELINE) as fh:
        baseline = json.load(fh)["metrics"]

    failures = []
    for m in GATE_METRICS:
        delta = current[m] - baseline[m]
        status = "OK  " if delta >= -TOLERANCE else "FAIL"
        print("%s %-12s baseline=%.4f current=%.4f delta=%+.4f" % (status, m, baseline[m], current[m], delta))
        if delta < -TOLERANCE:
            failures.append(m)

    if failures:
        print("\nRETRIEVAL REGRESSION on: %s (tolerance %.3f absolute)" % (", ".join(failures), TOLERANCE))
        return 1
    print("\nretrieval gate passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["matrix", "gate", "update-baseline", "heldout"])
    ap.add_argument("--plant-regression", action="store_true",
                    help="run the gate against a deliberately degraded retriever to prove it fails")
    args = ap.parse_args()
    if args.command == "heldout":
        out = run_heldout()
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "matrix":
        report = run_matrix()
        print()
        print(to_markdown(report))
        return 0
    return run_gate(update=args.command == "update-baseline", plant_regression=args.plant_regression)




# ---------------------------------------------------------------------------
# held-out slice -- the guard against golden-set overfitting
# ---------------------------------------------------------------------------

HELDOUT_FRACTION = 0.25


def split_golden(examples=None, fraction: float = HELDOUT_FRACTION, seed: int = 1234):
    """Deterministic dev/held-out split of the golden set.

    Every configuration decision in this repo has been made against the SAME 103
    examples. After enough iterations that stops measuring retrieval quality and
    starts measuring fit to this particular file -- and nothing in the harness
    would show it, because the number being optimised is the number being
    reported.

    The split is:
      * **stratified by category**, so the held-out slice is not accidentally all
        factual, which is the saturated class and would make it uninformative
      * **hashed on example id**, not shuffled, so an example stays in the same
        side forever. A reshuffle between runs would leak the held-out slice into
        tuning one example at a time.
    """
    import hashlib

    examples = examples if examples is not None else load_golden()
    by_cat = {}
    for ex in examples:
        by_cat.setdefault(ex["category"], []).append(ex)

    dev, held = [], []
    for cat, rows in sorted(by_cat.items()):
        rows = sorted(rows, key=lambda e: e["id"])
        for ex in rows:
            h = hashlib.blake2b(("%s:%s" % (seed, ex["id"])).encode(), digest_size=8).hexdigest()
            (held if (int(h, 16) % 10_000) / 10_000.0 < fraction else dev).append(ex)
    return dev, held


def run_heldout(mode: str = None, target_tokens: int = None) -> dict:
    """Score the gate configuration on dev and held-out separately.

    A large dev-vs-held-out gap is the signature of golden-set overfitting. A
    small gap does not prove there is none -- both slices come from one corpus
    written by one person -- but a large one is conclusive.
    """
    mode = mode or GATE_CONFIG["mode"]
    target_tokens = target_tokens or GATE_CONFIG["target_tokens"]
    _chunks, retriever = build(mode, target_tokens)
    dev, held = split_golden()

    dev_m = evaluate_retrieval(retriever, dev)
    held_m = evaluate_retrieval(retriever, held)
    gap = dev_m["overall"]["recall@5"] - held_m["overall"]["recall@5"]

    return {
        "config": {"mode": mode, "target_tokens": target_tokens},
        "n_dev": len(dev),
        "n_heldout": len(held),
        "dev": {k: dev_m["overall"][k] for k in ("recall@5", "mrr", "ndcg@10")},
        "heldout": {k: held_m["overall"][k] for k in ("recall@5", "mrr", "ndcg@10")},
        "recall@5_gap": gap,
        "heldout_categories": {c: len([e for e in held if e["category"] == c])
                               for c in sorted({e["category"] for e in held})},
        "verdict": ("no evidence of overfitting" if abs(gap) < 0.05
                    else "dev/held-out gap exceeds 5 points -- treat tuned numbers as optimistic"),
        "caveat": ("both slices come from one corpus written by one person, so a small gap "
                   "bounds overfitting to the SPLIT, not to the corpus or the question style."),
    }

if __name__ == "__main__":
    sys.exit(main())
