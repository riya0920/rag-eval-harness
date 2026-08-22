"""End-to-end generation evaluation: faithfulness, coverage, refusal, cost.

    python -m ragkit.generation_eval run
    python -m ragkit.generation_eval sample --n 30     # emit answers for hand-labelling
    python -m ragkit.generation_eval validate-judge

Refusal accuracy is reported as a **first-class metric**, split by category,
because a system that answers everything scores well on faithfulness by never
being asked a question it should decline.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .chunking import load_chunks
from .experiment import CORPUS, GATE_CONFIG, build, load_golden
from .generation import CachedGenerator, CostTracker, ExtractiveGenerator, ResponseCache
from .judge import (
    faithfulness,
    key_point_coverage,
    refusal_correct,
    validate_judge,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "eval", "results")
LABELS = os.path.join(ROOT, "eval", "human_labels.jsonl")


def _chunk_lookup(chunks):
    return {c.chunk_id: c for c in chunks}


def run_generation_eval(fetch_k: int = 5, mode: str = None, target_tokens: int = None,
                        cache_path: str = ":memory:") -> dict:
    mode = mode or GATE_CONFIG["mode"]
    target_tokens = target_tokens or GATE_CONFIG["target_tokens"]
    chunks, retriever = build(mode, target_tokens)
    lookup = _chunk_lookup(chunks)
    examples = load_golden()

    tracker = CostTracker()
    generator = CachedGenerator(ExtractiveGenerator(), ResponseCache(cache_path), tracker)

    rows = []
    t0 = time.perf_counter()
    for ex in examples:
        t_ret = time.perf_counter()
        hits = retriever.search(ex["question"], fetch_k)
        retrieve_ms = (time.perf_counter() - t_ret) * 1000.0
        ctx = [lookup[cid] for cid, _ in hits if cid in lookup]

        answer = generator.generate(ex["question"], ctx)
        faith = faithfulness(answer.text, [c.text for c in ctx])
        cover = key_point_coverage(answer.text, ex.get("key_points", []))

        rows.append({
            "id": ex["id"],
            "category": ex["category"],
            "refused": answer.refused,
            "refusal_correct": refusal_correct(answer.refused, ex["category"]),
            "faithfulness": faith.score,
            "n_claims": faith.n_claims,
            "coverage": cover["score"],
            "missing_points": cover["missing"],
            "retrieve_ms": retrieve_ms,
            "generate_ms": answer.latency_ms,
            "cached": answer.cached,
            "answer": answer.text,
        })

    wall = time.perf_counter() - t0

    def mean(key, subset=None):
        vals = [r[key] for r in (subset or rows)
                if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "refusal_accuracy": float(np.mean([r["refusal_correct"] for r in sub])),
            "refusal_rate": float(np.mean([r["refused"] for r in sub])),
            "faithfulness": mean("faithfulness", sub),
            "coverage": mean("coverage", sub),
        }

    answerable = [r for r in rows if r["category"] != "unanswerable"]
    return {
        "config": {"mode": mode, "target_tokens": target_tokens, "fetch_k": fetch_k,
                   "generator": "extractive"},
        "n_examples": len(rows),
        "overall": {
            "refusal_accuracy": float(np.mean([r["refusal_correct"] for r in rows])),
            "faithfulness": mean("faithfulness"),
            "coverage_answerable_only": mean("coverage", answerable),
            "p50_retrieve_ms": float(np.percentile([r["retrieve_ms"] for r in rows], 50)),
            "p50_generate_ms": float(np.percentile([r["generate_ms"] for r in rows], 50)),
            "wall_s": round(wall, 2),
        },
        "by_category": by_cat,
        "cost": tracker.summary(),
        "per_example": rows,
    }


def emit_sample(n: int, seed: int = 0) -> list:
    """Write answers for a random sample, for a human to label.

    Sampled rather than taking the first N, because the golden file is ordered by
    category and the first 30 would all be factual -- labelling only the easy
    class would inflate agreement.
    """
    result = run_generation_eval()
    rng = np.random.default_rng(seed)
    rows = result["per_example"]
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "sample", "validate-judge"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--fetch-k", type=int, default=5)
    ap.add_argument("--nli", action="store_true",
                    help="use the NLI fallback for claims the rules cannot adjudicate")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.command == "run":
        out = run_generation_eval(fetch_k=args.fetch_k)
        with open(os.path.join(RESULTS, "generation.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        summary = {k: v for k, v in out.items() if k != "per_example"}
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "sample":
        rows = emit_sample(args.n)
        for r in rows:
            print(json.dumps({"id": r["id"], "category": r["category"], "refused": r["refused"],
                              "judge_faithfulness": r["faithfulness"],
                              "answer": r["answer"][:300]}))
        return 0

    chunks, retriever = build(GATE_CONFIG["mode"], GATE_CONFIG["target_tokens"])
    generator = ExtractiveGenerator()
    examples = {e["id"]: e for e in load_golden()}
    out = validate_judge(LABELS, retriever, generator, examples, use_nli=args.nli)
    out['nli_enabled'] = args.nli
    with open(os.path.join(RESULTS, "judge_validation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
