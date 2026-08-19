"""Answer generation over retrieved context, plus a cost/cache layer.

**There is no LLM API key in this environment**, so the default generator is
extractive: it selects and stitches the sentences from the retrieved context that
best cover the question. That is a real, if unsophisticated, generator — it can
be wrong, it can hallucinate by selecting the wrong sentence, and it can refuse.

The point is not that the generator is good. The point is that the **evaluation
harness around it is real**, and swapping in an API-backed generator is one class
that implements `generate(question, chunks) -> Answer`. Everything downstream —
faithfulness, coverage, refusal accuracy, the judge, cost accounting — works
unchanged.

The extractive generator has one property that makes it a useful baseline rather
than a placeholder: because every sentence it emits is copied verbatim from the
context, its faithfulness is near-perfect BY CONSTRUCTION. That makes it the
right control for a faithfulness metric — if the metric cannot score a
copy-paste generator highly, the metric is broken.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field

from .retrieval import tokenize

# Phrases that signal the corpus does not answer the question. An extractive
# generator cannot invent a refusal, so refusal is a decision made here based on
# retrieval evidence rather than something the generator writes.
REFUSAL_TEXT = "The documentation does not cover this."


@dataclass
class Answer:
    text: str
    used_chunk_ids: list = field(default_factory=list)
    refused: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


class CostTracker:
    """Counts tokens and dollars. Every generator call goes through it.

    Cost per query is a first-class metric because it is the number that decides
    whether a RAG feature ships. A system whose quality is known and whose cost
    is not cannot be reasoned about.
    """

    def __init__(self, prompt_usd_per_1k: float = 0.003, completion_usd_per_1k: float = 0.015):
        self.prompt_rate = prompt_usd_per_1k
        self.completion_rate = completion_usd_per_1k
        self.calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def record(self, prompt_tokens: int, completion_tokens: int, cached: bool) -> float:
        self.calls += 1
        if cached:
            self.cache_hits += 1
            return 0.0
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        cost = (prompt_tokens / 1000.0) * self.prompt_rate + \
               (completion_tokens / 1000.0) * self.completion_rate
        self.cost_usd += cost
        return cost

    def summary(self) -> dict:
        billed = self.calls - self.cache_hits
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / self.calls, 4) if self.calls else 0.0,
            "billed_calls": billed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost_usd": round(self.cost_usd, 6),
            "cost_per_query_usd": round(self.cost_usd / self.calls, 6) if self.calls else 0.0,
            "cost_per_billed_call_usd": round(self.cost_usd / billed, 6) if billed else 0.0,
        }


class ResponseCache:
    """Content-addressed cache on (question, context). Survives process restarts.

    Keyed on the CONTEXT as well as the question: the same question against a
    changed corpus must not return a stale answer. That is the bug a
    question-only cache ships with, and it is invisible until the corpus updates.
    """

    def __init__(self, path: str = ":memory:"):
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            " key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.con.commit()

    @staticmethod
    def key(question: str, chunk_ids) -> str:
        h = hashlib.sha256()
        h.update(question.strip().lower().encode())
        for cid in chunk_ids:
            h.update(b"|")
            h.update(str(cid).encode())
        return h.hexdigest()

    def get(self, key: str):
        row = self.con.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict):
        self.con.execute(
            "INSERT OR REPLACE INTO cache (key, payload, created_at) VALUES (?,?,?)",
            (key, json.dumps(payload), time.time()),
        )
        self.con.commit()


class ExtractiveGenerator:
    """Selects the context sentences that best cover the question.

    Refusal rule: if no retrieved sentence shares enough content with the
    question, refuse rather than emit the best of a bad set. The threshold is a
    parameter because it trades refusal accuracy against answer coverage, and
    that trade belongs to the product, not to the code.
    """

    def __init__(self, max_sentences: int = 3, min_overlap: int = 2):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, chunks) -> Answer:
        t0 = time.perf_counter()
        q_terms = set(tokenize(question))
        scored = []
        for chunk in chunks:
            for sent in split_sentences(chunk.text):
                overlap = len(q_terms & set(tokenize(sent)))
                if overlap:
                    scored.append((overlap, chunk.chunk_id, sent))
        scored.sort(key=lambda x: -x[0])

        best = [s for s in scored if s[0] >= self.min_overlap][: self.max_sentences]
        prompt_tokens = sum(len(tokenize(c.text)) for c in chunks) + len(q_terms)

        if not best:
            return Answer(text=REFUSAL_TEXT, used_chunk_ids=[], refused=True,
                          prompt_tokens=prompt_tokens, completion_tokens=len(tokenize(REFUSAL_TEXT)),
                          latency_ms=(time.perf_counter() - t0) * 1000.0)

        text = " ".join(s[2] for s in best)
        return Answer(text=text, used_chunk_ids=[s[1] for s in best], refused=False,
                      prompt_tokens=prompt_tokens, completion_tokens=len(tokenize(text)),
                      latency_ms=(time.perf_counter() - t0) * 1000.0)


class CachedGenerator:
    """Wraps a generator with the cache and the cost tracker."""

    def __init__(self, generator, cache: ResponseCache | None = None,
                 tracker: CostTracker | None = None):
        self.generator = generator
        self.cache = cache or ResponseCache()
        self.tracker = tracker or CostTracker()

    def generate(self, question: str, chunks) -> Answer:
        chunk_ids = [c.chunk_id for c in chunks]
        key = ResponseCache.key(question, chunk_ids)
        hit = self.cache.get(key)
        if hit is not None:
            answer = Answer(**hit)
            answer.cached = True
            self.tracker.record(answer.prompt_tokens, answer.completion_tokens, cached=True)
            return answer

        answer = self.generator.generate(question, chunks)
        answer.cost_usd = self.tracker.record(answer.prompt_tokens, answer.completion_tokens,
                                              cached=False)
        payload = answer.as_dict()
        payload["cached"] = False
        self.cache.put(key, payload)
        return answer
