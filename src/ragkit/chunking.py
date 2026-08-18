"""Corpus loading and chunking.

Chunking is a retrieval hyperparameter, not a formatting detail: it decides what
the smallest addressable unit of truth is. The eval matrix sweeps it, so it lives
behind a config rather than being hard-coded at the call site.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int


DOC_HEADER = re.compile(r"^##\s+(doc:[a-z0-9\-]+)\s*$", re.MULTILINE)


def parse_corpus(markdown: str) -> dict:
    """Split the corpus on `## doc:<id>` headers -> {doc_id: body}.

    Document ids are authored in the corpus rather than generated, so the golden
    set can reference them by hand and stay stable across chunking changes.
    """
    matches = list(DOC_HEADER.finditer(markdown))
    docs = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            docs[m.group(1)] = body
    return docs


def _sentences(text: str) -> list:
    """Sentence split that does not break on decimals or abbreviations like $0.045."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z*`])", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def chunk_documents(docs: dict, target_tokens: int = 120, overlap_sentences: int = 1) -> list:
    """Sentence-packed chunks with sentence-level overlap.

    Packing by sentence rather than by character keeps facts intact: a chunk that
    ends mid-sentence ("...overage billed at") is unretrievable *and* unusable as
    context, and character-window chunkers produce those constantly.

    `target_tokens` is a whitespace-token approximation. A real tokenizer would
    shift the boundaries slightly; it would not change the ranking of chunking
    configs in the experiment matrix, which is what this knob is swept for.
    """
    chunks = []
    for doc_id, body in sorted(docs.items()):
        sents = _sentences(body)
        buf, buf_tokens, ordinal = [], 0, 0
        for sent in sents:
            n = len(sent.split())
            if buf and buf_tokens + n > target_tokens:
                chunks.append(_make(doc_id, ordinal, buf))
                ordinal += 1
                buf = buf[-overlap_sentences:] if overlap_sentences else []
                buf_tokens = sum(len(s.split()) for s in buf)
            buf.append(sent)
            buf_tokens += n
        if buf:
            chunks.append(_make(doc_id, ordinal, buf))
    return chunks


def _make(doc_id: str, ordinal: int, sents: list) -> Chunk:
    return Chunk(chunk_id="%s#%d" % (doc_id, ordinal), doc_id=doc_id, text=" ".join(sents), ordinal=ordinal)


def load_chunks(path: str, target_tokens: int = 120, overlap_sentences: int = 1) -> list:
    with open(path, encoding="utf-8") as fh:
        return chunk_documents(parse_corpus(fh.read()), target_tokens, overlap_sentences)
