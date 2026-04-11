"""TF-IDF retriever — built-in vector search with zero dependencies."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from eachmind.primitives.perspective import EncodedMemory


def _extract_text(value: Any) -> str:
    """Recursively extract text from encoded content."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_extract_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_extract_text(v) for v in value)
    return str(value)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into tokens."""
    return [
        token for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) >= 2
    ]


@dataclass
class TFIDFRetriever:
    """TF-IDF based retriever. Zero external dependencies."""

    def search(
        self,
        query: str,
        memories: list[EncodedMemory],
        limit: int = 10,
    ) -> list[EncodedMemory]:
        """Search memories using TF-IDF cosine similarity."""
        if not memories or not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Build document token lists
        doc_tokens: list[list[str]] = []
        for mem in memories:
            text = _extract_text(mem.encoded_content)
            doc_tokens.append(_tokenize(text))

        # Compute IDF
        num_docs = len(doc_tokens)
        all_terms = set(query_tokens)
        for tokens in doc_tokens:
            all_terms.update(tokens)

        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens:
            unique = set(tokens)
            for term in unique:
                doc_freq[term] += 1

        idf: dict[str, float] = {}
        for term in all_terms:
            idf[term] = math.log((num_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1

        # Query TF-IDF vector
        query_tf = Counter(query_tokens)
        query_len = len(query_tokens) or 1
        query_vec: dict[str, float] = {
            t: (c / query_len) * idf.get(t, 1.0)
            for t, c in query_tf.items()
        }

        # Score each document
        scored: list[tuple[float, int]] = []
        for idx, tokens in enumerate(doc_tokens):
            if not tokens:
                scored.append((0.0, idx))
                continue

            doc_tf = Counter(tokens)
            doc_len = len(tokens)

            # Cosine similarity
            dot = 0.0
            doc_norm = 0.0
            for term, count in doc_tf.items():
                tfidf_val = (count / doc_len) * idf.get(term, 1.0)
                doc_norm += tfidf_val ** 2
                if term in query_vec:
                    dot += tfidf_val * query_vec[term]

            query_norm = sum(v ** 2 for v in query_vec.values())
            denom = math.sqrt(query_norm) * math.sqrt(doc_norm)
            score = dot / denom if denom > 0 else 0.0
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scored[:limit]:
            if score > 0:
                results.append(memories[idx])
        return results
