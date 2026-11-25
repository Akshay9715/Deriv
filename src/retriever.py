import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "how", "i", "if", "in", "is", "it", "may", "my", "of", "on", "or",
    "that", "the", "their", "they", "this", "to", "what", "will", "with",
}


def normalize_token(token: str) -> str:
    """Lightweight deterministic normalizer so singular/plural variants align."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [normalize_token(t) for t in raw_tokens]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Retriever:
    """
    Deterministic Okapi BM25 retriever over chunk text enriched with
    document title and section metadata.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[dict[str, Any]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.doc_term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.num_docs: int = 0

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.num_docs = len(chunks)
        self.doc_freqs = Counter()
        self.doc_term_freqs = []
        self.doc_lengths = []

        for chunk in chunks:
            # Index doc_title + section + text so domain context boosts relevant articles
            composite_text = f"{chunk['doc_title']} {chunk['section']} {chunk['text']}"
            tokens = tokenize(composite_text, remove_stopwords=True)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0.0
        )

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score_chunk(self, query_tokens: list[str], doc_idx: int) -> float:
        if self.num_docs == 0 or self.avg_doc_len == 0.0:
            return 0.0

        tf_map = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0

        for term in query_tokens:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * ((tf * (self.k1 + 1.0)) / denom)

        return float(round(score, 6))

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        if top_k < 3:
            raise ValueError("top_k must be at least 3 per specification.")

        query_tokens = tokenize(question, remove_stopwords=True)
        scored: list[tuple[float, int, dict[str, Any]]] = []

        for idx, chunk in enumerate(self.chunks):
            score = self.score_chunk(query_tokens, idx)
            # Use negative idx as secondary sort key to preserve deterministic document order on ties
            scored.append((score, -idx, chunk))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = scored[:top_k]

        results: list[dict[str, Any]] = []
        for rank, (score, _, chunk) in enumerate(selected, start=1):
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "doc_title": chunk["doc_title"],
                    "score": float(score),
                    "chunk_text": chunk["text"],
                }
            )
        return results


def retrieve_all(
    retriever: BM25Retriever, queries: list[dict[str, Any]], top_k: int = 3
) -> list[dict[str, Any]]:
    retrieval_records: list[dict[str, Any]] = []
    for q in queries:
        results = retriever.retrieve(q["question"], top_k=top_k)
        retrieval_records.append(
            {
                "query_id": q["query_id"],
                "question": q["question"],
                "top_k": results,
            }
        )
    return retrieval_records


def save_retrieval(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)