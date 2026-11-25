import json
import re
from pathlib import Path
from typing import Any

from src.constants import AnswerLabel, format_citation
from src.retriever import tokenize

MIN_BM25_SCORE = 0.5
MIN_OVERLAP_COUNT = 1


def _detect_conflict(chunk_a: dict[str, Any], chunk_b: dict[str, Any], query_tokens: set[str]) -> bool:
    """
    Detects direct contradiction between two retrieved chunks that both address
    the query topic (e.g., opposite negation polarity or conflicting numbers).
    """
    tokens_a = set(tokenize(chunk_a["chunk_text"], remove_stopwords=True))
    tokens_b = set(tokenize(chunk_b["chunk_text"], remove_stopwords=True))

    shared_query_focus = (tokens_a & query_tokens) & (tokens_b & query_tokens)
    if len(shared_query_focus) < 2:
        return False

    negations = {"not", "no", "cannot", "never", "unexpired", "unavailable"}
    raw_a = set(re.findall(r"[a-z0-9]+", chunk_a["chunk_text"].lower()))
    raw_b = set(re.findall(r"[a-z0-9]+", chunk_b["chunk_text"].lower()))

    has_neg_a = bool(raw_a & negations)
    has_neg_b = bool(raw_b & negations)
    if has_neg_a != has_neg_b:
        return True

    nums_a = set(re.findall(r"\b\d+\b", chunk_a["chunk_text"]))
    nums_b = set(re.findall(r"\b\d+\b", chunk_b["chunk_text"]))
    if nums_a and nums_b and nums_a != nums_b and len(tokens_a & tokens_b) >= 3:
        return True

    return False


def generate_answer_for_query(retrieval_record: dict[str, Any]) -> dict[str, Any]:
    """
    Generates a citation-strict answer using ONLY the retrieved top_k chunks
    in retrieval_record. Never accesses raw kb/ files.
    """
    query_id = retrieval_record["query_id"]
    question = retrieval_record["question"]
    top_k = retrieval_record.get("top_k", [])

    if not top_k:
        return {
            "query_id": query_id,
            "answer_label": AnswerLabel.INSUFFICIENT_CONTEXT.value,
            "answer": "Insufficient context in the retrieved documents to answer this question.",
            "citations": [],
            "used_chunk_ids": [],
        }

    query_tokens = set(tokenize(question, remove_stopwords=True))

    # Score each retrieved chunk by combining BM25 score with exact query-token overlap
    scored_candidates: list[tuple[float, int, dict[str, Any]]] = []
    for item in top_k:
        chunk_tokens = set(tokenize(item["chunk_text"], remove_stopwords=True))
        title_tokens = set(tokenize(item["doc_title"], remove_stopwords=True))
        overlap = len(query_tokens & (chunk_tokens | title_tokens))
        combined_score = float(item["score"]) + (0.5 * overlap)
        scored_candidates.append((combined_score, overlap, item))

    scored_candidates.sort(key=lambda x: (x[0], -x[2]["rank"]), reverse=True)
    best_combined, best_overlap, best_chunk = scored_candidates[0]

    # Abstain when retrieval signal or lexical grounding is too weak
    if float(best_chunk["score"]) < MIN_BM25_SCORE or best_overlap < MIN_OVERLAP_COUNT:
        return {
            "query_id": query_id,
            "answer_label": AnswerLabel.INSUFFICIENT_CONTEXT.value,
            "answer": "Insufficient context in the retrieved documents to answer this question.",
            "citations": [],
            "used_chunk_ids": [],
        }

    # Check if top two candidates present conflicting rules on the same topic
    if len(scored_candidates) >= 2:
        _, second_overlap, second_chunk = scored_candidates[1]
        if second_overlap >= MIN_OVERLAP_COUNT and _detect_conflict(
            best_chunk, second_chunk, query_tokens
        ):
            cit_1 = format_citation(best_chunk["doc_title"], best_chunk["chunk_id"])
            cit_2 = format_citation(second_chunk["doc_title"], second_chunk["chunk_id"])
            answer_text = (
                f"Retrieved sources contain conflicting statements: "
                f"{best_chunk['chunk_text']} {cit_1} versus "
                f"{second_chunk['chunk_text']} {cit_2}"
            )
            return {
                "query_id": query_id,
                "answer_label": AnswerLabel.CONFLICTING_CONTEXT.value,
                "answer": answer_text,
                "citations": [cit_1, cit_2],
                "used_chunk_ids": [best_chunk["chunk_id"], second_chunk["chunk_id"]],
            }

    # Select primary supporting chunk (and a secondary chunk only if strongly relevant from same doc)
    selected = [best_chunk]
    if len(scored_candidates) >= 2:
        second_combined, second_overlap, second_chunk = scored_candidates[1]
        if (
            second_chunk["doc_title"] == best_chunk["doc_title"]
            and second_overlap >= 2
            and second_combined >= 0.75 * best_combined
        ):
            selected.append(second_chunk)

    answer_parts: list[str] = []
    citations: list[str] = []
    used_chunk_ids: list[str] = []

    for ch in selected:
        cit = format_citation(ch["doc_title"], ch["chunk_id"])
        answer_parts.append(f"{ch['chunk_text']} {cit}")
        citations.append(cit)
        used_chunk_ids.append(ch["chunk_id"])

    return {
        "query_id": query_id,
        "answer_label": AnswerLabel.GROUNDED_ANSWER.value,
        "answer": " ".join(answer_parts),
        "citations": citations,
        "used_chunk_ids": used_chunk_ids,
    }


def generate_all_answers(retrieval_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [generate_answer_for_query(rec) for rec in retrieval_records]


def save_answers(answers: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)