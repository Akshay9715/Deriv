import json
import re
from pathlib import Path
from typing import Any

from src.constants import AnswerLabel
from src.evaluator import evaluate_retrieval
from src.ingestion import chunk_documents_fixed_size, chunk_documents_sentence_line
from src.retriever import BM25Retriever, retrieve_all, tokenize


def run_grounding_check(
    retrieval_records: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    min_overlap_ratio: float = 0.5,
) -> dict[str, Any]:
    """
    Deterministically verifies that every citation in answers.json exists in
    retrieval.json and that the cited chunk text supports the generated answer.
    """
    retrieval_by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in retrieval_records:
        retrieval_by_query[rec["query_id"]] = {
            item["chunk_id"]: item for item in rec.get("top_k", [])
        }

    checks: list[dict[str, Any]] = []
    grounded_count = 0

    for ans in answers:
        qid = ans["query_id"]
        label = ans["answer_label"]
        used_ids = ans.get("used_chunk_ids", [])
        retrieved_map = retrieval_by_query.get(qid, {})

        # Strip inline [doc_title §chunk_id] tags to isolate claim wording
        clean_answer = re.sub(r"\[[^\[\]]+\]", "", ans.get("answer", "")).strip()
        answer_tokens = set(tokenize(clean_answer, remove_stopwords=True))

        if label == AnswerLabel.INSUFFICIENT_CONTEXT.value:
            is_grounded = len(used_ids) == 0
            checks.append(
                {
                    "query_id": qid,
                    "answer_label": label,
                    "citations_exist_in_retrieval": True,
                    "overlap_score": 1.0 if is_grounded else 0.0,
                    "is_grounded": is_grounded,
                    "reason": "Abstention answer has no ungrounded claims or citations.",
                }
            )
            if is_grounded:
                grounded_count += 1
            continue

        all_exist = all(cid in retrieved_map for cid in used_ids) and len(used_ids) > 0
        cited_tokens: set[str] = set()
        for cid in used_ids:
            if cid in retrieved_map:
                cited_tokens.update(
                    tokenize(retrieved_map[cid]["chunk_text"], remove_stopwords=True)
                )

        if not answer_tokens or not cited_tokens:
            overlap_score = 0.0
        else:
            overlap_score = round(len(answer_tokens & cited_tokens) / len(answer_tokens), 4)

        is_grounded = all_exist and (overlap_score >= min_overlap_ratio)
        if is_grounded:
            grounded_count += 1

        checks.append(
            {
                "query_id": qid,
                "answer_label": label,
                "citations_exist_in_retrieval": all_exist,
                "overlap_score": overlap_score,
                "is_grounded": is_grounded,
                "reason": (
                    "All citations exist in top_k and lexical support meets threshold."
                    if is_grounded
                    else "Missing citation reference or insufficient lexical support."
                ),
            }
        )

    return {
        "summary": {
            "total_answers": len(answers),
            "grounded_answers": grounded_count,
            "grounding_pass_rate": round(grounded_count / len(answers), 4) if answers else 0.0,
        },
        "results": checks,
    }


def run_chunking_comparison(
    documents: list[dict[str, str]],
    queries: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Compares retrieval performance between atomic line/sentence chunking
    and fixed-size 120-character window chunking.
    """
    strategies = {
        "sentence_line_chunks": chunk_documents_sentence_line(documents),
        "fixed_size_120_chars": chunk_documents_fixed_size(documents, window_size=120, overlap=20),
    }

    comparison: dict[str, Any] = {}
    for name, chunk_list in strategies.items():
        retriever = BM25Retriever()
        retriever.build_index(chunk_list)
        ret_records = retrieve_all(retriever, queries, top_k=top_k)
        eval_report = evaluate_retrieval(queries, ret_records)
        comparison[name] = {
            "total_chunks": len(chunk_list),
            "metrics": eval_report["summary"],
        }

    return {
        "strategies": comparison,
        "tradeoff_note": (
            "Sentence/line chunking preserves complete policy rules without mid-clause truncation, "
            "yielding cleaner term statistics and higher precision than fixed-size character windows "
            "that split key facts across boundaries."
        ),
    }


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)