import json
from pathlib import Path
from typing import Any

from src.constants import RetrievalStatus


def evaluate_retrieval(
    queries: list[dict[str, Any]], retrieval_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Deterministically evaluates retrieval quality against expected_doc_titles
    in queries.json and computes aggregate top-3 hit metrics.
    """
    retrieval_by_id = {rec["query_id"]: rec for rec in retrieval_records}

    per_query_results: list[dict[str, Any]] = []
    hits = 0
    partial_hits = 0
    misses = 0

    for q in queries:
        query_id = q["query_id"]
        expected_titles: list[str] = q.get("expected_doc_titles", [])
        rec = retrieval_by_id.get(query_id)

        if rec is None:
            raise ValueError(f"Missing retrieval record for query_id={query_id}")

        top3_items = rec.get("top_k", [])[:3]

        # Preserve rank order while deduplicating titles for clean reporting
        retrieved_titles_top3: list[str] = []
        first_rank_by_title: dict[str, int] = {}
        for item in top3_items:
            title = item["doc_title"]
            if title not in first_rank_by_title:
                first_rank_by_title[title] = int(item["rank"])
                retrieved_titles_top3.append(title)

        matched_titles = [t for t in expected_titles if t in first_rank_by_title]
        matched_expected = len(matched_titles) > 0

        if not expected_titles:
            status = RetrievalStatus.HIT if matched_expected else RetrievalStatus.MISS
            explanation = "No expected document titles specified."
        elif len(matched_titles) == len(expected_titles):
            status = RetrievalStatus.HIT
            hits += 1
            rank_details = ", ".join(
                f"'{t}' at rank {first_rank_by_title[t]}" for t in matched_titles
            )
            explanation = f"Expected title found: {rank_details}"
        elif len(matched_titles) > 0:
            status = RetrievalStatus.PARTIAL_HIT
            partial_hits += 1
            missing_titles = [t for t in expected_titles if t not in first_rank_by_title]
            rank_details = ", ".join(
                f"'{t}' at rank {first_rank_by_title[t]}" for t in matched_titles
            )
            explanation = (
                f"Partial match: found {rank_details}; missing {missing_titles} in top 3"
            )
        else:
            status = RetrievalStatus.MISS
            misses += 1
            explanation = (
                f"None of expected titles {expected_titles} appeared in top 3 retrieved documents"
            )

        per_query_results.append(
            {
                "query_id": query_id,
                "expected_doc_titles": expected_titles,
                "retrieved_doc_titles_top3": retrieved_titles_top3,
                "retrieval_status": status.value,
                "matched_expected_title": matched_expected,
                "explanation": explanation,
            }
        )

    total_queries = len(queries)
    top3_hit_rate = round(hits / total_queries, 4) if total_queries > 0 else 0.0

    summary = {
        "top3_hit_rate": top3_hit_rate,
        "total_queries": total_queries,
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses,
    }

    return {
        "summary": summary,
        "results": per_query_results,
    }


def save_evaluation(eval_report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)