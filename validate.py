import json
import re
import sys
from pathlib import Path
from typing import Any

from src.constants import (
    ANSWERS_ARTIFACT,
    CHUNKING_COMPARISON_ARTIFACT,
    CHUNKS_ARTIFACT,
    CITATION_PATTERN,
    EVAL_ARTIFACT,
    GROUNDING_ARTIFACT,
    QUERIES_PATH,
    RETRIEVAL_ARTIFACT,
    AnswerLabel,
    RetrievalStatus,
)


def _load_json(path: Path, errors: list[str]) -> Any:
    if not path.exists():
        errors.append(f"Missing required artifact: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {path}: {exc}")
        return None


def validate_all() -> tuple[bool, list[str]]:
    """
    Validates all required (and optional, if present) pipeline artifacts
    against schema, controlled vocabularies, and citation invariants.
    """
    errors: list[str] = []

    queries = _load_json(QUERIES_PATH, errors)
    chunks = _load_json(CHUNKS_ARTIFACT, errors)
    retrieval = _load_json(RETRIEVAL_ARTIFACT, errors)
    answers = _load_json(ANSWERS_ARTIFACT, errors)
    eval_report = _load_json(EVAL_ARTIFACT, errors)

    # Validate optional artifacts if they were generated
    if GROUNDING_ARTIFACT.exists():
        _load_json(GROUNDING_ARTIFACT, errors)
    if CHUNKING_COMPARISON_ARTIFACT.exists():
        _load_json(CHUNKING_COMPARISON_ARTIFACT, errors)

    if errors:
        return False, errors

    # 1. Validate queries & chunks structure
    if not isinstance(queries, list) or not queries:
        errors.append("queries.json must be a non-empty list.")
        return False, errors

    if not isinstance(chunks, list) or not chunks:
        errors.append("artifacts/chunks.json must be a non-empty list.")
    else:
        required_chunk_keys = {"chunk_id", "doc_title", "section", "text", "start_char", "end_char"}
        for idx, ch in enumerate(chunks):
            missing = required_chunk_keys - set(ch.keys())
            if missing:
                errors.append(f"Chunk at index {idx} missing keys: {sorted(missing)}")

    expected_query_ids = [q["query_id"] for q in queries]

    # 2. Validate retrieval.json
    retrieved_by_query: dict[str, dict[str, dict[str, Any]]] = {}
    if not isinstance(retrieval, list):
        errors.append("artifacts/retrieval.json must be a list.")
    else:
        ret_ids = [r.get("query_id") for r in retrieval]
        if ret_ids != expected_query_ids:
            errors.append(f"Retrieval query_ids {ret_ids} do not match queries.json {expected_query_ids}.")

        for rec in retrieval:
            qid = rec.get("query_id", "<unknown>")
            top_k = rec.get("top_k")
            if not isinstance(top_k, list) or len(top_k) < 3:
                errors.append(f"Query {qid}: retrieval top_k must contain at least 3 chunks.")
                continue

            chunk_map: dict[str, dict[str, Any]] = {}
            for item in top_k:
                score = item.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    errors.append(f"Query {qid}: non-numeric retrieval score {score!r}.")
                cid = item.get("chunk_id")
                if cid:
                    chunk_map[cid] = item
            retrieved_by_query[qid] = chunk_map

    # 3. Validate answers.json (vocabularies & citation strictness)
    allowed_labels = {label.value for label in AnswerLabel}
    if not isinstance(answers, list):
        errors.append("artifacts/answers.json must be a list.")
    else:
        ans_ids = [a.get("query_id") for a in answers]
        if ans_ids != expected_query_ids:
            errors.append(f"Answers query_ids {ans_ids} do not match queries.json {expected_query_ids}.")

        for ans in answers:
            qid = ans.get("query_id", "<unknown>")
            label = ans.get("answer_label")
            if label not in allowed_labels:
                errors.append(f"Query {qid}: invalid answer_label {label!r}.")

            citations = ans.get("citations", [])
            used_chunk_ids = ans.get("used_chunk_ids", [])
            answer_text = ans.get("answer", "")
            retrieved_chunks_for_q = retrieved_by_query.get(qid, {})

            if label == AnswerLabel.GROUNDED_ANSWER.value and len(citations) < 1:
                errors.append(f"Query {qid}: grounded_answer must include at least one citation.")

            for cit in citations:
                match = CITATION_PATTERN.match(cit)
                if not match:
                    errors.append(f"Query {qid}: citation {cit!r} violates [doc_title §chunk_id] format.")
                    continue
                cit_doc_title = match.group("doc_title")
                cit_chunk_id = match.group("chunk_id")

                if cit_chunk_id not in retrieved_chunks_for_q:
                    errors.append(
                        f"Query {qid}: cited chunk_id {cit_chunk_id!r} was not retrieved in top_k."
                    )
                else:
                    actual_title = retrieved_chunks_for_q[cit_chunk_id]["doc_title"]
                    if cit_doc_title != actual_title:
                        errors.append(
                            f"Query {qid}: citation doc_title {cit_doc_title!r} != retrieved {actual_title!r}."
                        )

                if cit not in answer_text:
                    errors.append(f"Query {qid}: citation {cit!r} not present in answer string.")

            for cid in used_chunk_ids:
                if cid not in retrieved_chunks_for_q:
                    errors.append(f"Query {qid}: used_chunk_id {cid!r} not found in retrieved top_k.")

            # Also check inline bracketed citations inside answer text
            inline_cits = re.findall(r"\[[^\[\]]+\]", answer_text)
            for inline_cit in inline_cits:
                if not CITATION_PATTERN.match(inline_cit):
                    errors.append(f"Query {qid}: malformed inline citation {inline_cit!r}.")

    # 4. Validate eval.json
    allowed_statuses = {st.value for st in RetrievalStatus}
    if not isinstance(eval_report, dict):
        errors.append("artifacts/eval.json must be a JSON object.")
    else:
        summary = eval_report.get("summary")
        required_summary_keys = {"top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"}
        if not isinstance(summary, dict) or not required_summary_keys.issubset(summary.keys()):
            errors.append(
                f"artifacts/eval.json summary missing required keys: {sorted(required_summary_keys)}."
            )

        results = eval_report.get("results")
        if not isinstance(results, list) or len(results) != len(queries):
            errors.append("artifacts/eval.json results must contain one record per query.")
        else:
            for res in results:
                qid = res.get("query_id", "<unknown>")
                status = res.get("retrieval_status")
                if status not in allowed_statuses:
                    errors.append(f"Query {qid}: invalid retrieval_status {status!r}.")

    return len(errors) == 0, errors


def main() -> None:
    ok, errors = validate_all()
    if ok:
        print("VALIDATION PASSED: All artifacts, schemas, vocabularies, and citations are valid.")
        sys.exit(0)
    else:
        print("VALIDATION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()