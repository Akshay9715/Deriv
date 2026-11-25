import json
import sys
from pathlib import Path
from typing import Any

from src.constants import (
    ANSWERS_ARTIFACT,
    CHUNKING_COMPARISON_ARTIFACT,
    CHUNKS_ARTIFACT,
    EVAL_ARTIFACT,
    GROUNDING_ARTIFACT,
    KB_DIR,
    QUERIES_PATH,
    RETRIEVAL_ARTIFACT,
    PipelineStage,
    PipelineStateMachine,
)
from src.evaluator import evaluate_retrieval, save_evaluation
from src.generator import generate_all_answers, save_answers
from src.grounding import run_chunking_comparison, run_grounding_check, save_json
from src.ingestion import chunk_documents_sentence_line, load_documents, save_chunks
from src.retriever import BM25Retriever, retrieve_all, save_retrieval
from validate import validate_all


def load_queries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("queries.json must be a non-empty list.")
    return data


def run_pipeline(top_k: int = 3) -> None:
    # Stage 1: INIT
    state = PipelineStateMachine()

    # Stage 2: DOCUMENTS_LOADED
    documents = load_documents(KB_DIR)
    queries = load_queries(QUERIES_PATH)
    state.advance_to(PipelineStage.DOCUMENTS_LOADED)

    # Stage 3: DOCUMENTS_CHUNKED
    chunks = chunk_documents_sentence_line(documents)
    save_chunks(chunks, CHUNKS_ARTIFACT)
    state.advance_to(PipelineStage.DOCUMENTS_CHUNKED)

    # Stage 4: INDEX_BUILT
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    state.advance_to(PipelineStage.INDEX_BUILT)

    # Stage 5: RETRIEVAL_COMPLETE
    retrieval_records = retrieve_all(retriever, queries, top_k=top_k)
    save_retrieval(retrieval_records, RETRIEVAL_ARTIFACT)
    state.advance_to(PipelineStage.RETRIEVAL_COMPLETE)

    # Guard: Final answers MUST NOT be generated before retrieval completes
    state.require_stage(PipelineStage.RETRIEVAL_COMPLETE)

    # Stage 6: ANSWERS_GENERATED
    answers = generate_all_answers(retrieval_records)
    save_answers(answers, ANSWERS_ARTIFACT)
    state.advance_to(PipelineStage.ANSWERS_GENERATED)

    # Stage 7: EVALUATION_COMPLETE
    eval_report = evaluate_retrieval(queries, retrieval_records)
    save_evaluation(eval_report, EVAL_ARTIFACT)

    # Should-Attempt #5 & #6 artifacts
    grounding_report = run_grounding_check(retrieval_records, answers)
    save_json(grounding_report, GROUNDING_ARTIFACT)

    comparison_report = run_chunking_comparison(documents, queries, top_k=top_k)
    save_json(comparison_report, CHUNKING_COMPARISON_ARTIFACT)
    state.advance_to(PipelineStage.EVALUATION_COMPLETE)

    # Stage 8: VALIDATION_COMPLETE
    ok, errors = validate_all()
    if not ok:
        for err in errors:
            print(f"[VALIDATION ERROR] {err}", file=sys.stderr)
        raise RuntimeError("Pipeline validation failed.")
    state.advance_to(PipelineStage.VALIDATION_COMPLETE)

    # Stage 9: RESULTS_FINALISED
    state.advance_to(PipelineStage.RESULTS_FINALISED)
    print(
        f"Done. Processed {len(documents)} documents ({len(chunks)} chunks) "
        f"and {len(queries)} queries. Top-3 Hit Rate: {eval_report['summary']['top3_hit_rate']:.2%}"
    )


if __name__ == "__main__":
    run_pipeline()