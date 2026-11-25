import re
from enum import Enum
from pathlib import Path


class AnswerLabel(str, Enum):
    GROUNDED_ANSWER = "grounded_answer"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CONFLICTING_CONTEXT = "conflicting_context"


class RetrievalStatus(str, Enum):
    HIT = "hit"
    PARTIAL_HIT = "partial_hit"
    MISS = "miss"


class PipelineStage(str, Enum):
    INIT = "INIT"
    DOCUMENTS_LOADED = "DOCUMENTS_LOADED"
    DOCUMENTS_CHUNKED = "DOCUMENTS_CHUNKED"
    INDEX_BUILT = "INDEX_BUILT"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    ANSWERS_GENERATED = "ANSWERS_GENERATED"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


ORDERED_STAGES = [
    PipelineStage.INIT,
    PipelineStage.DOCUMENTS_LOADED,
    PipelineStage.DOCUMENTS_CHUNKED,
    PipelineStage.INDEX_BUILT,
    PipelineStage.RETRIEVAL_COMPLETE,
    PipelineStage.ANSWERS_GENERATED,
    PipelineStage.EVALUATION_COMPLETE,
    PipelineStage.VALIDATION_COMPLETE,
    PipelineStage.RESULTS_FINALISED,
]

# Allowed citation format: [doc_title §chunk_id]
CITATION_PATTERN = re.compile(r"^\[(?P<doc_title>[^\[\]§]+?)\s§(?P<chunk_id>[^\[\]§\s]+)\]$")


def format_citation(doc_title: str, chunk_id: str) -> str:
    """Formats a citation strictly according to the controlled vocabulary."""
    return f"[{doc_title} §{chunk_id}]"


# Standard input/output paths
KB_DIR = Path("kb")
QUERIES_PATH = Path("queries.json")
ARTIFACTS_DIR = Path("artifacts")

CHUNKS_ARTIFACT = ARTIFACTS_DIR / "chunks.json"
RETRIEVAL_ARTIFACT = ARTIFACTS_DIR / "retrieval.json"
ANSWERS_ARTIFACT = ARTIFACTS_DIR / "answers.json"
EVAL_ARTIFACT = ARTIFACTS_DIR / "eval.json"
GROUNDING_ARTIFACT = ARTIFACTS_DIR / "grounding_check.json"
CHUNKING_COMPARISON_ARTIFACT = ARTIFACTS_DIR / "chunking_comparison.json"


class PipelineStateMachine:
    """Enforces sequential pipeline stage execution and prevents out-of-order steps."""

    def __init__(self) -> None:
        self.current_index = 0
        self.history: list[str] = [ORDERED_STAGES[0].value]
        print(f"[STAGE] {self.current_stage.value}")

    @property
    def current_stage(self) -> PipelineStage:
        return ORDERED_STAGES[self.current_index]

    def advance_to(self, target_stage: PipelineStage) -> None:
        expected_next_index = self.current_index + 1
        if expected_next_index >= len(ORDERED_STAGES):
            raise RuntimeError(f"Pipeline already finalised at {self.current_stage.value}.")

        expected_stage = ORDERED_STAGES[expected_next_index]
        if target_stage != expected_stage:
            raise RuntimeError(
                f"Invalid stage transition: attempted {target_stage.value} "
                f"from {self.current_stage.value}, expected {expected_stage.value}."
            )

        self.current_index = expected_next_index
        self.history.append(target_stage.value)
        print(f"[STAGE] {self.current_stage.value}")

    def require_stage(self, required_stage: PipelineStage) -> None:
        required_idx = ORDERED_STAGES.index(required_stage)
        if self.current_index < required_idx:
            raise RuntimeError(
                f"Stage guard failed: requires {required_stage.value}, "
                f"currently at {self.current_stage.value}."
            )