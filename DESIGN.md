# Mini-RAG Pipeline — System Design & Engineering Notes

## 1. Overview & Design Goals
This system implements a deterministic, replayable mini-RAG pipeline designed to run cleanly from a fresh checkout with zero reliance on hardcoded filenames, document ordering, or external network calls.

## 2. Architecture & Enforced Pipeline Stages
Execution is governed by an explicit finite state machine (`PipelineState`) enforcing strict stage ordering:
`INIT` -> `DOCUMENTS_LOADED` -> `DOCUMENTS_CHUNKED` -> `INDEX_BUILT` -> `RETRIEVAL_COMPLETE` -> `ANSWERS_GENERATED` -> `EVALUATION_COMPLETE` -> `VALIDATION_COMPLETE` -> `RESULTS_FINALISED`

Each stage persists its output to `artifacts/` before the next stage begins, preventing answer generation prior to retrieval completion.

## 3. Key Engineering Decisions

### A. Document Preprocessing & Chunking (`src/ingest.py`)
- **Discovery:** Files in `kb/` are discovered dynamically via `sorted(Path("kb").glob("*.txt"))` so swapping or renaming fixtures requires zero code changes.
- **Metadata Extraction:** `Title:` and `Section:` headers are parsed from the header block; the remaining text forms the document body.
- **Primary Chunking Strategy (Line/Sentence Granularity):** Product knowledge base articles consist of discrete policy statements. Splitting on non-empty lines/sentences preserves atomic facts, guarantees exact substring fidelity (`body[start_char:end_char] == chunk["text"]`), and prevents multi-topic dilution during retrieval.
- **Comparison Strategy (Fixed-Size Window):** Evaluated against 120-character fixed-window chunking in `artifacts/chunking_comparison.json` to measure precision vs. boundary-splitting tradeoffs.

### B. Retrieval Logic (`src/retriever.py`)
- **Algorithm:** Pure-Python BM25 (Okapi BM25 with $k_1 = 1.5, b = 0.75$) over normalized tokens, indexing `doc_title + section + chunk_text`.
- **Rationale:** BM25 handles exact domain terminology (e.g., "17:00 UTC", "2FA", "utility bills", "demo profit") reliably without external model downloads, ensuring fast, reproducible execution on any machine.

### C. Citation-Strict Grounded Generation (`src/generator.py`)
- **Approach:** Deterministic extractive synthesis over `top_k` retrieved chunks only.
- **Abstention & Conflict Handling:**
  - If the top retrieval score or non-stopword query overlap falls below a strict confidence threshold, the pipeline emits `insufficient_context` with empty citations rather than hallucinating.
  - If top-ranked chunks contain contradictory directives on the same subject, `conflicting_context` is flagged.
- **Citation Format:** Every factual claim is paired inline with `[doc_title §chunk_id]`, restricted strictly to chunks present in `artifacts/retrieval.json`.

### D. Evaluation & Validation (`src/evaluator.py`, `src/grounding.py`, `validate.py`)
- **Retrieval Evaluation:** Deterministically classifies each query as `hit`, `partial_hit`, or `miss` against `expected_doc_titles` in top-3 results and computes aggregate `top3_hit_rate`.
- **Grounding Check:** Verifies every cited `chunk_id` exists in `retrieval.json` and measures lexical overlap between the generated answer and cited `chunk_text`.
- **Standalone Validator:** `validate.py` verifies artifact existence, JSON validity, controlled vocabularies, score types, and citation integrity before `RESULTS_FINALISED`.