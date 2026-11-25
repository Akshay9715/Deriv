# Replayable Mini-RAG Pipeline

A deterministic, citation-strict Retrieval-Augmented Generation (RAG) pipeline built in Python as part of the **Deriv AI Engineer Hands-on Technical Challenge**.

The project implements an end-to-end pipeline that ingests local product knowledge-base documents, performs deterministic chunking and retrieval, generates grounded answers with citations, and evaluates retrieval quality against predefined ground-truth data.

The implementation prioritizes reproducibility, traceability, explainability, and reliable execution from a clean checkout.

## 1. Key Features

- **Document Ingestion:** Automatically discovers and loads `.txt` documents from the local `kb/` directory.
- **Deterministic Chunking:** Extracts document titles and sections and splits document content into reproducible chunks while preserving source text.
- **Local Retrieval:** Implements retrieval and ranking directly in Python without relying on external retrieval services.
- **Citation-Strict Answers:** Generates answers using retrieved context and attaches references in the required format.
- **Retrieval Evaluation:** Measures retrieval performance against expected document titles and ground-truth data.
- **Grounding Validation:** Checks whether answer citations reference retrieved chunks and whether the supporting content is present.
- **Pipeline Stage Tracking:** Enforces the required execution sequence.
- **Artifact Generation:** Saves intermediate and final results as structured JSON files.
- **Reproducibility:** Runs locally without external APIs or proprietary data.

## 2. Project Structure

```text
deriv-mini-rag/
│
├── kb/
│   ├── article_01.txt
│   ├── article_02.txt
│   └── ...
│
├── artifacts/
│   ├── chunks.json
│   ├── retrieval.json
│   ├── answers.json
│   ├── eval.json
│   ├── grounding_check.json
│   └── chunking_comparison.json
│
├── queries.json
├── run_pipeline.py
├── validate.py
├── README.md
├── requirements.txt
└── DESIGN.md
```

_Note: Generated artifacts are created by running the pipeline. Optional artifacts are included when their corresponding stages are implemented._

## 3. Requirements

- Python 3.10 or later
- No external API keys required
- No external LLM or hosted vector database required

The core implementation uses Python's standard library.

## 4. Running the Pipeline

### Step 1: Clone or extract the repository

```bash
git clone <repository-url>
cd deriv-mini-rag
```

Alternatively, extract the submitted project archive and navigate to its root directory.

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Execute the pipeline

```bash
python run_pipeline.py
```

The pipeline automatically reads documents from `kb/`, processes the queries from `queries.json`, and generates the required artifacts inside `artifacts/`.

### Step 4: Validate the results

```bash
python validate.py
```

The validation command checks the generated outputs against the challenge requirements.

### Clean execution

To verify reproducibility, remove previously generated artifacts and execute the pipeline again:

```bash
rm -rf artifacts
python run_pipeline.py
python validate.py
```

On Windows PowerShell:

```powershell
Remove-Item artifacts -Recurse -Force -ErrorAction SilentlyContinue
python run_pipeline.py
python validate.py
```

## 5. Pipeline Architecture

The implementation follows the required execution sequence:

```text
INIT
  |
  v
DOCUMENTS_LOADED
  |
  v
DOCUMENTS_CHUNKED
  |
  v
INDEX_BUILT
  |
  v
RETRIEVAL_COMPLETE
  |
  v
ANSWERS_GENERATED
  |
  v
EVALUATION_COMPLETE
  |
  v
VALIDATION_COMPLETE
  |
  v
RESULTS_FINALISED
```

Each stage is executed in sequence. Answer generation is performed only after retrieval has completed.

### Stage 1: Document Ingestion and Chunking

- Discovers knowledge-base documents dynamically.
- Extracts `Title:` and `Section:` metadata.
- Processes document bodies without depending on filenames or document ordering.
- Generates deterministic chunk identifiers.
- Tracks source character offsets.
- Preserves the original chunk text.

Output: `artifacts/chunks.json`

### Stage 2: Retrieval and Ranking

- Processes every question from `queries.json`.
- Retrieves relevant document chunks using a locally implemented ranking strategy.
- Assigns numerical relevance scores.
- Returns the top three or more ranked chunks for each query.

Output: `artifacts/retrieval.json`

### Stage 3: Citation-Strict Answer Generation

- Uses retrieved chunks as the source of factual information.
- Avoids introducing unsupported information.
- Supports the controlled answer labels:
  - `grounded_answer`
  - `insufficient_context`
  - `conflicting_context`
- Associates factual answers with their supporting chunk references.

Required citation format:

```text
[doc_title §chunk_id]
```

Output: `artifacts/answers.json`

### Stage 4: Deterministic Retrieval Evaluation

Retrieval quality is evaluated programmatically using the expected document titles and ground-truth answer points in `queries.json`.

The evaluation includes:

- Expected document title matching.
- Top-3 retrieval performance.
- Per-query retrieval status.
- Deterministic explanations.
- Aggregate retrieval statistics.

Allowed retrieval statuses:

- `hit`
- `partial_hit`
- `miss`

Output: `artifacts/eval.json`

### Stage 5: Answer Grounding Validation

The grounding validator checks:

- Whether citations exist in the retrieval results.
- Whether cited chunks were actually retrieved.
- Whether the supporting content is present in the cited source text.

Output: `artifacts/grounding_check.json`

### Stage 6: Chunking Strategy Comparison

Where implemented, multiple chunking strategies are evaluated against the same query set to compare retrieval performance and understand the trade-offs between chunk granularity and retrieval relevance.

Output: `artifacts/chunking_comparison.json`

## 6. Generated Artifacts

| Artifact                   | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `chunks.json`              | Extracted and chunked knowledge-base documents |
| `retrieval.json`           | Ranked retrieval results for each query        |
| `answers.json`             | Generated answers with citations               |
| `eval.json`                | Per-query and aggregate retrieval evaluation   |
| `grounding_check.json`     | Citation and answer-grounding validation       |
| `chunking_comparison.json` | Comparison of chunking strategies              |

All artifacts are generated programmatically during pipeline execution.

## 7. Validation

The validation script checks the integrity and consistency of the generated results.

It verifies:

1. Required artifact files exist.
2. JSON artifacts are valid.
3. Every input query has been processed.
4. Each query has at least three retrieved chunks.
5. Retrieval scores are numeric.
6. Answer labels belong to the allowed vocabulary.
7. Grounded answers contain citations.
8. Citations reference retrieved chunks only.
9. Evaluation statuses belong to the allowed vocabulary.
10. Aggregate evaluation metrics are present.

Run validation using:

```bash
python validate.py
```

## 8. Design Decisions

The implementation emphasizes deterministic and locally executable components for the core RAG workflow.

**Why deterministic retrieval?**

Retrieval and ranking are implemented in code rather than delegated entirely to an LLM. This makes retrieval behavior reproducible, measurable, and independently evaluable.

**Why citation-strict generation?**

Every factual answer must be traceable to retrieved knowledge-base content. If sufficient supporting context is unavailable, the system should communicate that limitation rather than fabricate an answer.

**Why structured artifacts?**

Persisting intermediate outputs makes the pipeline easier to debug, inspect, reproduce, and evaluate.

**Why local execution?**

The project avoids unnecessary external service dependencies, API credentials, and proprietary datasets, allowing another engineer to execute it from a clean checkout.

## 9. LLM Usage

This implementation uses a locally executable, deterministic approach for answer generation and does not require an external LLM API.

Consequently, `llm_calls.jsonl` is not required.

The pipeline's retrieval, ranking, evaluation, and validation logic are implemented in runnable code.

## 10. Reproducibility and Evaluation

The pipeline is designed to work with equivalent replacement knowledge-base documents and query files following the specified input schema.

It does not rely on:

- Exact sample filenames.
- Fixed document ordering.
- Hardcoded answers to sample questions.
- Precomputed retrieval or evaluation outputs.
- External proprietary knowledge sources.

To reproduce the results, execute:

```bash
python run_pipeline.py
python validate.py
```

---

**Developed by:** Akshay Choudhary

**Purpose:** Deriv AI Engineer — Hands-on Technical Challenge

**Language:** Python
