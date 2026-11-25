import json
import re
from pathlib import Path
from typing import Any


def parse_document(file_path: Path) -> dict[str, str]:
    """
    Parses a KB plain-text file with Title: and Section: headers.
    Preserves the document body text exactly as written after the header block.
    """
    raw_text = file_path.read_text(encoding="utf-8-sig")
    lines = raw_text.splitlines(keepends=True)

    doc_title = ""
    section = ""
    body_start_line_idx = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Title:"):
            doc_title = stripped[len("Title:") :].strip()
        elif stripped.startswith("Section:"):
            section = stripped[len("Section:") :].strip()
        elif stripped == "" and doc_title and section:
            # First blank line after headers marks the start of the body
            body_start_line_idx = idx + 1
            break
        elif doc_title and section:
            # Handle files with no blank line between Section: and body
            body_start_line_idx = idx
            break

    if not doc_title:
        doc_title = file_path.stem

    body = "".join(lines[body_start_line_idx:])
    return {
        "source_file": file_path.name,
        "doc_title": doc_title,
        "section": section,
        "body": body,
    }


def load_documents(kb_dir: Path) -> list[dict[str, str]]:
    """
    Loads all .txt files in kb_dir sorted alphabetically for determinism.
    Does not rely on specific filenames or document counts.
    """
    if not kb_dir.exists() or not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    files = sorted(kb_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in {kb_dir}")

    return [parse_document(f) for f in files]


def chunk_documents_sentence_line(documents: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Primary chunking strategy: splits body into atomic line/sentence units
    while preserving exact substrings and start_char / end_char offsets
    relative to the document body.
    """
    chunks: list[dict[str, Any]] = []
    chunk_counter = 1

    for doc in documents:
        body = doc["body"]
        # Match non-empty lines (each policy rule/sentence in KB docs)
        for match in re.finditer(r"[^\r\n]+", body):
            line_text = match.group(0).strip()
            if not line_text:
                continue

            # Compute exact character span of the trimmed text inside body
            leading_ws = len(match.group(0)) - len(match.group(0).lstrip())
            start_char = match.start() + leading_ws
            end_char = start_char + len(line_text)

            exact_slice = body[start_char:end_char]
            assert exact_slice == line_text, f"Offset mismatch: {exact_slice!r} != {line_text!r}"

            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_counter}",
                    "doc_title": doc["doc_title"],
                    "section": doc["section"],
                    "text": exact_slice,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )
            chunk_counter += 1

    return chunks


def chunk_documents_fixed_size(
    documents: list[dict[str, str]], window_size: int = 120, overlap: int = 20
) -> list[dict[str, Any]]:
    """
    Secondary chunking strategy (for Should-Attempt #6 comparison):
    Fixed-size character window with overlap, preserving exact body slices.
    """
    chunks: list[dict[str, Any]] = []
    chunk_counter = 1
    step = max(1, window_size - overlap)

    for doc in documents:
        body = doc["body"]
        n = len(body)
        pos = 0
        while pos < n:
            raw_end = min(n, pos + window_size)
            slice_text = body[pos:raw_end]
            stripped = slice_text.strip()
            if stripped:
                leading_ws = len(slice_text) - len(slice_text.lstrip())
                start_char = pos + leading_ws
                end_char = start_char + len(stripped)
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_counter}",
                        "doc_title": doc["doc_title"],
                        "section": doc["section"],
                        "text": body[start_char:end_char],
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                )
                chunk_counter += 1
            if raw_end == n:
                break
            pos += step

    return chunks


def save_chunks(chunks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)