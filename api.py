import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.constants import KB_DIR
from src.generator import generate_answer_for_query
from src.ingestion import chunk_documents_sentence_line, load_documents
from src.retriever import BM25Retriever

# Build index on startup from kb/
_DOCUMENTS = load_documents(KB_DIR)
_CHUNKS = chunk_documents_sentence_line(_DOCUMENTS)
_RETRIEVER = BM25Retriever()
_RETRIEVER.build_index(_CHUNKS)


class AnswerHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/answer":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            question = payload.get("question", "").strip()
            if not question:
                raise ValueError("Missing 'question' field.")
        except Exception as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        top_k = _RETRIEVER.retrieve(question, top_k=3)
        result = generate_answer_for_query(
            {"query_id": "API_QUERY", "question": question, "top_k": top_k}
        )
        response_body = {
            "answer_label": result["answer_label"],
            "answer": result["answer"],
            "citations": result["citations"],
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_body, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), AnswerHandler)
    print("Serving POST http://127.0.0.1:8000/answer ...")
    server.serve_forever()