"""Research Paper Assistant: Flask API with Qdrant-backed RAG."""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename

from backend.src.document_pipeline import process_pdf
from backend.src.rag import RAGService
from backend.src.llm import LLMService

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "uploads"
PAPER_DIR = ROOT / "data" / "papers"
FIGURE_DIR = ROOT / "data" / "figures"
for directory in (UPLOAD_DIR, PAPER_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__, template_folder=str(ROOT / "frontend" / "templates"), static_folder=str(ROOT / "frontend" / "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["JSON_SORT_KEYS"] = False
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Services are stateless at request level; documents are isolated by unguessable paper IDs.
rag_service = RAGService()
llm_service = LLMService()


def paper_path(paper_id: str) -> Path:
    """Resolve only UUID-shaped IDs to prevent path traversal."""
    try:
        safe_id = str(uuid.UUID(paper_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Invalid paper ID")
    return PAPER_DIR / f"{safe_id}.json"


def load_paper(paper_id: str) -> dict[str, Any]:
    path = paper_path(paper_id)
    if not path.exists():
        raise FileNotFoundError("Paper not found. Upload it again to continue.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."}), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/compare")
def compare_page():
    return render_template("compare.html")


@app.get("/api/paper/<paper_id>")
def get_paper_details(paper_id: str):
    """Return the selected paper's UI data without exposing full indexed text."""
    try:
        record = load_paper(paper_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({
        "paper_id": record["paper_id"], "title": record.get("title", ""),
        "filename": record.get("filename", ""),
        "page_count": record.get("page_count", 0),
        "topics": list(record.get("sections", {}).keys()),
        "table_count": len(record.get("tables", [])),
        "figure_count": len(record.get("figures", [])),
        "chunk_count": record.get("chunk_count", 0),
        "tables": [{k: t.get(k) for k in ("table_id", "page", "caption", "markdown", "row_count", "column_count")}
                   for t in record.get("tables", [])],
        "figures": [{k: f.get(k) for k in ("figure_id", "page", "caption", "description", "image_url")}
                    for f in record.get("figures", [])],
    })


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "research-paper-assistant", "vector_store": rag_service.backend_name})


@app.post("/api/upload")
def upload_pdf():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Choose a PDF file to upload."}), 400
    safe_name = secure_filename(uploaded.filename)
    if not safe_name or "." not in safe_name or safe_name.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Only PDF files are supported."}), 400

    paper_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{paper_id}.pdf"
    uploaded.save(pdf_path)
    try:
        # Parse with PyMuPDF first and reject invalid/empty PDFs.
        with fitz.open(pdf_path) as pdf:
            if pdf.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported yet.")
            if len(pdf) == 0:
                raise ValueError("The PDF has no pages.")
        processed = process_pdf(pdf_path, paper_id=paper_id, original_name=safe_name, figure_dir=FIGURE_DIR)
        if not processed["full_text"].strip() and not processed["tables"]:
            raise ValueError("No readable text or tables were found. Scanned PDFs may need OCR first.")
        rag_stats = rag_service.index_paper(paper_id, processed["chunks"])
        record = {
            "paper_id": paper_id,
            "filename": safe_name,
            "title": processed["title"],
            "authors": processed.get("authors", ""),
            "abstract": processed.get("abstract", ""),
            "page_count": processed["page_count"],
            "full_text": processed["full_text"],
            "sections": processed["sections"],
            "tables": processed["tables"],
            "figures": processed["figures"],
            "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "chunk_count": rag_stats["chunk_count"],
        }
        paper_path(paper_id).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        logger.info("Indexed paper %s (%s chunks)", paper_id, rag_stats["chunk_count"])
        return jsonify({
            "paper_id": paper_id,
            "filename": safe_name,
            "title": record["title"],
            "page_count": record["page_count"],
            "topics": list(record["sections"].keys()),
            "table_count": len(record["tables"]),
            "figure_count": len(record["figures"]),
            "chunk_count": record["chunk_count"],
            "figures": [{k: f.get(k) for k in ("figure_id", "page", "caption", "description", "image_url")} for f in record["figures"]],
            "tables": [{k: t.get(k) for k in ("table_id", "page", "caption", "markdown", "row_count", "column_count")} for t in record["tables"]],
            "message": "Paper processed and indexed successfully."
        }), 201
    except Exception as exc:
        logger.exception("PDF processing failed")
        try:
            pdf_path.unlink(missing_ok=True)
            paper_path(paper_id).unlink(missing_ok=True)
            rag_service.delete_paper(paper_id)
        except Exception:
            logger.exception("Cleanup after upload failure failed")
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "Could not process this PDF. Check server logs and configuration."
        return jsonify({"error": message}), 422


@app.get("/api/papers")
def list_papers():
    """Lightweight catalog. Do not return extracted full text here."""
    catalog = []
    for path in PAPER_DIR.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            catalog.append({key: record.get(key) for key in
                            ("paper_id", "title", "filename", "page_count", "created_at")})
        except (ValueError, OSError):
            logger.warning("Ignoring unreadable paper metadata file %s", path.name)
    catalog.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return jsonify({"papers": catalog})


def selected_papers(data):
    ids = data.get("paper_ids")
    if not isinstance(ids, list) or not 2 <= len(ids) <= 5:
        raise ValueError("Select between 2 and 5 papers for comparison.")
    if len(ids) != len(set(ids)):
        raise ValueError("Choose distinct papers for comparison.")
    return [load_paper(pid) for pid in ids]


@app.post("/api/compare")
def compare_papers():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question or len(question) > 4000:
        return jsonify({"error": "Enter a comparison question (maximum 4000 characters)."}), 400
    try:
        papers = selected_papers(data)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        context, sources = rag_service.retrieve_comparison_context(papers, question)
        return jsonify({"response": llm_service.compare(question, context), "sources": sources})
    except Exception:
        logger.exception("Comparison failed")
        return jsonify({"error": "Comparison failed. Check Qdrant and Groq configuration."}), 503


@app.post("/api/compare/stream")
def compare_papers_stream():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question or len(question) > 4000:
        return jsonify({"error": "Enter a comparison question (maximum 4000 characters)."}), 400
    try:
        papers = selected_papers(data)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400

    def events():
        try:
            context, sources = rag_service.retrieve_comparison_context(papers, question)
            for delta in llm_service.compare_stream(question, context):
                yield sse_event("delta", {"text": delta})
            yield sse_event("done", {"sources": sources})
        except Exception:
            logger.exception("Streaming comparison failed")
            yield sse_event("error", {"error": "Comparison failed. Check server logs and Groq settings."})
    return streamed_response(events())


@app.post("/api/summary")
def summary():
    data = request.get_json(silent=True) or {}
    paper_id, topic = data.get("paper_id"), data.get("topic")
    if not paper_id or not topic:
        return jsonify({"error": "paper_id and topic are required."}), 400
    try:
        paper = load_paper(paper_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    topic_text = paper["sections"].get(topic)
    if not topic_text:
        return jsonify({"error": "That section was not found in this paper."}), 404
    try:
        result = llm_service.summarize(topic_text, title=topic)
        return jsonify({"summary": result, "topic": topic})
    except Exception:
        logger.exception("Summary generation failed")
        return jsonify({"error": "Summary generation failed. Check GROQ_API_KEY and model configuration."}), 503


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    paper_id, message = data.get("paper_id"), (data.get("message") or "").strip()
    if not paper_id or not message:
        return jsonify({"error": "paper_id and message are required."}), 400
    if len(message) > 4000:
        return jsonify({"error": "Question is too long (maximum 4000 characters)."}), 400
    try:
        paper = load_paper(paper_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    try:
        answer, sources = rag_service.answer(paper_id, message, llm_service, paper=paper)
        return jsonify({"response": answer, "sources": sources})
    except Exception:
        logger.exception("RAG chat failed")
        return jsonify({"error": "Question answering failed. Check API key, Qdrant and model configuration."}), 503


def sse_event(event: str, data: dict) -> str:
    """SSE frame; JSON encoding keeps multiline content and Unicode safe."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def streamed_response(events):
    return Response(
        stream_with_context(events), mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/summary/stream")
def summary_stream():
    data = request.get_json(silent=True) or {}
    paper_id, topic = data.get("paper_id"), data.get("topic")
    if not paper_id or not topic:
        return jsonify({"error": "paper_id and topic are required."}), 400
    try:
        paper = load_paper(paper_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    section = paper["sections"].get(topic)
    if not section:
        return jsonify({"error": "That section was not found in this paper."}), 404

    def events():
        try:
            for token in llm_service.summarize_stream(section, title=topic):
                yield sse_event("delta", {"text": token})
            yield sse_event("done", {"topic": topic})
        except Exception:
            logger.exception("Streaming summary failed")
            yield sse_event("error", {"error": "Summary generation failed. Check the server logs and GROQ_API_KEY."})

    return streamed_response(events())


@app.post("/api/chat/stream")
def chat_stream():
    data = request.get_json(silent=True) or {}
    paper_id, message = data.get("paper_id"), (data.get("message") or "").strip()
    if not paper_id or not message:
        return jsonify({"error": "paper_id and message are required."}), 400
    if len(message) > 4000:
        return jsonify({"error": "Question is too long (maximum 4000 characters)."}), 400
    try:
        paper = load_paper(paper_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    def events():
        try:
            context, sources = rag_service.retrieve_context(paper_id, message, paper=paper)
            if not context:
                yield sse_event("delta", {"text": "No readable evidence was indexed for that question. Please re-upload the PDF."})
            else:
                for token in llm_service.answer_stream(message, context):
                    yield sse_event("delta", {"text": token})
            yield sse_event("done", {"sources": sources})
        except Exception:
            logger.exception("Streaming chat failed")
            yield sse_event("error", {"error": "Question answering failed. Check the server logs, Qdrant and GROQ_API_KEY."})

    return streamed_response(events())


@app.get("/api/figure/<paper_id>/<figure_name>")
def get_figure(paper_id: str, figure_name: str):
    try:
        paper_path(paper_id)
    except ValueError:
        return jsonify({"error": "Invalid paper ID"}), 400
    if not re.fullmatch(r"[a-f0-9-]+_[0-9]+\.(?:png|jpg|jpeg)", figure_name):
        return jsonify({"error": "Invalid figure name"}), 400
    folder = FIGURE_DIR / paper_id
    return send_from_directory(folder, figure_name, as_attachment=False)


@app.delete("/api/paper/<paper_id>")
def delete_paper(paper_id: str):
    try:
        path = paper_path(paper_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not path.exists():
        return jsonify({"error": "Paper not found."}), 404
    try:
        paper = load_paper(paper_id)
        rag_service.delete_paper(paper_id)
        path.unlink(missing_ok=True)
        (UPLOAD_DIR / f"{paper_id}.pdf").unlink(missing_ok=True)
        shutil_path = FIGURE_DIR / paper_id
        import shutil
        shutil.rmtree(shutil_path, ignore_errors=True)
        return jsonify({"deleted": True, "paper_id": paper_id})
    except Exception:
        logger.exception("Delete failed")
        return jsonify({"error": "Could not delete this paper."}), 500


if __name__ == "__main__":
    # Use Flask's development server only for local development. Deploy with gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
