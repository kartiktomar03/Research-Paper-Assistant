## Separate paper workspaces (new)

- `/` is the **single-paper workspace**: choose any paper from the library dropdown. Individual summaries, figures/tables and streamed Q&A use that selected paper ID.
- `/compare` is a **dedicated comparison page**: select 2–5 papers, stream cross-paper comparisons, and permanently delete individual papers from the library.
- `GET /api/paper/<paper_id>` returns the persisted selected paper's display details (not full-text) so older uploads can be reopened.
- Deleting a paper removes its indexed vectors, uploaded PDF, extracted figures and metadata; this action cannot be undone.
- Existing multi-file upload, single-paper streaming, and comparison endpoints remain unchanged. No authentication is provided; do not expose this demo on the public internet.

## Multi-paper comparison (new)

- Select up to **5 PDF files** in the upload picker. The browser uploads/indexes them sequentially, preserving successes if one file fails.
- Select **2–5 papers** in the **Compare research papers** panel and ask a comparison question.
- Comparison uses Qdrant retrieval filtered separately by each paper ID, adds per-paper metadata and key sections, then streams a Groq answer with paper-wise page sources.
- Single-paper summary and Q&A continue to work with the most recently uploaded PDF.
- The catalog persists on disk in `data/papers`. `GET /api/papers` lists available paper IDs and titles. `POST /api/compare/stream` accepts `{ "paper_ids": ["uuid1", "uuid2"], "message": "..." }`.
- **Title spacing fix:** the first-page title is reconstructed from PyMuPDF word coordinates, not artificially space-joined text spans. Tested with the included *Sufficient Context* PDF.
- Uploaded PDFs, Qdrant index, `.env` and virtual environments are **not shipped** in this clean ZIP. Re-upload PDFs on first start and reuse your private `.env` and `.venv` locally.
- Full end-to-end Groq/Qdrant comparison requires live credentials and local dependencies; syntax tests alone cannot validate answer quality.

# Research Paper Assistant — organized frontend + backend

## Project layout

```text
Research-Paper-Assistant/
├── app.py                     # Start here: python app.py
├── backend/
│   ├── app.py                 # Flask API and routes
│   └── src/
│       ├── document_pipeline.py  # PDF text/tables/figures/metadata
│       ├── rag.py             # Qdrant retrieval + lexical reranking
│       └── llm.py             # Groq prompt handling
├── frontend/
│   ├── templates/index.html
│   └── static/{css/style.css,js/app.js}
├── tests/                     # Includes provided sample PDF
├── data/                      # Generated during runtime
├── uploads/                   # Generated during runtime
├── requirements.txt
└── .env.example
```

## Start in Windows VS Code

Open the **project root** (the folder containing `app.py`). Move your existing `.venv` into this root, or create a new virtual environment. Keep your `.env` in the root; never share it.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. Upload PDFs **again** after upgrading: the old Qdrant local index is intentionally not bundled, and re-indexing uses the improved extraction pipeline.

## Improvements and caveats

- Tables: line-based + text-based detection, spatial deduplication, and table-caption validation to filter false positives. Extremely irregular/merged/scanned tables still need manual checks or OCR.
- Figures: crop rendered page regions near actual Figure captions (works with vectors), with raster fallback on uncaptioned pages. These are image crops, not guaranteed semantic segmentation; visually review crops.
- Metadata: first-page title and author heuristic + abstract extraction. Verify unusual author layouts.
- RAG: remove strict score cutoff, combine semantic results with keyword overlap, inject metadata for author/abstract questions and structured tables for table questions. Q&A requires Groq credentials and re-indexed PDFs.
- `GROQ_VISION_MODEL` remains optional. Without a working vision model, figures can be viewed, but the assistant **cannot reliably interpret numeric chart values**.
- The included PDF is a test fixture from the provided archive; remove `tests/sample_research_paper.pdf` if you do not want to keep it.

## Limitations and test status

Extraction was smoke-tested using the included research PDF. Python source was syntax-checked. Full Groq and Qdrant integration was not exercised in this isolated environment, so check Q&A with your API key and embedding model locally. No 100% table/figure accuracy is claimed.

---

# Research Paper Assistant — Qdrant RAG

A Flask-based research-paper assistant that extracts PDF text, tables, and embedded figures, indexes the content in Qdrant, and answers questions with retrieved source references.

## Features

- PDF upload validation and configurable upload-size limit.
- Page-aware text extraction with PyMuPDF.
- Table extraction with `pdfplumber`, rendered as Markdown and indexed for retrieval.
- Embedded figure extraction with captions/page references. Optional figure descriptions through a Groq vision model when `GROQ_VISION_MODEL` is configured.
- Sentence Transformers embeddings (`all-MiniLM-L6-v2`) and Qdrant vector search.
- Document isolation: every indexed chunk includes a unique `paper_id`, and retrieval filters by that ID.
- Section summaries and grounded Q&A with source/page/table/figure labels.
- Local embedded Qdrant mode for development; Docker Compose Qdrant server mode for a more realistic deployment.
- Delete endpoint removes the paper record, uploaded PDF, extracted figures, and its vector points.

## Important limitations

- This is a strong development/demo baseline, not a security-audited multi-tenant production service. It has no user accounts, ownership/authorization model, malware scanning, or distributed job queue. Anyone who obtains a paper ID can query/delete that document; add authentication and ownership checks before exposing it publicly.
- Scanned PDFs need OCR before text/table extraction. Current pipeline detects empty text and reports that OCR may be required; it does not run OCR automatically.
- PDF table extraction depends on layout and can fail on complex merged cells, rotated tables, or image-only tables.
- Figure extraction detects embedded raster images, not every vector drawing/chart. Optional vision descriptions are best-effort and must be verified for scientific accuracy. Exact numeric chart data should not be inferred from pixels.
- Qdrant is a vector database, not a PDF parser. Extraction and retrieval are separate pipeline stages.
- Section detection uses common heading patterns and falls back to a full-paper section when it cannot detect headings.
- Retrieval score threshold needs tuning on a representative test set; similarity scores are not calibrated probabilities.

## Requirements

- Python 3.11 recommended.
- A Groq API key for summaries and answer generation.
- First run needs network access to download the embedding model unless cached locally.

## Local setup (embedded Qdrant)

1. Create and activate a virtual environment.

   **Windows PowerShell**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

   **macOS/Linux**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

2. Add your real `GROQ_API_KEY` in `.env`. Do not commit `.env`.
3. Run the application:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000`.

By default, Qdrant uses persistent local storage at `./data/qdrant`. Do not run multiple app processes against the same embedded Qdrant path. For multiple workers or deployment, use a Qdrant server or Qdrant Cloud.

## Docker Compose (Qdrant server + Flask)

1. Copy `.env.example` to `.env` and add `GROQ_API_KEY`.
2. Start the services:
   ```bash
   docker compose up --build
   ```
3. Open `http://localhost:5000`; Qdrant dashboard/API is at `http://localhost:6333`.

The Compose setup overrides `QDRANT_URL` to `http://qdrant:6333`. The app uses one Gunicorn worker to avoid concurrent duplicate model loading; scale only after considering model memory, storage, request processing, and Qdrant server behavior.

## Optional figure understanding

Set `GROQ_VISION_MODEL` to a vision model currently available to your Groq account. Model names and availability change, so verify against the provider's current model list. If not configured or unavailable, images and captions are still extracted; the upload does not fail just because vision descriptions are unavailable.

## API overview

- `GET /api/health` — service health and vector backend mode.
- `POST /api/upload` — multipart form field `file` containing a PDF.
- `POST /api/summary` — JSON: `{"paper_id":"...","topic":"..."}`.
- `POST /api/chat` — JSON: `{"paper_id":"...","message":"..."}`; returns answer and retrieved sources.
- `GET /api/figure/<paper_id>/<figure_name>` — serves an extracted figure belonging to the paper path.
- `DELETE /api/paper/<paper_id>` — deletes the paper and indexed vectors.

## Suggested engineering next steps

1. Add authentication and document ownership checks before public deployment.
2. Add OCR fallback for scanned PDFs (for example, Tesseract or a document parsing service).
3. Add automated tests for cross-document retrieval isolation, malformed uploads, PDF table fixtures, and deletion.
4. Add background jobs for large PDFs and enforce API rate limits.
5. Evaluate recall@k and answer faithfulness on a small labelled set before adjusting chunking or retrieval thresholds.
6. Add citations linking to actual PDF page previews and use a more layout-aware parser for complex tables.

## License / attribution

This repository is based on the supplied `Research-Paper-Analysis-Project` source. Review the upstream repository's license and retain attribution/comply with it before publishing a modified version. Document your own changes accurately on your resume.

## Live streaming output

The UI uses `fetch()` + Server-Sent Events over POST to show real Groq token deltas as they arrive. New endpoints: `POST /api/summary/stream` with `{ "paper_id": "...", "topic": "..." }` and `POST /api/chat/stream` with `{ "paper_id": "...", "message": "..." }`. Each frame is `event: delta` (`{"text":"..."}`), `event: done` (sources for chat), or `event: error`. Existing `/api/summary` and `/api/chat` JSON routes remain available. Streaming requires a valid Groq key, access to Groq, and proxy buffering disabled. If your hosting proxy buffers responses, disable buffering for these routes.


---

## Free deployment on Render (Docker + Qdrant Cloud)

This setup is a **temporary public demo**, not permanent document storage. The
existing local Python workflow (`requirements.txt`, `.venv`, `python app.py`)
stays unchanged. `Dockerfile` installs `requirements.render.txt` instead of
PyTorch and runs `FastEmbed` (ONNX), with the same 384-dimension MiniLM model.
This reduces memory usage but **does not guarantee** a successful run in
Render's 512 MB free instance; large PDFs can still exceed memory/time limits.

1. Remove `.env`, `.venv`, personal PDFs and any API keys from the project before
   putting it on GitHub. `.gitignore` excludes generated artifacts.
2. Create a **Qdrant Cloud free cluster**, then copy its cluster URL and API key.
3. Push the project to GitHub. On [Render](https://dashboard.render.com), choose
   **New → Web Service**, connect the repository, choose **Docker** and **Free**.
   The `render.yaml` also documents the recommended settings.
4. In Render's **Environment** tab, set `GROQ_API_KEY`, `QDRANT_URL`, and
   `QDRANT_API_KEY` as secret environment variables (never commit real keys).
   `LLM_MODEL` must be a Groq model actually enabled for your account.
   Set `EMBEDDING_BACKEND=fastembed`, `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`,
   `EMBEDDING_DIMENSION=384`. The Dockerfile and render.yaml provide these defaults.
5. Deploy. Visit `/api/health`, then upload a **small text-based PDF** and check
   its single-paper summary, Q&A streaming, figure previews and `/compare`.

**IMPORTANT — Render Free loses document data.** Render Free's filesystem is
ephemeral and can be wiped after 15 minutes of inactivity, restart or redeploy.
Qdrant Cloud preserves **vectors**, but this app saves **paper JSON, titles,
sections, figures and PDFs locally** in `data/` and `uploads/`. After a restart,
the paper list will be empty and comparison cannot find old paper records, even
if vectors are still in Qdrant. Re-upload documents after a restart. To make
papers truly persistent, add an external persistent object/metadata store and
adapt the backend; Qdrant Cloud alone is not enough.

**Embedding note:** ONNX FastEmbed and local PyTorch MiniLM may give slightly
different embeddings. Re-index documents when changing backends; do not assume
that an existing cloud collection is compatible just because dimensions match.
For isolation, create a fresh Qdrant collection for the deployed app (e.g. set
`QDRANT_COLLECTION=research_paper_render` in Render environment).

**Privacy/security:** the current app does not require user login. Anyone with
the public URL can upload documents, list papers, delete documents and trigger
Groq API usage. Only upload non-sensitive, publicly shareable test PDFs; avoid
sharing the link widely until authentication and abuse protection are added.
`MAX_UPLOAD_MB=10` in the Render config limits a single upload but is not a
complete abuse prevention measure.

**If the deployment crashes:** inspect Render logs. A 512 MB out-of-memory
error cannot be fixed by editing README; try smaller PDFs or a more capable
service. Initial embedding model download/cold start can be slow.
