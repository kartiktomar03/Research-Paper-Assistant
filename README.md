# ResearchLab — AI-Powered Research Paper Assistant

An AI-powered research assistant that helps users analyze academic papers, extract tables and figures, generate summaries, and compare multiple research papers using Retrieval-Augmented Generation (RAG).

Built with **Flask, Python, Qdrant, Groq, Sentence Transformers, and FastEmbed**.

### Live Demo

[Launch ResearchLab](https://research-paper-assistant-a639.onrender.com)

> **Deployment note:** The application is hosted on Render's free tier. Initial loading and PDF indexing may be slow, and large PDF uploads may time out due to limited server resources. Uploaded documents are not permanently stored on Render.

## Overview

Reading and comparing research papers can be time-consuming, especially when important information is distributed across complex sections, tables, and figures.

ResearchLab aims to simplify this workflow through document-aware retrieval and LLM-powered analysis.

Users can upload research papers, explore their contents, ask natural-language questions, and compare findings across multiple documents.

## Key Features

### 1. Research Paper Processing

- Upload up to five PDF files through the interface.
- Extract page-aware text and document metadata.
- Identify research paper titles, authors, and abstracts using PDF parsing heuristics.
- Organize extracted content for retrieval and analysis.

### 2. AI-Powered Summarization

- Generate summaries of research paper sections.
- Explore methodologies, findings, conclusions, and limitations.
- Stream generated responses directly to the interface.

### 3. Document Question Answering

- Ask questions about an individual research paper.
- Retrieve relevant document chunks using semantic search.
- Generate context-grounded answers with source/page references.
- Stream responses using Server-Sent Events.

### 4. Multi-Paper Comparison

- Select two to five research papers for cross-document analysis.
- Compare research methodologies, findings, limitations, and experimental results.
- Retrieve relevant evidence separately from each selected document.
- Display paper-wise sources in generated responses.

### 5. Table and Figure Extraction

- Detect structured and borderless tables using multiple extraction strategies.
- Convert detected tables into a structured Markdown representation.
- Extract figures using rendered PDF regions and caption-based detection.
- Preserve available page references and captions.

_Complex tables, scanned pages, and figure boundaries may still require manual verification._

### 6. Research Paper Library

- Select previously uploaded papers for individual analysis.
- Navigate between single-paper and comparison workspaces.
- Delete papers and their associated indexed records.

## Technology Stack

| Layer           | Technologies                      |
| --------------- | --------------------------------- |
| Frontend        | HTML, CSS, JavaScript             |
| Backend         | Python, Flask                     |
| LLM Integration | Groq API                          |
| Vector Database | Qdrant                            |
| Embeddings      | Sentence Transformers / FastEmbed |
| PDF Processing  | PyMuPDF, pdfplumber               |
| Streaming       | Server-Sent Events (SSE)          |
| Deployment      | Docker, Render                    |
| Version Control | Git, GitHub                       |

## System Architecture

```text
                 Research Paper PDFs
                         |
                         v
                  Flask Backend
                         |
                         v
                 PDF Processing
                /       |        \
             Text     Tables    Figures
                \       |        /
                         v
                 Chunk Generation
                         |
                         v
                  Text Embeddings
                         |
                         v
                    Qdrant DB
                         |
                 Semantic Retrieval
                         |
                         v
                     Groq LLM
                         |
                         v
            Streaming Response + Sources
                         |
                         v
                   ResearchLab UI
```

For multi-paper comparison, relevant chunks are retrieved separately for each selected paper before being passed to the language model.

## Project Structure

```text
Research-Paper-Assistant/
|
|-- app.py
|-- backend/
|   |-- app.py
|   `-- src/
|       |-- document_pipeline.py
|       |-- rag.py
|       `-- llm.py
|
|-- frontend/
|   |-- templates/
|   `-- static/
|       |-- css/
|       `-- js/
|
|-- tests/
|-- data/
|-- uploads/
|-- Dockerfile
|-- docker-compose.yml
|-- render.yaml
|-- requirements.txt
|-- requirements.render.txt
|-- .env.example
`-- README.md
```

## Local Installation

### Prerequisites

- Python 3.11 recommended
- Git
- Groq API key
- Internet connection for initial model download

### 1. Clone the repository

```bash
git clone https://github.com/kartiktomar03/Research-Paper-Assistant.git
cd Research-Paper-Assistant
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

For local testing:

```env
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-120b

EMBEDDING_BACKEND=sentence-transformers

QDRANT_URL=
QDRANT_API_KEY=
```

Leaving `QDRANT_URL` empty enables local embedded Qdrant mode.

Never commit `.env` or actual API keys.

### 5. Start the application

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

The comparison workspace is available at:

`http://127.0.0.1:5000/compare`

## Deployment

The project includes a Docker-based configuration for Render.

The hosted version uses:

- **Render:** Flask application hosting
- **Qdrant Cloud:** Vector storage
- **Groq:** LLM inference
- **FastEmbed:** Lightweight ONNX-based embeddings

Required deployment environment variables include:

```env
GROQ_API_KEY=your_api_key
LLM_MODEL=openai/gpt-oss-120b

QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=research_paper_render

EMBEDDING_BACKEND=fastembed
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

### Free Hosting Limitations

Render Free has limited memory and CPU resources. Large research papers can result in slow indexing, request timeouts, or temporary service unavailability.

Uploaded PDFs, extracted figures, and paper metadata are stored on the instance's local filesystem, which is not persistent across restarts or redeployments. Qdrant Cloud vectors may remain even when local paper records are lost.

The current deployment is intended as a demonstration, not a production-ready document storage service.

## API Overview

| Endpoint                       | Purpose                            |
| ------------------------------ | ---------------------------------- |
| `GET /api/health`              | Application health status          |
| `GET /api/papers`              | List indexed papers                |
| `POST /api/upload`             | Upload and process a PDF           |
| `GET /api/paper/<paper_id>`    | Retrieve paper details             |
| `POST /api/summary/stream`     | Stream section summary             |
| `POST /api/chat/stream`        | Stream document Q&A                |
| `POST /api/compare/stream`     | Stream multi-paper comparison      |
| `DELETE /api/paper/<paper_id>` | Delete a paper and associated data |

## Current Limitations

- Complex merged-cell and borderless tables may not always be reconstructed accurately.
- Figure extraction can occasionally include surrounding content.
- Scanned PDFs require additional OCR support.
- LLM-generated information should be checked against the original paper.
- Public deployment currently lacks user authentication and document ownership controls.
- Render Free performance is limited for compute-intensive PDF processing.

## Future Improvements

- [ ] Background PDF processing with progress tracking
- [ ] Improved document-layout recognition
- [ ] OCR support for scanned research papers
- [ ] Session-based document isolation and automatic cleanup
- [ ] Faster embedding and indexing pipeline
- [ ] Persistent document storage
- [ ] Authentication and API rate limiting
- [ ] Improved extraction accuracy evaluation

## Testing

The repository includes tests for document processing, extraction logic, and streaming behavior.

```bash
python -m pytest
```

Additional end-to-end testing with live Groq and Qdrant Cloud services is needed, particularly under free hosting resource constraints.

## Attribution

This project was developed further from the supplied Research-Paper-Analysis-Project codebase. Upstream licensing and attribution requirements should be reviewed and preserved where applicable.

## Disclaimer

ResearchLab is designed as an academic research assistance tool. Generated summaries, comparisons, extracted numerical values, and citations should be verified against the original research papers before academic or professional use.

## Project Showcase

Explore the project's features, architecture,
screenshots, and local setup instructions.

[Visit ResearchLab Showcase](https://kartiktomar03.github.io/ResearchLab-Showcase/)
