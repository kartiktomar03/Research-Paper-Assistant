"""Qdrant vector indexing and document-filtered retrieval."""
from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

class _FastEmbedAdapter:
    """Small ONNX embedding backend for memory-constrained free hosting.

    Exposes the two SentenceTransformer methods used by this project, leaving
    all indexing/retrieval and API logic unchanged.
    """

    def __init__(self, name: str, dimension: int):
        from fastembed import TextEmbedding
        import os
        self._model = TextEmbedding(
            model_name=name,
            cache_dir=os.getenv("FASTEMBED_CACHE_DIR") or None,
            threads=1,
        )
        self._dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        single = isinstance(texts, str)
        docs = [texts] if single else list(texts)
        if not docs:
            return np.empty((0, self._dimension), dtype=np.float32)
        vectors = np.asarray(list(self._model.embed(docs, batch_size=8)), dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors[0] if single else vectors



class RAGService:
    def __init__(self):
        self.collection = os.getenv("QDRANT_COLLECTION", "research_paper_chunks")
        self.qdrant_url = os.getenv("QDRANT_URL", "").strip()
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        self.storage_path = os.getenv("QDRANT_LOCAL_PATH", str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "qdrant")).strip()
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._client = None
        # Local development defaults to the original Sentence Transformers backend.
        # Render Docker uses the lighter FastEmbed/ONNX backend.
        self.embedding_backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
        self._embedder = None
        self.vector_size = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    @property
    def backend_name(self):
        return "qdrant-server" if self.qdrant_url else "qdrant-local"

    @property
    def client(self):
        if self._client is None:
            if self.qdrant_url:
                self._client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=30)
            else:
                os.makedirs(self.storage_path, exist_ok=True)
                self._client = QdrantClient(path=self.storage_path)
            if not self._client.collection_exists(self.collection):
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
                )
                # Index fields used for document isolation and filtering.
                for field in ("paper_id", "content_type", "page"):
                    try:
                        self._client.create_payload_index(
                            collection_name=self.collection,
                            field_name=field,
                            field_schema=models.PayloadSchemaType.KEYWORD if field != "page" else models.PayloadSchemaType.INTEGER,
                        )
                    except Exception:
                        pass
        return self._client

    @property
    def embedder(self):
        if self._embedder is None:
            if self.embedding_backend == "fastembed":
                self._embedder = _FastEmbedAdapter(self.embedding_model_name, self.vector_size)
            elif self.embedding_backend == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name)
            else:
                raise ValueError(f"Unknown EMBEDDING_BACKEND: {self.embedding_backend}")
            self.vector_size = self._embedder.get_sentence_embedding_dimension()
        return self._embedder

    def _ensure_collection_dimension(self):
        # Initialize embeddings before creating collection to use the model's actual dimension.
        dimension = self.embedder.get_sentence_embedding_dimension()
        if self._client is None:
            if self.qdrant_url:
                self._client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=30)
            else:
                os.makedirs(self.storage_path, exist_ok=True)
                self._client = QdrantClient(path=self.storage_path)
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(collection_name=self.collection, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE))
            for field in ("paper_id", "content_type", "page"):
                try:
                    self._client.create_payload_index(collection_name=self.collection, field_name=field,
                        field_schema=models.PayloadSchemaType.INTEGER if field == "page" else models.PayloadSchemaType.KEYWORD)
                except Exception:
                    pass
        else:
            info = self._client.get_collection(self.collection)
            existing = info.config.params.vectors.size
            if existing != dimension:
                raise RuntimeError(f"Qdrant collection dimension is {existing}, but embedding model outputs {dimension}. Use a new collection name or matching embedding model.")
        return self._client

    def index_paper(self, paper_id: str, chunks: list[dict[str, Any]]) -> dict[str, int]:
        client = self._ensure_collection_dimension()
        self.delete_paper(paper_id)
        valid_chunks = [chunk for chunk in chunks if (chunk.get("text") or "").strip()]
        if not valid_chunks:
            raise ValueError("No indexable text, table, or figure content was extracted.")
        texts = [chunk["text"][:5000] for chunk in valid_chunks]
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        points = []
        for chunk, vector in zip(valid_chunks, vectors):
            payload = {
                "paper_id": paper_id,
                "text": chunk["text"][:5000],
                "content_type": chunk.get("content_type", "text"),
                "page": int(chunk.get("page", 1)),
                "label": chunk.get("label", ""),
                "caption": chunk.get("caption", ""),
            }
            points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
        # Batch upserts avoid a single huge request for long PDFs.
        for start in range(0, len(points), 64):
            client.upsert(collection_name=self.collection, points=points[start:start + 64], wait=True)
        return {"chunk_count": len(points)}

    def delete_paper(self, paper_id: str) -> None:
        # Don't initialize/download embeddings just to delete points. If no client/collection exists, there is nothing to delete.
        if self._client is None:
            return
        try:
            if self._client.collection_exists(self.collection):
                self._client.delete(collection_name=self.collection, points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))])
                ), wait=True)
        except Exception:
            # Cleanup should not mask the primary upload error.
            pass

    def retrieve_context(self, paper_id: str, question: str, paper: dict | None = None) -> tuple[str, list[dict[str, Any]]]:
        client = self._ensure_collection_dimension()
        vector = self.embedder.encode(question, normalize_embeddings=True, show_progress_bar=False).tolist()
        # Combine vector ranking with term overlap; don't discard legitimate low-score
        # scientific queries due to a poorly calibrated global similarity threshold.
        response = client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))]),
            limit=20, with_payload=True,
        )
        import re
        stopwords = {"the", "for", "and", "this", "that", "with", "from", "what", "which", "about", "does", "are", "were", "was", "can", "paper", "study", "tell", "give", "please", "explain", "their", "these", "those"}
        tokens = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2 and w not in stopwords}
        scored = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.get("text", "").lower()
            overlap = sum(1 for token in tokens if re.search(r"\b" + re.escape(token) + r"\b", content)) / max(1, len(tokens))
            score = float(point.score or 0)
            scored.append((score + 0.20 * overlap, point))
        scored.sort(key=lambda item: item[0], reverse=True)
        points = [p for _, p in scored[:9]]
        context_parts, sources = [], []
        # Basic bibliographic questions must use first-page metadata, not random
        # embedding matches with 0.2 similarity.
        if paper:
            title = paper.get("title", "")
            authors = paper.get("authors", "")
            abstract = paper.get("abstract", "")
            if any(w in question.lower() for w in ("author", "written by", "title", "abstract", "publication", "researcher")):
                context_parts.append(f"[Paper metadata — Page 1]\nTitle: {title}\nAuthors: {authors or 'Not reliably extracted'}\nAbstract: {abstract or 'Not reliably extracted'}")
                first_page = (paper.get("full_text", "").split("[PAGE 2]")[0])[:5500]
                context_parts.append("[Page 1]\n" + first_page)
                sources.append({"content_type": "metadata", "label": "Page 1 metadata", "page": 1, "score": 1.0, "snippet": f"Title: {title}; Authors: {authors}"[:320]})
            if any(w in question.lower() for w in ("table", "accuracy", "precision", "recall", "f1", "result", "comparison")):
                for table in (paper.get("tables") or [])[:12]:
                    context_parts.append(f"[Table — Page {table.get('page')}]\n{table.get('content', table.get('markdown', ''))[:2600]}")
            if any(w in question.lower() for w in ("figure", "diagram", "architecture", "graph", "chart")):
                for fig in (paper.get("figures") or [])[:12]:
                    context_parts.append(f"[Figure — Page {fig.get('page')}]\n{fig.get('content','')[:1500]}")
        seen = set()
        for point in points:
            payload = point.payload or {}
            kind = payload.get("content_type", "text")
            page = payload.get("page", "?")
            label = payload.get("label", "")
            text = payload.get("text", "")
            marker = (kind, page, label, text[:60])
            if marker in seen:
                continue
            seen.add(marker)
            context_parts.append(f"[{label or kind.title()} — Page {page}]\n{text}")
            sources.append({"content_type": kind, "label": label or f"Page {page}", "page": page, "score": round(float(point.score or 0), 3), "snippet": text[:320]})
        if not context_parts:
            return "", []
        return "\n\n---\n\n".join(context_parts), sources

    def retrieve_comparison_context(self, papers: list[dict], question: str) -> tuple[str, list[dict]]:
        """Retrieve evidence independently from EVERY selected paper.

        This avoids a single paper monopolising nearest-neighbour results.
        """
        contexts, sources = [], []
        per_paper_budget = min(6500, max(2400, 17000 // len(papers)))
        for index, paper in enumerate(papers, 1):
            context, paper_sources = self.retrieve_context(paper["paper_id"], question, paper=paper)
            label = f"Paper {index}: {paper['title']}"
            if not context:
                context = "No indexed evidence available."
            # First-page metadata plus representative methods/results sections
            # make broad cross-paper questions more robust than vector-only search.
            evidence = [f"Title: {paper['title']}",
                        f"Authors: {paper.get('authors') or 'Not reliably extracted'}",
                        f"Abstract: {(paper.get('abstract') or 'Not available')[:1600]}"]
            relevant_sections = [
                (heading, section) for heading, section in (paper.get('sections') or {}).items()
                if any(term in heading.lower() for term in
                       ('method', 'experiment', 'result', 'conclusion', 'limit', 'discussion'))
            ]
            for heading, section in relevant_sections[:3]:
                evidence.append(f"[Section: {heading}]\n{section[:750]}")
            prefix = "\n".join(evidence)
            remaining = max(500, per_paper_budget - len(prefix))
            contexts.append(f"=== {label} ===\n{prefix}\n\n[Retrieved evidence]\n{context[:remaining]}")
            for source in paper_sources[:7]:
                sources.append({**source, "paper_id": paper["paper_id"], "paper_title": paper["title"], "paper_label": f"Paper {index}"})
        return "\n\n".join(contexts), sources

    def answer(self, paper_id: str, question: str, llm_service, paper: dict | None = None) -> tuple[str, list[dict[str, Any]]]:
        context, sources = self.retrieve_context(paper_id, question, paper)
        if not context:
            return "No readable evidence was indexed for that question. Please re-upload the PDF.", []
        return llm_service.answer(question, context), sources
