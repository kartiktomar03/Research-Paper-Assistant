# Render Free / small CPU containers: use FastEmbed (ONNX) rather than PyTorch.
# Local development is unchanged: requirements.txt + python app.py.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_BACKEND=fastembed \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    EMBEDDING_DIMENSION=384 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=10000
WORKDIR /app
COPY requirements.render.txt ./
RUN pip install --no-cache-dir -r requirements.render.txt
COPY . .
RUN mkdir -p uploads data/papers data/figures data/qdrant
EXPOSE 10000
# A single process avoids loading the same embedding model in multiple workers.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 240 --graceful-timeout 30 app:app"]
