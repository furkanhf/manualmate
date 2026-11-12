# ManualMate on Hugging Face Spaces (Docker SDK, free CPU: 2 vCPU, 16 GB RAM).
#
# One container: FastAPI (/ask, /health, /docs) with the Gradio chat UI at "/",
# on port 7860. Real search: bge-m3 + bge-reranker-v2-m3 on CPU.
# Secrets come from the Space settings (never from the image):
#   DATABASE_URL    Neon connection string
#   GEMINI_API_KEY  Gemini API key
#
# Build on HF (no Docker on the laptop). The first build takes ~15-25 min
# (torch + 2 models, ~4.6 GB); later builds reuse the cached layers when only
# src/ changes.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONPATH=/app/src \
    MANUALMATE_SEARCH=full \
    GRADIO_ANALYTICS_ENABLED=False \
    TOKENIZERS_PARALLELISM=false

# HF Spaces runs the container as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# 1. torch, CPU build (the biggest layer; it changes rarely, so it stays cached).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -c "import torch; print('torch', torch.__version__)"

# 2. The app libraries.
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# 3. The models, baked into the image: the Space starts without downloading
#    4.6 GB each time it wakes up. Only the PyTorch weights + tokenizer files
#    (bge-m3 also ships ONNX and ColBERT/sparse weights that we do not use).
RUN python -c "\
from huggingface_hub import snapshot_download as get; \
skip = ['onnx/*', '*.onnx', 'colbert_linear.pt', 'sparse_linear.pt', 'imgs/*', '*.md']; \
get('BAAI/bge-m3', ignore_patterns=skip); \
get('BAAI/bge-reranker-v2-m3', ignore_patterns=skip); \
print('models downloaded')"

# 4. The code (last: it changes most often, so only this layer is rebuilt).
COPY configs configs
COPY src src
RUN chown -R user:user /app
USER user

EXPOSE 7860
# The models load at start (~1-2 min on CPU); the port opens when they are ready.
CMD ["uvicorn", "manualmate.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
