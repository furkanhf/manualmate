"""Turn text into vectors with bge-m3.

Rule: chunks and questions MUST use this same file, so the model and the
settings are always the same. Do not load the model anywhere else.

Works on Colab (GPU) and on a laptop (CPU). The model (~2.3 GB) downloads
once into HF_HOME (D:\\cache\\huggingface on the laptop).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "BAAI/bge-m3"
DIMENSIONS = 1024
MAX_SEQ_LENGTH = 1024   # tokens; our chunks are ~400 tokens, so this is enough


@lru_cache(maxsize=1)
def get_model():
    """Load bge-m3 once and keep it in memory."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    if device == "cuda":
        model.half()        # float16 on GPU: 2x faster, same quality
    print(f"Loaded {MODEL_NAME} on {device}")
    return model


def embed_texts(texts: list[str], batch_size: int = 32,
                show_progress: bool = True) -> np.ndarray:
    """Embed many texts. Returns a (len(texts), 1024) float32 array.

    The vectors are normalised (length 1), so cosine distance works directly.
    """
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(question: str) -> np.ndarray:
    """Embed one question. bge-m3 needs no special prefix for questions."""
    return embed_texts([question], batch_size=1, show_progress=False)[0]
