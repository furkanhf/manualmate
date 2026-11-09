"""Rerank: read the question and each candidate chunk TOGETHER, then sort again.

Why: dense search compares two vectors that were made separately (one for the
question, one for the chunk). That is fast, but it misses details, so the
right page is often at rank 2-5, or a contents page is at rank 1.
A cross-encoder reads "question + chunk" as one input and gives one score:
"how well does this chunk answer this question?". That is more exact, but
slow, so we use it only on the top N candidates from dense search.

Model: BAAI/bge-reranker-v2-m3 (~2.3 GB). It is multilingual, like bge-m3,
so Hindi and Telugu questions work later (week 4).

Runs on Colab (GPU). On the laptop there is no torch (on purpose).
Rule: like embedder.py, this is the only file that loads the reranker.
"""

from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 1024      # tokens for question + chunk; our chunks are ~400 tokens
CANDIDATES = 20        # how many dense results the reranker reads


@lru_cache(maxsize=1)
def get_model():
    """Load the reranker once and keep it in memory."""
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(MODEL_NAME, device=device, max_length=MAX_LENGTH)
    if device == "cuda":
        model.model.half()      # float16 on GPU: faster, same ranking
    print(f"Loaded {MODEL_NAME} on {device}")
    return model


def rerank(question: str, chunks: list[dict], top_k: int | None = None,
           batch_size: int = 16) -> list[dict]:
    """Return the chunks sorted by reranker score, best first.

    Each returned chunk is a copy with:
      score        the reranker score (0..1, higher = better answer)
      dense_score  the old score from dense search (kept for the eval)
    """
    if not chunks:
        return []
    pairs = [(question, c["text"]) for c in chunks]
    scores = get_model().predict(pairs, batch_size=batch_size,
                                 show_progress_bar=False, convert_to_numpy=True)
    out = []
    for chunk, score in zip(chunks, scores):
        c = dict(chunk)
        c["dense_score"] = chunk.get("score")
        c["score"] = float(score)
        out.append(c)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:top_k] if top_k else out
