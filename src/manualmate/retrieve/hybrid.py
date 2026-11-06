"""Hybrid search: dense (meaning) + BM25 (words), joined with RRF.

The two methods give scores on different scales (cosine 0..1, BM25 0..30+),
so we do not add the scores. Reciprocal Rank Fusion (RRF) uses only the
RANK of each chunk in each list:

    rrf(chunk) = sum over lists of  1 / (RRF_K + rank)

- A chunk high in both lists gets the best score.
- A chunk in only one list still gets some points.
- RRF_K = 60 is the standard value (Cormack et al., 2009). A large RRF_K
  makes the top ranks less dominant, so one list cannot "win" alone.

We take CANDIDATES chunks from each method, fuse them, and keep the top k.
"""

from __future__ import annotations

import numpy as np
import psycopg

from manualmate.index.bm25 import BM25Index
from manualmate.index.embedder import embed_query
from manualmate.retrieve.dense import search_by_vector as dense_search_by_vector

RRF_K = 60
CANDIDATES = 50   # chunks taken from each method before fusion


def rrf_fuse(result_lists: dict[str, list[dict]], k: int = 5,
             rrf_k: int = RRF_K) -> list[dict]:
    """Join ranked lists into one list with Reciprocal Rank Fusion.

    result_lists: {"dense": [...], "bm25": [...]}, each list best first,
    each item has a "chunk_id". Returns the top k chunks, best first.
    Each result has "score" (the RRF score) and "<method>_rank" for each
    list (None if the chunk was not in that list), to help find errors.
    """
    fused: dict[str, dict] = {}
    for method, results in result_lists.items():
        for rank, item in enumerate(results, start=1):
            entry = fused.get(item["chunk_id"])
            if entry is None:
                entry = {key: value for key, value in item.items() if key != "score"}
                entry["score"] = 0.0
                for name in result_lists:
                    entry[f"{name}_rank"] = None
                fused[item["chunk_id"]] = entry
            entry["score"] += 1.0 / (rrf_k + rank)
            entry[f"{method}_rank"] = rank
    return sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:k]


def search_by_vector(conn: psycopg.Connection, bm25_index: BM25Index,
                     question: str, vector: np.ndarray, k: int = 5,
                     candidates: int = CANDIDATES) -> list[dict]:
    """Hybrid search when the question vector is already computed (eval)."""
    return rrf_fuse({
        "dense": dense_search_by_vector(conn, vector, k=candidates),
        "bm25": bm25_index.search(question, k=candidates),
    }, k=k)


def search(conn: psycopg.Connection, bm25_index: BM25Index, question: str,
           k: int = 5) -> list[dict]:
    """Embed the question with bge-m3, then run hybrid search."""
    return search_by_vector(conn, bm25_index, question, embed_query(question), k)
