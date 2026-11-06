"""Dense search: find the chunks whose vectors are closest to the question.

This is the week 1 baseline. Week 2 adds BM25 (hybrid.py), the reranker,
query rewriting and metadata filters, and measures each change.
"""

from __future__ import annotations

import numpy as np
import psycopg

from manualmate.index.embedder import embed_query

# HNSW search is approximate. A higher ef_search checks more candidates:
# slower, but it does not miss close chunks. 100 is safe for our size.
EF_SEARCH = 100


def search_by_vector(conn: psycopg.Connection, vector: np.ndarray,
                     k: int = 5) -> list[dict]:
    """Return the k closest chunks to a vector, best first."""
    conn.execute(f"SET hnsw.ef_search = {EF_SEARCH};")
    rows = conn.execute(
        """
        SELECT chunk_id, manual_id, brand, model, category, page, text,
               1 - (embedding <=> %s) AS score
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (vector, vector, k),
    ).fetchall()
    keys = ["chunk_id", "manual_id", "brand", "model", "category", "page",
            "text", "score"]
    return [dict(zip(keys, row)) for row in rows]


def search(conn: psycopg.Connection, question: str, k: int = 5) -> list[dict]:
    """Embed the question with bge-m3, then return the k closest chunks."""
    return search_by_vector(conn, embed_query(question), k)
