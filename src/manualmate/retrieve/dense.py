"""Dense search: find the chunks whose vectors are closest to the question.

Week 1 baseline: search all chunks. Week 2: the router can limit the search
to some manuals (manual_ids).
"""

from __future__ import annotations

import numpy as np
import psycopg

from manualmate.index.embedder import embed_query

# HNSW search is approximate. A higher ef_search checks more candidates:
# slower, but it does not miss close chunks. 100 is safe for our size.
EF_SEARCH = 100

KEYS = ["chunk_id", "manual_id", "brand", "model", "category", "page", "text", "score"]
COLUMNS = ", ".join(KEYS[:-1])


def search_by_vector(conn: psycopg.Connection, vector: np.ndarray,
                     k: int = 5, manual_ids: list[str] | None = None) -> list[dict]:
    """Return the k closest chunks to a vector, best first.

    manual_ids: search only in these manuals (from the router). None = all.

    Why two SQL versions: the HNSW index first finds the closest chunks in the
    WHOLE table, and the WHERE filter comes after that. With a filter on one
    manual, most of those chunks are removed, so we can get fewer than k
    results. One manual has at most a few hundred chunks, so for a filtered
    search we compare the vector with every chunk of those manuals (exact
    search, no index). MATERIALIZED stops Postgres from using the HNSW index.
    """
    if manual_ids:
        sql = f"""
            WITH scoped AS MATERIALIZED (
                SELECT {COLUMNS}, embedding FROM chunks WHERE manual_id = ANY(%s)
            )
            SELECT {COLUMNS}, 1 - (embedding <=> %s) AS score
            FROM scoped
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params = (list(manual_ids), vector, vector, k)
    else:
        conn.execute(f"SET hnsw.ef_search = {EF_SEARCH};")
        sql = f"""
            SELECT {COLUMNS}, 1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params = (vector, vector, k)
    rows = conn.execute(sql, params).fetchall()
    return [dict(zip(KEYS, row)) for row in rows]


def search(conn: psycopg.Connection, question: str, k: int = 5,
           manual_ids: list[str] | None = None) -> list[dict]:
    """Embed the question with bge-m3, then return the k closest chunks."""
    return search_by_vector(conn, embed_query(question), k, manual_ids)
