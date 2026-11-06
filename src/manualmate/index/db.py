"""Connect to the Neon database and load chunks into it.

The connection string comes from the DATABASE_URL environment variable:
- laptop: from the .env file
- Colab:  from Colab Secrets (the notebook copies it into the environment)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

CHUNK_COLUMNS = [
    "chunk_id", "manual_id", "brand", "model", "category", "language",
    "filename", "page", "chunk_index", "text", "n_chars",
]


def get_connection() -> psycopg.Connection:
    """Open a connection to the database, with pgvector support."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (.env on laptop, Secrets on Colab).")
    conn = psycopg.connect(url)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    """Create the chunks table and its indexes (safe to run again)."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def reset_chunks(conn: psycopg.Connection) -> None:
    """Delete all chunks. Use before loading a new version of the chunks."""
    conn.execute("TRUNCATE chunks;")
    conn.commit()


def insert_chunks(conn: psycopg.Connection, chunks: list[dict],
                  embeddings: np.ndarray, batch_size: int = 200) -> int:
    """Insert chunks with their vectors. A chunk_id that exists is updated."""
    columns = CHUNK_COLUMNS + ["embedding"]
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns[1:])
    sql = (
        f"INSERT INTO chunks ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
    )
    rows = [
        [chunk[c] for c in CHUNK_COLUMNS] + [embeddings[i]]
        for i, chunk in enumerate(chunks)
    ]
    with conn.cursor() as cur:
        for start in range(0, len(rows), batch_size):
            cur.executemany(sql, rows[start:start + batch_size])
    conn.commit()
    return len(rows)


def count_chunks(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
