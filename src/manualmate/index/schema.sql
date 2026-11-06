-- ManualMate database schema (PostgreSQL + pgvector on Neon).
-- Safe to run many times: every statement uses IF NOT EXISTS.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per chunk. The labels come from manifest.csv (via the parser and
-- the chunker). embedding is the bge-m3 dense vector (1024 numbers).
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,        -- e.g. wm_004_p030_c1
    manual_id    TEXT NOT NULL,           -- e.g. wm_004
    brand        TEXT NOT NULL,
    model        TEXT NOT NULL,
    category     TEXT NOT NULL,           -- ac, washing-machine, ...
    language     TEXT NOT NULL,
    filename     TEXT NOT NULL,
    page         INTEGER NOT NULL,        -- PDF page number, from 1
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    n_chars      INTEGER NOT NULL,
    embedding    vector(1024) NOT NULL
);

-- Fast nearest-neighbour search with cosine distance (operator <=>).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Metadata filters (week 2) search by these columns.
CREATE INDEX IF NOT EXISTS chunks_manual_id ON chunks (manual_id);
CREATE INDEX IF NOT EXISTS chunks_category ON chunks (category);
CREATE INDEX IF NOT EXISTS chunks_brand ON chunks (brand);
