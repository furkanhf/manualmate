"""Embed all chunks with bge-m3 and load them into the Neon database.

Run on Colab (GPU) after uploading data/chunks/chunks.jsonl:
    python scripts/build_index.py

It deletes the old chunks first, so the database always matches the file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from manualmate.index.db import (count_chunks, get_connection, init_schema,
                                 insert_chunks, reset_chunks)
from manualmate.index.embedder import MODEL_NAME, embed_texts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.chunks, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    print(f"Read {len(chunks)} chunks from {args.chunks}")

    start = time.time()
    embeddings = embed_texts([c["text"] for c in chunks], batch_size=args.batch_size)
    print(f"Embedded with {MODEL_NAME} in {time.time() - start:.0f} s "
          f"-> shape {embeddings.shape}")

    with get_connection() as conn:
        init_schema(conn)
        reset_chunks(conn)
        insert_chunks(conn, chunks, embeddings)
        print(f"Database now has {count_chunks(conn)} chunks")


if __name__ == "__main__":
    main()
