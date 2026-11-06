"""Split parsed manual pages into chunks for embedding (week 1 baseline).

Input:  data/processed/<manual_id>.jsonl  (one record per page, from pdf_parser.py)
Output: data/chunks/chunks.jsonl          (one record per chunk)

Rules of this baseline chunker:
- A chunk never crosses a page, so every chunk has exactly one page number.
- Chunks are about CHUNK_SIZE characters. Cuts are at paragraph ends, then
  sentence ends, then word ends. A word is never cut.
- Each chunk starts with the last ~OVERLAP characters of the chunk before it,
  so a sentence near a cut is not lost.

Run from the project folder:
    python src/manualmate/ingest/chunker.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"

# --- Settings ----------------------------------------------------------------
CHUNK_SIZE = 1600        # target characters per chunk (about 400 tokens)
OVERLAP = 200            # characters copied from the end of the chunk before
MIN_LAST_CHUNK = 300     # a shorter last piece is joined to the chunk before
MIN_PAGE_CHARS = 20      # pages with less text than this give no chunk

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Labels copied from the page record to every chunk.
LABELS = ["manual_id", "brand", "model", "category", "language", "filename", "page"]


# --- Splitting ---------------------------------------------------------------
def split_long(text: str, max_chars: int) -> list[str]:
    """Split one long piece into parts <= max_chars, at sentence or word ends."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    for sentence in SENTENCE_END.split(text):
        if len(sentence) <= max_chars:
            parts.append(sentence)
            continue
        # A very long sentence (often a table row run together): cut at words.
        current = ""
        for word in sentence.split(" "):
            if current and len(current) + 1 + len(word) > max_chars:
                parts.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            parts.append(current)
    return parts


def pack(units: list[str], max_chars: int) -> list[str]:
    """Put units together, in order, into pieces of up to max_chars."""
    pieces, current = [], ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        pieces.append(current)
    # Join a very short last piece to the piece before it.
    if len(pieces) > 1 and len(pieces[-1]) < MIN_LAST_CHUNK:
        pieces[-2] = f"{pieces[-2]}\n{pieces[-1]}"
        pieces.pop()
    return pieces


def tail(text: str, max_chars: int) -> str:
    """The last <= max_chars characters of text, starting at a word."""
    if len(text) <= max_chars:
        return text
    cut = text[-max_chars:]
    space = cut.find(" ")
    return cut[space + 1:] if space != -1 else cut


def chunk_page(text: str) -> list[str]:
    """Split the text of one page into overlapping chunks."""
    body_size = CHUNK_SIZE - OVERLAP
    units = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            units.extend(split_long(paragraph, body_size))
    pieces = pack(units, body_size)

    chunks = []
    for i, piece in enumerate(pieces):
        if i == 0:
            chunks.append(piece)
        else:
            chunks.append(f"{tail(pieces[i - 1], OVERLAP)}\n{piece}")
    return chunks


# --- Run on all manuals ------------------------------------------------------
def read_pages(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_all() -> list[dict]:
    """Chunk every manual in data/processed and write data/chunks/chunks.jsonl."""
    page_files = sorted(PROCESSED_DIR.glob("*.jsonl"))
    if not page_files:
        raise SystemExit(f"No files in {PROCESSED_DIR}. Run pdf_parser.py first.")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for path in page_files:
            pages = read_pages(path)
            sizes = []
            for page in pages:
                if page["n_chars"] < MIN_PAGE_CHARS:
                    continue
                for index, text in enumerate(chunk_page(page["text"])):
                    record = {label: page[label] for label in LABELS}
                    record["chunk_id"] = (
                        f"{page['manual_id']}_p{page['page']:03d}_c{index}"
                    )
                    record["chunk_index"] = index
                    record["text"] = text
                    record["n_chars"] = len(text)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sizes.append(len(text))
            summary.append({
                "manual_id": path.stem,
                "pages": len(pages),
                "chunks": len(sizes),
                "avg": round(sum(sizes) / len(sizes)) if sizes else 0,
                "min": min(sizes, default=0),
                "max": max(sizes, default=0),
            })
    return summary


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'manual_id':<10}{'pages':>7}{'chunks':>8}{'avg':>7}{'min':>7}{'max':>7}")
    print("-" * 46)
    for row in summary:
        print(f"{row['manual_id']:<10}{row['pages']:>7}{row['chunks']:>8}"
              f"{row['avg']:>7}{row['min']:>7}{row['max']:>7}")
    print("-" * 46)
    total = sum(r["chunks"] for r in summary)
    print(f"{'TOTAL':<10}{sum(r['pages'] for r in summary):>7}{total:>8}"
          f"   ({len(summary)} manuals)")
    print(f"\nOutput file: {CHUNKS_PATH}")


if __name__ == "__main__":
    print_summary(chunk_all())
