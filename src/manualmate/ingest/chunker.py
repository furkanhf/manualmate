"""Split parsed manual pages into chunks for embedding.

Input:  data/processed/<manual_id>.jsonl  (one record per page, from pdf_parser.py)
Output: data/chunks/chunks.jsonl          (one record per chunk)

Two modes:
  --mode section (default, week 4): one topic per chunk, cut at the headings.
  --mode fixed   (week 1 baseline):  ~1600-character pieces, rules below.

Rules of --mode section:
- Every chunk starts with a title line: product + heading path,
  e.g. "LG MC2846BG microwave | OPERATION > Ghee/Bakery Mode".
- A chunk never holds a new heading: a heading always starts a new chunk.
- A chunk never crosses a page. Text that goes on to the next page gets a new
  chunk with the same title line (and the same section number in chunk_id).
- A long section is cut at sentence ends, like --mode fixed (no overlap).
- Small sections stay small (no merging); the title line gives them meaning.
- Safe mode: a manual with fewer than MIN_HEADINGS_PER_PAGE headings per page
  (headings not found) uses --mode fixed.

Rules of --mode fixed:
- A chunk never crosses a page, so every chunk has exactly one page number.
- Chunks are about CHUNK_SIZE characters. Cuts are at paragraph ends, then
  sentence ends, then word ends. A word is never cut.
- Each chunk starts with the last ~OVERLAP characters of the chunk before it,
  so a sentence near a cut is not lost.

Run from the project folder:
    python src/manualmate/ingest/chunker.py                 # section mode
    python src/manualmate/ingest/chunker.py --mode fixed    # week-1 baseline
"""

from __future__ import annotations

import argparse
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

MIN_HEADINGS_PER_PAGE = 0.3   # section mode: fewer headings -> safe mode (fixed)
MAX_PATH = 2             # section mode: at most 2 heading levels in the title line
MIN_BODY_LETTERS = 12    # section mode: a body with fewer letters gives no chunk ("English - 58")

CATEGORY_NAMES = {"ac": "air conditioner", "washing-machine": "washing machine",
                  "router": "Wi-Fi router"}

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


# --- Section mode ------------------------------------------------------------
def product_name(page: dict) -> str:
    """"LG MC2846BG microwave" from the manifest labels."""
    category = CATEGORY_NAMES.get(page["category"], page["category"])
    return f'{page["brand"].upper() if len(page["brand"]) <= 3 else page["brand"].title()} ' \
           f'{page["model"].upper()} {category}'


def heading_count(pages: list[dict]) -> int:
    return sum(1 for page in pages for p in page.get("paras", []) if p["h"])


def chunk_sections(pages: list[dict]) -> list[dict]:
    """Section-mode chunks of one manual: [{"page", "section_no", "path", "text"}]."""
    product = product_name(pages[0])
    stack: list[tuple[int, str]] = []          # open headings: (level, text)
    section_no = 0
    chunks = []
    for page in pages:
        body: list[str] = []

        def flush() -> None:
            text = "\n".join(body)
            body.clear()
            if len(re.findall(r"[A-Za-z]", text)) < MIN_BODY_LETTERS:
                return
            path = " > ".join(t for _, t in stack[-MAX_PATH:])
            title = f"{product} | {path}" if path else product
            room = max(CHUNK_SIZE - len(title) - 1, 400)
            units = []
            for line in text.split("\n"):
                units.extend(split_long(line, room))
            for piece in pack(units, room):
                chunks.append({"page": page["page"], "section_no": section_no,
                               "path": path, "text": f"{title}\n{piece}"})

        for para in page.get("paras", []):
            if para["h"]:
                flush()                        # rule: a new heading ends the chunk
                while stack and stack[-1][0] >= para["h"]:
                    stack.pop()
                stack.append((para["h"], para["text"]))
                section_no += 1
            else:
                body.append(para["text"])
        flush()                                # rule: a chunk never crosses a page
    return chunks


# --- Run on all manuals ------------------------------------------------------
def read_pages(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_all(mode: str = "section") -> list[dict]:
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
            used = mode
            if mode == "section":
                if "paras" not in pages[0]:
                    raise SystemExit(f"{path.name} has no 'paras'. Run pdf_parser.py again.")
                if heading_count(pages) < MIN_HEADINGS_PER_PAGE * len(pages):
                    used = "fixed (few headings)"
            if used == "section":
                labels = [l for l in LABELS if l != "page"]
                counters: dict[int, int] = {}
                for c in chunk_sections(pages):
                    index = counters.get(c["page"], 0)
                    counters[c["page"]] = index + 1
                    record = {label: pages[0][label] for label in labels}
                    record["page"] = c["page"]
                    record["chunk_id"] = (f"{pages[0]['manual_id']}_p{c['page']:03d}"
                                          f"_s{c['section_no']:04d}_c{index}")
                    record["chunk_index"] = index
                    record["section"] = c["path"]
                    record["text"] = c["text"]
                    record["n_chars"] = len(c["text"])
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sizes.append(len(c["text"]))
            else:                              # fixed mode (or safe mode)
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
                "mode": used,
                "pages": len(pages),
                "chunks": len(sizes),
                "avg": round(sum(sizes) / len(sizes)) if sizes else 0,
                "min": min(sizes, default=0),
                "max": max(sizes, default=0),
            })
    return summary


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'manual_id':<10}{'pages':>7}{'chunks':>8}{'avg':>7}{'min':>7}{'max':>7}  mode")
    print("-" * 68)
    for row in summary:
        print(f"{row['manual_id']:<10}{row['pages']:>7}{row['chunks']:>8}"
              f"{row['avg']:>7}{row['min']:>7}{row['max']:>7}  {row['mode']}")
    print("-" * 68)
    total = sum(r["chunks"] for r in summary)
    print(f"{'TOTAL':<10}{sum(r['pages'] for r in summary):>7}{total:>8}"
          f"   ({len(summary)} manuals)")
    print(f"\nOutput file: {CHUNKS_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunk the parsed manuals.")
    ap.add_argument("--mode", choices=["section", "fixed"], default="section")
    print_summary(chunk_all(ap.parse_args().mode))
