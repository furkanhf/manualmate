"""Parse product-manual PDFs into clean text, one record per page.

Input:  data/manifest.csv and the PDFs in data/raw/
Output: data/processed/<manual_id>.jsonl (one JSON line per page)

Run from the project folder:
    python src/manualmate/ingest/pdf_parser.py              # all manuals
    python src/manualmate/ingest/pdf_parser.py wm_004       # one manual
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

# --- Paths -------------------------------------------------------------------
# This file is at src/manualmate/ingest/pdf_parser.py, so parents[3] is the
# project folder (D:\Projects\manualmate).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Settings ----------------------------------------------------------------
LOW_TEXT_CHARS = 50          # a page with fewer characters is flagged low_text
REPEAT_RATIO = 0.5           # a line on >= 50% of pages is a header/footer
MIN_PAGES_FOR_REPEATS = 5    # do not look for repeats in very short manuals
MARGIN_RATIO = 0.10          # headers/footers sit in the top/bottom 10% of a page
STAMP_RATIO = 0.8            # a long line on >= 80% of pages is a stamp ...
STAMP_MIN_CHARS = 25         # ... if it has at least 25 characters

# Invisible characters that some PDFs contain (zero-width spaces, BOM,
# soft hyphen, word joiner).
INVISIBLE_CHARS = re.compile("[\u00ad\u200b-\u200f\u2060\ufeff]")
# Many kinds of space (em space, no-break space, ...) become a normal space.
ODD_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000\t]")
# A word broken at the end of a line: "instal-" + "lation".
HYPHEN_BREAK = re.compile(r"([a-z])-\s+([a-z])")
MULTI_SPACE = re.compile(r" {2,}")
# Some PDFs (the Daikin manuals) store "fi"/"fl" as one ligature glyph and the
# text comes out with a space after it: "fi lter", "airfl ow", "effi cient".
LIGATURE_GAP = re.compile(r"\b([A-Za-z]*?)(ffi|ffl|fi|fl) ([a-z]+)")
LIGATURE_MIN = 20   # fix only manuals with at least this many gaps
ONLY_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")   # a bare page number


# --- Manifest ----------------------------------------------------------------
def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    """Read manifest.csv and return one dict per manual."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- Cleaning helpers --------------------------------------------------------
def clean_chars(text: str) -> str:
    """Fix the characters of a string: unicode form, invisible chars, spaces."""
    text = unicodedata.normalize("NFKC", text)        # "ﬁ" -> "fi", etc.
    text = INVISIBLE_CHARS.sub("", text)
    text = ODD_SPACES.sub(" ", text)
    # Remove private-use characters. PDFs use them for icons (buttons, symbols)
    # and they have no meaning as text.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Co")
    return text


def line_key(line: str) -> str:
    """Normalise a line so the same header on different pages gives one key.

    Digits become '#', so "Page 5" and "Page 6" have the same key.
    """
    key = re.sub(r"\d+", "#", line.lower())
    return MULTI_SPACE.sub(" ", key).strip()


def join_block(lines: list[str]) -> str:
    """Join the lines of one text block into one paragraph."""
    text = " ".join(line.strip() for line in lines if line.strip())
    text = HYPHEN_BREAK.sub(r"\1\2", text)     # "re- quired" -> "required"
    return MULTI_SPACE.sub(" ", text).strip()


# --- Extraction --------------------------------------------------------------
def extract_blocks(page: pymupdf.Page) -> list[dict]:
    """Return the text blocks of a page.

    Each block is {"lines": [...cleaned lines...], "in_margin": bool}.
    in_margin is True when the block is in the top or bottom 10% of the page,
    where headers, footers and page numbers are.

    PyMuPDF gives blocks in the order the PDF stores them. For most manuals
    this is the reading order, also for two-column pages.
    """
    height = page.rect.height
    top_limit = page.rect.y0 + height * MARGIN_RATIO
    bottom_limit = page.rect.y1 - height * MARGIN_RATIO

    blocks = []
    for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks"):
        if block_type != 0:          # type 1 = image block
            continue
        lines = [clean_chars(line) for line in text.split("\n")]
        lines = [line for line in lines if line.strip()]
        if lines:
            in_margin = y1 <= top_limit or y0 >= bottom_limit
            blocks.append({"lines": lines, "in_margin": in_margin})
    return blocks


def find_repeated_lines(pages_blocks: list[list[dict]]) -> tuple[set[str], set[str]]:
    """Find noise lines that repeat on many pages.

    Returns two sets of line keys:
    - margin_repeats: lines in the top/bottom margin on >= 50% of pages
      (headers, footers). Only margin lines count, because content in the
      body of a page (recipes, login steps) can repeat too, and we keep it.
      A single word ("Ingredients") is a heading, so we keep it.
    - stamps: long lines anywhere on >= 80% of pages (watermarks such as
      "Downloaded from www.Manualslib.com", print marks such as "...indd 1").
    """
    n_pages = len(pages_blocks)
    if n_pages < MIN_PAGES_FOR_REPEATS:
        return set(), set()

    margin_counts, all_counts = Counter(), Counter()
    for blocks in pages_blocks:
        margin_keys, all_keys = set(), set()
        for block in blocks:
            for line in block["lines"]:
                key = line_key(line)
                all_keys.add(key)
                if block["in_margin"]:
                    margin_keys.add(key)
        margin_counts.update(margin_keys)
        all_counts.update(all_keys)

    margin_repeats = {
        key for key, count in margin_counts.items()
        if count >= n_pages * REPEAT_RATIO and len(key.split()) >= 2
    }
    stamps = {
        key for key, count in all_counts.items()
        if count >= n_pages * STAMP_RATIO and len(key) >= STAMP_MIN_CHARS
    }
    return margin_repeats, stamps


def is_noise(line: str, in_margin: bool,
             margin_repeats: set[str], stamps: set[str]) -> bool:
    """True if a line is a header, footer, stamp or bare page number."""
    key = line_key(line)
    if key in stamps:
        return True
    if in_margin:
        return key in margin_repeats or bool(ONLY_NUMBER.match(line))
    return False


def parse_pdf(pdf_path: Path, meta: dict) -> list[dict]:
    """Parse one PDF and return one record per page."""
    with pymupdf.open(pdf_path) as doc:
        pages_blocks = [extract_blocks(page) for page in doc]

    margin_repeats, stamps = find_repeated_lines(pages_blocks)

    page_texts = []
    for blocks in pages_blocks:
        paragraphs = []
        for block in blocks:
            kept = [
                line for line in block["lines"]
                if not is_noise(line, block["in_margin"], margin_repeats, stamps)
            ]
            paragraph = join_block(kept)
            if paragraph:
                paragraphs.append(paragraph)
        page_texts.append("\n".join(paragraphs))

    # Join broken ligatures only in a manual where the problem is common.
    # In other manuals a match is usually a real word ("burfi in").
    gaps = sum(len(LIGATURE_GAP.findall(text)) for text in page_texts)
    if gaps >= LIGATURE_MIN:
        page_texts = [LIGATURE_GAP.sub(r"\1\2\3", text) for text in page_texts]

    records = []
    for page_no, text in enumerate(page_texts, start=1):
        records.append({
            "manual_id": meta["manual_id"],
            "brand": meta["brand"],
            "model": meta["model"],
            "category": meta["category"],
            "language": meta["language"],
            "filename": meta["filename"],
            "page": page_no,                       # PDF page number, from 1
            "text": text,
            "n_chars": len(text),
            "low_text": len(text) < LOW_TEXT_CHARS,
        })
    return records


def save_jsonl(records: list[dict], out_path: Path) -> None:
    """Write the records as JSON Lines (one JSON object per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- Run on all manuals ------------------------------------------------------
def parse_all(only_ids: list[str] | None = None) -> list[dict]:
    """Parse every manual in the manifest (or only the given IDs)."""
    summary = []
    for meta in load_manifest():
        if only_ids and meta["manual_id"] not in only_ids:
            continue
        pdf_path = RAW_DIR / meta["filename"]
        if not pdf_path.exists():
            print(f"MISSING FILE: {pdf_path.name}")
            continue
        records = parse_pdf(pdf_path, meta)
        save_jsonl(records, PROCESSED_DIR / f"{meta['manual_id']}.jsonl")
        summary.append({
            "manual_id": meta["manual_id"],
            "filename": meta["filename"],
            "pages": len(records),
            "low_text": sum(r["low_text"] for r in records),
            "chars": sum(r["n_chars"] for r in records),
        })
    return summary


def print_summary(summary: list[dict]) -> None:
    print(f"\n{'manual_id':<10}{'pages':>7}{'low_text':>10}{'chars':>10}  filename")
    print("-" * 80)
    for row in summary:
        print(f"{row['manual_id']:<10}{row['pages']:>7}{row['low_text']:>10}"
              f"{row['chars']:>10}  {row['filename']}")
    print("-" * 80)
    print(f"{'TOTAL':<10}{sum(r['pages'] for r in summary):>7}"
          f"{sum(r['low_text'] for r in summary):>10}"
          f"{sum(r['chars'] for r in summary):>10}  ({len(summary)} manuals)")
    print(f"\nOutput folder: {PROCESSED_DIR}")


if __name__ == "__main__":
    print_summary(parse_all(sys.argv[1:] or None))
