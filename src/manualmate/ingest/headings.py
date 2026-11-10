"""Find section headings in manual PDFs (week 4, heading-aware chunking).

What it does
- Reads every text line of a PDF with its font size and bold flag (PyMuPDF "dict").
- Marks a line as a heading when it is bold or clearly bigger than the body text,
  and it does not look like a table cell, a step number or a piece of a sentence.
- Joins a heading that is broken over 2 lines ("6.5 Fruit and vegetable con-" + "tainer").
- Gives each heading a level: 1 = biggest style in the manual, 2 = next, ...

Used by pdf_parser.py / chunker.py (--mode section). Run alone to check the result:
    python src/manualmate/ingest/headings.py                 # all manuals, 10 samples each
    python src/manualmate/ingest/headings.py mw_001 --sample 30
    python src/manualmate/ingest/headings.py mw_003 --page 24  # every heading on one page
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

import pymupdf

from manualmate.ingest.pdf_parser import RAW_DIR, clean_chars, load_manifest

# --- Settings ----------------------------------------------------------------
MARGIN = 0.07            # lines in the top/bottom 7% of a page are never headings
BIGGER = 1.15            # a heading is >= 15% bigger than the body text ...
MUCH_BIGGER = 1.8        # ... and may start with a small letter only if >= 80% bigger
MAX_CHARS = 80           # a heading is short
MAX_WORDS = 12
MIN_LETTERS = 3          # "1", "24", "4." are step or page numbers, not headings
REPEAT_PAGES = 0.3       # a heading on >= 30% of pages is a running header
SAME_ROW = 2.0           # points: candidates on the same row = table column titles
JOIN_GAP = 0.9           # join 2 heading lines if the gap is < 0.9 x font size
JOIN_X = 40.0            # ... and they start at almost the same x (points)
MAX_PARTS = 2            # a "heading" of 3+ lines is a bold paragraph
LABEL_WORDS = 2          # a short text (<= 2 words) ...
LABEL_PAGES = 3          # ... used as a heading on >= 3 pages is a label ("Method:", "Done!")

SENTENCE_END = (".", ",", ";")
DOT_LEADER = re.compile(r"\.{4,}|(\. ){3,}")          # table of contents: "Safety ..... 8"
MID_SENTENCE = re.compile(r"[a-z][.!?] +[A-Z]")      # "operations. The temperature"
DOUBLED = re.compile(r"^(.+?)\d+ \1$")                # "Defrosting13 Defrosting"
BULLET = re.compile(r"^[▶►•·¡\-–*]\s*")              # "▶When using ..." = a list item
DANGLING = re.compile(r"\b(and|or|the|a|an|of|to|for|with|in|on|by)$", re.I)  # sentence goes on


# --- Read lines --------------------------------------------------------------
def read_lines(doc: pymupdf.Document) -> list[dict]:
    """Every horizontal text line of the document, with its style and position."""
    lines = []
    for page_no, page in enumerate(doc, start=1):
        height = page.rect.height
        for block_no, block in enumerate(page.get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                dx, dy = line["dir"]
                if abs(dy) > 0.01 or dx < 0:          # rotated or upside-down text
                    continue
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = clean_chars("".join(s["text"] for s in line["spans"])).strip()
                text = "".join(ch for ch in text if ch.isprintable() or ch == " ")
                text = re.sub(r"\s+", " ", text).strip()   # control chars, line breaks
                if not text:
                    continue
                main = max(spans, key=lambda s: len(s["text"].strip()))
                bold = all(s["flags"] & 16 or "bold" in s["font"].lower()
                           for s in spans)
                x0, y0, x1, y1 = line["bbox"]
                lines.append({
                    "idx": len(lines), "page": page_no, "block": block_no, "text": text,
                    "size": round(main["size"], 1), "bold": bool(bold),
                    "x0": x0, "y0": y0, "y1": y1,
                    "in_margin": y0 < height * MARGIN or y1 > height * (1 - MARGIN),
                    "heading": False, "level": 0,
                })
    return lines


def body_size(lines: list[dict]) -> float:
    """The font size that holds the most characters = the body text size."""
    sizes = Counter()
    for line in lines:
        sizes[line["size"]] += len(line["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


# --- Heading tests -----------------------------------------------------------
def looks_like_heading(line: dict, body: float) -> bool:
    text = line["text"]
    if line["in_margin"]:
        return False
    if len(text) > MAX_CHARS or len(text.split()) > MAX_WORDS:
        return False
    if len(re.findall(r"[A-Za-z]", text)) < MIN_LETTERS:
        return False
    if DOT_LEADER.search(text) or MID_SENTENCE.search(text) or BULLET.match(text):
        return False
    bigger = line["size"] >= body * BIGGER
    if not (line["bold"] or bigger):
        return False
    # A sentence piece: ends like a sentence (and is not an ALL CAPS title).
    if text.endswith(SENTENCE_END) and not text.isupper():
        return False
    # Starts with a small letter: a sentence piece, unless the font is much bigger
    # (Samsung writes titles in small letters: "troubleshooting and information code").
    first = re.search(r"[A-Za-z]", text).group()
    if first.islower() and line["size"] < body * MUCH_BIGGER:
        return False
    return True


def drop_table_rows(cands: list[dict]) -> list[dict]:
    """Column titles of a table ("Problem | Cause | Action") sit on one row.

    A real heading is alone on its row, so 2 or more candidates with the
    same y on the same page are table cells.
    """
    keep = []
    for line in cands:
        same_row = [c for c in cands
                    if c is not line and c["page"] == line["page"]
                    and abs(c["y0"] - line["y0"]) < SAME_ROW]
        if not same_row:
            keep.append(line)
    return keep


def join_broken(cands: list[dict]) -> list[dict]:
    """Join a heading that continues on the next line (same page, block and style)."""
    joined: list[dict] = []
    for line in cands:
        prev = joined[-1] if joined else None
        if (prev and prev["page"] == line["page"]
                and line["idx"] == prev["parts"][-1]["idx"] + 1
                and prev["size"] == line["size"] and prev["bold"] == line["bold"]
                and abs(prev["x0"] - line["x0"]) < JOIN_X
                and 0 <= line["y0"] - prev["y1"] < JOIN_GAP * line["size"]):
            if prev["text"].endswith("-"):
                prev["text"] = prev["text"][:-1] + line["text"]
            else:
                prev["text"] = f'{prev["text"]} {line["text"]}'
            prev["parts"].append(line)
            continue
        line["parts"] = [line]
        joined.append(line)
    return joined


def drop_paragraphs(cands: list[dict], lines: list[dict]) -> list[dict]:
    """Remove bold paragraphs and broken sentences that passed the line tests.

    - 3 or more joined lines: a bold paragraph (Bosch warnings), not a heading.
    - Still ends with "-" or with "and", "the", ...: the sentence goes on.
    - The next line has the same style and block: the bold text goes on as
      a paragraph. A real heading is followed by a change of style.
    """
    keep = []
    for c in cands:
        if (len(c["parts"]) > MAX_PARTS or c["text"].endswith("-")
                or DANGLING.search(c["text"])):
            continue
        last = c["parts"][-1]["idx"]
        nxt = lines[last + 1] if last + 1 < len(lines) else None
        if (nxt and not nxt.get("_cand") and nxt["page"] == c["page"]
                and nxt["block"] == c["block"] and nxt["size"] == c["size"]
                and nxt["bold"] == c["bold"]):
            continue
        keep.append(c)
    return keep


def drop_labels(cands: list[dict]) -> list[dict]:
    """Short texts used as a heading on many pages are labels inside a section
    ("Method:", "Ingredients", "---End", "Done!", "WARNING"). They stay body text,
    so a recipe is not cut away from its title.
    """
    key = lambda c: re.sub(r"[\d:.!]+", "", c["text"].lower()).strip()
    pages_with = Counter()
    for k, page in {(key(c), c["page"]) for c in cands}:
        pages_with[k] += 1
    return [c for c in cands
            if not (len(c["text"].split()) <= LABEL_WORDS
                    and pages_with[key(c)] >= LABEL_PAGES)]


def drop_running_headers(cands: list[dict], n_pages: int) -> list[dict]:
    """A heading text on >= 30% of pages is a running header, not a section."""
    if n_pages < 5:
        return cands
    key = lambda c: re.sub(r"\d+", "#", c["text"].lower())
    pages_with = Counter()
    for k, page in {(key(c), c["page"]) for c in cands}:
        pages_with[k] += 1
    return [c for c in cands if pages_with[key(c)] < n_pages * REPEAT_PAGES]


def set_levels(heads: list[dict]) -> None:
    """Level 1 = biggest heading style; bold beats not-bold at the same size."""
    styles = sorted({(h["size"], h["bold"]) for h in heads}, reverse=True)
    rank = {style: i + 1 for i, style in enumerate(styles)}
    for h in heads:
        h["level"] = rank[(h["size"], h["bold"])]


# --- Main entry --------------------------------------------------------------
def find_headings(doc: pymupdf.Document) -> tuple[list[dict], list[dict]]:
    """Return (all lines, headings). Heading lines have heading=True and a level."""
    lines = read_lines(doc)
    body = body_size(lines)
    cands = [l for l in lines if looks_like_heading(l, body)]
    cands = drop_table_rows(cands)
    for c in cands:
        c["_cand"] = True
    cands = join_broken(cands)
    cands = drop_paragraphs(cands, lines)
    cands = drop_running_headers(cands, len(doc))
    cands = drop_labels(cands)
    for c in cands:
        # Bosch prints the chapter title twice: "Defrosting13 Defrosting" -> "Defrosting"
        c["text"] = DOUBLED.sub(r"\1", c["text"])
    set_levels(cands)
    for h in cands:
        for part in h["parts"]:
            part["heading"] = True
            part["level"] = h["level"]
    return lines, cands


# --- Check from the command line ---------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*", help="manual ids (default: all)")
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--page", type=int, help="print every heading on this page")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for meta in load_manifest():
        if args.ids and meta["manual_id"] not in args.ids:
            continue
        path = Path(RAW_DIR) / meta["filename"]
        with pymupdf.open(path) as doc:
            lines, heads = find_headings(doc)
            n_pages = len(doc)
        total += len(heads)
        print(f'\n{meta["manual_id"]}  pages={n_pages}  body={body_size(lines)}'
              f'  headings={len(heads)}  per_page={len(heads) / n_pages:.1f}')
        if args.page:
            show = [h for h in heads if h["page"] == args.page]
        else:
            show = sorted(rng.sample(heads, min(args.sample, len(heads))),
                          key=lambda h: (h["page"], h["y0"]))
        for h in show:
            print(f'   p{h["page"]:<4} L{h["level"]}  {h["text"]}')
    print(f"\nTOTAL headings: {total}")


if __name__ == "__main__":
    main()
