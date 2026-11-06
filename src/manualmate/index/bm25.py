"""BM25 keyword search over the chunks.

Dense search (bge-m3) compares MEANING. BM25 compares WORDS. It is strong
where dense search is weak: error codes ("U0", "OE", "tAP"), model numbers
("FTKS35", "DIR-825") and rare words ("ghee").

How a chunk gets its score (sum over the words of the question):
  - rare words give more points (IDF),
  - a word that repeats gives more points, but with a limit (k1),
  - long chunks get a small penalty (b).

The index lives in memory. It is built from the same chunks that are in
Neon, so dense search and BM25 always search the same chunks:
  - BM25Index.from_db(conn)        -> app, Colab eval
  - BM25Index.from_jsonl(path)     -> laptop tests without the database

2,230 chunks build in less than 1 second, so we do not save the index
to disk. We build it again when the program starts.

Limit: BM25 only matches words that are the same. A Hindi or Telugu
question has no words in common with the English manuals, so BM25 gives
noise for it. Hybrid search (retrieve/hybrid.py) keeps dense search for
those questions.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi

# BM25 settings. These are the usual values; we do not tune them yet.
K1 = 1.5
B = 0.75

# Columns that each search result has (the same keys as retrieve/dense.py).
RESULT_KEYS = ["chunk_id", "manual_id", "brand", "model", "category", "page", "text"]

# A token is letters/digits, and it can contain "-", "/" or "." between
# them: "u0", "ftks35", "dir-825", "192.168.0.1", "on/off".
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")
SPLIT_RE = re.compile(r"[-/.]")

# Common English words that carry no meaning for search. Kept short on
# purpose: words like "not", "off", "on" matter in manuals ("does not
# drain", "turn off"), so they stay.
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does doing for from had
has have how i if in into is it its me my of or our please should so than
that the their them then there these they this those to too us was we were
what when where which while who why will with would you your
""".split())


def _stem(word: str) -> str:
    """A very small English suffix remover: cleaning/cleaned/cleans -> clean.

    It only touches long, letters-only words, so codes like "u0" or "oe" and
    model numbers stay the same. It is not a real stemmer, but it is enough
    to join the usual forms of a verb or a plural.
    """
    if len(word) <= 4 or not word.isalpha():
        return word
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    # "cleaning" -> "clean", but "stopping" -> "stopp" -> "stop"
    if len(word) >= 4 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Turn text into BM25 tokens.

    1. Lowercase and normalize (NFKC) the text.
    2. Find the tokens. Remove stopwords.
    3. For a joined token like "dir-825", keep it AND its parts
       ("dir", "825"). So "DIR 825" and "DIR-825" both match.
    4. Remove simple English endings (_stem).
    """
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text):
        if token in STOPWORDS:
            continue
        tokens.append(_stem(token))
        if SPLIT_RE.search(token):
            tokens.extend(_stem(p) for p in SPLIT_RE.split(token)
                          if p and p not in STOPWORDS)
    return tokens


class BM25Index:
    """BM25 over a list of chunks. Build once, search many times."""

    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk.")
        self.chunks = [{key: c[key] for key in RESULT_KEYS} for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks],
                              k1=K1, b=B)

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Index":
        """Build from data/chunks/chunks.jsonl."""
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])

    @classmethod
    def from_db(cls, conn) -> "BM25Index":
        """Build from the chunks table (the same chunks that dense search uses)."""
        rows = conn.execute(
            f"SELECT {', '.join(RESULT_KEYS)} FROM chunks ORDER BY chunk_id"
        ).fetchall()
        return cls([dict(zip(RESULT_KEYS, row)) for row in rows])

    def search(self, question: str, k: int = 5) -> list[dict]:
        """Return the k best chunks for the question, best first.

        Chunks with score 0 (no word in common) are not returned, so the
        list can be shorter than k.
        """
        scores = self.bm25.get_scores(tokenize(question))
        best = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])}
                for i in best if scores[i] > 0]
