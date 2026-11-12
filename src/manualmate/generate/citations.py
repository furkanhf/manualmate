"""Citations: find the [manual page] tags in an answer and check them.

What it is like: a teacher who checks the sources in an essay. For every
tag, the teacher asks: "Is this page one of the 5 pages I gave you?"
A tag to a page that was not given is a made-up source.

Why: a tag only helps the user if it is true. Gemini sometimes invents a
page number, or writes a fact with no tag. This file finds both problems,
so answer.py can flag the answer and the eval can count them.

Checks:
  1. Every tag must point to one of the 5 source chunks (manual_id + page).
  2. Every fact sentence should have a tag.
Gemini can write "[mw_003 p24]" or "[mw_003 p24, mw_003 p25]" or
"[mw_003 p.24]". We accept all three forms.

Runs on the laptop. Run this file to see the checks on 4 example answers.
"""

from __future__ import annotations

import re

from manualmate.generate.prompts import NOT_IN_MANUAL

BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")                       # [ ... ]
TAG_RE = re.compile(r"\b([a-z]{2}_\d{3})\s*p\.?\s*(\d+)\b", re.I)  # mw_003 p24
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# A full stop after these words does not end a sentence ("approx. 40 seconds").
ABBREV_RE = re.compile(r"\b(approx|ca|e\.g|i\.e|incl|fig|vs)\.(?=\s)"
                       r"|\b(no|min|max|sec|mins|secs|hrs)\.(?=\s*\d)", re.I)
DOT = "\u2024"         # placeholder for a protected full stop
MIN_WORDS = 4          # shorter pieces ("Steps:", "1.") are not fact sentences


def find_tags(answer: str) -> list[tuple[str, int]]:
    """Return every (manual_id, page) tag in the answer, in order."""
    tags = []
    for inside in BRACKET_RE.findall(answer):
        for manual_id, page in TAG_RE.findall(inside):
            tags.append((manual_id.lower(), int(page)))
    return tags


def strip_tags(answer: str) -> str:
    """The answer without tags (for RAGAS and for plain text)."""
    def drop(match: re.Match) -> str:
        return "" if TAG_RE.search(match.group(1)) else match.group(0)
    text = BRACKET_RE.sub(drop, answer)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()


def is_refusal(answer: str) -> bool:
    """True if Gemini said the sources do not answer the question."""
    return answer.strip().strip(".").upper() == NOT_IN_MANUAL


NOT_COVERED_RE = re.compile(
    r"\b(?:does|do)\s+not\s+(?:cover|mention|say|explain|include)|\bnot\s+covered\b",
    re.IGNORECASE)


def untagged_sentences(answer: str) -> list[str]:
    """Fact sentences that have no tag."""
    # A tag after the full stop ("... Ghee. [mw_003 p24]") belongs to the
    # sentence before it: move it in front of the full stop first.
    answer = re.sub(r"([.!?])[ \t]*((?:\[[^\[\]]+\][ \t]*)+)", r" \2\1", answer)
    answer = ABBREV_RE.sub(lambda m: (m.group(1) or m.group(2)) + DOT, answer)
    out = []
    last = None                   # the last fact sentence, while it has no tag
    for piece in SENTENCE_RE.split(answer):
        piece = piece.strip().replace(DOT, ".")
        if len(piece.split()) < MIN_WORDS:
            # A short piece with a tag ("▶ [ac_001 p9]") belongs to the
            # sentence before it.
            if last is not None and find_tags(piece):
                out.remove(last)
                last = None
            continue
        if piece.endswith(":"):   # a heading, for example "To clean the filter:"
            continue
        if NOT_COVERED_RE.search(piece):
            # Prompt rule 5: "say which part the manual does not cover".
            # That sentence states a gap, not a fact, so it needs no tag.
            continue
        if find_tags(piece):
            last = None
        else:
            out.append(piece)
            last = piece
    return out


def check(answer: str, chunks: list[dict]) -> dict:
    """Check the tags in one answer against its source chunks.

    Returns:
      refusal        True if the answer is NOT_IN_MANUAL (no other checks)
      tags           all tags found, as "mw_003 p24"
      invalid_tags   tags that are not one of the sources (made-up sources)
      untagged       fact sentences with no tag
      precision      share of tags that are valid (1.0 if no tags)
      ok             True if there is at least 1 tag, no invalid tag and
                     no untagged sentence
    """
    if is_refusal(answer):
        return {"refusal": True, "tags": [], "invalid_tags": [], "untagged": [],
                "precision": 1.0, "ok": True}

    allowed = {(c["manual_id"].lower(), int(c["page"])) for c in chunks}
    tags = find_tags(answer)
    invalid = [t for t in tags if t not in allowed]
    untagged = untagged_sentences(answer)
    precision = 1.0 if not tags else (len(tags) - len(invalid)) / len(tags)
    fmt = lambda t: f"{t[0]} p{t[1]}"
    return {
        "refusal": False,
        "tags": [fmt(t) for t in tags],
        "invalid_tags": [fmt(t) for t in invalid],
        "untagged": untagged,
        "precision": round(precision, 3),
        "ok": bool(tags) and not invalid and not untagged,
    }


if __name__ == "__main__":
    sources = [{"manual_id": "mw_003", "page": 24}, {"manual_id": "mw_003", "page": 157},
               {"manual_id": "ac_001", "page": 9}]
    examples = {
        "good": "Press Auto Cook until '6h' appears to select Ghee [mw_003 p24]. "
                "Then press Start [mw_003 p.24].",
        "made-up page": "Press Auto Cook until '6h' appears [mw_003 p25].",
        "no tag": "Press Auto Cook until '6h' appears [mw_003 p24]. "
                  "Then wait about 20 minutes for the butter to melt.",
        "abbreviation": "Start the programme and leave it to run for approx. 40 seconds "
                        "[mw_003 p24].",
        "tag after full stop": "Press Auto Cook until '6h' appears. [mw_003 p24] "
                               "Then press Start. [mw_003 p24]",
        "gemini q001": "Clean the filter every two weeks [ac_001 p9]. \n\nTo clean the filter:\n"
                       "1. Slide the filter off of the unit [ac_001 p9].\n"
                       "6. Reset the reminder: In operation ▶ ▶ Select Filter Reset. ▶ [ac_001 p9]",
        "refusal": "NOT_IN_MANUAL",
    }
    for name, answer in examples.items():
        result = check(answer, sources)
        print(f"--- {name}: ok={result['ok']}")
        for key in ("refusal", "tags", "invalid_tags", "untagged", "precision"):
            print(f"    {key}: {result[key]}")
    print("--- strip_tags:", strip_tags(examples["good"]))
