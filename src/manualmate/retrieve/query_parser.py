"""Understand the question before we search: which product, which problem.

The parser does NOT search. It reads the question and returns:
  - brand     ("daikin")             from the brand words in configs/aliases.yaml
  - category  ("ac")                 from the product words ("fridge", "washer", ...)
  - models    {"ac_002": 6, ...}     manuals whose model name is in the question,
                                     with the length of the match (longer = surer)
  - codes     ["U0"]                 error codes shown on the display
  - issue     "shows code U0. What does it mean?"
                                     the question without brand and model words;
                                     this is the text we search with

The router (retrieve/router.py) uses these fields to pick the manuals.

This version uses rules only (fast, free, same result each time). Hindi and
Telugu questions need Gemini to fill the same fields (week 4).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALIASES_PATH = PROJECT_ROOT / "configs" / "aliases.yaml"

MIN_PREFIX = 4   # "ftks35" may match "ftks35kvma", but "kgn" may not match "kgn56"

WORD_RE = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*")

# An error code is short and shown on the display: "U0", "OE", "tAP", "E21", "4C".
# We accept a word as a code if it is in quotes, or comes after one of these words.
CODE_TRIGGERS = r"(?:code|error|shows|showing|displays|display shows|blinks|flashes)"
CODE_RE = re.compile(
    rf"['\"‘’“”]([A-Za-z0-9]{{1,4}})['\"‘’“”]"          # 'tAP'
    rf"|\b{CODE_TRIGGERS}\s+(?:code\s+|error\s+)?([A-Za-z]{{0,3}}[0-9]{{0,3}})\b",
    re.IGNORECASE,
)
# Used in the issue text when the brand and model words are removed:
# "my Daikin FTKS35 is noisy" -> "my AC is noisy".
CATEGORY_LABELS = {"ac": "AC", "washing-machine": "washing machine",
                   "refrigerator": "fridge", "microwave": "microwave", "router": "router"}

NOT_CODES = {"a", "an", "the", "me", "on", "no", "up", "it", "is", "my", "this", "that",
             "what", "how", "why", "when"}


def _compact(text: str) -> str:
    """Lowercase and remove "-", "/", ".", and spaces: "DIR-825/AC" -> "dir825ac"."""
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


@dataclass
class Aliases:
    categories: dict[str, list[str]]           # category -> phrases
    brands: dict[str, list[str]]               # brand -> words
    manuals: dict[str, dict]                   # manual_id -> {brand, category, name, models}
    model_index: list[tuple[str, str]] = field(default_factory=list)  # (compact alias, manual_id)

    @classmethod
    def load(cls, path: Path = ALIASES_PATH) -> "Aliases":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        aliases = cls(data["categories"], data["brands"], data["manuals"])
        for manual_id, info in aliases.manuals.items():
            for model in info.get("models", []):
                aliases.model_index.append((_compact(model), manual_id))
        return aliases


@lru_cache(maxsize=1)
def get_aliases() -> Aliases:
    return Aliases.load()


@dataclass
class ParsedQuery:
    question: str
    brand: str | None = None
    category: str | None = None
    models: dict[str, int] = field(default_factory=dict)   # manual_id -> match length
    codes: list[str] = field(default_factory=list)
    issue: str = ""

    def summary(self) -> str:
        models = ",".join(f"{m}:{n}" for m, n in self.models.items()) or "-"
        return (f"brand={self.brand or '-'} category={self.category or '-'} "
                f"models={models} codes={','.join(self.codes) or '-'}")


def _phrase_found(phrase: str, text: str) -> tuple[int, int] | None:
    """Find a phrase as whole words in lowercase text. Returns (start, end)."""
    pattern = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    match = re.search(pattern, text)
    return match.span() if match else None


def _model_match(word: str, alias: str) -> int:
    """Length of the match between a question word and a model alias, or 0."""
    if word == alias:
        return len(word)
    shorter = min(len(word), len(alias))
    if shorter >= MIN_PREFIX and (word.startswith(alias) or alias.startswith(word)):
        return shorter
    return 0


def parse(question: str, aliases: Aliases | None = None) -> ParsedQuery:
    aliases = aliases or get_aliases()
    text = unicodedata.normalize("NFKC", question)
    lower = text.lower()
    parsed = ParsedQuery(question=question)
    remove: list[tuple[int, int]] = []    # character spans of brand/model words

    # 1. Brand: the first brand word in the question.
    best = None
    for brand, words in aliases.brands.items():
        for word in words:
            span = _phrase_found(word, lower)
            if span and (best is None or span[0] < best[1][0]):
                best = (brand, span)
    if best:
        parsed.brand = best[0]
        remove.append(best[1])

    # 2. Category: the longest product phrase ("washing machine" beats "washer").
    best_len = 0
    for category, phrases in aliases.categories.items():
        for phrase in phrases:
            if len(phrase) > best_len and _phrase_found(phrase, lower):
                parsed.category, best_len = category, len(phrase)

    # 3. Model: compare each word, and each pair of neighbouring words, with the aliases.
    words = [(m.group(), m.span()) for m in WORD_RE.finditer(lower)]
    candidates = [(_compact(w), span, False) for w, span in words]
    candidates += [(_compact(w1 + w2), (s1[0], s2[1]), True)
                   for (w1, s1), (w2, s2) in zip(words, words[1:])]
    brand_words = {_compact(b) for words_ in aliases.brands.values() for b in words_}
    for word, span, is_pair in candidates:
        if word in brand_words:
            continue
        for alias, manual_id in aliases.model_index:
            # A pair of words must be equal to an alias ("archer c6"), so that
            # "kgn56 freezer" does not count as one model name.
            length = (len(word) if word == alias else 0) if is_pair \
                else _model_match(word, alias)
            if length:
                if length > parsed.models.get(manual_id, 0):
                    parsed.models[manual_id] = length
                remove.append(span)

    # The category can also come from the model: "Daikin FTKS35" -> ac.
    if parsed.category is None and parsed.models:
        found = {aliases.manuals[m]["category"] for m in parsed.models}
        if len(found) == 1:
            parsed.category = found.pop()

    # 4. Error codes.
    for match in CODE_RE.finditer(text):
        code = match.group(1) or match.group(2)
        if code and code.lower() not in NOT_CODES and code not in parsed.codes \
                and any(c.isalpha() for c in code) and len(code) <= 4:
            parsed.codes.append(code)

    # 5. Issue text: the question without brand and model words.
    #    Words next to each other ("Daikin FTKS35") are removed as one block.
    #    If no product word is left, the first block becomes the product word.
    merged: list[list[int]] = []
    for start, end in sorted(remove):
        if merged and not text[merged[-1][1]:start].strip():
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    rest = "".join(text[a:b] for a, b in
                   zip([0] + [e for _, e in merged], [s_ for s_, _ in merged] + [len(text)]))
    needs_label = parsed.category is not None and not any(
        _phrase_found(p, rest.lower()) for p in aliases.categories[parsed.category])
    issue = text
    for i, (start, end) in reversed(list(enumerate(merged))):
        label = CATEGORY_LABELS.get(parsed.category, "") if needs_label and i == 0 else ""
        issue = issue[:start] + label + issue[end:]
    parsed.issue = re.sub(r"\s{2,}", " ", issue).strip()
    return parsed
