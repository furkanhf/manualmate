"""Translate the Hindi/Telugu test questions to English with Gemini, once.

For each question in eval/multilingual_queries.jsonl, one line goes to
eval/translations.jsonl: {"id", "question", "translation", "model"}.
Questions already in the file are skipped (use --force to redo them).

Run on the laptop (needs GEMINI_API_KEY in .env, no torch):
    python scripts/make_translations.py --only h07 t07     (test a few first)
    python scripts/make_translations.py
Then commit eval/translations.jsonl, so Colab can use it:
    python scripts/run_eval.py --method routed_rerank --queries eval/multilingual_queries.jsonl --translations
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from manualmate.retrieve.query_rewriter import MODEL_NAME
from manualmate.retrieve.translator import translate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "eval" / "multilingual_queries.jsonl"
TRANSLATIONS_PATH = PROJECT_ROOT / "eval" / "translations.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")          # Hindi/Telugu in the Windows console
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="only these ids (e.g. h07 t07)")
    parser.add_argument("--force", action="store_true", help="translate again")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds between Gemini calls (free tier limit)")
    args = parser.parse_args()

    queries = load_jsonl(QUERIES_PATH)
    if args.only:
        queries = [q for q in queries if q["id"] in set(args.only)]
    saved = {r["id"]: r for r in load_jsonl(TRANSLATIONS_PATH)}
    todo = [q for q in queries if args.force or q["id"] not in saved]
    print(f"Model: {MODEL_NAME}   questions: {len(queries)}   to translate: {len(todo)}\n")

    for i, q in enumerate(todo, start=1):
        english = translate(q["question"])
        saved[q["id"]] = {"id": q["id"], "question": q["question"],
                          "translation": english, "model": MODEL_NAME}
        print(f"[{i}/{len(todo)}] {q['id']}  {q['question'][:70]}")
        print(f"          -> {english or '(EMPTY: Gemini failed)'}")
        with open(TRANSLATIONS_PATH, "w", encoding="utf-8") as f:   # save after each one
            for key in sorted(saved):
                f.write(json.dumps(saved[key], ensure_ascii=False) + "\n")
        if i < len(todo):
            time.sleep(args.delay)

    empty = [k for k, r in saved.items() if not r["translation"]]
    print(f"\nSaved {len(saved)} translations to {TRANSLATIONS_PATH}"
          + (f"   EMPTY: {', '.join(sorted(empty))}" if empty else ""))


if __name__ == "__main__":
    main()
