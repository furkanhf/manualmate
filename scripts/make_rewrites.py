"""Rewrite the 50 test questions with Gemini and save them for the eval.

For each question in eval/text_queries.jsonl:
  1. the parser finds the category and the issue text (no brand/model words),
  2. Gemini writes the manual words (retrieve/query_rewriter.py),
  3. one line goes to eval/rewrites.jsonl:
     {"id", "question", "issue", "category", "rewrite", "model"}

Questions that are already in the file are skipped (Gemini is not called
again). Use --force to rewrite all of them again.

Run on the laptop (needs GEMINI_API_KEY in .env, no torch):
    python scripts/make_rewrites.py
    python scripts/make_rewrites.py --only q033 q035 q037   (test a few first)
    python scripts/make_rewrites.py --force
Then commit eval/rewrites.jsonl, so Colab can use it:
    python scripts/run_eval.py --method routed_rerank --rewrites
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from manualmate.retrieve.query_parser import parse
from manualmate.retrieve.query_rewriter import MODEL_NAME, rewrite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "eval" / "text_queries.jsonl"
REWRITES_PATH = PROJECT_ROOT / "eval" / "rewrites.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="rewrite only these question ids (e.g. q033 q035)")
    parser.add_argument("--force", action="store_true",
                        help="rewrite again, also the questions already in the file")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds between Gemini calls (free tier limit)")
    args = parser.parse_args()

    queries = load_jsonl(QUERIES_PATH)
    if args.only:
        queries = [q for q in queries if q["id"] in set(args.only)]
    saved = {r["id"]: r for r in load_jsonl(REWRITES_PATH)}

    todo = [q for q in queries if args.force or q["id"] not in saved]
    print(f"Model: {MODEL_NAME}   questions: {len(queries)}   to rewrite: {len(todo)}\n")

    for i, q in enumerate(todo, start=1):
        parsed = parse(q["question"])
        rewritten = rewrite(parsed.issue, parsed.category)
        saved[q["id"]] = {"id": q["id"], "question": q["question"],
                          "issue": parsed.issue, "category": parsed.category,
                          "rewrite": rewritten, "model": MODEL_NAME}
        print(f"[{i}/{len(todo)}] {q['id']}  {parsed.issue[:70]}")
        print(f"          -> {rewritten or '(EMPTY: Gemini failed)'}")
        # Save after every question, so a stop or an error loses nothing.
        with open(REWRITES_PATH, "w", encoding="utf-8") as f:
            for key in sorted(saved):
                f.write(json.dumps(saved[key], ensure_ascii=False) + "\n")
        if i < len(todo):
            time.sleep(args.delay)

    empty = [k for k, r in saved.items() if not r["rewrite"]]
    print(f"\nSaved {len(saved)} rewrites to {REWRITES_PATH}"
          + (f"   EMPTY: {', '.join(sorted(empty))}" if empty else ""))


if __name__ == "__main__":
    main()
