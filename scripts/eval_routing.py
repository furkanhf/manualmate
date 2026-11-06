"""Measure the query parser and the router (no database, no model needed).

Part 1 - eval/text_queries.jsonl (50 questions, each names its product):
  in_scope     the correct manual is in the manuals we search
  single       the router found exactly one manual, and it is correct
Part 2 - eval/clarify_queries.jsonl (10 questions, some do not name the model):
  ask_correct  the router asks when it must, and answers when it can
  scope_exact  the router selects exactly the expected manuals

Run:
    python scripts/eval_routing.py
"""

from __future__ import annotations

import json
from pathlib import Path

from manualmate.retrieve.query_parser import parse
from manualmate.retrieve.router import route

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> list[dict]:
    with open(PROJECT_ROOT / "eval" / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    # --- Part 1 --------------------------------------------------------------
    queries = load("text_queries.jsonl")
    in_scope = single = 0
    print("Part 1: text_queries.jsonl")
    for q in queries:
        parsed = parse(q["question"])
        r = route(parsed)
        ok = q["manual_id"] in r.manual_ids
        in_scope += ok
        single += ok and r.is_single
        if not (ok and r.is_single):
            print(f"  {q['id']} expected {q['manual_id']}  got {r.manual_ids} "
                  f"({r.reason})  | {parsed.summary()}")
    n = len(queries)
    print(f"  in_scope {in_scope / n:.2f}   single {single / n:.2f}   ({n} questions)\n")

    # --- Part 2 --------------------------------------------------------------
    queries = load("clarify_queries.jsonl")
    ask_ok = scope_ok = 0
    print("Part 2: clarify_queries.jsonl")
    for q in queries:
        r = route(parse(q["question"]))
        got = "ask" if r.ask else "answer"
        a, s = got == q["expect"], sorted(r.manual_ids) == sorted(q["manual_ids"])
        ask_ok += a
        scope_ok += s
        mark = "ok " if a and s else "ERR"
        buttons = ", ".join(name for _, name in r.options) or "-"
        print(f"  {mark} {q['id']} expect {q['expect']:<6} got {got:<6} "
              f"{len(r.manual_ids):>2} manual(s)  buttons: {buttons}")
    n = len(queries)
    print(f"  ask_correct {ask_ok / n:.2f}   scope_exact {scope_ok / n:.2f}   ({n} questions)")


if __name__ == "__main__":
    main()
