"""Week-6 app check: are the answers the USER sees correct?

Like run_answer_eval.py, but every question goes through the app path
(answer.ask, the same code as the chat UI and POST /ask):
  language detection -> live Gemini translation (Hindi/Telugu) -> router
  -> "Which model?" (the script clicks the labeled manual, like a user)
  -> dense + rerank -> gate 1 (on, as in the app) -> Gemini -> gate 2.

Question sets and what counts as correct:
  text     text_queries.jsonl (50) + text_queries_new.jsonl (14)
           -> answered AND a gold page is cited
  multi    multilingual_queries.jsonl (20, Hindi/Telugu)
           -> the same, AND the reply is in the question's language
  oos      out_of_scope.jsonl (30)  -> refused, or asked back
  clarify  clarify_queries.jsonl (10) -> asked back when "expect" is "ask"

Run on Colab after sections 1-4 (torch, DATABASE_URL, GEMINI_API_KEY):
    python scripts/run_app_eval.py --limit 3      # quick test, 3 per set
    python scripts/run_app_eval.py                # all 124 (~150 Gemini calls)
Laptop test of the script itself (BM25, weaker answers): --search bm25

Output: eval/results/<date>_app.jsonl and the list of wrong answers.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
LANGUAGES = {"en": "English", "hi": "Hindi", "te": "Telugu"}
SCRIPT = {"Hindi": re.compile(r"[ऀ-ॿ]"), "Telugu": re.compile(r"[ఀ-౿]")}


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (EVAL / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sets() -> dict[str, list[dict]]:
    return {"text": load("text_queries.jsonl") + load("text_queries_new.jsonl"),
            "multi": load("multilingual_queries.jsonl"),
            "oos": load("out_of_scope.jsonl"),
            "clarify": load("clarify_queries.jsonl")}


def judge(set_name: str, q: dict, row: dict) -> bool:
    if set_name in ("text", "multi"):
        ok = row["action"] == "answer" and row["gold_cited"]
        if set_name == "multi":
            want = LANGUAGES[q["language"]]
            ok = ok and row["language"] == want and bool(SCRIPT[want].search(row["text"]))
        return ok
    if set_name == "oos":
        return row["action"] in ("refuse", "ask")
    return (row["action"] == "ask") == (q.get("expect") == "ask")      # clarify


def run(service, limit: int | None, sleep: float) -> Path:
    from manualmate.api.schemas import AskRequest

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = EVAL / "results" / f"{stamp}_app.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for set_name, queries in sets().items():
            queries = queries[:limit] if limit else queries
            print(f"\n=== {set_name}: {len(queries)} questions")
            for q in queries:
                r = service.ask(AskRequest(question=q["question"]))
                asked_first = False
                labeled = q.get("manual_id")
                if (set_name in ("text", "multi") and r.action == "ask"
                        and labeled in {o.manual_id for o in r.options}):
                    asked_first = True             # the user clicks the right button
                    r = service.ask(AskRequest(question=q["question"], manual_id=labeled))
                cited = re.findall(r"\b([a-z]{2}_\d{3})\s*p\.?\s*(\d+)", r.text, re.I)
                cited = sorted({f"{m.lower()} p{int(p)}" for m, p in cited})
                gold = sorted(f"{labeled} p{p}" for p in q.get("pages", []))
                row = {"set": set_name, "id": q["id"], "question": q["question"],
                       "action": r.action, "reason": r.reason, "language": r.language,
                       "search_text": r.search_text, "asked_first": asked_first,
                       "options": [o.manual_id for o in r.options],
                       "top_score": r.top_score, "citations_ok": r.citations_ok,
                       "gold": gold, "cited": cited, "gold_cited": bool(set(gold) & set(cited)),
                       "sources": [s.tag for s in r.sources], "text": r.text,
                       "timings": r.timings.model_dump()}
                row["correct"] = judge(set_name, q, row)
                mark = "ok " if row["correct"] else "BAD"
                print(f"  {mark} {q['id']:5s} {r.action:6s} {r.reason:14s} "
                      f"{'(asked first) ' if asked_first else ''}cited={cited}")
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                if r.timings.gemini_ms is not None:
                    time.sleep(sleep)              # stay under the Gemini rate limit
    print(f"\nSaved {out_path}")
    return out_path


def summarize(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_set = defaultdict(list)
    for row in rows:
        by_set[row["set"]].append(row)
    print("\n================ RESULT ================")
    names = {"text": "English in-scope: answered + gold page cited",
             "multi": "Hindi/Telugu: answered + gold cited + right language",
             "oos": "Out-of-scope: refused or asked back",
             "clarify": "Vague questions: asked back when expected"}
    for name, label in names.items():
        rs = by_set.get(name, [])
        if rs:
            n = sum(r["correct"] for r in rs)
            print(f"  {label:55s} {n}/{len(rs)} = {n / len(rs):.2f}")
    text = by_set.get("text", []) + by_set.get("multi", [])
    answered = [r for r in text if r["action"] == "answer"]
    if answered:
        print(f"  citations clean (no made-up tag, no untagged fact)     "
              f"{sum(bool(r['citations_ok']) for r in answered)}/{len(answered)}")
    errors = [r for r in rows if r["action"] == "error"]
    if errors:
        print(f"  Gemini errors (timeouts/quota, run again): {[r['id'] for r in errors]}")
    wrong = [r for r in rows if not r["correct"]]
    print(f"\n================ WRONG ({len(wrong)}) ================")
    for r in wrong:
        print(f"\n{r['id']} [{r['set']}] {r['action']}/{r['reason']}  top score {r['top_score']}")
        print(f"  Q: {r['question']}")
        if r["search_text"] != r["question"]:
            print(f"  searched: {r['search_text']}")
        if r["gold"]:
            print(f"  expected {r['gold']}  cited {r['cited']}  read {r['sources']}")
        print(f"  A: {r['text'][:300]}".replace("\n", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="questions per set (quick test)")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between Gemini calls")
    parser.add_argument("--search", default="full", choices=["full", "bm25"])
    parser.add_argument("--summarize", type=Path, help="recount a saved run (no Gemini)")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
        return
    from manualmate.api.service import Service

    service = Service(args.search)
    service.start()
    summarize(run(service, args.limit, args.sleep))


if __name__ == "__main__":
    main()
