"""Week-3 eval: run the full pipeline on all test questions and count the scores.

What it is like: a teacher who marks 90 exams with an answer key.

Question sets:
  text      eval/text_queries.jsonl     50  must be ANSWERED, from the right page
  oos       eval/out_of_scope.jsonl     30  must be REFUSED (or "Which model?")
  clarify   eval/clarify_queries.jsonl  10  router must ask "Which model?" when expected

Gate 1 is OFF during the run (thresholds 0.0): Gemini answers every question,
and we save the top rerank score. Then summarize() tries many thresholds on
the saved results, with no new Gemini calls. We pick the threshold from data.

Run on Colab (needs torch for bge-m3 + reranker, and GEMINI_API_KEY):
    python scripts/run_answer_eval.py
    python scripts/run_answer_eval.py --limit 5          # quick test, 5 per set
Other in-scope questions (only this set runs, no oos/clarify):
    python scripts/run_answer_eval.py --queries eval/text_queries_new.jsonl
    python scripts/run_answer_eval.py --queries eval/multilingual_queries.jsonl --translations
      --translations: search with the saved English translation (eval/translations.jsonl),
      but Gemini gets the ORIGINAL question and answers in its language (hi/te field).
Recount a saved run (laptop, no torch, no Gemini):
    python scripts/run_answer_eval.py --summarize eval/results/<file>.jsonl

Output: eval/results/<date>_answers.jsonl (one line per question)
        eval/results/<date>_answers_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
LANGUAGES = {"en": "English", "hi": "Hindi", "te": "Telugu"}
THRESHOLDS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def load(name: str) -> list[dict]:
    path = EVAL / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run(limit: int | None, sleep: float, queries: Path | None = None,
        translations: bool = False) -> Path:
    from manualmate.generate.answer import generate, retrieve
    from manualmate.index.db import get_connection
    from manualmate.retrieve.query_parser import parse
    from manualmate.retrieve.router import route

    if queries:
        sets = {"text": [json.loads(line) for line in
                         queries.read_text(encoding="utf-8").splitlines() if line.strip()]}
    else:
        sets = {"text": load("text_queries.jsonl"), "oos": load("out_of_scope.jsonl"),
                "clarify": load("clarify_queries.jsonl")}
    english = {}
    if translations:
        english = {r["id"]: r["translation"] for r in load("translations.jsonl")}
        missing = [q["id"] for q in sets["text"] if not english.get(q["id"])]
        if missing:
            raise SystemExit(f"No translation for: {', '.join(missing)} "
                             "(run scripts/make_translations.py on the laptop)")
        print(f"Translations: {len(english)} loaded")
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = f"_{queries.stem}" if queries else ""
    out_path = EVAL / "results" / f"{stamp}_answers{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn, out_path.open("w", encoding="utf-8") as out:
        for set_name, queries in sets.items():
            queries = queries[:limit] if limit else queries
            print(f"\n=== {set_name}: {len(queries)} questions")
            for q in queries:
                row = {"set": set_name, "id": q["id"], "question": q["question"],
                       "expect": q.get("expect", "answer" if set_name == "text" else "refuse")}
                language = LANGUAGES.get(q.get("language", "en"), "English")
                search_text = english.get(q["id"], q["question"])   # translation or original
                if "language" in q:
                    row["language"] = q["language"]
                if q["id"] in english:
                    row["search_text"] = search_text
                if set_name == "clarify":          # router only, no Gemini
                    r = route(parse(q["question"]))
                    row.update(action="ask" if r.ask else "answer",
                               reason="ask_model" if r.ask else "ok",
                               manual_ids=r.manual_ids)
                else:
                    start = time.perf_counter()
                    chunks, r = retrieve(search_text, conn)
                    if set_name == "text" and r.ask:
                        # In-scope question, router asks "Which model?" (e.g. FTKS is in
                        # both Daikin K and L manuals). Act like the user: pick the
                        # labeled manual, then answer. Counted as asked_first.
                        row["asked_first"] = True
                        chunks, r = retrieve(search_text, conn, manual_id=q["manual_id"])
                    retrieve_ms = round((time.perf_counter() - start) * 1000)
                    a = generate(q["question"], chunks, route=r, language=language,
                                 refuse_below=0.0, clarify_below=0.0)
                    row.update(action=a.action, reason=a.reason, top_score=a.top_score,
                               raw=a.raw, citations=a.citations,
                               sources=[s["tag"] for s in a.sources],
                               retrieve_ms=retrieve_ms, gemini_ms=a.gemini_ms)
                    if set_name == "text":
                        gold = {f"{q['manual_id']} p{p}" for p in q["pages"]}
                        row["gold"] = sorted(gold)
                        row["gold_retrieved"] = any(s.strip("[]") in gold for s in row["sources"])
                        row["gold_cited"] = bool(gold & set((a.citations or {}).get("tags", [])))
                    if a.gemini_ms is not None:
                        time.sleep(sleep)          # stay under the Gemini rate limit
                score = row.get("top_score")
                print(f"  {q['id']:5s} {row['action']:6s} {row['reason']:14s} "
                      f"score={score if score is None else round(score, 3)}")
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    print(f"\nSaved {out_path}")
    return out_path


def decide(row: dict, threshold: float) -> str:
    """The final action at a gate-1 threshold (refuse below it)."""
    if row["action"] in ("ask", "error"):
        return row["action"]
    if (row.get("top_score") or 0.0) < threshold:
        return "refuse"
    return row["action"]            # "answer", or "refuse" from NOT_IN_MANUAL


def recheck(rows: list[dict]) -> None:
    """Run the CURRENT citation checker again on the saved Gemini answers.

    A fix in citations.py then changes the numbers without new Gemini calls.
    """
    from manualmate.generate.citations import check, find_tags

    for r in rows:
        if r.get("action") != "answer" or not r.get("raw") or not r.get("sources"):
            continue
        chunks = [{"manual_id": m, "page": p}
                  for tag in r["sources"] for m, p in find_tags(tag)]
        r["citations"] = check(r["raw"], chunks)
        if "gold" in r:
            r["gold_cited"] = bool(set(r["gold"]) & set(r["citations"]["tags"]))


def pct(n: int, d: int) -> float:
    return round(n / d, 3) if d else 0.0


def summarize(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    recheck(rows)
    text = [r for r in rows if r["set"] == "text"]
    oos = [r for r in rows if r["set"] == "oos"]
    clar = [r for r in rows if r["set"] == "clarify"]

    answered = [r for r in text if r["action"] == "answer"]
    cits = [r["citations"] for r in answered if r.get("citations")]
    summary = {
        "file": path.name,
        "text": {
            "n": len(text),
            "answered": pct(len(answered), len(text)),
            "wrongly_refused": pct(sum(r["action"] == "refuse" for r in text), len(text)),
            "errors": sum(r["action"] == "error" for r in text),
            "asked_first": [r["id"] for r in text if r.get("asked_first")],
            "gold_page_in_top5": pct(sum(r.get("gold_retrieved", False) for r in text), len(text)),
            "gold_page_cited": pct(sum(r.get("gold_cited", False) for r in text), len(text)),
            "citations_ok": pct(sum(c["ok"] for c in cits), len(cits)),
            "citation_precision": round(sum(c["precision"] for c in cits) / len(cits), 3) if cits else 0.0,
            "answers_with_invalid_tag": sum(bool(c["invalid_tags"]) for c in cits),
        },
        "oos": {
            "n": len(oos),
            "refused_or_asked": pct(sum(r["action"] in ("refuse", "ask") for r in oos), len(oos)),
            "answered_ids": [r["id"] for r in oos if r["action"] == "answer"],
        },
        "clarify": {
            "n": len(clar),
            "correct": pct(sum(r["action"] == ("ask" if r["expect"] == "ask" else "answer")
                               for r in clar), len(clar)),
        },
        "by_language": {
            lang: {"n": len(rs),
                   "answered": pct(sum(r["action"] == "answer" for r in rs), len(rs)),
                   "gold_page_cited": pct(sum(r.get("gold_cited", False) for r in rs), len(rs))}
            for lang in sorted({r["language"] for r in text if "language" in r})
            for rs in [[r for r in text if r.get("language") == lang]]
        },
        "gemini_ms_avg": round(sum(r["gemini_ms"] for r in rows if r.get("gemini_ms"))
                               / max(1, sum(1 for r in rows if r.get("gemini_ms")))),
        "thresholds": [],
    }
    for t in THRESHOLDS:
        summary["thresholds"].append({
            "refuse_below": t,
            "text_answered": pct(sum(decide(r, t) == "answer" for r in text), len(text)),
            "oos_refused": pct(sum(decide(r, t) in ("refuse", "ask") for r in oos), len(oos)),
        })

    s = summary
    print(f"\n===== SUMMARY ({path.name})")
    print(f"In-scope ({s['text']['n']}): answered {s['text']['answered']}, wrongly refused "
          f"{s['text']['wrongly_refused']}, errors {s['text']['errors']}, "
          f"asked first (then user picked the model) {s['text']['asked_first']}")
    print(f"  gold page in top 5 {s['text']['gold_page_in_top5']}, gold page cited "
          f"{s['text']['gold_page_cited']}")
    print(f"  citations ok {s['text']['citations_ok']}, precision "
          f"{s['text']['citation_precision']}, answers with a made-up tag "
          f"{s['text']['answers_with_invalid_tag']}")
    for lang, b in s["by_language"].items():
        print(f"  {lang}: n {b['n']}, answered {b['answered']}, gold page cited "
              f"{b['gold_page_cited']}")
    print(f"Out-of-scope ({s['oos']['n']}): refused or asked {s['oos']['refused_or_asked']}  "
          f"answered: {s['oos']['answered_ids']}")
    print(f"Clarify ({s['clarify']['n']}): correct {s['clarify']['correct']}")
    print(f"Gemini avg {s['gemini_ms_avg']} ms")
    print("\nGate-1 threshold | in-scope answered | out-of-scope refused")
    for row in s["thresholds"]:
        print(f"  {row['refuse_below']:5.2f}          | {row['text_answered']:5.2f}"
              f"             | {row['oos_refused']:5.2f}")

    out = path.with_name(path.stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N questions of each set")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds to wait after each Gemini call (rate limit)")
    parser.add_argument("--queries", type=Path,
                        help="run ONLY this in-scope question file (no oos/clarify)")
    parser.add_argument("--translations", action="store_true",
                        help="search with eval/translations.jsonl (Hindi/Telugu questions)")
    parser.add_argument("--summarize", type=Path, help="recount a saved answers .jsonl")
    args = parser.parse_args()
    path = args.summarize or run(args.limit, args.sleep, args.queries, args.translations)
    summarize(path)


if __name__ == "__main__":
    main()
