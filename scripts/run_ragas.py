"""RAGAS faithfulness on the saved answers of an answer-eval run.

What it is like: a fact-checker with the article and the reporter's notes.
For every claim in the article, the checker asks: "Do the notes support it?"
Score = supported claims / all claims (1.0 = every claim is in the manual).

Why: the citation check only proves that a tag points to a page we gave
Gemini. Faithfulness checks that the page really says what the answer says.
Resume target: 0.90.

How (for each answered in-scope question):
  1. Answer text without tags (strip_tags).
  2. Context = the text of the answer's 5 source pages, from Neon.
     NOTE: the 2026-11-10_1047 run saved page tags, not chunk ids, so the
     judge reads whole pages (a little more than Gemini saw -> slightly lenient).
  3. RAGAS Faithfulness with a Gemini judge (a stronger model than the one
     that wrote the answers): split into claims -> check each claim.

Runs on the laptop (no torch, no GPU). Needs GEMINI_API_KEY and DATABASE_URL in .env, and the openai package (comes with ragas).
    python scripts/run_ragas.py --limit 3          # quick test
    python scripts/run_ragas.py                    # all answered in-scope questions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
DEFAULT_ANSWERS = RESULTS / "2026-11-10_1047_answers.jsonl"
# Judge model. gemini-3.5-flash would be the better judge (it did not write the
# answers), but its free tier allows only 20 requests per day, and RAGAS needs
# 2+ calls per answer (claims, then verdicts). So the default is flash-lite, the
# same model that wrote the answers (possible self-bias: write this next to the score).
# Spot-check: python scripts/run_ragas.py --limit 8 --judge gemini-3.5-flash
JUDGE_MODEL = "gemini-3.5-flash-lite"
TARGET = 0.90


def page_contexts(conn, sources: list[str]) -> list[str]:
    """One context string per source page (unique, in rank order)."""
    from manualmate.generate.answer import chunks_for_pages
    from manualmate.generate.citations import find_tags

    pages = []
    for s in sources:
        for t in find_tags(s):
            if t not in pages:
                pages.append(t)
    out = []
    for manual_id, page in pages:
        chunks = chunks_for_pages(conn, manual_id, [page])
        text = "\n".join(c["text"] for c in chunks)
        if text.strip():
            # Say which manual the page is from. RAGAS rewrites "the appliance" in the
            # answer to "the Bosch KGN56 fridge" (no pronouns), but the page text has no
            # model name (the parser removes page headers). Without this line the judge
            # marks true claims as "model not mentioned" (q029, q048, q049 -> 0.0).
            c = chunks[0]
            label = " ".join(str(x) for x in (c.get("brand"), c.get("model"), c.get("category")) if x)
            out.append(f"Source: {label} manual ({manual_id}), page {page}\n{text}")
    return out


GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_metric(judge: str):
    """RAGAS Faithfulness with a Gemini judge.

    RAGAS scores with async calls, but it wraps the google-genai client in
    sync mode only (and instructor has a known Gemini safety-settings bug).
    So we use Gemini's official OpenAI-compatible endpoint with an async
    OpenAI client - the workaround that the RAGAS source code recommends.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    client = AsyncOpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_OPENAI_URL)
    # ragas default max_tokens = 1024. Gemini 3.5 "thinks" first, and the thinking
    # tokens count too -> IncompleteOutputException. So: more room, less thinking.
    llm = llm_factory(judge, provider="openai", client=client,
                      max_tokens=8192, reasoning_effort="low")
    return Faithfulness(llm=llm)


async def score_one(metric, question: str, answer: str, contexts: list[str],
                    retries: int = 3) -> tuple[float | None, list[dict]]:
    """The same 2 steps as Faithfulness.ascore (ragas 0.4.3), but we KEEP the
    claims, verdicts and reasons, so a low score can be checked by hand."""
    for attempt in range(1, retries + 1):
        try:
            statements = await metric._create_statements(question, answer)
            if not statements:
                return None, []
            verdicts = await metric._create_verdicts(statements, "\n".join(contexts))
            details = [{"claim": v.statement, "verdict": int(v.verdict), "reason": v.reason}
                       for v in verdicts.statements]
            if not details:
                return None, []
            return sum(d["verdict"] for d in details) / len(details), details
        except Exception as error:          # rate limit, network, bad JSON, ...
            if attempt == retries:
                print(f"  judge failed: {type(error).__name__}: {str(error)[:300]}")
                return None, []
            await asyncio.sleep(10 * attempt)
    return None, []


async def score_all(rows: list[dict], contexts_of: dict, judge: str, sleep: float,
                    debug: bool = False) -> list[dict]:
    """All judge calls in ONE event loop (the async client must not change loops)."""
    from manualmate.generate.citations import strip_tags

    metric = make_metric(judge)
    per_q = []
    for r in rows:
        contexts = contexts_of[r["id"]]
        start = time.perf_counter()
        score, details = await score_one(metric, r["question"], strip_tags(r["raw"]), contexts)
        ms = round((time.perf_counter() - start) * 1000)
        per_q.append({"id": r["id"], "faithfulness": score, "n_contexts": len(contexts),
                      "gold_retrieved": r.get("gold_retrieved"), "judge_ms": ms,
                      "claims": details})
        shown = "FAILED" if score is None else f"{score:.2f}"
        print(f"  {r['id']:5s} faithfulness={shown}  contexts={len(contexts)}  {ms} ms")
        if debug:
            for d in details:
                print(f"      [{d['verdict']}] {d['claim'][:110]}\n          reason: {d['reason'][:200]}")
        await asyncio.sleep(sleep)
    return per_q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--limit", type=int, help="only the first N answers")
    parser.add_argument("--judge", default=JUDGE_MODEL)
    parser.add_argument("--ids", help="only these question ids, e.g. q021,q049 (debug)")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds to wait after each question (rate limit)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    from manualmate.index.db import get_connection

    rows = [json.loads(l) for l in args.answers.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if r["set"] == "text" and r["action"] == "answer" and r.get("raw")]
    if args.ids:
        rows = [r for r in rows if r["id"] in args.ids.split(",")]
    rows = rows[:args.limit] if args.limit else rows
    print(f"{len(rows)} answers from {args.answers.name}, judge {args.judge}")

    with get_connection() as conn:          # page texts first (Neon, no Gemini)
        contexts_of = {r["id"]: page_contexts(conn, r["sources"]) for r in rows}
    per_q = asyncio.run(score_all(rows, contexts_of, args.judge, args.sleep, debug=bool(args.ids)))

    scored = [q["faithfulness"] for q in per_q if q["faithfulness"] is not None]
    mean = round(sum(scored) / len(scored), 3) if scored else None
    low = sorted((q for q in per_q if q["faithfulness"] is not None and q["faithfulness"] < 1.0),
                 key=lambda q: q["faithfulness"])
    summary = {
        "answers_file": args.answers.name, "judge": args.judge,
        "n": len(per_q), "scored": len(scored), "failed": len(per_q) - len(scored),
        "faithfulness_mean": mean, "target": TARGET,
        "below_1": [(q["id"], q["faithfulness"]) for q in low],
        "context": "whole source pages from Neon (run saved page tags, not chunk ids)",
        "per_question": per_q,
    }
    print(f"\n===== RAGAS faithfulness: {mean}  (target {TARGET}, scored {len(scored)}/{len(per_q)})")
    print("Below 1.0:", summary["below_1"])

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    suffix = (f"_limit{args.limit}" if args.limit else "") + ("_debug" if args.ids else "")
    out = RESULTS / f"{stamp}_ragas{suffix}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
