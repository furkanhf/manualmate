"""Answer: the full path from a question to a reply.

What it is like: the manager of a shop. The manager does not do the work.
They pass the question along the line and give one reply to the user:
  router -> dense search -> rerank -> gate 1 -> Gemini -> gate 2 -> citation check

Why: each file does one job. This file connects them into one function,
so the eval, the API and the UI (week 6) all call the same code.

Two parts:
  generate(question, chunks)  gate 1 + Gemini + gate 2 + citations.
                              Runs on the laptop (Gemini is an API call).
  answer(question, conn)      parser + router + dense + rerank + generate().
                              Needs torch (bge-m3 + reranker) -> Colab for now.

Run this file on the laptop: it takes real chunks from Neon (no search,
we pick the page by hand) and asks Gemini 2 questions:
  1. q001 (Samsung AC filter)  -> expect an answer with [ac_001 p9] tags
  2. "price of this AC"        -> expect a refusal (NOT_IN_MANUAL)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

from manualmate.generate.citations import check
from manualmate.generate.prompts import build_prompt, tag
from manualmate.generate.refusal import gate_after, gate_before

MODEL_NAME = "gemini-3.5-flash-lite"   # same model as the query rewriter; pinned
TOP_K = 5                              # chunks that Gemini reads
CANDIDATES = 20                        # dense results that the reranker reads
MAX_OUTPUT_TOKENS = 1024


@dataclass
class Answer:
    """One reply.

    action:    "answer" | "ask" | "refuse" | "error"
    reason:    "ok" | "ask_model" | "clarify" | "low_score" | "not_in_manual" | "gemini_error"
    text:      what the user sees (the answer with tags, or the message)
    raw:       Gemini's text as it came back ("" if Gemini was not called)
    citations: the result of citations.check() (empty if no answer)
    sources:   the chunks Gemini read: tag, brand, model, page, score
    """
    question: str
    action: str
    reason: str
    text: str
    raw: str = ""
    top_score: float | None = None
    options: list = field(default_factory=list)
    citations: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    gemini_ms: float | None = None


@lru_cache(maxsize=1)
def get_client():
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def call_gemini(system: str, user: str, model: str = MODEL_NAME,
                retries: int = 3) -> str | None:
    """Send the prompt to Gemini. Return its text, or None if it fails."""
    from google.genai import types

    config = types.GenerateContentConfig(system_instruction=system, temperature=0.0,
                                         max_output_tokens=MAX_OUTPUT_TOKENS)
    for attempt in range(1, retries + 1):
        try:
            response = get_client().models.generate_content(
                model=model, contents=user, config=config)
            return (response.text or "").strip()
        except Exception as error:          # rate limit, network, ...
            if attempt == retries:
                print(f"  Gemini failed: {error}")
                return None
            time.sleep(5 * attempt)          # wait 5 s, then 10 s
    return None


def _sources(chunks: list[dict]) -> list[dict]:
    return [{"tag": tag(c), "brand": c.get("brand"), "model": c.get("model"),
             "page": int(c["page"]), "score": c.get("score")} for c in chunks]


def generate(question: str, chunks: list[dict], route=None,
             language: str = "English", model: str = MODEL_NAME) -> Answer:
    """Gate 1 -> Gemini -> gate 2 -> citation check.

    question: the FULL user question.
    chunks:   reranked chunks, best first (each has "score").
    route:    the router's Route (for "Which model?"), or None.
    """
    chunks = chunks[:TOP_K]
    before = gate_before(route, chunks)
    if not before.answers:
        return Answer(question, before.action, before.reason, before.message,
                      top_score=before.top_score, options=before.options,
                      sources=_sources(chunks))

    system, user = build_prompt(question, chunks, language)
    start = time.perf_counter()
    raw = call_gemini(system, user, model)
    gemini_ms = round((time.perf_counter() - start) * 1000)
    if raw is None:
        return Answer(question, "error", "gemini_error",
                      "Sorry, the answer service is not available now. Please try again.",
                      top_score=before.top_score, sources=_sources(chunks),
                      gemini_ms=gemini_ms)

    after = gate_after(raw, before.top_score)
    if not after.answers:
        return Answer(question, after.action, after.reason, after.message, raw=raw,
                      top_score=before.top_score, sources=_sources(chunks),
                      gemini_ms=gemini_ms)

    return Answer(question, "answer", "ok", raw, raw=raw, top_score=before.top_score,
                  citations=check(raw, chunks), sources=_sources(chunks),
                  gemini_ms=gemini_ms)


def retrieve(question: str, conn) -> tuple[list[dict], object]:
    """Parser + router + dense + rerank: the week-2 pipeline (needs torch).

    Returns (top chunks, route). If the router must ask "Which model?",
    the chunk list is empty.
    """
    from manualmate.retrieve import dense, reranker
    from manualmate.retrieve.query_parser import parse
    from manualmate.retrieve.router import route as route_question

    parsed = parse(question)
    r = route_question(parsed)
    if r.ask:
        return [], r
    issue = parsed.issue or question
    candidates = dense.search(conn, issue, k=CANDIDATES, manual_ids=r.manual_ids)
    return reranker.rerank(issue, candidates, top_k=TOP_K), r   # reranker reads issue text


def answer(question: str, conn, language: str = "English") -> Answer:
    """The full pipeline for one question."""
    chunks, r = retrieve(question, conn)
    return generate(question, chunks, route=r, language=language)


def chunks_for_pages(conn, manual_id: str, pages: list[int]) -> list[dict]:
    """Get the chunks of some pages straight from Neon (no search, no torch).

    For laptop tests only.
    """
    rows = conn.execute(
        """SELECT chunk_id, manual_id, brand, model, category, page, text
           FROM chunks WHERE manual_id = %s AND page = ANY(%s)
           ORDER BY page, chunk_id""", (manual_id, pages)).fetchall()
    keys = ["chunk_id", "manual_id", "brand", "model", "category", "page", "text"]
    return [dict(zip(keys, row)) for row in rows]


def show(a: Answer) -> None:
    print(f"Q: {a.question}")
    print(f"-> {a.action} ({a.reason}), top score {a.top_score}, Gemini {a.gemini_ms} ms")
    print(f"   sources: {[s['tag'] for s in a.sources]}")
    print("   text:\n" + "\n".join("     " + line for line in a.text.splitlines()))
    if a.citations:
        c = a.citations
        print(f"   citations: ok={c['ok']} tags={c['tags']} invalid={c['invalid_tags']} "
              f"untagged={len(c['untagged'])}")
        for s in c["untagged"]:
            print(f"     untagged: {s}")


if __name__ == "__main__":
    from manualmate.index.db import get_connection

    with get_connection() as conn:
        # q001: Samsung AC, answer on page 9. Add page 10 as a "wrong" source,
        # like the real top 5 would have. The score 0.90 is a fake rerank score
        # (no reranker on the laptop), so gate 1 lets the question through.
        chunks = chunks_for_pages(conn, "ac_001", [9, 10])
        for c in chunks:
            c["score"] = 0.90
        print(f"Got {len(chunks)} chunks from Neon: {[c['chunk_id'] for c in chunks]}\n")

        show(generate("How often should I wash the filter of my Samsung split AC, "
                      "and how do I clean it?", chunks))
        print()
        show(generate("What is the price of this Samsung AC in India?", chunks))
