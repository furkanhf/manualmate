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
  ask(question, ...)          the app path (week 6): language + translation +
                              the user's manual pick + timings, then the above.

Run this file on the laptop: it takes real chunks from Neon (no search,
we pick the page by hand) and asks Gemini 2 questions:
  1. q001 (Samsung AC filter)  -> expect an answer with [ac_001 p9] tags
  2. "price of this AC"        -> expect a refusal (NOT_IN_MANUAL)
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from manualmate.generate.citations import check
from manualmate.generate.prompts import build_prompt, tag
from manualmate.generate.refusal import gate_after, gate_before, message

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
    # Filled by ask() (the app path); the eval leaves them empty.
    language: str = "English"
    search_text: str = ""          # the text the search used (English translation or the question)
    route_reason: str = ""
    manual_id: str | None = None   # the one manual searched (None if several / asked)
    translate_ms: float | None = None
    retrieve_ms: float | None = None
    total_ms: float | None = None


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
    return [{"tag": tag(c), "manual_id": c["manual_id"],
             "brand": c.get("brand"), "model": c.get("model"),
             "page": int(c["page"]), "score": c.get("score")} for c in chunks]


def generate(question: str, chunks: list[dict], route=None,
             language: str = "English", model: str = MODEL_NAME,
             refuse_below: float | None = None,
             clarify_below: float | None = None) -> Answer:
    """Gate 1 -> Gemini -> gate 2 -> citation check.

    question: the FULL user question.
    chunks:   reranked chunks, best first (each has "score").
    route:    the router's Route (for "Which model?"), or None.
    refuse_below, clarify_below: gate-1 thresholds; None = the defaults in
              refusal.py. The eval passes 0.0 so that Gemini always runs.
    """
    chunks = chunks[:TOP_K]
    limits = {k: v for k, v in (("refuse_below", refuse_below),
                                ("clarify_below", clarify_below)) if v is not None}
    before = gate_before(route, chunks, **limits)
    if not before.answers:
        return Answer(question, before.action, before.reason, before.message,
                      top_score=before.top_score, options=before.options,
                      sources=_sources(chunks))

    system, user = build_prompt(question, chunks, language)
    start = time.perf_counter()
    raw = call_gemini(system, user, model)
    gemini_ms = round((time.perf_counter() - start) * 1000)
    if raw is None:
        return Answer(question, "error", "gemini_error", message("gemini_error", language),
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


def retrieve(question: str, conn, manual_id: str | None = None) -> tuple[list[dict], object]:
    """Parser + router + dense + rerank: the week-2 pipeline (needs torch).

    Returns (top chunks, route). If the router must ask "Which model?",
    the chunk list is empty.
    manual_id: the user's answer to "Which model?" -> search only that manual.
    """
    from manualmate.retrieve import dense, reranker
    from manualmate.retrieve.query_parser import parse
    from manualmate.retrieve.router import route as route_question, route_to

    parsed = parse(question)
    r = route_to(manual_id) if manual_id else route_question(parsed)
    if r.ask:
        return [], r
    issue = parsed.issue or question
    candidates = dense.search(conn, issue, k=CANDIDATES, manual_ids=r.manual_ids)
    return reranker.rerank(issue, candidates, top_k=TOP_K), r   # reranker reads issue text


def retrieve_light(question: str, bm25, manual_id: str | None = None) -> tuple[list[dict], object]:
    """Like retrieve(), but BM25 instead of dense + rerank: no torch.

    For laptop tests of the API and UI only (MANUALMATE_SEARCH=bm25). It is a
    real search, but weaker, and BM25 scores are not 0..1 like rerank scores,
    so gate 1 (REFUSE_BELOW) only refuses when no chunk matches at all.
    """
    from manualmate.retrieve.query_parser import parse
    from manualmate.retrieve.router import route as route_question, route_to

    parsed = parse(question)
    r = route_to(manual_id) if manual_id else route_question(parsed)
    if r.ask:
        return [], r
    return bm25.search(parsed.issue or question, k=TOP_K, manual_ids=r.manual_ids), r


def answer(question: str, conn, language: str = "English") -> Answer:
    """The full pipeline for one question (eval path: no translation)."""
    chunks, r = retrieve(question, conn)
    return generate(question, chunks, route=r, language=language)


def keeps_manual(question: str, manual_id: str) -> bool:
    """Does the manual the user picked earlier still fit this question?

    The UI keeps the picked manual for the chat. But "how do I set up my router"
    after a Daikin AC pick is about another product (chat 5.4 bug): a brand,
    product type or model in the question that the picked manual does not have
    -> False, and ask() routes the question again.
    """
    from manualmate.retrieve.query_parser import get_aliases, parse

    info = get_aliases().manuals.get(manual_id)
    if info is None:
        return False
    parsed = parse(question)
    if parsed.brand and parsed.brand != info["brand"]:
        return False
    if parsed.category and parsed.category != info["category"]:
        return False
    return not parsed.models or manual_id in parsed.models


BRAND_NAMES = {"lg": "LG", "ifb": "IFB", "asus": "ASUS", "tp-link": "TP-Link",
               "d-link": "D-Link", "blue-star": "Blue Star"}


def brand_hint(manual_ids: list[str]) -> dict:
    """For "ask_brand": product name, its brands and an example manual, if all the
    manuals are one product type (e.g. 11 washing machines). Else {}."""
    from manualmate.generate.prompts import PRODUCT_NAMES
    from manualmate.retrieve.query_parser import get_aliases

    manuals = get_aliases().manuals
    infos = [manuals[m] for m in manual_ids if m in manuals]
    categories = {i["category"] for i in infos}
    if len(categories) != 1:
        return {}
    brands = sorted({i["brand"] for i in infos})
    return {"product": PRODUCT_NAMES.get(categories.pop(), "product"),
            "brands": ", ".join(BRAND_NAMES.get(b, b.title()) for b in brands),
            "example": min((i["name"] for i in infos), key=len)}


LANGUAGES = ("English", "Hindi", "Telugu")


# Common romanized Hindi / Telugu words (chat 5.4: "mera daikin ac bhot awaaz
# karta hai kya karu?" was read as English -> no translation -> refused).
# Words that are also English ("me", "ho", "to", "band", "na", "lo") are left out.
# 2+ hits = that language; checked against all English test questions (0 hits >= 2).
ROMAN_HINDI = frozenset("""
hai hain tha thi kya kyu kyun kyon kaise kaisa kab kitna kitni kitne kaun kahan
mera meri mere mujhe hamara apna apne karta karti karte karu karun karein kare karo
karna kiya nahi nahin nhi raha rahi rahe hota hoti hote ka ki ke mein mai
aur bahut bhot bohot bahot chahiye sakta sakti sakte batao bataiye bataye wala wali
kuch abhi toh yeh ye woh wo baad pehle saaf awaaz aawaz pani paani thanda garam chal
chalta chalti ruk raha jaata jata jaati
""".split())
ROMAN_TELUGU = frozenset("""
enti emiti emi ela elaa cheyali cheyyali chesthundi chestundi cheyyadam ledu ledhu
undi unnadi unnayi nenu naaku maa mee meeru enduku avutundi avtundi vastundi
ostundi raavatledu ravatledu kaavali kavali gurinchi chala chaala
cheyyatledu panicheyyatledu cheppandi cheppu adi idi ekkada eppudu entha
naa chesina chesanu tarvata taruvata kuda kooda edaina emaina aithe ayithe
""".split())
WORD_RE = re.compile(r"[a-z]+")


def detect_language(text: str) -> str:
    """Hindi or Telugu from the script or from common romanized words, else English.

    The UI dropdown can still force a language if this guesses wrong.
    """
    devanagari = sum("ऀ" <= ch <= "ॿ" for ch in text)
    telugu = sum("ఀ" <= ch <= "౿" for ch in text)
    if max(devanagari, telugu) > 0:
        return "Hindi" if devanagari >= telugu else "Telugu"
    words = WORD_RE.findall(text.lower())
    hindi = sum(w in ROMAN_HINDI for w in words)
    telugu_words = sum(w in ROMAN_TELUGU for w in words)
    if max(hindi, telugu_words) >= 2:
        return "Hindi" if hindi >= telugu_words else "Telugu"
    return "English"


def ask(question: str, conn=None, manual_id: str | None = None,
        language: str | None = None, bm25=None) -> Answer:
    """The app path (API and UI): translate -> search -> generate, with timings.

    manual_id: the manual the user picked after "Which model?" (or None).
    language:  "English" | "Hindi" | "Telugu", or None = detect from the script.
    bm25:      a BM25Index -> light search (laptop, no torch); None -> dense + rerank.

    Hindi/Telugu: the search uses Gemini's English translation (the eval did the
    same with saved translations); Gemini answers the ORIGINAL question in the
    user's language. If the translation fails, the search uses the question.
    """
    from manualmate.retrieve.translator import translate

    start = time.perf_counter()
    language = language if language in LANGUAGES else detect_language(question)

    search_text, translate_ms = question, None
    if language != "English":
        t0 = time.perf_counter()
        search_text = translate(question, retries=2) or question
        translate_ms = round((time.perf_counter() - t0) * 1000)

    if manual_id and not keeps_manual(search_text, manual_id):
        manual_id = None                     # new product in the question: route again

    t0 = time.perf_counter()
    if bm25 is not None:
        chunks, r = retrieve_light(search_text, bm25, manual_id)
    else:
        chunks, r = retrieve(search_text, conn, manual_id)
    retrieve_ms = round((time.perf_counter() - t0) * 1000)

    a = generate(question, chunks, route=r, language=language)
    if a.action != "answer":                 # "Which model?", refusals: user's language
        reason, fill = a.reason, {}
        if reason == "ask_model":
            # no buttons: product type known -> list its brands, else ask for brand +
            # product; a model that fits 2+ manuals (FTKS: Daikin K and L) -> say so
            if not a.options:
                fill = brand_hint(r.manual_ids)
                reason = "ask_brand" if fill else "ask_product"
            elif r.reason == "model":
                reason = "ask_series"
        a.text = (message(reason, language) or a.text).format(**fill)
    a.language, a.search_text, a.route_reason = language, search_text, r.reason
    a.manual_id = r.manual_ids[0] if r.is_single else None
    a.translate_ms, a.retrieve_ms = translate_ms, retrieve_ms
    a.total_ms = round((time.perf_counter() - start) * 1000)
    return a


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
