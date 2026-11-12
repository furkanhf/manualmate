"""Query translation: turn a Hindi/Telugu question into English before search.

Why: bge-m3 finds English pages for Hindi/Telugu in their own script well
(chat 5.2: 12/12 in the top 5), but romanized text ("mere microwave mein nami
kyun aati hai") is weak: 5/8 after rerank, and the reranker gives the right
page a score near 0. Gemini understands romanized Hindi/Telugu.

How:
  1. Gemini translates the whole question to plain English.
  2. Brand names, model numbers and error codes stay exactly as written,
     so the router still finds the manual.
  3. The search (router, dense, reranker) uses the English text. The answer
     is still written in the user's language (generate/, unchanged).

Runs on the laptop (API call, no torch). For the eval,
scripts/make_translations.py saves the translations once in
eval/translations.jsonl, so Colab needs no Gemini key and every run uses the
same text (same design as query_rewriter.py / eval/rewrites.jsonl).
"""

from __future__ import annotations

import time

from manualmate.retrieve.query_rewriter import MODEL_NAME, get_client

PROMPT = """Translate this question from a user of a home appliance (air conditioner, washing machine, refrigerator, microwave oven or Wi-Fi router) into simple, natural English.
The question may be in Hindi or Telugu, in their own script or written with English letters (romanized), or mixed with English words.

Question:
"{question}"

Rules:
- Output ONE line only: the English question. No quotes, no notes, no explanation.
- Keep brand names, model numbers, error codes and button names exactly as written (for example Samsung, FTKS35, KGN56, OE, tAP).
- Do not answer the question. Do not add information that is not in the question."""


TIMEOUT_MS = 15_000     # a one-line translation; a stuck call must not hang the chat


def translate(question: str, retries: int = 3, wait: float = 5.0) -> str:
    """Return the question in English, or "" if Gemini fails after all retries.

    wait: seconds before try 2 (x2 before try 3). The app uses 2 tries, 1 s apart.
    """
    from google.genai import types

    prompt = PROMPT.format(question=question.strip())
    config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=256,
                                         http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    for attempt in range(1, retries + 1):
        t0 = time.perf_counter()
        try:
            response = get_client().models.generate_content(
                model=MODEL_NAME, contents=prompt, config=config)
            text = (response.text or "").strip()
            line = text.splitlines()[0].strip().strip('"').strip() if text else ""
            if line:
                return line
        except Exception as exc:                       # network, quota, timeout, ...
            print(f"Translation failed (try {attempt}/{retries}, "
                  f"{time.perf_counter() - t0:.1f} s): {str(exc)[:300]}")
        if attempt < retries:
            time.sleep(wait * attempt)
    return ""
