"""Query rewriting: translate the user's words into the manual's words.

Why: users and manuals use different words for the same thing.
  user:   "my microwave smells of fish"
  manual: "DEODORIZE - this function can flash the air in the oven"
Dense search and the reranker cannot connect them, because the words
"smell" or "odour" are not on the page. Gemini knows both kinds of words.

How:
  1. Gemini reads the question (without brand and model words).
  2. Gemini writes 5-12 short phrases that the manual page would use.
  3. We search with: the question + those phrases (expand(), below).
     We ADD the phrases and keep the question, so a bad rewrite cannot
     hide what the user asked.
  4. The reranker still reads only the original question.

Runs on the laptop (Gemini is an API call: no torch needed).
For the eval, scripts/make_rewrites.py saves the rewrites of the 50 test
questions in eval/rewrites.jsonl, so Colab does not need the Gemini key and
every eval run uses the same rewrites.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

MODEL_NAME = "gemini-3.5-flash-lite"   # fast and cheap; pinned (not "-latest")

PRODUCT_NAMES = {"ac": "air conditioner", "washing-machine": "washing machine",
                 "refrigerator": "refrigerator", "microwave": "microwave oven",
                 "router": "Wi-Fi router"}

PROMPT = """You help a search engine find the right page in a home-appliance user manual.

A user asked this question about their {product}:
"{question}"

Write the words and short phrases that the manual page which answers this question would most likely contain:
section titles, function and button names, troubleshooting table wording, display messages.

Rules:
- Output ONE line only: 5 to 12 short phrases, separated by commas.
- Use manual wording, for example "deodorize", "remove odour", "the time is not displayed", "child lock", "drain filter".
- Do not answer the question. Do not explain.
- Do not write brand names, model names or error codes that are not in the question."""


@lru_cache(maxsize=1)
def get_client():
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def rewrite(question: str, category: str | None = None, retries: int = 3) -> str:
    """Return the manual words for a question, as one comma-separated line.

    question: the issue text from the parser (no brand or model words).
    category: from the parser ("microwave", ...); None = unknown product.
    Returns "" if Gemini fails after all retries (then we search with the
    question only, as before).
    """
    from google.genai import types

    product = PRODUCT_NAMES.get(category or "", "home appliance")
    prompt = PROMPT.format(product=product, question=question.strip())
    config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=256)

    for attempt in range(1, retries + 1):
        try:
            response = get_client().models.generate_content(
                model=MODEL_NAME, contents=prompt, config=config)
            text = (response.text or "").strip()
            # Keep the first line only, and remove quotes or a "Phrases:" label.
            line = text.splitlines()[0] if text else ""
            if ":" in line[:20]:
                line = line.split(":", 1)[1]
            return line.strip().strip('"').strip()
        except Exception as error:          # rate limit, network, ...
            if attempt == retries:
                print(f"  rewrite failed: {error}")
                return ""
            time.sleep(5 * attempt)          # wait 5 s, then 10 s
    return ""


def expand(question: str, rewritten: str) -> str:
    """The text we search with: the question, then the manual words."""
    if not rewritten:
        return question
    return f"{question}\nManual terms: {rewritten}"
