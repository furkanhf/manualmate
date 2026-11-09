"""Prompts for the answer step: an open-book exam.

What it is like: the model gets 5 pages from the manual and must answer
only from those pages. After every fact it writes the page it used,
like a student who must show the source for each line.

Why: without these rules, Gemini fills gaps from its own memory. Then the
answer sounds right but does not match this manual (for example a wrong
button name or a wrong error code).

The rules in the prompt:
  1. Use only the text in the SOURCES block.
  2. Put a tag like [mw_003 p24] after every fact.
     citations.py checks that every tag is one of the 5 sources.
  3. If the sources do not answer the question, write only NOT_IN_MANUAL.
     refusal.py turns that into a polite refusal.
  4. Answer in the user's language (English now; Hindi/Telugu in week 4).

Runs on the laptop (no torch). Run this file to print an example prompt.
"""

from __future__ import annotations

NOT_IN_MANUAL = "NOT_IN_MANUAL"

PRODUCT_NAMES = {"ac": "air conditioner", "washing-machine": "washing machine",
                 "refrigerator": "refrigerator", "microwave": "microwave oven",
                 "router": "Wi-Fi router"}

SYSTEM_PROMPT = f"""You are ManualMate. You answer questions about home appliances using ONLY the manual pages in the SOURCES block.

Rules:
1. Use only facts that are written in the SOURCES. Do not use your own knowledge, even if you are sure.
2. After every sentence that states a fact, write the tag of the source you used, exactly as it is shown, for example [mw_003 p24]. If two sources support a fact, write both tags.
3. Never write a tag that is not in the SOURCES.
4. If the SOURCES do not contain the answer, write only this word and nothing else: {NOT_IN_MANUAL}
5. If the SOURCES answer only part of the question, answer that part and say which part the manual does not cover.
6. Keep the manual's names for buttons, modes, menus and error codes exactly as written.
7. For a procedure, use numbered steps. Keep the answer short: at most about 150 words.
8. Write the answer in {{language}}. Keep tags, button names and error codes as they are in the SOURCES."""

USER_PROMPT = """SOURCES:
{sources}

QUESTION about the user's {product}:
{question}

ANSWER:"""


def tag(chunk: dict) -> str:
    """The citation tag for one chunk, for example "[mw_003 p24]".

    citations.py uses the same function, so the tag format is in one place.
    """
    return f"[{chunk['manual_id']} p{int(chunk['page'])}]"


def format_sources(chunks: list[dict]) -> str:
    """Write the chunks as numbered sources, each with its tag on top."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        header = f"Source {i} {tag(c)} - {c.get('brand', '')} {c.get('model', '')}".strip()
        parts.append(f"{header}\n{c['text'].strip()}")
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict],
                 language: str = "English") -> tuple[str, str]:
    """Return (system prompt, user prompt) for Gemini.

    question: the FULL user question (the answer must fit what they asked,
              not only the issue text that the search used).
    chunks:   the top 5 reranked chunks.
    language: the language of the answer.
    """
    category = chunks[0].get("category") if chunks else None
    product = PRODUCT_NAMES.get(category or "", "home appliance")
    system = SYSTEM_PROMPT.format(language=language)
    user = USER_PROMPT.format(sources=format_sources(chunks),
                              product=product, question=question.strip())
    return system, user


if __name__ == "__main__":
    # Example with 2 fake chunks: check that the prompt reads well.
    fake = [
        {"manual_id": "mw_003", "brand": "lg", "model": "mc2846bg",
         "category": "microwave", "page": 24,
         "text": "GHEE / BAKERY: press Auto Cook until '6h' appears for Ghee. ..."},
        {"manual_id": "mw_003", "brand": "lg", "model": "mc2846bg",
         "category": "microwave", "page": 157,
         "text": "Paneer Tikka. Ingredients: 2 tbsp ghee, 200 g paneer ..."},
    ]
    system, user = build_prompt("How do I make ghee in my LG MC2846BG?", fake)
    print("===== SYSTEM =====\n" + system)
    print("\n===== USER =====\n" + user)
