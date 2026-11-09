"""Refusal: decide when ManualMate must NOT answer.

What it is like: a careful shop assistant.
  - "Which model do you have?"        -> they do not know the product yet.
  - "Can you tell me more?"           -> they found only weak matches.
  - "That is not in the manual."      -> they found nothing useful, or they
                                         read the page and it does not answer.
Why: a wrong answer that sounds sure is worse than "not in the manual".
Target: refuse 90% of the questions that the manuals cannot answer
(eval/out_of_scope.jsonl), and still answer the questions they can.

Three gates, in order:
  Gate 0 (router):   more than one manual matches -> ask "Which model?".
  Gate 1 (score):    before Gemini. Look at the top rerank score.
                       score <  REFUSE_BELOW   -> refuse (no Gemini call)
                       score <  CLARIFY_BELOW  -> ask a follow-up question
                       else                    -> go to Gemini
  Gate 2 (Gemini):   after Gemini. Answer is NOT_IN_MANUAL -> refuse.

IMPORTANT: the two thresholds below are TEMPORARY guesses.
We set the real values from data: the top rerank scores of the 50 in-scope
questions and the ~30 out-of-scope questions (run on Colab).
A high score does not prove the page is right (q035: wrong page, 0.758),
so gate 2 is still needed.

Runs on the laptop (no torch). Run this file to see 6 example decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from manualmate.generate.citations import is_refusal

# TEMPORARY - set from data (see the docstring).
REFUSE_BELOW = 0.10
CLARIFY_BELOW = 0.30

MESSAGES = {
    "ask_model": "Which model do you have? I found more than one manual that fits.",
    "clarify": ("I am not sure which part of the manual you need. "
                "Can you describe the problem in more detail? "
                "For example: what you see on the display, or what the appliance does."),
    "low_score": ("I could not find this in the manual. "
                  "Please contact the brand's service centre, or ask about "
                  "how to use, clean, set up or troubleshoot the product."),
    "not_in_manual": ("The manual does not answer this question. "
                      "Please contact the brand's service centre."),
}


@dataclass
class Decision:
    """What to do with one question.

    action:    "answer" | "ask" | "refuse"
    reason:    "ok" | "ask_model" | "clarify" | "low_score" | "not_in_manual"
    message:   text for the user (empty when action is "answer")
    top_score: best rerank score (None if not known yet)
    options:   models to choose from (only for "ask_model")
    """
    action: str
    reason: str
    message: str = ""
    top_score: float | None = None
    options: list = field(default_factory=list)

    @property
    def answers(self) -> bool:
        return self.action == "answer"


def gate_before(route, chunks: list[dict],
                refuse_below: float = REFUSE_BELOW,
                clarify_below: float = CLARIFY_BELOW) -> Decision:
    """Gates 0 and 1: run them after rerank, before Gemini.

    route:  the Route from retrieve/router.py (uses .ask and .options),
            or None if the manual is already known.
    chunks: the reranked chunks, best first (each has "score").
    """
    if route is not None and getattr(route, "ask", False):
        return Decision("ask", "ask_model", MESSAGES["ask_model"],
                        options=list(getattr(route, "options", []) or []))

    top = float(chunks[0]["score"]) if chunks else 0.0
    if top < refuse_below:
        return Decision("refuse", "low_score", MESSAGES["low_score"], top)
    if top < clarify_below:
        return Decision("ask", "clarify", MESSAGES["clarify"], top)
    return Decision("answer", "ok", "", top)


def gate_after(answer: str, top_score: float | None = None) -> Decision:
    """Gate 2: run it on Gemini's answer."""
    if is_refusal(answer):
        return Decision("refuse", "not_in_manual", MESSAGES["not_in_manual"], top_score)
    return Decision("answer", "ok", "", top_score)


if __name__ == "__main__":
    @dataclass
    class FakeRoute:                     # same fields as router.Route
        ask: bool
        options: list = field(default_factory=list)

    one = FakeRoute(ask=False)
    many = FakeRoute(ask=True, options=["rf_001", "rf_002"])
    examples = [
        ("2 fridges match", many, [{"score": 0.90}], None),
        ("score 0.03", one, [{"score": 0.03}], None),
        ("score 0.20", one, [{"score": 0.20}], None),
        ("score 0.85, good answer", one, [{"score": 0.85}], "Press Start [mw_003 p24]."),
        ("score 0.85, NOT_IN_MANUAL", one, [{"score": 0.85}], "NOT_IN_MANUAL"),
        ("no chunks", one, [], None),
    ]
    for name, route, chunks, answer in examples:
        d = gate_before(route, chunks)
        if d.answers and answer is not None:
            d = gate_after(answer, d.top_score)
        print(f"{name:28s} -> {d.action:6s} ({d.reason}) {d.options or ''}")
