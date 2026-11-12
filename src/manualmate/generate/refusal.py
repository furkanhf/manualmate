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

Thresholds set from data (eval/results/2026-11-10_1047_answers.jsonl, Colab):
  out-of-scope top scores: max 0.102; 24 of 28 are below 0.05.
  in-scope top scores:     min 0.15 (q009), then 0.17, 0.18 - and those answers are fine.
  REFUSE_BELOW = 0.05 keeps a 3x gap to the lowest in-scope score (not tuned tight
  on the test set) and saves a Gemini call on most out-of-scope questions.
  CLARIFY_BELOW = REFUSE_BELOW -> the score-based clarify band is OFF: low-score
  in-scope questions still got good answers. Vague questions are handled by the
  router ("Which model?", clarify set 10/10).
A high score does not prove the page is right (q035: wrong page, 0.758;
o19 sushi: 0.868, refused by gate 2), so gate 2 is still needed.

Runs on the laptop (no torch). Run this file to see 6 example decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from manualmate.generate.citations import is_refusal

# Set from data on 2026-11-10 (see the docstring).
REFUSE_BELOW = 0.05
CLARIFY_BELOW = 0.05          # = REFUSE_BELOW -> clarify band off

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
    "gemini_error": "Sorry, the answer service is not available now. Please try again.",
    "pick": "Pick your model below.",
    # "ask_model" with too many manuals for buttons (no brand/product in the question):
    "ask_product": ("Which product is this about? Please add the brand and type to your "
                    "question, for example \"Daikin AC\" or \"LG microwave\"."),
}

# The same messages in Hindi and Telugu (chat 5.4): a Hindi question gets
# "Which model?" / the refusal in Hindi too. Brand/manual words stay English,
# as Indian users write them. Drafts by Claude; Furqan checks the wording.
MESSAGES_BY_LANGUAGE = {
    "English": MESSAGES,
    "Hindi": {
        "ask_model": "आपके पास कौन-सा मॉडल है? मुझे एक से ज़्यादा manual मिले जो फिट होते हैं।",
        "clarify": ("मुझे पक्का नहीं पता कि आपको manual का कौन-सा हिस्सा चाहिए। "
                    "क्या आप समस्या के बारे में थोड़ा और बता सकते हैं? "
                    "जैसे: डिस्प्ले पर क्या दिख रहा है, या उपकरण क्या कर रहा है।"),
        "low_score": ("मुझे यह manual में नहीं मिला। कृपया brand के service centre से संपर्क करें, "
                      "या product को इस्तेमाल करने, साफ़ करने, सेट-अप करने या "
                      "समस्या ठीक करने के बारे में पूछें।"),
        "not_in_manual": ("Manual में इस सवाल का जवाब नहीं है। "
                          "कृपया brand के service centre से संपर्क करें।"),
        "gemini_error": "माफ़ कीजिए, अभी जवाब देने वाली सेवा उपलब्ध नहीं है। कृपया फिर से कोशिश करें।",
        "pick": "नीचे से अपना मॉडल चुनें।",
        "ask_product": ("यह किस product के बारे में है? कृपया अपने सवाल में brand और product "
                        "का नाम लिखें, जैसे \"Daikin AC\" या \"LG microwave\"।"),
    },
    "Telugu": {
        "ask_model": "మీ దగ్గర ఏ మోడల్ ఉంది? సరిపోయే manuals ఒకటి కంటే ఎక్కువ దొరికాయి.",
        "clarify": ("మీకు manual లో ఏ భాగం కావాలో నాకు సరిగ్గా అర్థం కాలేదు. "
                    "సమస్య గురించి ఇంకొంచెం వివరంగా చెప్పగలరా? "
                    "ఉదాహరణకు: డిస్‌ప్లేలో ఏం కనిపిస్తోంది, లేదా ఉపకరణం ఏం చేస్తోంది."),
        "low_score": ("ఇది manual లో నాకు దొరకలేదు. దయచేసి brand service centre ను సంప్రదించండి, "
                      "లేదా product ను ఎలా వాడాలి, శుభ్రం చేయాలి, సెటప్ చేయాలి లేదా "
                      "సమస్యలు ఎలా సరిచేయాలి అని అడగండి."),
        "not_in_manual": ("ఈ ప్రశ్నకు manual లో సమాధానం లేదు. "
                          "దయచేసి brand service centre ను సంప్రదించండి."),
        "gemini_error": "క్షమించండి, ప్రస్తుతం సమాధానం ఇచ్చే సేవ అందుబాటులో లేదు. దయచేసి మళ్ళీ ప్రయత్నించండి.",
        "pick": "కింద మీ మోడల్‌ను ఎంచుకోండి.",
        "ask_product": ("ఇది ఏ product గురించి? దయచేసి మీ ప్రశ్నలో brand మరియు product పేరు "
                        "రాయండి, ఉదాహరణకు \"Daikin AC\" లేదా \"LG microwave\"."),
    },
}


def message(reason: str, language: str = "English") -> str:
    """The fixed message for a reason, in the user's language (English if unknown)."""
    table = MESSAGES_BY_LANGUAGE.get(language, MESSAGES)
    return table.get(reason) or MESSAGES.get(reason, "")


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
