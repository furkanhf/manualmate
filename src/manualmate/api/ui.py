"""The Gradio chat UI, mounted inside the FastAPI app (one server, port 7860).

What the user sees:
  - a chat box: ask in English, Hindi or Telugu
  - answers with page references "(p. 9)" and a Sources list under them
  - "Which model?" -> up to 6 buttons; the picked manual is kept for the
    rest of the chat (shown above the chat, with a "Change manual" button)
  - a small line with the time each part took (search, Gemini, total)

The UI calls the same Service as POST /ask (in the same process, no HTTP).
"""

from __future__ import annotations

import re
from typing import Callable

import gradio as gr

from manualmate.api.schemas import AskRequest, AskResponse
from manualmate.generate.answer import follow_up
from manualmate.generate.citations import BRACKET_RE, TAG_RE
from manualmate.generate.refusal import message

MAX_BUTTONS = 6          # = router.MAX_OPTIONS
LANGUAGE_CHOICES = ["auto", "English", "Hindi", "Telugu"]

EXAMPLES = [
    "How often should I wash the filter of my Samsung split AC, and how do I clean it?",
    "My Daikin AC outdoor unit is very noisy at night. How can I make it quieter?",
    "My IFB Senator washing machine shows 'tAP' on the display. What should I do?",
    "मेरे Samsung split AC का फ़िल्टर कितनी बार धोना चाहिए और इसे कैसे साफ़ करें?",
]

INTRO = """# ManualMate
Ask a question about your **air conditioner, washing machine, refrigerator, microwave or Wi-Fi router**
in English, Hindi or Telugu. Answers come only from the product manuals, with page numbers.
Add the brand or model to your question (for example *Samsung split AC*, *LG MC2846BG*)."""


# --- formatting ----------------------------------------------------------
def _seconds(ms: float | None) -> str:
    return f"{ms / 1000:.1f} s" if ms is not None else "-"


def format_answer(r: AskResponse) -> str:
    """Answer text with readable page references, a Sources list and timings."""
    names = {s.manual_id: s.name for s in r.sources}
    cited: dict[str, set[int]] = {}

    def replace(match: re.Match) -> str:
        tags = [(m.lower(), int(p)) for m, p in TAG_RE.findall(match.group(1))]
        if not tags:
            return match.group(0)            # a normal [bracket], not a tag
        for m, p in tags:
            cited.setdefault(m, set()).add(p)
        pages = ", ".join(str(p) for p in sorted({p for _, p in tags}))
        return f"*(p. {pages})*"

    text = BRACKET_RE.sub(replace, r.text)
    if r.language == "Hindi":
        text = text.replace("。", "।")      # Gemini sometimes uses the CJK full stop
    else:
        text = text.replace("。", ".")

    parts = [text]
    if cited:
        lines = [f"- {names.get(m, m)}: page {', '.join(str(p) for p in sorted(ps))}"
                 for m, ps in cited.items()]
        parts.append("**Sources**\n" + "\n".join(lines))
    t = r.timings
    timing = f"search {_seconds(t.retrieve_ms)} · answer {_seconds(t.gemini_ms)}"
    if t.gemini_tries > 1:
        timing += f" ({t.gemini_tries} tries)"
    if t.translate_ms is not None:
        timing = f"translate {_seconds(t.translate_ms)} · " + timing
    parts.append(f"<sub>{r.language} · {timing} · total {_seconds(t.total_ms)}</sub>")
    return "\n\n".join(parts)


def format_reply(r: AskResponse) -> str:
    if r.action == "answer":
        return format_answer(r)
    if r.action == "ask" and r.options:
        return r.text + "\n\n" + message("pick", r.language)
    return r.text


def manual_line(state: dict) -> str:
    if state.get("manual_id"):
        return f"Manual for this chat: **{state['manual_name']}**"
    return "Manual: chosen from your question"




# --- the chat logic (no Gradio: tests/test_app.py drives it directly) -------
ERROR_TEXT = "Sorry, something went wrong. Please try again."
MAX_QUESTION = 1000      # = AskRequest.question max_length


class Chat:
    """One chat's rules. State is a plain dict, kept by Gradio per browser tab:

      manual_id, manual_name  the manual kept for follow-ups (picked or routed)
      pending                 the question waiting for a model button
      options                 the buttons shown now [{manual_id, name}]
      pending_product         the question waiting for a product-only reply
                              ("Which washing machine?" -> "LG")

    Every method returns (history, state); state["options"] = the buttons to show.
    """

    def __init__(self, get_service: Callable):
        self.get_service = get_service

    def _run(self, question: str, history: list, state: dict, language: str):
        try:
            req = AskRequest(question=question[:MAX_QUESTION],
                             manual_id=state.get("manual_id"), language=language)
            r = self.get_service().ask(req)
        except Exception as error:                     # keep the UI alive
            print(f"ask failed: {error!r}")
            state = {**state, "options": [], "pending": None, "pending_product": None}
            return history + [{"role": "assistant", "content": ERROR_TEXT}], state
        options = [o.model_dump() for o in r.options] if r.action == "ask" else []
        state = {**state, "pending": question if options else None, "options": options,
                 # "Which product?" without buttons: the next product-only reply is
                 # joined to this question (answer.follow_up)
                 "pending_product": question if (r.action == "ask" and not options
                                                 and r.reason == "ask_model") else None}
        # Keep the one manual this question used for the follow-ups ("how do I clean
        # it?"). A question about another product routes again (answer.keeps_manual)
        # and replaces it; several manuals -> no manual kept.
        if r.manual:
            state = {**state, "manual_id": r.manual.manual_id, "manual_name": r.manual.name}
        elif r.route_reason not in ("user choice", "small talk"):
            state = {**state, "manual_id": None, "manual_name": None}
        return history + [{"role": "assistant", "content": format_reply(r)}], state

    def submit(self, question: str, history: list, state: dict, language: str = "auto"):
        question = (question or "").strip()
        if len(question) < 2:
            return history, state
        history = history + [{"role": "user", "content": question}]
        if state.get("pending_product"):
            question = follow_up(state["pending_product"], question) or question
        return self._run(question, history, state, language)

    def pick(self, index: int, history: list, state: dict, language: str = "auto"):
        options = state.get("options") or []
        if index >= len(options) or not state.get("pending"):
            return history, {**state, "options": []}
        choice = options[index]
        state = {**state, "manual_id": choice["manual_id"], "manual_name": choice["name"]}
        history = history + [{"role": "user", "content": choice["name"]}]
        return self._run(state["pending"], history, state, language)

    @staticmethod
    def change_manual(state: dict) -> dict:
        return {**state, "manual_id": None, "manual_name": None,
                "pending": None, "options": [], "pending_product": None}


# --- the Gradio wiring ----------------------------------------------------------
def build_ui(get_service: Callable) -> gr.Blocks:
    """get_service() returns the running Service (created at app start)."""
    logic = Chat(get_service)

    def buttons_for(state: dict) -> list:
        options = state.get("options") or []
        return [gr.update(value=options[i]["name"], visible=True) if i < len(options)
                else gr.update(visible=False) for i in range(MAX_BUTTONS)]

    def submit(question, history, state, language):
        history, state = logic.submit(question, history, state, language)
        return "", history, state, manual_line(state), *buttons_for(state)

    def pick(index, history, state, language):
        history, state = logic.pick(index, history, state, language)
        return history, state, manual_line(state), *buttons_for(state)

    def change_manual(state):
        state = logic.change_manual(state)
        return state, manual_line(state), *buttons_for(state)

    def new_chat():
        return [], {}, manual_line({}), *buttons_for({})

    with gr.Blocks(title="ManualMate") as ui:
        state = gr.State({})
        gr.Markdown(INTRO)
        with gr.Row():
            manual = gr.Markdown(manual_line({}))
            change = gr.Button("Change manual", size="sm", scale=0)
            clear = gr.Button("New chat", size="sm", scale=0)
        chat = gr.Chatbot(height=480, label="ManualMate", buttons=["copy"],
                          feedback_options=None)
        with gr.Row():
            buttons = [gr.Button(visible=False, size="sm", variant="secondary")
                       for _ in range(MAX_BUTTONS)]
        with gr.Row():
            box = gr.Textbox(placeholder="Ask about your appliance...", show_label=False,
                             scale=5, autofocus=True, max_length=MAX_QUESTION)
            language = gr.Dropdown(LANGUAGE_CHOICES, value="auto", label="Language",
                                   info="auto = from the script; pick Hindi/Telugu for "
                                        "questions typed in English letters", scale=2)
            send = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=box, label="Try")

        outputs = [box, chat, state, manual, *buttons]
        box.submit(submit, [box, chat, state, language], outputs)
        send.click(submit, [box, chat, state, language], outputs)
        for i, button in enumerate(buttons):
            button.click(lambda h, s, l, i=i: pick(i, h, s, l), [chat, state, language],
                         [chat, state, manual, *buttons])
        change.click(change_manual, [state], [state, manual, *buttons])
        clear.click(new_chat, None, [chat, state, manual, *buttons])
    return ui
