"""App tests (week 6): API, ask(), the chat rules, formatting, language detection.

Real: Neon, the parser, the router, BM25 search (light mode), the refusal gates,
the citation check, the chat state. Fake: Gemini and the translator, so the
tests are fast, free (no quota) and give the same result every time.
Not tested here: answer QUALITY of the real search (dense + rerank) -> Colab.

Run on the laptop (needs DATABASE_URL in .env; pytest is a dev tool, not in
requirements.txt: pip install pytest):
    & "D:\\Projects\\manualmate\\.conda\\python.exe" -m pytest tests/test_app.py -q
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Translations the fake translator knows (anything else: returned unchanged).
TRANSLATIONS = {
    "मेरे Samsung split AC का फ़िल्टर कितनी बार धोना चाहिए?":
        "How often should I wash the filter of my Samsung split AC?",
    "mera Daikin FTKS ac bhot awaaz karta hai kya karu?":
        "My Daikin FTKS AC makes a lot of noise, what should I do?",
    "washing machine reset kaise karein?": "How do I reset the washing machine?",
    "నా Samsung AC ధర ఎంత?": "What is the price of my Samsung AC?",
    "मेरा Daikin fridge ठंडा नहीं करता": "My Daikin fridge does not cool",
}
REFUSE_WORDS = re.compile(r"price|cricket|cost|spotify|cancer|ధర", re.I)
SOURCE_TAG = re.compile(r"Source 1 (\[[a-z]{2}_\d{3} p\d+\])")


class FakeGemini:
    """Answers with the tag of source 1, or NOT_IN_MANUAL for off-topic words."""

    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, system, user, model=None, retries=3, wait=5.0, stats=None):
        self.calls.append(user)
        self.retries = retries
        if stats is not None:
            stats["attempts"] = retries if self.fail else 1
        if self.fail:
            return None
        question = user.split("QUESTION about", 1)[-1]
        if REFUSE_WORDS.search(question):
            return "NOT_IN_MANUAL"
        tag = SOURCE_TAG.search(user)
        return f"Follow the steps on this page of the manual {tag.group(1)}."


class FakeTranslator:
    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, question, retries=3, wait=5.0):
        self.calls.append(question)
        self.wait = wait
        return "" if self.fail else TRANSLATIONS.get(question, question)


@pytest.fixture(scope="module")
def fakes():
    mp = pytest.MonkeyPatch()
    gemini, translator = FakeGemini(), FakeTranslator()
    mp.setattr("manualmate.generate.answer.call_gemini", gemini)
    mp.setattr("manualmate.retrieve.translator.translate", translator)
    yield gemini, translator
    mp.undo()


@pytest.fixture(scope="module")
def service(fakes):
    from manualmate.api.service import Service

    s = Service("bm25")
    s.start()
    return s


class Recorder:
    """Wraps the service and keeps the last response (to check search_text etc.)."""

    def __init__(self, service):
        self.service, self.last, self.requests = service, None, []

    def ask(self, req):
        self.requests.append(req)
        self.last = self.service.ask(req)
        return self.last

    def read_label(self, image):
        return self.service.read_label(image)


@pytest.fixture
def chat(service):
    from manualmate.api.ui import Chat

    rec = Recorder(service)
    return Chat(lambda: rec), rec


def ask(service, question, manual_id=None, language="auto"):
    from manualmate.api.schemas import AskRequest

    return service.ask(AskRequest(question=question, manual_id=manual_id, language=language))


def last_text(history):
    return history[-1]["content"]


# --- ask(): one question ------------------------------------------------------
def test_answer_english_branded(service):
    r = ask(service, "How often should I wash the filter of my Samsung split AC?")
    assert r.action == "answer" and r.citations_ok is True
    assert r.manual and r.manual.manual_id == "ac_001"
    assert r.sources and all(s.manual_id == "ac_001" for s in r.sources)
    assert r.timings.total_ms is not None and r.timings.translate_ms is None
    assert r.language == "English"


def test_refuse_off_topic(service):
    r = ask(service, "What is the price of the Samsung AR9500T AC in India?")
    assert r.action == "refuse" and r.reason == "not_in_manual"
    assert r.sources == []


def test_no_product_asks_product_with_types(service):
    r = ask(service, "how many modes are there")
    assert r.action == "ask" and r.options == []
    assert "air conditioners" in r.text and "Wi-Fi routers" in r.text


def test_category_only_lists_brands(service):
    r = ask(service, "how can I reset my washing machine")
    assert r.action == "ask" and r.options == []
    assert "Which washing machine" in r.text and "IFB" in r.text and "Whirlpool" in r.text
    assert "{" not in r.text


def test_model_in_two_manuals_says_series(service):
    r = ask(service, "How many modes does my Daikin FTKS AC have?")
    assert r.action == "ask" and len(r.options) == 2
    assert "more than one manual" in r.text and "label" in r.text
    assert {o.manual_id for o in r.options} == {"ac_002", "ac_004"}


def test_brand_category_buttons(service):
    r = ask(service, "My Daikin AC is noisy at night")
    assert r.action == "ask" and len(r.options) == 3
    assert r.text.startswith("Which model do you have?")


def test_brand_without_that_product(service):
    r = ask(service, "My Daikin fridge does not cool")
    assert r.action == "ask"
    assert "do not have a manual for a Daikin refrigerator" in r.text
    assert "Samsung" in r.text and "{" not in r.text


def test_hindi_script_translates_and_answers_in_hindi(service, fakes):
    _, translator = fakes
    q = "मेरे Samsung split AC का फ़िल्टर कितनी बार धोना चाहिए?"
    r = ask(service, q)
    assert r.language == "Hindi" and q in translator.calls
    assert r.search_text == TRANSLATIONS[q]
    assert r.action == "answer" and r.timings.translate_ms is not None


def test_romanized_hindi_ask_is_in_hindi(service):
    r = ask(service, "mera Daikin FTKS ac bhot awaaz karta hai kya karu?")
    assert r.language == "Hindi" and r.action == "ask" and len(r.options) == 2
    assert "manual" in r.text and re.search(r"[\u0900-\u097f]", r.text)


def test_telugu_refusal_is_in_telugu(service):
    r = ask(service, "నా Samsung AC ధర ఎంత?")
    assert r.language == "Telugu" and r.action == "refuse"
    assert re.search(r"[\u0c00-\u0c7f]", r.text)


def test_hindi_no_manual_message(service):
    r = ask(service, "मेरा Daikin fridge ठंडा नहीं करता")
    assert r.language == "Hindi" and "Daikin" in r.text and "{" not in r.text


def test_language_dropdown_overrides(service):
    r = ask(service, "How often should I wash the filter of my Samsung split AC?",
            language="Telugu")
    assert r.language == "Telugu"


def test_gemini_failure_is_error_message(service, fakes):
    gemini, _ = fakes
    gemini.fail = True
    try:
        r = ask(service, "How often should I wash the filter of my Samsung split AC?")
        h = ask(service, "मेरे Samsung split AC का फ़िल्टर कितनी बार धोना चाहिए?")
    finally:
        gemini.fail = False
    assert r.action == "error" and "try again" in r.text
    assert h.action == "error" and re.search(r"[\u0900-\u097f]", h.text)
    assert r.timings.gemini_tries == 2 and gemini.retries == 2   # the app: 2 tries, not 3


def test_app_retries_are_short(service, fakes):
    from manualmate.api.ui import format_answer

    gemini, translator = fakes
    ask(service, "\u092e\u0947\u0930\u0947 Samsung split AC \u0915\u093e \u092b\u093c\u093f\u0932\u094d\u091f\u0930 \u0915\u093f\u0924\u0928\u0940 \u092c\u093e\u0930 \u0927\u094b\u0928\u093e \u091a\u093e\u0939\u093f\u090f?")
    assert gemini.retries == 2 and translator.wait <= 1.0
    r = ask(service, "How often should I wash the filter of my Samsung split AC?")
    r.timings.gemini_tries = 2
    assert "(2 tries)" in format_answer(r)


def test_translation_failure_falls_back_to_question(service, fakes):
    _, translator = fakes
    translator.fail = True
    try:
        q = "मेरे Samsung split AC का फ़िल्टर कितनी बार धोना चाहिए?"
        r = ask(service, q)
    finally:
        translator.fail = False
    assert r.search_text == q and r.action in ("answer", "refuse", "ask")


def test_unknown_manual_id_routes_again(service):
    r = ask(service, "How often should I wash the filter of my Samsung split AC?",
            manual_id="xx_999")
    assert r.manual and r.manual.manual_id == "ac_001"


def test_picked_manual_kept_for_vague_question(service):
    r = ask(service, "How do I clean it?", manual_id="ac_002")
    assert r.route_reason == "user choice" and r.manual.manual_id == "ac_002"


def test_picked_manual_dropped_for_other_product(service):
    r = ask(service, "how do i setup my router", manual_id="ac_002")
    assert r.route_reason != "user choice" and r.manual is None
    assert "Wi-Fi router" in r.text


def test_searcher_hook_used_and_skipped_on_ask(service):
    """The ZeroGPU Space plugs its GPU search in as service.searcher (space/app.py)."""
    from manualmate.api.schemas import AskRequest
    from manualmate.api.service import Service

    calls = []

    def searcher(issue, manual_ids):
        calls.append((issue, manual_ids))
        return service.bm25.search(issue, k=5, manual_ids=manual_ids)

    s = Service("full")                  # no start(): no models; the hook does the search
    s.searcher, s._conn = searcher, service.conn()
    r = s.ask(AskRequest(question="How often should I wash the filter of my Samsung split AC?"))
    assert r.action == "answer" and calls == [(calls[0][0], ["ac_001"])]
    assert "Samsung" not in calls[0][0]                  # the issue text, brand removed
    r = s.ask(AskRequest(question="My Daikin AC is noisy at night"))
    assert r.action == "ask" and len(calls) == 1          # "Which model?": no GPU call


# --- the chat rules (what the UI does) ------------------------------------------
def test_chat_pick_button_then_follow_up(chat):
    c, rec = chat
    h, s = c.submit("How many modes does my Daikin FTKS AC have?", [], {})
    assert len(s["options"]) == 2 and s["pending"]
    k = next(i for i, o in enumerate(s["options"]) if o["manual_id"] == "ac_002")
    h, s = c.pick(k, h, s)
    assert s["manual_id"] == "ac_002" and s["options"] == []
    assert rec.last.action == "answer" and "modes" in rec.last.search_text
    h, s = c.submit("How do I clean the filter?", h, s)
    assert rec.requests[-1].manual_id == "ac_002" and s["manual_id"] == "ac_002"
    # another product: the manual changes
    h, s = c.submit("How do I reset my TP-Link Archer C6 router?", h, s)
    assert s.get("manual_id") != "ac_002"


def test_chat_routed_manual_kept_for_follow_up(chat):
    c, rec = chat
    h, s = c.submit("How do I clean the drum of my LG FHM1408 washing machine?", [], {})
    assert s["manual_id"] == "wm_003"
    h, s = c.submit("and how often?", h, s)
    assert rec.requests[-1].manual_id == "wm_003"


def test_chat_product_reply_joins_question(chat):
    c, rec = chat
    h, s = c.submit("how can I reset my washing machine", [], {})
    assert s["pending_product"] and not s["options"]
    h, s = c.submit("LG", h, s)
    assert "reset" in rec.requests[-1].question and "LG" in rec.requests[-1].question
    assert len(s["options"]) == 2                     # 2 LG washing machines
    h, s = c.pick(0, h, s)
    assert "reset" in rec.requests[-1].question
    assert rec.requests[-1].manual_id in {"wm_003", "wm_012"}
    assert h[-2]["role"] == "user" and h[-1]["role"] == "assistant"


def test_chat_new_question_not_joined(chat):
    c, rec = chat
    h, s = c.submit("how can I reset my washing machine", [], {})
    h, s = c.submit("How do I clean the drum of my LG FHM1408 washing machine?", h, s)
    assert "reset" not in rec.requests[-1].question


def test_chat_no_manual_then_brand_reply(chat):
    c, rec = chat
    h, s = c.submit("My Daikin fridge does not cool", [], {})
    assert s["pending_product"]
    h, s = c.submit("Samsung", h, s)
    assert "Samsung" in rec.requests[-1].question


def test_chat_hindi_series_pick_answers_in_hindi(chat):
    c, rec = chat
    h, s = c.submit("mera Daikin FTKS ac bhot awaaz karta hai kya karu?", [], {})
    assert len(s["options"]) == 2
    h, s = c.pick(0, h, s)
    assert rec.last.language == "Hindi" and rec.last.action in ("answer", "refuse")


def test_chat_bad_pick_and_empty_input(chat):
    c, rec = chat
    h, s = c.pick(3, [], {})
    assert h == [] and s["options"] == []
    h, s = c.submit("   ", [], {})
    assert h == [] and rec.requests == []
    h, s = c.submit("How many modes does my Daikin FTKS AC have?", [], {})
    h, s = c.pick(5, h, s)                            # only 2 buttons
    assert s["options"] == [] and len(h) == 2


def test_chat_change_manual_clears(chat):
    c, _ = chat
    h, s = c.submit("How many modes does my Daikin FTKS AC have?", [], {})
    s = c.change_manual({**s, "manual_id": "ac_002", "manual_name": "x"})
    assert not s["manual_id"] and not s["options"] and not s["pending"]


def test_chat_very_long_question(chat):
    c, rec = chat
    h, s = c.submit("How do I clean the filter of my Samsung split AC? " * 200, [], {})
    assert len(rec.requests[-1].question) <= 1000 and h[-1]["role"] == "assistant"
    assert h[-1]["content"] != ""


def test_chat_service_crash_keeps_ui_alive():
    from manualmate.api.ui import ERROR_TEXT, Chat

    class Broken:
        def ask(self, req):
            raise RuntimeError("database down")

    h, s = Chat(lambda: Broken()).submit("How do I clean my Samsung AC?", [],
                                         {"options": [{"manual_id": "a", "name": "b"}]})
    assert last_text(h) == ERROR_TEXT and s["options"] == []


# --- formatting --------------------------------------------------------------
def _response(text, language="English", sources=(("ac_002", 19),)):
    from manualmate.api.schemas import AskResponse, Source, Timings

    return AskResponse(
        action="answer", reason="ok", text=text, language=language, search_text="",
        route_reason="model", manual=None, options=[],
        sources=[Source(tag=f"[{m} p{p}]", manual_id=m, name="Daikin K", page=p, score=1.0)
                 for m, p in sources],
        citations_ok=True, top_score=1.0,
        timings=Timings(translate_ms=None, retrieve_ms=10, gemini_ms=1500, total_ms=1510))


def test_format_answer_tags_and_sources():
    from manualmate.api.ui import format_answer

    out = format_answer(_response("Use QUIET [ac_002 p19]. Also NIGHT [ac_002 p19, ac_002 p29]. "
                                  "Press [ON/OFF] button [AC_002 P.20]."))
    assert "[ac_002" not in out and "*(p. 19)*" in out and "*(p. 19, 29)*" in out
    assert "[ON/OFF]" in out                          # a normal bracket stays
    assert "*(p. 20)*" in out
    assert "Daikin K: page 19, 20, 29" in out
    assert "total 1.5 s" in out


def test_format_answer_hindi_full_stop():
    from manualmate.api.ui import format_answer

    out = format_answer(_response("फ़िल्टर साफ़ करें [ac_002 p19]。", language="Hindi"))
    assert "。" not in out and "।" in out


# --- language detection, follow-up joining, keeping the manual --------------------
def _load(name):
    return [json.loads(l) for l in (ROOT / "eval" / name).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def test_detect_language_on_eval_sets():
    from manualmate.generate.answer import detect_language

    english = [q["question"] for n in ("text_queries.jsonl", "text_queries_new.jsonl",
                                       "out_of_scope.jsonl", "clarify_queries.jsonl")
               for q in _load(n)]
    assert [q for q in english if detect_language(q) != "English"] == []
    want = {"hi": "Hindi", "te": "Telugu"}
    multi = _load("multilingual_queries.jsonl")
    assert [q["id"] for q in multi if detect_language(q["question"]) != want[q["language"]]] == []


@pytest.mark.parametrize("reply,joined", [
    ("LG washing machine", True), ("LG", True), ("its an LG FHM1408", True),
    ("Samsung", True), ("mera LG hai", True),
    ("LG washing machine what are the options", False),
    ("How do I clean the drum of my LG FHM1408?", False),
    ("what is the weather", False), ("no water comes in my LG", False),
])
def test_follow_up(reply, joined):
    from manualmate.generate.answer import follow_up

    out = follow_up("how do I reset my washing machine", reply)
    assert (out is not None) == joined
    if joined:
        assert out.startswith("how do I reset my washing machine")


@pytest.mark.parametrize("question,manual_id,keep", [
    ("How do I clean it?", "ac_002", True),
    ("My Daikin AC is noisy", "ac_002", True),
    ("My Daikin FTKS35 is noisy", "ac_002", True),        # FTKS fits K and L
    ("My Daikin FTXZ is noisy", "ac_002", False),         # another Daikin model
    ("how do i setup my router", "ac_002", False),
    ("My Samsung AC is noisy", "ac_002", False),
    ("anything", "xx_999", False),
])
def test_keeps_manual(question, manual_id, keep):
    from manualmate.generate.answer import keeps_manual

    assert keeps_manual(question, manual_id) is keep


@pytest.mark.parametrize("text,kind", [
    ("hi", "greeting"), ("Hello there!", "greeting"), ("नमस्ते", "greeting"),
    ("thanks!", "thanks"), ("thank you so much", "thanks"), ("धन्यवाद", "thanks"),
    ("ధన్యవాదాలు", "thanks"),
    ("hi my AC is broken", None), ("thanks, and how do I reset it?", None),
])
def test_small_talk(text, kind):
    from manualmate.generate.answer import small_talk

    assert small_talk(text) == kind


def test_chat_thanks_keeps_manual_and_skips_gemini(chat, fakes):
    gemini, _ = fakes
    c, rec = chat
    h, s = c.submit("How do I clean the drum of my LG FHM1408 washing machine?", [], {})
    n = len(gemini.calls)
    h, s = c.submit("thanks!", h, s)
    assert len(gemini.calls) == n and s["manual_id"] == "wm_003"
    assert "welcome" in last_text(h)
    h, s = c.submit("नमस्ते", [], {})
    assert re.search(r"[ऀ-ॿ]", last_text(h)) and not s.get("pending_product")


# --- label photo (week 5 light: Gemini vision reads the model label) --------------
READINGS = {   # fake Gemini vision: image name -> what it "reads"
    "daikin_k": {"brand": "Daikin", "model_number": "FTKS35KVMA",
                 "product_type": "air conditioner", "label_found": True},
    "daikin_ftks35": {"brand": "Daikin", "model_number": "FTKS35",
                      "product_type": "air conditioner", "label_found": True},
    "samsung_mw": {"brand": "Samsung", "model_number": "MC28H5025VK",
                   "product_type": "microwave oven", "label_found": True},
    "voltas": {"brand": "Voltas", "model_number": "183V CZT",
               "product_type": "air conditioner", "label_found": True},
    "lg_unknown": {"brand": "LG", "model_number": "XYZ999", "product_type": "washing machine",
                   "label_found": True},
    "blank": {"brand": None, "model_number": None, "product_type": "unknown",
              "label_found": False},
}


@pytest.fixture
def photos(monkeypatch, tmp_path):
    """Fake label reading: the photo file's name says what Gemini 'reads'."""
    from PIL import Image

    def fake_read(jpeg, **kwargs):
        return dict(READINGS[fake_read.current]) if fake_read.current != "fail" else None

    fake_read.current = None
    monkeypatch.setattr("manualmate.vision.image_to_query.read_label", fake_read)

    def make(name):
        path = tmp_path / f"{name}.jpg"
        Image.new("RGB", (4000, 3000), "white").save(path)
        fake_read.current = name
        return str(path)
    return make


def test_preprocess_upright_and_small(tmp_path):
    import io
    from PIL import Image
    from manualmate.vision.preprocess import MAX_SIDE, prepare

    img = Image.new("RGB", (4000, 3000), "white")
    exif = Image.Exif()
    exif[0x0112] = 6                                  # "rotate 90" phone photo
    path = tmp_path / "phone.jpg"
    img.save(path, exif=exif)
    out = Image.open(io.BytesIO(prepare(str(path))))
    assert out.format == "JPEG" and max(out.size) == MAX_SIDE
    assert out.size[1] > out.size[0]                  # now upright (portrait)


@pytest.mark.parametrize("name,status,manual", [
    ("daikin_k", "found", "ac_002"), ("samsung_mw", "found", "mw_002"),
    ("daikin_ftks35", "choose", None), ("voltas", "no_manual", None),
    ("lg_unknown", "no_manual", None), ("blank", "unreadable", None),
])
def test_label_match(name, status, manual):
    from manualmate.vision.image_to_query import match

    r = match(READINGS[name])
    assert r.status == status and r.manual_id == manual
    if name == "daikin_ftks35":
        assert {m for m, _ in r.options} == {"ac_002", "ac_004"}
    if name == "lg_unknown":                          # 2 LG washing machines: offered
        assert {m for m, _ in r.options} == {"wm_003", "wm_012"}
    if name == "voltas":
        assert r.options == []


def test_chat_photo_found_sets_manual(chat, photos):
    c, rec = chat
    h, s = c.photo(photos("daikin_k"), [], {})
    assert s["manual_id"] == "ac_002" and "FTKS35KVMA" in h[-2]["content"]
    assert "What would you like to know" in last_text(h)
    h, s = c.submit("How do I clean the filter?", h, s)
    assert rec.requests[-1].manual_id == "ac_002"


def test_chat_photo_answers_waiting_question(chat, photos):
    c, rec = chat
    h, s = c.submit("how can I reset my washing machine", [], {})       # "Which washing machine?"
    h, s = c.photo(photos("samsung_mw"), h, s)                          # (a microwave label)
    assert rec.requests[-1].manual_id == "mw_002" and "reset" in rec.requests[-1].question
    assert h[-1]["role"] == "assistant"


def test_chat_photo_choose_then_pick(chat, photos):
    c, rec = chat
    h, s = c.photo(photos("daikin_ftks35"), [], {})
    assert len(s["options"]) == 2 and not s.get("pending")
    n = len(rec.requests)
    h, s = c.pick(0, h, s)
    assert s["manual_id"] in {"ac_002", "ac_004"} and len(rec.requests) == n   # no search yet
    assert "What would you like to know" in last_text(h)


def test_chat_photo_no_manual_unreadable_error(chat, photos):
    c, _ = chat
    h, s = c.photo(photos("voltas"), [], {})
    assert "do not have a manual" in last_text(h) and s["options"] == []
    h, s = c.photo(photos("lg_unknown"), [], {})
    assert "Pick one only if it is your model" in last_text(h) and len(s["options"]) == 2
    h, s = c.photo(photos("blank"), [], {})
    assert "could not read a model number" in last_text(h)
    h, s = c.photo(photos("fail"), [], {})
    assert "could not read the photo" in last_text(h)
    assert c.photo(None, h, s) == (h, s)


def test_chat_photo_reply_in_chat_language(chat, photos):
    c, _ = chat
    h, s = c.submit("मेरे Samsung split AC का फ़िल्टर कितनी बार धोना चाहिए?", [], {})
    h, s = c.photo(photos("daikin_k"), h, s)
    assert re.search(r"[ऀ-ॿ]", last_text(h))


def test_http_read_label(fakes, photos, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MANUALMATE_SEARCH", "bm25")
    from manualmate.api.main import app

    with TestClient(app) as client:
        path = photos("daikin_k")
        with open(path, "rb") as f:
            r = client.post("/read-label", files={"photo": ("label.jpg", f, "image/jpeg")})
        assert r.status_code == 200 and r.json()["manual"]["manual_id"] == "ac_002"
        bad = client.post("/read-label", files={"photo": ("x.jpg", b"not an image", "image/jpeg")})
        assert bad.status_code == 415


# --- the HTTP API ---------------------------------------------------------------
def test_http_api(fakes, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MANUALMATE_SEARCH", "bm25")
    from manualmate.api.main import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["search"] == "bm25" and health["chunks"] > 0
        ok = client.post("/ask", json={"question": "How do I clean the filter of my Samsung split AC?"})
        assert ok.status_code == 200 and ok.json()["action"] == "answer"
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422
        assert client.post("/ask", json={"question": "hello there", "language": "French"}).status_code == 422
        assert client.get("/").status_code == 200        # the Gradio UI
