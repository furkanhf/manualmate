"""Label photo -> which manual (week 5, light: label reading, not page search).

  photo -> preprocess.prepare() -> ocr.read_label() (Gemini vision)
        -> "Daikin FTKS35KVMA air conditioner" -> query_parser + router
        -> one manual (the chat uses it), several (buttons), or none.

Only an exact MODEL match counts as "found". A label from a model we do not
have (e.g. Voltas 183V) must not silently open another model's manual: then
the result says "no manual for this model", and only offers the same brand's
manuals as buttons if there are a few (the user decides).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from manualmate.vision.ocr import read_label
from manualmate.vision.preprocess import prepare

PRODUCT_WORDS = {"air conditioner": "ac", "wi-fi router": "router",
                 "microwave oven": "microwave"}


@dataclass
class LabelResult:
    """status: "found" (one manual) | "choose" (several fit the model)
               | "no_manual" (read, but no manual for it) | "unreadable" | "error"
    """
    status: str
    brand: str | None = None
    model_number: str | None = None
    product_type: str | None = None
    manual_id: str | None = None
    options: list = field(default_factory=list)   # [(manual_id, name)]
    read_ms: float | None = None
    tries: int = 1

    @property
    def label_text(self) -> str:
        return " ".join(x for x in (self.brand, self.model_number) if x) or "?"


def to_query(reading: dict) -> str:
    """The reading as text the parser knows: "Daikin FTKS35KVMA ac"."""
    product = reading.get("product_type") or ""
    product = PRODUCT_WORDS.get(product, product)
    if product in ("other", "unknown"):
        product = ""
    return " ".join(x for x in (reading.get("brand"), reading.get("model_number"), product) if x)


def match(reading: dict) -> LabelResult:
    """Gemini's reading -> manual(s), with the same router as typed questions."""
    from manualmate.retrieve.query_parser import parse
    from manualmate.retrieve.router import route

    base = dict(brand=reading.get("brand"), model_number=reading.get("model_number"),
                product_type=reading.get("product_type"))
    if not reading.get("model_number") and not reading.get("brand"):
        return LabelResult("unreadable", **base)
    r = route(parse(to_query(reading)))
    if r.reason == "model":                       # the model number is one of ours
        if r.is_single:
            return LabelResult("found", manual_id=r.manual_ids[0], **base)
        return LabelResult("choose", options=list(r.options), **base)
    # Read, but not one of our models: offer the brand's manuals only if few.
    options = list(r.options) if "brand" in r.reason and "no manual" not in r.reason else []
    return LabelResult("no_manual", options=options, **base)


def label_to_manual(image) -> LabelResult:
    """The whole path for one photo (file path, bytes or PIL image)."""
    start, stats = time.perf_counter(), {}
    reading = read_label(prepare(image), stats=stats)
    ms = round((time.perf_counter() - start) * 1000)
    if reading is None:
        return LabelResult("error", read_ms=ms, tries=stats.get("attempts", 1))
    result = match(reading)
    result.read_ms, result.tries = ms, stats.get("attempts", 1)
    return result
