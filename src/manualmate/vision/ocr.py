"""Read a model-number label with Gemini vision (decided chat 5.4, not PaddleOCR).

What it is like: a service-centre person who looks at the sticker and reads
out "Daikin, FTKS35KVMA, air conditioner".

Why Gemini and not classic OCR: phone photos of stickers are blurry, angled
and full of other text (voltage, serial number, barcodes). Classic OCR returns
ALL the text; Gemini can say which line is the model number. No extra install
in the Space (same API as the answers).

Output: a dict {brand, model_number, product_type, label_found}, or None if
Gemini fails. Values are what is printed on the label; nothing is guessed.
"""

from __future__ import annotations

import json
import time

from manualmate.generate.answer import get_client
from manualmate.retrieve.query_rewriter import MODEL_NAME

TIMEOUT_MS = 30_000
PRODUCT_TYPES = ("air conditioner", "washing machine", "refrigerator", "microwave oven",
                 "wi-fi router", "other", "unknown")

PROMPT = f"""This photo should show a home appliance or its model / rating label (sticker or plate).
Read it and return JSON only, with these keys:
  "brand":        the brand name as printed (e.g. "Daikin", "LG", "TP-Link"), or null
  "model_number": the MODEL number exactly as printed, letters, digits and dashes
                  (e.g. "FTKS35KVMA", "MC28H5025VK", "Archer C6"), or null.
                  Not the serial number, not the voltage, not a part number.
                  If there are several (indoor and outdoor unit), give the main / indoor unit.
  "product_type": one of {list(PRODUCT_TYPES)}
  "label_found":  true if a model label or model number is visible, else false
Do not guess: if a value is not readable in the photo, use null."""


def read_label(jpeg: bytes, retries: int = 2, wait: float = 1.0,
               stats: dict | None = None) -> dict | None:
    """Gemini vision on one prepared JPEG (vision/preprocess.py). None if it fails."""
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=256, response_mime_type="application/json",
        http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT]
    for attempt in range(1, retries + 1):
        if stats is not None:
            stats["attempts"] = attempt
        t0 = time.perf_counter()
        try:
            response = get_client().models.generate_content(
                model=MODEL_NAME, contents=contents, config=config)
            data = json.loads(response.text or "{}")
            if isinstance(data, list):                # sometimes a one-item list
                data = data[0] if data else {}
            return {k: (data.get(k) or None) if k != "label_found" else bool(data.get(k))
                    for k in ("brand", "model_number", "product_type", "label_found")}
        except Exception as error:                    # network, quota, bad JSON, ...
            print(f"  Label reading failed (try {attempt}/{retries}, "
                  f"{time.perf_counter() - t0:.1f} s): {str(error)[:300]}")
            if attempt < retries:
                time.sleep(wait * attempt)
    return None
