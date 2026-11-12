"""Prepare a label photo for reading: upright, not too big, JPEG.

Phone photos are often 4000x3000 and stored sideways (the rotation is only an
EXIF tag). Gemini reads a 1600 px image just as well, and a smaller image is
faster and cheaper to send.

Later (eval baseline, idea from Furqan's number-plate project): an OpenCV
step that crops the largest 4-corner shape (the sticker) for Tesseract. Not
used for Gemini: a wrong crop could cut off the model number.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

MAX_SIDE = 1600          # px, the longest side after resizing
JPEG_QUALITY = 90


def prepare(image) -> bytes:
    """Image (file path, bytes or PIL image) -> upright RGB JPEG bytes, <= MAX_SIDE."""
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    else:
        img = image
    img = ImageOps.exif_transpose(img)            # phone photos: apply the EXIF rotation
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))           # keeps the aspect ratio, only shrinks
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()
