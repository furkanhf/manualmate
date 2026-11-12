"""The API endpoints.

GET  /health  is the app up? which search mode? how many chunks?
POST /ask     one question -> one reply (see schemas.AskResponse)
POST /read-label  photo of the model label -> brand, model number, manual

The endpoints are plain `def` (not async): the search and Gemini calls
block, so FastAPI runs them in its thread pool.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from manualmate.api.schemas import AskRequest, AskResponse, Health, LabelResponse

router = APIRouter()


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    service = request.app.state.service
    return Health(status="ok", search=service.search, chunks=service.n_chunks)


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    return request.app.state.service.ask(body)


MAX_PHOTO_BYTES = 15 * 1024 * 1024


@router.post("/read-label", response_model=LabelResponse)
def read_label(request: Request, photo: UploadFile = File(...)) -> LabelResponse:
    """Photo of the model label (JPEG/PNG/WebP) -> brand, model number, manual."""
    data = photo.file.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "Photo too large (max 15 MB).")
    try:
        from PIL import Image
        import io
        Image.open(io.BytesIO(data)).verify()        # a real image, not any file
    except Exception:
        raise HTTPException(415, "Not an image file.")
    return request.app.state.service.read_label(data)
