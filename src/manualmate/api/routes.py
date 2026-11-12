"""The API endpoints.

GET  /health  is the app up? which search mode? how many chunks?
POST /ask     one question -> one reply (see schemas.AskResponse)

The endpoints are plain `def` (not async): the search and Gemini calls
block, so FastAPI runs them in its thread pool.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from manualmate.api.schemas import AskRequest, AskResponse, Health

router = APIRouter()


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    service = request.app.state.service
    return Health(status="ok", search=service.search, chunks=service.n_chunks)


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    return request.app.state.service.ask(body)
