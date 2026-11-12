"""What goes into the API and what comes out (checked by FastAPI/pydantic).

AskRequest  -> POST /ask body
AskResponse -> its reply. "action" tells the UI what to show:
  answer   the answer text (with [ac_001 p9] tags) + sources
  ask      a question back: "Which model?" with buttons (options), or
           in words if more than 6 manuals match (options empty)
  refuse   "not in the manual" message
  error    Gemini failed (try again)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    manual_id: str | None = Field(
        None, description="The manual the user picked after 'Which model?'.")
    language: Literal["auto", "English", "Hindi", "Telugu"] = Field(
        "auto", description="auto = Hindi/Telugu from the script, else English.")


class Option(BaseModel):
    manual_id: str
    name: str


class Source(BaseModel):
    tag: str                 # "[ac_001 p9]", as in the answer text
    manual_id: str
    name: str                # "Samsung AR... air conditioner"
    page: int
    score: float | None


class Timings(BaseModel):
    translate_ms: float | None
    retrieve_ms: float | None
    gemini_ms: float | None
    gemini_tries: int = 1    # > 1: Gemini failed and was retried (shows why an answer was slow)
    total_ms: float | None


class AskResponse(BaseModel):
    action: Literal["answer", "ask", "refuse", "error"]
    reason: str
    text: str
    language: str
    search_text: str
    route_reason: str
    manual: Option | None    # the one manual searched (the UI keeps it for the chat)
    options: list[Option]
    sources: list[Source]
    citations_ok: bool | None
    top_score: float | None
    timings: Timings


class Health(BaseModel):
    status: str
    search: str              # "full" (dense + rerank) or "bm25" (laptop)
    chunks: int
