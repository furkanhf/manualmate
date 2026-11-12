"""The app's long-lived parts: the Neon connection and the search mode.

What it is like: the shop's back office. It opens once when the app starts
and stays open; every request uses it.

Search mode (env MANUALMATE_SEARCH):
  full  dense (bge-m3) + reranker: the real system (Colab, HF Spaces). Default.
  bm25  router + BM25 keyword search: no torch, for laptop tests of the
        API/UI. Weaker, and gate 1 works differently (see answer.retrieve_light).

Neon sleeps after ~5 minutes and drops idle connections, so every request
checks the connection and opens a new one if it is closed or broken.
"""

from __future__ import annotations

import os
import threading

import psycopg

from manualmate.generate.answer import Answer, ask
from manualmate.index.db import count_chunks, get_connection
from manualmate.retrieve.query_parser import get_aliases

from manualmate.api.schemas import AskRequest, AskResponse, Option, Source, Timings

SEARCH_MODES = ("full", "bm25")


class Service:
    def __init__(self, search: str | None = None):
        search = (search or os.environ.get("MANUALMATE_SEARCH", "full")).lower()
        if search not in SEARCH_MODES:
            raise ValueError(f"MANUALMATE_SEARCH must be one of {SEARCH_MODES}, not {search!r}")
        self.search = search
        self.bm25 = None
        self.searcher = None              # set by space/app.py: dense + rerank on ZeroGPU
        self.n_chunks = 0
        self._conn: psycopg.Connection | None = None
        self._lock = threading.Lock()     # one request at a time uses the connection

    # --- start ---------------------------------------------------------
    def start(self) -> None:
        """Connect, and load what the search needs, so request 1 is not slow."""
        conn = self.conn()
        self.n_chunks = count_chunks(conn)
        if self.search == "bm25":
            from manualmate.index.bm25 import BM25Index
            self.bm25 = BM25Index.from_db(conn)
        else:
            from manualmate.index.embedder import get_model as embedder
            from manualmate.retrieve.reranker import get_model as reranker
            embedder()
            reranker()
        print(f"ManualMate ready: search={self.search}, {self.n_chunks} chunks")

    # --- connection ----------------------------------------------------
    def _open(self) -> psycopg.Connection:
        conn = get_connection()
        conn.commit()                     # end the CREATE EXTENSION transaction
        conn.autocommit = True            # read-only app: no open transactions
        return conn

    def conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed or self._conn.broken:
            self._conn = self._open()
        return self._conn

    # --- ask -----------------------------------------------------------
    def ask(self, req: AskRequest) -> AskResponse:
        language = None if req.language == "auto" else req.language
        with self._lock:
            def run():
                return ask(req.question, self.conn(), req.manual_id, language,
                           bm25=self.bm25, searcher=self.searcher)
            try:
                a = run()
            except psycopg.OperationalError:
                # Neon dropped the connection (sleep): open a new one, try once more.
                self._conn = None
                a = run()
        return to_response(a)


def manual_name(manual_id: str) -> str:
    info = get_aliases().manuals.get(manual_id)
    return info["name"] if info else manual_id


def to_response(a: Answer) -> AskResponse:
    return AskResponse(
        action=a.action, reason=a.reason, text=a.text, language=a.language,
        search_text=a.search_text, route_reason=a.route_reason,
        manual=Option(manual_id=a.manual_id, name=manual_name(a.manual_id)) if a.manual_id else None,
        options=[Option(manual_id=m, name=n) for m, n in a.options],
        sources=[Source(tag=s["tag"], manual_id=s["manual_id"],
                        name=manual_name(s["manual_id"]), page=s["page"],
                        score=s["score"]) for s in a.sources] if a.action == "answer" else [],
        citations_ok=a.citations.get("ok") if a.citations else None,
        top_score=a.top_score,
        timings=Timings(translate_ms=a.translate_ms, retrieve_ms=a.retrieve_ms,
                        gemini_ms=a.gemini_ms, gemini_tries=a.gemini_tries,
                        total_ms=a.total_ms),
    )
