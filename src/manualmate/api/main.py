"""The ManualMate app: FastAPI (and, from step 3, the Gradio UI on top).

Run on the laptop (light search, no torch):
    $env:MANUALMATE_SEARCH = "bm25"
    & "D:\\Projects\\manualmate\\.conda\\python.exe" -m uvicorn manualmate.api.main:app --port 7860
Then open http://127.0.0.1:7860 (chat UI) or http://127.0.0.1:7860/docs (API).

On Colab / HF Spaces: MANUALMATE_SEARCH=full (the default).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from manualmate.api.routes import router
from manualmate.api.service import Service
from manualmate.api.ui import build_ui


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = Service()
    service.start()              # connect to Neon, load BM25 or the models
    app.state.service = service
    yield


app = FastAPI(title="ManualMate", version="0.1.0", lifespan=lifespan)
app.include_router(router)            # /health, /ask, /docs come before the UI at "/"
app = gr.mount_gradio_app(app, build_ui(lambda: app.state.service), path="/",
                          footer_links=["api"])
