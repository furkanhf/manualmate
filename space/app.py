"""ManualMate on Hugging Face Spaces (Gradio SDK, ZeroGPU).

The same app as on the laptop (src/manualmate: router, refusal gates,
Gemini answers, the Gradio chat), with one change: the search (bge-m3 dense
search + bge-reranker-v2-m3) runs on a ZeroGPU GPU, in gpu_search().

ZeroGPU rules (huggingface/skills, huggingface-spaces/references/zerogpu.md):
  1. `import spaces` before torch / sentence-transformers.
  2. Load the models at module scope with .to("cuda") (embedder/reranker
     get_model() do that: torch.cuda.is_available() is True under spaces).
  3. GPU work runs in a forked worker: only picklable data in and out. So
     gpu_search() opens its OWN Neon connection in the worker and returns
     plain dicts. Routing, Gemini and the chat state stay in the main process,
     so a visitor's GPU quota is used only for the ~1-2 s of search, not for
     the Gemini wait.

Secrets (Space settings): DATABASE_URL, GEMINI_API_KEY.
Deploy from the laptop: python scripts/deploy_space.py
"""

import spaces  # noqa: I001  -- MUST be first (patches torch.cuda)

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MANUALMATE_SEARCH", "full")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from manualmate.api.service import Service           # noqa: E402
from manualmate.api.ui import build_ui                # noqa: E402
from manualmate.generate.answer import search         # noqa: E402
from manualmate.index.db import get_connection        # noqa: E402
from manualmate.index.embedder import get_model as load_embedder   # noqa: E402
from manualmate.retrieve.reranker import get_model as load_reranker  # noqa: E402

# Rule 2: load both models now (module scope), on "cuda".
load_embedder()
load_reranker()

_worker_conn: dict = {}


def _conn():
    """One Neon connection per GPU worker process (never the main process's one:
    a forked socket must not be shared)."""
    pid = os.getpid()
    conn = _worker_conn.get(pid)
    if conn is None or conn.closed or conn.broken:
        conn = get_connection()
        conn.commit()
        conn.autocommit = True
        _worker_conn[pid] = conn
    return conn


@spaces.GPU(duration=20)
def gpu_search(issue: str, manual_ids: list[str]) -> list[dict]:
    """Dense search (bge-m3, top 20) + rerank (bge-reranker-v2-m3) -> top 5 chunks."""
    return search(issue, _conn(), manual_ids)


service = Service("full")
service.searcher = gpu_search
service.start()

demo = build_ui(lambda: service)
demo.queue(default_concurrency_limit=2)
demo.launch()
