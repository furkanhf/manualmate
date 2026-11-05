"""Check that the Gemini key and the Neon database work."""
import os

import psycopg
from dotenv import load_dotenv
from google import genai

load_dotenv()

# 1. Gemini
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
models = [m.name for m in client.models.list()]
print(f"Gemini OK: {len(models)} models available")

# 2. Neon + pgvector
with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    version = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
    ).fetchone()[0]
    print(f"Database OK: pgvector version {version}")