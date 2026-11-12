---
title: ManualMate
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
python_version: "3.12"
short_description: Answers from appliance manuals, with page citations
startup_duration_timeout: 30m
pinned: false
---

# ManualMate

Ask a question about your **air conditioner, washing machine, refrigerator,
microwave or Wi-Fi router** in **English, Hindi or Telugu**. ManualMate answers
only from the product's manual (48 manuals, 8,134 sections) and cites the pages.
If the manual does not cover the question, it says so instead of guessing.

How it works:

1. **Router**: brand and model from the question pick the manual. If several
   manuals fit, it asks "Which model?" with buttons.
2. **Search**: bge-m3 dense search in Postgres/pgvector (Neon), then
   bge-reranker-v2-m3 (on ZeroGPU).
3. **Answer**: Gemini writes the answer from the top 5 sections only, with a
   page tag on every fact; two refusal gates (low score, "not in manual").
4. **Hindi/Telugu** (own script or typed in English letters): the question is
   translated for the search; the answer comes back in the user's language.

Code: https://github.com/furkanhf/manualmate
