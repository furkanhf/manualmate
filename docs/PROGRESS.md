# ManualMate — PROGRESS (read this first)

This file is the single place to see where the project is and how to continue.
Claude: read this file at the start of every chat (via the folder access to D:\Projects\manualmate),
and update it at the end of every chat (status, sessions, next steps). Furqan commits it with the other files.

## ▶ START HERE — next chat (5.3): new test questions for the new manuals (step 2b)
Claude: do these steps with Furqan, about 5 per batch, one command at a time, and say where to run each and what to expect.
NEW (chat 5.2): Claude now runs in Claude Code inside VS Code → it CAN run PowerShell on the laptop itself (parser, chunker, checks). Use `& "D:\Projects\manualmate\.conda\python.exe" <script>` (= the activated env). Embedding + evals still run on Colab (Furqan runs, pastes output).
0. Check: `git status; git log --oneline -1` → clean, last commit = "Week 4: 48 manuals parsed, chunked, embedded + eval (chat 5.2)".
1. Step 2b: write ~10–15 new test questions for the NEW manuals, in a NEW file (e.g. eval/text_queries_new.jsonl; do not touch the 50). Same format as eval/text_queries.jsonl, labeled manual + page, Claude reads the pages in data/processed/ to label. Indian models first: Blue Star MDVRYO (ac_012), LG TS-Q19YNZE (ac_009); NOT Panasonic AKY ac_005 (mostly images, see Week 4 step 2 notes). Include questions that name the MODEL (routing to 1 manual) and some that name only the brand (router must ask / pick).
2. Colab 7e on the new file (may need a `--queries` option in run_eval.py; check first) → this is the fairer test than the 50.
3. Then step 3: 20 Hindi/Telugu questions (eval/multilingual_queries.jsonl).

## ▶ CURRENT STATUS (update every chat)
- Date: 2026-11-12 (chat 5.2)
- Last commit: "Week 4: 48 manuals parsed, chunked, embedded + eval (chat 5.2)". Before: cb30f92 (router batch, 48), 5badb88, 507a9ab, aadb9f4, 7097dd3, 32abf60 (week-4 step 1 closed).
- Week: 4 IN PROGRESS. Step 1 (section chunking) DONE. Step 2 DONE in chat 5.2: 48 manuals parsed (3,220 pages, 93 low_text), chunked (8,134 section chunks, 0 in safe mode; old 20 = 4,159 unchanged), embedded on Colab (Neon = 8,134 chunks), retrieval eval R@5 0.98 kept (see "Week 4 results — 48 manuals"). Next: step 2b (new test questions).
- Counts: AC 9, washing machine 11, refrigerator 9, microwave 9, router 10 = 48. Decided: keep 48 (no trimming to 9 each, no padding to 50); resume says "48 manuals".
- Chat 5.1 notes: see "Week 4 step 2 — manual collection (chat 5.1)" below (download script, drops, trims, scanned files, gotchas).
- From chat 4.1 (still true):
  - Last commit: f6cc9c7 (week 4: section chunking, parser paras, Colab 7e). Before: f890ff9 (headings.py), 7f51a85, 60458ce (week 3 closed).
  - Week: 4 IN PROGRESS. Step 1 (section chunking) DONE and KEPT → R@5 0.98 (see "Week 4 results" below). Neon now holds the 4,159 section chunks.
  - Scope cut (chat 4.1, Furqan has 2 more resume projects): ~60 manuals instead of 150 (decide later; the resume must say the real number); troubleshooting "one row = one chunk" and LLM keywords skipped; simple safe mode only. Weeks 5–6 to decide later (if week 5 is skipped, remove "OCR 91%" and "image query 84%" from the resume).
  - Done in chat 4.1 (no Gemini): src/manualmate/ingest/headings.py (bold/bigger font, 8 noise rules; Furqan spot check of 200 samples ≈ 5% wrong); pdf_parser.py also writes "paras" (headings as own paragraphs; "text" unchanged, checked on 1,671/1,671 pages); chunker.py --mode section (default) / --mode fixed (week-1 baseline, checked: identical 2,230 chunks).
  - Section mode rules: each chunk starts with a title line "<Brand> <MODEL> <category> | <parent> > <heading>" (2 levels); a new heading always starts a new chunk; a chunk never crosses a page; long sections cut at sentence ends (no overlap); no merging; body < 12 letters → no chunk; safe mode: < 0.3 headings/page → fixed (0 of 20 manuals). 4,159 chunks (fixed: 2,230), median 448 chars.
  - Not built yet: section expansion (send sibling parts of a section to Gemini); chunk_id has the section number (_sNNNN_) for later. Week-3 answer eval / RAGAS NOT re-run on the new chunks yet (needs Gemini).

## ▶ STOP RULE (agreed in chat 3.4 — no endless tuning)
Change something only if: (1) the measurement is wrong (bug), (2) a resume target is missed, or (3) real users would often get a wrong or unsafe answer (not 1 case in 50).
Otherwise: write it in this file and move on. If a target is missed: ONE round of fixes, measure again, write the result, move on. Never tune on the 50 test questions; never change labels.

## ▶ NEXT STEP (week 4)
1. DONE: section chunking, measured on Colab (7e), kept.
2. DONE in chat 5.2 (results: "Week 4 results — 48 manuals"). Was: laptop `python src/manualmate/ingest/pdf_parser.py` (all) → `python src/manualmate/ingest/chunker.py` (section mode) → check the per-manual "mode" column (safe mode = few headings) and chunk counts → Colab 5 (upload chunks.jsonl) – 6 (embed to Neon; old 20 are re-embedded too) → 7e (retrieval eval on the 50 questions; they cover only the old 20 manuals, so this checks that 28 more manuals did not hurt routing/R@5).
2b. Later: a few new test questions for the new manuals (Indian models first: Panasonic AKY, Blue Star, LG TS-Q19YNZE). Fairer test than the 50.
3. 20 Hindi/Telugu questions (eval/multilingual_queries.jsonl, labeled manual + page) → retrieval eval on Colab (no Gemini).
4. After the Gemini daily quota resets: quick answer check on the new chunks (7d --limit 3, then the full run if clean) + Hindi/Telugu answers.

## Old next step (done in chat 3.4, kept for reference)
1. Colab 🔑 Secrets: `DATABASE_URL` and `GEMINI_API_KEY` both listed, value only, Notebook access on.
2. Run notebook sections 1–4 in order (GPU → git pull → pip install → DATABASE_URL). They are REQUIRED before 7d in every new Colab session. Expect `Tesla T4`, `91a6ad8 …` (or newer), no pip errors, `DATABASE_URL loaded`.
3. Skip sections 5, 6, 7, 7b, 7c. (Section 5 = chunks.jsonl upload is NOT needed: the answer code reads chunks from Neon, not from data/.)
4. Run the first 7d cell (`--limit 3`, 1–2 min; models download first). Expect `GEMINI_API_KEY loaded`, `=== text: 3 questions` + `q001 answer ok score=0.9…`, `=== oos: 3` (mostly `refuse not_in_manual`), `=== clarify: 3` (`ask`/`answer`), `===== SUMMARY` + threshold table. Paste the whole output to Claude.
4b. Quick test done and clean. After the asked_first fix is pushed: run section 2 again (git pull → expect the new commit).
5. If clean: run the second 7d cell (full: 50 text + 30 oos + 10 clarify, ~80 Gemini calls, 5–8 min). If the output shows `Gemini failed` lines → re-run with `!python scripts/run_answer_eval.py --sleep 5`.
6. Run section **8** (the last cell, "Download the NEW results files") → save `<date>_answers.jsonl` + `<date>_answers_summary.json` in `D:\Projects\manualmate\eval\results\` → commit.
7. Claude reads the results on the laptop → sets REFUSE_BELOW / CLARIFY_BELOW in `src/manualmate/generate/refusal.py` (now 0.10 / 0.30, temporary) from the threshold sweep; looks at out-of-scope answered ids and in-scope wrongly refused.
8. Then: RAGAS faithfulness script (laptop, Gemini as judge, uses `strip_tags`) → target 0.90. Refusal target 90%. Optional: ~10 vague questions in clarify_queries.jsonl.

## Code facts that help (checked in chat 3.4)
- Code is in the src layout: `src/manualmate/<package>/…` (e.g. src/manualmate/generate/refusal.py). Docs below say `generate/refusal.py` as a short name.
- Gemini model: `gemini-3.5-flash-lite` (MODEL_NAME in answer.py, pinned). call_gemini() retries 3 times (waits 5 s, 10 s), then returns None and prints `Gemini failed: …`.
- run_answer_eval.py: `--limit N`, `--sleep S` (default 2.0 s after each Gemini call), `--summarize <file>` (re-make the summary from a saved answers file, no new Gemini calls).
- The old `claude/chat-3.2-summary.md` and `claude/chat-3.3-summary.md` are NOT in the project folder. This file replaces them; the week-3 details are in the code itself.

## How to start a new chat
1. VS Code → File → Open Folder → D:\Projects\manualmate.
2. Terminal (Ctrl+`): `conda activate D:\Projects\manualmate\.conda`
3. `git status; git log --oneline -1` → expect clean and the "Last commit" above.
4. New Claude chat (in the manualmate Project, opened in the Claude desktop app). First message:
   "Continue ManualMate. Ask for folder access to D:\Projects\manualmate, then read docs/PROGRESS.md and follow ▶ START HERE."
5. Claude asks for folder access → approve it (needed again in every new chat).


## What ManualMate is
Multimodal, multilingual RAG assistant for product manuals (portfolio project for AI engineer roles).
- Input: text question (English, Hindi, Telugu) and/or photo of a product or model-number label.
- Output: grounded answer with citations (manual + page), or a refusal for out-of-scope questions.
- Manuals are English only. "Multilingual" = cross-lingual queries: Hindi/Telugu questions find English chunks via bge-m3; Gemini answers in the user's language.
- Data target: 150 manuals, 5 categories (washing machine, AC, refrigerator, microwave, Wi-Fi router). Now: 48 (scope cut to ~50, chat 4.1/5.1).

## Resume targets (must be measured, not assumed)
Recall@5 0.70 → 0.88 | RAGAS faithfulness 0.74 → 0.90 | OCR accuracy 91% | image query correct page in top 3: 84% of 100 | correct refusals 90% | p95 latency 2.4 s.
- Resume says "Render" → change to "Hugging Face Spaces" (Render free tier = 512 MB RAM).
- Resume says "~9,000 chunks for 150 manuals": the current chunker gives 2,230 per 20 manuals (~17,000 for 150). Write the real measured count at the end.

## Tech stack
Python 3.11, PyMuPDF (import pymupdf, not fitz), pdfplumber, bge-m3 embeddings (sentence-transformers 6.2.0), bge-reranker, BM25 (rank-bm25), PostgreSQL + pgvector on Neon (free tier), Gemini API (google-genai), OpenCV + PaddleOCR, RAGAS, FastAPI, Gradio, Docker. Deploy on Hugging Face Spaces (Docker).

## Where each part runs
- Laptop / VS Code: all code, parsing, chunking, BM25, Gemini calls, evaluation, FastAPI, Docker, deploy.
- Google Colab (T4 GPU): heavy jobs — embedding all chunks with bge-m3 (notebooks/01_embed_on_colab.ipynb), hybrid eval, optional batch OCR. Colab clones the GitHub repo; chunks.jsonl is uploaded by hand (data/ is gitignored); vectors go straight to Neon. Colab keeps nothing; the model re-downloads each session (~15 s).
- Kaggle (2026-11-10): same jobs as Colab when Colab is not usable. Notebook GPU T4 x2, Internet On, Secrets via kaggle_secrets.UserSecretsClient; cells in the "Kaggle cells" section below. Results: right panel Output → download.
- Neon = shared database (Colab, Kaggle, laptop, app). GitHub = shared code.
- Rules: (1) same embedding model everywhere — only embedder.py loads it; (2) Colab keeps nothing; (3) pinned versions in requirements.txt (torch not listed there).

## Environment rules
- Project folder: D:\Projects\manualmate (Windows, PowerShell in VS Code). GitHub: https://github.com/furkanhf/manualmate (main).
- Conda env: D:\Projects\manualmate\.conda (Python 3.11.18). Each new terminal: conda activate D:\Projects\manualmate\.conda (terminals start in (base)).
- Use conda only to create the env. Install every library with pip, never conda install. torch on the laptop = CPU build (pip install torch --index-url https://download.pytorch.org/whl/cpu).
- Caches on D: HF_HOME=D:\cache\huggingface (verified), PIP_CACHE_DIR=D:\cache\pip.
- .env (top level) holds GEMINI_API_KEY and DATABASE_URL. Never pushed. .env.example kept with placeholders.
- VS Code interpreter = the .conda python; Furqan often runs files with the ▷ "Run Python File" button. Scripts need pip install -e . (pyproject.toml, src layout).
- Colab Secrets: DATABASE_URL = only the URL (postgresql://...?sslmode=require&channel_binding=require), no quotes, no "DATABASE_URL=", no extra lines (an extra line once caused a psycopg "channel_binding" error).
- GOTCHA: if Claude changes a file that is open in VS Code, close the tab / check there is no unsaved dot before Run — VS Code once saved the old buffer over Claude's new pdf_parser.py.
- LF→CRLF git warnings are harmless.
- Claude reaches D:\Projects\manualmate through the Claude desktop app (device bridge). It works only while the desktop app is open, signed in and online on the laptop, and the chat is opened in that app. Folder access must be granted again in each new chat. D:\PDF manuals = backup of the downloaded PDFs (optional).

## Sessions
- Day 1 (2026-11-05): plan, skeleton, env, check_setup, first commit.
- Day 2 (2026-11-06, chats 2.1 + 2.2): week 1 complete (20 manuals, parser, chunker, 50 questions, Colab embedding, dense baseline 0.72) + most of week 2 retrieval (BM25, hybrid, query parser, router).
- Day 2 follow-up (2026-11-08): plain-language guide ManualMate_Project_Explained.pdf (14 pages) made in chat 2.1. NOTE: it covers week 1 only; it does not include BM25/hybrid/router or the 0.90 result.
- Day 3 (2026-11-09, chat 3.1): reranker added (commit d0a6b0b) + notebook section 7b. Result: R@1 0.66, R@5 0.94, MRR 0.78 (rr-issue). Query rewriting (Gemini, commit 5377a1a) tested: NO gain → not used in the pipeline. Week 2 closed. Next: week 3 (generate/).
- Day 3 evening (2026-11-09, chat 3.2): week 3 code written and tested on the laptop: generate/prompts.py, citations.py, refusal.py, answer.py, eval/out_of_scope.jsonl (30), scripts/run_answer_eval.py + notebook 7d. Last commit 91a6ad8. Colab did not run.
- Day 4 (2026-11-10, chat 3.3): set up Kaggle as a GPU option (T4 x2 works). Eval not run yet. Kaggle cells: section below.
- Day 4 (2026-11-10, chat 3.4, later): full answer eval on Colab (1047), thresholds set, citation fix, RAGAS script (ragas 0.4.3 + Gemini judge via OpenAI endpoint) → 0.97. Week 3 closed. Gemini free quota: flash-lite 500/day + 15/min (answers + judge share it), flash 20/day.
- Day 4 (2026-11-10, chat 3.4): Colab works again → Colab is the main place for the eval, Kaggle the backup. Checked notebook 7d and the code; added the Colab run order (sections 1–4 first) to NEXT STEP. Eval not run yet.
- Day 5 (2026-11-11, chat 5.1): week 4 step 2 — manual collection 20 → 48 with scripts/download_manuals.py + data/download_list.csv; commits 7097dd3, aadb9f4, 507a9ab, 5badb88 + router batch. Not parsed/embedded yet.
- Day 6 (2026-11-12, chat 5.2, first chat in Claude Code / VS Code): parsed + chunked the 48 manuals on the laptop (Claude ran them), embedded 8,134 chunks on Colab, eval 7e → R@5 0.98 kept, R@1 0.72 (brand-only questions now match 2 manuals). Week 4 step 2 closed.

## Data: 48 manuals (data/manifest.csv: manual_id,brand,model,category,language,filename,source_url)
The list below is the first 20 (week 1). The 28 added in chat 5.1 are in data/manifest.csv and data/download_list.csv (with notes); summary in "Week 4 step 2" below.
AC: ac_001 samsung_ar-by_ac.pdf (12 p) | ac_002 daikin_ftks-k-series_ac.pdf (44) | ac_003 daikin_ftxz-us7_ac.pdf (48) | ac_004 daikin_ftxs-l-series_ac.pdf (44)
Washing machine: wm_001 ifb_senator-mbn-8014k (78) | wm_002 samsung_ww80t (72) | wm_003 lg_fhm1408bdl (44) | wm_004 bosch_waj20180za (52)
Refrigerator: rf_001 samsung_rf9000f (96) | rf_002 bosch_kgn (32) | rf_003 whirlpool_neo-ic355 (12, 7 low_text pages) | rf_004 bosch_kgn56 (32)
Microwave: mw_001 ifb_30brc2 (44) | mw_002 samsung_mc28h5025vk (64) | mw_003 lg_mc2846bg (232, Indian recipes) | mw_004 panasonic_nn-st64qw (88)
Router: rt_001 d-link_dir-825-ac (222) | rt_002 netgear_r6350 (204) | rt_003 tp-link_archer-c6 (99) | rt_004 tenda_ac10 (152)
- 5 scanned (no-text) PDFs kept in data/scanned for week-5 OCR tests.
- Daikin K and L manuals are almost identical (questions must name the model).
- Avoid manuals.plus/asin pages and "Download_PDFS" sites (AI-written fakes). The Panasonic site blocks direct PDF links.

## Week 4 step 2 — manual collection (chat 5.1, 2026-11-11)
- No usable bulk dataset exists (checked Kaggle/HF/E-Manual corpus = text only/PM209 = phones/CheckManual = LLM-made/Internet Archive = old+scanned). So: curated link list + script.
- scripts/download_manuals.py (new): reads data/download_list.csv (manifest columns + note), downloads to data/raw/, checks each PDF, writes data/download_report.csv. Statuses: OK, SHORT (<10 p), SCANNED (>50% pages <50 chars), NOT_EN (<80% a-z letters), DUPLICATE (same sha256 as a manifest PDF), BAD_FILE (HTML instead of PDF → renamed .bad), BY_HAND (no url: save the PDF as data/raw/<filename>, run again), BLOCKED_SITE (manuals.plus, manualslib, scribd, notice-facile …), DOWNLOAD_FAILED, IN_MANIFEST, SKIPPED (note starts with DROPPED/SCANNED). It never edits the manifest; Claude checks files, then adds manifest + aliases rows.
- Run: `python scripts/download_manuals.py --category router` or `... rf_007 wm_011`.
- Sites that block scripts (download by hand in the browser): Panasonic (HTTP 403), Godrej (404 even in browser → dropped), Croma (dead links), some retailers. Bosch/Siemens (media3), TP-Link, ASUS, Netgear, Haier AU, Electrolux AU, canadianappliance.ca worked.
- NOT_EN check misses Latin-script languages: Panasonic NA-FD135Z3 (Malay pages 58-112) and LG WT7005C (Spanish 39-80) were trimmed to the English pages (page numbers unchanged). Routers keep 0-5 non-English regulatory pages (not worth trimming).
- Added (28):
- ac (+5): ac_005 panasonic cs-su-aky-series, ac_006 mitsubishi msz-ap25-50vgkd2, ac_009 lg ts-q19ynze, ac_012 blue-star mdvryo-series, ac_013 panasonic cs-rs18uky-series
- washing-machine (+7): wm_005 bosch wga244a0me, wm_007 panasonic na-fd135z3, wm_008 haier hwf-an1-series, wm_009 electrolux ewf8024cdwa, wm_010 whirlpool fscr12430, wm_011 samsung wa47cg3500, wm_012 lg wt7005c
- refrigerator (+5): rf_006 haier hrf510-series, rf_008 electrolux ete-ebe-series, rf_011 samsung srt3500b, rf_009 panasonic nr-bx471-421, rf_007 lg lrtnc0915
- microwave (+5): mw_005 bosch bfl523m, mw_006 siemens bf722, mw_007 electrolux emb2527ba-emf2527ba, mw_008 whirlpool amw509ix, mw_010 sharp smc1169ks
- router (+6): rt_005 asus rt-ax92u, rt_006 tp-link archer-ax23, rt_007 tp-link deco-x20-x60, rt_008 netgear orbi-rbe770, rt_009 d-link dir-841, rt_010 mercusys mr60x
- Dropped: Carrier Midea (care/troubleshooting pages are images, no model), Hitachi 65MP3199 (cassette/ducted), Haier HSU14K (= scanned haier_inverter_ac.pdf), Samsung WA70A4022FS (2-page print sheet), Godrej washing machine + microwave (dead links), Liebherr (403 / HTML), Voltas (no official PDF online).
- Moved to data/scanned (week-5 OCR): lloyd_ac-common_ac.pdf (79 p, all images), fisher-paykel_rf381t-series_refrigerator.pdf (16 of 44 p images). data/scanned now has 7 files.
- ac_005 Panasonic CS-SU AKY (found in chat 5.2 parse): 14 of 32 pages low_text, only 8,228 chars; pages 6–10 (operation/remote) are empty = images. Kept in the index (install, ECO, Wi-Fi, PM 0.1 filter text is usable; 26 chunks). Do NOT use it for new test questions; candidate for week-5 OCR.
- New brands in aliases.yaml: mitsubishi, blue-star, godrej, electrolux, haier, siemens, sharp, mercusys, asus. Router test (query_parser.parse) checked for every new manual; old ones unchanged.
- GOTCHAS this chat: (1) VS Code saved its old buffer over Claude's new download_manuals.py once → close the tab (Don't Save) before running. (2) Moving files: the whole data/raw was once moved into data/scanned by mistake (Claude moved them back) → only move the files named. (3) Two Claude chats edited manifest/aliases at the same time (no damage) → use ONE chat at a time. (4) Backups of manifest/aliases/PROGRESS are in _to_delete/ (gitignored; delete it any time).

## Done so far (from GitHub, last commit c921ebf, 2026-11-06 21:50)
- ingest/pdf_parser.py — manifest → data/processed/<manual_id>.jsonl, one record per page (labels + page + text + n_chars + low_text). Removes margin headers/footers that repeat on ≥50% of pages, stamps/watermarks on ≥80% of pages, margin page numbers, invisible/icon chars; fixes ligature gaps ("fi lter") only in manuals with ≥20 gaps. 1,671 pages, 42 low_text, 2,131,555 chars.
- ingest/chunker.py — baseline: 1600 chars, 200 overlap, never crosses a page; chunk_id <manual_id>_p<page:03d>_c<index>; 2,230 chunks.
- index/embedder.py (bge-m3, 1024 dims, normalized, fp16 on GPU; the only place the model loads), index/db.py, index/schema.sql (chunks table, HNSW cosine, indexes on manual_id/category/brand), index/bm25.py.
- retrieve/reranker.py (BAAI/bge-reranker-v2-m3 CrossEncoder, max_length 1024, 20 candidates, fp16 on GPU; Colab only), run_eval --method routed_rerank --rerank-text issue|full --candidates N (prints rank changes + rerank ms)
- retrieve/dense.py, retrieve/hybrid.py (RRF), retrieve/query_parser.py + configs/aliases.yaml, retrieve/router.py (model → brand/category → all; asks "Which model?" when >1 manual matches).
- scripts/build_index.py, run_eval.py (--method dense|bm25|hybrid|routed|routed_hybrid, --query-text issue|full), eval_routing.py, notebooks/01_embed_on_colab.ipynb, pyproject.toml.
- eval/text_queries.jsonl (50 hand-labeled Qs, 10 per category; Furqan spot-checked 10/10), eval/clarify_queries.jsonl (10 Qs).
- eval/results/2026-11-06_1431_dense_baseline.json (week 1 baseline).

## Results so far (50 text questions)
| method | R@1 | R@3 | R@5 | R@10 | MRR@10 |
| bm25 only | 0.20 | 0.44 | 0.50 | 0.62 | 0.34 |
| hybrid RRF (all manuals) | 0.36 | 0.58 | 0.64 | 0.72 | 0.48 |
| dense baseline | 0.36 | 0.62 | 0.72 | 0.78 | 0.52 |
| routed dense, full query | 0.50 | 0.78 | 0.82 | 0.94 | 0.65 |
| routed hybrid, issue query | 0.58 | 0.76 | 0.88 | 0.94 | 0.70 |
| routed dense, issue query | 0.62 | 0.88 | 0.90 | 0.94 | 0.75 |
| routed + rerank, reranker reads full question | 0.56 | 0.90 | 0.94 | 0.94 | 0.72 |
| routed + rerank (bge-reranker-v2-m3, 20 cand.), reranker reads issue (BEST) | 0.66 | 0.90 | 0.94 | 0.94 | 0.78 |
Rerank time on Colab T4: avg 310 ms, p95 407 ms per question (CPU on HF Spaces will be much slower — check in week 6; options: fewer candidates, smaller reranker, ONNX).
Weakest category: microwave (R@5 0.70). The 3 misses left after rerank are all microwave and are vocabulary gaps, not ranking errors:
- q033 "smells of fish" → answer p39 says DEODORIZE ("flash the air"); the word smell/odour is not on the page.
- q035 "clock not showing" → p58 troubleshooting ("time is not displayed": power / Eco function); p12 SETTING THE TIME ranks first.
- q037 "make ghee" → p24 Ghee/Bakery auto mode ("6h: Ghee"); recipe pages that list ghee as an ingredient (p157) rank first.
Tried query rewriting (retrieve/query_rewriter.py, gemini-3.5-flash-lite, eval/rewrites.jsonl, run_eval --rewrites): routed dense + rw 0.60/0.88/0.90/0.94/0.74; rerank + rw 0.66/0.88/0.94/0.96/0.78 → same as without (±1 question = noise). The 3 misses stay: q033 reranker also has the word gap; q035 p12 is partly right (high score 0.758); q037 'making ghee' pulls recipe pages p222/223. Decision: do not ship the rewrite (extra Gemini call, no gain); keep the code for week 4. Do not tune more variants on the 50 test questions. The 3 misses are a chunking problem → fix in week 4 re-chunking. Do not change the labels to raise the score.
FINAL WEEK-2 PIPELINE: router → routed dense top 20 → bge-reranker-v2-m3 reads issue text → R@1 0.66, R@5 0.94, MRR 0.78.
Dense baseline misses (week 1): 7 wrong brand on top (fixed mainly by routing), 3 cover/contents pages on top, 5 right manual but nearby page.

## Week 3 files (done 2026-11-09, commit 91a6ad8)
- generate/prompts.py: open-book SYSTEM_PROMPT (8 rules), tag(chunk) → "[mw_003 p24]", build_prompt(full question, chunks, language).
- generate/citations.py: find_tags, check() → invalid (made-up) tags, untagged sentences, precision, ok; strip_tags (RAGAS), is_refusal.
- generate/refusal.py: gate 0 router ask; gate 1 top rerank score (REFUSE_BELOW 0.10, CLARIFY_BELOW 0.30 = TEMPORARY); gate 2 NOT_IN_MANUAL.
- generate/answer.py: generate() (gate 1 → gemini-3.5-flash-lite → gate 2 → citations), retrieve() (router → dense 20 → rerank → top 5, needs GPU), answer(), chunks_for_pages() (laptop test).
- eval/out_of_scope.jsonl: 30 Qs (8 price, 6 near_miss, 4 compare, 4 smart_home, 3 off_topic, 2 local, 2 opinion, 1 unsafe); router check 28/28.
- scripts/run_answer_eval.py: text 50 + oos 30 + clarify 10; gate 1 off during the run, top score saved, threshold sweep in the summary; --limit, --summarize, --sleep.
- Laptop smoke test: q001 answered from [ac_001 p9], citations ok; price question refused.

## Week 4 results — 48 manuals (Colab, 2026-11-12_0705 / _0706, chat 5.2)
Files: eval/results/2026-11-12_0705_routed_issue.json, 2026-11-12_0706_routed_rerank_issue_rr-issue_c20.json. Neon: 8,134 section chunks (48 manuals, 3,220 pages). Same 50 questions (they cover only the old 20 manuals).
Parse: 48 manuals, 3,220 pages, 93 low_text, 4,181,530 chars. Page counts = download_report.csv (wm_012 80 → 38 = Spanish trim, known). Chunk: 8,134 chunks, all 48 in section mode (0 safe mode); old 20 = exactly 4,159 (unchanged); new 28 = 3,975. (The 9–12k estimate was a guess; real count is 8,134.) Embed on T4: 143 s.
| method | R@1 | R@3 | R@5 | R@10 | MRR@10 | route_single |
| 20 manuals: routed dense, issue | 0.62 | 0.92 | 0.96 | 0.98 | 0.77 | — |
| **48 manuals: routed dense, issue** | 0.58 | 0.92 | 0.94 | 0.98 | 0.75 | 0.78 |
| 20 manuals: routed + rerank (issue) | 0.78 | 0.92 | 0.98 | 1.00 | 0.86 | 0.96 |
| **48 manuals: routed + rerank (issue) (BEST)** | 0.72 | 0.92 | **0.98** | 1.00 | 0.83 | 0.78 |
Rerank time on T4: avg 250 ms, p95 326 ms. manual@5 1.00, route_in_scope 1.00.
- Cause of the R@1 / MRR drop: 9 questions name the brand but not the model. With 48 manuals the brand+category now matches 2 manuals (Samsung WM wm_002+wm_011, LG WM wm_003+wm_012, Bosch WM wm_004+wm_005, Samsung RF rf_001+rf_011, D-Link rt_001+rt_009, Netgear rt_002+rt_008) → route_single 0.96 → 0.78; the search runs over both manuals and the other manual's page sometimes wins rank 1. Rank changes (rerank): q014 1→2 (wm_011 top), q016 2→3, q023 1→3 (rf_011 top), q046 1→2 (rt_008 top); q015/q018/q020/q022/q043 unchanged.
- In the real app the router asks "Which model?" when >1 manual matches (gate 0), so these users would pick the model first → the eval number is pessimistic for them. The eval does not simulate the answer to "which model?".
- Only miss at top 5: q009 (Daikin FTXZ humidity, known since week 3, rank 6). Microwave and WM R@5 now 1.00; AC 0.90.
- Stop rule: R@5 target (0.88) met, no bug → no change. Write the realistic number on the resume: "R@5 0.98 on 48 manuals".

## Week 4 results — section chunking (Colab, 2026-11-10_1538 / _1539, commit f6cc9c7)
Files: eval/results/2026-11-10_1538_routed_issue.json, 2026-11-10_1539_routed_rerank_issue_rr-issue_c20.json. 4,159 section chunks in Neon.
| method | R@1 | R@3 | R@5 | R@10 | MRR@10 |
| week 2: routed dense, issue (fixed chunks) | 0.62 | 0.88 | 0.90 | 0.94 | 0.75 |
| week 4: routed dense, issue (section chunks) | 0.62 | 0.92 | 0.96 | 0.98 | 0.77 |
| week 2: routed + rerank (issue), fixed chunks | 0.66 | 0.90 | 0.94 | 0.94 | 0.78 |
| **week 4: routed + rerank (issue), section chunks (BEST)** | **0.78** | 0.92 | **0.98** | **1.00** | **0.86** |
Rerank time on T4: avg 266 ms, p95 337 ms (20 candidates).
- The 3 microwave misses are fixed: q035 rank 13 → 2 (the title line "troubleshooting and information code" was enough, even without the table step); q037 → 4; q033 → 5 (just in, the weakest).
- New miss: q009 (Daikin FTXZ humidity, gold p12 SARARA DRY) rank 6 → ac R@5 0.90 (was 1.00). It was already weak in week 3 (rerank score 0.15, gold not cited). Logged, not tuned.
- Stop rule: one round. "No category drops" is strictly not met (ac −1 question), but net +3/−1, R@1 +0.12, MRR +0.08 → Furqan and Claude decided to KEEP section mode (chat 4.1).
- Caveat for the resume: the fix was designed while looking at 3 of the 50 test questions (the rule itself is general: cut at headings) → 0.98 may be slightly optimistic. The new manuals and the Hindi/Telugu questions are the fairer test.
- Resume line idea: "Recall@5 0.72 → 0.98 (routing, bge-reranker, heading-aware chunking)".

## Week 3 results — full answer eval (Colab, 2026-11-10_1047, commit 115584f)
File: eval/results/2026-11-10_1047_answers.jsonl (+ _summary.json). Gate 1 OFF during the run (scores saved, sweep offline).
| metric | result |
| in-scope answered (50) | 1.00, wrongly refused 0, errors 0; asked first then user picked model: q003, q004 |
| gold page in top 5 | 0.94 (same as week 2) |
| gold page cited | 0.88 |
| citations ok | 0.96 after the "approx." checker fix (0.90 before), tag precision 0.99, answers with a made-up tag: 1 (q008) |
| out-of-scope refused or asked (30) | 1.00 (28 NOT_IN_MANUAL, 2 ask) → refusal target 90% met |
| clarify correct (10) | 1.00 |
| Gemini ms | avg 2813, p50 1474, p95 9190 |
| retrieve ms (dense+rerank, T4) | p50 720, p95 922 |
| total (retrieve+Gemini) | p50 2239, p95 9904 |
Threshold sweep: in-scope answered 1.00 up to 0.10, 0.98 at 0.15, 0.94 at 0.20–0.30, 0.88 at 0.50; out-of-scope refused 1.00 at every threshold (gate 2 catches everything).
- Score gap: out-of-scope max 0.102 (o11), 0.10 (o14, o28), 0.093 (o27); in-scope min 0.15 (q009), then q046 0.17, q033 0.181, q022 0.321. o19 (sushi, LG) scored 0.868 but gate 2 refused it → gate 2 is needed.
- Low-score in-scope answers (q009, q046, q033) are reasonable answers → a score-based clarify gate at 0.30 would hurt them.
- Citation "not ok" (5 → 2): q020, q025, q028 = checker bug (sentence split after "approx.") → FIXED in citations.py (ABBREV_RE). Left, real, logged only: q046 = Gemini note line without a tag, q008 = made-up tag [ac_003 p36] (p36 not a source; gold p37 was retrieved).
- Re-score without new Gemini calls: `python scripts/run_answer_eval.py --summarize eval/results/2026-11-10_1047_answers.jsonl` (summarize now re-runs the current citation checker on the saved answers).
- Thresholds SET in refusal.py: REFUSE_BELOW 0.05, CLARIFY_BELOW 0.05 (clarify band off). Reason in the refusal.py docstring.
- Gold retrieved but not cited: q006 (cited p4/p12, gold p38), q009 (cited p13, gold p12). Retrieval misses (known, week-4 chunking): q033, q035, q037.
- Latency: average 2.8 s hides a long Gemini tail (p95 9.2 s). Do NOT put "p95 2.8 s" on the resume. Measure end-to-end in week 6 (options: streaming, lower max_output_tokens, check retries).

## Week 3 results — RAGAS faithfulness (laptop, scripts/run_ragas.py)
Setup: ragas 0.4.3 (+ instructor>=1.9, langchain-community<0.4, openai). Judge = gemini-3.5-flash-lite through Gemini's OpenAI-compatible endpoint (async client; the google-genai client is sync-only inside ragas). max_tokens 8192, reasoning_effort low (default 1024 tokens -> IncompleteOutputException). gemini-3.5-flash as judge: free tier only 20 requests/day -> not usable for 50 answers.
Context = whole source pages from Neon (the 1047 run saved page tags, not chunk ids) -> slightly lenient.
| run | code | faithfulness | note |
| 2026-11-10_1728 | old scoring (Faithfulness.ascore, no claims saved) | 0.734 | 6 answers scored 0.0; on re-run they scored 1.0 -> judge failures, cannot be checked (no claims saved) |
| 2026-11-10_1759 | current (keeps claims + verdicts + reasons) | **0.98** (50/50 scored) | below 1.0: q006 0.50, q009 0.80, q042 0.83, q032 0.89 |
| 2026-11-10_1822 | current | **0.959** (50/50 scored) | below 1.0: q029 0.00 ("rattling" judge error), q006 0.50, q009 0.75, q042 0.83, q044 0.89 |
**FINAL: RAGAS faithfulness 0.97 (mean of 2 judge runs with the current code, range 0.959–0.98) → target 0.90 MET.** A 3rd run hit the free daily quota (flash-lite 500 requests/day) → not needed (stop rule: both runs far above target).
Resume line: "RAGAS faithfulness 0.97 (2 judge runs, 0.96–0.98; Gemini judge)".
- The 4 "unsupported" claims of run 1759 are judge strictness, not hallucinations (e.g. q009: manual says SARARA DRY "mixes dehumidified cool air with room air"; judge wants the words "not too cold").
- The judge is not stable: q029 scored 0.40 at 17:43 (judge put the question word "rattling" into every claim) and 1.00 at 17:59 with the same code.
- Limits to write next to the number: judge = same model family as the writer (self-bias possible); page-level context; one judge run is noisy.
- Debug: `python scripts/run_ragas.py --ids q029,q006` prints every claim, verdict and reason.

## Still empty (not started)
filters.py, metadata.py, table_extractor.py, config.py, configs/settings.yaml, scripts/ingest.py, vision/, api/, ui/app.py, multilingual/image eval files, RAGAS faithfulness script.

## Next steps
1. Week 2 DONE (reranker kept, rewrite not kept). Rewrite result files committed (5df2c3a).
2. Week 3 NOW: code done. Next: run scripts/run_answer_eval.py on Colab section 7d (Kaggle = backup) (quick test --limit 3, then full) → download results → pick gate-1 threshold from the sweep → RAGAS faithfulness (target 0.90); refusal target 90%. Include Furqan's idea: when the question is vague / top rerank score is low → ask a follow-up question or refuse; add ~10 vague questions to clarify_queries.jsonl and measure 'asked when it should' separately (it does not raise Recall). Warning: score can be high on a wrong page (q035 0.758) → tune the threshold on data.
3. Later: week 4 (150 manuals, Hindi/Telugu questions), week 5 (vision), week 6 (FastAPI, Docker, Gradio, HF Spaces, latency; Neon free tier sleeps after ~5 min → cold start +0.5–1 s).

## Small TODOs
- Move paths/settings from pdf_parser.py and chunker.py into configs/settings.yaml + config.py.
- Optional: rename samsung_ar-by_ac.pdf → samsung_ar9500t_ac.pdf (then re-run parser + chunker + index).
- LG recipe tables come after the steps in reading order (fix in content-aware chunking).
- Optional: update ManualMate_Project_Explained.pdf with week 2 results and copy it into D:\Projects\manualmate\docs\.

## Kaggle cells (one code cell each, run one at a time)
Shortcut: the same cells (+ a download cell 7) are in `notebooks/02_eval_on_kaggle.ipynb`. In Kaggle: File → Import Notebook → upload that file.
Setup once: Create → New Notebook. Right panel → Session options: Accelerator **GPU T4 x2**, Internet **On** (phone-verified account).
Add-ons → Secrets: `DATABASE_URL`, `GEMINI_API_KEY` (value only, no quotes, no extra line). Tick both for this notebook.
If `nvidia-smi: command not found` → the session is CPU → select the GPU accelerator and restart.

Cell 1 — GPU (expect Tesla T4):
```
!nvidia-smi -L
```
Cell 2 — code (expect the last commit hash):
```
import os
if not os.path.exists("/kaggle/working/manualmate"):
    !git clone https://github.com/furkanhf/manualmate.git /kaggle/working/manualmate
%cd /kaggle/working/manualmate
!git pull -q
!git log --oneline -1
```
Cell 3 — libraries (Kaggle has torch; pip conflict warnings are OK):
```
!pip install -q sentence-transformers==6.2.0 rank-bm25==0.2.2 "psycopg[binary]" pgvector python-dotenv google-genai
!pip install -q -e . --no-deps
```
Cell 4 — secrets (expect `secrets loaded`):
```
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["DATABASE_URL"] = secrets.get_secret("DATABASE_URL")
os.environ["GEMINI_API_KEY"] = secrets.get_secret("GEMINI_API_KEY")
print("secrets loaded")
```
Cell 5 — quick test (expect `q001 answer ok score=…` lines, then `===== SUMMARY`):
```
!python scripts/run_answer_eval.py --limit 3
```
Cell 6 — full eval (only after cell 5 is clean):
```
!python scripts/run_answer_eval.py
```
After a `git push` from the laptop: run cell 2 again before cells 5/6.
If Neon does not connect on Kaggle → use the Colab notebook section 7d (now the main option anyway).

## How Furqan wants to work (commands and explanations)
Explanations
- Simplified Technical English (ASD-STE100): short sentences, one action per step, simple words, active voice.
- For a new component: analogy first ("what it is like"), then "why we need it", then what is in the file.
- Explain before writing code when he asks "what are we doing / why".

Steps and commands
- About 5 steps per batch. He sends the output, then the next batch.
- Each step says WHERE: laptop terminal, VS Code file, Colab/Kaggle cell.
- Each step says the EXPECTED OUTPUT, so he can check it himself.
- Commands in a code block, ready to paste. One command line per step where possible.
- Simple no-code steps (open a folder, delete a file, click a button) in plain words, not terminal commands.
- Commit command after each file: `git add <files>; git commit -m "<message>"; git push` → expect `N file(s) changed` and `main -> main`.
- LF→CRLF warnings are harmless; say so instead of letting him worry.

Workflow
- One file at a time. Claude writes the file on the laptop (device bridge) and tests it there when possible; Furqan runs it with ▷ "Run Python File", pastes the output, then commits.
- Before he runs a file that Claude changed: close the tab in VS Code (Don't Save), so VS Code does not save its old copy over the new file.
- New GPU jobs (Colab/Kaggle): quick test first (--limit 3), paste the output, then the full run.
- Put every Colab eval command into the notebook file and push it (cells added in the Colab tab are lost on reload). Kaggle saves its own notebook.
- Claude cannot see Colab/Kaggle outputs → Furqan pastes them, or downloads the result files into eval/results so Claude reads them on the laptop.
- Do not tune on the 50 test questions; do not change labels to raise a score.
