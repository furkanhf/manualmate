"""Measure retrieval quality on the text test questions.

For each question in eval/text_queries.jsonl:
  1. search the database for the top K chunks,
  2. a HIT means one of them has the expected manual_id and one of the
     expected pages.

Recall@K = hits / number of questions.
MRR      = mean of 1/rank of the first correct chunk (0 if none in top 10).
Manual@5 = the correct MANUAL is in the top 5 (page can be wrong).

Run (--method: dense, bm25 or hybrid; --name defaults to the method):
    python scripts/run_eval.py --method dense --name dense_baseline
    python scripts/run_eval.py --method bm25      (no bge-m3 needed: laptop is OK)
    python scripts/run_eval.py --method hybrid    (needs bge-m3: run on Colab)
    python scripts/run_eval.py --method routed    (parse -> route -> dense in scope)
    python scripts/run_eval.py --method routed --query-text full
    python scripts/run_eval.py --method routed_rerank   (routed dense top 20 ->
        bge-reranker -> new order; needs GPU: run on Colab)
        --rerank-text full (default): the reranker reads the whole question.
        issue: the reranker reads the question WITHOUT brand and model words.
        --candidates 20: how many dense results the reranker reads.
        --query-text issue (default): search with the question WITHOUT brand and
        model words. full: search with the whole question.
The results are saved to eval/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from manualmate.index.bm25 import BM25Index
from manualmate.index.db import get_connection
from manualmate.index.embedder import MODEL_NAME, embed_texts
from manualmate.retrieve import dense, hybrid, reranker
from manualmate.retrieve.query_parser import parse
from manualmate.retrieve.router import route

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "eval" / "text_queries.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
KS = [1, 3, 5, 10]
METHODS = ["dense", "bm25", "hybrid", "routed", "routed_hybrid", "routed_rerank"]


def first_hit_rank(results: list[dict], query: dict) -> int | None:
    """Rank (1-based) of the first correct chunk, or None."""
    for rank, r in enumerate(results, start=1):
        if r["manual_id"] == query["manual_id"] and r["page"] in query["pages"]:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="dense")
    parser.add_argument("--name", default=None,
                        help="name of the results file (default: the method)")
    parser.add_argument("--query-text", choices=["issue", "full"], default="issue",
                        help="routed methods: search text without brand/model words "
                             "(issue) or the whole question (full)")
    parser.add_argument("--rerank-text", choices=["issue", "full"], default="full",
                        help="routed_rerank: the text the reranker reads")
    parser.add_argument("--candidates", type=int, default=reranker.CANDIDATES,
                        help="routed_rerank: how many dense results to rerank")
    args = parser.parse_args()
    routed = args.method.startswith("routed")
    reranked = args.method == "routed_rerank"
    if args.name is None:
        args.name = f"{args.method}_{args.query_text}" if routed else args.method
        if reranked:
            args.name += f"_rr-{args.rerank_text}_c{args.candidates}"

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    # Routed methods: parse each question, pick the manuals, and (by default)
    # search with the issue text only.
    for q in queries:
        if routed:
            parsed = parse(q["question"])
            r = route(parsed)
            q["_scope"], q["_route"] = r.manual_ids, f"{r.reason}: {','.join(r.manual_ids)}"
            q["_parsed"] = parsed.summary()
            q["_text"] = parsed.issue if args.query_text == "issue" else q["question"]
        else:
            q["_scope"], q["_route"], q["_parsed"], q["_text"] = None, None, None, q["question"]

    # Only BM25 alone runs without bge-m3 (no torch needed).
    if args.method != "bm25":
        vectors = embed_texts([q["_text"] for q in queries], show_progress=False)
    else:
        vectors = [None] * len(queries)
    max_k = max(KS)

    per_query = []
    rerank_ms = []
    with get_connection() as conn:
        bm25_index = None
        if args.method in ("bm25", "hybrid", "routed_hybrid"):
            bm25_index = BM25Index.from_db(conn)
            print(f"BM25 index: {len(bm25_index)} chunks")
        for query, vector in zip(queries, vectors):
            scope = query["_scope"]
            dense_rank = None
            if args.method in ("dense", "routed"):
                results = dense.search_by_vector(conn, vector, k=max_k, manual_ids=scope)
            elif reranked:
                candidates = dense.search_by_vector(conn, vector, k=args.candidates,
                                                    manual_ids=scope)
                dense_rank = first_hit_rank(candidates, query)
                text = query["question"] if args.rerank_text == "full" else query["_text"]
                start = time.perf_counter()
                results = reranker.rerank(text, candidates, top_k=max_k)
                rerank_ms.append((time.perf_counter() - start) * 1000)
            elif args.method == "bm25":
                results = bm25_index.search(query["_text"], k=max_k)
            else:
                results = hybrid.search_by_vector(conn, bm25_index, query["_text"],
                                                  vector, k=max_k, manual_ids=scope)
            rank = first_hit_rank(results, query)
            per_query.append({
                "dense_rank": dense_rank,
                "id": query["id"],
                "category": query["category"],
                "type": query["type"],
                "question": query["question"],
                "expected": f"{query['manual_id']} p{query['pages']}",
                "rank": rank,
                "manual_in_top5": any(r["manual_id"] == query["manual_id"]
                                      for r in results[:5]),
                "top5": [f"{r['manual_id']} p{r['page']} ({r['score']:.3f})"
                         for r in results[:5]],
                "route": query["_route"],
                "parsed": query["_parsed"],
                "search_text": query["_text"],
            })

    n = len(per_query)
    metrics = {f"recall@{k}": sum(1 for q in per_query
                                  if q["rank"] and q["rank"] <= k) / n for k in KS}
    metrics["mrr@10"] = sum(1 / q["rank"] for q in per_query if q["rank"]) / n
    metrics["manual@5"] = sum(q["manual_in_top5"] for q in per_query) / n

    by_category = defaultdict(list)
    for q in per_query:
        by_category[q["category"]].append(q)
    category_recall = {
        cat: sum(1 for q in qs if q["rank"] and q["rank"] <= 5) / len(qs)
        for cat, qs in sorted(by_category.items())
    }

    # --- Print ---------------------------------------------------------------
    if routed:
        in_scope = sum(q["manual_id"] in q["_scope"] for q in queries) / len(queries)
        single = sum(len(q["_scope"]) == 1 and q["manual_id"] in q["_scope"]
                     for q in queries) / len(queries)
        metrics["route_in_scope"], metrics["route_single"] = in_scope, single
    print(f"\nRun: {args.name}   method: {args.method}   model: {MODEL_NAME}"
          f"   questions: {n}\n")
    for key, value in metrics.items():
        print(f"  {key:<10} {value:.2f}")
    if reranked:
        # The first call also loads the model, so leave it out of the timing.
        times = sorted(rerank_ms[1:]) or rerank_ms
        metrics["rerank_ms_avg"] = sum(times) / len(times)
        metrics["rerank_ms_p95"] = times[min(len(times) - 1, int(0.95 * len(times)))]
        moved = [q for q in per_query if q["dense_rank"] != q["rank"]]
        print(f"  reranker: {reranker.MODEL_NAME}, {args.candidates} candidates,"
              f" reads the {args.rerank_text} question")
        print(f"  rerank time per question: avg {metrics['rerank_ms_avg']:.0f} ms,"
              f" p95 {metrics['rerank_ms_p95']:.0f} ms")
        print(f"\nRank of the correct chunk, dense -> reranked ({len(moved)} changed):")
        for q in moved:
            print(f"  {q['id']}  {q['dense_rank']} -> {q['rank']}")
    print("\nRecall@5 by category:")
    for cat, value in category_recall.items():
        print(f"  {cat:<16} {value:.2f}")
    misses = [q for q in per_query if not q["rank"] or q["rank"] > 5]
    print(f"\nMisses at top 5 ({len(misses)}):")
    for q in misses:
        got = q["top5"][0] if q["top5"] else "(nothing)"
        print(f"  {q['id']} expected {q['expected']:<16} got {got}"
              f"   | {q['question'][:60]}")
        if q["route"]:
            print(f"        route {q['route']}   | search: {q['search_text'][:60]}")

    # --- Save ----------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = RESULTS_DIR / f"{stamp}_{args.name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": args.name,
            "method": args.method,
            "query_text": args.query_text if routed else None,
            "reranker": ({"model": reranker.MODEL_NAME, "candidates": args.candidates,
                          "rerank_text": args.rerank_text} if reranked else None),
            "model": MODEL_NAME,
            "date": stamp,
            "n_questions": n,
            "metrics": metrics,
            "recall@5_by_category": category_recall,
            "per_query": per_query,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
