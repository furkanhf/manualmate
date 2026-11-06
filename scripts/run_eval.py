"""Measure retrieval quality on the text test questions.

For each question in eval/text_queries.jsonl:
  1. search the database for the top K chunks,
  2. a HIT means one of them has the expected manual_id and one of the
     expected pages.

Recall@K = hits / number of questions.
MRR      = mean of 1/rank of the first correct chunk (0 if none in top 10).
Manual@5 = the correct MANUAL is in the top 5 (page can be wrong).

Run:
    python scripts/run_eval.py --name dense_baseline
The results are saved to eval/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from manualmate.index.db import get_connection
from manualmate.index.embedder import MODEL_NAME, embed_texts
from manualmate.retrieve.dense import search_by_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "eval" / "text_queries.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
KS = [1, 3, 5, 10]


def first_hit_rank(results: list[dict], query: dict) -> int | None:
    """Rank (1-based) of the first correct chunk, or None."""
    for rank, r in enumerate(results, start=1):
        if r["manual_id"] == query["manual_id"] and r["page"] in query["pages"]:
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="dense_baseline")
    args = parser.parse_args()

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    vectors = embed_texts([q["question"] for q in queries], show_progress=False)
    max_k = max(KS)

    per_query = []
    with get_connection() as conn:
        for query, vector in zip(queries, vectors):
            results = search_by_vector(conn, vector, k=max_k)
            rank = first_hit_rank(results, query)
            per_query.append({
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
    print(f"\nRun: {args.name}   model: {MODEL_NAME}   questions: {n}\n")
    for key, value in metrics.items():
        print(f"  {key:<10} {value:.2f}")
    print("\nRecall@5 by category:")
    for cat, value in category_recall.items():
        print(f"  {cat:<16} {value:.2f}")
    misses = [q for q in per_query if not q["rank"] or q["rank"] > 5]
    print(f"\nMisses at top 5 ({len(misses)}):")
    for q in misses:
        print(f"  {q['id']} expected {q['expected']:<16} got {q['top5'][0]}"
              f"   | {q['question'][:60]}")

    # --- Save ----------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = RESULTS_DIR / f"{stamp}_{args.name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": args.name,
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
