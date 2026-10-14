import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from ai_news_assistant.rag.base import Retriever
from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.config import DEFAULT_DB, DEFAULT_VECTOR_STORE, EVAL_RESULT_DIR
from ai_news_assistant.rag.eval.question_list import QueryConfs, query_confs
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever
from ai_news_assistant.rag.vector_retriever import VectorRetriever

RECALL_THS = (5, 10)  # We will report Recall@5, Recall@10


def evaluate_question(
    retriever: Retriever, query: str, gold: list[str], ks: tuple[int, ...] = RECALL_THS
) -> dict[str, Any]:
    """Evaluate a single question and report recalls for it."""
    results = retriever.search(query, k=max(ks))
    ranked = [hit.article_id for hit in results]
    gold_labels = set(gold)

    metrics: dict[str, dict[str, Any]] = {}
    for k in ks:
        top = set(ranked[:k])
        found = top & gold_labels
        metrics[str(k)] = {
            "gold_hits": len(found),
            "fp": len(top - gold_labels),
            "fn": len(gold_labels - top),
            "recall": len(found) / len(gold_labels),
        }

    return {
        "query": query,
        "n_gold": len(gold_labels),
        "metrics": metrics,
        "missed_at_max_k": sorted(gold_labels - set(ranked)),
        "retrieved": [
            {
                "article_id": hit.article_id,
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "gold": hit.article_id in gold_labels,
            }
            for hit in results
        ],
    }


def run_eval(retriever: Retriever, confs: QueryConfs) -> list[dict[str, Any]]:
    """Run the evaluation for each configuration"""
    rows: list[dict[str, Any]] = []
    for group in confs["queries"]:
        q_type = group["QueryType"]
        qs = group["Queries"]
        golds = group["GoldLabels"]

        if len(qs) != len(golds):
            raise ValueError(f"{group['QueryType']}: {len(qs)} queries but {len(golds)} gold lists")

        for q, gold in zip(qs, golds, strict=True):
            row = evaluate_question(retriever=retriever, query=q, gold=gold)
            row["query_type"] = q_type
            rows.append(row)
    return rows


def write_results(
    rows: list[dict[str, Any]], out_dir: Path, retriever_name: str, config_id: str
) -> Path:
    """Write the evaluation results to a file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now()
    path = out_dir / f"{created:%Y%m%d-%H%M%S}_{retriever_name}_{config_id}.json"
    payload = {
        "meta": {
            "retriever": retriever_name,
            "config_id": config_id,
            "ks": list(RECALL_THS),
            "created_at": created.isoformat(timespec="seconds"),
            "n_questions": len(rows),
        },
        "results": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_summary(rows: list[dict[str, Any]], config_id: str) -> None:
    print(f"----- Configuration: {config_id} -----")
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["query_type"], []).append(row)
    for query_type, group in by_type.items():
        print(f"{query_type}")
        for row in group:
            cells = []
            for k in RECALL_THS:
                recall = row["metrics"][str(k)]["recall"]
                cells.append(f"R@{k}={'n/a' if recall is None else f'{recall:.2f}'}")
            print(f"  {'  '.join(cells)}  {row['query']}")
    for k in RECALL_THS:
        values = [r["metrics"][str(k)]["recall"] for r in rows]
        scored = [v for v in values if v is not None]
        if scored:
            print(f"Mean recall@{k}: {mean(scored):.3f} over {len(scored)} questions")


def main() -> None:
    for config_id in query_confs["config_ids"]:
        retrievers: dict[str, Retriever] = {
            "vector": VectorRetriever(vector_store=DEFAULT_VECTOR_STORE, config_id=config_id),
            "bm25": BM25Retriever(db=DEFAULT_DB, config_id=config_id),
            f"hybrid_{config_id}": HybridRetriever(
                retrievers=[
                    VectorRetriever(vector_store=DEFAULT_VECTOR_STORE, config_id=config_id),
                    BM25Retriever(db=DEFAULT_DB, config_id=config_id),
                ]
            ),
        }
        for name, retriever in retrievers.items():
            rows = run_eval(retriever, query_confs)
            print_summary(rows, f"{name} / {config_id}")
            path = write_results(rows, EVAL_RESULT_DIR, retriever_name=name, config_id=config_id)
            print(f"Wrote {path}\n")


if __name__ == "__main__":
    main()
