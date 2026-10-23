"""Script 1 of the RAG eval: ask each question, save the raw chunk hits. No scoring.

Output: <out_dir>/<YYYYmmdd-HHMMSS>_<experiment>/run.json and one
<retriever>__<config_id>.jsonl per (retriever, chunking config), one line per question.
Scored by `evaluate.py`.
"""

import argparse
import json
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_news_assistant.rag.base import Retriever, SearchFilter
from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.config import EMBEDDING_MODEL
from ai_news_assistant.rag.eval.dataset import Dataset, Question, QuestionFilter, load_questions
from ai_news_assistant.rag.eval.run_config import RunConfig, load_run_config
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever
from ai_news_assistant.rag.store import RagStore
from ai_news_assistant.rag.vector_retriever import VectorRetriever

DEFAULT_RUN_CONFIG = Path("src/ai_news_assistant/rag/eval/rag_run.toml")


def build_retrievers(cfg: RunConfig, config_id: str) -> dict[str, Retriever]:
    """Build the retrievers named in `cfg` for one chunking config.

    The hybrid reuses the same vector and BM25 objects, so the model loads once.
    """
    names = set(cfg.retrievers)
    retrievers: dict[str, Retriever] = {}
    if names & {"vector", "hybrid"}:
        retrievers["vector"] = VectorRetriever(vector_store=cfg.vector_store, config_id=config_id)
    if names & {"bm25", "hybrid"}:
        retrievers["bm25"] = BM25Retriever(db=cfg.db, config_id=config_id)
    if "hybrid" in names:
        retrievers["hybrid"] = HybridRetriever(
            retrievers=[retrievers["vector"], retrievers["bm25"]],
            rrf_k=cfg.hybrid.rrf_k,
            depth=cfg.hybrid.depth,
        )
    return {name: retrievers[name] for name in cfg.retrievers}


def to_search_filter(question: Question, sources: Collection[str]) -> SearchFilter | None:
    """Turn a question's filter into a `SearchFilter`. `since` is UTC midnight, inclusive."""
    qf = question.filter
    if qf is None:
        return None
    if qf.source is not None and qf.source not in sources:
        raise ValueError(f"{question.id}: filter source {qf.source!r} is not in the corpus")
    return SearchFilter(
        sources=frozenset({qf.source}) if qf.source is not None else None,
        since=datetime(qf.since.year, qf.since.month, qf.since.day, tzinfo=UTC)
        if qf.since is not None
        else None,
    )


def run_question(
    retriever: Retriever, question: Question, k: int, search_filter: SearchFilter | None
) -> dict[str, Any]:
    """Search one question; the hits are saved as returned, best first."""
    start = time.perf_counter()
    hits = retriever.search(question.question, k=k, search_filter=search_filter)
    seconds = time.perf_counter() - start
    return {
        "question_id": question.id,
        "question": question.question,
        "filter": _filter_json(question.filter),
        "seconds": round(seconds, 4),
        "hits": [asdict(hit) for hit in hits],
    }


def run(
    cfg: RunConfig,
    dataset: Dataset,
    retrievers: Mapping[str, Mapping[str, Retriever]],
    sources: Collection[str],
    created: datetime | None = None,
) -> Path:
    """Ask every question with every (config id, retriever) pair; return the run folder.

    `retrievers` maps config id -> retriever name -> retriever. Filters are checked
    for all questions before the first search.
    """
    filters = {q.id: to_search_filter(q, sources) for q in dataset.questions}
    created = created or datetime.now()
    run_dir = cfg.out_dir / f"{created:%Y%m%d-%H%M%S}_{cfg.experiment}"
    run_dir.mkdir(parents=True, exist_ok=False)

    files: list[dict[str, str]] = []
    for config_id, by_name in retrievers.items():
        for name, retriever in by_name.items():
            rows = [
                run_question(retriever, q, k=cfg.chunk_k, search_filter=filters[q.id])
                for q in dataset.questions
            ]
            path = run_dir / f"{name}__{config_id}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            files.append({"retriever": name, "config_id": config_id, "path": path.name})
            total = sum(row["seconds"] for row in rows)
            print(f"{name} / {config_id}: {len(rows)} questions, {total:.1f} s")

    meta = {
        "experiment": cfg.experiment,
        "split": cfg.split,
        "created_at": created.isoformat(timespec="seconds"),
        "dataset_version": dataset.version,
        "question_ids": [q.id for q in dataset.questions],
        "embedding_model": EMBEDDING_MODEL,
        "config": asdict(cfg),
        "files": files,
    }
    (run_dir / "run.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return run_dir


def _filter_json(qf: QuestionFilter | None) -> dict[str, str | None] | None:
    if qf is None:
        return None
    return {"source": qf.source, "since": qf.since.isoformat() if qf.since else None}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG eval questions, save raw hits.")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_run_config(args.config)
    dataset = load_questions(cfg.dataset, split=cfg.split)
    with RagStore(cfg.db) as store:
        available = store.config_ids()
        sources = store.sources()
    missing = [c for c in cfg.config_ids if c not in available]
    if missing:
        parser.error(
            f"unknown config id(s) {', '.join(missing)}; available: {', '.join(available)}"
        )

    # Build everything before the first search, so a missing collection fails fast
    retrievers = {config_id: build_retrievers(cfg, config_id) for config_id in cfg.config_ids}
    run_dir = run(cfg, dataset, retrievers, sources)
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
