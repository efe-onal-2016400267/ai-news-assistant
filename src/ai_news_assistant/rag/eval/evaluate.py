"""Script 2 of the RAG eval: score a run folder from `run_retrieval.py`. Never searches.

`evaluate RUN_DIR` writes report.md and metrics.json into the run folder.
`evaluate --compare RUN_A RUN_B` scores both with the same labels and prints a comparison.
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_news_assistant.rag.base import Hit
from ai_news_assistant.rag.eval.dataset import Dataset, load_questions
from ai_news_assistant.rag.eval.report import render_compare, render_report, render_summary
from ai_news_assistant.rag.eval.scoring import (
    PairResult,
    score_question,
    summarize,
    summarize_by_type,
)
from ai_news_assistant.rag.store import RagStore


def read_meta(run_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return meta


def load_run(run_dir: Path, dataset: Dataset) -> tuple[dict[str, Any], list[PairResult]]:
    """Score every file of a run folder against `dataset`'s labels."""
    meta = read_meta(run_dir)
    by_id = {q.id: q for q in dataset.questions}
    question_ids: list[str] = meta["question_ids"]
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise ValueError(f"{run_dir.name}: question(s) not in the labels: {', '.join(missing)}")

    max_articles: int = meta["config"]["max_articles"]
    pairs: list[PairResult] = []
    for entry in meta["files"]:
        path = run_dir / entry["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if [row["question_id"] for row in rows] != question_ids:
            raise ValueError(f"{path.name}: lines do not match question_ids in run.json")
        scores = tuple(
            score_question(
                by_id[row["question_id"]], [Hit(**hit) for hit in row["hits"]], max_articles
            )
            for row in rows
        )
        pairs.append(
            PairResult(retriever=entry["retriever"], config_id=entry["config_id"], questions=scores)
        )
    return meta, pairs


def article_titles(db: Path, pairs: Iterable[PairResult]) -> dict[str, str]:
    """'Title (Source)' for every article shown in a report."""
    ids = {a for pair in pairs for q in pair.questions for a in q.top_articles}
    titles: dict[str, str] = {}
    with RagStore(db) as store:
        for article_id in sorted(ids):
            try:
                article = store.get_article(article_id)
            except KeyError:
                continue
            titles[article_id] = f"{article.title} ({article.source})"
    return titles


def metrics_json(meta: dict[str, Any], scored_version: int, pairs: Sequence[PairResult]) -> str:
    payload = {
        "experiment": meta["experiment"],
        "split": meta["split"],
        "dataset_version_run": meta["dataset_version"],
        "dataset_version_scored": scored_version,
        "pairs": [
            {
                "retriever": pair.retriever,
                "config_id": pair.config_id,
                "summary": asdict(summarize(pair.questions)),
                "by_type": [asdict(s) for s in summarize_by_type(pair)],
                "questions": [asdict(q) for q in pair.questions],
            }
            for pair in pairs
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a RAG eval run folder.")
    parser.add_argument("run_dir", type=Path, nargs="?", help="run folder to report on")
    parser.add_argument(
        "--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"), help="compare two runs"
    )
    parser.add_argument("--dataset", type=Path, help="labels to score with (default: the run's)")
    args = parser.parse_args(argv)
    if (args.run_dir is None) == (args.compare is None):
        parser.error("give either RUN_DIR or --compare RUN_A RUN_B")

    if args.compare is not None:
        run_a, run_b = args.compare
        meta_a = read_meta(run_a)
        dataset = load_questions(args.dataset or Path(meta_a["config"]["dataset"]))
        _, pairs_a = load_run(run_a, dataset)
        _, pairs_b = load_run(run_b, dataset)
        print(render_compare(run_a.name, pairs_a, run_b.name, pairs_b, dataset.version), end="")
        return

    run_dir: Path = args.run_dir
    meta = read_meta(run_dir)
    dataset = load_questions(args.dataset or Path(meta["config"]["dataset"]))
    meta, pairs = load_run(run_dir, dataset)
    titles = article_titles(Path(meta["config"]["db"]), pairs)

    report = render_report(meta, run_dir.name, dataset.version, pairs, titles)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        metrics_json(meta, dataset.version, pairs), encoding="utf-8"
    )
    print("\n".join(render_summary([(p.label, p) for p in pairs])))
    print(f"Wrote {run_dir / 'report.md'} and metrics.json")


if __name__ == "__main__":
    main()
