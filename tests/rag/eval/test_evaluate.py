import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from ai_news_assistant.rag.base import Hit, Retriever, SearchFilter
from ai_news_assistant.rag.eval import evaluate
from ai_news_assistant.rag.eval.dataset import Evidence, Question, load_questions
from ai_news_assistant.rag.eval.run_config import RunConfig
from ai_news_assistant.rag.eval.run_retrieval import run
from ai_news_assistant.rag.eval.scoring import PairResult, score_question, summarize_by_type

SOURCES = ("Food Blog", "Guardian", "TechCrunch", "Verge")

QUESTIONS = """dataset_version = {version}
[[question]]
id = "q-1"
type = "specific_detail"
split = "dev"
question = "What does the AI Act require?"
gold_article_ids = ["ai-act"]
gold_mode = "any"
[[question.evidence]]
article_id = "ai-act"
quote = "AI Act"

[[question]]
id = "q-2"
type = "multi_article"
split = "dev"
question = "Robots and bread?"
gold_article_ids = ["robotics", "sourdough"]
gold_mode = "all"
[[question.evidence]]
article_id = "robotics"
quote = "robotics"
[[question.evidence]]
article_id = "sourdough"
quote = "sourdough"

[[question]]
id = "q-3"
type = "no_answer"
split = "dev"
question = "Who wrote Ulysses?"
gold_article_ids = []
gold_mode = "any"
"""


class FixedRetriever:
    """Returns the same hits for one question id's text, nothing for others."""

    def __init__(self, hits_by_question: dict[str, list[Hit]]) -> None:
        self.hits_by_question = hits_by_question

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        return self.hits_by_question.get(query, [])[:k]


def hits(*article_ids: str) -> list[Hit]:
    return [
        Hit(chunk_id=f"{a}-{i}", article_id=a, score=1.0 - i / 100, text=f"{a} text, AI Act")
        for i, a in enumerate(article_ids)
    ]


def question(
    gold: tuple[str, ...] = ("g",),
    mode: Any = "any",
    q_type: Any = "specific_detail",
    stale: tuple[str, ...] = (),
) -> Question:
    return Question(
        id="q",
        type=q_type,
        split="dev",
        question="?",
        gold_article_ids=gold,
        gold_mode=mode,
        stale_article_ids=stale,
        evidence=tuple(Evidence(g, f"{g} text") for g in gold),
    )


def make_run(
    tmp_path: Path,
    rag_db: Path,
    first_question_hits: list[Hit],
    created: datetime,
    version: int = 1,
) -> Path:
    questions_file = tmp_path / f"questions-v{version}.toml"
    questions_file.write_text(QUESTIONS.format(version=version), encoding="utf-8")
    dataset = load_questions(questions_file, split="dev")
    cfg = RunConfig(
        experiment="test",
        split="dev",
        dataset=questions_file,
        config_ids=("cfg-a",),
        retrievers=("bm25",),
        db=rag_db,
        out_dir=tmp_path / "results",
        max_articles=10,
    )
    retriever = FixedRetriever(
        {
            "What does the AI Act require?": first_question_hits,
            "Robots and bread?": hits("robotics", "iphone"),
            "Who wrote Ulysses?": hits("iphone"),
        }
    )
    retrievers: dict[str, dict[str, Retriever]] = {"cfg-a": {"bm25": retriever}}
    return run(cfg, dataset, retrievers, SOURCES, created=created)


# score_question


def test_any_gold_at_rank_two() -> None:
    score = score_question(question(), hits("x", "x", "g", "y"), max_articles=50)

    assert score.first_gold_rank == 2
    assert (score.recall_5, score.recall_10, score.mrr) == (1.0, 1.0, 0.5)
    assert score.evidence_5 == 1.0
    assert score.n_articles == 3


def test_max_articles_cuts_the_ranking() -> None:
    score = score_question(question(), hits("a", "b", "c", "g"), max_articles=3)

    assert score.first_gold_rank is None
    assert score.recall_10 == 0.0
    assert score.n_articles == 3
    assert score.evidence_5 == 0.0


def test_no_answer_is_not_scored_but_keeps_top_score() -> None:
    score = score_question(question(gold=(), q_type="no_answer"), hits("x"), max_articles=50)

    assert not score.scored
    assert (score.recall_5, score.recall_10, score.mrr) == (None, None, None)
    assert score.top_score == 1.0
    assert score.top_articles == ("x",)


def test_stale_fields_only_with_stale_ids() -> None:
    plain = score_question(question(), hits("g"), max_articles=50)
    with_stale = score_question(question(stale=("s",)), hits("s", "g"), max_articles=50)

    assert (plain.stale_in_top_5, plain.gold_above_stale) == (None, None)
    assert (with_stale.stale_in_top_5, with_stale.gold_above_stale) == (1, False)


def test_copies_only_for_any_with_several_gold() -> None:
    one = score_question(question(gold=("g",)), hits("g"), max_articles=50)
    all_mode = score_question(question(gold=("g", "h"), mode="all"), hits("g"), max_articles=50)
    copies = score_question(question(gold=("g", "h")), hits("g", "x", "h"), max_articles=50)

    assert one.gold_copies_in_top_5 is None
    assert all_mode.gold_copies_in_top_5 is None
    assert copies.gold_copies_in_top_5 == 2


# summarize_by_type


def test_summarize_by_type_means_over_scored_only() -> None:
    scores = (
        score_question(question(), hits("g"), max_articles=50),
        score_question(question(), hits("x"), max_articles=50),
        score_question(question(gold=(), q_type="no_answer"), hits("x"), max_articles=50),
    )

    by_type = {s.type: s for s in summarize_by_type(PairResult("bm25", "cfg", scores))}

    assert list(by_type) == ["specific_detail", "no_answer"]
    detail = by_type["specific_detail"]
    assert (detail.n, detail.n_scored, detail.recall_5, detail.found_at_max) == (2, 2, 0.5, 1)
    no_answer = by_type["no_answer"]
    assert (no_answer.n, no_answer.n_scored, no_answer.recall_5, no_answer.mrr) == (
        1,
        0,
        None,
        None,
    )


# load_run


def test_load_run_scores_the_run_folder(tmp_path: Path, rag_db: Path) -> None:
    run_dir = make_run(tmp_path, rag_db, hits("iphone", "ai-act"), datetime(2026, 10, 22, 12))

    meta, pairs = evaluate.load_run(run_dir, load_questions(tmp_path / "questions-v1.toml"))

    (pair,) = pairs
    assert (pair.retriever, pair.config_id) == ("bm25", "cfg-a")
    by_id = {q.question_id: q for q in pair.questions}
    assert by_id["q-1"].first_gold_rank == 2
    assert by_id["q-2"].recall_5 == 0.5
    assert not by_id["q-3"].scored
    assert meta["question_ids"] == ["q-1", "q-2", "q-3"]


def test_load_run_question_missing_from_labels_raises(tmp_path: Path, rag_db: Path) -> None:
    run_dir = make_run(tmp_path, rag_db, hits("ai-act"), datetime(2026, 10, 22, 12))
    fewer = tmp_path / "fewer.toml"
    fewer.write_text(QUESTIONS.format(version=1).split('[[question]]\nid = "q-3"')[0])

    with pytest.raises(ValueError, match="q-3"):
        evaluate.load_run(run_dir, load_questions(fewer))


def test_load_run_short_jsonl_raises(tmp_path: Path, rag_db: Path) -> None:
    run_dir = make_run(tmp_path, rag_db, hits("ai-act"), datetime(2026, 10, 22, 12))
    path = run_dir / "bm25__cfg-a.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")

    with pytest.raises(ValueError, match="do not match question_ids"):
        evaluate.load_run(run_dir, load_questions(tmp_path / "questions-v1.toml"))


# main


def test_main_writes_report_and_metrics(
    tmp_path: Path, rag_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = make_run(
        tmp_path, rag_db, hits("iphone", "robotics", "ai-act"), datetime(2026, 10, 22, 12)
    )

    evaluate.main([str(run_dir)])

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "n=3 (2 scored, 1 no_answer not scored)" in report
    assert "Title of iphone (Verge)" in report  # q-2 finds 1 of 2 gold: a miss, titles shown
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    (pair,) = metrics["pairs"]
    assert pair["retriever"] == "bm25"
    assert [q["question_id"] for q in pair["questions"]] == ["q-1", "q-2", "q-3"]
    assert pair["summary"]["n_scored"] == 2
    assert "bm25 / cfg-a" in capsys.readouterr().out


def test_main_with_other_dataset_version_says_so(tmp_path: Path, rag_db: Path) -> None:
    run_dir = make_run(tmp_path, rag_db, hits("ai-act"), datetime(2026, 10, 22, 12))
    newer = tmp_path / "questions-v2.toml"
    newer.write_text(QUESTIONS.format(version=2), encoding="utf-8")

    evaluate.main([str(run_dir), "--dataset", str(newer)])

    assert "v1 at run time, scored with v2" in (run_dir / "report.md").read_text()


def test_main_needs_run_dir_or_compare(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        evaluate.main([])


def test_compare_lists_recall_changes(
    tmp_path: Path, rag_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a = make_run(tmp_path, rag_db, hits("ai-act"), datetime(2026, 10, 22, 12))
    run_b = make_run(
        tmp_path,
        rag_db,
        hits("a1", "a2", "a3", "a4", "a5", "a6", "ai-act"),
        datetime(2026, 10, 22, 13),
    )

    evaluate.main(["--compare", str(run_a), str(run_b)])

    out = capsys.readouterr().out
    assert "- q-1 (specific_detail), bm25 / cfg-a: R@5 1.00 -> 0.00 (rank 1 -> 7)" in out
    assert "q-2" not in out.split("## R@5 changes")[1].split("## Notes")[0]
    assert not (run_a / "report.md").exists()
