from dataclasses import replace
from typing import Any

from ai_news_assistant.rag.eval.report import render_matrix, render_report, render_summary
from ai_news_assistant.rag.eval.scoring import PairResult, QuestionScore

META: dict[str, Any] = {
    "experiment": "baseline",
    "split": "dev",
    "created_at": "2026-10-22T12:00:00",
    "dataset_version": 1,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "config": {"max_articles": 50, "chunk_overfetch": 3, "hybrid": {"rrf_k": 60, "depth": 50}},
}

BASE = QuestionScore(
    question_id="q-1",
    type="specific_detail",
    gold_mode="any",
    scored=True,
    n_articles=40,
    recall_5=1.0,
    recall_10=1.0,
    mrr=1.0,
    first_gold_rank=1,
    evidence_5=1.0,
    stale_in_top_5=None,
    gold_above_stale=None,
    hard_negative_above_gold=None,
    gold_copies_in_top_5=None,
    top_articles=("g", "x", "y"),
    top_score=0.9,
    filter=None,
)
MISS = replace(
    BASE,
    question_id="q-2",
    n_articles=20,
    recall_5=0.0,
    recall_10=0.0,
    mrr=0.0,
    first_gold_rank=None,
    evidence_5=0.0,
    top_articles=("x", "y", "z"),
)
NO_ANSWER = replace(
    BASE,
    question_id="q-3",
    type="no_answer",
    scored=False,
    recall_5=None,
    recall_10=None,
    mrr=None,
    first_gold_rank=None,
    evidence_5=None,
    top_articles=("x",),
    top_score=0.42,
)
TITLES = {"x": "Title X (Verge)", "y": "Title | Y (Guardian)", "z": "Title Z (TechCrunch)"}
PAIR = PairResult("bm25", "cfg-a", (BASE, MISS, NO_ANSWER))


def test_summary_row() -> None:
    (header, _, row) = render_summary([(PAIR.label, PAIR)])

    assert header.startswith("| Pair | n scored | R@5")
    assert row == "| bm25 / cfg-a | 2 | 0.50 | 0.50 | 0.50 | 1/2 | 20 / 40 |"


def test_matrix_cells() -> None:
    other = PairResult("vector", "cfg-a", (replace(BASE, first_gold_rank=7), MISS, NO_ANSWER))

    lines = render_matrix([(PAIR.label, PAIR), (other.label, other)])

    assert lines[0] == "| Question | Type | bm25 / cfg-a | vector / cfg-a |"
    assert lines[2:] == [
        "| q-1 | specific_detail | 1 | 7 |",
        "| q-2 | specific_detail | - | - |",
        "| q-3 | no_answer | n/a | n/a |",
    ]


def test_report_header_and_counts() -> None:
    report = render_report(META, "20261022-120000_baseline", 1, [PAIR], TITLES)

    assert "# RAG eval: baseline" in report
    assert "- Split: dev; dataset v1\n" in report
    assert "n=3 (2 scored, 1 no_answer not scored)" in report
    assert "150 chunks asked per retriever" in report


def test_report_shows_other_scored_version() -> None:
    report = render_report(META, "run", 2, [PAIR], TITLES)

    assert "dataset v1 at run time, scored with v2" in report


def test_miss_lists_top_titles_and_hit_does_not() -> None:
    lines = render_report(META, "run", 1, [PAIR], TITLES).splitlines()

    # Question-table rows (the rank matrix rows have no R@5 column)
    q2 = next(line for line in lines if line.startswith("| q-2 | specific_detail | - | 0.00"))
    assert q2.endswith("| Title X (Verge); Title \\| Y (Guardian); Title Z (TechCrunch) |")
    q1 = next(line for line in lines if line.startswith("| q-1 | specific_detail | 1 | 1.00"))
    assert q1.endswith("|  |")


def test_no_answer_row_is_not_scored_with_top_score() -> None:
    report = render_report(META, "run", 1, [PAIR], TITLES)

    row = next(
        line for line in report.splitlines() if line.startswith("| q-3 | no_answer | n/a | not")
    )
    assert "Title X (Verge), score 0.420" in row


def test_notes_for_stale_hard_negative_copies_filter() -> None:
    special = replace(
        BASE,
        stale_in_top_5=1,
        gold_above_stale=False,
        hard_negative_above_gold=True,
        gold_copies_in_top_5=3,
        filter="source=The Decoder",
    )
    report = render_report(META, "run", 1, [PairResult("bm25", "cfg-a", (special,))], TITLES)

    assert (
        "stale@5=1, gold>stale=no, hard neg above gold, copies@5=3, filter: source=The Decoder"
        in report
    )


def test_report_ends_with_notes() -> None:
    report = render_report(META, "run", 1, [PAIR], TITLES)

    notes = report.split("## Notes")[1]
    assert notes.count("\n- ") == 5
    # PAIR has 2 scored questions, so one question is half of mean recall
    assert "With 2 scored questions, one question is 0.50 of mean recall" in notes
