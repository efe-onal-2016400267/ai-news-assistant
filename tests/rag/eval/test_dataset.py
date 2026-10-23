import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ai_news_assistant.rag.eval import dataset
from ai_news_assistant.rag.eval.dataset import QuestionFilter, load_questions

REAL_FILE = Path(dataset.__file__).with_name("rag_questions_v1.toml")


def question(
    qid: str = "q-1", split: str = "dev", gold: tuple[str, ...] = ("a1",)
) -> dict[str, Any]:
    """A minimal valid question as a TOML table."""
    return {
        "id": qid,
        "type": "specific_detail",
        "split": split,
        "question": "What happened?",
        "gold_article_ids": list(gold),
        "gold_mode": "any",
        "evidence": [{"article_id": g, "quote": f"quote from {g}"} for g in gold],
    }


def write_toml(path: Path, questions: list[dict[str, Any]], version: int | None = 1) -> Path:
    """Write questions as TOML. JSON strings and string lists are valid TOML values."""
    lines = [] if version is None else [f"dataset_version = {version}"]
    for q in questions:
        lines.append("[[question]]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in q.items() if k not in ("evidence", "filter")]
        if "filter" in q:
            lines.append("[question.filter]")
            lines += [f"{k} = {json.dumps(v)}" for k, v in q["filter"].items()]
        for e in q.get("evidence", []):
            lines.append("[[question.evidence]]")
            lines += [f"{k} = {json.dumps(v)}" for k, v in e.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_real_file_loads() -> None:
    data = load_questions(REAL_FILE)
    by_id = {q.id: q for q in data.questions}

    assert data.version == 1
    assert len(data.questions) == 22
    assert by_id["rag-019"].filter == QuestionFilter(source="The Decoder")
    assert by_id["rag-020"].filter == QuestionFilter(since=date(2026, 10, 1))
    assert by_id["rag-021"].gold_article_ids == ()
    assert by_id["rag-022"].gold_article_ids == ()


@pytest.mark.parametrize(("split", "count"), [("dev", 13), ("test", 9)])
def test_real_file_split(split: Any, count: int) -> None:
    questions = load_questions(REAL_FILE, split=split).questions

    assert len(questions) == count
    assert {q.split for q in questions} == {split}


def test_legacy_type_is_dropped(tmp_path: Path) -> None:
    q = question() | {"legacy_type": "Natural Language"}

    loaded = load_questions(write_toml(tmp_path / "q.toml", [q])).questions[0]

    assert loaded.id == "q-1"
    assert not hasattr(loaded, "legacy_type")


def test_bare_toml_date_is_accepted(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "q.toml", [question()])
    path.write_text(path.read_text() + "[question.filter]\nsince = 2026-10-01\n")

    assert load_questions(path).questions[0].filter == QuestionFilter(since=date(2026, 10, 1))


def test_missing_dataset_version(tmp_path: Path) -> None:
    path = write_toml(tmp_path / "q.toml", [question()], version=None)

    with pytest.raises(ValueError, match="dataset_version"):
        load_questions(path)


Mutation = Callable[[list[dict[str, Any]]], None]


def set_key(key: str, value: Any) -> Mutation:
    def mutate(questions: list[dict[str, Any]]) -> None:
        questions[0][key] = value

    return mutate


def append(q: dict[str, Any]) -> Mutation:
    def mutate(questions: list[dict[str, Any]]) -> None:
        questions.append(q)

    return mutate


def no_answer_with_gold(questions: list[dict[str, Any]]) -> None:
    questions[0]["type"] = "no_answer"


def answerable_without_gold(questions: list[dict[str, Any]]) -> None:
    questions[0]["gold_article_ids"] = []
    questions[0]["evidence"] = []


def evidence_not_gold(questions: list[dict[str, Any]]) -> None:
    questions[0]["evidence"].append({"article_id": "a9", "quote": "elsewhere"})


@pytest.mark.parametrize(
    ("mutate", "bad_id"),
    [
        pytest.param(set_key("gold_articles", ["a1"]), "q-1", id="unknown-key"),
        pytest.param(set_key("type", "trivia"), "q-1", id="bad-type"),
        pytest.param(set_key("split", "train"), "q-1", id="bad-split"),
        pytest.param(set_key("gold_mode", "some"), "q-1", id="bad-gold-mode"),
        pytest.param(append(question("q-1", gold=("a2",))), "q-1", id="duplicate-id"),
        pytest.param(set_key("gold_article_ids", ["a1", "a1"]), "q-1", id="repeated-gold"),
        pytest.param(no_answer_with_gold, "q-1", id="no-answer-with-gold"),
        pytest.param(answerable_without_gold, "q-1", id="answerable-without-gold"),
        pytest.param(evidence_not_gold, "q-1", id="evidence-not-gold"),
        pytest.param(set_key("evidence", []), "q-1", id="gold-without-evidence"),
        pytest.param(set_key("stale_article_ids", ["a1"]), "q-1", id="gold-also-stale"),
        pytest.param(
            set_key("hard_negative_article_ids", ["a1"]), "q-1", id="gold-also-hard-negative"
        ),
        pytest.param(set_key("filter", {"until": "2026-01-01"}), "q-1", id="unknown-filter-key"),
        pytest.param(set_key("filter", {}), "q-1", id="empty-filter"),
        pytest.param(set_key("filter", {"since": "October 1"}), "q-1", id="bad-since"),
        pytest.param(append(question("q-2", split="test")), "q-2", id="article-in-both-splits"),
    ],
)
def test_invalid_file_names_the_question(tmp_path: Path, mutate: Mutation, bad_id: str) -> None:
    questions = [question()]
    mutate(questions)
    path = write_toml(tmp_path / "q.toml", questions)

    with pytest.raises(ValueError, match=bad_id):
        load_questions(path)
