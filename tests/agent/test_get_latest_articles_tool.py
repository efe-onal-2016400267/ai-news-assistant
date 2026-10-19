from pathlib import Path
from typing import Any

import pytest

from ai_news_assistant.agent.tools import MAX_DAYS, create_get_latest_articles_tool
from ai_news_assistant.rag.store import RagStore

# rag_db dates: ai-act 10-10 09:30 (newest, so "as of"), robotics 10-08 14:00,
# sourdough 09-30 07:00, iphone none.
AI_ACT = {
    "article_id": "ai-act",
    "title": "Title of ai-act",
    "source": "Guardian",
    "published_at": "2026-10-10T09:30:00+00:00",
    "date": "10 Oct 2026",
}
ROBOTICS = {
    "article_id": "robotics",
    "title": "Title of robotics",
    "source": "TechCrunch",
    "published_at": "2026-10-08T14:00:00+00:00",
    "date": "8 Oct 2026",
}


def latest(rag_db: Path, **args: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = create_get_latest_articles_tool(store_path=rag_db).invoke(args)
    return result


def ids(result: list[dict[str, Any]]) -> list[str]:
    return [a["article_id"] for a in result]


def test_two_days_hand_checked(rag_db: Path) -> None:
    # Cutoff 10-10 09:30 - 2 days = 10-08 09:30: ai-act and robotics, newest first.
    assert latest(rag_db, days=2) == [AI_ACT, ROBOTICS]


def test_default_is_seven_days(rag_db: Path) -> None:
    # Cutoff 10-03 09:30: sourdough (09-30) is too old.
    assert ids(latest(rag_db)) == ["ai-act", "robotics"]


def test_thirty_days_reaches_older_articles(rag_db: Path) -> None:
    # Cutoff 09-10 09:30. iphone has no date, so it never appears.
    assert ids(latest(rag_db, days=30)) == ["ai-act", "robotics", "sourdough"]


@pytest.mark.parametrize("days", [0, -3])
def test_days_below_one_clamped_to_one(rag_db: Path, days: int) -> None:
    # As days=1: cutoff 10-09 09:30, only ai-act.
    assert ids(latest(rag_db, days=days)) == ["ai-act"]


def test_days_above_max_clamped(rag_db: Path) -> None:
    assert MAX_DAYS == 30
    assert latest(rag_db, days=365) == latest(rag_db, days=MAX_DAYS)


def test_empty_corpus_gives_empty_list(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        store.replace_corpus([])

    assert latest(rag_db) == []


def test_corpus_without_dates_gives_empty_list(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        store.replace_corpus(["iphone"])

    assert latest(rag_db) == []


def test_listing_has_no_article_text(rag_db: Path) -> None:
    for item in latest(rag_db, days=30):
        assert set(item) == {"article_id", "title", "source", "published_at", "date"}


def test_tool_schema(rag_db: Path) -> None:
    tool = create_get_latest_articles_tool(store_path=rag_db)

    assert tool.name == "get_latest_articles"
    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert set(schema["properties"]) == {"days"}
    assert schema["properties"]["days"]["type"] == "integer"
    assert schema["properties"]["days"]["default"] == 7
    assert schema.get("required", []) == []
