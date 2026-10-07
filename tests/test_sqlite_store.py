from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ai_news_assistant.models import Article
from ai_news_assistant.storage.base import ArticleStore
from ai_news_assistant.storage.sqlite import SQLiteStore


def make_article(**overrides: Any) -> Article:
    fields: dict[str, Any] = {
        "id": "abc123",
        "url": "https://example.com/news/story",
        "title": "A headline",
        "source": "Example",
    }
    fields.update(overrides)
    return Article(**fields)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    with SQLiteStore(":memory:") as store:
        yield store


def test_store_satisfies_protocol(store: SQLiteStore) -> None:
    assert isinstance(store, ArticleStore)


def test_round_trip_with_all_fields(store: SQLiteStore) -> None:
    article = make_article(
        published_at=datetime(2026, 10, 6, 12, 0, 30, 123456, tzinfo=UTC),
        summary="A teaser.",
        text="Full text.",
        tags=["Regulation", "AI", "Europe"],
    )

    store.save([article])
    loaded = store.get(article.id)

    assert loaded == article
    assert loaded is not None
    assert loaded.tags == ["Regulation", "AI", "Europe"]
    assert loaded.published_at is not None
    assert loaded.published_at.tzinfo is not None


def test_round_trip_of_minimal_article(store: SQLiteStore) -> None:
    store.save([make_article()])
    loaded = store.get("abc123")

    assert loaded == make_article()
    assert loaded is not None
    assert loaded.published_at is None
    assert loaded.summary is None
    assert loaded.text is None
    assert loaded.tags == []


def test_non_utc_date_is_stored_as_same_instant_in_utc(store: SQLiteStore) -> None:
    plus_two = timezone(timedelta(hours=2))
    published = datetime(2026, 10, 6, 14, 0, tzinfo=plus_two)
    store.save([make_article(published_at=published)])

    loaded = store.get("abc123")

    assert loaded is not None
    assert loaded.published_at == published
    assert loaded.published_at is not None
    assert loaded.published_at.utcoffset() == timedelta(0)


def test_saving_same_article_twice_inserts_once(store: SQLiteStore) -> None:
    assert store.save([make_article()]) == 1
    assert store.save([make_article(title="Changed headline")]) == 0

    loaded = store.get("abc123")
    assert loaded is not None
    assert loaded.title == "A headline"


def test_mixed_batch_counts_only_new_articles(store: SQLiteStore) -> None:
    store.save([make_article(id="old")])

    assert store.save([make_article(id="old"), make_article(id="new")]) == 1
    assert store.get("new") is not None


def test_repeated_id_within_batch_counts_once(store: SQLiteStore) -> None:
    assert store.save([make_article(), make_article(title="Second copy")]) == 1


def test_save_empty_list_returns_zero(store: SQLiteStore) -> None:
    assert store.save([]) == 0


def test_save_accepts_generator(store: SQLiteStore) -> None:
    articles = (make_article(id=f"id-{n}") for n in range(3))

    assert store.save(articles) == 3


def test_get_unknown_id_returns_none(store: SQLiteStore) -> None:
    assert store.get("unknown") is None


def test_sql_in_article_text_is_stored_as_data(store: SQLiteStore) -> None:
    hostile = "'); DROP TABLE articles; --"
    article = make_article(id="x' OR '1'='1", title=hostile, summary=f'"{hostile}"', tags=[hostile])

    assert store.save([article]) == 1
    assert store.get("x' OR '1'='1") == article
    assert store.get("' OR '1'='1") is None
    assert store.save([make_article(id="after")]) == 1


def test_file_database_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    with SQLiteStore(path) as store:
        store.save([make_article()])

    with SQLiteStore(path) as reopened:
        assert reopened.get("abc123") == make_article()
        assert reopened.save([make_article()]) == 0
