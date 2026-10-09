import sqlite3
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


def test_round_trip_of_extracted_article(store: SQLiteStore) -> None:
    plus_two = timezone(timedelta(hours=2))
    extracted = datetime(2026, 10, 9, 10, 30, tzinfo=plus_two)
    article = make_article(text="Full text.", extraction_successful=True, extracted_at=extracted)
    store.save([article])

    loaded = store.get("abc123")

    assert loaded == article
    assert loaded is not None
    assert loaded.extraction_successful is True
    assert loaded.extracted_at == extracted
    assert loaded.extracted_at is not None
    assert loaded.extracted_at.utcoffset() == timedelta(0)


def test_round_trip_of_failed_extraction(store: SQLiteStore) -> None:
    article = make_article(extraction_successful=False)
    store.save([article])

    loaded = store.get("abc123")

    assert loaded == article
    assert loaded is not None
    assert loaded.extraction_successful is False
    assert loaded.extracted_at is None


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


# The articles table as first released, before the extraction columns.
_ORIGINAL_SCHEMA = """
CREATE TABLE articles (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TEXT,
    summary TEXT,
    text TEXT,
    tags TEXT NOT NULL
)
"""


def test_old_database_is_migrated_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(_ORIGINAL_SCHEMA)
        conn.execute(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "abc123",
                "https://example.com/news/story",
                "A headline",
                "Example",
                "2026-10-06T12:00:00+00:00",
                "A teaser.",
                None,
                '["AI"]',
            ),
        )
    conn.close()

    with SQLiteStore(path) as store:
        loaded = store.get("abc123")
        extracted = make_article(
            id="new",
            text="Full text.",
            extraction_successful=True,
            extracted_at=datetime(2026, 10, 9, 8, 30, tzinfo=UTC),
        )
        assert store.save([extracted]) == 1
        assert store.get("new") == extracted

    assert loaded == make_article(
        published_at=datetime(2026, 10, 6, 12, 0, tzinfo=UTC),
        summary="A teaser.",
        tags=["AI"],
    )
    assert loaded is not None
    assert loaded.extraction_successful is None
    assert loaded.extracted_at is None

    conn = sqlite3.connect(path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(articles)")]
    conn.close()
    assert columns[-2:] == ["extraction_successful", "extracted_at"]


def test_migration_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(_ORIGINAL_SCHEMA)
    conn.close()

    SQLiteStore(path).close()
    with SQLiteStore(path) as reopened:
        assert reopened.save([make_article()]) == 1
        assert reopened.get("abc123") == make_article()


def test_get_unextracted_on_empty_store(store: SQLiteStore) -> None:
    assert store.get_unextracted() == []


def test_get_unextracted_returns_only_never_tried(store: SQLiteStore) -> None:
    pending = make_article(id="pending", summary="A teaser.")
    failed = make_article(id="failed", extraction_successful=False)
    done = make_article(
        id="done",
        text="Full text.",
        extraction_successful=True,
        extracted_at=datetime(2026, 10, 9, 8, 30, tzinfo=UTC),
    )
    store.save([pending, failed, done])

    assert store.get_unextracted() == [pending]


def _save_dated_articles(store: SQLiteStore) -> None:
    store.save(
        [
            make_article(id="middle", published_at=datetime(2026, 10, 7, tzinfo=UTC)),
            make_article(id="undated"),
            make_article(id="newest", published_at=datetime(2026, 10, 8, tzinfo=UTC)),
            make_article(id="oldest", published_at=datetime(2026, 10, 6, tzinfo=UTC)),
        ]
    )


def test_get_unextracted_newest_first_undated_last(store: SQLiteStore) -> None:
    _save_dated_articles(store)

    ids = [article.id for article in store.get_unextracted()]

    assert ids == ["newest", "middle", "oldest", "undated"]


def test_get_unextracted_orders_by_instant_not_offset(store: SQLiteStore) -> None:
    plus_five = timezone(timedelta(hours=5))
    # 10:00+05:00 is 05:00 UTC, so it is older than 06:00 UTC despite the larger wall time.
    store.save(
        [
            make_article(id="earlier", published_at=datetime(2026, 10, 6, 10, tzinfo=plus_five)),
            make_article(id="later", published_at=datetime(2026, 10, 6, 6, tzinfo=UTC)),
        ]
    )

    assert [article.id for article in store.get_unextracted()] == ["later", "earlier"]


def test_get_unextracted_limit(store: SQLiteStore) -> None:
    _save_dated_articles(store)

    assert [article.id for article in store.get_unextracted(limit=2)] == ["newest", "middle"]
    assert store.get_unextracted(limit=0) == []


def test_get_unextracted_rejects_negative_limit(store: SQLiteStore) -> None:
    with pytest.raises(ValueError, match="limit"):
        store.get_unextracted(limit=-1)


def test_migrated_rows_count_as_never_tried(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    conn = sqlite3.connect(path)
    with conn:
        conn.execute(_ORIGINAL_SCHEMA)
        conn.execute(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "abc123",
                "https://example.com/news/story",
                "A headline",
                "Example",
                None,
                None,
                None,
                "[]",
            ),
        )
    conn.close()

    with SQLiteStore(path) as store:
        assert store.get_unextracted() == [make_article()]
