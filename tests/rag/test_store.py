from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_news_assistant.models import Article
from ai_news_assistant.rag.store import ArticleListing, RagStore
from ai_news_assistant.storage.sqlite import SQLiteStore

# rag_db dates: ai-act 10-10 09:30, robotics 10-08 14:00, sourdough 09-30 07:00, iphone none.
ROBOTICS_TIME = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)


def ids(listings: list[ArticleListing]) -> list[str]:
    return [a.id for a in listings]


def article(article_id: str, published_at: datetime, extracted: bool = True) -> Article:
    return Article(
        id=article_id,
        url=f"https://example.com/{article_id}",
        title=f"Title of {article_id}",
        source="Src",
        published_at=published_at,
        text="Body." if extracted else None,
        extraction_successful=extracted,
        extracted_at=datetime(2026, 10, 11, tzinfo=UTC) if extracted else None,
    )


@pytest.fixture
def small_db(tmp_path: Path) -> Path:
    """Ties, an article outside the corpus, and a corpus article whose extraction failed."""
    db = tmp_path / "small.db"
    same = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
    with SQLiteStore(db) as store:
        store.save(
            [
                article("b-tie", same),
                article("a-tie", same),
                article("outside", same + timedelta(hours=1)),
                article("failed", same + timedelta(hours=2), extracted=False),
            ]
        )
    with RagStore(db) as store:
        store.replace_corpus(["b-tie", "a-tie", "failed"])
    return db


# newest_publish_date


def test_newest_publish_date_is_newest_corpus_date(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        assert store.newest_publish_date() == "2026-10-10T09:30:00+00:00"


def test_newest_publish_date_none_for_empty_corpus(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        store.replace_corpus([])
        assert store.newest_publish_date() is None


def test_newest_publish_date_none_when_no_article_has_a_date(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        store.replace_corpus(["iphone"])
        assert store.newest_publish_date() is None


def test_newest_publish_date_ignores_outside_and_failed(small_db: Path) -> None:
    # "outside" (13:00) is not in the corpus, "failed" (14:00) failed extraction.
    with RagStore(small_db) as store:
        assert store.newest_publish_date() == "2026-10-05T12:00:00+00:00"


# get_articles_since


def test_articles_since_newest_first_exact(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        result = store.get_articles_since(datetime(2026, 10, 1, tzinfo=UTC))

    assert result == [
        ArticleListing(
            id="ai-act",
            title="Title of ai-act",
            source="Guardian",
            published_at="2026-10-10T09:30:00+00:00",
        ),
        ArticleListing(
            id="robotics",
            title="Title of robotics",
            source="TechCrunch",
            published_at="2026-10-08T14:00:00+00:00",
        ),
    ]


def test_articles_since_includes_exact_cutoff(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        assert ids(store.get_articles_since(ROBOTICS_TIME)) == ["ai-act", "robotics"]
        later = ROBOTICS_TIME + timedelta(seconds=1)
        assert ids(store.get_articles_since(later)) == ["ai-act"]


def test_articles_since_converts_other_timezones_to_utc(rag_db: Path) -> None:
    # 16:00 at +02:00 is 14:00 UTC, the robotics time.
    since = datetime(2026, 10, 8, 16, 0, tzinfo=timezone(timedelta(hours=2)))

    with RagStore(rag_db) as store:
        assert ids(store.get_articles_since(since)) == ["ai-act", "robotics"]


def test_articles_since_naive_datetime_raises(rag_db: Path) -> None:
    with RagStore(rag_db) as store, pytest.raises(ValueError):
        store.get_articles_since(datetime(2026, 10, 1))


def test_articles_since_skips_undated_articles(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        result = ids(store.get_articles_since(datetime(2000, 1, 1, tzinfo=UTC)))

    assert result == ["ai-act", "robotics", "sourdough"]  # iphone has no date


def test_articles_since_limit_applies_after_ordering(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        assert ids(store.get_articles_since(datetime(2000, 1, 1, tzinfo=UTC), limit=1)) == [
            "ai-act"
        ]


def test_articles_since_future_cutoff_gives_empty(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        assert store.get_articles_since(datetime(2030, 1, 1, tzinfo=UTC)) == []


def test_articles_since_ties_ordered_by_id(small_db: Path) -> None:
    with RagStore(small_db) as store:
        result = ids(store.get_articles_since(datetime(2026, 10, 1, tzinfo=UTC)))

    # Same published_at: a-tie before b-tie. "outside" and "failed" are excluded.
    assert result == ["a-tie", "b-tie"]
