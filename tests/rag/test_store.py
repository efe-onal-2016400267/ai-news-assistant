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


def article(
    article_id: str, published_at: datetime | None, extracted: bool = True, source: str = "Src"
) -> Article:
    return Article(
        id=article_id,
        url=f"https://example.com/{article_id}",
        title=f"Title of {article_id}",
        source=source,
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


# sources


@pytest.fixture
def sources_db(tmp_path: Path) -> Path:
    """Sources saved out of order, a duplicate, an undated article, and two that must not count."""
    db = tmp_path / "sources.db"
    day = datetime(2026, 10, 5, tzinfo=UTC)
    with SQLiteStore(db) as store:
        store.save(
            [
                article("w1", day, source="Wired"),
                article("w2", day, source="Wired"),
                article("ars", day, source="Ars Technica"),
                article("undated", None, source="Verge"),
                article("outside", day, source="Outsider"),
                article("failed", day, extracted=False, source="Failed Source"),
            ]
        )
    with RagStore(db) as store:
        store.replace_corpus(["w1", "w2", "ars", "undated", "failed"])
    return db


def test_sources_of_rag_db(rag_db: Path) -> None:
    # CORPUS sources: Guardian, Verge (undated), TechCrunch, Food Blog -> sorted.
    with RagStore(rag_db) as store:
        assert store.sources() == ["Food Blog", "Guardian", "TechCrunch", "Verge"]


def test_sources_are_distinct_and_sorted(sources_db: Path) -> None:
    # Saved as Wired, Wired, Ars Technica, Verge; Wired appears once.
    with RagStore(sources_db) as store:
        assert store.sources() == ["Ars Technica", "Verge", "Wired"]


def test_sources_skip_article_outside_corpus(sources_db: Path) -> None:
    with RagStore(sources_db) as store:
        assert "Outsider" not in store.sources()


def test_sources_skip_failed_extraction(sources_db: Path) -> None:
    # "failed" is in the corpus table but extraction_successful = 0.
    with RagStore(sources_db) as store:
        assert "Failed Source" not in store.sources()


def test_sources_empty_corpus_gives_empty(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        store.replace_corpus([])
        assert store.sources() == []


# date_cutoff


def test_date_cutoff_counts_back_from_newest_corpus_date(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        cutoff = store.date_cutoff(days=3)

    # Newest is ai-act 2026-10-10 09:30 UTC; minus 3 days = 2026-10-07 09:30 UTC.
    assert cutoff == datetime(2026, 10, 7, 9, 30, tzinfo=UTC)
    assert cutoff is not None and cutoff.tzinfo is not None


def test_date_cutoff_zero_days_is_newest_date(rag_db: Path) -> None:
    with RagStore(rag_db) as store:
        assert store.date_cutoff(days=0) == datetime(2026, 10, 10, 9, 30, tzinfo=UTC)


def test_date_cutoff_ignores_outside_and_failed(small_db: Path) -> None:
    # Newest corpus date is 10-05 12:00; "outside" and "failed" are later but do not count.
    with RagStore(small_db) as store:
        assert store.date_cutoff(days=1) == datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize("corpus", [[], ["iphone"]])
def test_date_cutoff_none_without_dates(rag_db: Path, corpus: list[str]) -> None:
    with RagStore(rag_db) as store:
        store.replace_corpus(corpus)
        assert store.date_cutoff(days=3) is None
