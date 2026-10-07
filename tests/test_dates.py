import time
from datetime import UTC, datetime, timedelta

import feedparser
import pytest

from ai_news_assistant.ingestion.dates import entry_published_at, struct_time_to_utc
from ai_news_assistant.models import Article


def rss_item(pub_date: str) -> bytes:
    return (
        '<rss version="2.0"><channel><item>'
        "<title>A headline</title><link>https://example.com/a</link>"
        f"<pubDate>{pub_date}</pubDate>"
        "</item></channel></rss>"
    ).encode()


def utc_struct(*fields: int) -> time.struct_time:
    return time.struct_time((*fields, 0, 1, 0))


def test_none_stays_none() -> None:
    assert struct_time_to_utc(None) is None


def test_struct_time_becomes_aware_utc() -> None:
    result = struct_time_to_utc(utc_struct(2026, 10, 5, 12, 30, 15))

    assert result == datetime(2026, 10, 5, 12, 30, 15, tzinfo=UTC)
    assert result is not None
    assert result.utcoffset() == timedelta(0)


def test_leap_second_is_capped() -> None:
    result = struct_time_to_utc(utc_struct(2026, 12, 31, 23, 59, 60))

    assert result == datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)


def test_plus_0200_pub_date_is_shifted_to_utc() -> None:
    entry = feedparser.parse(rss_item("Mon, 05 Oct 2026 14:30:00 +0200")).entries[0]

    assert entry_published_at(entry) == datetime(2026, 10, 5, 12, 30, tzinfo=UTC)


# Erroring on warnings guards against feedparser's updated_parsed -> published_parsed
# remapping, which warns when an entry has no updated date.
@pytest.mark.filterwarnings("error")
def test_unparseable_pub_date_gives_none() -> None:
    entry = feedparser.parse(rss_item("garbage")).entries[0]

    assert entry_published_at(entry) is None


def test_falls_back_to_updated() -> None:
    entry = {"updated_parsed": utc_struct(2026, 10, 5, 8, 0, 0)}

    assert entry_published_at(entry) == datetime(2026, 10, 5, 8, 0, tzinfo=UTC)


def test_published_wins_over_updated() -> None:
    entry = {
        "published_parsed": utc_struct(2026, 10, 5, 8, 0, 0),
        "updated_parsed": utc_struct(2026, 10, 6, 9, 0, 0),
    }

    assert entry_published_at(entry) == datetime(2026, 10, 5, 8, 0, tzinfo=UTC)


def test_no_dates_gives_none() -> None:
    assert entry_published_at({}) is None


def test_result_is_accepted_by_article() -> None:
    published = struct_time_to_utc(utc_struct(2026, 10, 5, 12, 30, 0))
    article = Article(
        id="abc",
        url="https://example.com/a",
        title="A headline",
        source="Example",
        published_at=published,
    )

    assert article.published_at == published
