from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import HttpUrl, ValidationError

from ai_news_assistant.models import Article


def make_article(**overrides: Any) -> Article:
    fields: dict[str, Any] = {
        "id": "abc123",
        "url": "https://example.com/news/story",
        "title": "A headline",
        "source": "Example",
    }
    fields.update(overrides)
    return Article(**fields)


def test_valid_article_with_all_fields() -> None:
    published = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
    article = make_article(
        published_at=published,
        summary="A teaser.",
        text="Full text.",
        tags=["AI", "Regulation"],
    )

    assert isinstance(article.url, HttpUrl)
    assert str(article.url) == "https://example.com/news/story"
    assert article.title == "A headline"
    assert article.published_at == published
    assert article.summary == "A teaser."
    assert article.text == "Full text."
    assert article.tags == ["AI", "Regulation"]


def test_minimal_article_uses_defaults() -> None:
    first = make_article()
    second = make_article()

    assert first.published_at is None
    assert first.summary is None
    assert first.text is None
    assert first.tags == []
    assert first.tags is not second.tags


@pytest.mark.parametrize("url", ["not a url", "ftp://x.com/a", ""])
def test_bad_url_raises(url: str) -> None:
    with pytest.raises(ValidationError):
        make_article(url=url)


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        make_article(published_at=datetime(2026, 10, 6, 12, 0))


def test_aware_non_utc_datetime_keeps_offset() -> None:
    plus_two = timezone(timedelta(hours=2))
    article = make_article(published_at=datetime(2026, 10, 6, 12, 0, tzinfo=plus_two))

    assert article.published_at is not None
    assert article.published_at.utcoffset() == timedelta(hours=2)


def test_title_is_stripped() -> None:
    assert make_article(title="  Hello  ").title == "Hello"


@pytest.mark.parametrize("title", ["", "   ", "\n\t"])
def test_empty_title_rejected(title: str) -> None:
    with pytest.raises(ValidationError):
        make_article(title=title)
