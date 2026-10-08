from pathlib import Path
from urllib.parse import urlsplit

import pytest

from ai_news_assistant.ingestion.feeds import FEEDS

FIXTURES = Path(__file__).parent / "fixtures"


def test_sources_are_unique_and_non_empty() -> None:
    sources = [feed.source for feed in FEEDS.values()]

    assert all(source == source.strip() and source for source in sources)
    assert len(set(sources)) == len(sources)


@pytest.mark.parametrize("name", FEEDS)
def test_url_is_https(name: str) -> None:
    parts = urlsplit(FEEDS[name].url)

    assert parts.scheme == "https"
    assert parts.netloc


@pytest.mark.parametrize("name", FEEDS)
def test_every_feed_has_a_fixture(name: str) -> None:
    # scripts/capture_fixtures.py saves feed "a-b" as a_b.xml.
    assert (FIXTURES / f"{name.replace('-', '_')}.xml").is_file()
