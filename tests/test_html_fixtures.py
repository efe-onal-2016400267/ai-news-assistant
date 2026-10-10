import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai_news_assistant.extraction.fetch import MAX_PAGE_BYTES, fetch_html
from ai_news_assistant.ingestion.feeds import FEEDS
from ai_news_assistant.ingestion.rss import parse_feed

FIXTURES = Path(__file__).parent / "text_html_fixtures"
RSS_FIXTURES = Path(__file__).parent / "rss_fixtures"
# Feeds whose pages our client can't fetch (see text_html_fixtures/README.md).
BLOCKED = {"openai-news"}
CAPTURED = sorted(FEEDS.keys() - BLOCKED)


def page_path(name: str) -> Path:
    return FIXTURES / f"{name.replace('-', '_')}.html"


def load_expected() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = json.loads(
        (FIXTURES / "expected.json").read_text(encoding="utf-8")
    )
    return expected


def test_blocked_feeds_are_known_and_have_no_page() -> None:
    assert BLOCKED <= FEEDS.keys()
    assert not any(page_path(name).exists() for name in BLOCKED)


@pytest.mark.parametrize("name", CAPTURED)
def test_page_fixture_comes_back_unchanged_through_fetch_html(name: str) -> None:
    path = page_path(name)
    assert path.stat().st_size <= MAX_PAGE_BYTES
    html = path.read_text(encoding="utf-8")
    response = httpx.Response(
        200, content=html.encode("utf-8"), headers={"Content-Type": "text/html; charset=utf-8"}
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda request: response))

    assert html.strip()
    assert fetch_html(load_expected()[name]["url"], client) == html


def test_expected_covers_exactly_the_captured_feeds() -> None:
    assert sorted(load_expected()) == CAPTURED


@pytest.mark.parametrize("name", CAPTURED)
def test_expected_entry_matches_first_feed_entry(name: str) -> None:
    entry = load_expected()[name]
    rss = (RSS_FIXTURES / f"{name.replace('-', '_')}.xml").read_bytes()

    assert entry["url"] == str(parse_feed(rss, FEEDS[name].source)[0].url)
    assert isinstance(entry["first"], str) and entry["first"].strip()
    assert isinstance(entry["last"], str) and entry["last"].strip()
