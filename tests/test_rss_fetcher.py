from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ai_news_assistant.ingestion.base import Fetcher
from ai_news_assistant.ingestion.rss import USER_AGENT, RSSFetcher
from ai_news_assistant.models import Article

FIXTURES = Path(__file__).parent / "rss_fixtures"
GUARDIAN = (FIXTURES / "guardian_ai.xml").read_bytes()
FEED_URL = "https://example.com/feed.xml"


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def serve(content: bytes) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, content=content)


def test_fetch_returns_articles() -> None:
    articles = RSSFetcher(FEED_URL, client=make_client(serve(GUARDIAN))).fetch()

    assert articles
    assert all(isinstance(article, Article) for article in articles)
    assert all(article.source == "The Guardian" for article in articles)


def test_fetch_sends_user_agent_to_feed_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=GUARDIAN)

    RSSFetcher(FEED_URL, client=make_client(handler)).fetch()

    [request] = requests
    assert request.url == FEED_URL
    assert request.headers["User-Agent"] == USER_AGENT


def test_fetch_passes_timeout() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=GUARDIAN)

    RSSFetcher(FEED_URL, client=make_client(handler), timeout=3.0).fetch()

    assert requests[0].extensions["timeout"] == httpx.Timeout(3.0).as_dict()


def test_fetch_uses_configured_source_over_feed_title() -> None:
    articles = RSSFetcher(FEED_URL, "Configured", client=make_client(serve(GUARDIAN))).fetch()

    assert articles
    assert all(article.source == "Configured" for article in articles)


def test_fetch_uses_configured_source_without_feed_title() -> None:
    feed = (
        b'<rss version="2.0"><channel>'
        b"<item><title>A headline</title><link>https://example.com/a</link></item>"
        b"</channel></rss>"
    )

    articles = RSSFetcher(FEED_URL, "Configured", client=make_client(serve(feed))).fetch()

    assert [article.source for article in articles] == ["Configured"]


def test_fetch_raises_on_server_error() -> None:
    client = make_client(lambda request: httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        RSSFetcher(FEED_URL, client=client).fetch()


def test_fetch_propagates_network_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(httpx.ConnectError):
        RSSFetcher(FEED_URL, client=make_client(handler)).fetch()


# Like test_base.py: isinstance only checks that `fetch` exists; mypy checks the signature in src.


def test_rss_fetcher_satisfies_protocol() -> None:
    assert isinstance(RSSFetcher(FEED_URL), Fetcher)
