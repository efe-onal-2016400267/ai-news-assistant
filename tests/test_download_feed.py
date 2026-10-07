from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from ai_news_assistant.ingestion.rss import FeedTooLargeError, download_feed

FIXTURES = Path(__file__).parent / "fixtures"
GUARDIAN = (FIXTURES / "guardian_ai.xml").read_bytes()
FEED_URL = "https://example.com/feed.xml"


class TrackingStream(httpx.SyncByteStream):
    """Serves chunks and records whether the body was read. Sends no Content-Length."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.read = False

    def __iter__(self) -> Iterator[bytes]:
        self.read = True
        yield from self.chunks


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def serve(response: httpx.Response) -> httpx.Client:
    return make_client(lambda request: response)


def test_under_limit_returns_body() -> None:
    client = serve(httpx.Response(200, content=GUARDIAN))

    assert download_feed(FEED_URL, client) == GUARDIAN


def test_exactly_at_limit_returns_body() -> None:
    body = b"x" * 100
    client = serve(httpx.Response(200, stream=TrackingStream([body[:60], body[60:]])))

    assert download_feed(FEED_URL, client, max_bytes=100) == body


def test_streamed_body_over_limit_raises() -> None:
    client = serve(httpx.Response(200, stream=TrackingStream([b"x" * 60, b"x" * 60])))

    with pytest.raises(FeedTooLargeError, match="exceeds 100 bytes"):
        download_feed(FEED_URL, client, max_bytes=100)


def test_content_length_over_limit_raises_without_reading_body() -> None:
    stream = TrackingStream([b"x" * 10])
    client = serve(httpx.Response(200, headers={"Content-Length": "101"}, stream=stream))

    with pytest.raises(FeedTooLargeError):
        download_feed(FEED_URL, client, max_bytes=100)

    assert not stream.read


def test_lying_content_length_still_raises() -> None:
    stream = TrackingStream([b"x" * 60, b"x" * 60])
    client = serve(httpx.Response(200, headers={"Content-Length": "10"}, stream=stream))

    with pytest.raises(FeedTooLargeError):
        download_feed(FEED_URL, client, max_bytes=100)


def test_invalid_content_length_falls_back_to_byte_count() -> None:
    stream = TrackingStream([b"x" * 50])
    client = serve(httpx.Response(200, headers={"Content-Length": "abc"}, stream=stream))

    assert download_feed(FEED_URL, client, max_bytes=100) == b"x" * 50


def test_server_error_raises_status_error_before_size_check() -> None:
    client = serve(httpx.Response(500, content=b"x" * 1000))

    with pytest.raises(httpx.HTTPStatusError):
        download_feed(FEED_URL, client, max_bytes=100)
