from collections.abc import Callable

import httpx
import pytest

from ai_news_assistant.downloads import USER_AGENT, ResponseTooLargeError, download
from ai_news_assistant.ingestion.rss import FeedTooLargeError

URL = "https://example.com/page"


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def serve(response: httpx.Response) -> httpx.Client:
    return make_client(lambda request: response)


def test_returns_body_and_headers() -> None:
    client = serve(
        httpx.Response(200, content=b"<html></html>", headers={"Content-Type": "text/html"})
    )

    result = download(URL, client, max_bytes=100)

    assert result.content == b"<html></html>"
    assert result.headers["Content-Type"] == "text/html"


def test_sends_user_agent_and_accept() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    download(URL, make_client(handler), max_bytes=100, accept="text/html")

    assert requests[0].headers["User-Agent"] == USER_AGENT
    assert requests[0].headers["Accept"] == "text/html"


def test_without_accept_keeps_httpx_default() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    download(URL, make_client(handler), max_bytes=100)

    assert requests[0].headers["Accept"] == "*/*"


def test_over_limit_raises_default_error() -> None:
    client = serve(httpx.Response(200, content=b"x" * 101))

    with pytest.raises(ResponseTooLargeError, match=f"Response {URL} exceeds 100 bytes"):
        download(URL, client, max_bytes=100)


def test_over_limit_raises_given_error_class() -> None:
    class PartTooLargeError(ResponseTooLargeError):
        kind = "Part"

    client = serve(httpx.Response(200, content=b"x" * 101))

    with pytest.raises(PartTooLargeError, match=f"Part {URL} exceeds 100 bytes") as info:
        download(URL, client, max_bytes=100, too_large=PartTooLargeError)

    assert info.value.url == URL
    assert info.value.max_bytes == 100


def test_feed_too_large_is_a_response_too_large() -> None:
    assert issubclass(FeedTooLargeError, ResponseTooLargeError)
