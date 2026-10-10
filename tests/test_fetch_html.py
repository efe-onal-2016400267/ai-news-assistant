from collections.abc import Callable, Iterator

import httpx
import pytest

from ai_news_assistant.downloads import USER_AGENT, ResponseTooLargeError
from ai_news_assistant.extraction.fetch import (
    NotHTMLError,
    PageTooLargeError,
    fetch_html,
)

URL = "https://example.com/news/story"
PAGE = "<html><body><p>Story text.</p></body></html>"
HTML = {"Content-Type": "text/html; charset=utf-8"}


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


def test_returns_html_as_text() -> None:
    client = serve(httpx.Response(200, content=PAGE.encode(), headers=HTML))

    assert fetch_html(URL, client) == PAGE


def test_sends_user_agent_and_html_accept() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=PAGE.encode(), headers=HTML)

    fetch_html(URL, make_client(handler))

    assert requests[0].headers["User-Agent"] == USER_AGENT
    assert requests[0].headers["Accept"] == "text/html,application/xhtml+xml"


def test_follows_redirect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": URL})
        return httpx.Response(200, content=PAGE.encode(), headers=HTML)

    assert fetch_html("https://example.com/old", make_client(handler)) == PAGE


def test_content_length_over_limit_raises_without_reading_body() -> None:
    stream = TrackingStream([b"x" * 10])
    headers = {**HTML, "Content-Length": "101"}
    client = serve(httpx.Response(200, headers=headers, stream=stream))

    with pytest.raises(PageTooLargeError, match=f"Page {URL} exceeds 100 bytes"):
        fetch_html(URL, client, max_bytes=100)

    assert not stream.read


def test_streamed_body_over_limit_raises() -> None:
    stream = TrackingStream([b"x" * 60, b"x" * 60])
    client = serve(httpx.Response(200, headers=HTML, stream=stream))

    with pytest.raises(PageTooLargeError):
        fetch_html(URL, client, max_bytes=100)


def test_exactly_at_limit_returns_body() -> None:
    stream = TrackingStream([b"x" * 60, b"x" * 40])
    client = serve(httpx.Response(200, headers=HTML, stream=stream))

    assert fetch_html(URL, client, max_bytes=100) == "x" * 100


def test_page_too_large_is_a_response_too_large() -> None:
    assert issubclass(PageTooLargeError, ResponseTooLargeError)


@pytest.mark.parametrize("status", [403, 404, 500])
def test_error_status_raises(status: int) -> None:
    client = serve(httpx.Response(status, content=b"Blocked", headers=HTML))

    with pytest.raises(httpx.HTTPStatusError):
        fetch_html(URL, client)


def test_timeout_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(httpx.ReadTimeout):
        fetch_html(URL, make_client(handler))


@pytest.mark.parametrize(
    ("content_type", "reported"),
    [
        ("application/pdf", "application/pdf"),
        ("Application/JSON; charset=utf-8", "application/json"),
    ],
)
def test_non_html_content_type_raises(content_type: str, reported: str) -> None:
    client = serve(httpx.Response(200, content=b"%PDF-1.7", headers={"Content-Type": content_type}))

    with pytest.raises(NotHTMLError, match=f"is {reported}, not HTML") as info:
        fetch_html(URL, client)

    assert info.value.url == URL
    assert info.value.content_type == reported


def test_long_content_type_is_cut_in_error() -> None:
    content_type = "application/" + "x" * 500
    client = serve(httpx.Response(200, content=b"data", headers={"Content-Type": content_type}))

    with pytest.raises(NotHTMLError) as info:
        fetch_html(URL, client)

    assert len(info.value.content_type) == 100


@pytest.mark.parametrize(
    "content_type",
    [
        "text/html",
        "text/html; charset=utf-8",
        'TEXT/HTML; Charset="UTF-8"',
        "application/xhtml+xml",
    ],
)
def test_html_content_types_are_accepted(content_type: str) -> None:
    client = serve(
        httpx.Response(200, content=PAGE.encode(), headers={"Content-Type": content_type})
    )

    assert fetch_html(URL, client) == PAGE


def test_missing_content_type_is_accepted() -> None:
    response = httpx.Response(200, content=PAGE.encode())  # Bytes content sets no Content-Type
    assert "Content-Type" not in response.headers

    assert fetch_html(URL, serve(response)) == PAGE


def test_declared_charset_is_used() -> None:
    headers = {"Content-Type": "text/html; charset=iso-8859-1"}
    client = serve(httpx.Response(200, content="<p>café</p>".encode("latin-1"), headers=headers))

    assert fetch_html(URL, client) == "<p>café</p>"


def test_invalid_bytes_are_replaced() -> None:
    client = serve(httpx.Response(200, content=b"<p>caf\xe9</p>", headers=HTML))

    assert fetch_html(URL, client) == "<p>caf�</p>"


@pytest.mark.parametrize("charset", ["nonsense", "rot13"])
def test_unusable_charset_falls_back_to_utf8(charset: str) -> None:
    headers = {"Content-Type": f"text/html; charset={charset}"}
    client = serve(httpx.Response(200, content="<p>café</p>".encode(), headers=headers))

    assert fetch_html(URL, client) == "<p>café</p>"
