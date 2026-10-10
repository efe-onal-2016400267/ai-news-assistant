from email.message import Message

import httpx

from ai_news_assistant.downloads import DEFAULT_TIMEOUT, ResponseTooLargeError, download

MAX_PAGE_BYTES = 5 * 1024 * 1024  # Revisit after measuring real pages in the HTML fixtures
HTML_TYPES = ("text/html", "application/xhtml+xml")
MAX_REPORTED_TYPE_LENGTH = 100  # The Content-Type header is untrusted input


class PageTooLargeError(ResponseTooLargeError):
    """The article page is larger than the download limit."""

    kind = "Page"


class NotHTMLError(Exception):
    """The response declares a content type that is not HTML, such as a PDF."""

    def __init__(self, url: str, content_type: str) -> None:
        content_type = content_type[:MAX_REPORTED_TYPE_LENGTH]
        super().__init__(f"Page {url} is {content_type}, not HTML")
        self.url = url
        self.content_type = content_type


def fetch_html(
    url: str,
    client: httpx.Client | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_PAGE_BYTES,
) -> str:
    """Download an article page and return its HTML as text.

    Nothing is caught or retried, so the caller decides which failures are permanent:
    - httpx.HTTPStatusError for 4xx and 5xx responses (paywalls and bot blocks included)
    - httpx.TransportError for timeouts, connection errors and too many redirects
    - PageTooLargeError when the body is larger than `max_bytes`
    - NotHTMLError when Content-Type is present and not HTML; a missing one is accepted

    The body is decoded with the charset from Content-Type, else UTF-8, and invalid bytes are
    replaced. Pages that declare their charset only in a <meta> tag are decoded as UTF-8.
    The content type is checked after the capped body has been read.
    """
    page = download(
        url,
        client,
        max_bytes=max_bytes,
        timeout=timeout,
        accept=",".join(HTML_TYPES),
        too_large=PageTooLargeError,
    )
    header = page.headers.get("Content-Type", "").strip()
    if not header:
        return page.content.decode("utf-8", errors="replace")

    media_type = header.split(";", 1)[0].strip().lower()
    if media_type not in HTML_TYPES:
        raise NotHTMLError(url, media_type)
    return _decode(page.content, _charset(header))


def _charset(content_type: str) -> str:
    message = Message()
    message["Content-Type"] = content_type
    return message.get_content_charset() or "utf-8"


def _decode(content: bytes, charset: str) -> str:
    # The charset is untrusted: unknown names and non-text codecs such as rot13 both raise
    # LookupError, so fall back to UTF-8 instead of failing the page.
    try:
        return content.decode(charset, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")
