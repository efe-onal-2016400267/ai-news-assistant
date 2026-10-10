from dataclasses import dataclass

import httpx

USER_AGENT = "ai-news-assistant/0.1"
DEFAULT_TIMEOUT = 15.0  # Seconds


class ResponseTooLargeError(Exception):
    """The response body is larger than the download limit."""

    kind = "Response"  # Subclasses name what was downloaded, e.g. "Feed"

    def __init__(self, url: str, max_bytes: int) -> None:
        super().__init__(f"{self.kind} {url} exceeds {max_bytes} bytes")
        self.url = url
        self.max_bytes = max_bytes


@dataclass(frozen=True)
class Download:
    content: bytes
    headers: httpx.Headers


def download(
    url: str,
    client: httpx.Client | None = None,
    *,
    max_bytes: int,
    timeout: float = DEFAULT_TIMEOUT,
    accept: str | None = None,
    too_large: type[ResponseTooLargeError] = ResponseTooLargeError,
) -> Download:
    """Download `url` and return the raw body bytes with the response headers.

    Raises `too_large` when the body is larger than `max_bytes`, checking Content-Length
    first so an oversized body is never read. HTTP and network errors propagate.
    """

    # To be able to do offline tests with httpx.MockTransport.
    # Tests will call with mock transport, production will call with None as client.
    if client is None:
        with httpx.Client() as own_client:
            return download(
                url,
                own_client,
                max_bytes=max_bytes,
                timeout=timeout,
                accept=accept,
                too_large=too_large,
            )

    headers = {"User-Agent": USER_AGENT}
    if accept is not None:
        headers["Accept"] = accept

    with client.stream(
        "GET",
        url,
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    ) as response:
        response.raise_for_status()
        if _content_length(response) > max_bytes:
            raise too_large(url, max_bytes)

        # iter_bytes yields decompressed bytes, so this also caps gzip bombs.
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise too_large(url, max_bytes)
            chunks.append(chunk)
        return Download(b"".join(chunks), response.headers)


def _content_length(response: httpx.Response) -> int:
    # The header is untrusted; a missing or invalid value leaves the byte count as the guard.
    try:
        return max(int(response.headers.get("Content-Length", 0)), 0)
    except ValueError:
        return 0
