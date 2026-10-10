import time
from collections.abc import Callable
from enum import StrEnum
from urllib.parse import urlsplit

import httpx

from ai_news_assistant.downloads import ResponseTooLargeError
from ai_news_assistant.extraction.fetch import NotHTMLError
from ai_news_assistant.extraction.robots import RobotsUnavailableError

DEFAULT_HOST_DELAY = 2.0  # Seconds between requests to the same host
MAX_HOST_DELAY = 60.0  # Upper bound for a Crawl-delay asked by robots.txt
RETRY_STATUSES = {408, 429}  # Request Timeout and Too Many Requests: try again later


class Outcome(StrEnum):
    """What happened to one article in an extraction run."""

    SAVED = "saved"  # Text extracted and stored
    NO_TEXT = "no_text"  # Page fetched, but no usable article text; recorded as a failure
    FAILED = "failed"  # Permanent failure, recorded so the article is not tried again
    RETRY = "retry"  # Temporary failure, nothing recorded, so the next run tries again
    ERROR = "error"  # Unexpected exception (a bug); nothing recorded


# httpx's own tree doesn't match "temporary": UnsupportedProtocol is a TransportError, while
# TooManyRedirects and DecodingError are not, so errors are listed by concrete type.
_TEMPORARY_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
    RobotsUnavailableError,
)
_PERMANENT_ERRORS = (
    ResponseTooLargeError,
    NotHTMLError,
    httpx.TooManyRedirects,  # A redirect loop
    httpx.DecodingError,  # A corrupt body
    httpx.UnsupportedProtocol,  # A link that isn't http(s)
    httpx.LocalProtocolError,
    httpx.InvalidURL,
)


def classify(exc: BaseException) -> Outcome:
    """Decide whether a failed article is retried next run or recorded as failed (D12)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status >= 500 or status in RETRY_STATUSES:
            return Outcome.RETRY
        return Outcome.FAILED
    if isinstance(exc, _TEMPORARY_ERRORS):
        return Outcome.RETRY
    if isinstance(exc, _PERMANENT_ERRORS):
        return Outcome.FAILED
    return Outcome.ERROR


class HostThrottle:
    """Keeps a minimum gap between requests to the same host; other hosts don't wait.

    The gap is `min_delay`, or a larger Crawl-delay from robots.txt, capped at `max_delay`.
    robots.txt downloads are not throttled: there is one per host per run.
    """

    def __init__(
        self,
        min_delay: float = DEFAULT_HOST_DELAY,
        max_delay: float = MAX_HOST_DELAY,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}

    def wait(self, url: str, delay: float | None = None) -> None:
        """Sleep until a request to this URL's host is allowed, then record the request."""
        host = urlsplit(url).netloc.lower()
        gap = min(max(self.min_delay, delay or 0.0), self.max_delay)
        last = self._last.get(host)
        if last is not None:
            remaining = last + gap - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last[host] = self._clock()
