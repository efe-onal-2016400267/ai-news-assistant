import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ai_news_assistant.downloads import ResponseTooLargeError
from ai_news_assistant.extraction.fetch import NotHTMLError, fetch_html
from ai_news_assistant.extraction.robots import RobotsPolicy, RobotsUnavailableError
from ai_news_assistant.extraction.text import extract_text_from_html
from ai_news_assistant.ingest import DEFAULT_DB_PATH, configure_logging
from ai_news_assistant.models import Article
from ai_news_assistant.storage.base import ArticleStore
from ai_news_assistant.storage.sqlite import SQLiteStore

DEFAULT_LOG_FILE = "logs/extract.log"
DEFAULT_LIMIT = 200  # Articles per run; the backlog drains over scheduled runs
DEFAULT_HOST_DELAY = 2.0  # Seconds between requests to the same host
MAX_HOST_DELAY = 60.0  # Upper bound for a Crawl-delay asked by robots.txt
RETRY_STATUSES = {408, 429}  # Request Timeout and Too Many Requests: try again later

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class ArticleResult:
    article_id: str
    outcome: Outcome


@dataclass(frozen=True)
class ExtractResult:
    articles: list[ArticleResult]

    def count(self, outcome: Outcome) -> int:
        return sum(article.outcome is outcome for article in self.articles)

    @property
    def all_retried(self) -> bool:
        """True when there were articles and every one has to be retried (network down?)."""
        return bool(self.articles) and self.count(Outcome.RETRY) == len(self.articles)


def extract(
    store: ArticleStore,
    client: httpx.Client | None = None,
    *,
    limit: int | None = DEFAULT_LIMIT,
    robots: RobotsPolicy | None = None,
    throttle: HostThrottle | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ExtractResult:
    """Fetch, extract and store the text of up to `limit` articles never tried, newest first.

    Each article ends with one Outcome. Saved, no-text and permanently failed articles are
    recorded, so they are not tried again; temporary failures and unexpected errors record
    nothing, so the next run retries them. One failing article never stops the others.
    """
    robots = robots if robots is not None else RobotsPolicy(client)
    throttle = throttle if throttle is not None else HostThrottle()
    results: list[ArticleResult] = []
    for article in store.get_unextracted(limit):
        try:
            outcome = _extract_one(article, store, client, robots, throttle, now)
        except Exception:
            logger.exception("Article %s: unexpected error", article.id, extra=_log_extra(article))
            outcome = Outcome.ERROR
        results.append(ArticleResult(article.id, outcome))

    result = ExtractResult(results)
    # Counts only: article and page text are untrusted and stay out of logs.
    logger.info(
        "Extract done: %d articles, %d saved, %d no text, %d failed, %d retry, %d errors",
        len(results),
        result.count(Outcome.SAVED),
        result.count(Outcome.NO_TEXT),
        result.count(Outcome.FAILED),
        result.count(Outcome.RETRY),
        result.count(Outcome.ERROR),
    )
    return result


def _extract_one(
    article: Article,
    store: ArticleStore,
    client: httpx.Client | None,
    robots: RobotsPolicy,
    throttle: HostThrottle,
    now: Callable[[], datetime],
) -> Outcome:
    """Handle one article. Unexpected errors propagate to the caller, which logs them."""
    url = str(article.url)
    text: str | None = None
    try:
        if not robots.allowed(url):
            outcome, reason = Outcome.FAILED, "robots.txt disallows"
        else:
            throttle.wait(url, robots.crawl_delay(url))
            text = extract_text_from_html(fetch_html(url, client))
            outcome, reason = (Outcome.SAVED, "") if text else (Outcome.NO_TEXT, "no article text")
    except Exception as exc:
        outcome = classify(exc)
        if outcome is Outcome.ERROR:
            raise
        reason = _describe(exc)

    if outcome is not Outcome.RETRY:
        store.save_extraction(article.id, text if outcome is Outcome.SAVED else None, now())
    logger.info(
        "Article %s (%s): %s%s",
        article.id,
        urlsplit(url).netloc,
        outcome,
        f", {reason}" if reason else "",
        extra=_log_extra(article),
    )
    return outcome


def _describe(exc: BaseException) -> str:
    # Error type and status only: messages can carry untrusted header values.
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _log_extra(article: Article) -> dict[str, str]:
    return {"article_id": article.id, "source": article.source}


def run(
    db_path: Path,
    client: httpx.Client | None = None,
    *,
    limit: int | None = DEFAULT_LIMIT,
    throttle: HostThrottle | None = None,
) -> int:
    """Extract articles in the SQLite database at `db_path`.

    Returns the exit code: 1 if every article had to be retried, otherwise 0.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with SQLiteStore(db_path) as store:
        result = extract(store, client, limit=limit, throttle=throttle)
    return 1 if result.all_retried else 0


def main() -> int:
    """Command-line entry point: NEWS_DB_PATH, NEWS_LOG_FILE and EXTRACT_LIMIT configure it."""
    db_path = Path(os.environ.get("NEWS_DB_PATH", DEFAULT_DB_PATH))
    configure_logging(Path(os.environ.get("NEWS_LOG_FILE", DEFAULT_LOG_FILE)))
    try:
        limit = int(os.environ.get("EXTRACT_LIMIT", DEFAULT_LIMIT))
    except ValueError:
        limit = -1
    if limit < 0:
        logger.error("EXTRACT_LIMIT must be a whole number, 0 or more")
        return 2
    with httpx.Client() as client:
        return run(db_path, client, limit=limit)


if __name__ == "__main__":
    sys.exit(main())
