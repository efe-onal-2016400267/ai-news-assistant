import logging
from collections.abc import Iterable, Mapping
from typing import Any

import feedparser
import httpx
from pydantic import ValidationError

from ai_news_assistant.downloads import (
    DEFAULT_TIMEOUT,
    USER_AGENT,
    ResponseTooLargeError,
    download,
)
from ai_news_assistant.ingestion.dates import entry_published_at
from ai_news_assistant.ingestion.summary import clean_summary, html_to_text
from ai_news_assistant.ingestion.urls import article_id
from ai_news_assistant.models import Article

__all__ = ["USER_AGENT", "FeedTooLargeError", "RSSFetcher", "download_feed", "parse_feed"]

MAX_SOURCE_LENGTH = 300  # The feed title is untrusted input
MAX_FEED_BYTES = 5 * 1024 * 1024  # Real feeds are well under 1 MB

logger = logging.getLogger(__name__)


class FeedTooLargeError(ResponseTooLargeError):
    """The feed response is larger than the download limit."""

    kind = "Feed"


def download_feed(
    url: str,
    client: httpx.Client | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_FEED_BYTES,
) -> bytes:
    """Download a feed and return the raw XML bytes.

    Raises FeedTooLargeError when the body is larger than `max_bytes`, checking
    Content-Length first so an oversized body is never read.
    """
    return download(
        url, client, max_bytes=max_bytes, timeout=timeout, too_large=FeedTooLargeError
    ).content


def parse_feed(content: bytes | str, source: str | None = None) -> list[Article]:
    """Parse raw feed XML into validated articles.

    The source name is `source` when given, since feed titles are untrusted and often not
    the publisher name. Without it, the publisher part of the feed title ("Section |
    Publisher") is used. Entry titles are reduced to plain text. Entries that fail validation
    or repeat an earlier link are skipped and logged. `text` is left empty for the
    later extraction step.
    """
    parsed = feedparser.parse(content)
    if parsed.bozo:
        logger.warning(
            "Malformed feed, using %d recovered entries: %s",
            len(parsed.entries),
            parsed.get("bozo_exception"),
            extra={"source": source},
        )

    source_name = _source_name(parsed.feed.get("title"), source)
    if source_name is None:
        logger.warning("Skipping feed with no title and no source name")
        return []

    articles: dict[str, Article] = {}
    for entry in parsed.entries:
        article = _parse_entry(entry, source_name)
        if article is None:
            continue
        # The id hashes the normalized link, so tracking-parameter variants count too.
        if article.id in articles:
            _log_skip(source_name, str(article.url), "duplicate link")
            continue
        articles[article.id] = article
    return list(articles.values())


class RSSFetcher:
    """Fetches one RSS or Atom feed and returns its articles.

    `source` is the publisher name for its articles; without it, the feed title is used.
    Pass `client` to share a
    connection pool or to test with httpx.MockTransport; with None, each fetch opens its own.
    """

    def __init__(
        self,
        url: str,
        source: str | None = None,
        *,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.url = url
        self.source = source
        self.client = client
        self.timeout = timeout

    def fetch(self) -> list[Article]:
        """Download and parse the feed. Entries that fail validation are skipped.

        HTTP and network errors propagate, so the caller decides what to do per feed.
        """
        content = download_feed(self.url, self.client, timeout=self.timeout)
        return parse_feed(content, self.source)


def _parse_entry(entry: Mapping[str, Any], source: str) -> Article | None:
    link = (entry.get("link") or "").strip()
    if not link:
        _log_skip(source, link, "missing link")
        return None
    try:
        return Article.model_validate(
            {
                "id": article_id(link),
                "url": link,
                "title": _clean_title(entry.get("title")),
                "source": source,
                "published_at": entry_published_at(entry),
                "summary": clean_summary(entry.get("summary")),
                "tags": _clean_tags(entry.get("tags") or []),
            }
        )
    except ValidationError as exc:
        # Field locations and messages only, so the untrusted input isn't echoed into logs.
        reason = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        _log_skip(source, link, reason)
        return None


def _log_skip(source: str, link: str, reason: str) -> None:
    logger.warning(
        "Skipping feed entry %s: %s",
        link or "<no link>",
        reason,
        extra={"source": source, "link": link, "reason": reason},
    )


def _clean_title(raw: str | None) -> str:
    # Some feeds (The Verge) double-escape titles, so feedparser leaves "&#8217;" in them.
    # Decoding can produce markup such as "<script>", so strip it like a summary.
    return html_to_text(raw or "")


def _source_name(feed_title: str | None, configured: str | None) -> str | None:
    name = " ".join((configured or "").split())
    if not name:
        title = " ".join((feed_title or "").split())
        name = title.rpartition(" | ")[2].strip()
    return name[:MAX_SOURCE_LENGTH] or None


def _clean_tags(tags: Iterable[Mapping[str, Any]]) -> list[str]:
    terms = (" ".join((tag.get("term") or "").split()) for tag in tags)
    return list(dict.fromkeys(term for term in terms if term))
