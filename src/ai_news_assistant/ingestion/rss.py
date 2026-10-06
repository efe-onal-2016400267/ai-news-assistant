import logging
from collections.abc import Iterable, Mapping
from typing import Any

import feedparser
import httpx
from pydantic import ValidationError

from ai_news_assistant.ingestion.dates import entry_published_at
from ai_news_assistant.ingestion.summary import clean_summary
from ai_news_assistant.ingestion.urls import article_id
from ai_news_assistant.models import Article

USER_AGENT = "ai-news-assistant/0.1"
MAX_SOURCE_LENGTH = 300  # The feed title is untrusted input

logger = logging.getLogger(__name__)


def download_feed(url: str, client: httpx.Client | None = None, *, timeout: float = 15.0) -> bytes:
    """Download a feed and return the raw XML bytes."""

    # To be able to do offline tests with httpx.MockTransport.
    # Tests will call with mock transport, production will call with None as client.
    if client is None:
        with httpx.Client() as own_client:
            return download_feed(url, own_client, timeout=timeout)

    response = client.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )

    response.raise_for_status()
    return response.content


def parse_feed(content: bytes | str, source: str | None = None) -> list[Article]:
    """Parse raw feed XML into validated articles.

    The source name is the publisher part of the feed title ("Section | Publisher"),
    falling back to `source` when the feed has no title. Entries that fail validation
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
                "title": entry.get("title", ""),
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


def _source_name(feed_title: str | None, fallback: str | None) -> str | None:
    title = " ".join((feed_title or "").split())
    name = title.rpartition(" | ")[2].strip() if title else (fallback or "").strip()
    return name[:MAX_SOURCE_LENGTH] or None


def _clean_tags(tags: Iterable[Mapping[str, Any]]) -> list[str]:
    terms = (" ".join((tag.get("term") or "").split()) for tag in tags)
    return list(dict.fromkeys(term for term in terms if term))
