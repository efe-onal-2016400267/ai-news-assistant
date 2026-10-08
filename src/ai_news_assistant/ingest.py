import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx

from ai_news_assistant.ingestion.base import Fetcher
from ai_news_assistant.ingestion.feeds import FEEDS
from ai_news_assistant.ingestion.rss import RSSFetcher
from ai_news_assistant.storage.base import ArticleStore
from ai_news_assistant.storage.sqlite import SQLiteStore

DEFAULT_DB_PATH = "data/news.db"
DEFAULT_LOG_FILE = "logs/ingest.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedResult:
    """Outcome of one feed: how many articles it returned and how many were new, or the error."""

    name: str
    fetched: int = 0
    new: int = 0
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class IngestResult:
    feeds: list[FeedResult]

    @property
    def ok_count(self) -> int:
        return sum(feed.ok for feed in self.feeds)

    @property
    def failed_count(self) -> int:
        return len(self.feeds) - self.ok_count

    @property
    def fetched(self) -> int:
        return sum(feed.fetched for feed in self.feeds)

    @property
    def new(self) -> int:
        return sum(feed.new for feed in self.feeds)

    @property
    def all_failed(self) -> bool:
        """True when no feed succeeded, including when there were no feeds at all."""
        return self.ok_count == 0


def ingest(fetchers: Mapping[str, Fetcher], store: ArticleStore) -> IngestResult:
    """Fetch every feed and save its articles. A failing feed is logged and skipped.

    Safe to rerun: articles already stored are not counted as new.
    """
    results: list[FeedResult] = []
    for name, fetcher in fetchers.items():
        try:
            articles = fetcher.fetch()
            new = store.save(articles)
        except Exception as exc:
            logger.exception("Feed %s failed", name, extra={"feed": name})
            results.append(FeedResult(name, error=exc))
        else:
            results.append(FeedResult(name, fetched=len(articles), new=new))

    result = IngestResult(results)
    # Counts and feed names only: article text is untrusted and stays out of logs.
    logger.info(
        "Ingest done: %d feeds ok, %d failed, %d articles fetched, %d new",
        result.ok_count,
        result.failed_count,
        result.fetched,
        result.new,
    )
    return result


def run(feeds: Mapping[str, str], db_path: Path, client: httpx.Client | None = None) -> int:
    """Ingest `feeds` (name -> URL) into the SQLite database at `db_path`.

    Returns the exit code: 1 if no feed succeeded, otherwise 0.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fetchers = {name: RSSFetcher(url, name, client=client) for name, url in feeds.items()}
    with SQLiteStore(db_path) as store:
        result = ingest(fetchers, store)
    return 1 if result.all_failed else 0


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def main() -> int:
    """Command-line entry point. Paths come from NEWS_DB_PATH and NEWS_LOG_FILE."""
    db_path = Path(os.environ.get("NEWS_DB_PATH", DEFAULT_DB_PATH))
    configure_logging(Path(os.environ.get("NEWS_LOG_FILE", DEFAULT_LOG_FILE)))
    with httpx.Client() as client:
        return run(FEEDS, db_path, client)


if __name__ == "__main__":
    sys.exit(main())
