from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ai_news_assistant.models import Article


@runtime_checkable
class ArticleStore(Protocol):
    """Persistent storage for articles, keyed by `Article.id`.

    SQLite locally and in tests, Postgres at deployment; callers only see this interface.
    """

    def save(self, articles: Iterable[Article]) -> int:
        """Insert articles not stored yet; existing ids are left unchanged.

        Returns how many articles were new, so repeated ingests are idempotent.
        """
        ...

    def get(self, article_id: str) -> Article | None:
        """Return the stored article, or None if the id is unknown."""
        ...

    def get_unextracted(self, limit: int | None = None) -> list[Article]:
        """Return articles whose extraction was never tried, newest first, undated last.

        Failed and successful extractions are excluded. `limit` caps the batch; None means all.
        """
        ...
