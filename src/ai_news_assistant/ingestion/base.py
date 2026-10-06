from typing import Protocol, runtime_checkable

from ai_news_assistant.models import Article


@runtime_checkable
class Fetcher(Protocol):
    """A news source that returns its current articles.

    Implementations take their configuration (URL, source name, credentials) in
    their constructor, so every source is fetched the same way.
    """

    def fetch(self) -> list[Article]:
        """Fetch the source and return its articles. Entries that fail validation are skipped."""
        ...
