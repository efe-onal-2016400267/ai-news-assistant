from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `score` is higher better."""

    chunk_id: str
    article_id: str
    score: float
    text: str


@dataclass(frozen=True)
class SearchFilter:
    """Filter with sources and since arguments."""

    sources: frozenset[str] | None = None  # Exact, canonical names, None = all
    since: datetime | None = None  # Timezone-aware, inclusive (>=)


@runtime_checkable
class Retriever(Protocol):
    """Interface shared by all retrievers (vector, BM25, ...)."""

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        """Return at most `k` hits for `query`, ordered best first.

        `score` is higher-is-better for every retriever (the vector retriever
        turns Chroma's distance into a similarity; BM25 scores already point that way).
        Scores from different retrievers are not comparable, so combining retrievers
        uses reciprocal rank fusion over the rankings.
        A query with no matches returns an empty list; it does not raise.

        A search filter enables filtering on `published_at` and `source` of an
        article. `sources` or `since` can be None, in that case non-None filter
        is applied. If both are None no filters apply.
        """
        ...
