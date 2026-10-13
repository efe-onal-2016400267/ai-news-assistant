from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `score` is higher better."""

    chunk_id: str
    article_id: str
    score: float
    text: str


@runtime_checkable
class Retriever(Protocol):
    """Interface shared by all retrievers (vector, BM25, ...)."""

    def search(self, query: str, k: int) -> list[Hit]:
        """Return at most `k` hits for `query`, ordered best first.

        `score` is higher-is-better for every retriever (the vector retriever
        turns Chroma's distance into a similarity; BM25 scores already point that way).
        Scores from different retrievers are not comparable, so combining retrievers
        uses reciprocal rank fusion over the rankings.
        A query with no matches returns an empty list; it does not raise.
        """
        ...
