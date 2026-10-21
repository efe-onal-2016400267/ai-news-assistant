import re
from datetime import datetime
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ai_news_assistant.rag.base import ArticleMetadata, Hit, SearchFilter
from ai_news_assistant.rag.store import RagStore

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|\w+")


def simple_tokenize(text: str) -> list[str]:
    """The tokenizer function that takes a text and returns the tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, db: str | Path, config_id: str) -> None:

        with RagStore(db) as store:
            self.chunks = store.get_chunks(config_id=config_id)
            if not self.chunks:
                raise ValueError(f"No chunks for config {config_id!r}")
            self.bm25 = BM25Okapi([simple_tokenize(c.text) for c in self.chunks])
            self.articles_metadata = {
                a.id: ArticleMetadata(
                    article_id=a.id,
                    source=a.source,
                    published_at=datetime.fromisoformat(str(a.published_at))
                    if a.published_at
                    else None,
                )
                for a in store.corpus_articles()
            }

    def matches(self, metadata: ArticleMetadata, search_filter: SearchFilter) -> bool:
        """Apply sources filter to an ArticleMetadata."""
        if search_filter.sources is not None and metadata.source not in search_filter.sources:
            return False
        if search_filter.since is not None and (
            metadata.published_at is None or metadata.published_at < search_filter.since
        ):
            return False
        return True

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        """Return top-k chunks based on bm25."""
        query_tokens = simple_tokenize(query)
        if k <= 0 or not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        # filter before getting the top k
        # for each chunk check if the chunks articles metadata passes the filter
        if search_filter is not None:
            keep = np.array(
                [
                    self.matches(self.articles_metadata[c.article_id], search_filter)
                    for c in self.chunks
                ]
            )
            scores = np.where(keep, scores, 0.0)
        # return the indices on the original scores list based on -scores
        top = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(
                chunk_id=self.chunks[c].chunk_id,
                article_id=self.chunks[c].article_id,
                score=float(scores[c]),
                text=self.chunks[c].text,
            )
            for c in top
            if scores[c] > 0
        ]
