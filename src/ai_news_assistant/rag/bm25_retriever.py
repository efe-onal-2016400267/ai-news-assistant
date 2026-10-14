import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ai_news_assistant.rag.base import Hit
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

    def search(self, query: str, k: int) -> list[Hit]:
        """Return top-k chunks based on bm25."""
        query_tokens = simple_tokenize(query)
        if k <= 0 or not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
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
