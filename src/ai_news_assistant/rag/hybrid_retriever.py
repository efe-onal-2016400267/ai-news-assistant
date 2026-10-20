from collections.abc import Iterable, Sequence
from dataclasses import replace

from ai_news_assistant.rag.base import Hit, Retriever, SearchFilter


def rrf_fuse(rankings: Iterable[list[Hit]], rrf_k: int = 60) -> list[Hit]:
    """Take the rankings from retrievers and return the fusion."""
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    fused = [replace(first_seen[chunk_id], score=scores[chunk_id]) for chunk_id in ordered]
    return fused


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, depth: int = 50) -> None:
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        """Search the query for all retrievers, then rrf_fuse."""
        if k <= 0:
            return []

        rankings = [
            retriever.search(query=query, k=self.depth, search_filter=search_filter)
            for retriever in self.retrievers
        ]
        fused = rrf_fuse(rankings=rankings, rrf_k=self.rrf_k)
        return fused[:k]


"""
config_id = "recursive-tok256-32"
question = "What did Mistral announce about their new Large model?"
vector_retriever = VectorRetriever(vector_store=DEFAULT_VECTOR_STORE, config_id=config_id)
bm25_retriever = BM25Retriever(db=DEFAULT_DB, config_id=config_id)

ranking_1 = vector_retriever.search(query=question, k=50)
ranking_2 = bm25_retriever.search(query=question, k=50)

rankings = [ranking_1, ranking_2]
print(rrf_fuse(rankings=rankings))

"""
