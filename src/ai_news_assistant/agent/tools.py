from dataclasses import asdict
from typing import Any

from langchain.tools import BaseTool, tool

from ai_news_assistant.rag.article_search import search_articles
from ai_news_assistant.rag.base import Retriever
from ai_news_assistant.rag.store import RagStore

MAX_K = 10


# RAG tool
def create_search_tool(retriever: Retriever, store: RagStore) -> BaseTool:
    @tool
    def search_news(query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search the news archive for articles relevant to a question.

        Use this before answering any question about news, AI models, companies or events;
        do not answer such questions from memory.
        Write `query` as a self-contained search with the names filled in, e.g. "Claude Haiku 5.5
        release", not "what about the other one?". `k` is the number of distinct articles to
        return (default 5, max 10).
        Each result has article_id, title, source, published_at (may be null), and snippet
        (excerpts from the article, separated by [...]). Use get_article with an article_id
        to read the full text. Cite the title and source of what you use.
        """
        k = max(1, min(k, MAX_K))
        return [
            asdict(s) for s in search_articles(retriever=retriever, store=store, query=query, k=k)
        ]

    return search_news


def main() -> None:
    from ai_news_assistant.rag.bm25_retriever import BM25Retriever
    from ai_news_assistant.rag.config import (
        DEFAULT_CHUNKING_CONFIG,
        DEFAULT_DB,
        DEFAULT_VECTOR_STORE,
    )
    from ai_news_assistant.rag.hybrid_retriever import HybridRetriever
    from ai_news_assistant.rag.vector_retriever import VectorRetriever

    vector_retriever = VectorRetriever(
        vector_store=DEFAULT_VECTOR_STORE, config_id=DEFAULT_CHUNKING_CONFIG
    )
    bm25_retriever = BM25Retriever(DEFAULT_DB, DEFAULT_CHUNKING_CONFIG)
    store = RagStore(DEFAULT_DB)

    retriever = HybridRetriever(retrievers=[vector_retriever, bm25_retriever])
    print(
        create_search_tool(retriever=retriever, store=store).invoke(
            input={"query": "What did Mistral announce about their new Large model?", "k": 5}
        )
    )


if __name__ == "__main__":
    main()
