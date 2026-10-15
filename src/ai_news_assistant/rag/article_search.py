from dataclasses import dataclass
from pathlib import Path

from ai_news_assistant.rag.base import Retriever
from ai_news_assistant.rag.store import RagStore

MAX_CHUNKS_PER_ARTICLE = 3
OVERFETCH = 3  # chunks repeat articles, so ask for more chunks than articles wanted
SEPARATOR = "\n[...]\n"


@dataclass
class ArticleSnippet:
    article_id: str
    title: str
    source: str
    published_at: str | None
    snippet: str  # Concatted chunk text from highest score to lowest, separated with [...]


def search_articles(
    retriever: Retriever, store_path: Path, query: str, k: int
) -> list[ArticleSnippet]:
    """Searches the articles and returns the ArticleSnippet.

    Called by search_news tool.
    If the agent decides it needs the full text of an article,
    it will call the full article extraction tool.
    """

    if k <= 0:
        return []

    hits = retriever.search(query=query, k=k * OVERFETCH)

    texts_by_article: dict[str, list[str]] = {}
    for hit in hits:
        texts = texts_by_article.setdefault(hit.article_id, [])
        if len(texts) < MAX_CHUNKS_PER_ARTICLE:
            texts.append(hit.text)

    snippets: list[ArticleSnippet] = []
    for article_id, texts in list(texts_by_article.items())[:k]:
        with RagStore(path=store_path) as store:
            article = store.get_article(article_id)
        snippets.append(
            ArticleSnippet(
                article_id=article_id,
                title=article.title,
                source=article.source,
                published_at=article.published_at,
                snippet=SEPARATOR.join(texts),
            )
        )

    return snippets
