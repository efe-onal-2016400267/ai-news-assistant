import asyncio
import json
import os
from pathlib import Path

from fastmcp import FastMCP

from ai_news_assistant.agent.agent import build_retriever
from ai_news_assistant.agent.tools import (
    GET_LATEST_ARTICLES_DESCRIPTION,
    GET_SINGLE_ARTICLE_DESCRIPTION,
    SEARCH_NEWS_DESCRIPTION,
    get_single_article,
    list_latest_articles,
    search_news_articles,
)
from ai_news_assistant.rag.base import Retriever
from ai_news_assistant.rag.config import DEFAULT_DB
from ai_news_assistant.rag.store import RagStore


def _dump(result: object) -> str:
    """Dumps json to return tool results."""
    return json.dumps(result, ensure_ascii=False)


def get_sources(store_path: Path) -> list[str]:
    """Get valid sources."""
    with RagStore(store_path) as store:
        sources = store.sources()
    return sources


def create_mcp_server(retriever: Retriever, store_path: Path) -> FastMCP:
    """Create the MCP server."""
    mcp = FastMCP("ai-news-tools")
    valid_sources = get_sources(
        store_path=store_path
    )  # Doesnt need to be async since it runs once at startup.

    # Search news articles tool definition
    # Create search description
    search_description = SEARCH_NEWS_DESCRIPTION.format(sources=", ".join(valid_sources))

    # Define the tool as an async function.
    @mcp.tool(name="search_news", description=search_description)
    async def search_news(
        query: str, k: int = 5, sources: list[str] | None = None, days: int | None = None
    ) -> str:
        result = await asyncio.to_thread(
            search_news_articles,
            retriever=retriever,
            store_path=store_path,
            gold_source_list=valid_sources,
            query=query,
            k=k,
            sources=sources,
            days=days,
        )
        return _dump(result)

    # Define get article tool
    @mcp.tool(name="get_article", description=GET_SINGLE_ARTICLE_DESCRIPTION)
    async def get_article(article_id: str) -> str:
        result = await asyncio.to_thread(
            get_single_article,
            article_id=article_id,
            store_path=store_path,
        )
        return _dump(result)

    # Define get latest articles tool
    @mcp.tool(name="get_latest_articles", description=GET_LATEST_ARTICLES_DESCRIPTION)
    async def get_latest_articles(days: int = 7) -> str:
        result = await asyncio.to_thread(
            list_latest_articles,
            days=days,
            store_path=store_path,
        )
        return _dump(result)

    return mcp


def main() -> None:
    server = create_mcp_server(
        retriever=build_retriever(),
        store_path=DEFAULT_DB,
    )
    server.run(transport="http", host=os.environ.get("TOOLS_HOST", "127.0.0.1"), port=9000)


if __name__ == "__main__":
    main()
