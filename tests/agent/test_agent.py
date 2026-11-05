from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_news_assistant.agent.agent import build_agent
from ai_news_assistant.rag.base import Hit, SearchFilter


class FakeRetriever:
    """Returns nothing; only its identity matters here."""

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        return []


def test_build_agent_registers_the_three_tools(
    rag_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ChatOpenAI needs a key to construct; it makes no network call until invoked.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    agent = build_agent(retriever=FakeRetriever(), trigger_tokens=10, store_path=rag_db)

    tools_node = agent.nodes["tools"].bound
    assert sorted(tools_node.tools_by_name) == ["get_article", "get_latest_articles", "search_news"]


def test_build_agent_defaults_to_an_in_memory_checkpointer(
    rag_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CLI and eval call build_agent without a checkpointer; they keep in-memory threads.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    agent = build_agent(retriever=FakeRetriever(), trigger_tokens=10, store_path=rag_db)

    assert isinstance(agent.checkpointer, InMemorySaver)


def test_build_agent_uses_the_given_checkpointer(
    rag_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    saver = InMemorySaver()

    agent = build_agent(
        retriever=FakeRetriever(), trigger_tokens=10, store_path=rag_db, checkpointer=saver
    )

    assert agent.checkpointer is saver
