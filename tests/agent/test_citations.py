from pathlib import Path
from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from ai_news_assistant.agent.citations import retrieved_ids
from ai_news_assistant.agent.config import CITATION_TOOLS
from ai_news_assistant.agent.tools import create_get_article_tool, create_search_tool
from ai_news_assistant.rag.base import Hit


class FakeRetriever:
    """Returns a fixed ranking, cut to k."""

    def __init__(self, article_ids: list[str]) -> None:
        self.ranking = [
            Hit(chunk_id=f"{a}-0", article_id=a, score=1.0, text=f"{a} chunk") for a in article_ids
        ]

    def search(self, query: str, k: int) -> list[Hit]:
        return self.ranking[:k]


def call(tool: BaseTool, args: dict[str, Any], call_id: str = "call-1") -> ToolMessage:
    """Run a real tool the way LangGraph's ToolNode does, so the ToolMessage has the real shape."""
    message = tool.invoke({"name": tool.name, "args": args, "id": call_id, "type": "tool_call"})
    assert isinstance(message, ToolMessage)
    return message


def search(rag_db: Path, article_ids: list[str], call_id: str = "call-1") -> ToolMessage:
    tool = create_search_tool(retriever=FakeRetriever(article_ids), store_path=rag_db)
    return call(tool, {"query": "q"}, call_id)


def get_article(rag_db: Path, article_id: str, call_id: str = "call-1") -> ToolMessage:
    return call(create_get_article_tool(store_path=rag_db), {"article_id": article_id}, call_id)


def asks(tool_name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])


# Happy path


def test_search_results_give_all_their_ids(rag_db: Path) -> None:
    messages: list[AnyMessage] = [search(rag_db, ["ai-act", "iphone"])]

    assert retrieved_ids(messages) == {"ai-act", "iphone"}


def test_get_article_success_gives_its_id(rag_db: Path) -> None:
    messages: list[AnyMessage] = [get_article(rag_db, "robotics")]

    assert retrieved_ids(messages) == {"robotics"}


def test_ids_from_several_turns_are_collected(rag_db: Path) -> None:
    messages: list[AnyMessage] = [
        HumanMessage(content="What did the EU pass?"),
        asks("search_news", {"query": "EU AI Act"}, "c1"),
        search(rag_db, ["ai-act", "iphone"], "c1"),
        asks("get_article", {"article_id": "robotics"}, "c2"),
        get_article(rag_db, "robotics", "c2"),
        AIMessage(content="The AI Act passed [1]."),
        HumanMessage(content="Anything else?"),
        asks("search_news", {"query": "bread"}, "c3"),
        search(rag_db, ["sourdough", "ai-act"], "c3"),
    ]

    # Turn 1: ai-act, iphone, robotics. Turn 2: sourdough, ai-act (already seen, counted once).
    assert retrieved_ids(messages) == {"ai-act", "iphone", "robotics", "sourdough"}


# Error paths


def test_get_article_error_gives_nothing(rag_db: Path) -> None:
    message = get_article(rag_db, "invented-id")
    assert "error" in message.text  # the real tool's error result, which names the id

    # Counting it would let S3 accept an id the model made up.
    assert retrieved_ids([message]) == set()


def test_tool_exception_message_is_ignored() -> None:
    message = ToolMessage(
        content="Error: database is locked",
        name="search_news",
        tool_call_id="c1",
        status="error",
    )

    assert retrieved_ids([message]) == set()


def test_non_json_content_is_ignored() -> None:
    message = ToolMessage(content="article_id: ai-act", name="get_article", tool_call_id="c1")

    assert retrieved_ids([message]) == set()


# Edge cases


def test_empty_search_result_gives_nothing(rag_db: Path) -> None:
    message = search(rag_db, [])
    assert message.content == []  # LangChain stores an empty list result as empty content

    assert retrieved_ids([message]) == set()


def test_no_messages_gives_empty_set() -> None:
    assert retrieved_ids([]) == set()


def test_only_human_and_ai_messages_give_nothing() -> None:
    # Asking for an id is not retrieving it.
    messages: list[AnyMessage] = [
        HumanMessage(content="Get article ai-act"),
        asks("get_article", {"article_id": "ai-act"}, "c1"),
    ]

    assert retrieved_ids(messages) == set()


def test_other_tool_names_are_ignored() -> None:
    message = ToolMessage(content='[{"article_id": "x"}]', name="other_tool", tool_call_id="c1")

    assert retrieved_ids([message]) == set()


def test_items_without_article_id_or_not_dicts_are_skipped() -> None:
    message = ToolMessage(
        content='[{"title": "t"}, "x", 3, {"article_id": "a"}]',
        name="search_news",
        tool_call_id="c1",
    )

    assert retrieved_ids([message]) == {"a"}


def test_citation_tools_match_real_tool_names(rag_db: Path) -> None:
    search_tool = create_search_tool(retriever=FakeRetriever([]), store_path=rag_db)
    article_tool = create_get_article_tool(store_path=rag_db)

    assert set(CITATION_TOOLS) == {search_tool.name, article_tool.name}
