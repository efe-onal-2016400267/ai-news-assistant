from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from ai_news_assistant.agent.citations import (
    CheckedCitations,
    check_citations,
    resolve_sources,
    retrieved_ids,
)
from ai_news_assistant.agent.config import CITATION_TOOLS
from ai_news_assistant.agent.tools import (
    create_get_article_tool,
    create_get_latest_articles_tool,
    create_search_tool,
)
from ai_news_assistant.models import ModelAnswer, Source
from ai_news_assistant.rag.base import Hit, SearchFilter


class FakeRetriever:
    """Returns a fixed ranking, cut to k."""

    def __init__(self, article_ids: list[str]) -> None:
        self.ranking = [
            Hit(chunk_id=f"{a}-0", article_id=a, score=1.0, text=f"{a} chunk") for a in article_ids
        ]

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
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
    latest_tool = create_get_latest_articles_tool(store_path=rag_db)

    assert set(CITATION_TOOLS) == {search_tool.name, article_tool.name, latest_tool.name}


def test_latest_articles_give_their_ids(rag_db: Path) -> None:
    message = call(create_get_latest_articles_tool(store_path=rag_db), {"days": 2})

    # Newest corpus date 10-10 09:30, minus 2 days: ai-act and robotics (iphone has no date).
    assert retrieved_ids([message]) == {"ai-act", "robotics"}


# check_citations


def check(text: str, citations: list[str], retrieved: set[str]) -> CheckedCitations:
    answer = ModelAnswer(text=text, citations=citations, grounded=True)
    return check_citations(answer, retrieved)


def test_all_markers_valid_text_unchanged() -> None:
    result = check("A [1]. B [2].", ["a", "b"], {"a", "b"})

    assert result == CheckedCitations(
        clean_text="A [1]. B [2].",
        valid_citations={1: "a", 2: "b"},
        dropped_ids=[],
        dropped_markers=[],
    )


def test_mixed_answer_hand_checked() -> None:
    # citations: [1] -> a (retrieved, used: kept), [2] -> x (not retrieved: dropped),
    # [3] -> b (retrieved, used: kept); [4] has no entry (out of range: removed).
    result = check("A [1]. B [2]. C [3]. D [4].", ["a", "x", "b"], {"a", "b"})

    assert result.clean_text == "A [1]. B. C [3]. D."
    assert result.valid_citations == {1: "a", 3: "b"}
    assert result.dropped_ids == ["x"]
    assert result.dropped_markers == [2, 4]


def test_marker_beyond_citations_removed() -> None:
    result = check("A [1]. B [3].", ["a", "b"], {"a", "b"})

    assert result.clean_text == "A [1]. B."
    assert result.valid_citations == {1: "a"}
    assert result.dropped_ids == []  # no id behind [3]; [2] -> b is simply unused
    assert result.dropped_markers == [3]


def test_unretrieved_id_marker_removed_and_id_dropped() -> None:
    result = check("Invented claim [1].", ["made-up"], {"a"})

    assert result.clean_text == "Invented claim."
    assert result.valid_citations == {}
    assert result.dropped_ids == ["made-up"]
    assert result.dropped_markers == [1]


def test_retrieved_citation_without_marker_not_valid_not_dropped() -> None:
    result = check("Only one claim [1].", ["a", "b"], {"a", "b"})

    assert result.valid_citations == {1: "a"}
    assert result.dropped_ids == []
    assert result.dropped_markers == []


def test_same_marker_twice_gives_one_entry() -> None:
    result = check("A [1]. Again [1].", ["a"], {"a"})

    assert result.clean_text == "A [1]. Again [1]."
    assert result.valid_citations == {1: "a"}


def test_same_bad_marker_twice_reported_once() -> None:
    result = check("A [2]. Again [2].", ["a"], {"a"})

    assert result.clean_text == "A. Again."
    assert result.dropped_markers == [2]


def test_no_markers_no_citations_unchanged() -> None:
    result = check("I do not know.", [], {"a"})

    assert result == CheckedCitations(
        clean_text="I do not know.", valid_citations={}, dropped_ids=[], dropped_markers=[]
    )


def test_marker_zero_removed() -> None:
    result = check("A [0]. B [1].", ["a"], {"a"})

    assert result.clean_text == "A. B [1]."
    assert result.dropped_markers == [0]


def test_numbers_not_renumbered() -> None:
    result = check("A [1]. B [2]. C [3].", ["a", "x", "c"], {"a", "c"})

    assert result.clean_text == "A [1]. B. C [3]."
    assert sorted(result.valid_citations) == [1, 3]


def test_unretrieved_id_dropped_even_without_marker() -> None:
    result = check("A [1].", ["a", "x"], {"a"})

    assert result.clean_text == "A [1]."
    assert result.dropped_ids == ["x"]
    assert result.dropped_markers == []


def test_empty_retrieved_drops_everything_in_citation_order() -> None:
    result = check("A [2]. B [1].", ["first", "second"], set())

    assert result.clean_text == "A. B."
    assert result.valid_citations == {}
    assert result.dropped_ids == ["first", "second"]
    assert result.dropped_markers == [1, 2]


def test_removed_marker_takes_its_spaces_but_not_newlines() -> None:
    result = check("Line one  \t[2]\nLine two [1].", ["a"], {"a"})

    assert result.clean_text == "Line one\nLine two [1]."


def test_kept_marker_keeps_its_space() -> None:
    result = check("Claim [1] and more.", ["a"], {"a"})

    assert result.clean_text == "Claim [1] and more."


def test_non_marker_brackets_kept() -> None:
    # Known gap: only [n] is a marker. The prompt asks for [1][2], never [1, 2] or [1-3].
    text = "See [a], [ 1 ] and [1, 2] or [1-3]."

    result = check(text, ["a"], {"a"})

    assert result.clean_text == text
    assert result.valid_citations == {}
    assert result.dropped_markers == []


def test_adjacent_markers_checked_separately() -> None:
    result = check("Claim [1][2].", ["a", "x"], {"a"})

    assert result.clean_text == "Claim [1]."
    assert result.valid_citations == {1: "a"}
    assert result.dropped_markers == [2]


# resolve_sources


def test_resolve_sources_fills_from_store(rag_db: Path) -> None:
    sources = resolve_sources({1: "ai-act", 2: "iphone"}, rag_db)

    assert sources == [
        Source(
            marker=1,
            article_id="ai-act",
            title="Title of ai-act",
            source="Guardian",
            published_at=datetime(2026, 10, 10, 9, 30, tzinfo=UTC),
        ),
        Source(
            marker=2,
            article_id="iphone",
            title="Title of iphone",
            source="Verge",
            published_at=None,
        ),
    ]


def test_resolve_sources_keeps_marker_gaps(rag_db: Path) -> None:
    sources = resolve_sources({1: "ai-act", 3: "robotics"}, rag_db)

    assert [(s.marker, s.article_id) for s in sources] == [(1, "ai-act"), (3, "robotics")]


def test_resolve_sources_sorted_by_marker(rag_db: Path) -> None:
    sources = resolve_sources({3: "robotics", 1: "ai-act"}, rag_db)

    assert [s.marker for s in sources] == [1, 3]


def test_resolve_sources_empty_gives_empty(rag_db: Path) -> None:
    assert resolve_sources({}, rag_db) == []


def test_resolve_sources_same_article_two_markers(rag_db: Path) -> None:
    sources = resolve_sources({1: "ai-act", 2: "ai-act"}, rag_db)

    assert [(s.marker, s.article_id) for s in sources] == [(1, "ai-act"), (2, "ai-act")]


def test_resolve_sources_unknown_id_raises(rag_db: Path) -> None:
    # Valid ids were retrieved from this same store, so a miss is a bug, not a user error.
    with pytest.raises(KeyError):
        resolve_sources({1: "gone"}, rag_db)
