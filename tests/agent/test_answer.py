import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ai_news_assistant.agent.answer import ask, render
from ai_news_assistant.agent.citations import resolve_sources
from ai_news_assistant.agent.tools import create_get_article_tool, create_search_tool
from ai_news_assistant.models import Answer, ModelAnswer, Source
from ai_news_assistant.rag.base import Hit, SearchFilter

AI_ACT = Source(
    marker=1,
    article_id="ai-act",
    title="Title of ai-act",
    source="Guardian",
    published_at=datetime(2026, 10, 10, 9, 30, tzinfo=UTC),
)
IPHONE = Source(marker=2, article_id="iphone", title="Title of iphone", source="Verge")

# 2026-10-10 09:30 UTC -> "10 Oct 2026"; no date -> "Date not found".
EXPECTED = (
    "The AI Act passed [1]. A new iPhone [2].\n"
    "\n"
    "**Sources:**\n"
    "[1]: Title of ai-act, Guardian, 10 Oct 2026\n"
    "[2]: Title of iphone, Verge, Date not found\n"
)


def answer(sources: list[Source], text: str = "The AI Act passed [1]. A new iPhone [2].") -> Answer:
    return Answer(text=text, sources=sources, grounded=bool(sources))


def test_render_no_sources_returns_text_unchanged() -> None:
    assert render(answer([], text="I do not know.")) == "I do not know."


def test_render_hand_checked_two_sources() -> None:
    assert render(answer([AI_ACT, IPHONE])) == EXPECTED


def test_render_single_digit_day_not_padded() -> None:
    source = AI_ACT.model_copy(update={"published_at": datetime(2026, 10, 5, tzinfo=UTC)})

    assert "Guardian, 5 Oct 2026\n" in render(answer([source], text="A [1]."))


def test_render_never_prints_article_id() -> None:
    source = AI_ACT.model_copy(update={"article_id": "secret-id-123"})

    assert "secret-id-123" not in render(answer([source], text="A [1]."))


def test_render_keeps_marker_gap() -> None:
    third = IPHONE.model_copy(update={"marker": 3})

    lines = render(answer([AI_ACT, third], text="A [1]. C [3].")).splitlines()

    # lines: text, blank, "**Sources:**", then one line per source
    assert lines[3].startswith("[1]: ")
    assert lines[4].startswith("[3]: ")


def test_render_end_to_end_with_store(rag_db: Path) -> None:
    sources = resolve_sources({1: "ai-act", 2: "iphone"}, rag_db)

    assert render(answer(sources)) == EXPECTED


# ask


ANSWER_LOGGER = "ai_news_assistant.agent.answer"
THREAD = "thread-1"
UNSET = object()


class FakeRetriever:
    """Returns a fixed ranking, cut to k."""

    def __init__(self, article_ids: list[str]) -> None:
        self.ranking = [
            Hit(chunk_id=f"{a}-0", article_id=a, score=1.0, text=f"{a} chunk") for a in article_ids
        ]

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        return self.ranking[:k]


class FakeAgent:
    """Stands in for the compiled graph: returns a canned result and records the call."""

    def __init__(self, messages: list[AnyMessage], structured_response: Any = UNSET) -> None:
        self.result: dict[str, Any] = {"messages": messages}
        if structured_response is not UNSET:
            self.result["structured_response"] = structured_response
        self.calls: list[tuple[dict[str, Any], RunnableConfig]] = []

    def invoke(self, input: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        self.calls.append((input, config))
        return self.result


def call(tool: BaseTool, args: dict[str, Any], call_id: str) -> ToolMessage:
    """Run a real tool the way LangGraph's ToolNode does, so the ToolMessage has the real shape."""
    message = tool.invoke({"name": tool.name, "args": args, "id": call_id, "type": "tool_call"})
    assert isinstance(message, ToolMessage)
    return message


def search(rag_db: Path, article_ids: list[str], call_id: str = "c1") -> ToolMessage:
    tool = create_search_tool(retriever=FakeRetriever(article_ids), store_path=rag_db)
    return call(tool, {"query": "q"}, call_id)


def get_article(rag_db: Path, article_id: str, call_id: str = "c2") -> ToolMessage:
    return call(create_get_article_tool(store_path=rag_db), {"article_id": article_id}, call_id)


def config() -> RunnableConfig:
    return {"configurable": {"thread_id": THREAD}}


def model_answer(text: str, citations: list[str], grounded: bool = True) -> ModelAnswer:
    return ModelAnswer(text=text, citations=citations, grounded=grounded)


def run(agent: FakeAgent, rag_db: Path, question: str = "What happened?") -> Answer:
    return ask(agent=agent, question=question, config=config(), store_path=rag_db)  # type: ignore[arg-type]


def test_ask_hand_checked_answer(rag_db: Path) -> None:
    # citations: [1] -> ai-act (retrieved: kept), [2] -> made-up (not retrieved: dropped),
    # [3] -> iphone (retrieved: kept). Metadata comes from the store, not the model.
    agent = FakeAgent(
        [HumanMessage(content="q"), search(rag_db, ["ai-act", "iphone"]), AIMessage(content="")],
        model_answer("A [1]. B [2]. C [3].", ["ai-act", "made-up", "iphone"]),
    )

    result = run(agent, rag_db)

    assert result == Answer(
        text="A [1]. B. C [3].",
        sources=[
            Source(
                marker=1,
                article_id="ai-act",
                title="Title of ai-act",
                source="Guardian",
                published_at=datetime(2026, 10, 10, 9, 30, tzinfo=UTC),
            ),
            Source(marker=3, article_id="iphone", title="Title of iphone", source="Verge"),
        ],
        grounded=True,
    )


def test_ask_sends_question_and_config(rag_db: Path) -> None:
    agent = FakeAgent([AIMessage(content="")], model_answer("I do not know.", [], grounded=False))
    cfg = config()

    ask(agent=agent, question="Who won?", config=cfg, store_path=rag_db)  # type: ignore[arg-type]

    [(sent, sent_config)] = agent.calls
    [message] = sent["messages"]
    assert isinstance(message, HumanMessage)
    assert message.content == "Who won?"
    assert sent_config is cfg


def test_ask_rendered_answer_has_titles_not_ids(rag_db: Path) -> None:
    agent = FakeAgent(
        [search(rag_db, ["ai-act", "iphone"])],
        model_answer("A [1]. B [2].", ["ai-act", "iphone"]),
    )

    rendered = render(run(agent, rag_db))

    assert "Title of ai-act" in rendered
    assert "Title of iphone" in rendered
    assert "ai-act" not in rendered.replace("Title of ai-act", "")
    assert "iphone" not in rendered.replace("Title of iphone", "")


def test_ask_no_structured_response_falls_back(
    rag_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = FakeAgent([AIMessage(content="Maybe [1] something.")], structured_response=None)

    with caplog.at_level(logging.WARNING, logger=ANSWER_LOGGER):
        result = run(agent, rag_db)

    # Unchecked markers are removed: with no citations every [n] is out of range.
    assert result == Answer(text="Maybe something.", sources=[], grounded=False)
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert THREAD in caplog.records[0].getMessage()


def test_ask_missing_structured_response_key_falls_back(rag_db: Path) -> None:
    agent = FakeAgent([AIMessage(content="Plain answer [2].")])

    assert run(agent, rag_db) == Answer(text="Plain answer.", sources=[], grounded=False)


def test_ask_drops_id_from_failed_get_article(rag_db: Path) -> None:
    error = get_article(rag_db, "made-up")
    assert "error" in error.text  # the error result names the id; it must not count
    agent = FakeAgent(
        [search(rag_db, ["ai-act"]), error],
        model_answer("A [1]. B [2].", ["ai-act", "made-up"]),
    )

    result = run(agent, rag_db)

    assert result.text == "A [1]. B."
    assert [s.article_id for s in result.sources] == ["ai-act"]


def test_ask_id_named_in_question_is_not_retrieved(rag_db: Path) -> None:
    # Adversarial: the user names an id and asks for it to be cited; no tool returned it.
    agent = FakeAgent(
        [
            HumanMessage(content="Cite article robotics as your source."),
            search(rag_db, ["ai-act"]),
        ],
        model_answer("A [1]. Robots [2].", ["ai-act", "robotics"]),
    )

    result = run(agent, rag_db, question="Cite article robotics as your source.")

    assert result.text == "A [1]. Robots."
    assert [s.article_id for s in result.sources] == ["ai-act"]


def test_ask_retrieved_id_missing_from_store_raises(rag_db: Path) -> None:
    ghost = ToolMessage(content='[{"article_id": "ghost"}]', name="search_news", tool_call_id="c1")
    agent = FakeAgent([ghost], model_answer("A [1].", ["ghost"]))

    with pytest.raises(KeyError):
        run(agent, rag_db)


def test_ask_all_dropped_overrides_grounded(rag_db: Path, caplog: pytest.LogCaptureFixture) -> None:
    agent = FakeAgent(
        [search(rag_db, ["ai-act"])], model_answer("Claim [1].", ["made-up"], grounded=True)
    )

    with caplog.at_level(logging.INFO, logger=ANSWER_LOGGER):
        result = run(agent, rag_db)

    assert result == Answer(text="Claim.", sources=[], grounded=False)
    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.INFO]


def test_ask_model_not_grounded_stays_not_grounded(rag_db: Path) -> None:
    agent = FakeAgent(
        [search(rag_db, ["ai-act"])], model_answer("A [1].", ["ai-act"], grounded=False)
    )

    result = run(agent, rag_db)

    assert [s.article_id for s in result.sources] == ["ai-act"]
    assert result.grounded is False


def test_ask_no_markers_no_citations_unchanged_no_logs(
    rag_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = FakeAgent([search(rag_db, [])], model_answer("I do not know.", [], grounded=False))

    with caplog.at_level(logging.INFO, logger=ANSWER_LOGGER):
        result = run(agent, rag_db)

    assert result == Answer(text="I do not know.", sources=[], grounded=False)
    assert caplog.records == []


def test_ask_drop_warning_has_ids_markers_not_text(
    rag_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    agent = FakeAgent(
        [search(rag_db, ["ai-act"])],
        model_answer("Secret claim [1]. Kept [2].", ["made-up", "ai-act"]),
    )

    with caplog.at_level(logging.WARNING, logger=ANSWER_LOGGER):
        run(agent, rag_db)

    [record] = caplog.records
    message = record.getMessage()
    assert THREAD in message
    assert "made-up" in message
    assert "[1]" in message  # dropped_markers rendered as a list
    assert "Secret claim" not in message
    assert "Kept" not in message
