import json
from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from ai_news_assistant.agent.citations import retrieved_ids
from ai_news_assistant.agent.config import (
    CLEARED_PLACEHOLDER,
    CONTEXT_TRIGGER_TOKENS,
    KEEP_TOOL_RESULTS,
)


def our_edit() -> ClearToolUsesEdit:
    """The edit as build_agent configures it."""
    return ClearToolUsesEdit(
        trigger=CONTEXT_TRIGGER_TOKENS, keep=KEEP_TOOL_RESULTS, placeholder=CLEARED_PLACEHOLDER
    )


def fixed_count(tokens: int) -> Any:
    """A token counter that always reports `tokens`, so the trigger is under test control."""

    def count(messages: Sequence[BaseMessage]) -> int:
        return tokens

    return count


OVER = fixed_count(CONTEXT_TRIGGER_TOKENS + 1)


def tool_turn(article_id: str, tool: str = "get_article") -> list[AnyMessage]:
    """An AI tool call and its result, shaped like ToolNode output."""
    call_id = f"call-{article_id}"
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": tool, "args": {"article_id": article_id}, "id": call_id}],
        ),
        ToolMessage(
            content=json.dumps({"article_id": article_id, "title": f"Title of {article_id}"}),
            name=tool,
            tool_call_id=call_id,
        ),
    ]


def thread(article_ids: list[str]) -> list[AnyMessage]:
    messages: list[AnyMessage] = [HumanMessage(content="What is new?")]
    for article_id in article_ids:
        messages += tool_turn(article_id)
    messages.append(AIMessage(content="Answer [1]."))
    return messages


def tool_contents(messages: list[AnyMessage]) -> list[Any]:
    return [m.content for m in messages if isinstance(m, ToolMessage)]


def is_cleared(message: AnyMessage) -> bool:
    return bool(message.response_metadata.get("context_editing", {}).get("cleared"))


# Trigger


def test_at_trigger_nothing_changes() -> None:
    messages = thread(["a1", "a2", "a3", "a4", "a5"])
    before = [m.model_copy() for m in messages]

    # `apply` edits only when tokens > trigger, so exactly the trigger is left alone.
    our_edit().apply(messages, count_tokens=fixed_count(CONTEXT_TRIGGER_TOKENS))

    assert messages == before


def test_over_trigger_clears_all_but_last_kept() -> None:
    messages = thread(["a1", "a2", "a3", "a4", "a5"])

    our_edit().apply(messages, count_tokens=OVER)

    # 5 tool results, keep 3: a1 and a2 cleared, a3-a5 intact.
    assert KEEP_TOOL_RESULTS == 3
    contents = tool_contents(messages)
    assert contents[:2] == [CLEARED_PLACEHOLDER, CLEARED_PLACEHOLDER]
    assert [json.loads(c)["article_id"] for c in contents[2:]] == ["a3", "a4", "a5"]


def test_real_approximate_counter_triggers_on_long_results() -> None:
    # Four results of 20,000 characters: about 5,000 tokens each with the approximate
    # counter (about 4 characters per token), well over CONTEXT_TRIGGER_TOKENS.
    messages = thread(["a1", "a2", "a3", "a4"])
    for i, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            messages[i] = message.model_copy(update={"content": "x" * 20_000})
    assert count_tokens_approximately(messages) > CONTEXT_TRIGGER_TOKENS

    our_edit().apply(messages, count_tokens=count_tokens_approximately)

    assert tool_contents(messages)[0] == CLEARED_PLACEHOLDER
    assert tool_contents(messages)[1:] == ["x" * 20_000] * 3


def test_short_thread_with_approximate_counter_untouched() -> None:
    messages = thread(["a1", "a2", "a3", "a4", "a5"])
    assert count_tokens_approximately(messages) < CONTEXT_TRIGGER_TOKENS
    before = [m.model_copy() for m in messages]

    our_edit().apply(messages, count_tokens=count_tokens_approximately)

    assert messages == before


# What a cleared message looks like


def test_cleared_message_keeps_id_and_name_and_is_marked() -> None:
    messages = thread(["a1", "a2", "a3", "a4"])

    our_edit().apply(messages, count_tokens=OVER)

    cleared = messages[2]
    assert isinstance(cleared, ToolMessage)
    assert cleared.content == CLEARED_PLACEHOLDER
    assert cleared.tool_call_id == "call-a1"
    assert cleared.name == "get_article"
    assert cleared.response_metadata["context_editing"] == {
        "cleared": True,
        "strategy": "clear_tool_uses",
    }


def test_tool_call_args_are_kept() -> None:
    # clear_tool_inputs is off: the model still sees which article it asked for.
    messages = thread(["a1", "a2", "a3", "a4"])

    our_edit().apply(messages, count_tokens=OVER)

    ai_call = messages[1]
    assert isinstance(ai_call, AIMessage)
    assert ai_call.tool_calls[0]["args"] == {"article_id": "a1"}


def test_non_tool_messages_untouched() -> None:
    messages = thread(["a1", "a2", "a3", "a4", "a5"])
    before = [m for m in messages if not isinstance(m, ToolMessage)]

    our_edit().apply(messages, count_tokens=OVER)

    assert [m for m in messages if not isinstance(m, ToolMessage)] == before


# Edge cases


def test_keep_or_fewer_results_nothing_cleared() -> None:
    messages = thread(["a1", "a2", "a3"])
    before = [m.model_copy() for m in messages]

    our_edit().apply(messages, count_tokens=OVER)

    assert messages == before


def test_applying_twice_is_stable() -> None:
    messages = thread(["a1", "a2", "a3", "a4", "a5"])
    our_edit().apply(messages, count_tokens=OVER)
    once = [m.model_copy() for m in messages]

    our_edit().apply(messages, count_tokens=OVER)

    assert messages == once
    assert sum(is_cleared(m) for m in messages) == 2


def test_parallel_tool_calls_matched_by_id() -> None:
    # One AI message calls two tools; both results are older than the last three kept.
    parallel: list[AnyMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_news", "args": {"query": "q"}, "id": "p1"},
                {"name": "get_latest_articles", "args": {"days": 7}, "id": "p2"},
            ],
        ),
        ToolMessage(content='[{"article_id": "s1"}]', name="search_news", tool_call_id="p1"),
        ToolMessage(
            content='[{"article_id": "l1"}]', name="get_latest_articles", tool_call_id="p2"
        ),
    ]
    messages = [HumanMessage(content="q"), *parallel, *thread(["a1", "a2", "a3"])[1:]]

    our_edit().apply(messages, count_tokens=OVER)

    assert tool_contents(messages)[:2] == [CLEARED_PLACEHOLDER, CLEARED_PLACEHOLDER]
    assert sum(is_cleared(m) for m in messages) == 2


def test_tool_message_without_matching_call_is_not_cleared() -> None:
    orphan = ToolMessage(content='{"article_id": "o1"}', name="get_article", tool_call_id="none")
    messages = [HumanMessage(content="q"), AIMessage(content=""), orphan]
    messages += thread(["a1", "a2", "a3"])[1:]

    our_edit().apply(messages, count_tokens=OVER)

    assert messages[2] == orphan


# Interaction with citations


def test_cleared_results_lose_their_ids() -> None:
    # The placeholder is not JSON, so retrieved_ids finds nothing in it.
    messages = thread(["a1", "a2", "a3", "a4", "a5"])

    our_edit().apply(messages, count_tokens=OVER)

    assert retrieved_ids(messages) == {"a3", "a4", "a5"}


class FakeRequest:
    """The two things ContextEditingMiddleware uses from a ModelRequest with a token counter."""

    def __init__(self, messages: list[AnyMessage]) -> None:
        self.messages = messages

    def override(self, *, messages: list[AnyMessage]) -> "FakeRequest":
        return FakeRequest(messages)


def test_middleware_edits_a_copy_not_the_thread_state() -> None:
    state_messages = thread(["a1", "a2", "a3", "a4", "a5"])
    seen: list[FakeRequest] = []
    middleware = ContextEditingMiddleware(edits=[our_edit()], token_counter=OVER)

    def handler(request: FakeRequest) -> AIMessage:
        seen.append(request)
        return AIMessage(content="ok")

    middleware.wrap_model_call(request=FakeRequest(state_messages), handler=handler)  # type: ignore[arg-type]

    # The model saw two cleared results ...
    [sent] = seen
    assert tool_contents(sent.messages)[:2] == [CLEARED_PLACEHOLDER, CLEARED_PLACEHOLDER]
    # ... but the thread state keeps every result, so citations of a1 and a2 still validate.
    assert CLEARED_PLACEHOLDER not in tool_contents(state_messages)
    assert retrieved_ids(state_messages) == {"a1", "a2", "a3", "a4", "a5"}
