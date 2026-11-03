import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ai_news_assistant.agent.citations import (
    CheckedCitations,
    check_citations,
    resolve_sources,
    retrieved_ids,
)
from ai_news_assistant.models import Answer, ModelAnswer

STATUS_TOOLS = frozenset({"search_news", "get_article", "get_latest_articles"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatusEvent:
    tool: str


@dataclass(frozen=True)
class AnswerEvent:
    answer: Answer


@dataclass(frozen=True)
class AskResult:
    """The whole run result for eval"""

    answer: Answer
    model_answer: ModelAnswer | None  # None = no structured output
    checked: CheckedCitations | None
    messages: list[AnyMessage]  # The whole thread so far, including earlier turns


def render(answer: Answer) -> str:
    """Add the sources at the end of the answer."""

    if not answer.sources:
        return answer.text

    text = answer.text + "\n\n**Sources:**\n"
    for source in answer.sources:
        text += "[" + str(source.marker) + "]: " + source.title + ", " + source.source
        if not source.published_at:
            date_text = "Date not found"
        else:
            date_text = f"{source.published_at.day} {source.published_at:%b %Y}"
        text += ", " + date_text + "\n"
    return text


def _finish(result: dict[str, Any], config: RunnableConfig, store_path: Path) -> AskResult:
    """Finish answer processing after agent invocation.

    Get result and perform everything after that:
    2. Call `retrieve_ids` to get the article_ids from tool calls.
    3. Call `check_citation` to get cleaned text and valid citations.
    4. Call `resolve_sources` to get the proper metadata for each cited article.
    5. Return the new answer.

    Errors propagate.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    structured_response = result.get("structured_response")
    messages = result["messages"]
    if not isinstance(structured_response, ModelAnswer):
        logger.warning(
            "Thread %s: no structured response, answering without sources",
            thread_id,
            extra={"thread_id": thread_id},
        )
        no_structure = ModelAnswer(
            text=result["messages"][-1].text,
            citations=[],
            grounded=False,
        )
        checked = check_citations(no_structure, set())
        return AskResult(
            answer=Answer(text=checked.clean_text, sources=[], grounded=False),
            model_answer=None,
            checked=checked,
            messages=messages,
        )

    retrieved = retrieved_ids(result["messages"])
    checked = check_citations(model_answer=structured_response, retrieved=retrieved)
    sources = resolve_sources(valid=checked.valid_citations, store_path=store_path)
    if checked.dropped_ids or checked.dropped_markers:
        logger.warning(
            "Thread %s: dropped citations: ids %s, markers %s",
            thread_id,
            checked.dropped_ids,
            checked.dropped_markers,
            extra={
                "thread_id": thread_id,
                "dropped_ids": checked.dropped_ids,
                "dropped_markers": checked.dropped_markers,
            },
        )
    if structured_response.grounded and not sources:
        logger.info(
            "Thread %s: model said grounded but no source survived, grounded set to False",
            thread_id,
            extra={"thread_id": thread_id},
        )
    answer = Answer(
        text=checked.clean_text,
        sources=sources,
        grounded=structured_response.grounded and len(sources) > 0,
    )
    ask_result = AskResult(
        answer=answer,
        model_answer=structured_response,
        checked=checked,
        messages=messages,
    )
    return ask_result


def ask_detailed(
    agent: CompiledStateGraph[Any, None, Any, Any],
    question: str,
    config: RunnableConfig,
    store_path: Path,
) -> AskResult:
    """Invoke the agent and call _finish."""
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return _finish(result, config, store_path)


async def aask(
    agent: CompiledStateGraph[Any, None, Any, Any],
    question: str,
    config: RunnableConfig,
    store_path: Path,
) -> Answer:
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config=config)
    finished = await asyncio.to_thread(_finish, result=result, config=config, store_path=store_path)
    return finished.answer


async def aask_stream(
    agent: CompiledStateGraph[Any, None, Any, Any],
    question: str,
    config: RunnableConfig,
    store_path: Path,
) -> AsyncIterator[StatusEvent | AnswerEvent]:
    stream = await agent.astream_events(
        {"messages": [HumanMessage(content=question)]}, config=config, version="v3"
    )
    async for call in stream.tool_calls:
        if call.tool_name in STATUS_TOOLS:
            yield StatusEvent(tool=call.tool_name)
    final = await stream.output()
    if final is None:
        raise RuntimeError("Agent run finished without a final state")
    finished = await asyncio.to_thread(_finish, result=final, config=config, store_path=store_path)
    yield AnswerEvent(finished.answer)


def ask(
    agent: CompiledStateGraph[Any, None, Any, Any],
    question: str,
    config: RunnableConfig,
    store_path: Path,
) -> Answer:
    """Return answer from ask_detailed(...)."""
    return ask_detailed(agent=agent, question=question, config=config, store_path=store_path).answer
