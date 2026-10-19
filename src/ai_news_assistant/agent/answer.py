import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ai_news_assistant.agent.citations import check_citations, resolve_sources, retrieved_ids
from ai_news_assistant.models import Answer, ModelAnswer

logger = logging.getLogger(__name__)


def render(answer: Answer) -> str:
    """Add the sources at the end of the answer."""

    if not answer.sources:
        return answer.text

    text = answer.text + "\n"
    for source in answer.sources:
        text += "[" + str(source.marker) + "]: " + source.title + ", " + source.source
        if not source.published_at:
            date_text = "Date not found"
        else:
            date_text = f"{source.published_at.day} {source.published_at:%b %Y}"
        text += ", " + date_text + "\n"
    return text


def ask(
    agent: CompiledStateGraph[Any, None, Any, Any],
    question: str,
    config: RunnableConfig,
    store_path: Path,
) -> Answer:
    """Orchestrate citation logic.

    1. invoke the agent to get the initial ModelAnswer with structured output.
    2. Call `retrieve_ids` to get the article_ids from tool calls.
    3. Call `check_citation` to get cleaned text and valid citations.
    4. Call `resolve_sources` to get the proper metadata for each cited article.
    5. Return the new answer.

    Errors propagate.
    """
    # Ids, markers and counts only: answer and article text are untrusted and stay out of logs.
    thread_id = config.get("configurable", {}).get("thread_id")
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    structured_response = result.get("structured_response")
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
        return Answer(
            text=checked.clean_text,
            sources=[],
            grounded=False,
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
    return answer
