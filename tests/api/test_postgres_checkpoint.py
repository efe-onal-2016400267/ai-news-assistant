"""Checkpoint round-trips against a real Postgres.

Skipped unless POSTGRES_USER, POSTGRES_PASSWORD and POSTGRES_DB are set (for example with
`docker compose up -d postgres` and a .env). Uses a small graph, not the agent, so no LLM runs.
"""

import asyncio
import operator
import os
from typing import Annotated, Any, TypedDict
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from ai_news_assistant.api.settings import db_uri
from ai_news_assistant.models import ModelAnswer

POSTGRES_VARS = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in POSTGRES_VARS),
    reason="needs POSTGRES_USER, POSTGRES_PASSWORD and POSTGRES_DB",
)


class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]


class AnswerState(TypedDict, total=False):
    answer: ModelAnswer


def chat_graph(saver: AsyncPostgresSaver) -> Any:
    """Appends one reply per turn, so the history length shows what was remembered."""

    async def reply(state: ChatState) -> dict[str, list[str]]:
        return {"messages": [f"reply to {state['messages'][-1]}"]}

    builder = StateGraph(ChatState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


def answer_graph(saver: AsyncPostgresSaver) -> Any:
    async def answer(_: AnswerState) -> dict[str, ModelAnswer]:
        return {"answer": ModelAnswer(text="A [1].", citations=["ai-act"], grounded=True)}

    builder = StateGraph(AnswerState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=saver)


def open_saver() -> Any:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[ModelAnswer])
    return AsyncPostgresSaver.from_conn_string(db_uri(), serde=serde)


def test_conversation_is_remembered_by_a_new_connection() -> None:
    config: RunnableConfig = {"configurable": {"thread_id": f"test-{uuid4().hex}"}}

    async def run() -> list[str]:
        # First turn on one connection, then the connection closes.
        async with open_saver() as first:
            await first.setup()
            await chat_graph(first).ainvoke({"messages": ["hello"]}, config=config)

        # Second turn on a fresh connection must see the first turn.
        async with open_saver() as second:
            await chat_graph(second).ainvoke({"messages": ["again"]}, config=config)
            snapshot = await chat_graph(second).aget_state(config)
        return list(snapshot.values["messages"])

    history = asyncio.run(run())

    assert history == ["hello", "reply to hello", "again", "reply to again"]


def test_structured_answer_comes_back_as_model_answer() -> None:
    config: RunnableConfig = {"configurable": {"thread_id": f"test-{uuid4().hex}"}}

    async def run() -> Any:
        async with open_saver() as writer:
            await writer.setup()
            await answer_graph(writer).ainvoke({}, config=config)

        async with open_saver() as reader:
            return await answer_graph(reader).aget_state(config)

    snapshot = asyncio.run(run())

    restored = snapshot.values["answer"]
    assert isinstance(restored, ModelAnswer)
    assert restored == ModelAnswer(text="A [1].", citations=["ai-act"], grounded=True)
