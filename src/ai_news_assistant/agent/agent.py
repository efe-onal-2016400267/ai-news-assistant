from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph

from ai_news_assistant.agent.config import (
    CHAT_MODEL,
    CLEARED_PLACEHOLDER,
    CONTEXT_TRIGGER_TOKENS,
    KEEP_TOOL_RESULTS,
    default_model_kwargs,
)
from ai_news_assistant.agent.prompts import SYSTEM_PROMPT
from ai_news_assistant.agent.tools import (
    create_get_article_tool,
    create_get_latest_articles_tool,
    create_search_tool,
)
from ai_news_assistant.models import ModelAnswer
from ai_news_assistant.rag.base import Retriever
from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.config import (
    DEFAULT_CHUNKING_CONFIG,
    DEFAULT_DB,
    DEFAULT_VECTOR_STORE,
)
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever
from ai_news_assistant.rag.vector_retriever import VectorRetriever


def build_retriever() -> Retriever:
    """Build HybridRetriever"""
    vector_retriever = VectorRetriever(
        vector_store=DEFAULT_VECTOR_STORE, config_id=DEFAULT_CHUNKING_CONFIG
    )

    bm25_retriever = BM25Retriever(db=DEFAULT_DB, config_id=DEFAULT_CHUNKING_CONFIG)

    return HybridRetriever([vector_retriever, bm25_retriever])


def build_agent(
    retriever: Retriever,
    model: str = CHAT_MODEL,
    model_kwargs: dict[str, Any] | None = None,
    trigger_tokens: int = CONTEXT_TRIGGER_TOKENS,
    store_path: Path = DEFAULT_DB,
) -> CompiledStateGraph[Any, None, Any, Any]:
    """Build the agent using the tool factories."""
    kwargs = {**default_model_kwargs(model), **(model_kwargs or {})}
    tools = [
        create_search_tool(retriever=retriever, store_path=store_path),
        create_get_article_tool(store_path=store_path),
        create_get_latest_articles_tool(store_path=store_path),
    ]

    return create_agent(
        model=init_chat_model(model, **kwargs),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[ModelAnswer])),
        response_format=ModelAnswer,
        middleware=[
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=trigger_tokens,
                        keep=KEEP_TOOL_RESULTS,
                        placeholder=CLEARED_PLACEHOLDER,
                    )
                ]
            )
        ],
    )


def main() -> None:
    load_dotenv()
    vector_retriever = VectorRetriever(
        vector_store=DEFAULT_VECTOR_STORE, config_id=DEFAULT_CHUNKING_CONFIG
    )

    bm25_retriever = BM25Retriever(db=DEFAULT_DB, config_id=DEFAULT_CHUNKING_CONFIG)

    retriever = HybridRetriever([vector_retriever, bm25_retriever])
    agent = build_agent(retriever=retriever)
    thread_config: RunnableConfig = {"configurable": {"thread_id": "efe_1"}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="What does new OpenAI personal agent do?")]},
        config=thread_config,
    )
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
