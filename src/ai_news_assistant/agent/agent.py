from typing import Any

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph

from ai_news_assistant.agent.config import CHAT_MODEL
from ai_news_assistant.agent.prompts import SYSTEM_PROMPT
from ai_news_assistant.agent.tools import (
    create_get_article_tool,
    create_get_latest_articles_tool,
    create_search_tool,
)
from ai_news_assistant.models import ModelAnswer
from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.config import (
    DEFAULT_CHUNKING_CONFIG,
    DEFAULT_DB,
    DEFAULT_VECTOR_STORE,
)
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever
from ai_news_assistant.rag.vector_retriever import VectorRetriever


def build_agent() -> CompiledStateGraph[Any, None, Any, Any]:
    """Build the agent using the tool factories."""
    vector_retriever = VectorRetriever(
        vector_store=DEFAULT_VECTOR_STORE, config_id=DEFAULT_CHUNKING_CONFIG
    )

    bm25_retriever = BM25Retriever(db=DEFAULT_DB, config_id=DEFAULT_CHUNKING_CONFIG)

    retriever = HybridRetriever([vector_retriever, bm25_retriever])

    tools = [
        create_search_tool(retriever=retriever, store_path=DEFAULT_DB),
        create_get_article_tool(store_path=DEFAULT_DB),
        create_get_latest_articles_tool(store_path=DEFAULT_DB),
    ]

    return create_agent(
        model=ChatOpenAI(
            model=CHAT_MODEL,
            reasoning={"effort": "medium"},
        ),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[ModelAnswer])),
        response_format=ModelAnswer,
    )


def main() -> None:
    agent = build_agent()
    thread_config: RunnableConfig = {"configurable": {"thread_id": "efe_1"}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="What does new OpenAI personal agent do?")]},
        config=thread_config,
    )
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
