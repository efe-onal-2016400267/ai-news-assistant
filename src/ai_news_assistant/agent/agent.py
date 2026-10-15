from langchain.agents import create_agent
from ai_news_assistant.rag.store import RagStore

from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.config import (
    DEFAULT_CHUNKING_CONFIG,
    DEFAULT_DB,
    DEFAULT_VECTOR_STORE,
)
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever
from ai_news_assistant.rag.vector_retriever import VectorRetriever
from langgraph.graph.state import CompiledStateGraph
from ai_news_assistant.agent.tools import create_search_tool
from ai_news_assistant.agent.config import CHAT_MODEL
from ai_news_assistant.agent.propmts import SYSTEM_PROMPT
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from dotenv import load_dotenv


def build_agent() -> CompiledStateGraph:
    """Build the agent using the tool factories."""
    vector_retriever = VectorRetriever(
        vector_store=DEFAULT_VECTOR_STORE,
        config_id=DEFAULT_CHUNKING_CONFIG
        )

    bm25_retriever = BM25Retriever(
        db=DEFAULT_DB,
        config_id=DEFAULT_CHUNKING_CONFIG
    )

    retriever = HybridRetriever(
        [vector_retriever, bm25_retriever]
    )

    tools = [create_search_tool(retriever=retriever, store_path=DEFAULT_DB)]

    return create_agent(
        model=ChatOpenAI(name=CHAT_MODEL, temperature=0,),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


load_dotenv()
agent = build_agent()
config = {"configurable": {"thread_id": "efe_1"}}
result = agent.invoke({"messages": [HumanMessage(content="What does new OpenAI personal agent do?")]})
for message in result["messages"]:
    message.pretty_print()
