import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from ai_news_assistant.agent.agent import build_agent
from ai_news_assistant.agent.answer import ask, render
from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.config import (
    DEFAULT_CHUNKING_CONFIG,
    DEFAULT_DB,
    DEFAULT_VECTOR_STORE,
)
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever
from ai_news_assistant.rag.vector_retriever import VectorRetriever


def print_result_messages(results: dict[str, Any]) -> None:
    print("------------------ DEBUG MESSAGES PRINT STARTS ------------------")
    for message in results["messages"]:
        message.pretty_print()
    print("------------------ DEBUG MESSAGES PRINT ENDS ------------------")


def main() -> None:
    load_dotenv()
    vector_retriever = VectorRetriever(
        vector_store=DEFAULT_VECTOR_STORE, config_id=DEFAULT_CHUNKING_CONFIG
    )

    bm25_retriever = BM25Retriever(db=DEFAULT_DB, config_id=DEFAULT_CHUNKING_CONFIG)

    retriever = HybridRetriever([vector_retriever, bm25_retriever])
    agent = build_agent(
        retriever=retriever,
    )
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("News assistant. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        answer = ask(
            agent=agent,
            question=question,
            config=config,
            store_path=DEFAULT_DB,
        )

        text_with_source = render(answer=answer)
        print(f"\nAssistant> {text_with_source}")


if __name__ == "__main__":
    main()
