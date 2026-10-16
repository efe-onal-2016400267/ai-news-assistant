import uuid

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ai_news_assistant.agent.agent import build_agent


def main() -> None:
    load_dotenv()
    agent = build_agent()
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

        result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        print(f"\nAssistant> {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
