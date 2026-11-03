import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ai_news_assistant.agent.agent import build_agent, build_retriever
from ai_news_assistant.agent.answer import aask
from ai_news_assistant.api.schemas import ChatRequest, ChatResponse
from ai_news_assistant.api.settings import preflight
from ai_news_assistant.rag.config import DEFAULT_DB

logger = logging.getLogger(__name__)


# App
def create_app(
    agent: CompiledStateGraph[Any, None, Any, Any] | None = None, store_path: Path = DEFAULT_DB
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if agent is None:
            load_dotenv()
            preflight(store_path)
            app.state.agent = build_agent(retriever=build_retriever(), store_path=store_path)
        else:
            app.state.agent = agent
        yield

    app = FastAPI(lifespan=lifespan)

    # Endpoints
    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        thread_id = req.thread_id or uuid4().hex
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        try:
            answer = await aask(
                agent=request.app.state.agent,
                question=req.question,
                config=config,
                store_path=store_path,
            )
        except Exception:
            logger.error("chat failed", extra={"thread_id": thread_id}, exc_info=True)
            raise HTTPException(status_code=500, detail="The assistant could not answer.") from None
        return ChatResponse(thread_id=thread_id, answer=answer)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
