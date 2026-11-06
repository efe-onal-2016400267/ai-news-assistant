import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph

from ai_news_assistant.agent.agent import build_agent
from ai_news_assistant.agent.answer import StatusEvent, aask, aask_stream
from ai_news_assistant.agent.tools import load_mcp_tools
from ai_news_assistant.api.schemas import ChatRequest, ChatResponse
from ai_news_assistant.api.settings import db_uri, preflight
from ai_news_assistant.models import ModelAnswer
from ai_news_assistant.rag.config import DEFAULT_DB

logger = logging.getLogger(__name__)


def sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def tools_url() -> str:
    return os.environ["TOOLS_URL"]


# App
def create_app(
    agent: CompiledStateGraph[Any, None, Any, Any] | None = None, store_path: Path = DEFAULT_DB
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if agent is not None:
            app.state.agent = agent
            yield
            return
        load_dotenv()
        preflight(store_path)
        async with AsyncPostgresSaver.from_conn_string(
            db_uri(), serde=JsonPlusSerializer(allowed_msgpack_modules=[ModelAnswer])
        ) as checkpointer:
            await checkpointer.setup()
            tools = await load_mcp_tools(url=tools_url())
            app.state.agent = build_agent(
                tools=tools,
                store_path=store_path,
                checkpointer=checkpointer,
            )
            yield

    app = FastAPI(lifespan=lifespan)

    # Endpoints
    # Regular chat endpoint
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

    # event streaming endpoint
    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
        thread_id = req.thread_id or uuid4().hex
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        async def events() -> AsyncIterator[str]:
            try:
                async for event in aask_stream(
                    agent=request.app.state.agent,
                    question=req.question,
                    config=config,
                    store_path=store_path,
                ):
                    if isinstance(event, StatusEvent):
                        yield sse("status", json.dumps({"tool": event.tool}))
                    else:
                        payload = ChatResponse(thread_id=thread_id, answer=event.answer)
                        yield sse("answer", payload.model_dump_json())
            except Exception as exc:
                logger.error(
                    "chat stream failed: %s", type(exc).__name__, extra={"thread_id": thread_id}
                )
                yield sse("error", json.dumps({"detail": "FAILED"}))

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
