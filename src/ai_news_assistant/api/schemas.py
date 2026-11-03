"""Pydantic models for chat request and response."""

from typing import Annotated

from pydantic import BaseModel, StringConstraints

from ai_news_assistant.models import Answer


class ChatRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    thread_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")] | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: Answer
