import json
import logging
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph.state import CompiledStateGraph

import ai_news_assistant.api.app as app_module
from ai_news_assistant.api.app import create_app
from ai_news_assistant.api.settings import REQUIRED_ENV
from ai_news_assistant.models import ModelAnswer

APP_LOGGER = "ai_news_assistant.api.app"
FAILED_DETAIL = "The assistant could not answer."


class FakeAgent:
    """Stands in for the compiled graph: returns a fixed result or raises, and records calls."""

    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((payload, config))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def search_message(article_ids: list[str]) -> ToolMessage:
    """A search_news result, shaped the way the agent's tool returns it."""
    items = [{"article_id": article_id} for article_id in article_ids]
    return ToolMessage(content=json.dumps(items), name="search_news", tool_call_id="call-1")


def model_result(
    text: str,
    citations: list[str],
    grounded: bool,
    retrieved: list[str],
) -> dict[str, Any]:
    messages: list[AnyMessage] = [
        HumanMessage(content="question"),
        search_message(retrieved),
        AIMessage(content=text),
    ]
    structured = ModelAnswer(text=text, citations=citations, grounded=grounded)
    return {"messages": messages, "structured_response": structured}


def plain_result(text: str) -> dict[str, Any]:
    """The agent finished without structured output."""
    return {"messages": [HumanMessage(content="question"), AIMessage(content=text)]}


@pytest.fixture
def client(rag_db: Path) -> Iterator[TestClient]:
    """Client with a fake agent that answers from the rag_db corpus; it is replaced per test."""
    with TestClient(create_app(agent=FakeAgent(), store_path=rag_db)) as test_client:
        yield test_client


def make_client(agent: FakeAgent, store_path: Path) -> TestClient:
    """Client for a test that needs its own agent; the caller enters it as a context manager."""
    return TestClient(
        create_app(
            agent=cast(CompiledStateGraph[Any, None, Any, Any], agent), store_path=store_path
        )
    )


def chat(client: TestClient, question: str = "What happened?", **extra: Any) -> Any:
    return client.post("/chat", json={"question": question, **extra})


# Happy path


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_grounded_answer_returns_sources_from_the_store(rag_db: Path) -> None:
    agent = FakeAgent(model_result("The AI Act passed [1].", ["ai-act"], True, ["ai-act"]))

    with make_client(agent, rag_db) as client:
        response = chat(client, thread_id="thread-1")

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "thread-1"
    assert body["answer"]["text"] == "The AI Act passed [1]."
    assert body["answer"]["grounded"] is True
    assert body["answer"]["sources"] == [
        {
            "marker": 1,
            "article_id": "ai-act",
            "title": "Title of ai-act",
            "source": "Guardian",
            "published_at": body["answer"]["sources"][0]["published_at"],
        }
    ]
    # 2026-10-10 09:30 UTC, as stored in the corpus.
    published = datetime.fromisoformat(body["answer"]["sources"][0]["published_at"])
    assert published == datetime(2026, 10, 10, 9, 30, tzinfo=UTC)


def test_missing_thread_id_is_generated_and_passed_to_the_agent(rag_db: Path) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        response = chat(client)

    thread_id = response.json()["thread_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", thread_id)
    assert agent.calls[0][1] == {"configurable": {"thread_id": thread_id}}


def test_same_thread_id_is_passed_on_every_call(rag_db: Path) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        first = chat(client, thread_id="thread-1")
        second = chat(client, "And then?", thread_id="thread-1")

    assert first.json()["thread_id"] == "thread-1"
    assert second.json()["thread_id"] == "thread-1"
    assert [config["configurable"]["thread_id"] for _, config in agent.calls] == [
        "thread-1",
        "thread-1",
    ]


def test_question_is_stripped_before_the_agent_sees_it(rag_db: Path) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        chat(client, "  What happened?  ")

    message = agent.calls[0][0]["messages"][0]
    assert message.content == "What happened?"


# Model answers that are checked before they reach the user


def test_unretrieved_citation_is_dropped_and_its_marker_removed(rag_db: Path) -> None:
    # "B [2]." loses " [2]" -> "B." (the marker and the space before it).
    # iphone was never retrieved, so it is dropped; only ai-act is kept.
    agent = FakeAgent(
        model_result("A [1]. B [2].", ["ai-act", "iphone"], True, retrieved=["ai-act"])
    )

    with make_client(agent, rag_db) as client:
        answer = chat(client).json()["answer"]

    assert answer["text"] == "A [1]. B."
    assert answer["grounded"] is True
    assert [source["article_id"] for source in answer["sources"]] == ["ai-act"]


def test_grounded_claim_with_no_surviving_source_is_not_grounded(rag_db: Path) -> None:
    agent = FakeAgent(model_result("Claim [1].", ["made-up"], True, retrieved=["ai-act"]))

    with make_client(agent, rag_db) as client:
        answer = chat(client).json()["answer"]

    assert answer == {"text": "Claim.", "sources": [], "grounded": False}


def test_missing_structured_response_answers_without_sources(rag_db: Path) -> None:
    agent = FakeAgent(plain_result("I cannot help with that."))

    with make_client(agent, rag_db) as client:
        answer = chat(client).json()["answer"]

    assert answer == {"text": "I cannot help with that.", "sources": [], "grounded": False}


# Request validation


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_is_rejected_before_the_agent_runs(rag_db: Path, question: str) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        response = chat(client, question)

    assert response.status_code == 422
    assert agent.calls == []


def test_question_at_the_length_limit_is_accepted(rag_db: Path) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        response = chat(client, "x" * 2000)

    assert response.status_code == 200


def test_question_over_the_length_limit_is_rejected(rag_db: Path) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        response = chat(client, "x" * 2001)

    assert response.status_code == 422
    assert agent.calls == []


@pytest.mark.parametrize("thread_id", ["has space", "bad/id", "x" * 65, ""])
def test_invalid_thread_id_is_rejected(rag_db: Path, thread_id: str) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        response = chat(client, thread_id=thread_id)

    assert response.status_code == 422
    assert agent.calls == []


def test_thread_id_at_the_length_limit_is_accepted(rag_db: Path) -> None:
    agent = FakeAgent(plain_result("Hello."))

    with make_client(agent, rag_db) as client:
        response = chat(client, thread_id="x" * 64)

    assert response.status_code == 200
    assert response.json()["thread_id"] == "x" * 64


def test_chat_only_accepts_post(client: TestClient) -> None:
    assert client.get("/chat").status_code == 405


# Errors from the agent


def test_agent_failure_returns_generic_500_without_internal_detail(
    rag_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    question = "private question text 123"
    agent = FakeAgent(error=RuntimeError("internal secret detail"))

    with caplog.at_level(logging.ERROR, logger=APP_LOGGER):
        with make_client(agent, rag_db) as client:
            response = chat(client, question)

    assert response.status_code == 500
    assert response.json() == {"detail": FAILED_DETAIL}
    assert "internal secret detail" not in response.text
    assert "chat failed" in caplog.text
    assert question not in caplog.text


# App startup


def test_injected_agent_is_used_without_building_one(
    rag_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**_: Any) -> None:
        raise AssertionError("build_agent must not run when an agent is injected")

    monkeypatch.setattr(app_module, "build_agent", fail)
    agent = FakeAgent()

    app: FastAPI = create_app(agent=cast(Any, agent), store_path=rag_db)
    with TestClient(app):
        assert app.state.agent is agent


class FakeSaver:
    """Stands in for AsyncPostgresSaver: records setup and reports it to the spy."""

    def __init__(self, spy: "SaverSpy") -> None:
        self.spy = spy

    async def setup(self) -> None:
        self.spy.events.append("setup")
        if self.spy.setup_error is not None:
            raise self.spy.setup_error


class SaverSpy:
    """Replaces the AsyncPostgresSaver class in app.py; records open, setup and close in order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.conn_strings: list[str] = []
        self.serdes: list[Any] = []
        self.setup_error: Exception | None = None
        self.saver: FakeSaver | None = None

    def from_conn_string(
        self, conn_string: str, *, serde: Any = None
    ) -> AbstractAsyncContextManager[FakeSaver]:
        self.conn_strings.append(conn_string)
        self.serdes.append(serde)
        return self._open()

    @asynccontextmanager
    async def _open(self) -> AsyncIterator[FakeSaver]:
        self.saver = FakeSaver(self)
        self.events.append("open")
        try:
            yield self.saver
        finally:
            self.events.append("close")


@pytest.fixture
def saver_spy(monkeypatch: pytest.MonkeyPatch) -> SaverSpy:
    spy = SaverSpy()
    monkeypatch.setattr(app_module, "AsyncPostgresSaver", spy)
    return spy


@pytest.fixture
def startup_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Valid settings and an existing db file; the real .env and retriever stay out of the test."""
    db = tmp_path / "news.db"
    db.touch()
    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "test-value")
    monkeypatch.setenv("POSTGRES_USER", "efe")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss")
    monkeypatch.setenv("POSTGRES_DB", "news")
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.setattr(app_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(app_module, "build_retriever", lambda: object())
    return db


@pytest.fixture
def build_calls(monkeypatch: pytest.MonkeyPatch, saver_spy: SaverSpy) -> list[dict[str, Any]]:
    """Replaces build_agent; records its kwargs and the event order, returns a sentinel agent."""
    calls: list[dict[str, Any]] = []

    def record_build_agent(**kwargs: Any) -> object:
        calls.append(kwargs)
        saver_spy.events.append("build")
        return built_agent

    built_agent = object()
    monkeypatch.setattr(app_module, "build_agent", record_build_agent)
    return calls


def test_startup_builds_the_agent_with_the_postgres_checkpointer(
    startup_env: Path,
    saver_spy: SaverSpy,
    build_calls: list[dict[str, Any]],
) -> None:
    app = create_app(store_path=startup_env)

    with TestClient(app):
        assert app.state.agent is not None
        assert saver_spy.events == ["open", "setup", "build"]

    assert len(build_calls) == 1
    assert build_calls[0]["store_path"] == startup_env
    assert build_calls[0]["checkpointer"] is saver_spy.saver
    assert saver_spy.conn_strings == ["postgresql://efe:p%40ss@localhost:5432/news"]


def test_startup_closes_the_saver_on_shutdown(
    startup_env: Path, saver_spy: SaverSpy, build_calls: list[dict[str, Any]]
) -> None:
    with TestClient(create_app(store_path=startup_env)):
        assert saver_spy.events == ["open", "setup", "build"]

    assert saver_spy.events == ["open", "setup", "build", "close"]


def test_startup_fails_and_builds_nothing_when_setup_fails(
    startup_env: Path,
    saver_spy: SaverSpy,
    build_calls: list[dict[str, Any]],
) -> None:
    saver_spy.setup_error = RuntimeError("tables could not be created")

    with pytest.raises(RuntimeError, match="tables could not be created"):
        with TestClient(create_app(store_path=startup_env)):
            pass

    assert build_calls == []
    assert saver_spy.events == ["open", "setup", "close"]


def test_checkpointer_serializer_round_trips_a_model_answer(
    startup_env: Path,
    saver_spy: SaverSpy,
    build_calls: list[dict[str, Any]],
) -> None:
    # The serde given to the saver must rebuild ModelAnswer, or stored answers come back as dicts.
    with TestClient(create_app(store_path=startup_env)):
        serde = saver_spy.serdes[0]
        answer = ModelAnswer(text="A [1].", citations=["ai-act"], grounded=True)

        restored = serde.loads_typed(serde.dumps_typed(answer))

    assert isinstance(serde, JsonPlusSerializer)
    assert restored == answer


def test_injected_agent_does_not_open_the_postgres_saver(rag_db: Path, saver_spy: SaverSpy) -> None:
    with make_client(FakeAgent(), rag_db):
        pass

    assert saver_spy.conn_strings == []
    assert saver_spy.events == []


def test_startup_fails_when_keys_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "news.db"
    db.touch()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "load_dotenv", lambda: None)

    def fail(**_: Any) -> None:
        raise AssertionError("build_agent must not run when preflight fails")

    monkeypatch.setattr(app_module, "build_agent", fail)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY, OPENAI_API_KEY"):
        with TestClient(create_app(store_path=db)):
            pass


def test_startup_fails_when_db_is_missing(
    startup_env: Path, saver_spy: SaverSpy, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="News db not found"):
        with TestClient(create_app(store_path=tmp_path / "absent.db")):
            pass

    assert saver_spy.conn_strings == []
