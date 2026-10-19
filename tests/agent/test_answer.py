from datetime import UTC, datetime
from pathlib import Path

from ai_news_assistant.agent.answer import render
from ai_news_assistant.agent.citations import resolve_sources
from ai_news_assistant.models import Answer, Source

AI_ACT = Source(
    marker=1,
    article_id="ai-act",
    title="Title of ai-act",
    source="Guardian",
    published_at=datetime(2026, 10, 10, 9, 30, tzinfo=UTC),
)
IPHONE = Source(marker=2, article_id="iphone", title="Title of iphone", source="Verge")

# 2026-10-10 09:30 UTC -> "10 Oct 2026"; no date -> "Date not found".
EXPECTED = (
    "The AI Act passed [1]. A new iPhone [2].\n"
    "[1]: Title of ai-act, Guardian, 10 Oct 2026\n"
    "[2]: Title of iphone, Verge, Date not found\n"
)


def answer(sources: list[Source], text: str = "The AI Act passed [1]. A new iPhone [2].") -> Answer:
    return Answer(text=text, sources=sources, grounded=bool(sources))


def test_render_no_sources_returns_text_unchanged() -> None:
    assert render(answer([], text="I do not know.")) == "I do not know."


def test_render_hand_checked_two_sources() -> None:
    assert render(answer([AI_ACT, IPHONE])) == EXPECTED


def test_render_single_digit_day_not_padded() -> None:
    source = AI_ACT.model_copy(update={"published_at": datetime(2026, 10, 5, tzinfo=UTC)})

    assert "Guardian, 5 Oct 2026\n" in render(answer([source], text="A [1]."))


def test_render_never_prints_article_id() -> None:
    source = AI_ACT.model_copy(update={"article_id": "secret-id-123"})

    assert "secret-id-123" not in render(answer([source], text="A [1]."))


def test_render_keeps_marker_gap() -> None:
    third = IPHONE.model_copy(update={"marker": 3})

    lines = render(answer([AI_ACT, third], text="A [1]. C [3].")).splitlines()

    assert lines[1].startswith("[1]: ")
    assert lines[2].startswith("[3]: ")


def test_render_end_to_end_with_store(rag_db: Path) -> None:
    sources = resolve_sources({1: "ai-act", 2: "iphone"}, rag_db)

    assert render(answer(sources)) == EXPECTED
