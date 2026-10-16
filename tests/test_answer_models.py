from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from ai_news_assistant.agent.answer import ModelAnswer
from ai_news_assistant.models import Answer, Source

# The store writes dates as UTC ISO text (SQLiteStore._to_utc_text) and RagStore returns them as is.
STORE_DATE = "2026-10-10T09:30:00+00:00"


def source(**overrides: Any) -> Source:
    fields: dict[str, Any] = {
        "marker": 1,
        "article_id": "id-a",
        "title": "Mistral releases Large 3",
        "source": "The Decoder",
        "published_at": STORE_DATE,
    }
    return Source(**(fields | overrides))


# ModelAnswer: what the LLM returns, parsed but not trusted


def test_model_answer_with_citations() -> None:
    answer = ModelAnswer(
        text="Mistral released Large 3 [1], available via API [2].",
        citations=["id-a", "id-b"],
        grounded=True,
    )

    # Order is the contract: citations[0] backs [1], citations[1] backs [2].
    assert answer.citations == ["id-a", "id-b"]
    assert answer.grounded is True


def test_model_answer_citations_default_to_empty() -> None:
    answer = ModelAnswer(text="I do not know.", grounded=False)

    assert answer.citations == []


def test_model_answer_parses_from_json() -> None:
    raw = '{"text": "Large 3 is out [1].", "citations": ["id-a"], "grounded": true}'

    answer = ModelAnswer.model_validate_json(raw)

    assert answer == ModelAnswer(text="Large 3 is out [1].", citations=["id-a"], grounded=True)


def test_citation_ids_are_stripped() -> None:
    answer = ModelAnswer(text="x [1] [2]", citations=[" id-a ", "id-b\n"], grounded=True)

    # Stripped, so they compare equal to the ids the tools returned.
    assert answer.citations == ["id-a", "id-b"]


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_empty_or_blank_citation_id_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        ModelAnswer(text="x [1]", citations=[bad_id], grounded=True)


def test_grounded_is_required() -> None:
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate({"text": "x", "citations": []})


def test_model_answer_schema() -> None:
    # This JSON schema is what the model reads when asked for structured output.
    schema = ModelAnswer.model_json_schema()
    properties = schema["properties"]

    assert schema["required"] == ["text", "grounded"]
    for name in ("text", "citations", "grounded"):
        assert properties[name]["description"].strip()
    assert properties["citations"]["items"] == {"type": "string", "minLength": 1}
    assert "first id backs marker [1]" in properties["citations"]["description"]


# Source: built from the store, never by the model


def test_source_from_store_iso_text() -> None:
    result = source(published_at=STORE_DATE)

    assert result.published_at == datetime(2026, 10, 10, 9, 30, tzinfo=UTC)
    assert result.published_at.tzinfo is not None


def test_source_published_at_none_accepted() -> None:
    result = source(published_at=None)

    assert result.published_at is None


def test_source_published_at_defaults_to_none() -> None:
    result = Source(marker=1, article_id="id-a", title="Title", source="Verge")

    assert result.published_at is None


@pytest.mark.parametrize("marker", [0, -1])
def test_source_marker_below_1_rejected(marker: int) -> None:
    with pytest.raises(ValidationError):
        source(marker=marker)


def test_source_marker_1_accepted() -> None:
    assert source(marker=1).marker == 1


@pytest.mark.parametrize("field", ["article_id", "title"])
@pytest.mark.parametrize("value", ["", "   "])
def test_source_blank_article_id_or_title_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        source(**{field: value})


@pytest.mark.parametrize("naive", [datetime(2026, 10, 10, 9, 30), "2026-10-10T09:30:00"])
def test_source_naive_datetime_rejected(naive: datetime | str) -> None:
    with pytest.raises(ValidationError):
        source(published_at=naive)


# Answer: what ask() returns to the CLI, API and evals


def test_answer_without_sources() -> None:
    # The "no evidence" shape, also the fallback when structured_response is None.
    answer = Answer(text="I do not know.", grounded=False)

    assert answer.sources == []
    assert answer.grounded is False


def test_answer_with_sources_keeps_order() -> None:
    # [2] was dropped by the check; markers are not renumbered, so 1 and 3 remain.
    first = source(marker=1, article_id="id-a")
    third = source(marker=3, article_id="id-c")

    answer = Answer(text="Claim [1]. Other claim [3].", sources=[first, third], grounded=True)

    assert [s.marker for s in answer.sources] == [1, 3]
    assert [s.article_id for s in answer.sources] == ["id-a", "id-c"]


def test_answer_grounded_is_required() -> None:
    with pytest.raises(ValidationError):
        Answer.model_validate({"text": "x", "sources": []})
