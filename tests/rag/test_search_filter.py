import dataclasses
from datetime import UTC, datetime

import pytest

from ai_news_assistant.rag.base import SearchFilter


def test_search_filter_defaults_to_no_filtering() -> None:
    search_filter = SearchFilter()

    assert search_filter.sources is None
    assert search_filter.since is None


@pytest.mark.parametrize(
    ("kwargs", "sources", "since"),
    [
        ({"sources": frozenset({"TechCrunch"})}, frozenset({"TechCrunch"}), None),
        (
            {"since": datetime(2026, 10, 8, tzinfo=UTC)},
            None,
            datetime(2026, 10, 8, tzinfo=UTC),
        ),
    ],
)
def test_search_filter_takes_one_field_alone(
    kwargs: dict[str, object], sources: frozenset[str] | None, since: datetime | None
) -> None:
    search_filter = SearchFilter(**kwargs)  # type: ignore[arg-type]

    assert search_filter.sources == sources
    assert search_filter.since == since


def test_search_filter_is_frozen() -> None:
    search_filter = SearchFilter(sources=frozenset({"TechCrunch"}))

    with pytest.raises(dataclasses.FrozenInstanceError):
        search_filter.sources = frozenset({"Wired"})  # type: ignore[misc]
