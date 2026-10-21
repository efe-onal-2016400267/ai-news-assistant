from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from conftest import AI_ACT_FIRST_CHUNK, CONFIG_ID, make_chunks

from ai_news_assistant.rag.base import ArticleMetadata, Hit, Retriever, SearchFilter
from ai_news_assistant.rag.bm25_retriever import BM25Retriever, simple_tokenize

# Every query term is in at most 2 of the 6 chunks, so BM25Okapi gives it a positive IDF.
QUERY = "EU lawmakers approve rules for artificial intelligence"
MULTI_QUERY = "AI Act iPhone phone"
ROBOTICS_QUERY = "warehouse robotics startup"
MULTI_QUERY_MATCHES = {
    f"{CONFIG_ID}:ai-act:0",
    f"{CONFIG_ID}:ai-act:1",
    f"{CONFIG_ID}:iphone:0",
    f"{CONFIG_ID}:iphone:1",
}


@pytest.fixture
def retriever(rag_db: Path) -> BM25Retriever:
    return BM25Retriever(rag_db, CONFIG_ID)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Apple's iPhone!", ["apple", "s", "iphone"]),
        ("GPT-4.5 and v1.2.3", ["gpt", "4.5", "and", "v1", "2.3"]),
        ("$799", ["799"]),
        ("", []),
        ("   ", []),
        ("?!", []),
    ],
)
def test_simple_tokenize(text: str, expected: list[str]) -> None:
    assert simple_tokenize(text) == expected


def test_is_a_retriever(retriever: BM25Retriever) -> None:
    assert isinstance(retriever, Retriever)


def test_obvious_chunk_is_top_hit(retriever: BM25Retriever) -> None:
    hits = retriever.search(QUERY, 3)

    top = hits[0]
    assert top.chunk_id == f"{CONFIG_ID}:ai-act:0"
    assert top.article_id == "ai-act"
    assert top.text == AI_ACT_FIRST_CHUNK


@pytest.mark.parametrize("k", [1, 3])
def test_k_is_respected(retriever: BM25Retriever, k: int) -> None:
    assert len(retriever.search(MULTI_QUERY, k)) == k


def test_scores_positive_and_descending(retriever: BM25Retriever) -> None:
    scores = [hit.score for hit in retriever.search(MULTI_QUERY, 100)]

    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_only_matching_chunks_returned(retriever: BM25Retriever) -> None:
    # Unlike the vector retriever, chunks that share no term with the query are dropped.
    hits = retriever.search(MULTI_QUERY, 100)

    assert {hit.chunk_id for hit in hits} == MULTI_QUERY_MATCHES


def test_only_loads_chunks_of_its_config(retriever: BM25Retriever) -> None:
    assert [c.chunk_id for c in retriever.chunks] == [c.chunk_id for c in make_chunks(CONFIG_ID)]


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty(retriever: BM25Retriever, k: int) -> None:
    assert retriever.search(QUERY, k) == []


def test_search_filter_none_matches_no_argument(retriever: BM25Retriever) -> None:
    assert retriever.search(MULTI_QUERY, 100, search_filter=None) == retriever.search(
        MULTI_QUERY, 100
    )


# Metadata lookup and matches
# Fixture: ai-act Guardian 10-10 09:30, iphone Verge undated, robotics TechCrunch 10-08 14:00,
# sourdough Food Blog 09-30 07:00 (all UTC).

ROBOTICS_TIME = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
ROBOTICS = ArticleMetadata(article_id="robotics", source="TechCrunch", published_at=ROBOTICS_TIME)
UNDATED = ArticleMetadata(article_id="iphone", source="Verge", published_at=None)


def test_metadata_lookup_parses_dates(retriever: BM25Retriever) -> None:
    assert retriever.articles_metadata["robotics"] == ROBOTICS
    assert retriever.articles_metadata["iphone"] == UNDATED
    assert set(retriever.articles_metadata) == {"ai-act", "iphone", "robotics", "sourdough"}


@pytest.mark.parametrize(
    ("search_filter", "expected"),
    [
        (SearchFilter(), True),
        (SearchFilter(sources=frozenset({"TechCrunch", "Verge"})), True),
        (SearchFilter(sources=frozenset({"Guardian"})), False),
        (SearchFilter(since=ROBOTICS_TIME), True),  # inclusive
        (SearchFilter(since=ROBOTICS_TIME + timedelta(seconds=1)), False),
        (SearchFilter(sources=frozenset({"TechCrunch"}), since=ROBOTICS_TIME), True),
        (SearchFilter(sources=frozenset({"Guardian"}), since=ROBOTICS_TIME), False),
        (
            SearchFilter(
                sources=frozenset({"TechCrunch"}), since=datetime(2026, 10, 9, tzinfo=UTC)
            ),
            False,
        ),
    ],
)
def test_matches_dated_article(
    retriever: BM25Retriever, search_filter: SearchFilter, expected: bool
) -> None:
    assert retriever.matches(ROBOTICS, search_filter) is expected


@pytest.mark.parametrize(
    ("search_filter", "expected"),
    [
        (SearchFilter(sources=frozenset({"Verge"})), True),
        (SearchFilter(since=datetime(2000, 1, 1, tzinfo=UTC)), False),
    ],
)
def test_matches_undated_article_only_without_since(
    retriever: BM25Retriever, search_filter: SearchFilter, expected: bool
) -> None:
    assert retriever.matches(UNDATED, search_filter) is expected


# Filtered search


def chunk_ids(hits: list[Hit]) -> list[str]:
    return [hit.chunk_id for hit in hits]


def test_source_only_keeps_that_source_including_undated(retriever: BM25Retriever) -> None:
    hits = retriever.search(
        MULTI_QUERY, 100, search_filter=SearchFilter(sources=frozenset({"Verge"}))
    )

    assert set(chunk_ids(hits)) == {f"{CONFIG_ID}:iphone:0", f"{CONFIG_ID}:iphone:1"}


def test_date_only_drops_older_and_undated(retriever: BM25Retriever) -> None:
    # MULTI_QUERY matches ai-act (10-10) and iphone (undated); since 10-08 keeps ai-act only.
    hits = retriever.search(
        MULTI_QUERY, 100, search_filter=SearchFilter(since=datetime(2026, 10, 8, tzinfo=UTC))
    )

    assert set(chunk_ids(hits)) == {f"{CONFIG_ID}:ai-act:0", f"{CONFIG_ID}:ai-act:1"}


def test_source_and_date_combined(retriever: BM25Retriever) -> None:
    search_filter = SearchFilter(
        sources=frozenset({"Guardian", "Verge"}), since=datetime(2026, 10, 8, tzinfo=UTC)
    )

    hits = retriever.search(MULTI_QUERY, 100, search_filter=search_filter)

    assert set(chunk_ids(hits)) == {f"{CONFIG_ID}:ai-act:0", f"{CONFIG_ID}:ai-act:1"}


def test_filter_matching_nothing_returns_empty(retriever: BM25Retriever) -> None:
    # robotics (TechCrunch) shares no term with MULTI_QUERY.
    hits = retriever.search(
        MULTI_QUERY, 100, search_filter=SearchFilter(sources=frozenset({"TechCrunch"}))
    )

    assert hits == []


def test_filter_applies_before_top_k_cut(retriever: BM25Retriever) -> None:
    unfiltered_top = retriever.search(MULTI_QUERY, 1)[0]
    assert unfiltered_top.article_id == "iphone"  # precondition: Guardian is not top unfiltered

    hits = retriever.search(
        MULTI_QUERY, 1, search_filter=SearchFilter(sources=frozenset({"Guardian"}))
    )

    # Cutting to k=1 before filtering would leave nothing; filtering first gives ai-act.
    assert [hit.article_id for hit in hits] == ["ai-act"]


def test_filter_keeps_unfiltered_scores_and_order(retriever: BM25Retriever) -> None:
    unfiltered = retriever.search(MULTI_QUERY, 100)

    hits = retriever.search(
        MULTI_QUERY, 100, search_filter=SearchFilter(sources=frozenset({"Verge"}))
    )

    assert hits == [hit for hit in unfiltered if hit.article_id == "iphone"]


def test_empty_search_filter_is_no_filter(retriever: BM25Retriever) -> None:
    assert retriever.search(MULTI_QUERY, 100, search_filter=SearchFilter()) == retriever.search(
        MULTI_QUERY, 100
    )


def test_since_boundary_is_inclusive_across_timezones(retriever: BM25Retriever) -> None:
    # 16:00 at +02:00 is 14:00 UTC, the robotics publication time.
    plus_two = timezone(timedelta(hours=2))
    at = SearchFilter(since=datetime(2026, 10, 8, 16, 0, tzinfo=plus_two))
    after = SearchFilter(since=datetime(2026, 10, 8, 16, 0, 1, tzinfo=plus_two))

    assert {h.article_id for h in retriever.search(ROBOTICS_QUERY, 10, search_filter=at)} == {
        "robotics"
    }
    assert retriever.search(ROBOTICS_QUERY, 10, search_filter=after) == []


@pytest.mark.parametrize("query", ["quantum", "?!", ""])
def test_query_without_known_terms_returns_empty(retriever: BM25Retriever, query: str) -> None:
    assert retriever.search(query, 5) == []


def test_query_is_case_insensitive(retriever: BM25Retriever) -> None:
    hits = retriever.search("IPHONE", 5)

    assert hits == retriever.search("iphone", 5)
    assert [hit.chunk_id for hit in hits] == [f"{CONFIG_ID}:iphone:0"]


def test_unknown_config_raises(rag_db: Path) -> None:
    with pytest.raises(ValueError, match="nope"):
        BM25Retriever(rag_db, "nope")
