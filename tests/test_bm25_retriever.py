from pathlib import Path

import pytest
from conftest import AI_ACT_FIRST_CHUNK, CONFIG_ID, make_chunks

from ai_news_assistant.rag.base import Retriever
from ai_news_assistant.rag.bm25_retriever import BM25Retriever, simple_tokenize

# Every query term is in at most 2 of the 6 chunks, so BM25Okapi gives it a positive IDF.
QUERY = "EU lawmakers approve rules for artificial intelligence"
MULTI_QUERY = "AI Act iPhone phone"
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
