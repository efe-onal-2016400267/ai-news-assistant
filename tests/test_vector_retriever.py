from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import chromadb
import pytest
from conftest import AI_ACT_FIRST_CHUNK, CONFIG_ID, make_chunks
from sentence_transformers import SentenceTransformer

from ai_news_assistant.rag.base import Hit, Retriever, SearchFilter
from ai_news_assistant.rag.config import BGE_QUERY_PREFIX
from ai_news_assistant.rag.vector_retriever import VectorRetriever

QUERY = "EU lawmakers approve rules for artificial intelligence"
N_CHUNKS = len(make_chunks(CONFIG_ID))


@pytest.fixture
def retriever(built_store: Path) -> VectorRetriever:
    return VectorRetriever(built_store, CONFIG_ID)


def test_is_a_retriever(retriever: VectorRetriever) -> None:
    assert isinstance(retriever, Retriever)


def test_obvious_chunk_is_top_hit(retriever: VectorRetriever) -> None:
    hits = retriever.search(QUERY, 3)

    top = hits[0]
    assert top.chunk_id == f"{CONFIG_ID}:ai-act:0"
    assert top.article_id == "ai-act"
    assert top.text == AI_ACT_FIRST_CHUNK


@pytest.mark.parametrize("k", [1, 3])
def test_k_is_respected(retriever: VectorRetriever, k: int) -> None:
    assert len(retriever.search(QUERY, k)) == k


def test_scores_descending(retriever: VectorRetriever) -> None:
    scores = [hit.score for hit in retriever.search(QUERY, N_CHUNKS)]

    assert scores == sorted(scores, reverse=True)
    assert all(-1 <= s <= 1 for s in scores)


def test_k_larger_than_count_returns_all(retriever: VectorRetriever) -> None:
    hits = retriever.search(QUERY, 100)

    assert len(hits) == N_CHUNKS
    assert {hit.chunk_id for hit in hits} == {c.chunk_id for c in make_chunks(CONFIG_ID)}


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty(retriever: VectorRetriever, k: int) -> None:
    assert retriever.search(QUERY, k) == []


def test_empty_collection_returns_empty(
    tmp_path: Path, use_cached_model: SentenceTransformer
) -> None:
    vector_store = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(vector_store))
    client.create_collection("empty", metadata={"hnsw:space": "cosine"})

    retriever = VectorRetriever(vector_store, "empty")

    assert retriever.search(QUERY, 5) == []


def test_query_gets_bge_prefix(retriever: VectorRetriever, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    encode = retriever.model.encode

    def recording_encode(sentences: Any, **kwargs: Any) -> Any:
        calls.append((sentences, kwargs))
        return encode(sentences, **kwargs)

    monkeypatch.setattr(retriever.model, "encode", recording_encode)

    retriever.search(QUERY, 1)

    assert calls == [([BGE_QUERY_PREFIX + QUERY], {"normalize_embeddings": True})]


# chroma_where (no model)

ROBOTICS_TIME = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
# 2026-10-08 is 20,734 days after 1970-01-01: 20,734 * 86,400 + 14 * 3,600 = 1,791,468,000.
ROBOTICS_TS = 1_791_468_000.0


@pytest.fixture
def bare() -> VectorRetriever:
    """A VectorRetriever without __init__: chroma_where uses no state, so skip loading the model."""
    return VectorRetriever.__new__(VectorRetriever)


@pytest.mark.parametrize("search_filter", [None, SearchFilter()])
def test_where_is_none_without_filter_fields(
    bare: VectorRetriever, search_filter: SearchFilter | None
) -> None:
    assert bare.chroma_where(search_filter) is None


def test_where_source_only_has_no_and(bare: VectorRetriever) -> None:
    where = bare.chroma_where(SearchFilter(sources=frozenset({"TechCrunch"})))

    assert where == {"source": {"$in": ["TechCrunch"]}}


def test_where_lists_every_source(bare: VectorRetriever) -> None:
    where = bare.chroma_where(SearchFilter(sources=frozenset({"TechCrunch", "Verge", "Wired"})))

    assert where is not None and set(where) == {"source"}
    assert sorted(where["source"]["$in"]) == ["TechCrunch", "Verge", "Wired"]


def test_where_date_only_uses_unix_seconds(bare: VectorRetriever) -> None:
    assert bare.chroma_where(SearchFilter(since=ROBOTICS_TIME)) == {
        "published_at": {"$gte": ROBOTICS_TS}
    }


def test_where_date_in_other_timezone_gives_same_seconds(bare: VectorRetriever) -> None:
    # 16:00 at +02:00 is 14:00 UTC.
    since = datetime(2026, 10, 8, 16, 0, tzinfo=timezone(timedelta(hours=2)))

    assert bare.chroma_where(SearchFilter(since=since)) == {"published_at": {"$gte": ROBOTICS_TS}}


def test_where_both_fields_use_and(bare: VectorRetriever) -> None:
    where = bare.chroma_where(SearchFilter(sources=frozenset({"Guardian"}), since=ROBOTICS_TIME))

    assert where == {
        "$and": [
            {"source": {"$in": ["Guardian"]}},
            {"published_at": {"$gte": ROBOTICS_TS}},
        ]
    }


# Filtered search on the built store
# Fixture: ai-act Guardian 10-10, iphone Verge undated, robotics TechCrunch 10-08 14:00,
# sourdough Food Blog 09-30 (all UTC).


def chunk_ids(hits: list[Hit]) -> set[str]:
    return {hit.chunk_id for hit in hits}


def test_search_source_only_keeps_undated(retriever: VectorRetriever) -> None:
    hits = retriever.search(QUERY, 100, search_filter=SearchFilter(sources=frozenset({"Verge"})))

    assert chunk_ids(hits) == {f"{CONFIG_ID}:iphone:0", f"{CONFIG_ID}:iphone:1"}


def test_search_date_only_drops_older_and_undated(retriever: VectorRetriever) -> None:
    hits = retriever.search(QUERY, 100, search_filter=SearchFilter(since=ROBOTICS_TIME))

    # since is inclusive, so robotics (exactly 14:00) stays.
    assert chunk_ids(hits) == {
        f"{CONFIG_ID}:ai-act:0",
        f"{CONFIG_ID}:ai-act:1",
        f"{CONFIG_ID}:robotics:0",
    }


def test_search_source_and_date_combined(retriever: VectorRetriever) -> None:
    search_filter = SearchFilter(sources=frozenset({"Guardian", "Verge"}), since=ROBOTICS_TIME)

    hits = retriever.search(QUERY, 100, search_filter=search_filter)

    assert chunk_ids(hits) == {f"{CONFIG_ID}:ai-act:0", f"{CONFIG_ID}:ai-act:1"}


def test_search_unknown_source_returns_empty(retriever: VectorRetriever) -> None:
    hits = retriever.search(QUERY, 5, search_filter=SearchFilter(sources=frozenset({"Nobody"})))

    assert hits == []


def test_search_filter_applies_before_top_k(retriever: VectorRetriever) -> None:
    assert retriever.search(QUERY, 1)[0].article_id == "ai-act"  # precondition

    hits = retriever.search(QUERY, 1, search_filter=SearchFilter(sources=frozenset({"Verge"})))

    assert [hit.article_id for hit in hits] == ["iphone"]


def test_search_filter_keeps_unfiltered_scores(retriever: VectorRetriever) -> None:
    unfiltered = {hit.chunk_id: hit.score for hit in retriever.search(QUERY, 100)}

    hits = retriever.search(QUERY, 100, search_filter=SearchFilter(sources=frozenset({"Verge"})))

    assert [hit.score for hit in hits] == pytest.approx([unfiltered[h.chunk_id] for h in hits])
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
