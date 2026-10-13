from pathlib import Path
from typing import Any

import chromadb
import pytest
from conftest import AI_ACT_FIRST_CHUNK, CONFIG_ID, make_chunks
from sentence_transformers import SentenceTransformer

from ai_news_assistant.rag.base import Retriever
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
