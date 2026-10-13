from datetime import datetime
from pathlib import Path

import chromadb
import pytest
from conftest import CONFIG_ID, OTHER_CONFIG_ID, make_chunks
from sentence_transformers import SentenceTransformer

from ai_news_assistant.rag import build_vectors
from ai_news_assistant.rag.build_vectors import build, main, metadata
from ai_news_assistant.rag.store import Chunk, CorpusArticle, RagStore


def make_article(published_at: str | None) -> CorpusArticle:
    return CorpusArticle(
        id="a1", title="A title", source="Guardian", published_at=published_at, text="Text."
    )


def get_collection(vector_store: Path, name: str = CONFIG_ID) -> chromadb.Collection:
    return chromadb.PersistentClient(path=str(vector_store)).get_collection(name)


def test_metadata_has_article_fields() -> None:
    published_at = "2026-10-10T09:30:00+00:00"
    chunk = Chunk(config_id=CONFIG_ID, article_id="a1", chunk_index=3, text="Text.")

    meta = metadata(chunk, {"a1": make_article(published_at)})

    assert meta == {
        "article_id": "a1",
        "chunk_index": 3,
        "source": "Guardian",
        "published_at": datetime.fromisoformat(published_at).timestamp(),
    }


def test_metadata_omits_missing_published_at() -> None:
    chunk = Chunk(config_id=CONFIG_ID, article_id="a1", chunk_index=0, text="Text.")

    meta = metadata(chunk, {"a1": make_article(None)})

    assert "published_at" not in meta


def test_main_rejects_unknown_config(rag_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> None:
        raise AssertionError("build must not run for an unknown config")

    monkeypatch.setattr(build_vectors, "build", fail)

    with pytest.raises(SystemExit) as exc_info:
        main(["--config", "nope", "--db", str(rag_db)])

    assert exc_info.value.code == 2


def test_build_stores_one_vector_per_chunk_of_config(built_store: Path) -> None:
    expected = make_chunks(CONFIG_ID)

    stored = get_collection(built_store).get(include=["documents"])

    assert sorted(stored["ids"]) == sorted(c.chunk_id for c in expected)
    assert stored["documents"] is not None
    documents = dict(zip(stored["ids"], stored["documents"], strict=True))
    assert documents == {c.chunk_id: c.text for c in expected}
    assert not any(i.startswith(f"{OTHER_CONFIG_ID}:") for i in stored["ids"])


def test_build_stores_metadata(rag_db: Path, built_store: Path) -> None:
    with RagStore(rag_db) as store:
        chunks = store.get_chunks(CONFIG_ID)
        articles = {a.id: a for a in store.corpus_articles()}

    stored = get_collection(built_store).get(include=["metadatas"])

    assert stored["metadatas"] is not None
    metadatas = dict(zip(stored["ids"], stored["metadatas"], strict=True))
    assert metadatas == {c.chunk_id: metadata(c, articles) for c in chunks}


def test_build_uses_cosine_space(built_store: Path) -> None:
    collection_metadata = get_collection(built_store).metadata

    assert collection_metadata is not None
    assert collection_metadata["hnsw:space"] == "cosine"


def test_index_sanity_self_query(built_store: Path, embedding_model: SentenceTransformer) -> None:
    chunk = make_chunks(CONFIG_ID)[2]
    embedding = embedding_model.encode([chunk.text], normalize_embeddings=True)

    result = get_collection(built_store).query(query_embeddings=embedding.tolist(), n_results=1)

    assert result["ids"][0] == [chunk.chunk_id]
    assert result["distances"] is not None
    assert result["distances"][0][0] == pytest.approx(0, abs=1e-3)


def test_rebuild_replaces_collection(rag_db: Path, built_store: Path) -> None:
    kept = make_chunks(CONFIG_ID)[:2]
    with RagStore(rag_db) as store:
        store.replace_chunks(CONFIG_ID, kept)

    build(rag_db, built_store, CONFIG_ID)

    stored = get_collection(built_store).get()
    assert sorted(stored["ids"]) == sorted(c.chunk_id for c in kept)
