from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sentence_transformers import SentenceTransformer

from ai_news_assistant.models import Article
from ai_news_assistant.rag import build_vectors, vector_retriever
from ai_news_assistant.rag.config import EMBEDDING_MODEL
from ai_news_assistant.rag.store import Chunk, RagStore
from ai_news_assistant.storage.sqlite import SQLiteStore

CONFIG_ID = "test-cfg"
OTHER_CONFIG_ID = "other-cfg"

# article_id -> (source, published_at, chunk texts). Topics are far apart so the right
# chunk for a query is obvious.
CORPUS: dict[str, tuple[str, datetime | None, list[str]]] = {
    "ai-act": (
        "Guardian",
        datetime(2026, 10, 10, 9, 30, tzinfo=UTC),
        [
            "The European Parliament passed the AI Act, the first comprehensive law "
            "regulating artificial intelligence systems in the EU.",
            "Under the AI Act, high-risk AI systems must meet transparency and "
            "human-oversight requirements before they reach the market.",
        ],
    ),
    "iphone": (
        "Verge",
        None,
        [
            "Apple unveiled a new iPhone with a faster chip and a brighter display.",
            "The new phone ships next month and starts at $799 in the United States.",
        ],
    ),
    "robotics": (
        "TechCrunch",
        datetime(2026, 10, 8, 14, 0, tzinfo=UTC),
        [
            "A warehouse robotics startup raised $50 million in a Series B round "
            "led by venture capital investors.",
        ],
    ),
    "sourdough": (
        "Food Blog",
        datetime(2026, 9, 30, 7, 0, tzinfo=UTC),
        [
            "To bake sourdough bread, feed your starter the night before and let "
            "the dough proof slowly in the fridge.",
        ],
    ),
}

AI_ACT_FIRST_CHUNK = CORPUS["ai-act"][2][0]


def make_chunks(config_id: str) -> list[Chunk]:
    return [
        Chunk(config_id=config_id, article_id=article_id, chunk_index=i, text=text)
        for article_id, (_, _, texts) in CORPUS.items()
        for i, text in enumerate(texts)
    ]


@pytest.fixture(scope="session")
def embedding_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except OSError:
        pytest.skip("BGE model unavailable (needs HF Hub download or cache)")


@pytest.fixture
def use_cached_model(
    embedding_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> SentenceTransformer:
    """Make build() and VectorRetriever() reuse the session model instead of reloading it."""

    def load(_name: str, *args: Any, **kwargs: Any) -> SentenceTransformer:
        return embedding_model

    monkeypatch.setattr(build_vectors, "SentenceTransformer", load)
    monkeypatch.setattr(vector_retriever, "SentenceTransformer", load)
    return embedding_model


@pytest.fixture
def rag_db(tmp_path: Path) -> Path:
    """A SQLite file with articles, a corpus and chunks for CONFIG_ID and OTHER_CONFIG_ID."""
    db = tmp_path / "news.db"
    extracted_at = datetime(2026, 10, 11, tzinfo=UTC)
    articles = [
        Article(
            id=article_id,
            url=f"https://example.com/{article_id}",
            title=f"Title of {article_id}",
            source=source,
            published_at=published_at,
            text="\n\n".join(texts),
            extraction_successful=True,
            extracted_at=extracted_at,
        )
        for article_id, (source, published_at, texts) in CORPUS.items()
    ]
    with SQLiteStore(db) as store:
        store.save(articles)
    with RagStore(db) as store:
        store.replace_corpus(CORPUS)
        store.replace_chunks(CONFIG_ID, make_chunks(CONFIG_ID))
        store.replace_chunks(
            OTHER_CONFIG_ID,
            [Chunk(config_id=OTHER_CONFIG_ID, article_id="ai-act", chunk_index=0, text="Other.")],
        )
    return db


@pytest.fixture
def built_store(rag_db: Path, tmp_path: Path, use_cached_model: SentenceTransformer) -> Path:
    """A Chroma directory with CONFIG_ID embedded by build_vectors.build."""
    vector_store = tmp_path / "chroma"
    build_vectors.build(rag_db, vector_store, CONFIG_ID)
    return vector_store
