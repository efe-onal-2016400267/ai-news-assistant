import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ai_news_assistant.rag.config import DEFAULT_DB, DEFAULT_VECTOR_STORE, MODEL_NAME
from ai_news_assistant.rag.store import Chunk, CorpusArticle, RagStore

BATCH = 1000  # chunks per Chroma `add` call


def build(db: Path, vector_store: Path, config_id: str) -> None:
    """Build the vector store by vectorising the chunks from the db."""

    # Get chunks and articles from the store.
    # Get chunks by config_id
    with RagStore(db) as store:
        chunks = store.get_chunks(config_id)
        articles = {a.id: a for a in store.corpus_articles()}

    # Define embedding model
    model = SentenceTransformer(MODEL_NAME)
    # Create embeddings, normalized so other metrics work too (e.g.: dot product)
    embeddings = model.encode(
        [c.text for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )

    # Create client and collention
    client = chromadb.PersistentClient(path=str(vector_store))
    try:
        client.delete_collection(config_id)
    except Exception:  # raises if it doesn't exist; will narrow it once I see the type
        pass
    collection = client.create_collection(config_id, metadata={"hnsw:space": "cosine"})
    batch_size = min(BATCH, client.get_max_batch_size())

    # Store embeddings in the store batch by batch
    for start in range(0, len(chunks), batch_size):
        part = chunks[start : start + batch_size]
        collection.add(
            ids=[c.chunk_id for c in part],
            embeddings=embeddings[start : start + batch_size].tolist(),
            documents=[c.text for c in part],
            metadatas=[metadata(c, articles) for c in part],
        )

    print(f"{config_id}: {collection.count()} vectors stored ({len(chunks)} chunks)")


def metadata(chunk: Chunk, articles: dict[str, CorpusArticle]) -> dict[str, str | int | float]:
    article = articles[chunk.article_id]
    meta: dict[str, str | int | float] = {
        "article_id": chunk.article_id,
        "chunk_index": chunk.chunk_index,
        "source": article.source,
    }

    if article.published_at:
        meta["published_at"] = datetime.fromisoformat(article.published_at).timestamp()
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Embed one chunking config into a Chroma collection."
    )
    parser.add_argument("--config", required=True, help="config_id from the chunks table")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--vector-store", type=Path, default=DEFAULT_VECTOR_STORE)
    args = parser.parse_args(argv)

    with RagStore(args.db) as store:
        available = store.config_ids()
    if args.config not in available:
        parser.error(f"unknown config {args.config!r}; available: {', '.join(available)}")
    build(args.db, args.vector_store, args.config)


if __name__ == "__main__":
    main()
